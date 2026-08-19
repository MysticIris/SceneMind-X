"""Persistent Qwen3-VL service for Phase 1 server inference."""

from __future__ import annotations

import ast
import gc
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Sequence

from scenemindx.prompts.loader import CorePromptRegistry, load_core_prompt_registry, load_prompt_manifest

from .contracts import ServiceResult


def _extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        candidate = text[index:]
        try:
            value, _ = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            # Repair object-key quotes only; values remain model-authored text.
            candidate = re.sub(
                r"([,{]\s*)[\"'\u201c\u201d]([^\"'\u201c\u201d]+)[\"'\u201c\u201d]\s*:",
                r'\1"\2":',
                candidate,
            )
            try:
                value, _ = decoder.raw_decode(candidate)
            except json.JSONDecodeError:
                try:
                    value = ast.literal_eval(candidate)
                except (SyntaxError, ValueError):
                    continue
        if isinstance(value, dict):
            return value
    raise ValueError("model output did not contain a JSON object")


def _parse_rank_positions(text: str, expected_count: int) -> list[int] | None:
    value = text.strip()
    if not re.fullmatch(r"\d+(?:\s*[,，>]\s*\d+)*", value):
        return None
    positions = [int(item) for item in re.findall(r"\d+", value)]
    if len(positions) != expected_count or set(positions) != set(range(1, expected_count + 1)):
        return None
    return positions


def _normalize_plain_text(text: str) -> str:
    value = text.strip()
    value = re.sub(r"^```(?:text|markdown)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    return re.sub(r"\s+", " ", value).strip().strip('"')


def _normalize_simplified_chinese(text: str) -> tuple[str, bool]:
    from opencc import OpenCC

    normalized = OpenCC("t2s").convert(text)
    return normalized, normalized != text


def _strip_description_meta_text(text: str) -> tuple[str, bool]:
    meta_markers = ("字数", "字符范围", "符合要求", "符合150", "满足150", "核对完毕", "自检", "最终输出为纯文本")
    kept_lines = []
    removed = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and any(marker in stripped for marker in meta_markers):
            removed = True
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines), removed


class DisabledVLMService:
    """Provide disabled v l m service behavior."""
    def status(self) -> dict[str, Any]:
        """Execute the status operation."""
        return {"status": "disabled", "loaded": False}

    def analyze_image(self, image_path: Path, prompt_version: str | None = None) -> ServiceResult:
        """Execute the analyze image operation."""
        return ServiceResult(status="disabled", error="vlm_service_disabled")

    def describe_image(self, image_path: Path, core_facts: dict[str, Any], options: dict[str, Any]) -> ServiceResult:
        """Execute the describe image operation."""
        return ServiceResult(status="disabled", error="vlm_service_disabled")

    def task_prompt_identity(self, prompt_id: str) -> dict[str, str]:
        """Execute the task prompt identity operation."""
        raise RuntimeError("vlm_service_disabled")

    def answer_question(self, image_path: Path, question: str, evidence: dict[str, Any]) -> ServiceResult:
        """Execute the answer question operation."""
        return ServiceResult(status="disabled", error="vlm_service_disabled")

    def generate_content(self, image_paths: Sequence[Path], facts: Sequence[dict[str, Any]], options: dict[str, Any]) -> ServiceResult:
        """Execute the generate content operation."""
        return ServiceResult(status="disabled", error="vlm_service_disabled")

    def compare_images(self, image_paths: Sequence[Path], instruction: str | None = None) -> ServiceResult:
        """Execute the compare images operation."""
        return ServiceResult(status="disabled", error="vlm_service_disabled")

    def run_course_prompt(
        self,
        image_paths: Sequence[Path],
        prompt: str,
        *,
        prompt_id: str,
        prompt_sha256: str,
        max_new_tokens: int = 512,
        image_labels: Sequence[str] | None = None,
        min_new_tokens: int | None = None,
    ) -> ServiceResult:
        """Execute the run course prompt operation."""
        return ServiceResult(status="disabled", error="vlm_service_disabled")

    def run_text_repair(
        self,
        prompt: str,
        *,
        prompt_id: str,
        prompt_sha256: str,
        max_new_tokens: int = 384,
    ) -> ServiceResult:
        """Execute the run text repair operation."""
        return ServiceResult(status="disabled", error="vlm_service_disabled")


class PersistentQwen3VLService:
    """Load Qwen3-VL once and serialize batch-size-one requests."""

    def __init__(
        self,
        model_path: Path,
        prompt_root: Path,
        *,
        model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
        model_revision: str = "ebb281ec70b05090aa6165b016eac8ec08e71b17",
        dtype: str = "bfloat16",
        gpu_max_memory_gib: float = 14.0,
        core_registry_path: Path | None = None,
        core_prompt_version: str | None = None,
    ) -> None:
        self.model_path = model_path.resolve()
        self.prompt_root = prompt_root.resolve()
        self.model_id = model_id
        self.model_revision = model_revision
        self.dtype = dtype
        self.gpu_max_memory_gib = gpu_max_memory_gib
        registry_path = core_registry_path or self.prompt_root.parent / "gate1" / "p3_registry.json"
        self.core_registry: CorePromptRegistry = load_core_prompt_registry(registry_path)
        self.core_prompt_version = core_prompt_version or self.core_registry.default_prompt
        if self.core_prompt_version not in self.core_registry.prompts:
            raise ValueError(f"unknown core Prompt version: {self.core_prompt_version}")
        self.task_prompts = load_prompt_manifest(self.prompt_root / "registry.json")
        schema_path = self.prompt_root.parents[1] / "data" / "schemas" / "gate1_d3_semantic_review_payload_p3_v1_1.schema.json"
        self.core_schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.model: Any | None = None
        self.processor: Any | None = None
        self.model_device: Any | None = None
        self.load_metrics: dict[str, Any] = {}
        self._lock = threading.Lock()

    def _validate_files(self) -> None:
        required = ("config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json")
        missing = [name for name in required if not (self.model_path / name).is_file()]
        weights = list(self.model_path.glob("*.safetensors"))
        if not self.model_path.is_dir() or missing or not weights:
            raise FileNotFoundError(f"incomplete local model directory: missing={missing}, weight_files={len(weights)}")

    def load(self) -> dict[str, Any]:
        """Load the requested value."""
        if self.model is not None:
            return self.load_metrics
        self._validate_files()
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the Phase 1 Qwen3-VL service")
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if free_bytes < 10 * 1024**3:
            raise RuntimeError(f"insufficient free GPU memory before Qwen load: {free_bytes} bytes")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            min_pixels=65536,
            max_pixels=1003520,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=getattr(torch, self.dtype),
            device_map="auto",
            max_memory={0: f"{self.gpu_max_memory_gib}GiB", "cpu": "64GiB"},
            local_files_only=True,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        self.model.eval()
        self.model_device = next(self.model.parameters()).device
        self.load_metrics = {
            "status": "ready",
            "loaded": True,
            "model": self.model_id,
            "model_revision": self.model_revision,
            "load_seconds": time.perf_counter() - started,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "gpu_free_before_bytes": int(free_bytes),
            "gpu_total_bytes": int(total_bytes),
            "device_map": getattr(self.model, "hf_device_map", None),
        }
        return self.load_metrics

    def status(self) -> dict[str, Any]:
        """Execute the status operation."""
        core = self.prompt_identity()
        if self.model is None:
            return {"status": "not_loaded", "loaded": False, "model": self.model_id, "model_revision": self.model_revision, "core_prompt": core}
        return {**self.load_metrics, "core_prompt": core}

    def unload(self) -> dict[str, Any]:
        """Release only this service's model objects after a failed local load."""

        self.model = None
        self.processor = None
        self.model_device = None
        self.load_metrics = {}
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        return self.status()

    def prompt_identity(self, prompt_version: str | None = None) -> dict[str, str]:
        """Execute the prompt identity operation."""
        prompt_id = prompt_version or self.core_prompt_version
        try:
            prompt = self.core_registry.prompts[prompt_id]
        except KeyError:
            raise ValueError(f"unknown core Prompt version: {prompt_id}") from None
        return {"prompt_id": prompt.prompt_id, "prompt_version": prompt.version, "prompt_sha256": prompt.bundle_sha256}

    def task_prompt_identity(self, prompt_id: str) -> dict[str, str]:
        """Execute the task prompt identity operation."""
        try:
            prompt = self.task_prompts[prompt_id]
        except KeyError:
            raise ValueError(f"unknown task Prompt: {prompt_id}") from None
        return {"prompt_id": prompt.prompt_id, "prompt_version": prompt.version, "prompt_sha256": prompt.sha256}

    def _prompt(self, name: str) -> str:
        return (self.prompt_root / name).read_text(encoding="utf-8").strip()

    def _generate_raw(
        self,
        image_paths: Sequence[Path],
        prompt: str,
        *,
        max_new_tokens: int,
        min_new_tokens: int | None = None,
        repetition_penalty: float = 1.05,
        no_repeat_ngram_size: int = 6,
        image_labels: Sequence[str] | None = None,
        system_prompt: str | None = None,
        history_messages: Sequence[dict[str, str]] | None = None,
    ) -> ServiceResult:
        if self.model is None or self.processor is None:
            raise RuntimeError("load() must be called before inference")
        import torch
        from PIL import Image

        images = []
        inputs = None
        generated = None
        try:
            for path in image_paths:
                with Image.open(path) as handle:
                    images.append(handle.convert("RGB"))
            if image_labels is not None and len(image_labels) != len(images):
                raise ValueError("multiturn_chat_image_label_count_mismatch")
            content = []
            for index, image in enumerate(images):
                if image_labels is not None:
                    content.append({"type": "text", "text": f"{image_labels[index]}："})
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": prompt})
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            for message in history_messages or []:
                role = str(message.get("role", ""))
                text = str(message.get("content", "")).strip()
                if role not in {"user", "assistant"} or not text:
                    raise ValueError("invalid_multiturn_chat_history_message")
                messages.append({"role": role, "content": text})
            messages.append({"role": "user", "content": content})
            with self._lock:
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.model_device)
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                with torch.inference_mode():
                    generation_options = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": False,
                        "repetition_penalty": repetition_penalty,
                    }
                    if no_repeat_ngram_size > 0:
                        generation_options["no_repeat_ngram_size"] = no_repeat_ngram_size
                    if min_new_tokens is not None:
                        generation_options["min_new_tokens"] = min_new_tokens
                    generated = self.model.generate(**inputs, **generation_options)
                torch.cuda.synchronize()
                latency = time.perf_counter() - started
                trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated)]
                text = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                peak = int(torch.cuda.max_memory_allocated())
                input_tokens = int(inputs.input_ids.shape[-1])
                output_tokens = len(trimmed[0])
                max_new_tokens_hit = output_tokens >= max_new_tokens
            return ServiceResult(
                status="success",
                data={
                    "raw_output": text,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "finish_reason": (
                        "max_new_tokens"
                        if max_new_tokens_hit
                        else "eos_or_model_stop"
                    ),
                    "max_new_tokens": max_new_tokens,
                    "max_new_tokens_hit": max_new_tokens_hit,
                    "generation_parameters": {
                        "do_sample": False,
                        "repetition_penalty": repetition_penalty,
                        "no_repeat_ngram_size": no_repeat_ngram_size,
                        "min_new_tokens": min_new_tokens,
                        "max_new_tokens": max_new_tokens,
                    },
                    "message_summary": {
                        "system_message_present": bool(system_prompt),
                        "history_message_count": len(history_messages or []),
                        "image_count": len(image_paths),
                        "image_labels": (
                            list(image_labels)
                            if image_labels is not None
                            else None
                        ),
                        "roles": [message["role"] for message in messages],
                    },
                },
                model=self.model_id,
                model_revision=self.model_revision,
                latency_seconds=latency,
                peak_vram_bytes=peak,
            )
        finally:
            del inputs, generated
            for image in images:
                image.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _generate(
        self,
        image_paths: Sequence[Path],
        prompt: str,
        *,
        max_new_tokens: int,
        repetition_penalty: float = 1.05,
        no_repeat_ngram_size: int = 6,
    ) -> ServiceResult:
        result = self._generate_raw(
            image_paths,
            prompt,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        return ServiceResult(
            status=result.status,
            data={**result.data, "parsed_output": _extract_json_object(result.data["raw_output"])},
            error=result.error,
            model=result.model,
            model_revision=result.model_revision,
            latency_seconds=result.latency_seconds,
            peak_vram_bytes=result.peak_vram_bytes,
        )

    def analyze_image(self, image_path: Path, prompt_version: str | None = None) -> ServiceResult:
        """Execute the analyze image operation."""
        prompt_id = prompt_version or self.core_prompt_version
        identity = self.prompt_identity(prompt_id)
        core_prompt = self.core_registry.prompts[prompt_id]
        normalized: dict[str, str] = {}
        stages: list[dict[str, Any]] = []
        errors: list[str] = []
        total_latency = 0.0
        peak_vram = 0
        for stage in core_prompt.stages:
            result = self._generate_raw([image_path], stage.prompt.text, max_new_tokens=stage.max_new_tokens)
            raw_output = str(result.data["raw_output"])
            parsed: dict[str, Any] | None = None
            stage_error: str | None = None
            try:
                parsed = _extract_json_object(raw_output)
            except ValueError as exc:
                if len(stage.fields) == 1 and raw_output.strip():
                    normalized[stage.fields[0]] = raw_output.strip().replace("```json", "").replace("```", "").strip()
                    stage_error = f"raw_text_fallback: {exc}"
                else:
                    stage_error = str(exc)
            if parsed is not None:
                if set(parsed) != set(stage.fields) or any(not isinstance(parsed.get(field), str) for field in stage.fields):
                    stage_error = f"expected exactly string fields {sorted(stage.fields)}, got {sorted(parsed)}"
                else:
                    normalized.update({field: parsed[field] for field in stage.fields})
            if stage_error:
                errors.append(f"{stage.stage_id}: {stage_error}")
            total_latency += float(result.latency_seconds or 0.0)
            peak_vram = max(peak_vram, int(result.peak_vram_bytes or 0))
            stages.append(
                {
                    "stage_id": stage.stage_id,
                    "prompt_sha256": stage.prompt.sha256,
                    "raw_output": raw_output,
                    "parsed_output": parsed,
                    "output_tokens": result.data.get("output_tokens"),
                    "latency_seconds": result.latency_seconds,
                    "peak_vram_bytes": result.peak_vram_bytes,
                    "stage_error": stage_error,
                }
            )
        from jsonschema import Draft202012Validator

        schema_errors = [
            f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
            for error in sorted(Draft202012Validator(self.core_schema).iter_errors(normalized), key=lambda item: list(item.absolute_path))
        ]
        raw_output = "\n\n".join(f"[{stage['stage_id']}]\n{stage['raw_output']}" for stage in stages)
        return ServiceResult(
            status="success",
            data={
                **identity,
                "raw_output": raw_output,
                "normalized_output": normalized,
                "parsed_output": normalized,
                "stage_outputs": stages,
                "parse_error": "; ".join(errors) if errors else None,
                "schema_valid": not schema_errors,
                "schema_errors": schema_errors,
                "output_tokens": sum(int(stage.get("output_tokens") or 0) for stage in stages),
            },
            model=self.model_id,
            model_revision=self.model_revision,
            latency_seconds=total_latency,
            peak_vram_bytes=peak_vram,
        )

    def describe_image(self, image_path: Path, core_facts: dict[str, Any], options: dict[str, Any]) -> ServiceResult:
        """Execute the describe image operation."""
        spec = self.task_prompts["natural_chinese_detailed_description_v1"]
        prompt = spec.text.format(
            facts=json.dumps(core_facts, ensure_ascii=False),
            options=json.dumps(options, ensure_ascii=False),
        )
        target_length = min(max(int(options.get("length", 180)), 150), 350)
        minimum_generation_tokens = min(max(target_length - 20, 160), 330)
        result = self._generate_raw(
            [image_path],
            prompt,
            max_new_tokens=512,
            min_new_tokens=minimum_generation_tokens,
        )
        content_output, meta_text_removed = _strip_description_meta_text(str(result.data["raw_output"]))
        normalized_output = _normalize_plain_text(content_output)
        final_output, simplified_conversion_applied = _normalize_simplified_chinese(normalized_output)
        chinese_characters = len(re.findall(r"[\u4e00-\u9fff]", final_output))
        return ServiceResult(
            status=result.status,
            data={
                "prompt_id": spec.prompt_id,
                "prompt_version": spec.version,
                "prompt_sha256": spec.sha256,
                "raw_output": result.data["raw_output"],
                "final_output": final_output,
                "output_tokens": result.data.get("output_tokens"),
                "minimum_generation_tokens": minimum_generation_tokens,
                "simplified_chinese_normalizer": "OpenCC t2s",
                "simplified_conversion_applied": simplified_conversion_applied,
                "prohibited_meta_text_removed": meta_text_removed,
                "chinese_character_count": chinese_characters,
                "length_contract_pass": 150 <= chinese_characters <= 350,
            },
            error=result.error,
            model=result.model,
            model_revision=result.model_revision,
            latency_seconds=result.latency_seconds,
            peak_vram_bytes=result.peak_vram_bytes,
        )

    def answer_question(self, image_path: Path, question: str, evidence: dict[str, Any]) -> ServiceResult:
        """Execute the answer question operation."""
        prompt = self._prompt("vqa_v1.txt").format(
            question=question,
            evidence=json.dumps(evidence, ensure_ascii=False),
        )
        return self._generate([image_path], prompt, max_new_tokens=320)

    def generate_content(self, image_paths: Sequence[Path], facts: Sequence[dict[str, Any]], options: dict[str, Any]) -> ServiceResult:
        """Execute the generate content operation."""
        prompt = self._prompt("content_v1.txt").format(
            facts=json.dumps(list(facts), ensure_ascii=False),
            options=json.dumps(options, ensure_ascii=False),
        )
        return self._generate(image_paths, prompt, max_new_tokens=384)

    def compare_images(self, image_paths: Sequence[Path], instruction: str | None = None) -> ServiceResult:
        """Execute the compare images operation."""
        if len(image_paths) < 2:
            raise ValueError("compare_images requires at least two images")
        ranking_mode = bool(instruction and instruction.startswith("RANK_CONTRACT:"))
        if ranking_mode:
            user_instruction = instruction.removeprefix("RANK_CONTRACT:").strip()
            asset_ids = [path.name for path in image_paths]
            image_mapping = [
                {"image_position": index, "asset_id": asset_id}
                for index, asset_id in enumerate(asset_ids, start=1)
            ]
            prompt = (
                "Compare every supplied image and rank all of them for the requested criterion. "
                f"image_position_mapping={json.dumps(image_mapping, ensure_ascii=False)}. "
                f"Output only {len(asset_ids)} unique image_position integers from best to worst, separated by ASCII commas. "
                f"Use every integer from 1 through {len(asset_ids)} exactly once. "
                "Do not output asset ids, JSON, brackets, explanations, labels, or any other text. "
                f"User criterion: {user_instruction}"
            )
            result = self._generate_raw(
                image_paths,
                prompt,
                max_new_tokens=32,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
            )
            positions = _parse_rank_positions(str(result.data.get("raw_output", "")), len(asset_ids))
            if positions is not None:
                normalized = {
                    "ranking": [
                        {"asset_id": asset_ids[position - 1], "rank": rank, "reason": ""}
                        for rank, position in enumerate(positions, start=1)
                    ]
                }
                return ServiceResult(
                    status=result.status,
                    data={**result.data, "parsed_output": normalized},
                    error=result.error,
                    model=result.model,
                    model_revision=result.model_revision,
                    latency_seconds=result.latency_seconds,
                    peak_vram_bytes=result.peak_vram_bytes,
                )
            return ServiceResult(
                status="invalid_output",
                data={**result.data, "parsed_output": None},
                error="ranking_position_contract_violation",
                model=result.model,
                model_revision=result.model_revision,
                latency_seconds=result.latency_seconds,
                peak_vram_bytes=result.peak_vram_bytes,
            )
        prompt = self._prompt("compare_v1.txt")
        if instruction:
            mapping = "；".join(f"图片{index + 1}={path.name}" for index, path in enumerate(image_paths))
            prompt = (
                f"{prompt}\n\n输入映射：{mapping}\n"
                f"用户的比较或排序要求：{instruction}\n"
                "必须逐张引用上述 asset 文件名，不得遗漏任何图片。若要求排序，"
                "JSON 还必须包含 ranking 数组，每项含 asset_id、rank、reason。"
            )
        return self._generate(image_paths, prompt, max_new_tokens=384)

    def run_course_prompt(
        self,
        image_paths: Sequence[Path],
        prompt: str,
        *,
        prompt_id: str,
        prompt_sha256: str,
        max_new_tokens: int = 512,
        image_labels: Sequence[str] | None = None,
        min_new_tokens: int | None = None,
    ) -> ServiceResult:
        """Execute the run course prompt operation."""
        if not 1 <= len(image_paths) <= 5:
            raise ValueError("course_prompt_requires_1_to_5_images")
        if not prompt.strip():
            raise ValueError("course_prompt_must_not_be_empty")
        if image_labels is not None and len(image_paths) != len(image_labels):
            raise ValueError("course_prompt_image_label_count_mismatch")
        if min_new_tokens is not None and not 1 <= min_new_tokens < max_new_tokens:
            raise ValueError("course_prompt_min_new_tokens_must_be_below_max")
        raw = self._generate_raw(
            image_paths,
            prompt,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            image_labels=image_labels,
            no_repeat_ngram_size=0 if prompt_id.startswith("phase6_0a_") else 6,
        )
        parse_error = None
        try:
            parsed_output = _extract_json_object(str(raw.data["raw_output"]))
        except ValueError as exc:
            # Course Prompt format recovery belongs to the product-side
            # validator. Preserve every model-authored raw output here so a
            # malformed envelope never becomes a transport-level 500 before
            # bounded repair, semantic salvage, or an honest failure response
            # can inspect it.
            parsed_output = {}
            parse_error = str(exc)
        return ServiceResult(
            status=raw.status,
            data={
                **raw.data,
                "parsed_output": parsed_output,
                "prompt_id": prompt_id,
                "prompt_sha256": prompt_sha256,
                "candidate_id": (
                    "SCENEMINDX_MULTI_IMAGE_CONTENT_V2_CANDIDATE"
                    if prompt_id == "phase5_2_multi_image_content_v2"
                    else (
                        "SCENEMINDX_MULTI_IMAGE_CONTENT_V3_CANDIDATE"
                        if prompt_id.startswith("phase5_2d_")
                        else (
                        "SCENEMINDX_CONVERSATION_POLICY_V1_CANDIDATE"
                        if prompt_id.startswith("phase5_2c_")
                        else "SCENEMINDX_PHASE5_2_COURSE_PROMPT_CANDIDATE_V1"
                        )
                    )
                ),
                "image_labels": list(image_labels) if image_labels is not None else None,
                "min_new_tokens": min_new_tokens,
                "parse_error": parse_error,
            },
            error=raw.error,
            model=raw.model,
            model_revision=raw.model_revision,
            latency_seconds=raw.latency_seconds,
            peak_vram_bytes=raw.peak_vram_bytes,
        )

    def run_text_repair(
        self,
        prompt: str,
        *,
        prompt_id: str,
        prompt_sha256: str,
        max_new_tokens: int = 384,
    ) -> ServiceResult:
        """Run a text-only, semantic-preserving contract repair."""

        if not prompt.strip():
            raise ValueError("text_repair_prompt_must_not_be_empty")
        result = self._generate_raw(
            [],
            prompt,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=0,
        )
        parse_error = None
        try:
            parsed_output = _extract_json_object(result.data["raw_output"])
        except ValueError as exc:
            parsed_output = None
            parse_error = str(exc)
        return ServiceResult(
            status=result.status,
            data={
                **result.data,
                "parsed_output": parsed_output,
                "parse_error": parse_error,
                "prompt_id": prompt_id,
                "prompt_sha256": prompt_sha256,
            },
            error=result.error,
            model=result.model,
            model_revision=result.model_revision,
            latency_seconds=result.latency_seconds,
            peak_vram_bytes=result.peak_vram_bytes,
        )

    def run_multiturn_chat(
        self,
        image_paths: Sequence[Path],
        image_labels: Sequence[str],
        *,
        system_prompt: str,
        history_messages: Sequence[dict[str, str]],
        current_prompt: str,
        prompt_id: str,
        prompt_sha256: str,
        max_new_tokens: int = 512,
    ) -> ServiceResult:
        """Execute the run multiturn chat operation."""
        if not 1 <= len(image_paths) <= 5:
            raise ValueError("multiturn_chat_requires_1_to_5_images")
        if len(image_paths) != len(image_labels):
            raise ValueError("multiturn_chat_image_label_count_mismatch")
        result = self._generate_raw(
            image_paths,
            current_prompt,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=0,
            image_labels=image_labels,
            system_prompt=system_prompt,
            history_messages=history_messages,
        )
        parse_error = None
        try:
            parsed_output = _extract_json_object(result.data["raw_output"])
        except ValueError as exc:
            parsed_output = None
            parse_error = str(exc)
        return ServiceResult(
            status=result.status,
            data={
                **result.data,
                "parsed_output": parsed_output,
                "parse_error": parse_error,
                "prompt_id": prompt_id,
                "prompt_sha256": prompt_sha256,
                "candidate_id": "SCENEMINDX_MULTITURN_CHAT_V2_CANDIDATE",
                "image_labels": list(image_labels),
                "history_message_count": len(history_messages),
                "message_organization": "system + bounded role history + current IMG_n visual user turn",
                "chat_generation_settings": {
                    "do_sample": False,
                    "repetition_penalty": 1.05,
                    "no_repeat_ngram_size": 0,
                },
            },
            error=result.error,
            model=result.model,
            model_revision=result.model_revision,
            latency_seconds=result.latency_seconds,
            peak_vram_bytes=result.peak_vram_bytes,
        )
