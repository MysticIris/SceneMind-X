"""Evidence-aware orchestration for the Phase 1 application."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

from scenemindx.services.contracts import VLMService
from scenemindx.services.retrieval import RetrievalService

from .library import LibraryRepository
from .tracing import TraceStore


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def _service_identity(status: dict[str, Any]) -> tuple[str | None, str | None]:
    return status.get("model"), status.get("model_revision")


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class Orchestrator:
    """Provide orchestrator behavior."""
    def __init__(
        self,
        library: LibraryRepository,
        vlm: VLMService,
        retrieval: RetrievalService,
        traces: TraceStore,
        output_root: Path,
    ) -> None:
        self.library = library
        self.vlm = vlm
        self.retrieval = retrieval
        self.traces = traces
        self.output_root = output_root

    def route(self, task_type: str) -> list[str]:
        """Execute the route operation."""
        routes = {
            "analyze": ["library", "vlm"],
            "search": ["embedding", "retrieval"],
            "vqa": ["library", "historical_p3", "ocr_evidence", "vlm"],
            "generate": ["library", "historical_p3", "vlm"],
            "describe": ["library", "production_core", "vlm"],
            "build_index": ["library", "embedding", "retrieval"],
        }
        if task_type not in routes:
            raise ValueError(f"unsupported task type: {task_type}")
        return routes[task_type]

    def collect_evidence(self, image_id: str) -> dict[str, Any]:
        """Execute the collect evidence operation."""
        return {
            "current_core_facts": self.current_facts(image_id),
            "historical_p3_v1_3": self.library.historical_intelligence(image_id),
            "ocr": self.library.ocr_evidence(image_id),
        }

    def current_facts(self, image_id: str) -> dict[str, Any]:
        """Execute the current facts operation."""
        identity = self._core_identity(None)
        cache_path = self._core_cache_path(image_id, str(identity["prompt_id"]))
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("image_id") == image_id and cached.get("core_prompt") == identity:
                data = cached.get("result", {}).get("data", {})
                facts = data.get("normalized_output") or data.get("parsed_output")
                if isinstance(facts, dict) and facts:
                    return {
                        "status": "available",
                        "source_version": str(identity["prompt_version"]),
                        "source": "current_core_cache",
                        "image_id": image_id,
                        **facts,
                    }
        return self.library.historical_intelligence(image_id)

    @staticmethod
    def facts_text(facts: dict[str, Any]) -> str:
        """Execute the facts text operation."""
        ignored = {"status", "source", "source_version", "source_path", "image_id", "schema_valid"}
        values = [
            str(value).strip()
            for key, value in facts.items()
            if key not in ignored and value is not None and value != "" and value != "not_available_in_v1_3"
        ]
        return "；".join(dict.fromkeys(values))

    @staticmethod
    def compose_response(payload: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
        """Execute the compose response operation."""
        return {**payload, "request_id": trace["request_id"], "trace": trace}

    def _start_trace(
        self,
        task_type: str,
        image_ids: list[str],
        *,
        prompt_version: str,
        schema_version: str,
        model_status: dict[str, Any],
        prompt_sha256: str | None = None,
        input_facts_sha256: str | None = None,
    ) -> dict[str, Any]:
        model, revision = _service_identity(model_status)
        return self.traces.start(
            task_type,
            image_ids,
            model=model,
            model_revision=revision,
            prompt_version=prompt_version,
            schema_version=schema_version,
            services=self.route(task_type),
            prompt_sha256=prompt_sha256,
            input_facts_sha256=input_facts_sha256,
        )

    def _core_identity(self, prompt_version: str | None) -> dict[str, str | None]:
        if hasattr(self.vlm, "prompt_identity"):
            return self.vlm.prompt_identity(prompt_version)
        core = self.vlm.status().get("core_prompt", {})
        return {
            "prompt_id": prompt_version or core.get("prompt_id") or "p3_v1_4",
            "prompt_version": core.get("prompt_version") or prompt_version or "P3 v1.4",
            "prompt_sha256": core.get("prompt_sha256"),
        }

    def _core_cache_path(self, image_id: str, prompt_id: str) -> Path:
        return self.output_root / "core_cache" / prompt_id / f"{Path(image_id).stem}.json"

    def _run_core_analysis(self, image_id: str, prompt_version: str | None, *, allow_cache: bool) -> tuple[dict[str, Any], str]:
        identity = self._core_identity(prompt_version)
        prompt_id = str(identity["prompt_id"])
        cache_path = self._core_cache_path(image_id, prompt_id)
        if allow_cache and cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("image_id") == image_id and cached.get("core_prompt") == identity:
                return cached["result"], "cache"
        image_path = self.library.image_path(image_id, verify_hash=True)
        if prompt_version is None:
            result = self.vlm.analyze_image(image_path)
        else:
            result = self.vlm.analyze_image(image_path, prompt_version=prompt_version)
        if result.status != "success":
            raise RuntimeError(result.error or result.status)
        result_payload = result.as_dict()
        _write_json_atomic(cache_path, {"image_id": image_id, "core_prompt": identity, "result": result_payload})
        return result_payload, "generated"

    def analyze(self, image_id: str, prompt_version: str | None = None) -> dict[str, Any]:
        """Execute the analyze operation."""
        identity = self._core_identity(prompt_version)
        trace = self._start_trace(
            "analyze",
            [image_id],
            prompt_version=str(identity["prompt_version"]),
            prompt_sha256=identity.get("prompt_sha256"),
            schema_version="gate1_d3_semantic_review_payload_p3_v1_1",
            model_status=self.vlm.status(),
        )
        try:
            result, core_source = self._run_core_analysis(image_id, prompt_version, allow_cache=False)
            payload = {
                "namespace": "production_core_prompt_v1",
                "historical_p3_v1_3_unchanged": True,
                "image_id": image_id,
                "core_prompt": identity,
                "core_source": core_source,
                "result": result,
            }
            output_path = self.output_root / "analyze" / f"{trace['request_id']}.json"
            _write_json_atomic(output_path, payload)
            finished = self.traces.finish(trace, status="success", output_path=str(output_path))
            return self.compose_response(payload, finished)
        except Exception as exc:
            self.traces.finish(trace, status="failed", error=exc)
            raise

    def build_index(self, extra_items: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Build index."""
        trace = self._start_trace(
            "build_index",
            [],
            prompt_version="not_applicable",
            schema_version="phase1_retrieval_index_v1",
            model_status=self.retrieval.embedding.status(),
        )
        try:
            items = [
                {
                    "image_id": asset["image_id"],
                    "asset_id": asset["image_id"],
                    "sha256": asset["sha256"],
                    "library_id": "default",
                    "image_path": str(self.library.image_path(asset["image_id"], verify_hash=True)),
                    "retrieval_text": self.facts_text(self.current_facts(asset["image_id"])),
                    "source": "frozen_library",
                    "image_url": f"/library/{asset['image_id']}/image",
                }
                for asset in self.library.list_assets()
            ]
            items.extend(list(extra_items or []))
            status = self.retrieval.build_index(items)
            finished = self.traces.finish(trace, status="success", output_path=str(self.retrieval.index_path))
            return self.compose_response({"index": status}, finished)
        except Exception as exc:
            self.traces.finish(trace, status="failed", error=exc)
            raise

    def search(self, query: str, top_k: int) -> dict[str, Any]:
        """Execute the search operation."""
        trace = self._start_trace(
            "search",
            [],
            prompt_version="not_applicable",
            schema_version="phase1_search_response_v1",
            model_status=self.retrieval.embedding.status(),
        )
        try:
            results = self.retrieval.search(query, top_k)
            if results:
                trace["retrieval"] = {
                    key: results[0].get(key)
                    for key in (
                        "retrieval_backend",
                        "model",
                        "revision",
                        "index_version",
                        "fallback_used",
                        "fallback_reason",
                    )
                }
            finished = self.traces.finish(trace, status="success", output_path=str(self.retrieval.index_path))
            return self.compose_response({"query": query, "results": results}, finished)
        except Exception as exc:
            self.traces.finish(trace, status="failed", error=exc)
            raise

    def answer_question(self, image_id: str, question: str, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Execute the answer question operation."""
        trace = self._start_trace(
            "vqa",
            [image_id],
            prompt_version="phase1_mvp_vqa_v1",
            schema_version="phase1_mvp_vqa_v1",
            model_status=self.vlm.status(),
        )
        try:
            image_path = self.library.image_path(image_id, verify_hash=True)
            evidence = self.collect_evidence(image_id)
            if history:
                evidence["conversation_history"] = history
            result = self.vlm.answer_question(image_path, question, evidence)
            if result.status != "success":
                raise RuntimeError(result.error or result.status)
            payload = {"image_id": image_id, "question": question, "evidence": evidence, "result": result.as_dict()}
            output_path = self.output_root / "vqa" / f"{trace['request_id']}.json"
            _write_json_atomic(output_path, payload)
            finished = self.traces.finish(trace, status="success", output_path=str(output_path))
            return self.compose_response(payload, finished)
        except Exception as exc:
            self.traces.finish(trace, status="failed", error=exc)
            raise

    def generate_content(self, image_ids: Sequence[str], options: dict[str, Any]) -> dict[str, Any]:
        """Execute the generate content operation."""
        ids = list(image_ids)
        trace = self._start_trace(
            "generate",
            ids,
            prompt_version="phase1_mvp_grounded_content_v1",
            schema_version="phase1_mvp_content_v1",
            model_status=self.vlm.status(),
        )
        try:
            paths = [self.library.image_path(image_id, verify_hash=True) for image_id in ids]
            facts = [self.current_facts(image_id) for image_id in ids]
            result = self.vlm.generate_content(paths, facts, options)
            if result.status != "success":
                raise RuntimeError(result.error or result.status)
            payload = {"image_ids": ids, "options": options, "grounding_facts": facts, "result": result.as_dict()}
            output_path = self.output_root / "generate" / f"{trace['request_id']}.json"
            _write_json_atomic(output_path, payload)
            finished = self.traces.finish(trace, status="success", output_path=str(output_path))
            return self.compose_response(payload, finished)
        except Exception as exc:
            self.traces.finish(trace, status="failed", error=exc)
            raise

    def describe_image(self, image_id: str, options: dict[str, Any]) -> dict[str, Any]:
        """Execute the describe image operation."""
        task_prompt_id = "natural_chinese_detailed_description_v1"
        if hasattr(self.vlm, "task_prompt_identity"):
            task_identity = self.vlm.task_prompt_identity(task_prompt_id)
        else:
            task_identity = {
                "prompt_id": task_prompt_id,
                "prompt_version": "NATURAL_CHINESE_DETAILED_DESCRIPTION_V1",
                "prompt_sha256": None,
            }
        core_identity = self._core_identity(None)
        trace = self._start_trace(
            "describe",
            [image_id],
            prompt_version=str(task_identity["prompt_version"]),
            prompt_sha256=task_identity.get("prompt_sha256"),
            schema_version="natural_chinese_detailed_description_response_v1",
            model_status=self.vlm.status(),
        )
        try:
            core_result, core_source = self._run_core_analysis(image_id, None, allow_cache=True)
            core_data = core_result.get("data", {})
            core_facts = core_data.get("normalized_output") or core_data.get("parsed_output")
            if not isinstance(core_facts, dict) or not core_facts:
                raise RuntimeError("production core facts are unavailable")
            facts_sha256 = _sha256_json(core_facts)
            trace["input_facts_sha256"] = facts_sha256
            image_path = self.library.image_path(image_id, verify_hash=True)
            result = self.vlm.describe_image(image_path, core_facts, options)
            if result.status != "success":
                raise RuntimeError(result.error or result.status)
            payload = {
                "namespace": "natural_chinese_detailed_description_v1",
                "image_id": image_id,
                "options": options,
                "core_prompt": core_identity,
                "core_source": core_source,
                "input_core_facts_sha256": facts_sha256,
                "core_facts": core_facts,
                "result": result.as_dict(),
            }
            output_path = self.output_root / "describe" / f"{trace['request_id']}.json"
            _write_json_atomic(output_path, payload)
            finished = self.traces.finish(trace, status="success", output_path=str(output_path))
            return self.compose_response(payload, finished)
        except Exception as exc:
            self.traces.finish(trace, status="failed", error=exc)
            raise
