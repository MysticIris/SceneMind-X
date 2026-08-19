"""Offline-only Qwen3-VL runtime used by Gate 1 experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GenerationSettings:
    """Frozen generation controls for a reproducible single-image run."""

    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0

    def as_generate_kwargs(self) -> dict[str, Any]:
        """Execute the as generate kwargs operation."""
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if self.no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size cannot be negative")
        controls: dict[str, Any] = {}
        if self.repetition_penalty != 1.0:
            controls["repetition_penalty"] = self.repetition_penalty
        if self.no_repeat_ngram_size:
            controls["no_repeat_ngram_size"] = self.no_repeat_ngram_size
        if self.do_sample:
            return {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": True,
                "temperature": self.temperature,
                "top_p": self.top_p,
                **controls,
            }
        return {"max_new_tokens": self.max_new_tokens, "do_sample": False, **controls}


class Qwen3VLRuntime:
    """Load an audited local Qwen3-VL checkpoint and run one image at a time."""

    def __init__(self, model_path: str | Path, *, dtype: str = "bfloat16") -> None:
        self.model_path = Path(model_path).resolve()
        self.dtype_name = dtype
        self.model: Any | None = None
        self.processor: Any | None = None
        self.load_seconds: float | None = None

    def validate_local_files(self) -> None:
        """Validate local files."""
        required = (
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        )
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"model directory not found: {self.model_path}")
        missing = [name for name in required if not (self.model_path / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"model directory is incomplete; missing: {', '.join(missing)}"
            )

    def load(self) -> dict[str, Any]:
        """Load processor and weights without any network fallback."""

        self.validate_local_files()
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the Gate 1 Qwen3-VL baseline")
        dtype = getattr(torch, self.dtype_name, None)
        if dtype is None:
            raise ValueError(f"unsupported torch dtype: {self.dtype_name}")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=dtype,
            device_map="auto",
            local_files_only=True,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        self.model.eval()
        self.load_seconds = time.perf_counter() - started
        device_map = getattr(self.model, "hf_device_map", None)
        return {
            "model_path": str(self.model_path),
            "dtype": str(self.model.dtype),
            "device_map": dict(device_map) if device_map else {"model_device": str(next(self.model.parameters()).device)},
            "load_seconds": self.load_seconds,
            "peak_vram_bytes_after_load": torch.cuda.max_memory_allocated(),
        }

    def generate(
        self,
        image_path: str | Path,
        prompt: str,
        settings: GenerationSettings,
    ) -> dict[str, Any]:
        """Generate text for one local image and report latency and peak VRAM."""

        if self.model is None or self.processor is None:
            raise RuntimeError("load() must be called before generate()")
        image = Path(image_path).resolve()
        if not image.is_file():
            raise FileNotFoundError(f"image not found: {image}")

        import torch
        from PIL import Image

        with Image.open(image) as handle:
            rgb_image = handle.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        inputs = inputs.to(model_device)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                **settings.as_generate_kwargs(),
            )
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - started
        trimmed = [
            output[len(source) :]
            for source, output in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return {
            "output_text": output_text,
            "input_tokens": int(inputs.input_ids.shape[-1]),
            "output_tokens": int(trimmed[0].shape[-1]),
            "inference_seconds": inference_seconds,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        }
