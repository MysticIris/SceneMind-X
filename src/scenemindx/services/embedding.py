"""ChineseCLIP image-text embedding adapters."""

from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Any, Sequence


class DeterministicVisualEmbeddingService:
    """Dependency-light visual baseline for product retrieval.

    This is an explicit engineering baseline, not a trained Retriever. It
    combines RGB histograms with a coarse luminance grid so image search uses
    real image content even when no neural embedding checkpoint is configured.
    """

    model_id = "scenemindx/deterministic-visual-baseline"
    model_revision = "color-grid-v1"

    def load(self) -> dict[str, Any]:
        """Load the requested value."""
        return self.status()

    def status(self) -> dict[str, Any]:
        """Execute the status operation."""
        return {
            "status": "ready",
            "loaded": True,
            "backend": "deterministic_visual_baseline",
            "model": self.model_id,
            "model_revision": self.model_revision,
            "device": "cpu",
            "trained": False,
        }

    @staticmethod
    def _normalize(values: list[float]) -> list[float]:
        norm = sum(value * value for value in values) ** 0.5
        return [value / max(norm, 1e-12) for value in values]

    def encode_image(self, image_path: Path) -> list[float]:
        """Execute the encode image operation."""
        from PIL import Image

        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
            histogram = image.histogram()
            color = []
            for channel in range(3):
                base = channel * 256
                total = max(1, sum(histogram[base : base + 256]))
                color.extend(
                    sum(histogram[base + index : base + index + 32]) / total
                    for index in range(0, 256, 32)
                )
            grid_image = image.convert("L").resize((4, 4), Image.Resampling.BILINEAR)
            grid = [value / 255.0 for value in list(grid_image.getdata())]
        return self._normalize([*color, *grid])

    def encode_text(self, text: str) -> list[float]:
        """Execute the encode text operation."""
        raise RuntimeError("deterministic_visual_baseline_uses_lexical_text_scoring")


class DisabledEmbeddingService:
    """Provide disabled embedding service behavior."""
    def status(self) -> dict[str, Any]:
        """Execute the status operation."""
        return {"status": "disabled", "loaded": False}

    def encode_image(self, image_path: Path) -> list[float]:
        """Execute the encode image operation."""
        raise RuntimeError("embedding_service_disabled")

    def encode_text(self, text: str) -> list[float]:
        """Execute the encode text operation."""
        raise RuntimeError("embedding_service_disabled")


class ChineseCLIPEmbeddingService:
    """Use one audited local ChineseCLIP checkpoint for both modalities."""

    def __init__(
        self,
        model_path: Path,
        *,
        model_id: str = "OFA-Sys/chinese-clip-vit-base-patch16",
        model_revision: str = "pending_server_inventory",
        prefer_cuda: bool = True,
    ) -> None:
        self.model_path = model_path.resolve()
        self.model_id = model_id
        self.model_revision = model_revision
        self.prefer_cuda = prefer_cuda
        self.model: Any | None = None
        self.processor: Any | None = None
        self.device = "cpu"
        self.load_seconds: float | None = None

    def load(self) -> dict[str, Any]:
        """Load the requested value."""
        if self.model is not None:
            return self.status()
        if not self.model_path.is_dir():
            raise FileNotFoundError(self.model_path)
        import torch
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor

        self.device = "cuda" if self.prefer_cuda and torch.cuda.is_available() else "cpu"
        started = time.perf_counter()
        self.processor = ChineseCLIPProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.model = ChineseCLIPModel.from_pretrained(self.model_path, local_files_only=True).to(self.device)
        self.model.eval()
        self.load_seconds = time.perf_counter() - started
        return self.status()

    def status(self) -> dict[str, Any]:
        """Execute the status operation."""
        return {
            "status": "ready" if self.model is not None else "not_loaded",
            "loaded": self.model is not None,
            "model": self.model_id,
            "model_revision": self.model_revision,
            "device": self.device,
            "load_seconds": self.load_seconds,
        }

    @staticmethod
    def _normalized(values: Any) -> list[float]:
        import torch

        vector = values[0].float()
        vector = vector / torch.linalg.vector_norm(vector).clamp_min(1e-12)
        return [float(value) for value in vector.detach().cpu().tolist()]

    def encode_image(self, image_path: Path) -> list[float]:
        """Execute the encode image operation."""
        if self.model is None or self.processor is None:
            raise RuntimeError("load() must be called before encode_image")
        import torch
        from PIL import Image

        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                values = self.model.get_image_features(**inputs)
            return self._normalized(values)
        finally:
            image.close()

    def encode_text(self, text: str) -> list[float]:
        """Execute the encode text operation."""
        if self.model is None or self.processor is None:
            raise RuntimeError("load() must be called before encode_text")
        import torch

        inputs = self.processor(text=[text], padding=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            values = self.model.get_text_features(**inputs)
        return self._normalized(values)


class ModelScopeChineseCLIPEmbeddingService:
    """Load the audited Alibaba IIC Chinese-CLIP checkpoint directly.

    ModelScope's generic multi-modal pipeline imports unrelated audio/OFA
    preprocessors. This adapter uses the same official model class and mirrors
    its CLIPPreprocessor image and token transforms without that import fanout.
    """

    IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
    IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        model_path: Path,
        *,
        model_id: str = "iic/multi-modal_clip-vit-base-patch16_zh",
        model_revision: str = "v1.0.1",
        prefer_cuda: bool = True,
    ) -> None:
        self.model_path = model_path.resolve()
        self.model_id = model_id
        self.model_revision = model_revision
        self.prefer_cuda = prefer_cuda
        self.model: Any | None = None
        self.image_transform: Any | None = None
        self.device = "cpu"
        self.load_seconds: float | None = None

    def load(self) -> dict[str, Any]:
        """Load the requested value."""
        if self.model is not None:
            return self.status()
        if not self.model_path.is_dir():
            raise FileNotFoundError(self.model_path)

        import torch
        from PIL import Image
        from modelscope.models.multi_modal.clip.model import CLIPForMultiModalEmbedding
        from torchvision.transforms import Compose, Normalize, Resize, ToTensor

        self.device = "cuda" if self.prefer_cuda and torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            raise RuntimeError("modelscope_iic_chinese_clip_requires_cuda")

        config_path = self.model_path / "vision_model_config.json"
        with config_path.open("r", encoding="utf-8") as handle:
            image_resolution = int(json.load(handle)["image_resolution"])

        started = time.perf_counter()
        self.model = CLIPForMultiModalEmbedding(str(self.model_path))
        self.model.eval()
        self.image_transform = Compose(
            [
                Resize((image_resolution, image_resolution), interpolation=Image.Resampling.BICUBIC),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(self.IMAGE_MEAN, self.IMAGE_STD),
            ]
        )
        self.load_seconds = time.perf_counter() - started
        return self.status()

    def status(self) -> dict[str, Any]:
        """Execute the status operation."""
        return {
            "status": "ready" if self.model is not None else "not_loaded",
            "loaded": self.model is not None,
            "backend": "modelscope_iic",
            "model": self.model_id,
            "model_revision": self.model_revision,
            "device": self.device,
            "load_seconds": self.load_seconds,
        }

    @staticmethod
    def _tokenize(tokenizer: Any, texts: Sequence[str], context_length: int = 52) -> Any:
        import torch

        rows: list[list[int]] = []
        for text in texts:
            tokens = [tokenizer.vocab["[CLS]"]]
            tokens.extend(
                tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))[: context_length - 2]
            )
            tokens.append(tokenizer.vocab["[SEP]"])
            rows.append(tokens)
        result = torch.zeros(len(rows), context_length, dtype=torch.long)
        for index, tokens in enumerate(rows):
            result[index, : len(tokens)] = torch.tensor(tokens)
        return result

    @staticmethod
    def _as_list(values: Any) -> list[float]:
        return [float(value) for value in values[0].detach().float().cpu().tolist()]

    def encode_image(self, image_path: Path) -> list[float]:
        """Execute the encode image operation."""
        if self.model is None or self.image_transform is None:
            raise RuntimeError("load() must be called before encode_image")
        from PIL import Image

        with Image.open(image_path) as handle:
            tensor = self.image_transform(handle).unsqueeze(0)
        values = self.model({"img": tensor})["img_embedding"]
        return self._as_list(values)

    def encode_text(self, text: str) -> list[float]:
        """Execute the encode text operation."""
        if self.model is None:
            raise RuntimeError("load() must be called before encode_text")
        tensor = self._tokenize(self.model.tokenizer, [text])
        values = self.model({"text": tensor})["text_embedding"]
        return self._as_list(values)
