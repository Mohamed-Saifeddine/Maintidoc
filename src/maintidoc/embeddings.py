from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from maintidoc.config import Settings


def _normalise(vector: list[float] | np.ndarray) -> list[float]:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    return (array / norm if norm else array).tolist()


def _feature_tensor(output, modality: str):
    """Unifie les sorties CLIP de Transformers 4 et 5."""
    if hasattr(output, "norm"):
        return output
    modality_attribute = f"{modality}_embeds"
    if hasattr(output, modality_attribute):
        return getattr(output, modality_attribute)
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError(f"Format de sortie CLIP inattendu pour {modality}: {type(output)!r}")


class OllamaTextEmbedder:
    """Embeddings textuels produits localement par BGE-M3 via l'API Ollama."""

    def __init__(self, config: Settings) -> None:
        self.base_url = config.ollama_base_url.rstrip("/")
        self.model = config.embed_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=300,
        )
        response.raise_for_status()
        vectors = response.json().get("embeddings", [])
        if len(vectors) != len(texts):
            raise RuntimeError("Ollama n'a pas renvoyé un embedding pour chaque texte.")
        return [_normalise(vector) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class ClipEncoder:
    """Encode les images et les requêtes texte dans le même espace CLIP."""

    def __init__(self, config: Settings) -> None:
        self.model_name = config.clip_model
        self._model = None
        self._processor = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name).to(self._device)
        self._model.eval()

    def embed_images(self, paths: list[Path]) -> list[list[float]]:
        if not paths:
            return []
        self._load()
        import torch

        images = [Image.open(path).convert("RGB") for path in paths]
        inputs = self._processor(images=images, return_tensors="pt", padding=True)
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.inference_mode():
            vectors = _feature_tensor(self._model.get_image_features(**inputs), "image")
            vectors = vectors / vectors.norm(dim=-1, keepdim=True)
        return vectors.cpu().numpy().astype(np.float32).tolist()

    def embed_text(self, text: str) -> list[float]:
        self._load()
        import torch

        inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.inference_mode():
            vector = _feature_tensor(self._model.get_text_features(**inputs), "text")
            vector = vector / vector.norm(dim=-1, keepdim=True)
        return vector[0].cpu().numpy().astype(np.float32).tolist()


class OllamaVisionCaptioner:
    """Description optionnelle des images avec Qwen2.5-VL exécuté par Ollama."""

    def __init__(self, config: Settings) -> None:
        self.base_url = config.ollama_base_url.rstrip("/")
        self.model = config.vision_model

    def describe(self, image_path: Path, page_context: str = "") -> str:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        prompt = (
            "Décris précisément cette image extraite d'un rapport. Identifie les personnes, "
            "objets, graphiques, nombres et textes visibles. Ne déduis rien qui n'est pas visible. "
            f"Contexte textuel de la page : {page_context[:1200]}"
        )
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "images": [encoded], "stream": False},
            timeout=300,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
