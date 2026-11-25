from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    root_dir: Path = ROOT_DIR
    data_dir: Path = ROOT_DIR / "data"
    qdrant_path: Path = ROOT_DIR / "data" / "qdrant"
    extracted_dir: Path = ROOT_DIR / "data" / "extracted"
    uploads_dir: Path = ROOT_DIR / "data" / "uploads"

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3:latest")
    vision_model: str = os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:7b")
    agent_model: str = os.getenv("OLLAMA_AGENT_MODEL", "qwen3.5:9b")
    clip_model: str = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")

    qdrant_url: str | None = os.getenv("QDRANT_URL") or None
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    text_collection: str = "maintidoc_text"
    image_collection: str = "maintidoc_images"

    text_embedding_size: int = 1024
    image_embedding_size: int = 512
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    text_top_k: int = int(os.getenv("TEXT_TOP_K", "5"))
    image_top_k: int = int(os.getenv("IMAGE_TOP_K", "4"))
    enable_vlm_captions: bool = _as_bool(os.getenv("ENABLE_VLM_CAPTIONS"), False)
    max_images_per_document: int = int(os.getenv("MAX_IMAGES_PER_DOCUMENT", "40"))

    @property
    def default_pdf(self) -> Path:
        return self.data_dir / "source" / "unicef_bulgaria_annual_report_2024.pdf"

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.extracted_dir, self.uploads_dir):
            path.mkdir(parents=True, exist_ok=True)
        if self.qdrant_url is None:
            self.qdrant_path.mkdir(parents=True, exist_ok=True)


settings = Settings()
