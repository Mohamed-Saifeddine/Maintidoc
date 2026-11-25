from __future__ import annotations

from pathlib import Path

from maintidoc.config import Settings
from maintidoc.embeddings import ClipEncoder, OllamaTextEmbedder
from maintidoc.extraction import MultimodalPDFExtractor, ProgressCallback
from maintidoc.models import ElementType, IndexingStats
from maintidoc.vector_store import QdrantVectorStore


class DocumentIndexer:
    def __init__(self, config: Settings) -> None:
        config.ensure_directories()
        self.config = config
        self.extractor = MultimodalPDFExtractor(config)
        self.text_embedder = OllamaTextEmbedder(config)
        self.clip = ClipEncoder(config)
        self.store = QdrantVectorStore(config)

    def index(self, pdf_path: Path, progress: ProgressCallback | None = None) -> IndexingStats:
        elements, page_count = self.extractor.extract(pdf_path, progress)
        text_elements = [
            element for element in elements if element.element_type != ElementType.IMAGE
        ]
        image_elements = [
            element for element in elements if element.element_type == ElementType.IMAGE
        ]

        if progress:
            progress("Embeddings BGE-M3 du texte et des tableaux", 0.86)
        text_vectors = self.text_embedder.embed([element.content for element in text_elements])

        if progress:
            progress("Embeddings CLIP des images", 0.91)
        image_vectors = self.clip.embed_images(
            [element.image_path for element in image_elements if element.image_path is not None]
        )

        if progress:
            progress("Écriture dans Qdrant", 0.97)
        document_id = elements[0].document_id if elements else pdf_path.stem
        self.store.replace_document(
            document_id,
            text_elements,
            text_vectors,
            image_elements,
            image_vectors,
        )
        if progress:
            progress("Indexation terminée", 1.0)

        return IndexingStats(
            document_id=document_id,
            pages=page_count,
            text_chunks=sum(element.element_type == ElementType.TEXT for element in text_elements),
            tables=sum(element.element_type == ElementType.TABLE for element in text_elements),
            images=len(image_elements),
        )
