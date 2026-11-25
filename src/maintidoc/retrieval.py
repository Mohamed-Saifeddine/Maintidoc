from __future__ import annotations

from maintidoc.config import Settings
from maintidoc.embeddings import ClipEncoder, OllamaTextEmbedder
from maintidoc.models import SearchHit
from maintidoc.vector_store import QdrantVectorStore


def reciprocal_rank_fusion(result_sets: list[list[SearchHit]], k: int = 60) -> list[SearchHit]:
    scores: dict[str, float] = {}
    hits: dict[str, SearchHit] = {}
    for result_set in result_sets:
        for rank, hit in enumerate(result_set, start=1):
            key = f"{hit.document_id}:{hit.element_id}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            hits[key] = hit
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [hits[key].model_copy(update={"score": scores[key]}) for key in ordered]


class MultimodalRetriever:
    def __init__(self, config: Settings) -> None:
        config.ensure_directories()
        self.config = config
        self.text_embedder = OllamaTextEmbedder(config)
        self.clip = ClipEncoder(config)
        self.store = QdrantVectorStore(config)

    def search_text(self, query: str, limit: int | None = None) -> list[SearchHit]:
        vector = self.text_embedder.embed_query(query)
        return self.store.search_text(vector, limit or self.config.text_top_k)

    def search_images(self, query: str, limit: int | None = None) -> list[SearchHit]:
        vector = self.clip.embed_text(query)
        return self.store.search_images(vector, limit or self.config.image_top_k)

    def search_all(self, query: str, limit: int = 8) -> list[SearchHit]:
        return reciprocal_rank_fusion([self.search_text(query), self.search_images(query)])[:limit]
