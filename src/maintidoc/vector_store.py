from __future__ import annotations

import atexit
import uuid
from functools import lru_cache

from qdrant_client import QdrantClient, models

from maintidoc.config import Settings
from maintidoc.models import DocumentElement, ElementType, SearchHit

_OPEN_CLIENTS: list[QdrantClient] = []


@atexit.register
def _close_clients() -> None:
    for client in _OPEN_CLIENTS:
        client.close()


@lru_cache(maxsize=4)
def _shared_client(url: str | None, api_key: str | None, local_path: str) -> QdrantClient:
    """Partage le client local pour éviter plusieurs verrous sur le même dossier."""
    if url:
        client = QdrantClient(url=url, api_key=api_key)
    else:
        client = QdrantClient(path=local_path, force_disable_check_same_thread=True)
    _OPEN_CLIENTS.append(client)
    return client


class QdrantVectorStore:
    def __init__(self, config: Settings) -> None:
        self.config = config
        self.client = _shared_client(
            config.qdrant_url,
            config.qdrant_api_key,
            str(config.qdrant_path),
        )
        self._ensure_collection(config.text_collection, config.text_embedding_size)
        self._ensure_collection(config.image_collection, config.image_embedding_size)

    def _ensure_collection(self, name: str, size: int) -> None:
        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
            )

    @staticmethod
    def _point_id(element_id: str, document_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"maintidoc:{document_id}:{element_id}"))

    def replace_document(
        self,
        document_id: str,
        text_elements: list[DocumentElement],
        text_vectors: list[list[float]],
        image_elements: list[DocumentElement],
        image_vectors: list[list[float]],
    ) -> None:
        self.delete_document(document_id)
        self._upsert(self.config.text_collection, text_elements, text_vectors)
        self._upsert(self.config.image_collection, image_elements, image_vectors)

    def delete_document(self, document_id: str) -> None:
        selector = models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=document_id)
                    )
                ]
            )
        )
        for collection in (self.config.text_collection, self.config.image_collection):
            self.client.delete(collection_name=collection, points_selector=selector, wait=True)

    def _upsert(
        self,
        collection: str,
        elements: list[DocumentElement],
        vectors: list[list[float]],
    ) -> None:
        if len(elements) != len(vectors):
            raise ValueError("Le nombre d'éléments et de vecteurs doit être identique.")
        if not elements:
            return
        points = []
        for element, vector in zip(elements, vectors, strict=True):
            payload = {
                "document_id": element.document_id,
                "file_name": element.file_name,
                "page": element.page,
                "element_id": element.element_id,
                "element_type": element.element_type.value,
                "content": element.content,
                "image_path": str(element.image_path) if element.image_path else None,
            }
            points.append(
                models.PointStruct(
                    id=self._point_id(element.element_id, element.document_id),
                    vector=vector,
                    payload=payload,
                )
            )
        self.client.upsert(collection_name=collection, points=points, wait=True)

    def search_text(self, vector: list[float], limit: int) -> list[SearchHit]:
        return self._search(self.config.text_collection, vector, limit)

    def search_images(self, vector: list[float], limit: int) -> list[SearchHit]:
        return self._search(self.config.image_collection, vector, limit)

    def _search(self, collection: str, vector: list[float], limit: int) -> list[SearchHit]:
        results = self.client.query_points(
            collection_name=collection,
            query=vector,
            limit=limit,
            with_payload=True,
        ).points
        hits: list[SearchHit] = []
        for point in results:
            payload = point.payload or {}
            hits.append(
                SearchHit(
                    document_id=str(payload["document_id"]),
                    file_name=str(payload["file_name"]),
                    page=int(payload["page"]),
                    element_id=str(payload["element_id"]),
                    element_type=ElementType(str(payload["element_type"])),
                    content=str(payload.get("content", "")),
                    score=float(point.score),
                    image_path=payload.get("image_path"),
                )
            )
        return hits

    def counts(self) -> dict[str, int]:
        return {
            "text_and_tables": self.client.count(self.config.text_collection, exact=True).count,
            "images": self.client.count(self.config.image_collection, exact=True).count,
        }
