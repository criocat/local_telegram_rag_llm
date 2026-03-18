from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import Settings


@dataclass
class QdrantStore:
    client: QdrantClient
    collection_name: str

    def upsert_points(self, points: list[models.PointStruct]) -> None:
        """Insert or update a list of points in the Qdrant collection."""
        if not points:
            return
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def search(
        self,
        vector: list[float],
        query_filter: models.Filter | None = None,
        limit: int = 30,
        with_payload: bool = True,
    ) -> list[Any]:
        """Search for similar points in the collection using a query vector."""
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=with_payload,
            with_vectors=True,
        )
        return response.points

    def retrieve_by_ids(self, point_ids: list[str | int]) -> list[Any]:
        """Retrieve specific points from the collection by their IDs."""
        if not point_ids:
            return []
        return self.client.retrieve(
            collection_name=self.collection_name,
            ids=point_ids,
            with_payload=True,
            with_vectors=True,
        )

    def scroll(
        self,
        query_filter: models.Filter | None = None,
        limit: int = 256,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> tuple[list[Any], Any]:
        """Scroll through points in the collection, optionally applying a filter."""
        return self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )


def build_qdrant_client(settings: Settings) -> QdrantClient:
    """Initialize and return a QdrantClient instance using the provided settings."""
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def create_collection_if_not_exists(client: QdrantClient, settings: Settings) -> None:
    """Create the Qdrant collection with appropriate vector and quantization settings if it does not already exist."""
    collections = client.get_collections().collections
    existing = {c.name for c in collections}
    if settings.qdrant_collection in existing:
        return

    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=models.VectorParams(
            size=settings.qdrant_vector_size,
            distance=models.Distance.COSINE,
            on_disk=False,
        ),
        quantization_config=models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(always_ram=True),
        ),
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=10000,
            memmap_threshold=20000,
        ),
    )


def create_store(settings: Settings) -> QdrantStore:
    """Set up the Qdrant client, ensure the collection exists, and return a QdrantStore instance."""
    client = build_qdrant_client(settings)
    create_collection_if_not_exists(client, settings)
    return QdrantStore(client=client, collection_name=settings.qdrant_collection)
