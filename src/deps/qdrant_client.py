from uuid import uuid4
from typing import List, Dict, Optional, Union

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse


class QdrantVectorStore:
    def __init__(self, uri: str, api_key: str):
        self.uri = uri
        self.api_key = api_key
        self.client = QdrantClient(url=self.uri, api_key=self.api_key)

    def create_collection(
        self,
        collection_name: str,
        embedding_size: int,
        distance: Union[str, models.Distance] = "cosine",
    ) -> None:
        if isinstance(distance, str):
            distance = self._parse_distance_metric(distance)

        try:
            if not self.client.collection_exists(collection_name=collection_name):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=embedding_size,
                        distance=distance,
                    ),
                    on_disk_payload=True,
                )
        except UnexpectedResponse as e:
            raise RuntimeError(f"Failed to create collection '{collection_name}': {e}")

    def list_collections(self) -> list[str]:
        """List all collection names in Qdrant"""
        collections = self.client.get_collections()
        return [c.name for c in collections.collections]

    def get_collection_info(self, collection_name: str) -> dict | None:
        """Return basic stats for a collection, or None if it doesn't exist."""
        if not self.client.collection_exists(collection_name):
            return None
        info = self.client.get_collection(collection_name)
        vectors_config = info.config.params.vectors
        if hasattr(vectors_config, "size"):
            dimensions = vectors_config.size
            distance = str(vectors_config.distance.value).lower()
        else:
            first = next(iter(vectors_config.values()))
            dimensions = first.size
            distance = str(first.distance.value).lower()
        return {
            "vectors_count": info.points_count or 0,
            "dimensions": dimensions,
            "distance": distance,
            "status": str(info.status.value).lower(),
        }

    def delete_collection(self, collection_name: str) -> None:
        try:
            if self.client.collection_exists(collection_name):
                print(f"Removing existing Qdrant collection: {collection_name}")
                self.client.delete_collection(collection_name)
        except Exception as e:
            print(f"Warning: could not delete collection {collection_name}: {e}")
            raise RuntimeError(f"Failed to delete collection '{collection_name}': {e}")

    def add_embeddings(
        self,
        collection_name: str,
        embeddings: List[List[float]],
        payloads: Optional[List[Dict]] = None,
        ids: Optional[List[Union[str, int]]] = None,
        batch_size: int = 32,
    ) -> None:
        if payloads and len(payloads) != len(embeddings):
            raise ValueError("Payloads must match the number of embeddings.")

        if ids and len(ids) != len(embeddings):
            raise ValueError("IDs must match the number of embeddings.")

        points = [
            models.PointStruct(
                id=ids[i] if ids else str(uuid4()),
                vector=embedding,
                payload=payloads[i] if payloads else None,
            )
            for i, embedding in enumerate(embeddings)
        ]

        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=collection_name, points=batch)

    def query(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
    ) -> List[models.ScoredPoint]:
        return self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

    @staticmethod
    def _parse_distance_metric(metric: str) -> models.Distance:
        """Convert a distance metric string to the corresponding Qdrant Distance enum."""
        metric = metric.lower()
        match metric:
            case "cosine":
                return models.Distance.COSINE
            case "euclid":
                return models.Distance.EUCLID
            case "dot":
                return models.Distance.DOT
            case "manhattan":
                return models.Distance.MANHATTAN
            case _:
                raise ValueError(f"Unsupported distance metric: '{metric}'")
