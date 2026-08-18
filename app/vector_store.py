from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue

from app.config import Settings
from app.models import Source


def point_id(document_id: str, chunk_index: int) -> int:
    """Generate a unique point ID from document_id and chunk_index."""
    # Combine document_id hash and chunk_index into a unique integer
    doc_hash = abs(hash(document_id)) % (2**31)
    return (doc_hash << 16) | (chunk_index & 0xFFFF)


class VectorStore:
    """Qdrant vector store for RAG retrieval."""

    def __init__(self, settings: Settings) -> None:
        self.client = QdrantClient(url=settings.qdrant_url)
        self.settings = settings

    def ensure_collection(self, vector_size: int) -> None:
        """Create collection if it doesn't exist."""
        try:
            self.client.get_collection(self.settings.qdrant_collection)
        except Exception:
            # Collection doesn't exist, create it
            self.client.create_collection(
                collection_name=self.settings.qdrant_collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def upsert(self, points: list) -> None:
        """Add or update points in the collection."""
        self.client.upsert(
            collection_name=self.settings.qdrant_collection,
            points=points,
        )

    def search(self, query_vector: list[float], top_k: int, document_ids: list[str] | None = None) -> list[Source]:
        """Search for similar vectors and return relevant sources."""
        # Build filter if document_ids are specified
        search_filter = None
        if document_ids:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=doc_id),
                    )
                    for doc_id in document_ids
                ]
            )

        results = self.client.search(
            collection_name=self.settings.qdrant_collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=search_filter,
        )

        sources = []
        for result in results:
            payload = result.payload
            sources.append(
                Source(
                    document_id=payload["document_id"],
                    filename=payload["filename"],
                    chunk_index=payload["chunk_index"],
                    text=payload["text"],
                    relevance_score=result.score,
                )
            )

        return sources
