"""
Endee Vector Database client wrapper.
Handles index creation, vector storage, and similarity search.
"""

import logging
from typing import Optional
from endee import Endee, Precision
from app.config import settings

logger = logging.getLogger(__name__)

# Global client and index references
_client: Optional[Endee] = None
_index = None


def get_client() -> Endee:
    """Get or create the Endee client singleton."""
    global _client
    if _client is None:
        _client = Endee(settings.ENDEE_AUTH_TOKEN or "")
        _client.set_base_url(settings.ENDEE_URL)
        logger.info(f"Connected to Endee at {settings.ENDEE_URL}")
    return _client


def init_index() -> None:
    """
    Initialize the vector index in Endee.
    Creates the index if it doesn't exist (384-dim, cosine, INT8).
    """
    global _index
    client = get_client()

    try:
        _index = client.get_index(name=settings.INDEX_NAME)
        logger.info(f"Using existing index: {settings.INDEX_NAME}")
    except Exception:
        # Index doesn't exist — create it
        client.create_index(
            name=settings.INDEX_NAME,
            dimension=settings.INDEX_DIMENSION,
            space_type="cosine",
            precision=Precision.INT8D,
        )
        _index = client.get_index(name=settings.INDEX_NAME)
        logger.info(f"Created new index: {settings.INDEX_NAME}")


def get_index():
    """Get the initialized index, creating it if needed."""
    global _index
    if _index is None:
        init_index()
    return _index


def store_vectors(
    ids: list[str],
    vectors: list[list[float]],
    metadata: list[dict],
) -> None:
    """
    Store vectors with metadata in Endee.

    Args:
        ids: Unique identifier for each vector
        vectors: Embedding vectors (384-dim)
        metadata: Payload metadata (source, text, chunk_index, etc.)
    """
    index = get_index()
    items = [
        {"id": vid, "vector": vec, "meta": meta}
        for vid, vec, meta in zip(ids, vectors, metadata)
    ]

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        index.upsert(batch)
        logger.debug(f"Upserted batch {i // batch_size + 1}")

    logger.info(f"Stored {len(items)} vectors in Endee")


def search_vectors(
    query_vector: list[float],
    top_k: int = 5,
) -> list[dict]:
    """
    Search for similar vectors in Endee.

    Args:
        query_vector: Query embedding (384-dim)
        top_k: Number of results to return

    Returns:
        List of results with id, similarity score, and metadata
    """
    index = get_index()
    results = index.query(vector=query_vector, top_k=top_k)

    formatted = []
    for result in results:
        formatted.append({
            "id": result.id if hasattr(result, "id") else result.get("id", ""),
            "score": result.similarity if hasattr(result, "similarity") else result.get("similarity", 0.0),
            "metadata": result.meta if hasattr(result, "meta") else result.get("meta", {}),
        })

    logger.info(f"Found {len(formatted)} results for query")
    return formatted


def delete_vectors_by_document(document_id: str) -> None:
    """
    Delete all vectors associated with a specific document.

    Args:
        document_id: The document identifier to remove
    """
    index = get_index()
    try:
        # Endee SDK uses delete_vector() for single vector deletion
        index.delete_vector(document_id)
        logger.info(f"Deleted vector: {document_id}")
    except Exception as e:
        logger.warning(f"Could not delete vectors for {document_id}: {e}")
