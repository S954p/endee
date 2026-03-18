"""
Embedding generation module using Hugging Face Inference API.
Provides vector representations of text for similarity search.

Uses the free Hugging Face Inference API — no PyTorch or native
DLL dependencies required.
"""

import logging
import numpy as np
from huggingface_hub import InferenceClient

from app.config import settings

logger = logging.getLogger(__name__)

# Global client reference
_client: InferenceClient | None = None


def _get_client() -> InferenceClient:
    """Get or create the Hugging Face InferenceClient singleton."""
    global _client
    if _client is None:
        logger.info(f"Initializing HF Inference client for model: {settings.EMBEDDING_MODEL}")
        token = settings.HF_TOKEN if settings.HF_TOKEN else None
        _client = InferenceClient(token=token)
        logger.info("HF Inference client ready")
    return _client


def _normalize(vectors: list[list[float]]) -> list[list[float]]:
    """L2-normalize a batch of embedding vectors."""
    arr = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-9, a_max=None)
    normalized = arr / norms
    return normalized.tolist()


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed

    Returns:
        List of embedding vectors (384-dim float arrays)
    """
    client = _get_client()
    model_name = f"sentence-transformers/{settings.EMBEDDING_MODEL}"

    # Process in batches to avoid payload limits
    batch_size = 32
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = client.feature_extraction(batch, model=model_name)

        # result is a numpy array or nested list of shape (batch, dim)
        if hasattr(result, "tolist"):
            batch_embeddings = result.tolist()
        else:
            batch_embeddings = result

        all_embeddings.extend(batch_embeddings)

    # Normalize embeddings
    all_embeddings = _normalize(all_embeddings)

    return all_embeddings


def get_single_embedding(text: str) -> list[float]:
    """
    Generate an embedding for a single text string.

    Args:
        text: Text to embed

    Returns:
        Embedding vector (384-dim float array)
    """
    return get_embeddings([text])[0]
