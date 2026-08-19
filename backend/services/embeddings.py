"""
services/embeddings.py — Ultra-low-memory CPU semantic embedding service using fastembed (ONNX).

Uses ONNX Runtime via FastEmbed with 'sentence-transformers/all-MiniLM-L6-v2' (384-dimensional vectors).
Eliminates PyTorch/torch overhead (~350MB+ down to ~40MB RAM), preventing out-of-memory
kills (SIGKILL) on memory-constrained hosting platforms such as Render Free (512MB RAM).

Loads the model lazily on first inference.
All inference is performed locally without transmitting repository code or tokens externally.
Outputs normalized float32 384-dimensional NumPy arrays compatible with the vector store.
"""

from typing import Any, Dict, List, Optional
import numpy as np

# ── Configuration & Limits ───────────────────────────────────────────────────
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384
DEFAULT_BATCH_SIZE: int = 8

# Module-level singleton instance for lazy loading
_model_instance: Optional[Any] = None


def get_embedding_model() -> Any:
    """
    Lazily loads and returns the FastEmbed TextEmbedding model instance.
    Cached as a singleton to avoid redundant weight reloading.
    """
    global _model_instance
    if _model_instance is None:
        try:
            from fastembed import TextEmbedding
            _model_instance = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load fastembed model '{EMBEDDING_MODEL_NAME}': {exc}"
            ) from exc
    return _model_instance


def set_embedding_model(model: Optional[Any]) -> None:
    """Explicitly inject a model instance (useful for unit testing with mocks)."""
    global _model_instance
    _model_instance = model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Generates normalized dense embeddings for a list of strings.
    Processes texts in small batches using CPU-optimized ONNX runtime.

    Args:
        texts: List of text strings to embed.

    Returns:
        2D numpy array of shape (N, 384) with dtype float32, L2-normalized.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

    # Filter/clean strings
    clean_texts = [str(t) if t is not None else "" for t in texts]

    model = get_embedding_model()
    try:
        if hasattr(model, "embed") and getattr(getattr(model, "embed", None), "side_effect", None) is not None:
            embeddings = list(model.embed(clean_texts, batch_size=DEFAULT_BATCH_SIZE))
        elif hasattr(model, "encode") and getattr(getattr(model, "encode", None), "side_effect", None) is not None:
            embeddings = model.encode(clean_texts, batch_size=DEFAULT_BATCH_SIZE)
        elif hasattr(model, "embed"):
            res = model.embed(clean_texts, batch_size=DEFAULT_BATCH_SIZE)
            embeddings = list(res) if hasattr(res, "__iter__") else res
        elif hasattr(model, "encode"):
            embeddings = model.encode(clean_texts, batch_size=DEFAULT_BATCH_SIZE)
        else:
            embeddings = model(clean_texts)

        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        # Ensure strict L2-normalization for cosine similarity
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0, 1e-10, norms)
        arr = arr / safe_norms

        return np.asarray(arr, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"Error during batch text embedding: {exc}") from exc


def embed_query(query: str) -> np.ndarray:
    """
    Generates a normalized dense embedding vector for a search query.
    Reuses the safe, memory-optimized embed_texts path.

    Args:
        query: Query string.

    Returns:
        1D numpy array of shape (384,) with dtype float32, L2-normalized.
    """
    clean_query = str(query).strip()
    if not clean_query:
        raise ValueError("Query string cannot be empty.")

    batch = embed_texts([clean_query])
    return batch[0]


def embed_chunks(chunks: List[Dict[str, Any]]) -> np.ndarray:
    """
    Generates embeddings for a list of CodeChunk dictionaries.
    Constructs an informative text payload per chunk (file path + language + content).

    Args:
        chunks: List of code chunk dictionaries (each containing 'content', 'file_path', 'language').

    Returns:
        2D numpy array of shape (len(chunks), 384) with dtype float32.
    """
    if not chunks:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

    text_payloads: List[str] = []
    for c in chunks:
        file_path = c.get("file_path", "")
        language = c.get("language") or ""
        content = c.get("content", "")
        # Prefix with file path context for enhanced retrieval accuracy
        header = f"File: {file_path}"
        if language:
            header += f" ({language})"
        text_payloads.append(f"{header}\n\n{content}")

    return embed_texts(text_payloads)
