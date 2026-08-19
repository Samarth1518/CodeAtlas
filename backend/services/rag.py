"""
services/rag.py — Retrieval-Augmented Generation pipeline for CodeAtlas.

Orchestrates the full RAG flow:
  1. Validate repository URL
  2. Embed the user's question using the local embedding service
  3. Search the persistent vector index for relevant code chunks
  4. Deduplicate identical chunk IDs
  5. Build a compact context and send to the LLM service
  6. Return the grounded answer and source citations

This module reuses the existing embedding and vector-store services and does not
duplicate any semantic-search logic.

Security:
    - GitHub tokens and LLM API keys are never passed through this module.
    - Only sanitised, already-filtered code chunks from the vector index are used.
    - Private/inaccessible repositories are excluded by the upstream indexing pipeline.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from services.embeddings import embed_query
from services.llm import generate_answer
from services.vector_store import default_vector_store

# ── Configuration ─────────────────────────────────────────────────────────────

_GITHUB_RE = re.compile(
    r"^https://github\.com/([A-Za-z0-9._-]+)/([A-Za-z0-9._-]+?)(?:\.git)?/?$"
)

_MAX_CHUNKS: int = 10        # Hard ceiling on chunks sent to LLM context
_DEFAULT_TOP_K: int = 6      # Default number of chunks to retrieve


# ── Public API ────────────────────────────────────────────────────────────────

def answer_question(
    repo_url: str,
    question: str,
    top_k: int = _DEFAULT_TOP_K,
) -> Dict[str, Any]:
    """
    Full RAG pipeline: embed question → search index → generate LLM answer.

    Args:
        repo_url: Full GitHub repository URL (must be a public repository).
        question: Natural-language developer question.
        top_k: Number of top-ranked chunks to retrieve (capped at _MAX_CHUNKS).

    Returns:
        Dictionary with keys:
            "answer"  — Markdown-formatted answer string from the LLM.
            "sources" — List of source citation dicts (subset of chunk metadata).

    Raises:
        ValueError: On invalid repo URL or empty question.
        RuntimeError: On embedding errors, LLM errors, or missing index.
    """
    # 1. Validate inputs
    repo_url = _validate_repo_url(repo_url)
    question = str(question).strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    top_k = max(1, min(int(top_k), _MAX_CHUNKS))

    # 2. Check if repository vector index exists
    index_data = default_vector_store._load_repo_index(repo_url)
    if not index_data or len(index_data.get("metadata", [])) == 0:
        return {
            "answer": (
                "**Search Index Not Found**\n\n"
                "The vector search index for this repository has not been built yet or is no longer available in server memory.\n\n"
                "Please click **⚡ Build Search Index** in the Semantic Search section above so that I can retrieve and analyze the relevant code chunks."
            ),
            "sources": [],
        }

    # 3. Embed the question
    try:
        query_embedding = embed_query(question)
    except Exception as exc:
        raise RuntimeError(f"Failed to generate question embedding: {exc}") from exc

    # 4. Search vector index
    try:
        raw_results = default_vector_store.search(
            repo_url=repo_url,
            query_embedding=query_embedding,
            top_k=top_k,
            query_text=question,
        )
    except Exception as exc:
        raise RuntimeError(f"Vector store search failed: {exc}") from exc

    # 5. Deduplicate by chunk_id (keep highest-scored occurrence)
    chunks = _deduplicate_chunks(raw_results)

    # 6. Generate grounded answer
    # Extract repository full name for prompt context (e.g. "owner/repo")
    repo_display = _extract_repo_name(repo_url)

    answer = generate_answer(
        question=question,
        context_chunks=chunks,
        repository=repo_display,
    )

    # 7. Build source citations (exclude raw content to keep response lean)
    sources = _build_sources(chunks)

    return {
        "answer": answer,
        "sources": sources,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_repo_url(repo_url: str) -> str:
    """Validates and normalises a GitHub repository URL."""
    url = str(repo_url).strip()
    match = _GITHUB_RE.match(url)
    if not match:
        raise ValueError(
            "Invalid GitHub repository URL. "
            "Expected format: https://github.com/owner/repository"
        )
    owner, repo = match.groups()
    return f"https://github.com/{owner}/{repo}"


def _deduplicate_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Removes duplicate chunk IDs, keeping the first occurrence (highest score
    since results are pre-sorted by descending similarity).
    """
    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for chunk in chunks:
        cid = chunk.get("chunk_id", "")
        if cid and cid in seen:
            continue
        seen.add(cid)
        deduped.append(chunk)
    return deduped


def _build_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts raw chunk records into lean source citation objects for the API response.
    Excludes full content (already used in the prompt) to keep response payloads small.
    """
    sources: List[Dict[str, Any]] = []
    for chunk in chunks:
        sources.append(
            {
                "chunk_id": chunk.get("chunk_id", ""),
                "file_path": chunk.get("file_path", ""),
                "language": chunk.get("language"),
                "start_line": chunk.get("start_line", 1),
                "end_line": chunk.get("end_line", 1),
                "score": chunk.get("score", 0.0),
            }
        )
    return sources


def _extract_repo_name(repo_url: str) -> str:
    """Returns 'owner/repo' from a full GitHub URL, or the URL itself on failure."""
    match = _GITHUB_RE.match(repo_url)
    if match:
        owner, repo = match.groups()
        return f"{owner}/{repo}"
    return repo_url
