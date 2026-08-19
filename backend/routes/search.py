"""
routes/search.py — Semantic code search endpoint.

Provides POST /api/search which embeds a natural language query and performs
cosine similarity search against the repository's local vector index.
"""

import re
from flask import Blueprint, jsonify, request

from services.embeddings import embed_query
from services.vector_store import default_vector_store

search_bp = Blueprint("search", __name__)

_GITHUB_RE = re.compile(
    r"^https://github\.com/([A-Za-z0-9._-]+)/([A-Za-z0-9._-]+?)(?:\.git)?/?$"
)

_MAX_QUERY_LENGTH: int = 500
_MAX_TOP_K: int = 20
_DEFAULT_TOP_K: int = 5


@search_bp.route("", methods=["POST"])
@search_bp.route("/", methods=["POST"])
def search():
    """
    POST /api/search

    Request:
    {
        "repo_url": "https://github.com/owner/repo",
        "query": "Where is authentication handled?",
        "top_k": 5
    }

    Response:
    {
        "success": true,
        "query": "Where is authentication handled?",
        "results": [
            {
                "score": 0.8742,
                "chunk_id": "src_auth.py:L10-L38:abc12345",
                "file_path": "src/auth.py",
                "language": "Python",
                "start_line": 10,
                "end_line": 38,
                "content": "...",
                "line_count": 29,
                "char_count": 820
            }
        ]
    }
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"success": False, "error": "Request body must be valid JSON."}), 400

    raw_url = str(body.get("repo_url", "")).strip()
    if not raw_url:
        return jsonify({"success": False, "error": "Missing 'repo_url' in request body."}), 400

    match = _GITHUB_RE.match(raw_url)
    if not match:
        return (
            jsonify(
                {
                    "success": False,
                    "error": (
                        "Invalid GitHub repository URL. "
                        "Expected format: https://github.com/owner/repository"
                    ),
                }
            ),
            422,
        )

    owner, repo = match.groups()
    if repo.endswith(".git"):
        repo = repo[:-4]

    repo_url = f"https://github.com/{owner}/{repo}"

    raw_query = body.get("query")
    if raw_query is None or not str(raw_query).strip():
        return jsonify({"success": False, "error": "Query string cannot be empty."}), 400

    query = str(raw_query).strip()
    if len(query) > _MAX_QUERY_LENGTH:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Query too long ({len(query)} chars). Maximum length is {_MAX_QUERY_LENGTH} characters.",
                }
            ),
            400,
        )

    # Validate top_k
    top_k = body.get("top_k", _DEFAULT_TOP_K)
    try:
        top_k = int(top_k)
        if top_k < 1:
            top_k = _DEFAULT_TOP_K
        top_k = min(top_k, _MAX_TOP_K)
    except (ValueError, TypeError):
        top_k = _DEFAULT_TOP_K

    # 1. Embed query vector
    try:
        query_emb = embed_query(query)
    except Exception as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Failed to generate query embedding: {exc}",
                }
            ),
            500,
        )

    # 2. Search local vector store
    try:
        results = default_vector_store.search(
            repo_url=repo_url,
            query_embedding=query_emb,
            top_k=top_k,
        )
    except Exception as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Vector search error: {exc}",
                }
            ),
            500,
        )

    return (
        jsonify(
            {
                "success": True,
                "query": query,
                "results": results,
            }
        ),
        200,
    )
