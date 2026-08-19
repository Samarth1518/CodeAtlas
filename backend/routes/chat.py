"""
routes/chat.py — AI Codebase Assistant endpoint.

Provides POST /api/chat which orchestrates the full RAG pipeline:
  1. Validates the repository URL and question.
  2. Delegates to the RAG service (embed → search → LLM generate).
  3. Returns the grounded answer and source citations.

Security:
    - GITHUB_TOKEN and LLM_API_KEY are never returned or logged.
    - Question length is capped at 500 characters.
    - top_k is capped at 10.
"""

import re
from flask import Blueprint, jsonify, request

from services.rag import answer_question

chat_bp = Blueprint("chat", __name__)

_GITHUB_RE = re.compile(
    r"^https://github\.com/([A-Za-z0-9._-]+)/([A-Za-z0-9._-]+?)(?:\.git)?/?$"
)

_MAX_QUESTION_LENGTH: int = 500
_DEFAULT_TOP_K: int = 6
_MAX_TOP_K: int = 10


@chat_bp.route("", methods=["POST"])
@chat_bp.route("/", methods=["POST"])
def chat():
    """
    POST /api/chat

    Request JSON:
    {
        "repo_url": "https://github.com/owner/repository",
        "question": "How does authentication work?",
        "top_k": 6
    }

    Response JSON (success):
    {
        "success": true,
        "answer": "...",
        "sources": [
            {
                "file_path": "backend/auth.py",
                "start_line": 10,
                "end_line": 42,
                "language": "Python",
                "score": 0.82,
                "chunk_id": "..."
            }
        ]
    }

    Response JSON (error):
    {
        "success": false,
        "error": "..."
    }
    """
    # ── Parse body ───────────────────────────────────────────────────────────
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"success": False, "error": "Request body must be valid JSON."}), 400

    # ── Validate repo_url ────────────────────────────────────────────────────
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
    repo_url = f"https://github.com/{owner}/{repo}"

    # ── Validate question ────────────────────────────────────────────────────
    raw_question = body.get("question")
    if raw_question is None or not str(raw_question).strip():
        return jsonify({"success": False, "error": "Question cannot be empty."}), 400

    question = str(raw_question).strip()
    if len(question) > _MAX_QUESTION_LENGTH:
        return (
            jsonify(
                {
                    "success": False,
                    "error": (
                        f"Question too long ({len(question)} characters). "
                        f"Maximum length is {_MAX_QUESTION_LENGTH} characters."
                    ),
                }
            ),
            400,
        )

    # ── Validate top_k ───────────────────────────────────────────────────────
    top_k = body.get("top_k", _DEFAULT_TOP_K)
    try:
        top_k = int(top_k)
        if top_k < 1:
            top_k = _DEFAULT_TOP_K
        top_k = min(top_k, _MAX_TOP_K)
    except (ValueError, TypeError):
        top_k = _DEFAULT_TOP_K

    # ── Run RAG pipeline ─────────────────────────────────────────────────────
    try:
        result = answer_question(
            repo_url=repo_url,
            question=question,
            top_k=top_k,
        )
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 422
    except RuntimeError as exc:
        error_msg = str(exc)
        # Return 503 for LLM not configured, 500 for other runtime errors
        status = 503 if "LLM_API_KEY is not configured" in error_msg else 500
        return jsonify({"success": False, "error": error_msg}), status
    except Exception as exc:
        return jsonify({"success": False, "error": f"Unexpected error: {exc}"}), 500

    return (
        jsonify(
            {
                "success": True,
                "answer": result["answer"],
                "sources": result["sources"],
            }
        ),
        200,
    )
