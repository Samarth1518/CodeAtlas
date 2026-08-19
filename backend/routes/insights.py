"""
routes/insights.py — Repository architecture and code insights endpoint.

Provides POST /api/insights/analyze which takes already-retrieved repository
data (tree + optional source files) and returns structured architectural insights
extracted deterministically — no hallucination, no invented facts.

The optional LLM narrative summary is clearly separated from raw detected facts
and grounded strictly in the insights produced by the deterministic service.

Security:
    - Never exposes GITHUB_TOKEN or LLM_API_KEY.
    - Private repositories are blocked at the GitHub metadata level (Phase 1).
    - File content must come from the Phase 3 safe filter results.
"""

import re
from flask import Blueprint, jsonify, request

from services.insights import analyze_repository

insights_bp = Blueprint("insights", __name__)

_GITHUB_RE = re.compile(
    r"^https://github\.com/([A-Za-z0-9._-]+)/([A-Za-z0-9._-]+?)(?:\.git)?/?$"
)

_MAX_FILES_INPUT: int = 200   # Maximum source files accepted in a single request
_MAX_TREE_INPUT: int = 10_000 # Maximum tree items accepted


@insights_bp.route("/analyze", methods=["POST"])
def analyze():
    """
    POST /api/insights/analyze

    Request JSON:
    {
        "repo_url": "https://github.com/owner/repository",
        "metadata": {
            "name": "repo",
            "owner": "owner",
            "full_name": "owner/repo",
            "primary_language": "Python",
            "default_branch": "main",
            "visibility": "public"
        },
        "tree": [
            { "path": "src/app.py", "type": "file", "size": 1024 }
        ],
        "source_files": [
            { "path": "package.json", "content": "...", "language": "JSON" }
        ]
    }

    Response JSON (success):
    {
        "success": true,
        "insights": {
            "repo_url": "...",
            "metadata": {...},
            "statistics": {...},
            "technologies": {...},
            "dependencies": [...],
            "important_files": [...],
            "directory_structure": {...},
            "ci_cd": [...],
            "documentation": {...},
            "architecture_notes": [...]
        }
    }
    """
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

    # ── Validate metadata ────────────────────────────────────────────────────
    metadata = body.get("metadata")
    if not metadata or not isinstance(metadata, dict):
        return jsonify({"success": False, "error": "Missing or invalid 'metadata' object."}), 400

    # ── Validate tree ────────────────────────────────────────────────────────
    tree = body.get("tree")
    if tree is None:
        return jsonify({"success": False, "error": "Missing 'tree' in request body."}), 400
    if not isinstance(tree, list):
        return jsonify({"success": False, "error": "'tree' must be a list of file/directory items."}), 400
    if len(tree) > _MAX_TREE_INPUT:
        tree = tree[:_MAX_TREE_INPUT]

    # ── Validate source_files (optional) ─────────────────────────────────────
    source_files = body.get("source_files")
    if source_files is not None:
        if not isinstance(source_files, list):
            return jsonify({"success": False, "error": "'source_files' must be a list."}), 400
        # Cap and sanitise
        source_files = [
            f for f in source_files[:_MAX_FILES_INPUT]
            if isinstance(f, dict) and "path" in f
        ]
    else:
        source_files = []

    # ── Run deterministic analysis ───────────────────────────────────────────
    try:
        insights = analyze_repository(
            repo_url=repo_url,
            metadata=metadata,
            tree=tree,
            source_files=source_files,
        )
    except Exception as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Insight analysis failed: {exc}",
                }
            ),
            500,
        )

    return (
        jsonify(
            {
                "success": True,
                "insights": insights,
            }
        ),
        200,
    )
