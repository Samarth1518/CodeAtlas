"""
routes/chunks.py — Source-code chunking and index preparation endpoint.

Provides POST /api/chunks/generate which takes source files (either directly
provided in the payload or retrieved on-demand via the safe GitHub service)
and splits them into structured chunks ready for embedding and vector indexing.
"""

import re
from flask import Blueprint, jsonify, request

from services.chunker import process_repository_files
from services.file_filter import classify_file, get_extension_language
from services.github import GitHubAPIError, get_file_content, get_repo_metadata

chunks_bp = Blueprint("chunks", __name__)

_GITHUB_RE = re.compile(
    r"^https://github\.com/([A-Za-z0-9._-]+)/([A-Za-z0-9._-]+?)(?:\.git)?/?$"
)

_MAX_FILES_TO_PROCESS: int = 100


@chunks_bp.route("/generate", methods=["POST"])
def generate_chunks():
    """
    POST /api/chunks/generate

    Request Body (Mode 1 — direct files):
    {
        "repo_url": "https://github.com/owner/repo",
        "files": [
            {
                "path": "src/app.py",
                "content": "...",
                "language": "Python"
            }
        ]
    }

    Request Body (Mode 2 — fetch on demand):
    {
        "repo_url": "https://github.com/owner/repo",
        "paths": ["src/app.py", "README.md"],
        "ref": "main"
    }

    Response:
    {
        "success": true,
        "repo_url": "https://github.com/owner/repo",
        "total_chunks": 15,
        "total_files_processed": 2,
        "chunks": [
            {
                "chunk_id": "src_app.py:L1-L35:abc12345",
                "repo_url": "https://github.com/owner/repo",
                "file_path": "src/app.py",
                "language": "Python",
                "start_line": 1,
                "end_line": 35,
                "content": "...",
                "line_count": 35,
                "char_count": 890
            }
        ],
        "summary": {
            "language_breakdown": { "Python": 12, "Markdown": 3 },
            "total_lines_chunked": 420,
            "truncated": false
        }
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

    # Mode 1: Files supplied directly in request
    files_input = body.get("files")
    # Mode 2: Paths supplied to fetch
    paths_input = body.get("paths")

    if files_input is None and paths_input is None:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Either 'files' (list of file objects) or 'paths' (list of repository paths) must be provided.",
                }
            ),
            400,
        )

    files_to_chunk: list[dict] = []

    if files_input is not None:
        if not isinstance(files_input, list):
            return jsonify({"success": False, "error": "'files' must be a list."}), 400
        if len(files_input) > _MAX_FILES_TO_PROCESS:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Too many files provided ({len(files_input)}). Maximum is {_MAX_FILES_TO_PROCESS}.",
                    }
                ),
                400,
            )

        for f in files_input:
            if isinstance(f, dict) and "path" in f and "content" in f:
                path = str(f["path"])
                # Apply safety filter on directly provided files
                should_fetch, _ = classify_file(path, len(f.get("content", "")))
                if should_fetch:
                    files_to_chunk.append({
                        "path": path,
                        "content": str(f["content"]),
                        "language": f.get("language") or get_extension_language(path),
                    })

    elif paths_input is not None:
        if not isinstance(paths_input, list):
            return jsonify({"success": False, "error": "'paths' must be a list of strings."}), 400
        if len(paths_input) > _MAX_FILES_TO_PROCESS:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Too many paths requested ({len(paths_input)}). Maximum is {_MAX_FILES_TO_PROCESS}.",
                    }
                ),
                400,
            )

        # Enforce public repository check
        try:
            metadata = get_repo_metadata(owner, repo)
        except GitHubAPIError as err:
            return jsonify({"success": False, "error": err.message}), err.status_code

        ref = str(body.get("ref", "") or "").strip() or metadata["default_branch"]

        for path in paths_input:
            path_str = str(path).strip()
            should_fetch, _ = classify_file(path_str, None)
            if not should_fetch:
                continue

            try:
                file_data = get_file_content(owner, repo, path_str, ref)
                files_to_chunk.append({
                    "path": file_data["path"],
                    "content": file_data["content"],
                    "language": get_extension_language(path_str),
                })
            except Exception:
                # Silently skip single file fetch errors during batch chunking
                continue

    # Execute chunking service
    result = process_repository_files(repo_url=repo_url, files=files_to_chunk)

    return (
        jsonify(
            {
                "success": True,
                "repo_url": repo_url,
                "total_chunks": result["total_chunks"],
                "total_files_processed": result["total_files_processed"],
                "chunks": result["chunks"],
                "summary": result["summary"],
            }
        ),
        200,
    )
