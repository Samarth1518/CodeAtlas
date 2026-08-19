"""
routes/index.py — Vector index generation and management endpoint.

Provides POST /api/index/build which validates the repository, applies safe
file filtering, chunks the source files, generates embeddings via sentence-transformers,
and stores the vector index in the local persistent vector store.
"""

import re
from flask import Blueprint, jsonify, request

from services.chunker import process_repository_files
from services.embeddings import embed_chunks
from services.file_filter import classify_file, get_extension_language
from services.github import GitHubAPIError, get_file_content, get_repo_metadata
from services.vector_store import default_vector_store

index_bp = Blueprint("index", __name__)

_GITHUB_RE = re.compile(
    r"^https://github\.com/([A-Za-z0-9._-]+)/([A-Za-z0-9._-]+?)(?:\.git)?/?$"
)

_MAX_FILES_TO_INDEX: int = 100
_MAX_CHUNKS_PER_REQUEST: int = 2000


@index_bp.route("/build", methods=["POST"])
def build_index():
    """
    POST /api/index/build

    Request:
    {
        "repo_url": "https://github.com/owner/repo",
        "files": [
            { "path": "src/app.py", "content": "...", "language": "Python" }
        ],
        "paths": ["src/app.py"],
        "ref": "main"
    }

    Response:
    {
        "success": true,
        "repo_url": "https://github.com/owner/repo",
        "summary": {
            "files_processed": 5,
            "chunks_indexed": 32,
            "languages": { "Python": 28, "Markdown": 4 }
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

    files_input = body.get("files")
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

    files_to_process: list[dict] = []

    if files_input is not None:
        if not isinstance(files_input, list):
            return jsonify({"success": False, "error": "'files' must be a list."}), 400
        if len(files_input) > _MAX_FILES_TO_INDEX:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Too many files provided ({len(files_input)}). Maximum is {_MAX_FILES_TO_INDEX}.",
                    }
                ),
                400,
            )

        for f in files_input:
            if isinstance(f, dict) and "path" in f and "content" in f:
                path = str(f["path"])
                # Apply security and safety filter
                should_fetch, _ = classify_file(path, len(f.get("content", "")))
                if should_fetch:
                    files_to_process.append({
                        "path": path,
                        "content": str(f["content"]),
                        "language": f.get("language") or get_extension_language(path),
                    })

    elif paths_input is not None:
        if not isinstance(paths_input, list):
            return jsonify({"success": False, "error": "'paths' must be a list of strings."}), 400
        if len(paths_input) > _MAX_FILES_TO_INDEX:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Too many paths requested ({len(paths_input)}). Maximum is {_MAX_FILES_TO_INDEX}.",
                    }
                ),
                400,
            )

        # Validate public repository access
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
                files_to_process.append({
                    "path": file_data["path"],
                    "content": file_data["content"],
                    "language": get_extension_language(path_str),
                })
            except Exception:
                continue

    # 1. Chunk the source files
    chunking_result = process_repository_files(
        repo_url=repo_url,
        files=files_to_process,
        max_total_chunks=_MAX_CHUNKS_PER_REQUEST,
    )
    chunks = chunking_result["chunks"]

    if not chunks:
        # Build empty index cleanly
        summary = default_vector_store.build_index(repo_url, [], np.empty((0, 384), dtype=np.float32))
        return (
            jsonify(
                {
                    "success": True,
                    "repo_url": repo_url,
                    "summary": {
                        "files_processed": 0,
                        "chunks_indexed": 0,
                        "languages": {},
                    },
                }
            ),
            200,
        )

    # 2. Generate embeddings for the chunks
    try:
        embeddings = embed_chunks(chunks)
    except Exception as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Embedding generation failed: {exc}",
                }
            ),
            500,
        )

    # 3. Store in persistent vector index
    try:
        summary = default_vector_store.build_index(
            repo_url=repo_url,
            chunks=chunks,
            embeddings=embeddings,
        )
    except Exception as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Failed to persist vector index: {exc}",
                }
            ),
            500,
        )

    return (
        jsonify(
            {
                "success": True,
                "repo_url": repo_url,
                "summary": {
                    "files_processed": summary["files_processed"],
                    "chunks_indexed": summary["chunks_indexed"],
                    "languages": summary["languages"],
                },
            }
        ),
        200,
    )
