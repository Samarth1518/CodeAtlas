"""
routes/contents.py — Source-file content retrieval endpoint.

Provides POST /api/contents/fetch which accepts a validated GitHub repository
reference and a list of file paths, applies the file-filter safety rules, and
returns decoded source-file contents for all files that pass.

Architecture note: kept separate from routes/repos.py so that the initial
/api/repos/analyze response stays lightweight (metadata + tree only) and
content fetching is an explicit, on-demand operation.
"""

import re
from flask import Blueprint, jsonify, request

from services.file_filter import MAX_FILE_SIZE_BYTES, classify_file, get_extension_language
from services.github import GitHubAPIError, get_file_content, get_repo_metadata

contents_bp = Blueprint("contents", __name__)

# Same URL pattern as in routes/repos.py
_GITHUB_RE = re.compile(
    r"^https://github\.com/([A-Za-z0-9._-]+)/([A-Za-z0-9._-]+?)(?:\.git)?/?$"
)

#: Hard cap on how many files we will fetch in a single request, to prevent
#: abuse and keep response sizes manageable.
_MAX_FILES_PER_REQUEST: int = 50


@contents_bp.route("/fetch", methods=["POST"])
def fetch_contents():
    """
    POST /api/contents/fetch

    Request body::

        {
            "repo_url":       "https://github.com/owner/repo",
            "ref":            "main",          // optional, defaults to repository default branch
            "paths":          ["src/app.py", "README.md"]  // optional; fetch ALL eligible files if omitted
        }

    Response (success)::

        {
            "success": true,
            "owner": "owner",
            "repo": "repo",
            "ref": "main",
            "files": [
                {
                    "path":     "src/app.py",
                    "name":     "app.py",
                    "size":     1234,
                    "language": "Python",
                    "encoding": "utf-8",
                    "content":  "import os\\n...",
                    "sha":      "abc123",
                    "html_url": "https://github.com/..."
                }
            ],
            "skipped": [
                { "path": "logo.png",   "reason": "skipped: binary/media file extension '.png'" },
                { "path": "huge.sql",   "reason": "skipped: file too large (600 KB > 512 KB limit)" }
            ],
            "summary": {
                "fetched":  2,
                "skipped":  2,
                "errors":   0
            }
        }
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"success": False, "error": "Request body must be valid JSON."}), 400

    # ── Validate repo_url ─────────────────────────────────────────────────────
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

    # ── Validate ref / paths ──────────────────────────────────────────────────
    requested_paths = body.get("paths")  # None means "fetch all eligible"

    if requested_paths is not None:
        if not isinstance(requested_paths, list):
            return jsonify({"success": False, "error": "'paths' must be a list of strings."}), 400
        if len(requested_paths) == 0:
            return jsonify({"success": False, "error": "'paths' list is empty."}), 400
        if len(requested_paths) > _MAX_FILES_PER_REQUEST:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": (
                            f"Too many paths requested ({len(requested_paths)}). "
                            f"Maximum is {_MAX_FILES_PER_REQUEST} files per request."
                        ),
                    }
                ),
                400,
            )

    # ── Verify repository is public (reuse get_repo_metadata guard) ───────────
    try:
        metadata = get_repo_metadata(owner, repo)
    except GitHubAPIError as err:
        return jsonify({"success": False, "error": err.message}), err.status_code

    ref = str(body.get("ref", "") or "").strip() or metadata["default_branch"]

    # ── Build the list of (path, size) pairs to process ──────────────────────
    # If the caller supplied explicit paths we use those (size=None so we skip
    # the size pre-check and let the API tell us), otherwise we'd need the tree.
    # For explicit paths, classify with size=None and rely on GitHub's response.
    if requested_paths is not None:
        candidates = [(str(p).strip(), None) for p in requested_paths]
    else:
        # No paths given → not supported via this endpoint; require caller to
        # pass paths (they get them from /api/repos/analyze tree response).
        return (
            jsonify(
                {
                    "success": False,
                    "error": (
                        "The 'paths' field is required. Retrieve the file tree first via "
                        "POST /api/repos/analyze, then pass the desired file paths here."
                    ),
                }
            ),
            400,
        )

    # ── Process each path ─────────────────────────────────────────────────────
    fetched = []
    skipped = []
    error_count = 0

    for path, size in candidates:
        should_fetch, reason = classify_file(path, size)
        if not should_fetch:
            skipped.append({"path": path, "reason": reason})
            continue

        try:
            file_data = get_file_content(owner, repo, path, ref)

            # Post-fetch size guard (the tree may not always report accurate sizes)
            actual_size = file_data.get("size", 0) or 0
            if actual_size > MAX_FILE_SIZE_BYTES:
                skipped.append(
                    {
                        "path": path,
                        "reason": (
                            f"skipped: file too large after fetch "
                            f"({actual_size // 1024} KB > {MAX_FILE_SIZE_BYTES // 1024} KB limit)"
                        ),
                    }
                )
                continue

            fetched.append(
                {
                    "path": file_data["path"],
                    "name": file_data["name"],
                    "size": file_data["size"],
                    "language": get_extension_language(path),
                    "encoding": file_data["encoding"],
                    "content": file_data["content"],
                    "sha": file_data["sha"],
                    "html_url": file_data["html_url"],
                }
            )

        except GitHubAPIError as err:
            error_count += 1
            skipped.append({"path": path, "reason": f"error: {err.message}"})
        except ValueError as err:
            error_count += 1
            skipped.append({"path": path, "reason": f"decode error: {err}"})

    return (
        jsonify(
            {
                "success": True,
                "owner": owner,
                "repo": repo,
                "ref": ref,
                "files": fetched,
                "skipped": skipped,
                "summary": {
                    "fetched": len(fetched),
                    "skipped": len(skipped),
                    "errors": error_count,
                },
            }
        ),
        200,
    )
