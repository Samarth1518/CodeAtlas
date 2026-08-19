"""
routes/repos.py — Repository analysis endpoints.
"""

import re
from flask import Blueprint, jsonify, request
from services.github import GitHubAPIError, get_repo_metadata, get_repo_tree

repos_bp = Blueprint("repos", __name__)

# Accepts https://github.com/<owner>/<repo> (ignoring trailing slash or .git)
_GITHUB_RE = re.compile(
    r"^https://github\.com/([A-Za-z0-9._-]+)/([A-Za-z0-9._-]+?)(?:\.git)?/?$"
)


@repos_bp.route("/analyze", methods=["POST"])
def analyze():
    """
    POST /api/repos/analyze
    Body:    { "repo_url": "https://github.com/owner/repo" }
    Returns: {
        "success": true,
        "repo_url": "https://github.com/owner/repo",
        "name": "repo",
        "owner": "owner",
        "full_name": "owner/repo",
        "html_url": "https://github.com/owner/repo",
        "default_branch": "main",
        "visibility": "public",
        "primary_language": "Python",
        "metadata": { ... },
        "tree": [
            { "path": "src/app.py", "type": "file", "size": 1024 },
            { "path": "src", "type": "directory", "size": null }
        ],
        "tree_summary": {
            "total_files": 1,
            "total_dirs": 1,
            "truncated": false
        }
    }
    """
    body = request.get_json(silent=True)

    if not body or "repo_url" not in body:
        return (
            jsonify({"success": False, "error": "Missing 'repo_url' in request body."}),
            400,
        )

    raw_url = str(body["repo_url"]).strip()
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

    try:
        # Step 1: Retrieve repository metadata
        metadata = get_repo_metadata(owner, repo)

        # Step 2: Retrieve file tree using the repository's default branch
        default_branch = metadata.get("default_branch") or "main"
        tree_result = get_repo_tree(owner, repo, default_branch)

        return (
            jsonify(
                {
                    "success": True,
                    "repo_url": repo_url,
                    "name": metadata["name"],
                    "owner": metadata["owner"],
                    "full_name": metadata["full_name"],
                    "html_url": metadata["html_url"],
                    "default_branch": metadata["default_branch"],
                    "visibility": metadata["visibility"],
                    "primary_language": metadata["primary_language"],
                    "metadata": metadata,
                    "tree": tree_result["tree"],
                    "tree_summary": {
                        "total_files": tree_result["total_files"],
                        "total_dirs": tree_result["total_dirs"],
                        "truncated": tree_result["truncated"],
                    },
                }
            ),
            200,
        )
    except GitHubAPIError as err:
        return jsonify({"success": False, "error": err.message}), err.status_code
