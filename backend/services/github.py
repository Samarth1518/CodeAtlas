"""
services/github.py — GitHub REST API client for repository metadata,
file tree retrieval, and individual source-file content fetching.
"""

import base64
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


class GitHubAPIError(Exception):
    """Custom exception for GitHub API errors with an associated HTTP status code."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _make_github_request(url: str) -> Dict[str, Any]:
    """Helper to perform authenticated GitHub API requests and handle common HTTP/network errors."""
    token = os.getenv("GITHUB_TOKEN", "").strip()
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "CodeAtlas-App",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers, method="GET")

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            if response.status != 200:
                raise GitHubAPIError(
                    response.status,
                    f"GitHub API returned unexpected status code: {response.status}",
                )
            return json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            pass

        # Check for rate limit
        remaining = e.headers.get("x-ratelimit-remaining")
        if remaining == "0" or e.code == 429:
            raise GitHubAPIError(
                429,
                "GitHub API rate limit exceeded. Please try again later.",
            )

        if e.code == 404:
            if "empty" in error_body.lower():
                # Empty repository indicator
                return {"tree": [], "truncated": False, "is_empty": True}
            raise GitHubAPIError(
                404,
                "Repository or requested resource not found on GitHub.",
            )
        elif e.code == 409:
            # 409 Conflict often returned when repo is empty
            return {"tree": [], "truncated": False, "is_empty": True}
        elif e.code == 401:
            raise GitHubAPIError(
                401,
                "GitHub authentication failed. Please verify the configured credentials.",
            )
        elif e.code == 403:
            raise GitHubAPIError(
                403,
                "Access denied. The repository may be private or restricted.",
            )
        elif e.code >= 500:
            raise GitHubAPIError(
                502,
                f"GitHub API is currently unavailable (HTTP {e.code}). Please try again later.",
            )
        else:
            raise GitHubAPIError(
                e.code,
                f"GitHub API request failed with status code {e.code}.",
            )

    except urllib.error.URLError:
        raise GitHubAPIError(
            503,
            "Could not connect to GitHub API. Please check your network connection.",
        )
    except TimeoutError:
        raise GitHubAPIError(
            504,
            "Request to GitHub API timed out. Please try again.",
        )
    except GitHubAPIError:
        raise
    except Exception:
        raise GitHubAPIError(
            500,
            "An unexpected error occurred while communicating with the GitHub API.",
        )


def get_repo_metadata(owner: str, repo: str) -> Dict[str, Any]:
    """
    Fetches basic repository metadata from the GitHub REST API.

    Returns:
        Dict with name, owner, full_name, html_url, default_branch, visibility, primary_language.

    Raises:
        GitHubAPIError: If not found, private/inaccessible, rate-limited, or on API failure.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}"
    data = _make_github_request(url)

    owner_data = data.get("owner", {})
    owner_login = (
        owner_data.get("login")
        if isinstance(owner_data, dict)
        else str(owner_data)
    )

    is_private = data.get("private", False)
    visibility = data.get("visibility")
    if not visibility:
        visibility = "private" if is_private else "public"

    # Enforce public repository restriction
    if is_private or visibility.lower() == "private":
        raise GitHubAPIError(
            403,
            f"Repository '{owner}/{repo}' is private. CodeAtlas currently only supports public repositories.",
        )

    return {
        "name": data.get("name", repo),
        "owner": owner_login or owner,
        "full_name": data.get("full_name", f"{owner}/{repo}"),
        "html_url": data.get("html_url", f"https://github.com/{owner}/{repo}"),
        "default_branch": data.get("default_branch", "main"),
        "visibility": visibility,
        "primary_language": data.get("language"),
    }


def get_repo_tree(owner: str, repo: str, branch: str = "main") -> Dict[str, Any]:
    """
    Retrieves the recursive Git file tree for a repository from GitHub's Git Trees API.

    Returns:
        Dict containing:
            - tree: List[Dict] with path, type ("file" or "directory"), and size (int or None)
            - total_files: int
            - total_dirs: int
            - truncated: bool
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    data = _make_github_request(url)

    raw_tree = data.get("tree", [])
    truncated = bool(data.get("truncated", False))

    clean_tree: List[Dict[str, Any]] = []
    file_count = 0
    dir_count = 0

    for item in raw_tree:
        git_type = item.get("type", "")
        if git_type == "tree":
            item_type = "directory"
            size = None
            dir_count += 1
        elif git_type == "blob":
            item_type = "file"
            size = item.get("size")
            file_count += 1
        else:
            # Submodules/commits or other types
            item_type = "file"
            size = item.get("size")
            file_count += 1

        clean_tree.append({
            "path": item.get("path", ""),
            "type": item_type,
            "size": size,
        })

    return {
        "tree": clean_tree,
        "total_files": file_count,
        "total_dirs": dir_count,
        "truncated": truncated,
    }


def get_file_content(owner: str, repo: str, path: str, ref: str = "main") -> Dict[str, Any]:
    """
    Fetches the content of a single file from GitHub's Contents API and decodes it.

    Uses GitHub's /repos/{owner}/{repo}/contents/{path}?ref={ref} endpoint,
    which returns Base64-encoded content for file blobs.

    Args:
        owner:  Repository owner login.
        repo:   Repository name.
        path:   Repository-relative file path (POSIX, e.g. ``"src/app.py"``).
        ref:    Git ref (branch name, tag, or commit SHA). Defaults to ``"main"``.

    Returns:
        Dict with:
            - path (str)
            - name (str)
            - size (int)  — raw byte size reported by GitHub
            - encoding (str)  — always ``"utf-8"`` after decoding
            - content (str)   — decoded UTF-8 text
            - sha (str)       — blob SHA
            - html_url (str)

    Raises:
        GitHubAPIError: On 404, rate-limit, auth failure, GitHub outage, or
                        if the API returns a non-file object (directory, submodule).
        ValueError: If the GitHub response does not contain Base64-encoded content
                    or the decoded bytes are not valid UTF-8.
    """
    # URL-encode path separators are fine; GitHub accepts forward slashes directly.
    encoded_path = path.replace("%", "%25")  # guard against literal % in paths
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{encoded_path}?ref={ref}"
    data = _make_github_request(url)

    # GitHub returns a list when `path` points to a directory
    if isinstance(data, list):
        raise GitHubAPIError(
            400,
            f"Path '{path}' points to a directory, not a file.",
        )

    obj_type = data.get("type", "")
    if obj_type != "file":
        raise GitHubAPIError(
            400,
            f"Path '{path}' is not a regular file (type: '{obj_type}').",
        )

    encoding = data.get("encoding", "")
    raw_content = data.get("content", "")

    if encoding != "base64" or not raw_content:
        raise ValueError(
            f"Unexpected encoding '{encoding}' for file '{path}'. "
            "Only 'base64' is supported."
        )

    # GitHub includes newlines in the Base64 payload — strip them before decoding.
    try:
        decoded_bytes = base64.b64decode(raw_content.replace("\n", ""))
    except Exception as exc:
        raise ValueError(f"Failed to Base64-decode content of '{path}': {exc}") from exc

    try:
        text = decoded_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"File '{path}' contains non-UTF-8 bytes and cannot be displayed as text."
        ) from exc

    return {
        "path": data.get("path", path),
        "name": data.get("name", path.split("/")[-1]),
        "size": data.get("size", len(decoded_bytes)),
        "encoding": "utf-8",
        "content": text,
        "sha": data.get("sha", ""),
        "html_url": data.get("html_url", ""),
    }
