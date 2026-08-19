"""
routes/index.py — Asynchronous Vector Index Generation & Status Endpoint.

Provides:
  - POST /api/index/build: Accepts repository source files/paths, chunks them,
    and kicks off background batch embedding with FastEmbed. Returns immediately (202 Accepted)
    so Gunicorn workers never time out on Render Free.
  - GET  /api/index/status: Returns current indexing status (indexing, ready, failed, not_indexed),
    chunk processing progress, and final index summary.
"""

import re
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
from flask import Blueprint, current_app, jsonify, request

from services.chunker import process_repository_files
from services.embeddings import embed_chunks
from services.file_filter import classify_file, get_extension_language
from services.github import GitHubAPIError, get_file_content, get_repo_metadata
from services.vector_store import _normalize_repo_key, default_vector_store

index_bp = Blueprint("index", __name__)

_GITHUB_RE = re.compile(
    r"^https://github\.com/([A-Za-z0-9._-]+)/([A-Za-z0-9._-]+?)(?:\.git)?/?$"
)

_MAX_FILES_TO_INDEX: int = 100
_MAX_CHUNKS_PER_REQUEST: int = 2000
_BATCH_SIZE: int = 8

# In-memory thread-safe tracking for background indexing jobs
_indexing_jobs: Dict[str, Dict[str, Any]] = {}
_indexing_lock = threading.Lock()


def _run_indexing_job(repo_url: str, chunks: List[Dict[str, Any]], files_processed_count: int, lang_counts: Dict[str, int]) -> None:
    """
    Background worker function that encodes chunks in small batches (batch_size=8)
    and commits them to the local vector store without blocking the HTTP worker.
    """
    key = _normalize_repo_key(repo_url)
    try:
        total = len(chunks)
        if total == 0:
            summary = default_vector_store.build_index(repo_url, [], np.empty((0, 384), dtype=np.float32))
            with _indexing_lock:
                _indexing_jobs[key] = {
                    "repo_url": repo_url,
                    "status": "ready",
                    "progress": {"chunks_processed": 0, "total_chunks": 0, "percent": 100},
                    "summary": {
                        "files_processed": 0,
                        "chunks_indexed": 0,
                        "languages": {},
                    },
                    "error": None,
                    "updated_at": time.time(),
                }
            return

        all_embeddings: List[np.ndarray] = []
        for i in range(0, total, _BATCH_SIZE):
            batch = chunks[i : i + _BATCH_SIZE]
            batch_embs = embed_chunks(batch)
            all_embeddings.append(batch_embs)
            processed = min(i + len(batch), total)
            with _indexing_lock:
                if key in _indexing_jobs:
                    _indexing_jobs[key]["progress"] = {
                        "chunks_processed": processed,
                        "total_chunks": total,
                        "percent": int((processed / total) * 100) if total else 100,
                    }
                    _indexing_jobs[key]["updated_at"] = time.time()

        final_embeddings = np.vstack(all_embeddings)

        summary = default_vector_store.build_index(
            repo_url=repo_url,
            chunks=chunks,
            embeddings=final_embeddings,
        )

        with _indexing_lock:
            _indexing_jobs[key] = {
                "repo_url": repo_url,
                "status": "ready",
                "progress": {
                    "chunks_processed": total,
                    "total_chunks": total,
                    "percent": 100,
                },
                "summary": {
                    "files_processed": summary["files_processed"],
                    "chunks_indexed": summary["chunks_indexed"],
                    "languages": summary["languages"],
                },
                "error": None,
                "updated_at": time.time(),
            }
    except Exception as exc:
        with _indexing_lock:
            _indexing_jobs[key] = {
                "repo_url": repo_url,
                "status": "failed",
                "error": str(exc),
                "updated_at": time.time(),
            }


@index_bp.route("/build", methods=["POST"])
def build_index():
    """
    POST /api/index/build
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

    files_to_process: List[Dict[str, Any]] = []

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

    # Chunk the source files
    chunking_result = process_repository_files(
        repo_url=repo_url,
        files=files_to_process,
        max_total_chunks=_MAX_CHUNKS_PER_REQUEST,
    )
    chunks = chunking_result["chunks"]

    key = _normalize_repo_key(repo_url)
    total_chunks = len(chunks)
    files_processed_count = len(set(c.get("file_path", "") for c in chunks))
    lang_counts: Dict[str, int] = {}
    for c in chunks:
        l = c.get("language") or "Other"
        lang_counts[l] = lang_counts.get(l, 0) + 1

    with _indexing_lock:
        _indexing_jobs[key] = {
            "repo_url": repo_url,
            "status": "indexing",
            "progress": {
                "chunks_processed": 0,
                "total_chunks": total_chunks,
                "percent": 0,
            },
            "summary": {
                "files_processed": files_processed_count,
                "chunks_indexed": 0,
                "languages": lang_counts,
            },
            "error": None,
            "started_at": time.time(),
            "updated_at": time.time(),
        }

    # Execute synchronously if requested (or in testing environment)
    is_sync = body.get("sync") is True or current_app.config.get("TESTING", False)
    if is_sync:
        _run_indexing_job(repo_url, chunks, files_processed_count, lang_counts)
        with _indexing_lock:
            job = _indexing_jobs.get(key, {})
            if job.get("status") == "failed":
                return jsonify({"success": False, "error": job.get("error")}), 500
            return jsonify({
                "success": True,
                "repo_url": repo_url,
                "status": "ready",
                "summary": job.get("summary", {
                    "files_processed": files_processed_count,
                    "chunks_indexed": total_chunks,
                    "languages": lang_counts,
                }),
            }), 200

    # Start non-blocking background thread for production
    thread = threading.Thread(
        target=_run_indexing_job,
        args=(repo_url, chunks, files_processed_count, lang_counts),
        daemon=True,
    )
    thread.start()

    return jsonify({
        "success": True,
        "repo_url": repo_url,
        "status": "indexing",
        "message": "Indexing started in background.",
        "progress": {
            "chunks_processed": 0,
            "total_chunks": total_chunks,
            "percent": 0,
        },
        "summary": {
            "files_processed": files_processed_count,
            "chunks_indexed": 0,
            "languages": lang_counts,
        },
    }), 200


@index_bp.route("/status", methods=["GET"])
def index_status():
    """
    GET /api/index/status?repo_url=https://github.com/owner/repo
    Returns real-time progress and completion status for a repository index.
    """
    raw_url = request.args.get("repo_url", "").strip()
    if not raw_url:
        return jsonify({"success": False, "error": "Missing 'repo_url' query parameter."}), 400

    key = _normalize_repo_key(raw_url)
    with _indexing_lock:
        if key in _indexing_jobs:
            job = _indexing_jobs[key]
            return jsonify({
                "success": True,
                "repo_url": raw_url,
                "status": job["status"],
                "progress": job.get("progress"),
                "summary": job.get("summary"),
                "error": job.get("error"),
            }), 200

    # If not in active memory, check if already indexed on disk
    index_data = default_vector_store._load_repo_index(raw_url)
    if index_data and len(index_data["metadata"]) > 0:
        metadata = index_data["metadata"]
        files = set(m.get("file_path", "") for m in metadata)
        langs: Dict[str, int] = {}
        for m in metadata:
            l = m.get("language") or "Other"
            langs[l] = langs.get(l, 0) + 1
        return jsonify({
            "success": True,
            "repo_url": raw_url,
            "status": "ready",
            "progress": {
                "chunks_processed": len(metadata),
                "total_chunks": len(metadata),
                "percent": 100,
            },
            "summary": {
                "files_processed": len(files),
                "chunks_indexed": len(metadata),
                "languages": langs,
            },
            "error": None,
        }), 200

    return jsonify({
        "success": True,
        "repo_url": raw_url,
        "status": "not_indexed",
        "progress": None,
        "summary": None,
        "error": None,
    }), 200
