"""
services/chunker.py — Source-code chunking and processing service.

Splits safe source files into structured, semantically meaningful chunks with
complete metadata (file path, line ranges, language, deterministic IDs).
Supports AST-based chunking for Python, header-based chunking for Markdown,
declaration/block-based chunking for C-style and web languages, and a robust
line-and-paragraph fallback for all other formats.
"""

import ast
import hashlib
import re
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

# ── Configurable Chunking Defaults ────────────────────────────────────────────

MIN_CHUNK_LINES: int = 4          # Minimum lines to form an independent chunk
TARGET_CHUNK_LINES: int = 40      # Target line count per chunk
MAX_CHUNK_LINES: int = 120        # Hard ceiling on lines per chunk before sub-splitting
DEFAULT_OVERLAP_LINES: int = 3    # Line overlap when windowing large blocks
MAX_CHUNKS_PER_FILE: int = 200    # Guard against pathological files
MAX_TOTAL_CHUNKS: int = 2000      # Safeguard for entire repository request


# ── Chunk ID Generator ────────────────────────────────────────────────────────

def generate_chunk_id(repo_url: str, file_path: str, start_line: int, end_line: int, content: str) -> str:
    """Generates a deterministic unique chunk ID based on repo, path, line range, and content hash."""
    seed = f"{repo_url}:{file_path}:{start_line}:{end_line}:{content}"
    content_hash = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
    safe_path = file_path.replace("/", "_").replace("\\", "_")
    return f"{safe_path}:L{start_line}-L{end_line}:{content_hash}"


# ── Generic Line-Range Window Helper ──────────────────────────────────────────

def _slice_lines(lines: List[str], start_line: int, end_line: int) -> str:
    """Extracts lines (1-indexed inclusive) into a joined string."""
    s = max(1, start_line) - 1
    e = min(len(lines), end_line)
    return "\n".join(lines[s:e])


def _window_large_range(
    lines: List[str],
    start_line: int,
    end_line: int,
    target_lines: int = TARGET_CHUNK_LINES,
    max_lines: int = MAX_CHUNK_LINES,
    overlap: int = DEFAULT_OVERLAP_LINES,
) -> List[Tuple[int, int]]:
    """
    Subdivides a line range [start_line, end_line] that exceeds max_lines
    into smaller overlapping windows.
    """
    ranges: List[Tuple[int, int]] = []
    total = end_line - start_line + 1
    if total <= max_lines:
        return [(start_line, end_line)]

    curr_start = start_line
    while curr_start <= end_line:
        curr_end = min(curr_start + target_lines - 1, end_line)
        # Avoid creating a tiny trailing chunk
        if end_line - curr_end < MIN_CHUNK_LINES and curr_end < end_line:
            curr_end = end_line
        ranges.append((curr_start, curr_end))
        if curr_end >= end_line:
            break
        curr_start = curr_end - overlap + 1
        if curr_start <= ranges[-1][0]:
            curr_start = ranges[-1][0] + 1
    return ranges


# ── Strategy 1: Python AST Chunking ───────────────────────────────────────────

def _chunk_python(lines: List[str], source: str) -> Optional[List[Tuple[int, int]]]:
    """
    Extracts top-level classes, functions, and standalone statement blocks using Python's AST.
    Returns list of (start_line, end_line) tuples, or None if AST parse fails.
    """
    try:
        tree = ast.parse(source)
    except Exception:
        return None

    if not tree.body:
        return None

    ranges: List[Tuple[int, int]] = []
    prev_end = 0

    for node in tree.body:
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if start is None or end is None:
            continue

        # Catch-up any header or comments between blocks if non-trivial
        if prev_end > 0 and start > prev_end + 1:
            gap_start = prev_end + 1
            gap_end = start - 1
            # Only record gap if it has non-blank lines
            gap_text = _slice_lines(lines, gap_start, gap_end).strip()
            if gap_text:
                if (gap_end - gap_start + 1) <= MAX_CHUNK_LINES:
                    ranges.append((gap_start, gap_end))
                else:
                    ranges.extend(_window_large_range(lines, gap_start, gap_end))

        # Check if node is a class with individual methods
        if isinstance(node, ast.ClassDef) and (end - start + 1) > MAX_CHUNK_LINES:
            # Sub-chunk class methods if class is large
            class_header_end = start
            for subnode in node.body:
                if isinstance(subnode, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    class_header_end = max(start, subnode.lineno - 1)
                    break
            ranges.append((start, class_header_end))

            for subnode in node.body:
                sub_start = getattr(subnode, "lineno", None)
                sub_end = getattr(subnode, "end_lineno", None)
                if sub_start and sub_end:
                    if (sub_end - sub_start + 1) <= MAX_CHUNK_LINES:
                        ranges.append((sub_start, sub_end))
                    else:
                        ranges.extend(_window_large_range(lines, sub_start, sub_end))
        else:
            if (end - start + 1) <= MAX_CHUNK_LINES:
                ranges.append((start, end))
            else:
                ranges.extend(_window_large_range(lines, start, end))

        prev_end = max(prev_end, end)

    # Trailing lines
    if prev_end < len(lines):
        trailing_text = _slice_lines(lines, prev_end + 1, len(lines)).strip()
        if trailing_text:
            ranges.append((prev_end + 1, len(lines)))

    return ranges


# ── Strategy 2: Markdown Header Chunking ──────────────────────────────────────

_MD_HEADER_RE = re.compile(r"^(#{1,4})\s+(.+)$")


def _chunk_markdown(lines: List[str]) -> List[Tuple[int, int]]:
    """Splits Markdown by heading boundaries (#, ##, ###, ####)."""
    header_indices: List[int] = []
    for idx, line in enumerate(lines, start=1):
        if _MD_HEADER_RE.match(line.strip()):
            header_indices.append(idx)

    if not header_indices:
        return _chunk_by_paragraphs_or_lines(lines)

    ranges: List[Tuple[int, int]] = []
    # If content before first header
    if header_indices[0] > 1:
        prefix_text = _slice_lines(lines, 1, header_indices[0] - 1).strip()
        if prefix_text:
            ranges.extend(_window_large_range(lines, 1, header_indices[0] - 1))

    for i in range(len(header_indices)):
        start = header_indices[i]
        end = header_indices[i + 1] - 1 if i + 1 < len(header_indices) else len(lines)
        if (end - start + 1) <= MAX_CHUNK_LINES:
            ranges.append((start, end))
        else:
            ranges.extend(_window_large_range(lines, start, end))

    return ranges


# ── Strategy 3: Block / Declaration Chunking (JS/TS/Java/C/C++/Go/Rust) ───────

_DECLARATION_RE = re.compile(
    r"^(?:"
    r"(?:export\s+(?:default\s+)?)?"
    r"(?:async\s+)?"
    r"(?:function|class|interface|type|enum|struct|trait|impl|record|actor)\b"
    r"|(?:export\s+)?(?:const|let|var)\s+[A-Za-z0-9_$]+\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z0-9_$]+)\s*=>"
    r"|(?:public|private|protected|static|final|abstract|override|fn|func|def)\s+"
    r")",
    re.MULTILINE,
)


def _chunk_code_declarations(lines: List[str]) -> List[Tuple[int, int]]:
    """Detects top-level declaration boundaries for languages with C/Java/JS-style syntax."""
    boundaries: List[int] = []
    for idx, line in enumerate(lines, start=1):
        # Only check unindented or single-level indented declarations
        if line and not line.startswith(("    \t", "\t\t", "      ")):
            if _DECLARATION_RE.match(line.strip()):
                boundaries.append(idx)

    if len(boundaries) <= 1:
        return _chunk_by_paragraphs_or_lines(lines)

    ranges: List[Tuple[int, int]] = []
    if boundaries[0] > 1:
        prefix = _slice_lines(lines, 1, boundaries[0] - 1).strip()
        if prefix:
            ranges.extend(_window_large_range(lines, 1, boundaries[0] - 1))

    for i in range(len(boundaries)):
        start = boundaries[i]
        end = boundaries[i + 1] - 1 if i + 1 < len(boundaries) else len(lines)
        if (end - start + 1) <= MAX_CHUNK_LINES:
            ranges.append((start, end))
        else:
            ranges.extend(_window_large_range(lines, start, end))

    return ranges


# ── Strategy 4: Paragraph & Line-Based Fallback ───────────────────────────────

def _chunk_by_paragraphs_or_lines(
    lines: List[str],
    target_lines: int = TARGET_CHUNK_LINES,
    max_lines: int = MAX_CHUNK_LINES,
    min_lines: int = MIN_CHUNK_LINES,
) -> List[Tuple[int, int]]:
    """
    Fallback chunker: groups lines by blank line / paragraph breaks
    up to target_lines/max_lines.
    """
    total_lines = len(lines)
    if total_lines == 0:
        return []
    if total_lines <= max_lines:
        return [(1, total_lines)]

    ranges: List[Tuple[int, int]] = []
    chunk_start = 1
    curr_line = 1

    while curr_line <= total_lines:
        span = curr_line - chunk_start + 1

        # Check if line is a natural boundary (blank line)
        is_blank = (lines[curr_line - 1].strip() == "")
        reached_target = span >= target_lines
        hit_max = span >= max_lines

        if (reached_target and is_blank) or hit_max or curr_line == total_lines:
            end_line = curr_line
            # If hit max on non-blank, find last blank within range if available
            if hit_max and not is_blank and end_line > chunk_start + min_lines:
                # look back up to 10 lines for a blank line
                for lookback in range(end_line, max(chunk_start + min_lines, end_line - 15), -1):
                    if lines[lookback - 1].strip() == "":
                        end_line = lookback
                        break

            ranges.append((chunk_start, end_line))
            chunk_start = end_line + 1
            curr_line = chunk_start
        else:
            curr_line += 1

    # Merge very small trailing chunk into previous chunk if possible
    if len(ranges) > 1:
        last_start, last_end = ranges[-1]
        last_span = last_end - last_start + 1
        prev_start, prev_end = ranges[-2]
        if last_span < min_lines and (last_end - prev_start + 1) <= max_lines:
            ranges.pop()
            ranges[-1] = (prev_start, last_end)

    return ranges


# ── Main Chunking Function ────────────────────────────────────────────────────

def chunk_file(
    repo_url: str,
    file_path: str,
    content: str,
    language: Optional[str] = None,
    max_chunks: int = MAX_CHUNKS_PER_FILE,
) -> List[Dict[str, Any]]:
    """
    Splits a single source file's content into structured chunks with complete metadata.

    Args:
        repo_url:  Repository full name or URL.
        file_path: Repository-relative file path (e.g. ``"src/app.py"``).
        content:   Full raw text content of the source file.
        language:  Language label if known (e.g. ``"Python"``, ``"TypeScript"``).
        max_chunks: Maximum chunks to generate for this single file.

    Returns:
        List of chunk dicts, each with:
            - chunk_id (str)
            - repo_url (str)
            - file_path (str)
            - language (str or None)
            - start_line (int)
            - end_line (int)
            - content (str)
            - line_count (int)
            - char_count (int)
    """
    if not content or not content.strip():
        return []

    lines = content.split("\n")
    total_lines = len(lines)

    # Detect language if not provided
    if not language:
        from services.file_filter import get_extension_language
        language = get_extension_language(file_path)

    lang_lower = (language or "").lower()
    suffix = PurePosixPath(file_path).suffix.lower()

    # Determine chunking strategy
    ranges: Optional[List[Tuple[int, int]]] = None

    if lang_lower == "python" or suffix in (".py", ".pyi"):
        ranges = _chunk_python(lines, content)

    if ranges is None and (lang_lower == "markdown" or suffix in (".md", ".mdx")):
        ranges = _chunk_markdown(lines)

    if ranges is None and (
        lang_lower in ("javascript", "typescript", "java", "c", "c++", "c#", "rust", "go", "php", "swift")
        or suffix in (".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".cpp", ".cc", ".h", ".hpp", ".cs", ".rs", ".go")
    ):
        ranges = _chunk_code_declarations(lines)

    # Fallback to paragraph/line chunking if no strategy succeeded or returned empty
    if not ranges:
        ranges = _chunk_by_paragraphs_or_lines(lines)

    # Post-process ranges to build final chunk objects
    chunks: List[Dict[str, Any]] = []

    for start_line, end_line in ranges[:max_chunks]:
        chunk_content = _slice_lines(lines, start_line, end_line)
        # Skip purely whitespace chunks
        if not chunk_content.strip():
            continue

        c_id = generate_chunk_id(repo_url, file_path, start_line, end_line, chunk_content)
        chunk_obj = {
            "chunk_id": c_id,
            "repo_url": repo_url,
            "file_path": file_path,
            "language": language,
            "start_line": start_line,
            "end_line": end_line,
            "content": chunk_content,
            "line_count": end_line - start_line + 1,
            "char_count": len(chunk_content),
        }
        chunks.append(chunk_obj)

    return chunks


def process_repository_files(
    repo_url: str,
    files: List[Dict[str, Any]],
    max_total_chunks: int = MAX_TOTAL_CHUNKS,
) -> Dict[str, Any]:
    """
    Processes a collection of retrieved source files and produces all chunks.

    Args:
        repo_url: Full URL or name of repository.
        files: List of file dictionaries (from Phase 3), each containing:
               path, content, and optionally language, size.
        max_total_chunks: Hard limit on total chunks generated.

    Returns:
        Dict with:
            - total_chunks (int)
            - total_files_processed (int)
            - chunks (List[Dict])
            - summary (Dict): language breakdown, total lines, truncated flag
    """
    all_chunks: List[Dict[str, Any]] = []
    language_counts: Dict[str, int] = {}
    files_processed = 0
    total_lines = 0
    truncated = False

    for file_info in files:
        if len(all_chunks) >= max_total_chunks:
            truncated = True
            break

        path = file_info.get("path", "")
        content = file_info.get("content", "")
        language = file_info.get("language")

        if not content:
            continue

        remaining_quota = max_total_chunks - len(all_chunks)
        file_chunks = chunk_file(
            repo_url=repo_url,
            file_path=path,
            content=content,
            language=language,
            max_chunks=min(MAX_CHUNKS_PER_FILE, remaining_quota),
        )

        if file_chunks:
            files_processed += 1
            lang_key = language or "Other"
            language_counts[lang_key] = language_counts.get(lang_key, 0) + len(file_chunks)
            for c in file_chunks:
                total_lines += c["line_count"]
                all_chunks.append(c)

    return {
        "total_chunks": len(all_chunks),
        "total_files_processed": files_processed,
        "chunks": all_chunks,
        "summary": {
            "language_breakdown": language_counts,
            "total_lines_chunked": total_lines,
            "truncated": truncated,
        },
    }
