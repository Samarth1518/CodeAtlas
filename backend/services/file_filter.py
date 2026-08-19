"""
services/file_filter.py — Rules for which repository files are safe to retrieve.

This module defines:
  - Which file extensions / paths are considered valid text/source files.
  - Which files must be unconditionally skipped (binaries, secrets, generated dirs).
  - A maximum per-file size limit to prevent memory exhaustion.
"""

from pathlib import PurePosixPath
from typing import Tuple

# ── Limits ────────────────────────────────────────────────────────────────────

#: Maximum raw byte size of a single file we will fetch. Files larger than this
#: are reported as skipped rather than downloaded into memory.
MAX_FILE_SIZE_BYTES: int = 512 * 1024  # 512 KB

# ── Allowed source / text extensions ──────────────────────────────────────────

ALLOWED_EXTENSIONS: frozenset = frozenset(
    {
        # Python ecosystem
        ".py", ".pyi", ".pyx", ".pxd",
        # JavaScript / TypeScript
        ".js", ".jsx", ".mjs", ".cjs",
        ".ts", ".tsx", ".mts", ".cts",
        # Web
        ".html", ".htm", ".css", ".scss", ".sass", ".less", ".svelte", ".vue",
        # JVM
        ".java", ".kt", ".kts", ".groovy", ".scala",
        # C / C++ / C#
        ".c", ".h", ".cpp", ".cxx", ".cc", ".hpp", ".hxx", ".cs",
        # Systems
        ".rs", ".go", ".swift", ".m", ".mm",
        # Ruby / PHP / Perl
        ".rb", ".rake", ".php", ".pl", ".pm",
        # Shell
        ".sh", ".bash", ".zsh", ".fish", ".ps1", ".psm1", ".bat", ".cmd",
        # Data / config
        ".json", ".jsonc", ".json5",
        ".yaml", ".yml",
        ".toml", ".ini", ".cfg", ".conf",
        ".xml", ".xsd", ".xsl",
        ".csv", ".tsv",
        ".env.example", ".env.sample",   # example env files only, not real ones
        # Documentation
        ".md", ".mdx", ".rst", ".txt", ".adoc", ".asciidoc",
        # SQL / data
        ".sql", ".graphql", ".gql",
        # Build / CI
        ".cmake", ".mk", ".make",
        # Notebooks (raw text form)
        ".ipynb",
        # Other common text
        ".lock",   # package-lock / Pipfile.lock etc. (text)
        ".gradle", ".properties",
        ".tf", ".tfvars",           # Terraform
        ".dart",
        ".r", ".R",                 # R language
        ".lua", ".ex", ".exs",     # Elixir / Lua
        ".clj", ".cljs",            # Clojure
        ".hs", ".lhs",              # Haskell
        ".erl", ".hrl",             # Erlang
        ".elm",
        ".jl",                      # Julia
        ".proto",                   # protobuf
        ".thrift",
        ".dockerfile",
    }
)

#: Special filenames (no extension) that are safe text files.
ALLOWED_BASENAMES: frozenset = frozenset(
    {
        "Makefile", "makefile", "GNUmakefile",
        "Dockerfile", "dockerfile",
        "Vagrantfile",
        "Jenkinsfile",
        "Procfile",
        "Gemfile",
        "Rakefile",
        "Brewfile",
        "LICENSE", "LICENCE",
        "NOTICE",
        "README",
        "CHANGELOG",
        "CONTRIBUTING",
        "AUTHORS",
        "CODEOWNERS",
        ".editorconfig",
        ".gitattributes",
        ".gitignore",
        ".npmignore",
        ".dockerignore",
        ".prettierrc",
        ".eslintrc",
        ".babelrc",
        ".browserslistrc",
        # Example env files are safe to expose
        ".env.example",
        ".env.sample",
    }
)

# ── Sensitive / secret file patterns ──────────────────────────────────────────

#: Exact filenames that must never be retrieved, regardless of extension.
SENSITIVE_FILENAMES: frozenset = frozenset(
    {
        ".env",
        ".env.local", ".env.development", ".env.production",
        ".env.staging", ".env.test", ".env.ci",
        ".npmrc",         # may contain auth tokens
        ".yarnrc",
        ".pypirc",        # PyPI credentials
        "id_rsa", "id_rsa.pub",
        "id_ed25519", "id_ed25519.pub",
        "id_ecdsa", "id_ecdsa.pub",
        "id_dsa", "id_dsa.pub",
        ".ssh",
        "credentials",
        "secrets.yaml", "secrets.yml",
        "secret.yaml", "secret.yml",
        "service-account.json", "serviceaccount.json",
    }
)

#: Extensions that are inherently secret / credential-bearing.
SENSITIVE_EXTENSIONS: frozenset = frozenset(
    {
        ".pem", ".key", ".p12", ".pfx", ".p8",
        ".cer", ".crt", ".der",
        ".jks", ".keystore",
        ".asc",       # GPG armored
        ".gpg", ".pgp",
    }
)

# ── Skip-prefix directories ────────────────────────────────────────────────────

#: Top-level or nested path segments that indicate generated / vendor content.
SKIP_DIR_SEGMENTS: frozenset = frozenset(
    {
        "node_modules",
        ".git",
        "vendor",
        "dist",
        "build",
        "out",
        ".next",
        ".nuxt",
        ".svelte-kit",
        "coverage",
        ".nyc_output",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".tox",
        ".eggs",
        "*.egg-info",
        ".venv",
        "venv",
        "env",
        ".env",          # directory named .env (different from .env file)
        "htmlcov",
        "target",        # Maven / Cargo build output
        ".gradle",
        ".idea",
        ".vscode",
        ".vs",
        "tmp",
        "temp",
        ".cache",
        "logs",
        ".sass-cache",
    }
)

# ── Binary / media extensions ──────────────────────────────────────────────────

BINARY_EXTENSIONS: frozenset = frozenset(
    {
        # Images
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
        ".tiff", ".tif", ".avif", ".heic", ".heif",
        # Video
        ".mp4", ".mkv", ".mov", ".avi", ".wmv", ".flv", ".webm",
        # Audio
        ".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a",
        # Fonts
        ".ttf", ".otf", ".woff", ".woff2", ".eot",
        # Archives
        ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".jar", ".war", ".ear",
        # Executables / compiled
        ".exe", ".dll", ".so", ".dylib", ".bin", ".obj", ".o", ".a", ".lib",
        ".class", ".pyc", ".pyo",
        # Office / PDF
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        # Database blobs
        ".db", ".sqlite", ".sqlite3",
        # Other binary
        ".wasm",
    }
)


# ── Public API ────────────────────────────────────────────────────────────────


def classify_file(path: str, size: int | None) -> Tuple[bool, str]:
    """
    Decide whether a file at the given repository path should be fetched.

    Args:
        path: Repository-relative POSIX path (e.g. ``"src/utils/helper.py"``).
        size: File size in bytes as reported by the Git Trees API, or ``None``.

    Returns:
        A ``(should_fetch, reason)`` tuple where:
          - ``should_fetch`` is ``True`` if the file is safe and worth fetching.
          - ``reason`` is an empty string when ``should_fetch`` is ``True``,
            or a short human-readable explanation when it is ``False``.
    """
    p = PurePosixPath(path)
    parts = p.parts          # e.g. ("src", "utils", "helper.py")
    filename = p.name        # e.g. "helper.py"
    suffix = p.suffix.lower()  # e.g. ".py"  (empty string if no extension)

    # ── 1. Skip skipped directory segments ───────────────────────────────────
    for part in parts[:-1]:   # only directory components, not the filename itself
        if part in SKIP_DIR_SEGMENTS:
            return False, f"skipped: inside excluded directory '{part}'"

    # ── 2. Sensitive filenames (exact match on the filename) ────────────────────────
    # Explicit allow for example/sample env files before the sensitive block
    if filename in (".env.example", ".env.sample"):
        return True, ""

    if filename in SENSITIVE_FILENAMES:
        return False, f"skipped: sensitive file '{filename}'"

    # Catch all remaining .env.* variants (e.g. .env.production, .env.local)
    if filename.startswith(".env."):
        return False, f"skipped: sensitive env file '{filename}'"

    # ── 3. Sensitive extensions ───────────────────────────────────────────────
    if suffix in SENSITIVE_EXTENSIONS:
        return False, f"skipped: sensitive file extension '{suffix}'"

    # ── 4. Binary / media extensions ─────────────────────────────────────────
    if suffix in BINARY_EXTENSIONS:
        return False, f"skipped: binary/media file extension '{suffix}'"

    # ── 5. Oversized files ────────────────────────────────────────────────────
    if size is not None and size > MAX_FILE_SIZE_BYTES:
        kb = size // 1024
        return False, f"skipped: file too large ({kb} KB > {MAX_FILE_SIZE_BYTES // 1024} KB limit)"

    # ── 6. Accept by allowed extension or bare filename ──────────────────────
    if suffix in ALLOWED_EXTENSIONS:
        return True, ""

    if filename in ALLOWED_BASENAMES:
        return True, ""

    # ── 7. Default-deny: unknown extension / no extension ────────────────────
    return False, f"skipped: unsupported file type (extension: '{suffix or 'none'}')"


def get_extension_language(path: str) -> str | None:
    """
    Return a simple language label derived from the file extension, or None.
    This is a best-effort display helper — not a full language detector.
    """
    suffix = PurePosixPath(path).suffix.lower()
    _EXT_LANG: dict[str, str] = {
        ".py": "Python", ".pyi": "Python", ".pyx": "Python",
        ".js": "JavaScript", ".mjs": "JavaScript", ".jsx": "JavaScript",
        ".ts": "TypeScript", ".tsx": "TypeScript",
        ".java": "Java",
        ".kt": "Kotlin", ".kts": "Kotlin",
        ".scala": "Scala",
        ".groovy": "Groovy",
        ".c": "C", ".h": "C",
        ".cpp": "C++", ".cxx": "C++", ".cc": "C++", ".hpp": "C++",
        ".cs": "C#",
        ".rs": "Rust",
        ".go": "Go",
        ".rb": "Ruby",
        ".php": "PHP",
        ".swift": "Swift",
        ".sh": "Shell", ".bash": "Shell", ".zsh": "Shell",
        ".ps1": "PowerShell",
        ".html": "HTML", ".htm": "HTML",
        ".css": "CSS", ".scss": "SCSS", ".sass": "Sass",
        ".json": "JSON", ".jsonc": "JSON",
        ".yaml": "YAML", ".yml": "YAML",
        ".toml": "TOML",
        ".xml": "XML",
        ".sql": "SQL",
        ".md": "Markdown", ".mdx": "Markdown",
        ".rst": "reStructuredText",
        ".txt": "Text",
        ".graphql": "GraphQL", ".gql": "GraphQL",
        ".tf": "Terraform", ".tfvars": "Terraform",
        ".dart": "Dart",
        ".r": "R", ".R": "R",
        ".lua": "Lua",
        ".ex": "Elixir", ".exs": "Elixir",
        ".clj": "Clojure", ".cljs": "ClojureScript",
        ".hs": "Haskell",
        ".erl": "Erlang",
        ".elm": "Elm",
        ".jl": "Julia",
        ".proto": "Protocol Buffers",
        ".ipynb": "Jupyter Notebook",
        ".vue": "Vue",
        ".svelte": "Svelte",
    }
    return _EXT_LANG.get(suffix)
