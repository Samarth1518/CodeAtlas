"""
services/insights.py — Repository Architecture & Code Insights service.

Analyzes already-retrieved repository data (tree + source files) to produce
structured, deterministic architectural insights without inventing repository facts.

All extracted data is grounded strictly in:
  - The file/directory tree structure
  - Parsed manifest files (package.json, requirements.txt, etc.)
  - Source file paths and extensions
  - Repository metadata

The service NEVER fabricates file names, dependencies, or architecture patterns
not present in the retrieved data.

Security:
  - Never requests additional GitHub API calls beyond what's already retrieved.
  - Sensitive files are excluded by the existing file_filter.py rules.
  - GitHub tokens and LLM API keys are never passed through this module.
"""

import json
import os
import re
from collections import Counter
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Set, Tuple

from services.file_filter import get_extension_language

# ── Manifest file names ───────────────────────────────────────────────────────

#: Dependency manifest filenames we know how to parse
KNOWN_MANIFESTS: Dict[str, str] = {
    "package.json":          "Node.js / npm",
    "requirements.txt":      "Python / pip",
    "requirements-dev.txt":  "Python / pip (dev)",
    "requirements-test.txt": "Python / pip (test)",
    "pyproject.toml":        "Python / pyproject",
    "Pipfile":               "Python / Pipenv",
    "setup.py":              "Python / setuptools",
    "setup.cfg":             "Python / setuptools",
    "Cargo.toml":            "Rust / Cargo",
    "go.mod":                "Go / modules",
    "pom.xml":               "Java / Maven",
    "build.gradle":          "Java / Gradle",
    "build.gradle.kts":      "Kotlin / Gradle",
    "Gemfile":               "Ruby / Bundler",
    "composer.json":         "PHP / Composer",
    "pubspec.yaml":          "Dart / Flutter",
    "Package.swift":         "Swift / SPM",
    "mix.exs":               "Elixir / Mix",
}

#: CI/CD configuration files
CI_FILES: Dict[str, str] = {
    ".github/workflows":    "GitHub Actions",
    ".travis.yml":          "Travis CI",
    ".circleci/config.yml": "CircleCI",
    "Jenkinsfile":          "Jenkins",
    ".gitlab-ci.yml":       "GitLab CI",
    "azure-pipelines.yml":  "Azure Pipelines",
    "bitbucket-pipelines.yml": "Bitbucket Pipelines",
    "Dockerfile":           "Docker",
    "docker-compose.yml":   "Docker Compose",
    "docker-compose.yaml":  "Docker Compose",
    ".dockerignore":        "Docker",
    "kubernetes":           "Kubernetes",
    "k8s":                  "Kubernetes",
    "Makefile":             "Make",
    "makefile":             "Make",
    "GNUmakefile":          "Make",
    "Taskfile.yml":         "Taskfile",
    "Taskfile.yaml":        "Taskfile",
}

#: Signals indicating specific frameworks/libraries by file patterns
FRAMEWORK_SIGNALS: List[Tuple[str, str]] = [
    # JavaScript / TypeScript frameworks
    (r"next\.config\.(js|ts|mjs)$",     "Next.js"),
    (r"nuxt\.config\.(js|ts)$",         "Nuxt.js"),
    (r"vite\.config\.(js|ts)$",         "Vite"),
    (r"svelte\.config\.(js|ts)$",       "SvelteKit"),
    (r"angular\.json$",                  "Angular"),
    (r"gatsby-config\.(js|ts)$",         "Gatsby"),
    (r"remix\.config\.(js|ts)$",         "Remix"),
    (r"astro\.config\.(mjs|ts)$",        "Astro"),
    (r"webpack\.config\.(js|ts)$",       "Webpack"),
    (r"\.eslintrc\.(js|json|ya?ml|cjs)$","ESLint"),
    (r"tailwind\.config\.(js|ts)$",      "Tailwind CSS"),
    (r"jest\.config\.(js|ts)$",          "Jest"),
    (r"vitest\.config\.(js|ts)$",        "Vitest"),
    # Python frameworks
    (r"manage\.py$",                     "Django"),
    (r"wsgi\.py$",                       "WSGI (Django/Flask)"),
    (r"asgi\.py$",                       "ASGI (Django/FastAPI)"),
    (r"alembic\.ini$",                   "Alembic (DB migrations)"),
    (r"celery\.py$",                     "Celery"),
    (r"conftest\.py$",                   "pytest"),
    (r"pytest\.ini$",                    "pytest"),
    (r"tox\.ini$",                       "tox"),
    # Config
    (r"\.pre-commit-config\.yaml$",      "pre-commit"),
    (r"renovate\.json$",                 "Renovate"),
    (r"terraform\.tfvars$",              "Terraform"),
    (r"\.terraform$",                    "Terraform"),
    (r"serverless\.yml$",                "Serverless Framework"),
]

#: Common important entry-point file names (checked by basename)
ENTRY_POINT_NAMES: Set[str] = {
    "main.py", "app.py", "server.py", "run.py", "wsgi.py", "asgi.py",
    "manage.py", "index.js", "index.ts", "index.jsx", "index.tsx",
    "app.js", "app.ts", "server.js", "server.ts", "main.js", "main.ts",
    "main.go", "main.rs", "main.c", "main.cpp", "main.java",
    "index.html", "App.tsx", "App.jsx", "App.vue", "App.svelte",
    "Program.cs", "Startup.cs", "__init__.py",
}

#: Common test directory names
TEST_DIR_PATTERNS: Set[str] = {
    "test", "tests", "spec", "specs", "__tests__", "e2e", "integration",
    "unit", "test_", "tests_",
}

#: Documentation file names
DOC_FILE_PATTERNS: Set[str] = {
    "readme.md", "readme.rst", "readme.txt", "readme",
    "contributing.md", "changelog.md", "changelog.rst",
    "license", "license.md", "license.txt",
    "docs", "documentation",
}

# ── Manifest parsers ──────────────────────────────────────────────────────────

def _parse_package_json(content: str) -> Dict[str, Any]:
    """Extracts package name, version, scripts, and dependencies from package.json."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return {"error": "Could not parse package.json"}

    deps = list((data.get("dependencies") or {}).keys())
    dev_deps = list((data.get("devDependencies") or {}).keys())
    scripts = list((data.get("scripts") or {}).keys())
    peer_deps = list((data.get("peerDependencies") or {}).keys())

    return {
        "name": data.get("name"),
        "version": data.get("version"),
        "description": data.get("description"),
        "main": data.get("main"),
        "scripts": scripts[:20],
        "dependencies": deps[:50],
        "devDependencies": dev_deps[:50],
        "peerDependencies": peer_deps[:20],
        "total_dependencies": len(deps),
        "total_devDependencies": len(dev_deps),
    }


def _parse_requirements_txt(content: str) -> Dict[str, Any]:
    """Extracts package names from requirements.txt (pip format)."""
    packages: List[str] = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        # Strip version specifiers: flask>=3.0 → flask
        pkg = re.split(r"[>=<!~\[\s;]", line)[0].strip()
        if pkg:
            packages.append(pkg)
    return {
        "packages": packages[:100],
        "total_packages": len(packages),
    }


def _parse_pyproject_toml(content: str) -> Dict[str, Any]:
    """Extracts basic metadata from pyproject.toml without external TOML parsers."""
    result: Dict[str, Any] = {}
    deps: List[str] = []

    # Extract project name
    m = re.search(r'^\s*name\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    if m:
        result["name"] = m.group(1)

    # Extract version
    m = re.search(r'^\s*version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    if m:
        result["version"] = m.group(1)

    # Extract dependencies block (simplified)
    in_deps = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped in ("[project.dependencies]", "dependencies = ["):
            in_deps = True
            continue
        if in_deps:
            if stripped.startswith("[") and not stripped.startswith('"'):
                in_deps = False
                continue
            pkg_m = re.match(r'["\']?([A-Za-z0-9_\-]+)[>=<!~\[\"\']?', stripped)
            if pkg_m and stripped:
                deps.append(pkg_m.group(1))

    result["dependencies"] = deps[:50]
    result["total_packages"] = len(deps)
    return result


def _parse_cargo_toml(content: str) -> Dict[str, Any]:
    """Extracts package metadata and dependencies from Cargo.toml."""
    result: Dict[str, Any] = {}
    deps: List[str] = []

    m = re.search(r'^\s*name\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if m:
        result["name"] = m.group(1)

    m = re.search(r'^\s*version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if m:
        result["version"] = m.group(1)

    in_deps = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped in ("[dependencies]", "[dev-dependencies]", "[build-dependencies]"):
            in_deps = True
            continue
        if stripped.startswith("[") and in_deps:
            in_deps = False
            continue
        if in_deps and "=" in stripped and not stripped.startswith("#"):
            pkg = stripped.split("=")[0].strip().strip('"')
            if pkg:
                deps.append(pkg)

    result["dependencies"] = deps[:50]
    result["total_packages"] = len(deps)
    return result


def _parse_go_mod(content: str) -> Dict[str, Any]:
    """Extracts module name and dependencies from go.mod."""
    deps: List[str] = []
    module_name = None

    for line in content.splitlines():
        line = line.strip()
        if line.startswith("module "):
            module_name = line[7:].strip()
        elif line.startswith("require (") or line.startswith("require("):
            continue
        elif line and not line.startswith("//") and not line.startswith(")"):
            parts = line.split()
            if len(parts) >= 2 and "/" in parts[0]:
                deps.append(parts[0])

    return {
        "module": module_name,
        "dependencies": deps[:50],
        "total_packages": len(deps),
    }


def _parse_gemfile(content: str) -> Dict[str, Any]:
    """Extracts gems from a Ruby Gemfile."""
    gems: List[str] = []
    for line in content.splitlines():
        m = re.match(r"^\s*gem\s+['\"]([^'\"]+)['\"]", line)
        if m:
            gems.append(m.group(1))
    return {"gems": gems[:50], "total_packages": len(gems)}


def _parse_composer_json(content: str) -> Dict[str, Any]:
    """Extracts packages from composer.json."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return {"error": "Could not parse composer.json"}
    deps = list((data.get("require") or {}).keys())
    dev_deps = list((data.get("require-dev") or {}).keys())
    return {
        "name": data.get("name"),
        "dependencies": deps[:50],
        "devDependencies": dev_deps[:50],
        "total_packages": len(deps) + len(dev_deps),
    }


#: Registry of manifest parsers
_MANIFEST_PARSERS = {
    "package.json":      _parse_package_json,
    "requirements.txt":  _parse_requirements_txt,
    "requirements-dev.txt": _parse_requirements_txt,
    "requirements-test.txt": _parse_requirements_txt,
    "pyproject.toml":    _parse_pyproject_toml,
    "Cargo.toml":        _parse_cargo_toml,
    "go.mod":            _parse_go_mod,
    "Gemfile":           _parse_gemfile,
    "composer.json":     _parse_composer_json,
}


# ── Core analysis ─────────────────────────────────────────────────────────────

def analyze_repository(
    repo_url: str,
    metadata: Dict[str, Any],
    tree: List[Dict[str, Any]],
    source_files: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Produces structured architectural insights from already-retrieved repository data.

    Args:
        repo_url:     Full GitHub repository URL.
        metadata:     Repository metadata dict (name, owner, full_name, etc.).
        tree:         List of tree items (each has 'path', 'type', 'size').
        source_files: Optional list of source file dicts with 'path' and 'content'.

    Returns:
        Dict with structured insights including statistics, technologies, dependencies,
        important files, directory structure, CI/CD, and architecture summary.
    """
    source_files = source_files or []
    source_map: Dict[str, str] = {f["path"]: f.get("content", "") for f in source_files}

    # ── 1. File statistics ────────────────────────────────────────────────────
    stats = _compute_statistics(tree)

    # ── 2. Technology / language detection ────────────────────────────────────
    technologies = _detect_technologies(tree, source_map, metadata)

    # ── 3. Dependency extraction ──────────────────────────────────────────────
    dependencies = _extract_dependencies(tree, source_map)

    # ── 4. Important files & entry points ────────────────────────────────────
    important_files = _find_important_files(tree)

    # ── 5. Directory structure analysis ──────────────────────────────────────
    directory_structure = _analyze_directory_structure(tree)

    # ── 6. CI/CD & DevOps detection ──────────────────────────────────────────
    ci_cd = _detect_ci_cd(tree)

    # ── 7. Documentation presence ─────────────────────────────────────────────
    docs = _detect_documentation(tree)

    # ── 8. Architecture-level insights (deterministic) ───────────────────────
    architecture_notes = _generate_architecture_notes(
        tree, technologies, dependencies, directory_structure, metadata, source_map
    )

    return {
        "repo_url": repo_url,
        "metadata": {
            "name": metadata.get("name"),
            "full_name": metadata.get("full_name"),
            "owner": metadata.get("owner"),
            "primary_language": metadata.get("primary_language"),
            "default_branch": metadata.get("default_branch"),
            "visibility": metadata.get("visibility"),
        },
        "statistics": stats,
        "technologies": technologies,
        "dependencies": dependencies,
        "important_files": important_files,
        "directory_structure": directory_structure,
        "ci_cd": ci_cd,
        "documentation": docs,
        "architecture_notes": architecture_notes,
    }


# ── Statistics ────────────────────────────────────────────────────────────────

def _compute_statistics(tree: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes file/directory statistics from the tree."""
    total_files = sum(1 for i in tree if i.get("type") == "file")
    total_dirs = sum(1 for i in tree if i.get("type") == "directory")
    total_size = sum(
        (i.get("size") or 0)
        for i in tree
        if i.get("type") == "file" and i.get("size") is not None
    )

    # Language distribution by file count
    lang_counts: Counter = Counter()
    for item in tree:
        if item.get("type") != "file":
            continue
        path = item.get("path", "")
        lang = get_extension_language(path)
        if lang:
            lang_counts[lang] += 1

    top_languages = dict(lang_counts.most_common(10))

    return {
        "total_files": total_files,
        "total_directories": total_dirs,
        "total_items": total_files + total_dirs,
        "total_size_bytes": total_size,
        "language_distribution": top_languages,
    }


# ── Technology detection ──────────────────────────────────────────────────────

def _detect_technologies(
    tree: List[Dict[str, Any]],
    source_map: Dict[str, str],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Detects programming languages, frameworks, and tools from file patterns."""
    all_paths = [i.get("path", "") for i in tree]
    all_basenames = {PurePosixPath(p).name for p in all_paths}

    detected_frameworks: Set[str] = set()

    # Check framework signals against all file paths
    for pattern, framework in FRAMEWORK_SIGNALS:
        for path in all_paths:
            if re.search(pattern, path, re.IGNORECASE):
                detected_frameworks.add(framework)
                break

    # Detect from package.json dependencies if content is available
    pkg_json_content = _find_manifest_content(source_map, "package.json")
    if pkg_json_content:
        try:
            pkg = json.loads(pkg_json_content)
            all_deps = {
                **((pkg.get("dependencies") or {})),
                **((pkg.get("devDependencies") or {})),
            }
            dep_keys = set(all_deps.keys())
            # Map known npm packages → framework names
            _NPM_FRAMEWORK_MAP = {
                "react": "React",
                "react-dom": "React",
                "next": "Next.js",
                "vue": "Vue.js",
                "@angular/core": "Angular",
                "svelte": "Svelte",
                "gatsby": "Gatsby",
                "@remix-run/react": "Remix",
                "express": "Express.js",
                "fastify": "Fastify",
                "koa": "Koa.js",
                "hapi": "Hapi.js",
                "nestjs": "NestJS",
                "@nestjs/core": "NestJS",
                "socket.io": "Socket.IO",
                "graphql": "GraphQL",
                "apollo-server": "Apollo Server",
                "mongoose": "Mongoose (MongoDB)",
                "sequelize": "Sequelize (SQL ORM)",
                "prisma": "Prisma",
                "typeorm": "TypeORM",
                "redux": "Redux",
                "@reduxjs/toolkit": "Redux Toolkit",
                "zustand": "Zustand",
                "mobx": "MobX",
                "jest": "Jest",
                "vitest": "Vitest",
                "cypress": "Cypress",
                "playwright": "@playwright/test",
                "tailwindcss": "Tailwind CSS",
                "styled-components": "styled-components",
                "antd": "Ant Design",
                "@mui/material": "Material UI",
                "axios": "Axios",
                "typescript": "TypeScript",
                "webpack": "Webpack",
                "vite": "Vite",
                "rollup": "Rollup",
                "babel": "Babel",
                "eslint": "ESLint",
                "prettier": "Prettier",
            }
            for pkg_name, framework in _NPM_FRAMEWORK_MAP.items():
                if pkg_name in dep_keys:
                    detected_frameworks.add(framework)
        except (json.JSONDecodeError, ValueError):
            pass

    # Detect from requirements.txt
    req_content = _find_manifest_content(source_map, "requirements.txt")
    if req_content:
        req_lower = req_content.lower()
        _PIP_FRAMEWORK_MAP = {
            "flask": "Flask",
            "django": "Django",
            "fastapi": "FastAPI",
            "starlette": "Starlette",
            "tornado": "Tornado",
            "aiohttp": "aiohttp",
            "sanic": "Sanic",
            "uvicorn": "Uvicorn (ASGI)",
            "gunicorn": "Gunicorn (WSGI)",
            "sqlalchemy": "SQLAlchemy",
            "alembic": "Alembic",
            "pydantic": "Pydantic",
            "celery": "Celery",
            "numpy": "NumPy",
            "pandas": "Pandas",
            "scikit-learn": "scikit-learn",
            "tensorflow": "TensorFlow",
            "torch": "PyTorch",
            "transformers": "Hugging Face Transformers",
            "sentence-transformers": "sentence-transformers",
            "openai": "OpenAI SDK",
            "anthropic": "Anthropic SDK",
            "boto3": "AWS Boto3",
            "google-cloud": "Google Cloud SDK",
            "pytest": "pytest",
            "redis": "Redis",
            "pymongo": "PyMongo (MongoDB)",
            "psycopg2": "psycopg2 (PostgreSQL)",
            "asyncpg": "asyncpg (PostgreSQL)",
            "motor": "Motor (MongoDB async)",
            "httpx": "HTTPX",
            "requests": "Requests",
            "click": "Click (CLI)",
            "typer": "Typer (CLI)",
        }
        for pkg_name, framework in _PIP_FRAMEWORK_MAP.items():
            if pkg_name in req_lower:
                detected_frameworks.add(framework)

    # Detect from extension distribution
    extension_langs: Set[str] = set()
    for item in tree:
        if item.get("type") == "file":
            lang = get_extension_language(item.get("path", ""))
            if lang:
                extension_langs.add(lang)

    # Sort for determinism
    return {
        "primary_language": metadata.get("primary_language"),
        "detected_languages": sorted(extension_langs),
        "detected_frameworks": sorted(detected_frameworks),
        "manifest_files": [
            p for p in all_paths
            if PurePosixPath(p).name in KNOWN_MANIFESTS
        ][:20],
    }


# ── Dependency extraction ─────────────────────────────────────────────────────

def _extract_dependencies(
    tree: List[Dict[str, Any]],
    source_map: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Extracts and parses all known dependency manifests found in the tree."""
    all_paths = [i.get("path", "") for i in tree]
    result: List[Dict[str, Any]] = []

    for path in all_paths:
        basename = PurePosixPath(path).name
        if basename not in _MANIFEST_PARSERS:
            continue

        content = source_map.get(path, "")
        if not content:
            # Manifest found in tree but content not retrieved — still report it
            result.append({
                "file": path,
                "ecosystem": KNOWN_MANIFESTS.get(basename, basename),
                "parsed": False,
                "reason": "File content was not retrieved",
            })
            continue

        try:
            parsed = _MANIFEST_PARSERS[basename](content)
            result.append({
                "file": path,
                "ecosystem": KNOWN_MANIFESTS.get(basename, basename),
                "parsed": True,
                "data": parsed,
            })
        except Exception as exc:
            result.append({
                "file": path,
                "ecosystem": KNOWN_MANIFESTS.get(basename, basename),
                "parsed": False,
                "reason": f"Parse error: {exc}",
            })

    return result


# ── Important files ───────────────────────────────────────────────────────────

def _find_important_files(tree: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identifies important/notable files from the repository tree."""
    important: List[Dict[str, Any]] = []

    for item in tree:
        if item.get("type") != "file":
            continue
        path = item.get("path", "")
        basename = PurePosixPath(path).name
        basename_lower = basename.lower()

        # Entry points
        if basename in ENTRY_POINT_NAMES:
            important.append({
                "path": path,
                "role": "Entry point",
                "reason": f"'{basename}' is a common application entry point",
            })
            continue

        # README / documentation
        if basename_lower in DOC_FILE_PATTERNS or basename_lower.startswith("readme"):
            important.append({
                "path": path,
                "role": "Documentation",
                "reason": "Project documentation file",
            })
            continue

        # Dependency manifests
        if basename in KNOWN_MANIFESTS:
            important.append({
                "path": path,
                "role": "Dependency manifest",
                "reason": f"{KNOWN_MANIFESTS[basename]} dependency file",
            })
            continue

        # CI/CD files
        for ci_path, ci_name in CI_FILES.items():
            if path == ci_path or path.startswith(ci_path + "/") or basename == ci_path:
                important.append({
                    "path": path,
                    "role": "CI/CD configuration",
                    "reason": f"{ci_name} configuration file",
                })
                break

    # Sort for determinism and cap at 30
    important.sort(key=lambda x: (x["role"], x["path"]))
    return important[:30]


# ── Directory structure ───────────────────────────────────────────────────────

def _analyze_directory_structure(tree: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyzes top-level directories and their apparent roles."""
    top_level_dirs: List[str] = []
    top_level_files: List[str] = []
    dir_file_counts: Counter = Counter()

    for item in tree:
        path = item.get("path", "")
        parts = path.split("/")
        if len(parts) == 1:
            if item.get("type") == "directory":
                top_level_dirs.append(path)
            else:
                top_level_files.append(path)
        elif len(parts) >= 2:
            dir_file_counts[parts[0]] += 1

    # Infer roles of top-level directories
    dir_roles: List[Dict[str, str]] = []
    for d in sorted(set(list(top_level_dirs) + list(dir_file_counts.keys()))):
        role = _infer_directory_role(d)
        dir_roles.append({
            "name": d,
            "role": role,
            "file_count": dir_file_counts.get(d, 0),
        })

    return {
        "top_level_directories": sorted(top_level_dirs)[:30],
        "top_level_files": sorted(top_level_files)[:20],
        "directory_roles": dir_roles[:30],
    }


def _infer_directory_role(dirname: str) -> str:
    """Infers the architectural role of a directory from its name."""
    d = dirname.lower()
    roles = {
        "src": "Source code",
        "source": "Source code",
        "lib": "Library / utilities",
        "libs": "Library / utilities",
        "app": "Application code",
        "apps": "Application code",
        "api": "API layer",
        "routes": "Route handlers",
        "controllers": "Controllers",
        "handlers": "Request handlers",
        "middleware": "Middleware",
        "models": "Data models",
        "schemas": "Data schemas",
        "services": "Business logic / services",
        "utils": "Utilities / helpers",
        "helpers": "Utilities / helpers",
        "common": "Shared utilities",
        "shared": "Shared code",
        "components": "UI components",
        "pages": "Application pages",
        "views": "View layer",
        "templates": "Templates",
        "static": "Static assets",
        "public": "Public assets",
        "assets": "Static assets",
        "styles": "Stylesheets",
        "css": "Stylesheets",
        "images": "Images",
        "img": "Images",
        "fonts": "Fonts",
        "icons": "Icons",
        "tests": "Test suite",
        "test": "Test suite",
        "spec": "Test suite",
        "specs": "Test suite",
        "__tests__": "Test suite",
        "e2e": "End-to-end tests",
        "docs": "Documentation",
        "documentation": "Documentation",
        "scripts": "Build / utility scripts",
        "tools": "Developer tools",
        "config": "Configuration",
        "configs": "Configuration",
        "settings": "Configuration",
        "database": "Database layer",
        "db": "Database layer",
        "migrations": "Database migrations",
        "data": "Data files",
        "dist": "Build output (distribution)",
        "build": "Build output",
        "out": "Build output",
        "node_modules": "Node.js dependencies (external)",
        "venv": "Python virtual environment (external)",
        ".github": "GitHub configuration (CI/Actions)",
        ".vscode": "VS Code editor configuration",
        "kubernetes": "Kubernetes configuration",
        "k8s": "Kubernetes configuration",
        "terraform": "Infrastructure-as-code (Terraform)",
        "infra": "Infrastructure configuration",
        "infrastructure": "Infrastructure configuration",
        "docker": "Docker configuration",
        "ci": "CI/CD configuration",
        "deploy": "Deployment configuration",
        "frontend": "Frontend application",
        "backend": "Backend application",
        "client": "Client-side application",
        "server": "Server-side application",
        "web": "Web application layer",
        "cmd": "CLI command entrypoints (Go convention)",
        "pkg": "Reusable packages (Go convention)",
        "internal": "Internal packages (Go convention)",
        "proto": "Protocol Buffer definitions",
        "protos": "Protocol Buffer definitions",
    }
    return roles.get(d, "Unknown")


# ── CI/CD detection ───────────────────────────────────────────────────────────

def _detect_ci_cd(tree: List[Dict[str, Any]]) -> List[str]:
    """Detects CI/CD and DevOps tooling from file/directory presence."""
    detected: Set[str] = set()
    all_paths = [i.get("path", "") for i in tree]

    for path in all_paths:
        # GitHub Actions
        if path.startswith(".github/workflows/") and path.endswith((".yml", ".yaml")):
            detected.add("GitHub Actions")
        # Other CI patterns
        for ci_path, ci_name in CI_FILES.items():
            basename = PurePosixPath(path).name
            if path == ci_path or basename == ci_path or path.startswith(ci_path + "/"):
                detected.add(ci_name)

    return sorted(detected)


# ── Documentation detection ───────────────────────────────────────────────────

def _detect_documentation(tree: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Detects documentation files and directories."""
    doc_files: List[str] = []
    has_readme = False

    for item in tree:
        if item.get("type") != "file":
            continue
        path = item.get("path", "")
        basename = PurePosixPath(path).name.lower()

        if basename.startswith("readme"):
            has_readme = True
            doc_files.append(path)
        elif basename in {"contributing.md", "changelog.md", "changelog.rst",
                          "changelog.txt", "authors.md", "license", "license.md",
                          "license.txt", "code_of_conduct.md", "security.md"}:
            doc_files.append(path)
        elif path.startswith("docs/") or path.startswith("documentation/"):
            doc_files.append(path)

    return {
        "has_readme": has_readme,
        "documentation_files": doc_files[:20],
        "documentation_file_count": len(doc_files),
    }


# ── Architecture insights (deterministic) ─────────────────────────────────────

def _generate_architecture_notes(
    tree: List[Dict[str, Any]],
    technologies: Dict[str, Any],
    dependencies: List[Dict[str, Any]],
    directory_structure: Dict[str, Any],
    metadata: Dict[str, Any],
    source_map: Dict[str, str],
) -> List[str]:
    """
    Generates deterministic, evidence-based architecture observations.
    These are logical deductions from the structure — no fabrication.
    """
    notes: List[str] = []
    all_paths = {i.get("path", "") for i in tree}
    top_dirs = {d["name"].lower() for d in directory_structure.get("directory_roles", [])}
    frameworks = set(technologies.get("detected_frameworks", []))
    languages = set(technologies.get("detected_languages", []))

    # Monorepo detection
    if any(d in top_dirs for d in ("frontend", "backend", "client", "server", "api", "web")):
        if len(top_dirs) >= 3:
            notes.append(
                "The repository appears to use a monorepo structure containing multiple application layers "
                "(e.g., frontend and backend) within the same repository."
            )

    # Full-stack detection
    frontend_indicators = frameworks & {"React", "Vue.js", "Angular", "Svelte", "Next.js", "Nuxt.js"}
    backend_indicators = frameworks & {"Flask", "Django", "FastAPI", "Express.js", "Fastify", "NestJS"}
    if frontend_indicators and backend_indicators:
        notes.append(
            f"This is a full-stack application combining "
            f"{', '.join(sorted(frontend_indicators))} (frontend) with "
            f"{', '.join(sorted(backend_indicators))} (backend)."
        )

    # SPA detection
    if any(f in frameworks for f in ["React", "Vue.js", "Angular", "Svelte"]):
        if not any(f in frameworks for f in ["Next.js", "Nuxt.js", "SvelteKit", "Gatsby", "Remix"]):
            notes.append(
                "The project appears to be a Single Page Application (SPA) using a client-side framework "
                "without a server-side rendering framework."
            )

    # API-first patterns
    if "routes" in top_dirs or "controllers" in top_dirs or "handlers" in top_dirs:
        notes.append(
            "The repository follows a route/controller-based architectural pattern for API handling."
        )

    # Test coverage presence
    if any(d in top_dirs for d in ("tests", "test", "spec", "specs", "__tests__")):
        notes.append(
            "The repository includes a dedicated test directory, indicating a test-driven or test-aware development approach."
        )

    # Docker/containerization
    if any(p in all_paths for p in ("Dockerfile", "docker-compose.yml", "docker-compose.yaml")):
        notes.append(
            "The project is containerized using Docker. A Dockerfile and/or Docker Compose configuration is present."
        )

    # TypeScript usage
    if "TypeScript" in languages or "TypeScript" in frameworks:
        notes.append("The project uses TypeScript for type-safe JavaScript development.")

    # Database patterns
    db_frameworks = frameworks & {"SQLAlchemy", "Prisma", "Mongoose (MongoDB)", "TypeORM", "Sequelize (SQL ORM)"}
    if db_frameworks:
        notes.append(
            f"Database access is managed via: {', '.join(sorted(db_frameworks))}."
        )

    # Migration presence
    if "migrations" in top_dirs or any(p.startswith("migrations/") for p in all_paths):
        notes.append(
            "Database migration files are present, suggesting a relational database with schema versioning."
        )

    # Config-driven setup
    if any(p in all_paths for p in (".env.example", ".env.sample")):
        notes.append(
            "The repository includes a sample environment file (.env.example), indicating environment-variable-based configuration."
        )

    # Documentation quality
    if any(p in all_paths or p.startswith("docs/") for p in all_paths if "readme" in p.lower()):
        notes.append("A README file is present at the root level.")

    # Infrastructure as code
    if "Terraform" in frameworks or "terraform" in top_dirs:
        notes.append(
            "Infrastructure-as-Code (Terraform) configuration is present, suggesting cloud infrastructure management."
        )

    # Return deduplicated, capped list
    seen: Set[str] = set()
    unique_notes: List[str] = []
    for note in notes:
        if note not in seen:
            seen.add(note)
            unique_notes.append(note)

    return unique_notes[:15]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_manifest_content(source_map: Dict[str, str], filename: str) -> Optional[str]:
    """Finds manifest file content regardless of subdirectory depth."""
    # Prefer root-level
    if filename in source_map:
        return source_map[filename]
    # Check any path ending with the filename
    for path, content in source_map.items():
        if PurePosixPath(path).name == filename:
            return content
    return None
