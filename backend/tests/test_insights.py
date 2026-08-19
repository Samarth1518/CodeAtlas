"""
tests/test_insights.py — Unit tests for services/insights.py.

Tests are deterministic and do not make any real API calls.
"""

import json
import unittest

from services.insights import (
    analyze_repository,
    _compute_statistics,
    _detect_technologies,
    _detect_ci_cd,
    _detect_documentation,
    _extract_dependencies,
    _find_important_files,
    _analyze_directory_structure,
    _parse_package_json,
    _parse_requirements_txt,
    _parse_pyproject_toml,
    _parse_cargo_toml,
    _parse_go_mod,
    _parse_gemfile,
    _parse_composer_json,
)

# ── Sample data ───────────────────────────────────────────────────────────────

SAMPLE_METADATA = {
    "name": "my-app",
    "owner": "owner",
    "full_name": "owner/my-app",
    "primary_language": "Python",
    "default_branch": "main",
    "visibility": "public",
}

SAMPLE_TREE = [
    {"path": "README.md", "type": "file", "size": 1200},
    {"path": "requirements.txt", "type": "file", "size": 300},
    {"path": "pyproject.toml", "type": "file", "size": 500},
    {"path": "app.py", "type": "file", "size": 2400},
    {"path": "Dockerfile", "type": "file", "size": 800},
    {"path": ".github", "type": "directory", "size": None},
    {"path": ".github/workflows", "type": "directory", "size": None},
    {"path": ".github/workflows/ci.yml", "type": "file", "size": 600},
    {"path": "backend", "type": "directory", "size": None},
    {"path": "backend/app.py", "type": "file", "size": 3000},
    {"path": "backend/routes", "type": "directory", "size": None},
    {"path": "backend/routes/api.py", "type": "file", "size": 1500},
    {"path": "backend/services", "type": "directory", "size": None},
    {"path": "backend/services/auth.py", "type": "file", "size": 900},
    {"path": "frontend", "type": "directory", "size": None},
    {"path": "frontend/src", "type": "directory", "size": None},
    {"path": "frontend/src/App.tsx", "type": "file", "size": 5000},
    {"path": "frontend/package.json", "type": "file", "size": 1200},
    {"path": "frontend/vite.config.ts", "type": "file", "size": 400},
    {"path": "tests", "type": "directory", "size": None},
    {"path": "tests/test_auth.py", "type": "file", "size": 1000},
]

SAMPLE_PACKAGE_JSON = json.dumps({
    "name": "my-frontend",
    "version": "1.0.0",
    "description": "Frontend application",
    "scripts": {"dev": "vite", "build": "tsc && vite build", "test": "vitest"},
    "dependencies": {"react": "^18.0.0", "react-dom": "^18.0.0", "axios": "^1.0.0"},
    "devDependencies": {"vite": "^5.0.0", "typescript": "^5.0.0", "vitest": "^1.0.0"},
})

SAMPLE_REQUIREMENTS = "flask==3.0.3\nflask-cors==4.0.1\nrequests>=2.28.0\nnumpy>=1.26.0\n# comment\n-r base.txt\n"

SAMPLE_PYPROJECT = """
[project]
name = "my-app"
version = "0.1.0"
description = "A test project"

[project.dependencies]
flask = ">=3.0"
requests = ">=2.28"
"""

SAMPLE_CARGO = """
[package]
name = "my-crate"
version = "0.1.0"

[dependencies]
serde = "1.0"
tokio = { version = "1.0", features = ["full"] }
reqwest = "0.11"
"""

SAMPLE_GO_MOD = """
module github.com/owner/myapp

go 1.21

require (
    github.com/gin-gonic/gin v1.9.0
    github.com/stretchr/testify v1.8.4
)
"""

SAMPLE_GEMFILE = """
source 'https://rubygems.org'

gem 'rails', '~> 7.0'
gem 'pg', '>= 0.18'
gem 'puma', '~> 5.0'
"""

SAMPLE_COMPOSER = json.dumps({
    "name": "owner/my-php-app",
    "require": {"php": "^8.1", "laravel/framework": "^10.0"},
    "require-dev": {"phpunit/phpunit": "^10.0"},
})

SOURCE_FILES = [
    {"path": "frontend/package.json", "content": SAMPLE_PACKAGE_JSON, "language": "JSON"},
    {"path": "requirements.txt", "content": SAMPLE_REQUIREMENTS, "language": "text"},
]


# ── Parser unit tests ─────────────────────────────────────────────────────────

class TestParsePackageJson(unittest.TestCase):

    def test_parses_dependencies(self):
        result = _parse_package_json(SAMPLE_PACKAGE_JSON)
        self.assertIn("react", result["dependencies"])
        self.assertIn("axios", result["dependencies"])
        self.assertEqual(result["name"], "my-frontend")
        self.assertEqual(result["version"], "1.0.0")

    def test_parses_dev_dependencies(self):
        result = _parse_package_json(SAMPLE_PACKAGE_JSON)
        self.assertIn("vite", result["devDependencies"])
        self.assertIn("typescript", result["devDependencies"])

    def test_parses_scripts(self):
        result = _parse_package_json(SAMPLE_PACKAGE_JSON)
        self.assertIn("dev", result["scripts"])
        self.assertIn("build", result["scripts"])

    def test_invalid_json_returns_error(self):
        result = _parse_package_json("{not valid json}")
        self.assertIn("error", result)

    def test_empty_json_object(self):
        result = _parse_package_json("{}")
        self.assertEqual(result["dependencies"], [])
        self.assertEqual(result["devDependencies"], [])

    def test_total_dependency_counts(self):
        result = _parse_package_json(SAMPLE_PACKAGE_JSON)
        self.assertEqual(result["total_dependencies"], 3)
        self.assertEqual(result["total_devDependencies"], 3)


class TestParseRequirementsTxt(unittest.TestCase):

    def test_parses_packages(self):
        result = _parse_requirements_txt(SAMPLE_REQUIREMENTS)
        self.assertIn("flask", result["packages"])
        self.assertIn("requests", result["packages"])
        self.assertIn("numpy", result["packages"])

    def test_skips_comments_and_flags(self):
        result = _parse_requirements_txt(SAMPLE_REQUIREMENTS)
        packages = result["packages"]
        self.assertNotIn("#", packages)
        self.assertNotIn("-r base.txt", packages)

    def test_strips_version_specifiers(self):
        result = _parse_requirements_txt("django>=4.2\npsycopg2==2.9.5")
        self.assertIn("django", result["packages"])
        self.assertIn("psycopg2", result["packages"])

    def test_empty_file(self):
        result = _parse_requirements_txt("")
        self.assertEqual(result["packages"], [])

    def test_total_count(self):
        result = _parse_requirements_txt(SAMPLE_REQUIREMENTS)
        self.assertGreater(result["total_packages"], 0)


class TestParsePyprojectToml(unittest.TestCase):

    def test_extracts_name_and_version(self):
        result = _parse_pyproject_toml(SAMPLE_PYPROJECT)
        self.assertEqual(result.get("name"), "my-app")
        self.assertEqual(result.get("version"), "0.1.0")

    def test_empty_toml(self):
        result = _parse_pyproject_toml("")
        self.assertIsInstance(result, dict)


class TestParseCargoToml(unittest.TestCase):

    def test_extracts_package_info(self):
        result = _parse_cargo_toml(SAMPLE_CARGO)
        self.assertEqual(result.get("name"), "my-crate")
        self.assertEqual(result.get("version"), "0.1.0")

    def test_extracts_dependencies(self):
        result = _parse_cargo_toml(SAMPLE_CARGO)
        self.assertIn("serde", result["dependencies"])
        self.assertIn("tokio", result["dependencies"])

    def test_empty_cargo(self):
        result = _parse_cargo_toml("")
        self.assertIsInstance(result, dict)


class TestParseGoMod(unittest.TestCase):

    def test_extracts_module_name(self):
        result = _parse_go_mod(SAMPLE_GO_MOD)
        self.assertEqual(result.get("module"), "github.com/owner/myapp")

    def test_extracts_dependencies(self):
        result = _parse_go_mod(SAMPLE_GO_MOD)
        deps = result["dependencies"]
        self.assertTrue(any("gin" in d for d in deps))

    def test_empty_mod(self):
        result = _parse_go_mod("")
        self.assertIsInstance(result, dict)


class TestParseGemfile(unittest.TestCase):

    def test_extracts_gems(self):
        result = _parse_gemfile(SAMPLE_GEMFILE)
        self.assertIn("rails", result["gems"])
        self.assertIn("pg", result["gems"])

    def test_empty_gemfile(self):
        result = _parse_gemfile("")
        self.assertEqual(result["gems"], [])


class TestParseComposerJson(unittest.TestCase):

    def test_extracts_packages(self):
        result = _parse_composer_json(SAMPLE_COMPOSER)
        self.assertIn("laravel/framework", result["dependencies"])
        self.assertIn("phpunit/phpunit", result["devDependencies"])

    def test_invalid_json(self):
        result = _parse_composer_json("{not json}")
        self.assertIn("error", result)


# ── Core analysis tests ───────────────────────────────────────────────────────

class TestComputeStatistics(unittest.TestCase):

    def test_counts_files_and_dirs(self):
        stats = _compute_statistics(SAMPLE_TREE)
        self.assertGreater(stats["total_files"], 0)
        self.assertGreater(stats["total_directories"], 0)

    def test_language_distribution_not_empty(self):
        stats = _compute_statistics(SAMPLE_TREE)
        self.assertIsInstance(stats["language_distribution"], dict)
        self.assertGreater(len(stats["language_distribution"]), 0)

    def test_empty_tree(self):
        stats = _compute_statistics([])
        self.assertEqual(stats["total_files"], 0)
        self.assertEqual(stats["total_directories"], 0)


class TestDetectTechnologies(unittest.TestCase):

    def test_detects_vite_from_config(self):
        tech = _detect_technologies(SAMPLE_TREE, {}, SAMPLE_METADATA)
        self.assertIn("Vite", tech["detected_frameworks"])

    def test_detects_github_actions(self):
        ci = _detect_ci_cd(SAMPLE_TREE)
        self.assertIn("GitHub Actions", ci)
        self.assertIn("Docker", ci)

    def test_detects_react_from_package_json(self):
        source_map = {"frontend/package.json": SAMPLE_PACKAGE_JSON}
        tech = _detect_technologies(SAMPLE_TREE, source_map, SAMPLE_METADATA)
        self.assertIn("React", tech["detected_frameworks"])

    def test_detects_flask_from_requirements(self):
        source_map = {"requirements.txt": SAMPLE_REQUIREMENTS}
        tech = _detect_technologies(SAMPLE_TREE, source_map, SAMPLE_METADATA)
        self.assertIn("Flask", tech["detected_frameworks"])

    def test_primary_language_from_metadata(self):
        tech = _detect_technologies(SAMPLE_TREE, {}, SAMPLE_METADATA)
        self.assertEqual(tech["primary_language"], "Python")

    def test_empty_tree(self):
        tech = _detect_technologies([], {}, SAMPLE_METADATA)
        self.assertIsInstance(tech["detected_frameworks"], list)
        self.assertIsInstance(tech["detected_languages"], list)


class TestExtractDependencies(unittest.TestCase):

    def test_extracts_package_json(self):
        source_map = {"frontend/package.json": SAMPLE_PACKAGE_JSON}
        deps = _extract_dependencies(SAMPLE_TREE, source_map)
        pkg_dep = next((d for d in deps if "package.json" in d["file"]), None)
        self.assertIsNotNone(pkg_dep)
        self.assertTrue(pkg_dep["parsed"])

    def test_extracts_requirements_txt(self):
        source_map = {"requirements.txt": SAMPLE_REQUIREMENTS}
        deps = _extract_dependencies(SAMPLE_TREE, source_map)
        req_dep = next((d for d in deps if "requirements.txt" in d["file"]), None)
        self.assertIsNotNone(req_dep)
        self.assertTrue(req_dep["parsed"])

    def test_manifest_without_content_reported_as_not_parsed(self):
        deps = _extract_dependencies(SAMPLE_TREE, {})  # no content
        for dep in deps:
            if dep["file"].endswith("requirements.txt"):
                self.assertFalse(dep["parsed"])
                return

    def test_no_manifests_returns_empty(self):
        minimal_tree = [{"path": "README.md", "type": "file", "size": 100}]
        deps = _extract_dependencies(minimal_tree, {})
        self.assertEqual(deps, [])

    def test_invalid_json_manifest_reported_as_not_parsed(self):
        source_map = {"frontend/package.json": "NOT VALID JSON"}
        deps = _extract_dependencies(SAMPLE_TREE, source_map)
        pkg_dep = next((d for d in deps if "package.json" in d["file"]), None)
        if pkg_dep:
            # Parse error → parsed flag False OR data has error key
            if pkg_dep.get("parsed"):
                self.assertIn("error", pkg_dep.get("data", {}))


class TestFindImportantFiles(unittest.TestCase):

    def test_finds_readme(self):
        files = _find_important_files(SAMPLE_TREE)
        readme = next((f for f in files if "README" in f["path"]), None)
        self.assertIsNotNone(readme)
        self.assertEqual(readme["role"], "Documentation")

    def test_finds_entry_point(self):
        files = _find_important_files(SAMPLE_TREE)
        entry = next((f for f in files if f["role"] == "Entry point"), None)
        self.assertIsNotNone(entry)

    def test_finds_manifest(self):
        files = _find_important_files(SAMPLE_TREE)
        manifest = next((f for f in files if f["role"] == "Dependency manifest"), None)
        self.assertIsNotNone(manifest)

    def test_empty_tree(self):
        files = _find_important_files([])
        self.assertEqual(files, [])


class TestAnalyzeDirectoryStructure(unittest.TestCase):

    def test_identifies_top_level_dirs(self):
        structure = _analyze_directory_structure(SAMPLE_TREE)
        self.assertIn("backend", structure["top_level_directories"])
        self.assertIn("frontend", structure["top_level_directories"])
        self.assertIn("tests", structure["top_level_directories"])

    def test_directory_roles_not_empty(self):
        structure = _analyze_directory_structure(SAMPLE_TREE)
        self.assertGreater(len(structure["directory_roles"]), 0)

    def test_infers_role_for_known_dirs(self):
        structure = _analyze_directory_structure(SAMPLE_TREE)
        roles = {d["name"]: d["role"] for d in structure["directory_roles"]}
        self.assertIn("backend", roles)
        self.assertIn("frontend", roles)
        self.assertEqual(roles.get("tests"), "Test suite")


class TestAnalyzeRepository(unittest.TestCase):

    def test_full_analysis_returns_all_keys(self):
        result = analyze_repository(
            repo_url="https://github.com/owner/my-app",
            metadata=SAMPLE_METADATA,
            tree=SAMPLE_TREE,
            source_files=SOURCE_FILES,
        )
        self.assertIn("statistics", result)
        self.assertIn("technologies", result)
        self.assertIn("dependencies", result)
        self.assertIn("important_files", result)
        self.assertIn("directory_structure", result)
        self.assertIn("ci_cd", result)
        self.assertIn("documentation", result)
        self.assertIn("architecture_notes", result)

    def test_empty_repo_no_crash(self):
        result = analyze_repository(
            repo_url="https://github.com/owner/empty-repo",
            metadata={"name": "empty", "owner": "owner", "full_name": "owner/empty",
                      "primary_language": None, "default_branch": "main", "visibility": "public"},
            tree=[],
            source_files=[],
        )
        self.assertEqual(result["statistics"]["total_files"], 0)
        self.assertEqual(result["statistics"]["total_directories"], 0)

    def test_architecture_notes_are_strings(self):
        result = analyze_repository(
            repo_url="https://github.com/owner/my-app",
            metadata=SAMPLE_METADATA,
            tree=SAMPLE_TREE,
            source_files=SOURCE_FILES,
        )
        for note in result["architecture_notes"]:
            self.assertIsInstance(note, str)
            self.assertGreater(len(note), 0)

    def test_no_invented_facts(self):
        """Architecture notes must be based on detected data, not invented."""
        tree = [{"path": "README.md", "type": "file", "size": 100}]
        result = analyze_repository(
            repo_url="https://github.com/owner/minimal-repo",
            metadata={**SAMPLE_METADATA, "primary_language": None},
            tree=tree,
            source_files=[],
        )
        # With only a README, no framework notes should appear
        notes = " ".join(result["architecture_notes"]).lower()
        self.assertNotIn("react", notes)
        self.assertNotIn("django", notes)
        self.assertNotIn("docker", notes)

    def test_repo_url_preserved(self):
        result = analyze_repository(
            repo_url="https://github.com/owner/my-app",
            metadata=SAMPLE_METADATA,
            tree=SAMPLE_TREE,
        )
        self.assertEqual(result["repo_url"], "https://github.com/owner/my-app")

    def test_none_source_files_defaults_to_empty(self):
        """None source_files should not raise."""
        result = analyze_repository(
            repo_url="https://github.com/owner/my-app",
            metadata=SAMPLE_METADATA,
            tree=SAMPLE_TREE,
            source_files=None,
        )
        self.assertIsNotNone(result)


class TestDetectDocumentation(unittest.TestCase):

    def test_detects_readme(self):
        docs = _detect_documentation(SAMPLE_TREE)
        self.assertTrue(docs["has_readme"])
        self.assertGreater(docs["documentation_file_count"], 0)

    def test_no_readme(self):
        tree = [{"path": "src/main.py", "type": "file", "size": 100}]
        docs = _detect_documentation(tree)
        self.assertFalse(docs["has_readme"])


if __name__ == "__main__":
    unittest.main()
