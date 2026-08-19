"""
tests/test_file_filter.py — Unit tests for services/file_filter.py
"""

import sys
import unittest
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from services.file_filter import MAX_FILE_SIZE_BYTES, classify_file, get_extension_language


class ClassifyFileTestCase(unittest.TestCase):
    # ── Should fetch ──────────────────────────────────────────────────────────

    def test_python_file_allowed(self):
        ok, reason = classify_file("src/app.py", 1024)
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_typescript_file_allowed(self):
        ok, reason = classify_file("frontend/src/App.tsx", 2048)
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_markdown_readme_allowed(self):
        ok, reason = classify_file("README.md", 512)
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_dockerfile_bare_name_allowed(self):
        ok, reason = classify_file("Dockerfile", 300)
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_makefile_bare_name_allowed(self):
        ok, reason = classify_file("Makefile", 400)
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_yaml_config_allowed(self):
        ok, reason = classify_file(".github/workflows/ci.yml", 800)
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_json_config_allowed(self):
        ok, reason = classify_file("package.json", 1200)
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_env_example_allowed(self):
        ok, reason = classify_file(".env.example", 200)
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_env_sample_allowed(self):
        ok, reason = classify_file(".env.sample", 200)
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    # ── Binary / media ────────────────────────────────────────────────────────

    def test_png_image_skipped(self):
        ok, reason = classify_file("assets/logo.png", 20000)
        self.assertFalse(ok)
        self.assertIn("binary/media", reason)

    def test_jpg_skipped(self):
        ok, reason = classify_file("docs/screenshot.jpg", 50000)
        self.assertFalse(ok)
        self.assertIn("binary/media", reason)

    def test_wasm_skipped(self):
        # dist/ directory is excluded first; .wasm is also binary but dir check wins
        ok, reason = classify_file("dist/app.wasm", 300000)
        self.assertFalse(ok)
        # Either reason is acceptable: dir exclusion or binary extension
        self.assertTrue(
            "dist" in reason or "binary/media" in reason,
            msg=f"Unexpected skip reason: {reason}",
        )

    def test_zip_archive_skipped(self):
        ok, reason = classify_file("release.zip", 1024 * 1024)
        self.assertFalse(ok)
        self.assertIn("binary/media", reason)

    def test_ttf_font_skipped(self):
        ok, reason = classify_file("fonts/Inter.ttf", 180000)
        self.assertFalse(ok)
        self.assertIn("binary/media", reason)

    # ── Sensitive files ───────────────────────────────────────────────────────

    def test_dotenv_skipped(self):
        ok, reason = classify_file(".env", 100)
        self.assertFalse(ok)
        self.assertIn("sensitive", reason)

    def test_env_production_skipped(self):
        ok, reason = classify_file(".env.production", 100)
        self.assertFalse(ok)
        self.assertIn("sensitive", reason)

    def test_env_local_skipped(self):
        ok, reason = classify_file(".env.local", 100)
        self.assertFalse(ok)
        self.assertIn("sensitive", reason)

    def test_pem_cert_skipped(self):
        ok, reason = classify_file("certs/server.pem", 2000)
        self.assertFalse(ok)
        self.assertIn("sensitive", reason)

    def test_private_key_skipped(self):
        ok, reason = classify_file("id_rsa", 1600)
        self.assertFalse(ok)
        self.assertIn("sensitive", reason)

    def test_service_account_json_skipped(self):
        ok, reason = classify_file("service-account.json", 800)
        self.assertFalse(ok)
        self.assertIn("sensitive", reason)

    # ── Excluded directories ──────────────────────────────────────────────────

    def test_node_modules_skipped(self):
        ok, reason = classify_file("node_modules/lodash/index.js", 5000)
        self.assertFalse(ok)
        self.assertIn("node_modules", reason)

    def test_dot_git_skipped(self):
        ok, reason = classify_file(".git/config", 200)
        self.assertFalse(ok)
        self.assertIn(".git", reason)

    def test_dist_dir_skipped(self):
        ok, reason = classify_file("dist/bundle.js", 50000)
        self.assertFalse(ok)
        self.assertIn("dist", reason)

    def test_venv_dir_skipped(self):
        ok, reason = classify_file("venv/lib/python3.11/site.py", 1000)
        self.assertFalse(ok)
        self.assertIn("venv", reason)

    def test_pycache_skipped(self):
        ok, reason = classify_file("src/__pycache__/app.cpython-311.pyc", 5000)
        self.assertFalse(ok)
        self.assertIn("__pycache__", reason)

    # ── Size limit ────────────────────────────────────────────────────────────

    def test_file_at_limit_allowed(self):
        ok, reason = classify_file("big.py", MAX_FILE_SIZE_BYTES)
        self.assertTrue(ok)

    def test_file_over_limit_skipped(self):
        ok, reason = classify_file("huge.py", MAX_FILE_SIZE_BYTES + 1)
        self.assertFalse(ok)
        self.assertIn("too large", reason)

    def test_none_size_not_blocked_by_size_check(self):
        # When size is unknown (None), we don't pre-block on size
        ok, reason = classify_file("main.py", None)
        self.assertTrue(ok)

    # ── Unknown extensions ────────────────────────────────────────────────────

    def test_unknown_extension_skipped(self):
        ok, reason = classify_file("data.xyz123", 500)
        self.assertFalse(ok)
        self.assertIn("unsupported file type", reason)


class GetExtensionLanguageTestCase(unittest.TestCase):
    def test_python(self):
        self.assertEqual(get_extension_language("app.py"), "Python")

    def test_typescript(self):
        self.assertEqual(get_extension_language("App.tsx"), "TypeScript")

    def test_javascript(self):
        self.assertEqual(get_extension_language("index.js"), "JavaScript")

    def test_markdown(self):
        self.assertEqual(get_extension_language("README.md"), "Markdown")

    def test_yaml(self):
        self.assertEqual(get_extension_language("config.yaml"), "YAML")

    def test_unknown_returns_none(self):
        self.assertIsNone(get_extension_language("binary.xyz"))

    def test_no_extension_returns_none(self):
        self.assertIsNone(get_extension_language("Makefile"))


if __name__ == "__main__":
    unittest.main()
