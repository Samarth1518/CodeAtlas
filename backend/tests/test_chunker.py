"""
tests/test_chunker.py — Unit tests for services/chunker.py
"""

import sys
import unittest
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from services.chunker import chunk_file, generate_chunk_id, process_repository_files


class ChunkerTestCase(unittest.TestCase):
    # ── Python AST chunking ───────────────────────────────────────────────────

    def test_chunk_python_functions_and_classes(self):
        source = (
            "import os\n"
            "import sys\n\n"
            "def add(a: int, b: int) -> int:\n"
            "    \"\"\"Add two numbers.\"\"\"\n"
            "    return a + b\n\n"
            "class Calculator:\n"
            "    def multiply(self, x: int, y: int) -> int:\n"
            "        return x * y\n"
        )
        chunks = chunk_file(
            repo_url="https://github.com/owner/repo",
            file_path="src/calc.py",
            content=source,
            language="Python",
        )

        self.assertGreater(len(chunks), 0)
        # Check metadata
        for c in chunks:
            self.assertEqual(c["repo_url"], "https://github.com/owner/repo")
            self.assertEqual(c["file_path"], "src/calc.py")
            self.assertEqual(c["language"], "Python")
            self.assertIn("chunk_id", c)
            self.assertGreaterEqual(c["start_line"], 1)
            self.assertLessEqual(c["start_line"], c["end_line"])
            self.assertTrue(len(c["content"]) > 0)
            self.assertEqual(c["line_count"], c["end_line"] - c["start_line"] + 1)

    def test_python_syntax_error_falls_back_gracefully(self):
        source = (
            "def broken_function(\n"
            "    print('syntax error missing parenthesis'\n"
            "    x = 10\n"
            "    y = 20\n"
        )
        chunks = chunk_file(
            repo_url="https://github.com/owner/repo",
            file_path="src/broken.py",
            content=source,
            language="Python",
        )
        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0]["file_path"], "src/broken.py")

    # ── Markdown chunking ─────────────────────────────────────────────────────

    def test_chunk_markdown_headers(self):
        source = (
            "# Introduction\n\n"
            "Welcome to the project.\n\n"
            "## Getting Started\n\n"
            "Run npm install.\n\n"
            "## Architecture\n\n"
            "This application uses microservices.\n"
        )
        chunks = chunk_file(
            repo_url="https://github.com/owner/repo",
            file_path="README.md",
            content=source,
            language="Markdown",
        )
        self.assertGreaterEqual(len(chunks), 2)
        # Check that markdown headings are present
        found_getting_started = any("Getting Started" in c["content"] for c in chunks)
        self.assertTrue(found_getting_started)

    # ── TypeScript / JavaScript chunking ──────────────────────────────────────

    def test_chunk_typescript_declarations(self):
        source = (
            "import React from 'react';\n\n"
            "export interface UserProps {\n"
            "  name: string;\n"
            "  age: number;\n"
            "}\n\n"
            "export function UserCard(props: UserProps) {\n"
            "  return <div>{props.name}</div>;\n"
            "}\n\n"
            "export const formatUserName = (user: UserProps) => {\n"
            "  return user.name.toUpperCase();\n"
            "};\n"
        )
        chunks = chunk_file(
            repo_url="https://github.com/owner/repo",
            file_path="src/UserCard.tsx",
            content=source,
            language="TypeScript",
        )
        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0]["language"], "TypeScript")

    # ── Plain text / JSON fallback chunking ───────────────────────────────────

    def test_chunk_json_file(self):
        source = "{\n" + "\n".join(f'  "key_{i}": "value_{i}",' for i in range(50)) + "\n}"
        chunks = chunk_file(
            repo_url="https://github.com/owner/repo",
            file_path="config.json",
            content=source,
            language="JSON",
        )
        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0]["file_path"], "config.json")

    # ── Empty and whitespace files ────────────────────────────────────────────

    def test_empty_content_returns_no_chunks(self):
        chunks = chunk_file("https://github.com/owner/repo", "empty.py", "")
        self.assertEqual(chunks, [])

    def test_whitespace_only_content_returns_no_chunks(self):
        chunks = chunk_file("https://github.com/owner/repo", "space.py", "   \n\n  \t\n")
        self.assertEqual(chunks, [])

    # ── Large file windowing & line bounds ────────────────────────────────────

    def test_large_file_splits_into_manageable_chunks(self):
        lines = [f"print('Line {i}')" for i in range(1, 300)]
        source = "\n".join(lines)
        chunks = chunk_file(
            repo_url="https://github.com/owner/repo",
            file_path="large_script.py",
            content=source,
            language="Python",
        )
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            # Each chunk should not exceed maximum line ceiling
            self.assertLessEqual(c["line_count"], 150)
            self.assertGreaterEqual(c["start_line"], 1)

    # ── Process repository files batch helper ─────────────────────────────────

    def test_process_repository_files(self):
        files = [
            {"path": "src/app.py", "content": "def main():\n    print('hello')\n", "language": "Python"},
            {"path": "README.md", "content": "# Title\n\nSome doc content.\n", "language": "Markdown"},
        ]
        res = process_repository_files("https://github.com/owner/repo", files)
        self.assertEqual(res["total_files_processed"], 2)
        self.assertGreaterEqual(res["total_chunks"], 2)
        self.assertIn("Python", res["summary"]["language_breakdown"])
        self.assertIn("Markdown", res["summary"]["language_breakdown"])
        self.assertFalse(res["summary"]["truncated"])

    def test_process_repository_files_respects_max_total_chunks(self):
        files = [
            {"path": f"src/file_{i}.py", "content": f"def func_{i}():\n    pass\n", "language": "Python"}
            for i in range(20)
        ]
        res = process_repository_files("https://github.com/owner/repo", files, max_total_chunks=5)
        self.assertLessEqual(res["total_chunks"], 5)
        self.assertTrue(res["summary"]["truncated"])


if __name__ == "__main__":
    unittest.main()
