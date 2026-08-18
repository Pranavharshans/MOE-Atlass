from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ProjectFilesTests(unittest.TestCase):
    def test_packaging_and_deferred_validation_docs_are_present(self) -> None:
        required_files = (
            ROOT / "pyproject.toml",
            ROOT / "LICENSE",
            ROOT / "docs" / "architecture.md",
            ROOT / "docs" / "development.md",
            ROOT / "docs" / "model-validation-ledger.md",
        )
        for path in required_files:
            with self.subTest(path=path):
                self.assertTrue(path.is_file(), path)

    def test_validation_ledger_keeps_model_execution_deferred(self) -> None:
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        self.assertIn("Status: deferred", ledger)
        self.assertIn("No model files are downloaded", ledger)
        self.assertIn("final VM", ledger)


if __name__ == "__main__":
    unittest.main()
