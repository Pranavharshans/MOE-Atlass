from __future__ import annotations

import io
import json
import sys
import unittest
from contextlib import redirect_stdout

import moeatlas
from moeatlas.cli import main
from moeatlas.diagnostics import collect_doctor_report


class FoundationTests(unittest.TestCase):
    def test_package_metadata_is_stable(self) -> None:
        self.assertEqual(moeatlas.PRODUCT_NAME, "MoEAtlas")
        self.assertRegex(moeatlas.__version__, r"^\d+\.\d+\.\d+$")

    def test_version_command_is_model_free(self) -> None:
        output = io.StringIO()
        with self.assertRaises(SystemExit) as exit_context, redirect_stdout(output):
            main(["--version"])

        self.assertEqual(exit_context.exception.code, 0)
        self.assertEqual(output.getvalue().strip(), f"moeatlas {moeatlas.__version__}")

    def test_doctor_json_reports_deferred_validation(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(["doctor", "--json"])

        self.assertEqual(exit_code, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["product"], "MoEAtlas")
        self.assertTrue(report["python"]["supported"])
        validation = report["validation"]["model_and_gpu"]
        self.assertEqual(validation["status"], "deferred")
        self.assertFalse(validation["model_downloads_performed"])

    def test_doctor_report_only_checks_optional_package_presence(self) -> None:
        report = collect_doctor_report()
        package_status = report["optional_runtime_packages"]
        self.assertEqual(set(package_status), {"torch", "transformers", "safetensors"})
        for info in package_status.values():
            self.assertIsInstance(info["available"], bool)

    def test_diagnostics_do_not_import_model_runtime_modules(self) -> None:
        runtime_names = {"torch", "transformers", "safetensors"}
        before = {name for name in runtime_names if name in sys.modules}
        collect_doctor_report()
        after = {name for name in runtime_names if name in sys.modules}
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
