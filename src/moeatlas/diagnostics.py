"""Model-free environment diagnostics for the initial MoEAtlas foundation.

This module deliberately inspects package metadata without importing PyTorch,
Transformers, or any model loader. A diagnostic report is useful before the
model runtime exists, and makes the deferred validation boundary visible to
users and CI.
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from typing import Any

from . import PRODUCT_NAME, __version__

_OPTIONAL_RUNTIME_PACKAGES = ("torch", "transformers", "safetensors")


def _is_importable(module_name: str) -> bool:
    """Return whether a package can be found without importing it."""

    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        # ``find_spec`` can raise when a partially initialized module is in
        # sys.modules. Diagnostics should remain best-effort in that case.
        return False


def collect_doctor_report() -> dict[str, Any]:
    """Build a JSON-serializable, model-free environment report.

    The validation status is intentionally explicit. Presence of an optional
    package does not imply that a model was loaded or that a GPU test passed.
    """

    return {
        "product": PRODUCT_NAME,
        "package": "moeatlas",
        "package_version": __version__,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "supported": sys.version_info >= (3, 11),
        },
        "optional_runtime_packages": {
            name: {"available": _is_importable(name)} for name in _OPTIONAL_RUNTIME_PACKAGES
        },
        "validation": {
            "model_and_gpu": {
                "status": "deferred",
                "reason": (
                    "Checkpoint downloads and model/GPU execution are deferred "
                    "to the final VM validation phase."
                ),
                "model_downloads_performed": False,
            }
        },
    }
