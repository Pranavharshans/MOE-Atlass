"""MoEAtlas: local-first observability for Mixture-of-Experts models.

The foundation package intentionally has no model-runtime import side effects.
PyTorch, Transformers, and checkpoint loading will be added behind explicit
features in later implementation slices.
"""

from __future__ import annotations

__all__ = ["__version__", "PRODUCT_NAME"]

PRODUCT_NAME = "MoEAtlas"
__version__ = "0.1.0"
