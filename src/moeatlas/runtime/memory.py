"""Best-effort release of accelerator allocator state after runtime work.

Model loading can fail after a framework has reserved CUDA blocks but before a
model object is returned to the caller. Python reference collection followed
by the PyTorch allocator's cache release makes those blocks available to a
subsequent job in the same server process. The helper is deliberately lazy
and optional-runtime-free at import time.
"""

from __future__ import annotations

import gc
import sys


def release_accelerator_memory() -> None:
    """Release unreferenced model objects and cached CUDA blocks, best effort.

    This never imports PyTorch and never turns cleanup into a new job failure.
    Allocated tensors that are still owned by a live model cannot be released
    by this function; callers must close that model first.
    """

    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        cuda = getattr(torch, "cuda", None)
        if cuda is None:
            return
        is_available = getattr(cuda, "is_available", None)
        if callable(is_available) and not is_available():
            return
        empty_cache = getattr(cuda, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
        ipc_collect = getattr(cuda, "ipc_collect", None)
        if callable(ipc_collect):
            ipc_collect()
    except Exception:
        # Cleanup must not mask the original model/load/run failure.
        return


__all__ = ["release_accelerator_memory"]
