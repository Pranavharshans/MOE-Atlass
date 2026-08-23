"""Scoped, model-neutral remote-code compatibility bridge contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from moeatlas.runtime.remote_code_compat import remote_code_compatibility


def _transformers(import_utils: object) -> object:
    return SimpleNamespace(utils=SimpleNamespace(import_utils=import_utils))


def test_missing_legacy_fx_predicate_is_scoped_and_reported() -> None:
    import_utils = SimpleNamespace()
    with remote_code_compatibility(_transformers(import_utils), enabled=True) as bridges:
        assert import_utils.is_torch_fx_available() is True
        assert [bridge.name for bridge in bridges] == [
            "transformers.is_torch_fx_available"
        ]
    assert not hasattr(import_utils, "is_torch_fx_available")


def test_existing_transformers_symbol_is_never_replaced() -> None:
    def sentinel() -> bool:
        return False

    import_utils = SimpleNamespace(is_torch_fx_available=sentinel)
    with remote_code_compatibility(_transformers(import_utils), enabled=True) as bridges:
        assert bridges == ()
        assert import_utils.is_torch_fx_available is sentinel
    assert import_utils.is_torch_fx_available is sentinel


def test_disabled_bridge_does_not_mutate_transformers() -> None:
    import_utils = SimpleNamespace()
    with remote_code_compatibility(_transformers(import_utils), enabled=False) as bridges:
        assert bridges == ()
        assert not hasattr(import_utils, "is_torch_fx_available")


def test_bridge_is_removed_when_remote_import_raises() -> None:
    import_utils = SimpleNamespace()
    with pytest.raises(RuntimeError, match="remote import failed"):
        with remote_code_compatibility(_transformers(import_utils), enabled=True):
            raise RuntimeError("remote import failed")
    assert not hasattr(import_utils, "is_torch_fx_available")
