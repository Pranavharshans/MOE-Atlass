from __future__ import annotations

import pytest

from moeatlas.runtime.contracts import ModelObservationError
from moeatlas.runtime.observation import observed_parameter_dtype_inventory


class _DType:
    def __init__(self, name: str) -> None:
        self.name = name


class _Parameter:
    def __init__(self, dtype: str, numel: int, element_size: int) -> None:
        self.dtype = _DType(dtype)
        self._numel = numel
        self._element_size = element_size

    def numel(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return self._element_size


class _Model:
    def __init__(self, rows: tuple[tuple[str, _Parameter], ...]) -> None:
        self._rows = rows

    def named_parameters(self):
        return iter(self._rows)


def test_parameter_dtype_inventory_covers_every_tensor_and_aggregates_bytes() -> None:
    model = _Model(
        (
            ("embed.weight", _Parameter("float16", 100, 2)),
            ("layers.0.experts.0.weight", _Parameter("uint8", 50, 1)),
            ("layers.0.experts.1.weight", _Parameter("uint8", 75, 1)),
        )
    )

    inventory, warnings = observed_parameter_dtype_inventory(model)

    assert inventory["status"] == "available"
    assert inventory["tensor_count"] == 3
    assert inventory["element_count"] == 225
    assert inventory["logical_bytes"] == 325
    assert inventory["mixed_dtype"] is True
    assert inventory["dtype_rows"] == [
        {
            "dtype": "float16",
            "tensor_count": 1,
            "sized_tensor_count": 1,
            "element_count": 100,
            "logical_bytes": 200,
        },
        {
            "dtype": "uint8",
            "tensor_count": 2,
            "sized_tensor_count": 2,
            "element_count": 125,
            "logical_bytes": 125,
        },
    ]
    assert str(inventory["inventory_digest"]).startswith("sha256:")
    assert warnings == ("loaded model parameters use multiple dtypes",)


def test_parameter_dtype_inventory_digest_binds_parameter_names() -> None:
    first, _ = observed_parameter_dtype_inventory(
        _Model((("first", _Parameter("float16", 4, 2)),))
    )
    second, _ = observed_parameter_dtype_inventory(
        _Model((("second", _Parameter("float16", 4, 2)),))
    )
    assert first["dtype_rows"] == second["dtype_rows"]
    assert first["inventory_digest"] != second["inventory_digest"]


def test_parameter_dtype_inventory_reports_absence_without_guessing() -> None:
    inventory, warnings = observed_parameter_dtype_inventory(object())
    assert inventory["status"] == "unavailable"
    assert inventory["dtype_rows"] == []
    assert inventory["inventory_digest"] is None
    assert warnings == ("parameter dtype inventory is unavailable",)


def test_parameter_dtype_inventory_is_budget_bounded() -> None:
    model = _Model(
        (
            ("first", _Parameter("float32", 1, 4)),
            ("second", _Parameter("float32", 1, 4)),
        )
    )
    with pytest.raises(ModelObservationError, match="tensor budget"):
        observed_parameter_dtype_inventory(model, max_parameter_tensors=1)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, None])
def test_parameter_dtype_inventory_rejects_invalid_budgets(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        observed_parameter_dtype_inventory(_Model(()), max_parameter_tensors=value)  # type: ignore[arg-type]
