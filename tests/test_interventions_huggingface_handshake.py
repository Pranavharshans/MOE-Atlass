"""Model-free contracts for the reversible HF expert-backend handshake."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from moeatlas.interventions import (
    ExpertBackendHandshakeStatus,
    TransformersInterventionError,
    run_huggingface_expert_handshake,
)


class _Registry(dict[str, Callable[..., object]]):
    def register(self, name: str, function: Callable[..., object]) -> None:
        if name in self:
            raise KeyError(name)
        self[name] = function

    def unregister(self, name: str) -> None:
        del self[name]


class _Model:
    def __init__(
        self,
        registry: _Registry,
        implementation: str = "grouped_mm",
        *,
        expert: object | None = None,
    ) -> None:
        self.registry = registry
        self.implementations: dict[str, str | None] = {"": implementation}
        self.expert = expert if expert is not None else object()

    def get_experts_implementation(self) -> dict[str, str | None]:
        return dict(self.implementations)

    def set_experts_implementation(self, value: str | dict[str, str | None]) -> None:
        requested = {"": value} if isinstance(value, str) else dict(value)
        for implementation in requested.values():
            if (
                implementation is not None
                and implementation != "eager"
                and implementation not in self.registry
            ):
                raise KeyError(implementation)
        self.implementations = requested

    def forward(self, value: object) -> object:
        implementation = self.implementations[""]
        assert isinstance(implementation, str)
        return self.registry[implementation](self.expert, value)


def _fixture() -> tuple[_Model, _Registry, object]:
    expected = object()
    registry = _Registry(grouped_mm=lambda _module, _value: expected)
    return _Model(registry), registry, expected


def test_pass_through_handshake_exercises_delegate_and_restores_exact_state() -> None:
    model, registry, expected = _fixture()
    original_registry = dict(registry)

    output, report = run_huggingface_expert_handshake(
        model,
        lambda: model.forward(object()),
        registry=registry,
    )

    assert output is expected
    assert report.status is ExpertBackendHandshakeStatus.VERIFIED
    assert report.invocation_counts == (("grouped_mm", 1),)
    assert report.restored is True
    assert model.implementations == {"": "grouped_mm"}
    assert registry == original_registry


def test_handshake_restores_model_and_registry_when_forward_raises() -> None:
    model, registry, _expected = _fixture()
    original_registry = dict(registry)
    failure = RuntimeError("forward failed")

    def fail() -> None:
        model.forward(object())
        raise failure

    with pytest.raises(RuntimeError) as raised:
        run_huggingface_expert_handshake(model, fail, registry=registry)

    assert raised.value is failure
    assert model.implementations == {"": "grouped_mm"}
    assert registry == original_registry


def test_partially_failed_backend_switch_restores_then_runs_normally() -> None:
    class PartialFailureModel(_Model):
        failed = False

        def set_experts_implementation(self, value: str | dict[str, str | None]) -> None:
            super().set_experts_implementation(value)
            if not self.failed and self.implementations[""].startswith("moeatlas_passthrough_"):
                self.failed = True
                raise RuntimeError("partial switch failure")

    _model, registry, expected = _fixture()
    model = PartialFailureModel(registry)
    original_registry = dict(registry)

    output, report = run_huggingface_expert_handshake(
        model,
        lambda: model.forward(object()),
        registry=registry,
    )

    assert output is expected
    assert report.status is ExpertBackendHandshakeStatus.UNAVAILABLE
    assert model.implementations == {"": "grouped_mm"}
    assert registry == original_registry


def test_restoration_failure_is_fatal_and_registry_is_still_cleaned() -> None:
    class RestoreFailureModel(_Model):
        switched = False

        def set_experts_implementation(self, value: str | dict[str, str | None]) -> None:
            requested = {"": value} if isinstance(value, str) else dict(value)
            if self.switched and requested == {"": "grouped_mm"}:
                raise RuntimeError("restore failed")
            super().set_experts_implementation(requested)
            if str(self.implementations[""]).startswith("moeatlas_passthrough_"):
                self.switched = True

    _model, registry, expected = _fixture()
    model = RestoreFailureModel(registry)

    with pytest.raises(TransformersInterventionError, match="restoration failed"):
        run_huggingface_expert_handshake(
            model,
            lambda: expected,
            registry=registry,
        )

    assert set(registry) == {"grouped_mm"}


def test_handshake_reports_forward_that_does_not_exercise_experts() -> None:
    model, registry, expected = _fixture()

    output, report = run_huggingface_expert_handshake(
        model,
        lambda: expected,
        registry=registry,
    )

    assert output is expected
    assert report.status is ExpertBackendHandshakeStatus.NOT_EXERCISED
    assert report.invocation_counts == (("grouped_mm", 0),)
    assert model.implementations == {"": "grouped_mm"}
    assert set(registry) == {"grouped_mm"}


def test_unavailable_interface_runs_the_forward_without_mutation() -> None:
    expected = object()

    output, report = run_huggingface_expert_handshake(object(), lambda: expected)

    assert output is expected
    assert report.status is ExpertBackendHandshakeStatus.UNAVAILABLE
    assert report.restored is True


def test_model_using_a_private_registry_falls_back_after_clean_restore() -> None:
    expected = object()
    public_registry = _Registry(grouped_mm=lambda _module, _value: object())
    private_registry = _Registry(grouped_mm=lambda _module, _value: expected)
    model = _Model(public_registry)

    def execute() -> object:
        implementation = model.implementations[""]
        assert isinstance(implementation, str)
        return private_registry[implementation](model.expert, object())

    output, report = run_huggingface_expert_handshake(
        model,
        execute,
        registry=public_registry,
    )

    assert output is expected
    assert report.status is ExpertBackendHandshakeStatus.UNAVAILABLE
    assert model.implementations == {"": "grouped_mm"}
    assert set(public_registry) == {"grouped_mm"}


def test_mixed_backends_are_wrapped_independently() -> None:
    registry = _Registry(
        grouped_mm=lambda _module, value: ("grouped", value),
        sonicmoe=lambda _module, value: ("sonic", value),
    )
    model = _Model(registry)
    model.implementations = {"text": "grouped_mm", "vision": "sonicmoe"}

    def execute() -> tuple[object, object]:
        text_backend = model.implementations["text"]
        vision_backend = model.implementations["vision"]
        assert isinstance(text_backend, str) and isinstance(vision_backend, str)
        return (
            registry[text_backend](model.expert, 1),
            registry[vision_backend](model.expert, 2),
        )

    output, report = run_huggingface_expert_handshake(
        model,
        execute,
        registry=registry,
    )

    assert output == (("grouped", 1), ("sonic", 2))
    assert report.invocation_counts == (("grouped_mm", 1), ("sonicmoe", 1))
    assert model.implementations == {"text": "grouped_mm", "vision": "sonicmoe"}
    assert set(registry) == {"grouped_mm", "sonicmoe"}


def test_eager_backend_delegates_to_decorator_original_forward() -> None:
    expected = object()

    def original_forward(_expert: object, _value: object) -> object:
        return expected

    def decorated_forward(_expert: object, _value: object) -> object:
        raise AssertionError("temporary backend should replace decorated dispatch")

    decorated_forward.__wrapped__ = original_forward  # type: ignore[attr-defined]
    expert_type = type("Expert", (), {"forward": decorated_forward})
    registry = _Registry()
    model = _Model(registry, "eager", expert=expert_type())

    output, report = run_huggingface_expert_handshake(
        model,
        lambda: model.forward(object()),
        registry=registry,
    )

    assert output is expected
    assert report.status is ExpertBackendHandshakeStatus.VERIFIED
    assert report.invocation_counts == (("eager", 1),)
    assert model.implementations == {"": "eager"}
    assert registry == {}
