"""Torch-free executable hook surface for lifecycle tests."""

from __future__ import annotations

from collections import defaultdict


class SyntheticHookHandle:
    def __init__(
        self,
        owner: SyntheticHookModule,
        hook_name: str,
        callback: object,
    ) -> None:
        self.owner = owner
        self.hook_name = hook_name
        self.callback = callback
        self.removed = False

    def remove(self) -> None:
        self.owner.removal_log.append((self.owner.name, self.hook_name))
        if self.hook_name in self.owner.fail_removals:
            raise RuntimeError(f"synthetic removal failure: {self.owner.name}")
        remaining_failures = self.owner.transient_removals.get(self.hook_name, 0)
        if remaining_failures:
            self.owner.transient_removals[self.hook_name] = remaining_failures - 1
            raise RuntimeError(f"synthetic transient removal failure: {self.owner.name}")
        self.removed = True
        callbacks = self.owner.callbacks[self.hook_name]
        if self.callback in callbacks:
            callbacks.remove(self.callback)


class SyntheticHookModule:
    def __init__(
        self,
        name: str,
        *,
        registration_log: list[tuple[str, str]],
        removal_log: list[tuple[str, str]],
        fail_registrations: set[str] | None = None,
        fail_removals: set[str] | None = None,
        transient_removals: dict[str, int] | None = None,
    ) -> None:
        self.name = name
        self.registration_log = registration_log
        self.removal_log = removal_log
        self.fail_registrations = set(fail_registrations or ())
        self.fail_removals = set(fail_removals or ())
        self.transient_removals = dict(transient_removals or {})
        self.callbacks: dict[str, list[object]] = defaultdict(list)

    def _register(self, hook_name: str, callback: object) -> SyntheticHookHandle:
        self.registration_log.append((self.name, hook_name))
        if hook_name in self.fail_registrations:
            raise RuntimeError(f"synthetic registration failure: {self.name}")
        self.callbacks[hook_name].append(callback)
        return SyntheticHookHandle(self, hook_name, callback)

    def register_forward_pre_hook(self, callback: object) -> SyntheticHookHandle:
        return self._register("forward_pre", callback)

    def register_forward_hook(self, callback: object) -> SyntheticHookHandle:
        return self._register("forward", callback)

    def register_full_backward_hook(self, callback: object) -> SyntheticHookHandle:
        return self._register("full_backward", callback)

    def fire(self, hook_name: str, *args: object) -> list[object]:
        return [callback(*args) for callback in tuple(self.callbacks[hook_name])]


class SyntheticHookModel:
    def __init__(
        self,
        *,
        modules: dict[str, SyntheticHookModule] | None = None,
    ) -> None:
        self.registration_log: list[tuple[str, str]] = []
        self.removal_log: list[tuple[str, str]] = []
        self.modules = modules or {
            "layers.0.router": SyntheticHookModule(
                "layers.0.router",
                registration_log=self.registration_log,
                removal_log=self.removal_log,
            ),
            "layers.1.router": SyntheticHookModule(
                "layers.1.router",
                registration_log=self.registration_log,
                removal_log=self.removal_log,
            ),
        }

    def named_modules(self):
        yield "", self
        yield from self.modules.items()


__all__ = ["SyntheticHookHandle", "SyntheticHookModel", "SyntheticHookModule"]
