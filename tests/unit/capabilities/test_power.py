"""Tests for the power port: the two protocols, and the stopgaps until the tier router lands."""

from __future__ import annotations

import importlib
import inspect
import sys
from collections.abc import Iterable, Mapping
from typing import cast

import pytest

from backend.capabilities import power
from backend.capabilities.power import (
    FixedPower,
    FixedTiers,
    GpuYieldPower,
    PowerLevel,
    PowerState,
    SingleModelTiers,
    TierProviders,
)
from backend.llm.provider import StreamingLLMProvider
from tests.unit.capabilities.fakes import FakePower

PROVIDER = cast(StreamingLLMProvider, object())
SMART = cast(StreamingLLMProvider, object())


class FakeGpuYield:
    """A `GpuYield` whose state a test sets, and which counts each read."""

    def __init__(self, state: str = "normal") -> None:
        self.current = state
        self.reads = 0

    def state(self) -> str:
        self.reads += 1
        return self.current


class CountingGetter:
    """The provider getter `SingleModelTiers` takes, answering `current` and counting calls."""

    def __init__(self, provider: StreamingLLMProvider | None) -> None:
        self.current = provider
        self.calls = 0

    def __call__(self) -> StreamingLLMProvider | None:
        self.calls += 1
        return self.current


class TestModule:
    def test_imports_with_neither_typed_module_loaded(self, monkeypatch):
        for name in ("backend.gpu_yield", "backend.llm.provider", "backend.capabilities.power"):
            monkeypatch.delitem(sys.modules, name, raising=False)

        module = importlib.import_module("backend.capabilities.power")

        assert module.SingleModelTiers is not None
        assert "backend.gpu_yield" not in sys.modules
        assert "backend.llm.provider" not in sys.modules

    def test_declares_the_two_protocols_with_one_method_each(self):
        assert [name for name in vars(PowerState) if name == "level"] == ["level"]
        assert [name for name in vars(TierProviders) if name == "provider"] == ["provider"]
        assert list(inspect.signature(TierProviders.provider).parameters) == ["self", "tier"]

    def test_holds_no_level_to_tier_table(self):
        levels = {"full", "low", "off"}
        tiers = {"interaction", "smart"}
        for value in vars(power).values():
            if not isinstance(value, Mapping):
                continue
            keyed_by_level = bool(value) and set(value) <= levels
            assert not (keyed_by_level and any(_names_a_tier(v, tiers) for v in value.values()))


def _names_a_tier(value: object, tiers: set[str]) -> bool:
    if isinstance(value, str):
        return value in tiers
    if isinstance(value, Iterable):
        return any(item in tiers for item in value)
    return False


class TestFixedPower:
    def test_defaults_to_full(self):
        assert FixedPower().level() == "full"

    @pytest.mark.parametrize("level", ["full", "low", "off"])
    def test_reads_the_level_it_was_given(self, level: PowerLevel):
        assert FixedPower(level).level() == level


class TestGpuYieldPower:
    @pytest.mark.parametrize(
        ("state", "level"), [("normal", "full"), ("yielded", "off"), ("resume_failed", "off")]
    )
    def test_maps_each_yield_state(self, state: str, level: str):
        assert GpuYieldPower(FakeGpuYield(state)).level() == level  # type: ignore[arg-type]

    def test_reads_the_state_at_each_call(self):
        gpu_yield = FakeGpuYield("normal")
        state = GpuYieldPower(gpu_yield)  # type: ignore[arg-type]

        first = state.level()
        gpu_yield.current = "yielded"
        second = state.level()
        gpu_yield.current = "normal"
        third = state.level()

        assert (first, second, third) == ("full", "off", "full")
        assert gpu_yield.reads == 3

    def test_no_gpu_yield_reads_full(self):
        assert GpuYieldPower(None).level() == "full"


class TestSingleModelTiers:
    @pytest.mark.parametrize("level", ["full", "low"])
    @pytest.mark.parametrize("tier", ["interaction", "smart"])
    def test_serves_both_tiers_from_the_one_provider(self, level: PowerLevel, tier):
        tiers = SingleModelTiers(CountingGetter(PROVIDER), FakePower(level))

        assert tiers.provider(tier) is PROVIDER

    @pytest.mark.parametrize("tier", ["interaction", "smart"])
    def test_serves_nothing_at_off(self, tier):
        tiers = SingleModelTiers(CountingGetter(PROVIDER), FakePower("off"))

        assert tiers.provider(tier) is None

    def test_serves_nothing_while_the_getter_returns_none(self):
        getter = CountingGetter(None)
        tiers = SingleModelTiers(getter, FakePower("full"))

        assert tiers.provider("interaction") is None
        getter.current = PROVIDER
        assert tiers.provider("interaction") is PROVIDER

    def test_calls_the_getter_and_reads_the_level_at_each_call(self):
        getter = CountingGetter(PROVIDER)
        level = FakePower("full")
        tiers = SingleModelTiers(getter, level)

        tiers.provider("interaction")
        tiers.provider("smart")
        level.current = "off"
        at_off = tiers.provider("interaction")
        level.current = "low"
        at_low = tiers.provider("interaction")

        assert getter.calls >= 3
        assert level.reads == 4
        assert (at_off, at_low) == (None, PROVIDER)


class TestFixedTiers:
    def test_serves_each_tier_it_was_given_and_none_for_the_rest(self):
        tiers = FixedTiers({"interaction": PROVIDER, "smart": None})

        assert tiers.provider("interaction") is PROVIDER
        assert tiers.provider("smart") is None
        assert FixedTiers({"smart": SMART}).provider("interaction") is None
