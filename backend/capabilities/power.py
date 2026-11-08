"""The power port: the level MIST runs at, and which model tiers can serve a turn now.

Capabilities consume this interface and the tier router implements it. It holds no policy:
capabilities only show and log the level, and whether a decode can run is whether the
`interaction` tier has a provider. Which tiers a level allows is the router's policy, so this
module has no level-to-tier table.

Until the router lands, two stopgaps serve the daily build: `GpuYieldPower` reads the level from
gaming mode, and `SingleModelTiers` serves both tiers from the one current provider.

`GpuYield` and `StreamingLLMProvider` are imported for typing only, so importing this module
loads neither the gaming-mode watcher nor the LLM provider stack.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from backend.gpu_yield import GpuYield
    from backend.llm.provider import StreamingLLMProvider

PowerLevel = Literal["full", "low", "off"]
ModelTier = Literal["interaction", "smart"]


class PowerState(Protocol):
    """The level MIST runs at; the router implements it. Non-blocking, safe from any thread."""

    def level(self) -> PowerLevel:
        """Return the current level, which capabilities show and log but never decide on."""
        ...


class TierProviders(Protocol):
    """The provider of each model tier; the router implements it. Non-blocking, any thread."""

    def provider(self, tier: ModelTier) -> StreamingLLMProvider | None:
        """Return the tier's provider, or None when it cannot serve now.

        None covers both a level that forbids the tier and a tier that is not loaded yet.
        """
        ...


class FixedPower:
    """A level that never changes: for tests, and for builds with no gaming mode."""

    __slots__ = ("_level",)

    def __init__(self, level: PowerLevel = "full") -> None:
        self._level: PowerLevel = level

    def level(self) -> PowerLevel:
        """Return the level given at construction."""
        return self._level


class GpuYieldPower:
    """The stopgap level: gaming mode's `GpuYield` state, read at each call.

    `normal` is full. `yielded` and `resume_failed` are off, because both mean the GPU models
    are released or not yet reloaded. No `GpuYield` (gaming mode not built) is full. A state
    this code does not know reads as off, the side `GpuYield` itself errs on when unsure.
    """

    __slots__ = ("_gpu_yield",)

    def __init__(self, gpu_yield: GpuYield | None) -> None:
        self._gpu_yield = gpu_yield

    def level(self) -> PowerLevel:
        """Return full or off from the yield state at this moment."""
        if self._gpu_yield is None:
            return "full"
        return "full" if self._gpu_yield.state() == "normal" else "off"


class SingleModelTiers:
    """The stopgap tiers: today's one model serves both tiers, and none serves at off.

    The provider is read through `get_provider` at each call, because it exists only after the
    voice processor's `initialize()`; until then the getter returns None and so does every
    tier.
    """

    __slots__ = ("_get_provider", "_power")

    def __init__(
        self, get_provider: Callable[[], StreamingLLMProvider | None], power: PowerState
    ) -> None:
        self._get_provider = get_provider
        self._power = power

    def provider(self, tier: ModelTier) -> StreamingLLMProvider | None:
        """Return the one current provider for either tier, or None at off or before it exists."""
        if self._power.level() == "off":
            return None
        return self._get_provider()


class FixedTiers:
    """A provider per tier, or None, fixed at construction: for tests."""

    __slots__ = ("_providers",)

    def __init__(self, providers: Mapping[ModelTier, StreamingLLMProvider | None]) -> None:
        self._providers = dict(providers)

    def provider(self, tier: ModelTier) -> StreamingLLMProvider | None:
        """Return the provider given for `tier`, or None when none was given."""
        return self._providers.get(tier)
