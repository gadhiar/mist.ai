"""Confirmation policy: whether a validated call runs, waits for a yes, or is refused.

The mode is per tool: `ToolSpec.default_policy`, optionally overridden by a YAML file
(`MIST_CAPABILITY_POLICY_FILE`, `{tool_name: mode}`). The decision depends on the call's source.
The fast path's anchored grammar is the one deterministic test of an explicit command, so
`confirm_inferred` runs a fast-path call and confirms a model call. A call run by an approved
proposal skips policy, because the approval was the confirmation; only `deny` still refuses it.
The file is the precursor of ADR-020's proposed `permissions` block.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Literal

import yaml
from pydantic import BaseModel

from backend.errors import CapabilityConfigError

from .contracts import Capability, ConfirmPolicy, InvocationContext, ToolSpec

logger = logging.getLogger(__name__)

_MODES: frozenset[str] = frozenset({"allow", "confirm_inferred", "confirm", "deny"})

Decision = Literal["allow", "confirm", "deny"]


class PolicyTable:
    """The effective confirmation mode of each tool."""

    def __init__(self, modes: Mapping[str, ConfirmPolicy]) -> None:
        self._modes = dict(modes)

    @classmethod
    def from_specs(cls, specs: Iterable[ToolSpec], overrides_path: str | None) -> PolicyTable:
        """Build the table from each tool's default, then apply the overrides file.

        An unknown tool name in the file is logged once (all names in one warning) and ignored,
        so a file written for a capability that is not built still loads. Any invalid mode
        refuses the whole file, so a typo never leaves one tool on a weaker mode than intended.

        Args:
            specs: Every registered tool.
            overrides_path: The YAML file, or None for defaults only.

        Raises:
            CapabilityConfigError: The file cannot be read or parsed, is not a mapping of tool
                names to modes, or names an invalid mode. The message is meant to be reported
                to the user while the defaults stay in force.
        """
        modes: dict[str, ConfirmPolicy] = {spec.name: spec.default_policy for spec in specs}
        if overrides_path is None:
            return cls(modes)

        overrides = _read_overrides(overrides_path)
        invalid = sorted(
            f"{name}: {mode!r}"
            for name, mode in overrides.items()
            if not (isinstance(mode, str) and mode in _MODES)
        )
        if invalid:
            raise CapabilityConfigError(
                f"policy file {overrides_path} has invalid modes ({'; '.join(invalid)}); "
                f"valid modes: {', '.join(sorted(_MODES))}"
            )
        unknown = sorted(name for name in overrides if name not in modes)
        if unknown:
            logger.warning(
                "Policy file %s names unknown tools, ignored: %s",
                overrides_path,
                ", ".join(unknown),
            )
        for name, mode in overrides.items():
            if name in modes:
                modes[name] = mode  # type: ignore[assignment]  # checked against _MODES above
        return cls(modes)

    def decide(
        self, spec: ToolSpec, args: BaseModel, ctx: InvocationContext, capability: Capability
    ) -> Decision:
        """Decide a validated call by its tool's mode and the call's source.

        `capability.escalate` can only raise an allowed `model` or `fastpath` call to confirm. It
        is not consulted for source `proposal`: that call was already confirmed once, and a fresh
        proposal would stop it from ever running.
        """
        mode = self._modes.get(spec.name, spec.default_policy)
        if mode == "deny":
            return "deny"
        if ctx.source == "proposal":
            return "allow"
        if mode == "confirm" or (mode == "confirm_inferred" and ctx.source == "model"):
            return "confirm"
        if capability.escalate(spec.name, args):
            return "confirm"
        return "allow"


def _read_overrides(path: str) -> dict[str, object]:
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except OSError as exc:
        raise CapabilityConfigError(f"policy file {path} cannot be read: {exc}") from exc
    except yaml.YAMLError as exc:
        raise CapabilityConfigError(f"policy file {path} is not valid YAML: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
        raise CapabilityConfigError(f"policy file {path} must map tool names to modes")
    return loaded
