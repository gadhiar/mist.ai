"""Gate thresholds: repository defaults overlaid by the root's calibrated values.

The defaults live in `gates_default.json`, each with a note saying whether H6 calibrates it on
the rehearsal. H6 writes its measured values to `<root>/config/gates.json`, so calibration is
data rather than a code change. The overlay may set only keys the defaults define, and only to
numbers: a typo in a key would otherwise leave the default silently in force.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .errors import InputError

DEFAULTS_PATH = Path(__file__).with_name("gates_default.json")
OVERLAY_REL = "config/gates.json"


@dataclass(frozen=True, slots=True)
class GateDefault:
    """A gate's default value, whether H6 calibrates it, and what it means."""

    value: float
    h6: bool
    note: str


@dataclass(frozen=True, slots=True)
class Gates:
    """The gate defaults and the overlay applied to them."""

    defaults: Mapping[str, GateDefault]
    overlay: Mapping[str, float]

    def value(self, key: str) -> float:
        """Return a gate's effective value.

        Raises:
            InputError: The key is not a known gate.
        """
        if key not in self.defaults:
            raise InputError(f"unknown gate {key}")
        return self.overlay.get(key, self.defaults[key].value)

    def effective(self) -> dict[str, float]:
        """Return every gate's effective value: the thresholds a report records it used."""
        return {key: self.overlay.get(key, d.value) for key, d in self.defaults.items()}


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _read(path: Path, what: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InputError(f"the {what} file is missing") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError(f"the {what} file is not readable JSON") from exc


def load_defaults(path: Path = DEFAULTS_PATH) -> dict[str, GateDefault]:
    """Load the repository's gate defaults.

    Raises:
        InputError: The default file is missing or malformed.
    """
    data = _read(path, "gate default")
    gates = data.get("gates") if isinstance(data, dict) else None
    if not isinstance(gates, dict):
        raise InputError("the gate default file has no gates object")
    defaults: dict[str, GateDefault] = {}
    for key, entry in gates.items():
        if (
            not isinstance(entry, dict)
            or set(entry) != {"value", "h6", "note"}
            or not _is_number(entry["value"])
            or not isinstance(entry["h6"], bool)
            or not isinstance(entry["note"], str)
        ):
            raise InputError(f"the gate default file's entry {key} is malformed")
        defaults[key] = GateDefault(value=entry["value"], h6=entry["h6"], note=entry["note"])
    return defaults


def load(root_path: Path | None, *, defaults_path: Path = DEFAULTS_PATH) -> Gates:
    """Load the defaults and, when the root has one, its `config/gates.json` overlay.

    Args:
        root_path: The root directory, or `None` for defaults only.
        defaults_path: The default file; tests point it elsewhere.

    Raises:
        InputError: The default file is missing or malformed, or the overlay is not an object,
            names an unknown gate, or sets a value that is not a number.
    """
    defaults = load_defaults(defaults_path)
    overlay: dict[str, float] = {}
    overlay_path = root_path / OVERLAY_REL if root_path is not None else None
    if overlay_path is not None and overlay_path.exists():
        data = _read(overlay_path, OVERLAY_REL)
        if not isinstance(data, dict):
            raise InputError(f"{OVERLAY_REL} must be a JSON object of gate values")
        for key, value in data.items():
            if key not in defaults:
                raise InputError(f"{OVERLAY_REL} names an unknown gate {key}")
            if not _is_number(value):
                raise InputError(f"{OVERLAY_REL} sets {key} to something that is not a number")
            overlay[key] = value
    return Gates(defaults=MappingProxyType(defaults), overlay=MappingProxyType(overlay))
