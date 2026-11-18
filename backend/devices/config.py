"""The device listener's settings, read from the environment.

The listener is off unless `MIST_DEVICE_LISTENER=on`, so a backend without the devices overlay
opens no second socket. The paths default to the read-only mount the overlay gives
`mist-backend` (`/app/devices`, written on the host by `scripts/daily/devices.py`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from backend.errors import MistError

DEFAULT_PORT = 8443
DEFAULT_DIR = Path("/app/devices")
# Inside the backend's own container: the forwarder reaches it over the compose network, and the
# host side is published by the forwarder on one LAN address only.
ALL_INTERFACES = "0.0.0.0"  # nosec B104 - the container's interfaces, published by the forwarder


class DeviceListenerConfigError(MistError):
    """A `MIST_DEVICE_*` variable has a value the listener cannot use."""


@dataclass(frozen=True, slots=True)
class DeviceListenerConfig:
    """Where and how the device listener serves.

    `host` is not read from the environment: the listener always binds every interface of the
    backend's container, and tests bind loopback.
    """

    enabled: bool = False
    port: int = DEFAULT_PORT
    tls_cert: Path = DEFAULT_DIR / "tls" / "server.pem"
    tls_key: Path = DEFAULT_DIR / "tls" / "server.key"
    registry: Path = DEFAULT_DIR / "registry.json"
    host: str = ALL_INTERFACES

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> DeviceListenerConfig:
        """Read the five `MIST_DEVICE_*` variables; an unset one takes its default.

        Raises:
            DeviceListenerConfigError: `MIST_DEVICE_LISTENER` is not `on` or `off`, or
                `MIST_DEVICE_PORT` is not a port number.
        """
        switch = env.get("MIST_DEVICE_LISTENER", "off").strip().lower()
        if switch not in ("on", "off"):
            raise DeviceListenerConfigError(
                f"MIST_DEVICE_LISTENER must be on or off, not {switch!r}"
            )
        raw_port = env.get("MIST_DEVICE_PORT", str(DEFAULT_PORT)).strip()
        try:
            port = int(raw_port)
        except ValueError as exc:
            raise DeviceListenerConfigError(
                f"MIST_DEVICE_PORT must be a port number, not {raw_port!r}"
            ) from exc
        if not 1 <= port <= 65535:
            raise DeviceListenerConfigError(f"MIST_DEVICE_PORT {port} is not a port number")
        defaults = cls()
        return cls(
            enabled=switch == "on",
            port=port,
            tls_cert=Path(env.get("MIST_DEVICE_TLS_CERT") or defaults.tls_cert),
            tls_key=Path(env.get("MIST_DEVICE_TLS_KEY") or defaults.tls_key),
            registry=Path(env.get("MIST_DEVICE_REGISTRY") or defaults.registry),
        )
