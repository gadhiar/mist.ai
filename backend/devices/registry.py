"""The device registry: which room devices may connect, and how each proves it.

`scripts/daily/devices.py` writes `<devices dir>/registry.json` on the host, and the backend reads
it from its read-only mount:

    {"version": 1, "devices": [{"name", "area", "token_sha256", "revoked", "created"}]}

A device's bearer token is `<name>.<secret>`, the secret being 32 random bytes in unpadded
base64url. The registry keeps only the secret's sha256, so the file, which the backend can read,
cannot be replayed as a token.

A malformed file refuses to load. A malformed entry is skipped and logged, so it never
authenticates, and the rest still load. A reload that finds the file broken or gone drops every
record: a registry that cannot be read admits no device.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from backend.errors import MistError

logger = logging.getLogger(__name__)

REGISTRY_VERSION = 1
SECRET_BYTES = 32
TOKEN_SEPARATOR = "."  # nosec B105 - the separator between name and secret, not a secret
NAME_PATTERN = re.compile(r"[a-z][a-z0-9-]{0,23}", re.ASCII)
# Endpoint names MIST already uses or reserves; a device named one would be confused with it.
RESERVED_NAMES = frozenset({"desktop", "all", "room"})
SHA256_HEX = re.compile(r"[0-9a-f]{64}", re.ASCII)
# Compared against when the claimed name is unknown, so a miss costs what a hit does.
_NO_DEVICE_DIGEST = "0" * 64


class DeviceRegistryError(MistError):
    """The registry file is missing, unreadable or not a version-1 registry."""


@dataclass(frozen=True, slots=True)
class DeviceRecord:
    """One registered device; `token_sha256` is the hex sha256 of its 32-byte secret."""

    name: str
    area: str
    token_sha256: str
    revoked: bool
    created: datetime


def valid_name(name: str) -> bool:
    """Whether `name` is usable as a device name, which is also its endpoint name."""
    return NAME_PATTERN.fullmatch(name) is not None and name not in RESERVED_NAMES


def claimed_name(bearer: str) -> str | None:
    """The device name a bearer token claims, unchecked; for logging a refused attempt."""
    name, separator, _secret = bearer.partition(TOKEN_SEPARATOR)
    if not separator or not name:
        return None
    return name


def _decode_secret(encoded: str) -> bytes | None:
    try:
        secret = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    except (binascii.Error, ValueError):
        return None
    return secret if len(secret) == SECRET_BYTES else None


def _parse_created(raw: object) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        moment = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else None


def _record(index: int, raw: object) -> DeviceRecord | None:
    """The entry as a record, or None (logged) when any field is off the format."""
    if not isinstance(raw, dict):
        logger.warning("device_registry_entry_skipped index=%d reason=not_an_object", index)
        return None
    name = raw.get("name")
    area = raw.get("area")
    digest = raw.get("token_sha256")
    revoked = raw.get("revoked")
    created = _parse_created(raw.get("created"))
    problem = None
    if not isinstance(name, str) or not valid_name(name):
        problem = "name"
    elif not isinstance(area, str) or not area.strip():
        problem = "area"
    elif not isinstance(digest, str) or SHA256_HEX.fullmatch(digest) is None:
        problem = "token_sha256"
    elif not isinstance(revoked, bool):
        problem = "revoked"
    elif created is None:
        problem = "created"
    if problem is not None:
        shown = name if isinstance(name, str) and valid_name(name) else "?"
        logger.warning(
            "device_registry_entry_skipped index=%d device=%s reason=%s", index, shown, problem
        )
        return None
    return DeviceRecord(name, area, digest, revoked, created)


def _stamp(path: Path) -> tuple[int, int, int] | None:
    """What changes when the file is rewritten: the CLI replaces it, so the inode moves too."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_ino, stat.st_size)


def _read(path: Path) -> tuple[DeviceRecord, ...]:
    """Every well-formed record in the file.

    Raises:
        DeviceRegistryError: When the file cannot be read, is not a version-1 registry, or
            lists a name twice.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise DeviceRegistryError(f"cannot read the device registry {path}: {exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DeviceRegistryError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("version") != REGISTRY_VERSION:
        raise DeviceRegistryError(f"{path} is not a version-{REGISTRY_VERSION} device registry")
    entries = payload.get("devices")
    if not isinstance(entries, list):
        raise DeviceRegistryError(f"{path} has no devices list")
    records: list[DeviceRecord] = []
    seen: set[str] = set()
    for index, raw in enumerate(entries):
        record = _record(index, raw)
        if record is None:
            continue
        if record.name in seen:
            raise DeviceRegistryError(f"{path} lists the device {record.name} twice")
        seen.add(record.name)
        records.append(record)
    return tuple(records)


class DeviceRegistry:
    """The registered devices, re-read when the file changes.

    `verify` is the listener's whole authentication: it never raises, and it answers None for
    anything but a well-formed token of a registered, unrevoked device.
    """

    def __init__(
        self,
        path: Path,
        records: tuple[DeviceRecord, ...],
        stamp: tuple[int, int, int] | None,
    ) -> None:
        self._path = path
        self._records = records
        self._by_name = {record.name: record for record in records}
        self._stamp = stamp

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> DeviceRegistry:
        """Read the registry at `path`.

        Raises:
            DeviceRegistryError: When the file is missing, unreadable or malformed.
        """
        registry_path = Path(path)
        stamp = _stamp(registry_path)
        return cls(registry_path, _read(registry_path), stamp)

    def devices(self) -> tuple[DeviceRecord, ...]:
        """Every well-formed record, revoked ones included, in file order."""
        return self._records

    def area_of(self, name: str) -> str | None:
        """The area a device was registered for, or None for an unknown name."""
        record = self._by_name.get(name)
        return record.area if record is not None else None

    def verify(self, bearer: str) -> DeviceRecord | None:
        """The record whose secret `bearer` carries, or None."""
        name, separator, encoded = bearer.partition(TOKEN_SEPARATOR)
        if not separator:
            return None
        secret = _decode_secret(encoded)
        if secret is None:
            return None
        record = self._by_name.get(name)
        expected = record.token_sha256 if record is not None else _NO_DEVICE_DIGEST
        matched = hmac.compare_digest(hashlib.sha256(secret).hexdigest(), expected)
        if record is None or not matched or record.revoked:
            return None
        return record

    def reload_if_changed(self) -> bool:
        """Re-read the file when its modification time, inode or size changed.

        Returns:
            True when the records were replaced, including by none at all because the file
            became unreadable, malformed or missing; False when the file is unchanged.
        """
        stamp = _stamp(self._path)
        if stamp == self._stamp and stamp is not None:
            return False
        if stamp is None and self._stamp is None and not self._records:
            return False
        try:
            records = _read(self._path)
        except DeviceRegistryError as exc:
            logger.error("device_registry_unreadable error=%s; admitting no device", exc)
            records = ()
        self._records = records
        self._by_name = {record.name: record for record in records}
        self._stamp = stamp
        return True
