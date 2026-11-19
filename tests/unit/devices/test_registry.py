"""Tests for `backend/devices/registry.py`: what the device listener trusts.

The registry file is written by `scripts/daily/devices.py`; the round-trip tests drive that CLI
with a fake runner, so no `openssl` runs and no key is made. Every file lives under `tmp_path`.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from backend.devices import registry as registry_module
from backend.devices.registry import DeviceRecord, DeviceRegistry, DeviceRegistryError
from backend.errors import MistError
from scripts.daily import devices

CREATED = "2026-11-17T09:00:00Z"
SERVER_IP = "192.168.1.20"
SAN_OUTPUT = f"X509v3 Subject Alternative Name: \n    IP Address:{SERVER_IP}\n".encode()


def secret_for(seed: int) -> bytes:
    return bytes([seed]) * 32


def token_of(name: str, secret: bytes) -> str:
    return f"{name}.{base64.urlsafe_b64encode(secret).decode().rstrip('=')}"


def entry(name: str, secret: bytes, *, area: str = "example-area", revoked: bool = False) -> dict:
    return {
        "name": name,
        "area": area,
        "token_sha256": hashlib.sha256(secret).hexdigest(),
        "revoked": revoked,
        "created": CREATED,
    }


def write_registry(path: Path, entries: list, *, version: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": version, "devices": entries}), encoding="utf-8")
    return path


@pytest.fixture
def path(tmp_path) -> Path:
    return tmp_path / "devices" / "registry.json"


# --- Loading ---------------------------------------------------------------------------------


def test_load_reads_each_record(path):
    write_registry(path, [entry("example-room", secret_for(1), area="example-area")])

    registry = DeviceRegistry.load(path)

    assert registry.devices() == (
        DeviceRecord(
            name="example-room",
            area="example-area",
            token_sha256=hashlib.sha256(secret_for(1)).hexdigest(),
            revoked=False,
            created=datetime(2026, 11, 17, 9, 0, tzinfo=UTC),
        ),
    )


def test_device_registry_error_is_a_mist_error():
    assert issubclass(DeviceRegistryError, MistError)


def test_load_refuses_a_missing_file(path):
    with pytest.raises(DeviceRegistryError, match="registry.json"):
        DeviceRegistry.load(path)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("not json", id="not-json"),
        pytest.param("[]", id="not-an-object"),
        pytest.param('{"version": 2, "devices": []}', id="unknown-version"),
        pytest.param('{"version": 1, "devices": {}}', id="devices-not-a-list"),
        pytest.param('{"version": 1}', id="no-devices"),
    ],
)
def test_load_refuses_a_malformed_file(path, text):
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")

    with pytest.raises(DeviceRegistryError):
        DeviceRegistry.load(path)


def test_load_refuses_a_name_listed_twice(path):
    write_registry(
        path, [entry("example-room", secret_for(1)), entry("example-room", secret_for(2))]
    )

    with pytest.raises(DeviceRegistryError, match="example-room"):
        DeviceRegistry.load(path)


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"name": "Example"}, id="name-off-pattern"),
        pytest.param({"name": "desktop"}, id="reserved-name"),
        pytest.param({"area": ""}, id="empty-area"),
        pytest.param({"token_sha256": "ab" * 31}, id="short-hash"),
        pytest.param({"token_sha256": "AB" * 32}, id="uppercase-hash"),
        pytest.param({"revoked": "no"}, id="revoked-not-bool"),
        pytest.param({"created": "yesterday"}, id="created-unparseable"),
        pytest.param({"created": "2026-11-17T09:00:00"}, id="created-without-zone"),
    ],
)
def test_a_malformed_entry_is_skipped_and_never_verifies(path, change):
    bad = {**entry("example-bad", secret_for(1)), **change}
    write_registry(path, [bad, entry("example-good", secret_for(2))])

    registry = DeviceRegistry.load(path)

    assert [record.name for record in registry.devices()] == ["example-good"]
    assert registry.verify(token_of(bad["name"], secret_for(1))) is None


def test_an_entry_that_is_not_an_object_is_skipped(path):
    write_registry(path, ["example-room", entry("example-good", secret_for(2))])

    assert [r.name for r in DeviceRegistry.load(path).devices()] == ["example-good"]


# --- Verification ----------------------------------------------------------------------------


def test_verify_returns_the_record_for_its_token(path):
    write_registry(
        path, [entry("example-room", secret_for(1)), entry("example-hall", secret_for(2))]
    )
    registry = DeviceRegistry.load(path)

    record = registry.verify(token_of("example-hall", secret_for(2)))

    assert record is not None and record.name == "example-hall"


@pytest.mark.parametrize(
    "bearer",
    [
        pytest.param(token_of("example-room", secret_for(9)), id="wrong-secret"),
        pytest.param(token_of("example-hall", secret_for(1)), id="another-devices-secret"),
        pytest.param(token_of("example-none", secret_for(1)), id="unknown-name"),
        pytest.param("example-room", id="no-secret"),
        pytest.param("example-room.", id="empty-secret"),
        pytest.param("example-room.!!!!", id="not-base64"),
        pytest.param(
            f"example-room.{base64.urlsafe_b64encode(b'x' * 16).decode().rstrip('=')}",
            id="short-secret",
        ),
        pytest.param("", id="empty"),
    ],
)
def test_verify_refuses_a_wrong_or_malformed_bearer(path, bearer):
    write_registry(
        path, [entry("example-room", secret_for(1)), entry("example-hall", secret_for(2))]
    )

    assert DeviceRegistry.load(path).verify(bearer) is None


def test_verify_refuses_a_revoked_device(path):
    write_registry(path, [entry("example-room", secret_for(1), revoked=True)])

    assert DeviceRegistry.load(path).verify(token_of("example-room", secret_for(1))) is None


def test_verify_compares_with_hmac_compare_digest(path, monkeypatch):
    write_registry(path, [entry("example-room", secret_for(1))])
    registry = DeviceRegistry.load(path)
    compared: list[tuple[str, str]] = []
    original = hmac.compare_digest

    def spy(first, second):
        compared.append((first, second))
        return original(first, second)

    monkeypatch.setattr(registry_module.hmac, "compare_digest", spy)

    assert registry.verify(token_of("example-room", secret_for(1))) is not None
    assert registry.verify(token_of("example-none", secret_for(1))) is None
    # The unknown name is compared too, against a hash no secret has, so a miss takes as long.
    assert len(compared) == 2


def test_claimed_name_reads_the_id_without_checking_the_secret():
    assert registry_module.claimed_name(token_of("example-room", secret_for(1))) == "example-room"
    assert registry_module.claimed_name("example-room.anything") == "example-room"
    assert registry_module.claimed_name("no-separator") is None
    assert registry_module.claimed_name(".secret") is None


def test_area_of_names_the_devices_area(path):
    write_registry(path, [entry("example-room", secret_for(1), area="example-area")])
    registry = DeviceRegistry.load(path)

    assert registry.area_of("example-room") == "example-area"
    assert registry.area_of("example-none") is None


# --- Reloading -------------------------------------------------------------------------------


def bump_mtime(path: Path) -> None:
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))


def test_reload_if_changed_is_false_while_the_file_is_unchanged(path):
    write_registry(path, [entry("example-room", secret_for(1))])
    registry = DeviceRegistry.load(path)

    assert registry.reload_if_changed() is False
    assert registry.verify(token_of("example-room", secret_for(1))) is not None


def test_reload_if_changed_picks_up_a_revocation(path):
    write_registry(path, [entry("example-room", secret_for(1))])
    registry = DeviceRegistry.load(path)

    write_registry(path, [entry("example-room", secret_for(1), revoked=True)])
    bump_mtime(path)

    assert registry.reload_if_changed() is True
    assert registry.verify(token_of("example-room", secret_for(1))) is None
    assert registry.devices()[0].revoked is True


def test_reload_if_changed_picks_up_a_new_device(path):
    write_registry(path, [entry("example-room", secret_for(1))])
    registry = DeviceRegistry.load(path)

    write_registry(
        path, [entry("example-room", secret_for(1)), entry("example-hall", secret_for(2))]
    )
    bump_mtime(path)

    assert registry.reload_if_changed() is True
    assert registry.verify(token_of("example-hall", secret_for(2))) is not None


@pytest.mark.parametrize("damage", ["unreadable", "missing"])
def test_reload_of_a_broken_file_fails_closed(path, damage):
    write_registry(path, [entry("example-room", secret_for(1))])
    registry = DeviceRegistry.load(path)

    if damage == "missing":
        path.unlink()
    else:
        path.write_text("{", encoding="utf-8")
        bump_mtime(path)

    assert registry.reload_if_changed() is True
    assert registry.devices() == ()
    assert registry.verify(token_of("example-room", secret_for(1))) is None


# --- Round trip with the provisioning CLI ----------------------------------------------------


class FakeOpenssl:
    """Writes a placeholder for each `-out` and answers the SAN query; never makes a key."""

    def __call__(self, argv):
        argv = [str(a) for a in argv]
        if "-out" in argv:
            Path(argv[argv.index("-out") + 1]).write_bytes(b"placeholder\n")
        if "subjectAltName" in argv and "-ext" in argv:
            return devices.CommandResult(0, SAN_OUTPUT, b"")
        return devices.CommandResult(0, b"", b"")


def cli(tmp_path: Path, *argv: str) -> int:
    ctx = devices.Context(
        runner=FakeOpenssl(),
        environ={},
        clock=lambda: datetime(2026, 11, 17, 9, 0, tzinfo=UTC),
        which=lambda name: "/usr/bin/openssl",
    )
    return devices.main(list(argv), ctx)


def read_secrets(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and ":" in line:
            key, _, value = line.partition(":")
            values[key.strip()] = value.strip().strip('"')
    return values


@pytest.fixture
def provisioned(tmp_path) -> tuple[Path, Path]:
    device_dir = tmp_path / "devices"
    ca_dir = tmp_path / "devices-ca"
    assert (
        cli(tmp_path, "init", "--dir", str(device_dir), "--ca-dir", str(ca_dir), "--ip", SERVER_IP)
        == 0
    )
    return device_dir, ca_dir


def test_the_registry_verifies_the_token_the_cli_wrote(tmp_path, provisioned):
    device_dir, _ca_dir = provisioned
    out = tmp_path / "private" / "example-room"
    args = (
        "add",
        "example-room",
        "--area",
        "example-area",
        "--out",
        str(out),
        "--dir",
        str(device_dir),
    )
    assert cli(tmp_path, *args) == 0

    registry = DeviceRegistry.load(device_dir / "registry.json")
    token = read_secrets(out / "secrets.yaml")["mist_device_token"]

    record = registry.verify(token)
    assert record is not None
    assert (record.name, record.area, record.revoked) == ("example-room", "example-area", False)
    assert record.created == datetime(2026, 11, 17, 9, 0, tzinfo=UTC)


def test_the_registry_refuses_the_old_token_after_rotate_and_every_token_after_revoke(
    tmp_path, provisioned
):
    device_dir, _ca_dir = provisioned
    out = tmp_path / "private" / "example-room"
    dir_args = ("--dir", str(device_dir))
    assert (
        cli(tmp_path, "add", "example-room", "--area", "example-area", "--out", str(out), *dir_args)
        == 0
    )
    old = read_secrets(out / "secrets.yaml")["mist_device_token"]
    registry = DeviceRegistry.load(device_dir / "registry.json")

    assert cli(tmp_path, "rotate", "example-room", "--out", str(out), *dir_args) == 0
    new = read_secrets(out / "secrets.yaml")["mist_device_token"]
    bump_mtime(device_dir / "registry.json")
    registry.reload_if_changed()
    assert old != new
    assert registry.verify(old) is None
    assert registry.verify(new) is not None

    assert cli(tmp_path, "revoke", "example-room", *dir_args) == 0
    bump_mtime(device_dir / "registry.json")
    registry.reload_if_changed()
    assert registry.verify(new) is None
