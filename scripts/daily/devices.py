"""Host CLI that provisions room devices: `python scripts/daily/devices.py <command> ...`.

    init --ip <PC LAN IP> [--dir <devices dir>] [--ca-dir <CA dir>]
    add <name> --area <area> --out <private dir> [--dir ...] [--ca-dir ...] [--port 8443]
    list [--dir ...]
    revoke <name> [--dir ...]
    rotate <name> --out <private dir> [--dir ...] [--ca-dir ...] [--port 8443]
    cert --ip <PC LAN IP> [--dir ...] [--ca-dir ...]

Two directories, kept apart. The devices dir (`--dir`, else `MIST_DEVICE_DIR`, else
`<MIST_DAILY_ROOT>/devices`) holds `registry.json` and `tls/` with the server's certificate and
key and the CA's public certificate; it is the one directory a container mounts, read-only. The
CA dir (`--ca-dir`, else `<MIST_DAILY_ROOT>/devices-ca`) holds `ca.key`, which no service mounts
and only `init` and `cert` read. Both live outside every git work tree.

`add` and `rotate` write a device's token only into the private directory's `secrets.yaml`, never
to the terminal or into an `openssl` argument; the registry keeps the secret's sha256 alone.

Like `instance.py`, it runs on host Python with the standard library alone, and every `openssl`
call goes through `Context.runner`, so the tests assert exact argument lists without making a
key. `openssl` is the one on `PATH`, else the copy Git for Windows ships.

Exit codes: 0 done; 2 refused, nothing changed; 1 failed part-way, with the state named.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import ipaddress
import json
import os
import re
import secrets
import shutil
import subprocess  # nosec B404 - runs fixed openssl argument lists, never a shell
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2

REGISTRY_FILE = "registry.json"
REGISTRY_VERSION = 1
TLS_DIR = "tls"
CA_CERT = "ca.pem"
CA_KEY = "ca.key"
SERVER_CERT = "server.pem"
SERVER_KEY = "server.key"
SECRETS_FILE = "secrets.yaml"
DEVICE_CA_FILE = "mist_ca.pem"
DEVICES_SUBDIR = "devices"
CA_SUBDIR = "devices-ca"
DEFAULT_PORT = 8443
CERT_DAYS = "3650"
CA_SUBJECT = "/CN=MIST device CA"
EC_KEY_ARGS = ("genpkey", "-algorithm", "EC", "-pkeyopt", "ec_paramgen_curve:P-256")
GIT_OPENSSL = Path("C:/Program Files/Git/usr/bin/openssl.exe")

# The same rules as `backend/devices/registry.py`, spelled here because this file runs where
# `backend` is not importable; a unit test round-trips the two.
NAME_PATTERN = re.compile(r"[a-z][a-z0-9-]{0,23}", re.ASCII)
RESERVED_NAMES = ("desktop", "all", "room")
SECRET_BYTES = 32
TOKEN_SEPARATOR = "."
AREA_MAX = 64
OTA_PASSWORD_BYTES = 16
PRIVATE_NETWORKS = tuple(
    ipaddress.IPv4Network(net) for net in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)
SAN_IP = re.compile(r"IP Address:([0-9.]+)")
SECRET_KEYS = (
    "mist_device_id",
    "mist_device_token",
    "mist_server_host",
    "mist_server_port",
    "mist_ota_password",
)
SECRETS_HEADER = (
    "# Written by scripts/daily/devices.py for one room device. Private: never commit it.\n"
    "# Add the device's Wi-Fi lines (wifi_ssid, wifi_password) below; add and rotate keep them.\n"
)


@dataclass(frozen=True, slots=True)
class CommandResult:
    """What one `openssl` command returned."""

    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""


class Runner(Protocol):
    """Runs one command and returns its exit code and captured output."""

    def __call__(self, argv: Sequence[str]) -> CommandResult:
        """Run `argv`."""
        ...


def subprocess_runner(argv: Sequence[str]) -> CommandResult:
    """Run `argv` without a shell; a missing program reads as exit 127."""
    command = [str(arg) for arg in argv]
    try:
        completed = subprocess.run(  # nosec B603 - fixed argument list, no shell
            command, stdin=subprocess.DEVNULL, capture_output=True, check=False
        )
    except FileNotFoundError as exc:
        return CommandResult(127, b"", f"{command[0]}: {exc}".encode())
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class Context:
    """Everything a command reads from outside its arguments, injectable for tests."""

    runner: Runner = subprocess_runner
    environ: Mapping[str, str] = field(default_factory=lambda: os.environ)
    clock: Callable[[], datetime] = _utc_now
    which: Callable[[str], str | None] = shutil.which
    git_openssl: Path = GIT_OPENSSL


class DevicesError(Exception):
    """A command stopped; `exit_code` says whether anything changed."""

    exit_code = EXIT_FAILED
    label = "failed"


class RefusedError(DevicesError):
    """A check failed before anything was written."""

    exit_code = EXIT_REFUSED
    label = "refused, nothing changed"


class FailedError(DevicesError):
    """A step failed after earlier steps took effect; the message names the state left."""

    exit_code = EXIT_FAILED
    label = "failed part-way"


# --- Paths -----------------------------------------------------------------------------------


def _resolve(raw: str) -> Path:
    return Path(raw).expanduser().resolve()


def _inside(path: Path, parent: Path) -> bool:
    """Whether `path` is `parent` or below it, compared the way the host's file system does."""
    child = os.path.normcase(str(path))
    base = os.path.normcase(str(parent))
    return child == base or child.startswith(base.rstrip(os.sep) + os.sep)


def _git_work_tree_holding(path: Path) -> Path | None:
    """The nearest directory at or above `path` holding `.git` (a directory or a worktree file)."""
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _refuse_in_work_tree(path: Path, what: str) -> None:
    work_tree = _git_work_tree_holding(path)
    if work_tree is not None:
        raise RefusedError(
            f"the {what} {path} is inside the git work tree {work_tree}; keep it outside every "
            "work tree, so no git or repository tool can reach it"
        )


def _root_default(ctx: Context, subdir: str, flag: str) -> Path:
    root = ctx.environ.get("MIST_DAILY_ROOT", "").strip()
    if not root:
        raise RefusedError(f"pass {flag}, or set MIST_DAILY_ROOT so it defaults to <root>/{subdir}")
    return _resolve(root) / subdir


def _device_dir(args: argparse.Namespace, ctx: Context) -> Path:
    if args.dir:
        return _resolve(args.dir)
    configured = ctx.environ.get("MIST_DEVICE_DIR", "").strip()
    if configured:
        return _resolve(configured)
    return _root_default(ctx, DEVICES_SUBDIR, "--dir")


def _ca_dir(args: argparse.Namespace, ctx: Context) -> Path:
    if args.ca_dir:
        return _resolve(args.ca_dir)
    return _root_default(ctx, CA_SUBDIR, "--ca-dir")


def _check_ca_dir(ca_dir: Path, device_dir: Path) -> None:
    if _inside(ca_dir, device_dir):
        raise RefusedError(
            f"the CA dir {ca_dir} is inside the devices dir {device_dir}, which a container "
            "mounts; keep the CA key apart"
        )
    _refuse_in_work_tree(ca_dir, "CA dir")


def _check_private_dir(out: Path, device_dir: Path, ca_dir: Path | None) -> None:
    _refuse_in_work_tree(out, "private directory")
    if _inside(out, device_dir):
        raise RefusedError(
            f"the private directory {out} is inside the devices dir {device_dir}, which a "
            "container mounts"
        )
    if ca_dir is not None and _inside(out, ca_dir):
        raise RefusedError(f"the private directory {out} is inside the CA dir {ca_dir}")


def _check_ip(raw: str) -> str:
    try:
        address = ipaddress.IPv4Address(raw)
    except ValueError as exc:
        raise RefusedError(f"--ip {raw!r} is not an IPv4 address") from exc
    if not any(address in network for network in PRIVATE_NETWORKS):
        raise RefusedError(
            f"--ip {raw} is not a private LAN address (10/8, 172.16/12 or 192.168/16)"
        )
    return str(address)


def _write_atomic(path: Path, data: bytes) -> None:
    """Write through a private temporary file in the same directory, then replace."""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


# --- openssl ---------------------------------------------------------------------------------


def _find_openssl(ctx: Context) -> str:
    found = ctx.which("openssl")
    if found:
        return found
    if ctx.git_openssl.is_file():
        return str(ctx.git_openssl)
    raise RefusedError(
        f"no openssl found: none on PATH and none at {GIT_OPENSSL.as_posix()} (Git for Windows "
        "ships one there); install Git for Windows or put openssl on PATH"
    )


def _stderr_tail(result: CommandResult) -> str:
    lines = result.stderr.decode("utf-8", "replace").strip().splitlines()
    tail = " | ".join(lines[-5:])
    return f"exit {result.returncode}" + (f": {tail}" if tail else "")


def _openssl(ctx: Context, openssl: str, step: str, *args: str) -> CommandResult:
    result = ctx.runner([openssl, *args])
    if result.returncode != 0:
        raise FailedError(f"openssl could not {step} ({_stderr_tail(result)})")
    return result


def _server_extensions(ip: str) -> str:
    return (
        "basicConstraints=critical,CA:FALSE\n"
        "keyUsage=critical,digitalSignature\n"
        "extendedKeyUsage=serverAuth\n"
        "subjectKeyIdentifier=hash\n"
        "authorityKeyIdentifier=keyid:always\n"
        f"subjectAltName=IP:{ip}\n"
    )


def _issue_server_cert(ctx: Context, openssl: str, device_dir: Path, ca_dir: Path, ip: str) -> None:
    """A fresh server key and certificate for `ip`, signed by the CA; the old pair stays on failure.

    The key and certificate are made beside their final names and replace them only once both
    exist, so a failed step leaves a running listener's pair as it was.
    """
    tls = device_dir / TLS_DIR
    key_new = tls / f"{SERVER_KEY}.new"
    cert_new = tls / f"{SERVER_CERT}.new"
    csr = tls / "server.csr"
    ext = tls / "server.ext"
    serial = f"0x{int(ctx.clock().timestamp() * 1_000_000):x}"
    try:
        _openssl(ctx, openssl, "make the server key", *EC_KEY_ARGS, "-out", str(key_new))
        _openssl(
            ctx,
            openssl,
            "make the server certificate request",
            "req",
            "-new",
            "-key",
            str(key_new),
            "-subj",
            f"/CN={ip}",
            "-out",
            str(csr),
        )
        ext.write_text(_server_extensions(ip), encoding="utf-8")
        _openssl(
            ctx,
            openssl,
            "sign the server certificate",
            "x509",
            "-req",
            "-in",
            str(csr),
            "-CA",
            str(tls / CA_CERT),
            "-CAkey",
            str(ca_dir / CA_KEY),
            "-set_serial",
            serial,
            "-days",
            CERT_DAYS,
            "-sha256",
            "-extfile",
            str(ext),
            "-out",
            str(cert_new),
        )
        os.replace(key_new, tls / SERVER_KEY)
        os.replace(cert_new, tls / SERVER_CERT)
    finally:
        for leftover in (key_new, cert_new, csr, ext):
            with contextlib.suppress(FileNotFoundError):
                leftover.unlink()


def _server_host(ctx: Context, openssl: str, device_dir: Path) -> str:
    """The IP address in the server certificate's SAN, which every device connects to."""
    cert = device_dir / TLS_DIR / SERVER_CERT
    result = ctx.runner([openssl, "x509", "-in", str(cert), "-noout", "-ext", "subjectAltName"])
    found = SAN_IP.findall(result.stdout.decode("utf-8", "replace"))
    if result.returncode != 0 or len(found) != 1:
        raise RefusedError(
            f"cannot read one IP address from {cert}'s subjectAltName ({_stderr_tail(result)}); "
            "run cert --ip <PC LAN IP>"
        )
    return found[0]


# --- The registry ----------------------------------------------------------------------------


def _registry_path(device_dir: Path) -> Path:
    return device_dir / REGISTRY_FILE


def _read_registry(device_dir: Path) -> list[dict]:
    path = _registry_path(device_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RefusedError(f"{path} does not exist; run init first") from exc
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise RefusedError(f"cannot read {path}: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("version") != REGISTRY_VERSION
        or not isinstance(payload.get("devices"), list)
    ):
        raise RefusedError(f"{path} is not a version-{REGISTRY_VERSION} device registry")
    return [entry for entry in payload["devices"] if isinstance(entry, dict)]


def _write_registry(device_dir: Path, entries: list[dict]) -> None:
    payload = {"version": REGISTRY_VERSION, "devices": entries}
    _write_atomic(_registry_path(device_dir), (json.dumps(payload, indent=2) + "\n").encode())


def _find(entries: list[dict], name: str) -> dict | None:
    return next((entry for entry in entries if entry.get("name") == name), None)


def _check_name(name: str) -> None:
    if NAME_PATTERN.fullmatch(name) is None:
        raise RefusedError(
            f"device name {name!r} must match {NAME_PATTERN.pattern}: a lowercase letter, then "
            "up to 23 lowercase letters, digits or hyphens"
        )
    if name in RESERVED_NAMES:
        raise RefusedError(f"device name {name!r} is reserved ({', '.join(RESERVED_NAMES)})")


def _check_area(area: str) -> str:
    cleaned = area.strip()
    if not cleaned or len(cleaned) > AREA_MAX or not cleaned.isprintable():
        raise RefusedError(f"--area must be 1 to {AREA_MAX} printable characters")
    return cleaned


def _format_utc(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- The private directory -------------------------------------------------------------------


def _new_token(name: str) -> tuple[str, str]:
    """A token `<name>.<secret>` and the hex sha256 of its 32-byte secret."""
    secret = secrets.token_bytes(SECRET_BYTES)
    encoded = base64.urlsafe_b64encode(secret).decode("ascii").rstrip("=")
    return f"{name}{TOKEN_SEPARATOR}{encoded}", hashlib.sha256(secret).hexdigest()


def _merge_secrets(existing: str | None, values: Mapping[str, str]) -> str:
    """`existing` with each key's line replaced, or appended; every other line kept as it was."""
    lines = (existing if existing is not None else SECRETS_HEADER).splitlines()
    remaining = dict(values)
    merged = []
    for line in lines:
        key = line.partition(":")[0].strip()
        if not line.startswith("#") and key in remaining:
            merged.append(f"{key}: {remaining.pop(key)}")
        else:
            merged.append(line)
    merged.extend(f"{key}: {value}" for key, value in remaining.items())
    return "\n".join(merged) + "\n"


def _existing_value(text: str | None, key: str) -> str | None:
    for line in (text or "").splitlines():
        found, separator, value = line.partition(":")
        if separator and not line.startswith("#") and found.strip() == key:
            return value.strip()
    return None


def _write_private(
    out: Path, device_dir: Path, name: str, token: str, host: str, port: int
) -> None:
    """Write or update `secrets.yaml` and copy the CA certificate into the private directory.

    The OTA password is made once and kept by later runs, so a rotation never locks the
    device out of an over-the-air update.
    """
    secrets_path = out / SECRETS_FILE
    try:
        out.mkdir(parents=True, exist_ok=True)
        try:
            existing = secrets_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            existing = None
        values = {
            "mist_device_id": f'"{name}"',
            "mist_device_token": f'"{token}"',
            "mist_server_host": f'"{host}"',
            "mist_server_port": str(port),
        }
        if _existing_value(existing, "mist_ota_password") is None:
            values["mist_ota_password"] = f'"{secrets.token_hex(OTA_PASSWORD_BYTES)}"'
        _write_atomic(secrets_path, _merge_secrets(existing, values).encode("utf-8"))
        _write_atomic(out / DEVICE_CA_FILE, (device_dir / TLS_DIR / CA_CERT).read_bytes())
    except OSError as exc:
        raise FailedError(
            f"writing {out} stopped at {exc}; the registry is unchanged, so the device is not "
            "registered"
        ) from exc


# --- Commands --------------------------------------------------------------------------------


def cmd_init(args: argparse.Namespace, ctx: Context) -> None:
    """Make the device CA, the server certificate for the PC's LAN address and an empty registry."""
    device_dir = _device_dir(args, ctx)
    ca_dir = _ca_dir(args, ctx)
    ip = _check_ip(args.ip)
    _check_ca_dir(ca_dir, device_dir)
    _refuse_in_work_tree(device_dir, "devices dir")
    for existing in (
        _registry_path(device_dir),
        device_dir / TLS_DIR / CA_CERT,
        ca_dir / CA_KEY,
    ):
        if existing.exists():
            raise RefusedError(
                f"{existing} already exists; init makes a new CA, which would cut off every "
                "provisioned device. Use cert to reissue the server certificate"
            )
    openssl = _find_openssl(ctx)

    try:
        (device_dir / TLS_DIR).mkdir(parents=True, exist_ok=True)
        ca_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise FailedError(f"cannot create {device_dir} or {ca_dir}: {exc}") from exc
    ca_key = ca_dir / CA_KEY
    try:
        _openssl(ctx, openssl, "make the CA key", *EC_KEY_ARGS, "-out", str(ca_key))
        _openssl(
            ctx,
            openssl,
            "make the CA certificate",
            "req",
            "-x509",
            "-new",
            "-key",
            str(ca_key),
            "-sha256",
            "-days",
            CERT_DAYS,
            "-subj",
            CA_SUBJECT,
            "-addext",
            "basicConstraints=critical,CA:TRUE",
            "-addext",
            "keyUsage=critical,keyCertSign,cRLSign",
            "-out",
            str(device_dir / TLS_DIR / CA_CERT),
        )
        _issue_server_cert(ctx, openssl, device_dir, ca_dir, ip)
        _write_registry(device_dir, [])
    except (FailedError, OSError) as exc:
        raise FailedError(
            f"init stopped: {exc}. Delete {device_dir} and {ca_dir} and run init again"
        ) from exc

    print(f"initialized the devices dir {device_dir} (mount it read-only via MIST_DEVICE_DIR)")
    print(f"the CA key is in {ca_dir}; keep it out of every mount and backup")
    print(f"the server certificate names {ip}")


def cmd_add(args: argparse.Namespace, ctx: Context) -> None:
    """Register a device and write its token, server address and CA into its private directory."""
    _check_name(args.name)
    area = _check_area(args.area)
    device_dir = _device_dir(args, ctx)
    ca_dir = _ca_dir(args, ctx) if args.ca_dir or ctx.environ.get("MIST_DAILY_ROOT") else None
    out = _resolve(args.out)
    _check_private_dir(out, device_dir, ca_dir)
    entries = _read_registry(device_dir)
    if _find(entries, args.name) is not None:
        raise RefusedError(
            f"{args.name} is already registered; rotate gives it a new token, and a revoked "
            "name stays revoked"
        )
    openssl = _find_openssl(ctx)
    host = _server_host(ctx, openssl, device_dir)

    token, digest = _new_token(args.name)
    _write_private(out, device_dir, args.name, token, host, args.port)
    entries.append(
        {
            "name": args.name,
            "area": area,
            "token_sha256": digest,
            "revoked": False,
            "created": _format_utc(ctx.clock()),
        }
    )
    try:
        _write_registry(device_dir, entries)
    except OSError as exc:
        raise FailedError(
            f"wrote {out / SECRETS_FILE}, but the registry write stopped at {exc}: {args.name} "
            "is not registered; run add again"
        ) from exc
    print(f"added {args.name} in area {area}")
    print(f"wrote {out / SECRETS_FILE} and {out / DEVICE_CA_FILE}; add the Wi-Fi lines there")


def cmd_list(args: argparse.Namespace, ctx: Context) -> None:
    """Print each device's name, area, state and creation time; never its hash."""
    entries = _read_registry(_device_dir(args, ctx))
    if not entries:
        print("no devices")
    for entry in entries:
        state = "revoked" if entry.get("revoked") else "active"
        print(
            f"{entry.get('name')}  area {entry.get('area')}  {state}  created {entry.get('created')}"
        )


def cmd_revoke(args: argparse.Namespace, ctx: Context) -> None:
    """Mark a device revoked; the backend refuses it from its next registry read."""
    device_dir = _device_dir(args, ctx)
    entries = _read_registry(device_dir)
    entry = _find(entries, args.name)
    if entry is None:
        raise RefusedError(f"{args.name} is not registered")
    if entry.get("revoked") is True:
        print(f"{args.name} was already revoked")
        return
    entry["revoked"] = True
    try:
        _write_registry(device_dir, entries)
    except OSError as exc:
        raise FailedError(f"cannot write the registry: {exc}; {args.name} is not revoked") from exc
    print(f"revoked {args.name}")


def cmd_rotate(args: argparse.Namespace, ctx: Context) -> None:
    """Give a device a new token; the backend refuses the old one from its next registry read.

    That holds for new connections only. The backend closes a live session only when its device
    is revoked or removed, so a session already open on the old token stays open until it ends.
    """
    device_dir = _device_dir(args, ctx)
    ca_dir = _ca_dir(args, ctx) if args.ca_dir or ctx.environ.get("MIST_DAILY_ROOT") else None
    out = _resolve(args.out)
    _check_private_dir(out, device_dir, ca_dir)
    entries = _read_registry(device_dir)
    entry = _find(entries, args.name)
    if entry is None:
        raise RefusedError(f"{args.name} is not registered")
    if entry.get("revoked") is True:
        raise RefusedError(f"{args.name} is revoked; a revoked name stays revoked")
    openssl = _find_openssl(ctx)
    host = _server_host(ctx, openssl, device_dir)

    token, digest = _new_token(args.name)
    _write_private(out, device_dir, args.name, token, host, args.port)
    entry["token_sha256"] = digest
    try:
        _write_registry(device_dir, entries)
    except OSError as exc:
        raise FailedError(
            f"wrote the new token to {out / SECRETS_FILE}, but the registry write stopped at "
            f"{exc}: the registry still holds the old one; run rotate again"
        ) from exc
    print(f"rotated {args.name}'s token; reflash the device from {out / SECRETS_FILE}")


def cmd_cert(args: argparse.Namespace, ctx: Context) -> None:
    """Reissue the server certificate, for a new LAN address, from the existing CA."""
    device_dir = _device_dir(args, ctx)
    ca_dir = _ca_dir(args, ctx)
    ip = _check_ip(args.ip)
    _check_ca_dir(ca_dir, device_dir)
    for needed in (ca_dir / CA_KEY, device_dir / TLS_DIR / CA_CERT):
        if not needed.is_file():
            raise RefusedError(f"{needed} does not exist; run init first")
    openssl = _find_openssl(ctx)
    _issue_server_cert(ctx, openssl, device_dir, ca_dir, ip)
    print(f"reissued {device_dir / TLS_DIR / SERVER_CERT} for {ip}")
    print("the backend reads it when its device listener starts; devices keep their mist_ca.pem")


# --- Entry point -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The command-line parser; each subcommand names its handler in `handler`."""
    parser = argparse.ArgumentParser(
        prog="devices.py", description="Provision the room devices that may reach MIST."
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    def with_dirs(sub: argparse.ArgumentParser, *, ca: bool) -> None:
        sub.add_argument("--dir", help="devices dir; MIST_DEVICE_DIR or <MIST_DAILY_ROOT>/devices")
        if ca:
            sub.add_argument("--ca-dir", help="CA dir; <MIST_DAILY_ROOT>/devices-ca")

    init = commands.add_parser("init", help="make the CA, the server certificate and a registry")
    init.add_argument("--ip", required=True, help="the PC's LAN address the devices connect to")
    with_dirs(init, ca=True)
    init.set_defaults(handler=cmd_init)

    add = commands.add_parser("add", help="register a device and write its private secrets")
    add.add_argument("name")
    add.add_argument("--area", required=True, help="the area its lights commands act on")
    add.add_argument("--out", required=True, help="private directory outside every work tree")
    add.add_argument("--port", type=int, default=DEFAULT_PORT, help="the LAN port it connects to")
    with_dirs(add, ca=True)
    add.set_defaults(handler=cmd_add)

    listing = commands.add_parser("list", help="list the registered devices")
    with_dirs(listing, ca=False)
    listing.set_defaults(handler=cmd_list)

    revoke = commands.add_parser("revoke", help="refuse a device from now on")
    revoke.add_argument("name")
    with_dirs(revoke, ca=False)
    revoke.set_defaults(handler=cmd_revoke)

    rotate = commands.add_parser("rotate", help="give a device a new token")
    rotate.add_argument("name")
    rotate.add_argument("--out", required=True, help="the device's private directory")
    rotate.add_argument("--port", type=int, default=DEFAULT_PORT)
    with_dirs(rotate, ca=True)
    rotate.set_defaults(handler=cmd_rotate)

    cert = commands.add_parser("cert", help="reissue the server certificate for a new address")
    cert.add_argument("--ip", required=True)
    with_dirs(cert, ca=True)
    cert.set_defaults(handler=cmd_cert)
    return parser


def main(argv: Sequence[str] | None = None, ctx: Context | None = None) -> int:
    """Run one command and return its exit code (argument errors exit 2 through argparse)."""
    args = build_parser().parse_args(argv)
    try:
        args.handler(args, ctx if ctx is not None else Context())
    except DevicesError as exc:
        print(f"devices.py {args.command}: {exc.label}: {exc}", file=sys.stderr)
        return exc.exit_code
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
