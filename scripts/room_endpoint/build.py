# SPDX-License-Identifier: MIT
# Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.
"""Compile the room endpoint's firmware offline, in a throwaway build directory.

    python -m scripts.room_endpoint.build compile spike|room [--config <path>] [--out <dir>]

It runs inside the mist.ai worker image, which holds ESPHome, ESP-IDF and every external asset
under `/opt/mist-firmware-deps/`, so a compile needs no network. Each run makes a new directory
under `/tmp`, copies `firmware/room-endpoint/` into it as `room-endpoint/`, points ESPHome's data
directory (`ESPHOME_DATA_DIR`) and `HOME` inside it, and runs `esphome compile` there. Nothing is
written in the work tree, so a compile also runs from a read-only mount, and the directory is
removed when the run ends.

Without `--config` it builds the repository's `spike.yaml` or `room.yaml` with the image's paths
as substitutions and a `secrets.yaml` and `mist_ca.pem` it generates in the build directory with
throwaway values (a stand-in Wi-Fi network, token and CA), so a worker's compile check never needs
a secret. The spike build's stand-in wake is the stock `stop` model itself. The room build cannot
take that: `mist_prewake` refuses a room build in which another model shares the name model's
word. So the room compile check wakes on a manifest written into the build directory, the stop
manifest with its word changed to "compile check" and its `model` naming the image's stop model
file. That image still opens on "stop", so `--out` is refused for it: a room image leaves the
container only when built from a private YAML with `--config`.

With `--config <path>` it builds a private YAML made from `personal.example.yaml`: the file's
directory (without its `images/`) is copied in as `config/`, beside `room-endpoint/`, so the
YAML's `!include ../room-endpoint/<file>.yaml` finds the repository's config, and its own
`secrets.yaml` and `mist_ca.pem` are the ones read.

`--out <dir>` copies the factory and OTA images out of the build directory into `<dir>` and prints
their paths there; without it, their paths in the build directory are printed and they are
discarded with it.

`voice_kit` takes the XMOS front end's firmware only by URL, so for the compile's length the
image's dependency directory is served on `127.0.0.1:8765`, which only this container reaches.

Exit codes: 0 built; 1 the compile failed or produced no image; 2 refused (no such config, or
`--out` for the room compile check).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import shutil
import socket
import subprocess  # nosec B404 - fixed argument lists, never a shell
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from io import TextIOBase
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIRMWARE_DIR = REPO_ROOT / "firmware" / "room-endpoint"
DEPS_DIR = Path("/opt/mist-firmware-deps")
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8765
SERVER_PYTHON = "/opt/python312/bin/python3.12"
SERVER_READY_TIMEOUT_S = 10.0
TARGETS = ("spike", "room")

# Every external asset the configs name, at its path in the worker image. `room.yaml` and
# `spike.yaml` carry the same values as their defaults; a private YAML may override them.
STAND_INS: dict[str, str] = {
    "voice_kit_path": str(DEPS_DIR / "home-assistant-voice-pe" / "esphome" / "components"),
    "xmos_firmware": f"http://{SERVER_HOST}:{SERVER_PORT}/xmos/ffva_v1.3.1_upgrade.bin",
    "stop_model": str(DEPS_DIR / "models" / "stop" / "stop.json"),
    "vad_model": str(DEPS_DIR / "models" / "vad" / "vad.json"),
    "wake_model": str(DEPS_DIR / "models" / "stop" / "stop.json"),
    "websocket_client_path": str(DEPS_DIR / "esp_websocket_client"),
}

TREE_NAME = "room-endpoint"
PRIVATE_NAME = "config"
COMPONENTS_SUBSTITUTION = "mist_components_path"
SECRETS_FILE = "secrets.yaml"
CA_FILE = "mist_ca.pem"
FACTORY_IMAGE = "firmware.factory.bin"
OTA_IMAGE = "firmware.ota.bin"
COPY_IGNORE = shutil.ignore_patterns(".esphome", "__pycache__", "images")

STAND_IN_DEVICE = "stand-in"
STAND_IN_SECRET_BYTES = 32

# The room compile check's name model: the stop model under a word no other model carries.
ROOM_TARGET = "room"
STAND_IN_WAKE_FILE = "stand-in-wake.json"
STAND_IN_WAKE_WORD = "compile check"
KEY_WAKE_WORD = "wake_word"
KEY_MODEL = "model"


class BuildError(Exception):
    """A step of the build failed; the message says which."""


Compile = Callable[[Sequence[str], Mapping[str, str], Path], int]
Serve = Callable[[Sequence[str]], AbstractContextManager[None]]
MakeCa = Callable[[Path, Path], None]


@dataclass(frozen=True, slots=True)
class Toolchain:
    """What a build calls outside Python; the tests pass fakes."""

    compile: Compile
    serve: Serve
    make_ca: MakeCa
    tmp_root: Path = Path("/tmp")  # nosec B108 - the container's own throwaway directory
    stop_manifest: Path = Path(STAND_INS["stop_model"])


def server_python() -> str:
    """The Python that serves the dependency directory: `python3.12`, else the image's copy."""
    return shutil.which("python3.12") or SERVER_PYTHON


def server_argv(python: str) -> list[str]:
    """Serve `/opt/mist-firmware-deps` on the loopback address only, isolated from the work tree."""
    return [
        python,
        "-I",
        "-m",
        "http.server",
        str(SERVER_PORT),
        "--bind",
        SERVER_HOST,
        "--directory",
        str(DEPS_DIR),
    ]


def compile_argv(config: Path, substitutions: Mapping[str, str]) -> list[str]:
    """`esphome` with each substitution as `-s <key> <value>`, then `compile <config>`."""
    argv = ["esphome"]
    for key, value in substitutions.items():
        argv += ["-s", key, value]
    return [*argv, "compile", str(config)]


def build_env(build_dir: Path, base: Mapping[str, str]) -> dict[str, str]:
    """`base` with ESPHome's data directory and `HOME` moved into the build directory."""
    return {**base, "HOME": str(build_dir / "home"), "ESPHOME_DATA_DIR": str(build_dir / "data")}


def make_stand_in_ca(pem: Path, work_dir: Path) -> None:
    """Write a throwaway self-signed CA certificate to `pem`; its key stays in `work_dir`."""
    argv = [
        "openssl",
        "req",
        "-x509",
        "-newkey",
        "ec",
        "-pkeyopt",
        "ec_paramgen_curve:prime256v1",
        "-nodes",
        "-days",
        "1",
        "-subj",
        "/CN=MIST build stand-in CA",
        "-keyout",
        str(work_dir / "stand-in-ca.key"),
        "-out",
        str(pem),
    ]
    result = subprocess.run(argv, capture_output=True, text=True, check=False)  # nosec B603
    if result.returncode != 0:
        raise BuildError(f"openssl could not make the stand-in CA: {result.stderr.strip()}")


def write_stand_in_secrets(directory: Path, make_ca: MakeCa, work_dir: Path) -> None:
    """Write `secrets.yaml` and `mist_ca.pem` with throwaway values, as `devices.py add` would.

    The keys are those `scripts/daily/devices.py` writes plus the two Wi-Fi lines Raj adds; the
    values only have to pass ESPHome's validation, and nothing here is a real secret.
    """
    secret = base64.urlsafe_b64encode(secrets.token_bytes(STAND_IN_SECRET_BYTES))
    token = f"{STAND_IN_DEVICE}.{secret.decode('ascii').rstrip('=')}"
    lines = [
        "# Throwaway values written by scripts/room_endpoint/build.py for a compile check.",
        'wifi_ssid: "mist-build-stand-in"',
        'wifi_password: "mist-build-stand-in"',
        f'mist_device_id: "{STAND_IN_DEVICE}"',
        f'mist_device_token: "{token}"',
        f'mist_server_host: "{SERVER_HOST}"',
        "mist_server_port: 8443",
        f'mist_ota_password: "{secrets.token_hex(16)}"',
    ]
    (directory / SECRETS_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    make_ca(directory / CA_FILE, work_dir)


def write_stand_in_wake(directory: Path, stop_manifest: Path) -> Path:
    """Write the room compile check's name model manifest into `directory` and return its path.

    It is `stop_manifest` with only its word changed, so `mist_prewake`'s check that no other
    model shares the name model's word passes, and its `model` is the stop model file's absolute
    path: ESPHome 2026.10.2 loads `manifest_path.parent / manifest["model"]`
    (`micro_wake_word/__init__.py:452`), and an absolute right operand replaces the parent.
    """
    stop = stop_manifest.resolve()
    manifest = json.loads(stop.read_text(encoding="utf-8"))
    manifest[KEY_WAKE_WORD] = STAND_IN_WAKE_WORD
    manifest[KEY_MODEL] = str(stop.parent / manifest[KEY_MODEL])
    path = directory / STAND_IN_WAKE_FILE
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def refuse_stand_in_out(target: str, config: Path | None, out: Path | None) -> None:
    """Keep the room compile check's image in the build directory.

    Its stand-in name model is the stop model, so that image would open the uplink on "stop".
    """
    if target == ROOM_TARGET and config is None and out is not None:
        raise BuildError(
            "--out is refused for a room build without --config: its stand-in wake is the stop "
            'model, so the image would open the uplink on "stop"; build the room image from '
            "the private YAML with --config"
        )


def _run_compile(argv: Sequence[str], env: Mapping[str, str], cwd: Path) -> int:
    return subprocess.run(list(argv), env=dict(env), cwd=cwd, check=False).returncode  # nosec


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


@contextmanager
def _serve(argv: Sequence[str]) -> Iterator[None]:
    server = subprocess.Popen(  # nosec B603
        list(argv), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        deadline = time.monotonic() + SERVER_READY_TIMEOUT_S
        while not _port_open(SERVER_HOST, SERVER_PORT):
            if server.poll() is not None or time.monotonic() > deadline:
                raise BuildError(
                    f"the dependency server did not start on {SERVER_HOST}:{SERVER_PORT}"
                )
            time.sleep(0.1)
        yield
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


def default_toolchain() -> Toolchain:
    return Toolchain(compile=_run_compile, serve=_serve, make_ca=make_stand_in_ca)


def _find_image(build_dir: Path, name: str) -> Path:
    found = sorted((build_dir / "data" / "build").glob(f"*/build/{name}"))
    if len(found) != 1:
        raise BuildError(f"the compile left {len(found)} copies of {name}, not one")
    return found[0]


def _prepare(build_dir: Path, target: str, config: Path | None, toolchain: Toolchain) -> list[str]:
    tree = build_dir / TREE_NAME
    shutil.copytree(FIRMWARE_DIR, tree, ignore=COPY_IGNORE)
    (build_dir / "home").mkdir()
    components = {COMPONENTS_SUBSTITUTION: str(tree / "components")}
    if config is None:
        write_stand_in_secrets(tree, toolchain.make_ca, build_dir)
        substitutions = {**STAND_INS, **components}
        if target == ROOM_TARGET:
            wake = write_stand_in_wake(build_dir, toolchain.stop_manifest)
            substitutions["wake_model"] = str(wake)
        return compile_argv(tree / f"{target}.yaml", substitutions)
    private = build_dir / PRIVATE_NAME
    shutil.copytree(config.parent, private, ignore=COPY_IGNORE)
    return compile_argv(private / config.name, components)


def run(
    target: str,
    config: Path | None,
    out: Path | None,
    toolchain: Toolchain,
    stdout: TextIOBase,
) -> int:
    """Build one image set; see the module docstring for the steps and exit codes."""
    if config is not None and not config.is_file():
        print(f"refused: {config} is not a file", file=stdout)
        return 2
    try:
        refuse_stand_in_out(target, config, out)
    except BuildError as exc:
        print(f"refused: {exc}", file=stdout)
        return 2
    build_dir = Path(tempfile.mkdtemp(prefix="mist-room-endpoint-", dir=toolchain.tmp_root))
    try:
        argv = _prepare(build_dir, target, config, toolchain)
        with toolchain.serve(server_argv(server_python())):
            rc = toolchain.compile(argv, build_env(build_dir, os.environ), build_dir)
        if rc != 0:
            print(f"compile failed (esphome exit {rc})", file=stdout)
            return 1
        images = {
            "factory": _find_image(build_dir, FACTORY_IMAGE),
            "ota": _find_image(build_dir, OTA_IMAGE),
        }
        if out is None:
            for label, path in images.items():
                print(f"{label} image: {path} (discarded with the build directory)", file=stdout)
            return 0
        out.mkdir(parents=True, exist_ok=True)
        for label, path in images.items():
            copied = out / path.name
            shutil.copyfile(path, copied)
            print(f"{label} image: {copied}", file=stdout)
        return 0
    except (BuildError, OSError) as exc:
        print(f"build failed: {exc}", file=stdout)
        return 1
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.room_endpoint.build")
    commands = parser.add_subparsers(dest="command", required=True)
    compile_cmd = commands.add_parser("compile", help="compile the spike or room firmware")
    compile_cmd.add_argument("target", choices=TARGETS)
    compile_cmd.add_argument(
        "--config", type=Path, help="a private YAML to build instead of the repository's"
    )
    compile_cmd.add_argument(
        "--out", type=Path, help="copy the factory and OTA images into this directory"
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    toolchain: Toolchain | None = None,
    stdout: TextIOBase | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    return run(
        args.target,
        None if args.config is None else args.config.resolve(),
        args.out,
        toolchain or default_toolchain(),
        stdout or sys.stdout,
    )


if __name__ == "__main__":
    sys.exit(main())
