"""Contract of the opt-in device overlay, `deploy/daily/compose.devices.yml`.

Hermetic: parses the committed YAML, the env example and the README; no docker, no network. The
overlay is what lets a room device reach the backend over the LAN, so each test pins one way an
edit could widen it: a port on `mist-backend`, a mount or env file on the forwarder, a bind that
compose would accept unset, or a key nobody reviewed.
"""

import re
import shlex
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DAILY_DIR = REPO_ROOT / "deploy" / "daily"
OVERLAY_PATH = DAILY_DIR / "compose.devices.yml"
COMPOSE_PATH = DAILY_DIR / "compose.yml"
ENV_EXAMPLE_PATH = DAILY_DIR / "daily.env.example"
README_PATH = DAILY_DIR / "README.md"
FORWARDER_SOURCE = REPO_ROOT / "scripts" / "daily" / "device_forward.py"

BACKEND = "mist-backend"
FORWARDER = "mist-device-forward"
DAILY_ENV_VARIABLES = ("MIST_DAILY_DEVICE_BIND", "MIST_DAILY_DEVICE_PORT", "MIST_DEVICE_DIR")
WORKSTATION_ONLY = "MIST_DEVICE_FORWARD_IMAGE"
EXAMPLE_ASSIGNMENT = re.compile(r"^(#\s*)?([A-Z][A-Z0-9_]*)=(.*)$")
INTERPOLATION = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)")


@pytest.fixture(scope="module")
def overlay() -> dict:
    return yaml.safe_load(OVERLAY_PATH.read_text(encoding="utf-8"))


def _example_assignments() -> dict[str, tuple[bool, str]]:
    found = {}
    for line in ENV_EXAMPLE_PATH.read_text(encoding="utf-8").splitlines():
        match = EXAMPLE_ASSIGNMENT.match(line.strip())
        if match:
            found[match.group(2)] = (bool(match.group(1)), match.group(3))
    return found


def test_the_overlay_holds_the_two_services_and_nothing_else(overlay):
    assert set(overlay) == {"services"}
    assert set(overlay["services"]) == {BACKEND, FORWARDER}


def test_the_backend_gains_only_the_listener_switch_and_the_read_only_mount(overlay):
    backend = overlay["services"][BACKEND]

    assert backend == {
        "environment": ["MIST_DEVICE_LISTENER=on"],
        "volumes": ["${MIST_DEVICE_DIR:?set MIST_DEVICE_DIR in daily.env}:/app/devices:ro"],
    }
    assert "ports" not in backend


def test_the_forwarder_has_exactly_the_specified_keys(overlay):
    forwarder = overlay["services"][FORWARDER]

    assert set(forwarder) == {
        "profiles",
        "image",
        "pull_policy",
        "user",
        "command",
        "ports",
        "mem_limit",
        "restart",
    }
    assert "volumes" not in forwarder and "env_file" not in forwarder
    assert "environment" not in forwarder


def test_the_forwarder_is_in_the_devices_profile_only(overlay):
    assert overlay["services"][FORWARDER]["profiles"] == ["devices"]


def test_the_forwarder_runs_the_release_image_without_pulling(overlay):
    forwarder = overlay["services"][FORWARDER]

    assert forwarder["image"] == "${MIST_DEVICE_FORWARD_IMAGE:-mist-backend:${MIST_RELEASE_TAG}}"
    assert forwarder["pull_policy"] == "never"
    assert forwarder["user"] == "1000:1000"
    assert forwarder["mem_limit"] == "128m"
    assert forwarder["restart"] == "unless-stopped"


def test_the_forwarder_publishes_one_port_on_the_required_bind(overlay):
    assert overlay["services"][FORWARDER]["ports"] == [
        "${MIST_DAILY_DEVICE_BIND:?set MIST_DAILY_DEVICE_BIND in daily.env}"
        ":${MIST_DAILY_DEVICE_PORT:-8443}:8443"
    ]


def test_the_forwarder_relays_to_the_backends_listener(overlay):
    command = overlay["services"][FORWARDER]["command"]
    argv = shlex.split(command) if isinstance(command, str) else [str(a) for a in command]

    assert argv == [
        "python",
        "-m",
        "scripts.daily.device_forward",
        "--listen",
        "0.0.0.0:8443",
        "--to",
        "mist-backend:8443",
    ]
    assert FORWARDER_SOURCE.is_file()


def test_the_base_compose_file_still_knows_nothing_of_devices():
    text = COMPOSE_PATH.read_text(encoding="utf-8")
    for word in ("MIST_DEVICE", "8443", FORWARDER, "compose.devices.yml"):
        assert word not in text, word


def test_the_env_example_offers_each_daily_env_key_commented_out():
    assignments = _example_assignments()

    for name in DAILY_ENV_VARIABLES:
        commented, value = assignments[name]
        assert commented is True, f"{name} must be commented out, so the listener stays off"
        assert re.fullmatch(r"<[a-z0-9][a-z0-9-]*>", value), f"{name}={value}"
    assert WORKSTATION_ONLY not in assignments


def test_the_env_example_names_every_variable_the_overlay_reads():
    text = "\n".join(
        line
        for line in OVERLAY_PATH.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )
    named = set(_example_assignments())

    assert set(INTERPOLATION.findall(text)) - named == {WORKSTATION_ONLY}


def test_the_readme_section_covers_the_overlay():
    text = README_PATH.read_text(encoding="utf-8")
    match = re.search(r"^## Room devices\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    assert match is not None, "README.md has no '## Room devices' section"
    section = " ".join(match.group(1).split())
    for phrase in (
        "`MIST_DAILY_DEVICE_BIND`",
        "`MIST_DAILY_DEVICE_PORT`",
        "`MIST_DEVICE_DIR`",
        "`MIST_DEVICE_FORWARD_IMAGE`",
        "compose.devices.yml",
        "--profile devices",
        "devices.py init",
        "devices.py add",
        "devices.py revoke",
        "devices.py rotate",
        "device-forward:",
        "example-room",
    ):
        assert phrase in section, phrase
