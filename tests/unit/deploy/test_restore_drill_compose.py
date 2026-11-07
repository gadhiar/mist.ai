"""Contract of the restore drill project, `deploy/daily/restore-drill.yml` (plan v1 item 8).

Hermetic: parses the committed YAML and the env example only; no docker, no network. The drill
restores an artifact into a Neo4j on a fresh labelled volume and into a standby directory, so
each test guards one way an edit could let the drill reach the live instance: a published port,
a mount of the instance root, a writable backup root, a graph URI outside the dev allowlist, or
a volume compose would create and own instead of the one `instance.py drill-start` labelled.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from backend.knowledge.eval_isolation import assert_neo4j_dev_isolated

REPO_ROOT = Path(__file__).resolve().parents[3]
DAILY_DIR = REPO_ROOT / "deploy" / "daily"
DRILL_PATH = DAILY_DIR / "restore-drill.yml"
DAILY_COMPOSE_PATH = DAILY_DIR / "compose.yml"
ENV_EXAMPLE_PATH = DAILY_DIR / "daily.env.example"

NEO4J = "mist-neo4j-dev"
RESTORE = "mist-restore"
DRILL_VOLUME = "drill-neo4j"
DRILL_URI = "bolt://mist-neo4j-dev:7687"
RELEASE_IMAGE_PREFIX = "mist-backend:${MIST_RELEASE_TAG:?"
# Set by `instance.py` in the subprocess environment only, never in daily.env (item 8).
DRILL_ONLY_VARIABLES = {"MIST_DRILL_NEO4J_VOLUME", "MIST_DRILL_ROOT"}
HEAP_AND_CACHE = (
    "NEO4J_server_memory_heap_initial__size",
    "NEO4J_server_memory_heap_max__size",
    "NEO4J_server_memory_pagecache_size",
)
INTERPOLATION = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)")
EXAMPLE_ASSIGNMENT = re.compile(r"^#?\s*([A-Z][A-Z0-9_]*)=")


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def drill() -> dict:
    return _load(DRILL_PATH)


@pytest.fixture(scope="module")
def daily() -> dict:
    return _load(DAILY_COMPOSE_PATH)


def _environment(service: dict) -> dict[str, str]:
    raw = service.get("environment") or {}
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    pairs = (str(entry).partition("=") for entry in raw)
    return {key: value for key, _sep, value in pairs}


def _split_mount(spec: str) -> list[str]:
    """Split a short-syntax volume on the colons outside `${...}`."""
    parts, current, depth = [], [], 0
    for char in spec:
        if char == "{" and current and current[-1] == "$":
            depth += 1
        elif char == "}" and depth:
            depth -= 1
        if char == ":" and not depth:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    parts.append("".join(current))
    return parts


def _mounts(service: dict) -> list[tuple[str, str, str]]:
    """(source, target, mode) for each volume; mode is "rw" when none is given."""
    mounts = []
    for spec in service.get("volumes", []):
        assert isinstance(spec, str), f"long-syntax volume cannot be checked: {spec!r}"
        source, target, *mode = _split_mount(spec)
        mounts.append((source, target, mode[0] if mode else "rw"))
    return mounts


def _text_without_comments(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#"))


class TestProject:
    def test_has_no_top_level_name_so_the_cli_names_the_project(self, drill):
        assert "name" not in drill

    def test_services_are_exactly_the_drill_neo4j_and_the_restore(self, drill):
        assert set(drill["services"]) == {NEO4J, RESTORE}

    @pytest.mark.parametrize("key", ["container_name", "build", "ports"])
    def test_no_service_sets(self, drill, key):
        offenders = [name for name, service in drill["services"].items() if key in service]
        assert not offenders, f"{key} set on {offenders}"

    def test_mounts_nothing_from_the_instance_root(self):
        assert "${MIST_DAILY_ROOT" not in DRILL_PATH.read_text(encoding="utf-8")

    def test_never_names_the_instance_graph_volume(self):
        text = _text_without_comments(DRILL_PATH)
        assert "MIST_DAILY_NEO4J_VOLUME" not in text
        assert "neo4j-data" not in text

    def test_every_variable_is_in_daily_env_or_set_by_the_cli(self):
        example = {
            match.group(1)
            for line in ENV_EXAMPLE_PATH.read_text(encoding="utf-8").splitlines()
            if (match := EXAMPLE_ASSIGNMENT.match(line.strip()))
        }
        used = set(INTERPOLATION.findall(_text_without_comments(DRILL_PATH)))
        assert used - example == DRILL_ONLY_VARIABLES

    def test_the_drill_only_variables_are_not_in_daily_env(self):
        text = ENV_EXAMPLE_PATH.read_text(encoding="utf-8")
        for name in DRILL_ONLY_VARIABLES:
            assert name not in text, name


class TestDrillNeo4j:
    def test_uses_the_instance_pinned_image(self, drill, daily):
        image = drill["services"][NEO4J]["image"]
        assert image == daily["services"]["mist-neo4j"]["image"]
        assert "@sha256:" in image

    def test_publishes_no_port(self, drill):
        assert "ports" not in drill["services"][NEO4J]

    def test_takes_the_instance_password_from_the_env_file(self, drill):
        auth = _environment(drill["services"][NEO4J])["NEO4J_AUTH"]
        assert auth.startswith("${NEO4J_USERNAME:-neo4j}/${NEO4J_PASSWORD:?")

    @pytest.mark.parametrize("name", HEAP_AND_CACHE)
    def test_heap_and_page_cache_match_the_instance(self, drill, daily, name):
        drill_env = _environment(drill["services"][NEO4J])
        daily_env = _environment(daily["services"]["mist-neo4j"])
        assert drill_env[name] == daily_env[name]

    def test_memory_limit_matches_the_instance(self, drill, daily):
        # One variable and one default size both Neo4js, so a raised instance limit covers the
        # drill too.
        assert drill["services"][NEO4J]["mem_limit"] == daily["services"]["mist-neo4j"]["mem_limit"]

    def test_mounts_only_the_drill_volume_at_data(self, drill):
        assert _mounts(drill["services"][NEO4J]) == [(DRILL_VOLUME, "/data", "rw")]

    def test_the_drill_volume_is_external_and_named_by_the_cli(self, drill):
        volume = drill["volumes"][DRILL_VOLUME]
        assert volume["external"] is True
        assert volume["name"].startswith("${MIST_DRILL_NEO4J_VOLUME:?")

    def test_declares_no_other_volume(self, drill):
        assert set(drill["volumes"]) == {DRILL_VOLUME}


class TestRestoreService:
    def test_runs_only_on_request(self, drill):
        assert drill["services"][RESTORE]["profiles"] == ["run"]

    def test_runs_the_release_image_without_pulling(self, drill):
        service = drill["services"][RESTORE]
        assert service["image"].startswith(RELEASE_IMAGE_PREFIX)
        assert service["pull_policy"] == "never"
        assert service["user"] == "1000:1000"

    def test_its_entrypoint_is_python(self, drill):
        assert drill["services"][RESTORE]["entrypoint"] == ["python"]

    def test_environment_carries_the_graph_credentials(self, drill):
        env = _environment(drill["services"][RESTORE])
        assert env["NEO4J_URI"] == DRILL_URI
        assert env["NEO4J_USERNAME"] == "${NEO4J_USERNAME:-neo4j}"
        assert env["NEO4J_PASSWORD"].startswith("${NEO4J_PASSWORD:?")

    def test_its_graph_uri_passes_the_dev_allowlist(self, drill, monkeypatch):
        monkeypatch.delenv("MIST_DEV_NEO4J_HOSTS", raising=False)
        assert_neo4j_dev_isolated(_environment(drill["services"][RESTORE])["NEO4J_URI"])

    def test_mounts_the_backup_root_read_only(self, drill):
        mounts = {
            target: (source, mode) for source, target, mode in _mounts(drill["services"][RESTORE])
        }
        source, mode = mounts["/backup"]
        assert source.startswith("${MIST_BACKUP_HOST_PATH:?")
        assert mode == "ro"

    def test_the_only_writable_bind_is_the_drill_root(self, drill):
        writable = [
            (source, target)
            for source, target, mode in _mounts(drill["services"][RESTORE])
            if mode != "ro"
        ]
        assert len(writable) == 1
        source, target = writable[0]
        assert source.startswith("${MIST_DRILL_ROOT:?")
        assert target == "/drill"

    def test_waits_for_the_drill_neo4j_to_be_healthy(self, drill):
        depends = drill["services"][RESTORE]["depends_on"]
        assert depends == {NEO4J: {"condition": "service_healthy"}}
        assert "healthcheck" in drill["services"][NEO4J]
