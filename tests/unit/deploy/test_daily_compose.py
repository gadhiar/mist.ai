"""Contract of the mist-daily compose project, `deploy/daily/compose.yml`.

Hermetic: parses the committed YAML and the env example only; no docker, no network. The daily
instance runs a pinned release image with its state in an instance root outside every git work
tree, so each test guards one way an edit could make it reach the working tree, publish a port
beyond loopback, lose a literal that must not depend on the env file, or drift from the
workstation stack's `docker-compose.yml` where the two must agree.

The parity tests fail on a deliberate change to either file until the other follows: that is
their purpose, not a defect.
"""

import re
import shlex
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DAILY_DIR = REPO_ROOT / "deploy" / "daily"
COMPOSE_PATH = DAILY_DIR / "compose.yml"
ENV_EXAMPLE_PATH = DAILY_DIR / "daily.env.example"
README_PATH = DAILY_DIR / "README.md"
ROOT_COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"

BACKEND = "mist-backend"
NEO4J = "mist-neo4j"
LLM = "mist-llm"
BACKUP = "mist-backup"
SERVICES = (BACKEND, NEO4J, LLM, BACKUP)
RELEASE_IMAGE_PREFIX = "mist-backend:${MIST_RELEASE_TAG:?"
BIND_SOURCE_PREFIXES = ("${MIST_DAILY_ROOT", "${MODELS_DIR:?", "${MIST_BACKUP_HOST_PATH:?")
TAURI_ORIGINS = "http://tauri.localhost,https://tauri.localhost,tauri://localhost"
SHARE_NETWORK = "llm-share"

STORE_PATH_LITERALS = {
    "EVENT_STORE_DB_PATH": "/app/data/event_store.db",
    "EVENT_STORE_AUDIO_DIR": "/app/data/audio",
    "VECTOR_STORE_DATA_DIR": "/app/data/vector_store",
    "MIST_VAULT_ROOT": "/app/mist-memory",
    "MIST_SIDECAR_DB_PATH": "/app/data/vault_sidecar.db",
}
BACKEND_LITERALS = {
    "NEO4J_URI": "bolt://mist-neo4j:7687",
    "LLM_SERVER_URL": "http://mist-llm:8080",
    "LLM_BACKEND": "llamacpp",
    "MIST_SESSION_ORIGIN": "real",
    "TTS_ENABLED": "false",
    "EMBEDDING_DEVICE": "cpu",
    **STORE_PATH_LITERALS,
    "MIST_LOG_DIR": "/app/logs",
    "MIST_OPS_DIR": "/app/ops",
    "MIST_BACKUP_STATUS_PATH": "/app/ops/backup-status.json",
}
BACKUP_ENVIRONMENT = {
    "NEO4J_URI": "bolt://mist-neo4j:7687",
    "MIST_BACKUP_ROOT": "/backup",
    "MIST_BACKUP_STATUS_PATH": "/app/ops/backup-status.json",
    "MIST_BACKUP_INTERVAL_HOURS": "${MIST_BACKUP_INTERVAL_HOURS:-24}",
    "MIST_BACKUP_RETAIN": "${MIST_BACKUP_RETAIN:-7}",
    "MIST_BACKUP_CHECK_MINUTES": "${MIST_BACKUP_CHECK_MINUTES:-10}",
    "MIST_BACKUP_RETRY_MINUTES": "${MIST_BACKUP_RETRY_MINUTES:-60}",
    "MIST_BACKUP_MIN_NODE_RATIO": "${MIST_BACKUP_MIN_NODE_RATIO:-0.5}",
    **STORE_PATH_LITERALS,
}
MEM_LIMITS = {
    BACKEND: "${MIST_DAILY_BACKEND_MEM_LIMIT:-8g}",
    NEO4J: "${MIST_DAILY_NEO4J_MEM_LIMIT:-2g}",
    LLM: "${MIST_DAILY_LLM_MEM_LIMIT:-4g}",
    BACKUP: "${MIST_DAILY_BACKUP_MEM_LIMIT:-4g}",
}
LOGGING = {"driver": "json-file", "options": {"max-size": "20m", "max-file": "5"}}

# Read by the backend from `daily.env` or by `instance.py`, not interpolated by compose.
EXAMPLE_EXTRA_VARIABLES = (
    "MIST_EXTRACTION_INFERENCE",
    "MIST_EXTRACTION_SERVICE_URL",
    "MIST_USAGE_TIMEZONE",
    "MIST_INSTANCE_NAME",
    "MODEL",
)
REHEARSAL_PORT_VARIABLES = (
    "MIST_DAILY_WS_PORT",
    "MIST_DAILY_BOLT_PORT",
    "MIST_DAILY_NEO4J_HTTP_PORT",
)
# `instance.py init` fills or rewrites these lines in place, so they must be live assignments.
LIVE_EXAMPLE_VARIABLES = ("MIST_RELEASE_TAG", "MIST_DAILY_ROOT", "MIST_INSTANCE_NAME", "MODELS_DIR")

INTERPOLATION = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)")
DEFAULT_INTERPOLATION = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}")
REQUIRED_INTERPOLATION = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):\?")
EXAMPLE_ASSIGNMENT = re.compile(r"^(#\s*)?([A-Z][A-Z0-9_]*)=(.*)$")
PLACEHOLDER = re.compile(r"^<[a-z0-9][a-z0-9-]*>$")
NAMED_VOLUME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def compose() -> dict:
    return _load(COMPOSE_PATH)


@pytest.fixture(scope="module")
def root_compose() -> dict:
    return _load(ROOT_COMPOSE_PATH)


def _environment(service: dict) -> dict[str, str]:
    raw = service.get("environment") or {}
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    pairs = (str(entry).partition("=") for entry in raw)
    return {key: value for key, _sep, value in pairs}


def _split_mount(spec: str) -> list[str]:
    """Split a short-syntax volume on the colons outside `${...}`.

    A required-variable message or a default can hold a colon, so a plain `split(":")` would
    cut the source in two.
    """
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


def _mount_at(service: dict, target: str) -> tuple[str, str, str]:
    matches = [mount for mount in _mounts(service) if mount[1] == target]
    assert len(matches) == 1, f"expected one mount at {target}, found {matches!r}"
    return matches[0]


def _compose_text() -> str:
    """The compose file without its comment lines, whose prose may show `${VAR}` examples."""
    lines = COMPOSE_PATH.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#"))


def _example_assignments() -> list[tuple[bool, str, str]]:
    """(commented, name, value) for every `NAME=value` line, live or commented out."""
    assignments = []
    for line in ENV_EXAMPLE_PATH.read_text(encoding="utf-8").splitlines():
        match = EXAMPLE_ASSIGNMENT.match(line.strip())
        if match:
            assignments.append((bool(match.group(1)), match.group(2), match.group(3)))
    return assignments


class TestServices:
    def test_project_is_named_mist_daily(self, compose):
        assert compose["name"] == "mist-daily"

    def test_services_are_exactly_the_four(self, compose):
        assert set(compose["services"]) == set(SERVICES)

    @pytest.mark.parametrize("key", ["container_name", "build"])
    def test_no_service_sets(self, compose, key):
        offenders = [name for name, service in compose["services"].items() if key in service]
        assert not offenders, f"{key} set on {offenders}: containers must stay project-scoped"

    @pytest.mark.parametrize("name", [BACKEND, BACKUP])
    def test_runs_the_release_image_without_pulling(self, compose, name):
        service = compose["services"][name]

        assert service["image"].startswith(RELEASE_IMAGE_PREFIX)
        assert service["pull_policy"] == "never"
        assert service["user"] == "1000:1000"

    def test_backup_runs_the_backend_image(self, compose):
        services = compose["services"]
        assert services[BACKUP]["image"] == services[BACKEND]["image"]

    def test_backup_runs_the_scheduler(self, compose):
        command = compose["services"][BACKUP]["command"]
        argv = shlex.split(command) if isinstance(command, str) else [str(a) for a in command]
        assert argv == ["python", "-m", "scripts.backup.scheduler"]

    def test_backend_reads_the_instance_env_file(self, compose):
        (entry,) = compose["services"][BACKEND]["env_file"]
        path = entry["path"] if isinstance(entry, dict) else entry

        assert path.startswith("${MIST_DAILY_ROOT:?") and path.endswith("}/daily.env")
        assert not isinstance(entry, dict) or entry.get("required", True) is True

    @pytest.mark.parametrize("name", SERVICES)
    def test_restarts_unless_stopped(self, compose, name):
        assert compose["services"][name]["restart"] == "unless-stopped"


class TestMounts:
    @pytest.mark.parametrize("name", SERVICES)
    def test_no_bind_mount_is_repository_relative(self, compose, name):
        for source, target, _mode in _mounts(compose["services"][name]):
            if NAMED_VOLUME.match(source):
                assert source in compose["volumes"], f"{target}: undeclared volume {source!r}"
                continue
            assert source.startswith(BIND_SOURCE_PREFIXES), f"{name} binds {source!r}"

    def test_backend_volumes(self, compose):
        assert _mounts(compose["services"][BACKEND]) == [
            ("${MIST_DAILY_ROOT}/state", "/app/data", "rw"),
            ("${MIST_DAILY_ROOT}/logs", "/app/logs", "rw"),
            ("${MIST_DAILY_ROOT}/mist-memory", "/app/mist-memory", "rw"),
            ("${MIST_DAILY_ROOT}/ops", "/app/ops", "ro"),
            ("mist-hf-cache", "/home/appuser/.cache/huggingface", "rw"),
            ("mist-torch-cache", "/home/appuser/.cache/torch", "rw"),
        ]

    def test_ops_is_read_only_in_the_backend(self, compose):
        assert _mount_at(compose["services"][BACKEND], "/app/ops")[2] == "ro"

    def test_ops_is_read_write_in_the_backup(self, compose):
        assert _mount_at(compose["services"][BACKUP], "/app/ops")[2] == "rw"

    def test_backup_volumes(self, compose):
        mounts = _mounts(compose["services"][BACKUP])

        assert mounts[:3] == [
            ("${MIST_DAILY_ROOT}/state", "/app/data", "rw"),
            ("${MIST_DAILY_ROOT}/mist-memory", "/app/mist-memory", "ro"),
            ("${MIST_DAILY_ROOT}/ops", "/app/ops", "rw"),
        ]
        ((source, target, mode),) = mounts[3:]
        assert source.startswith("${MIST_BACKUP_HOST_PATH:?") and (target, mode) == (
            "/backup",
            "rw",
        )

    def test_neo4j_volumes(self, compose):
        assert _mounts(compose["services"][NEO4J]) == [
            ("neo4j-data", "/data", "rw"),
            ("${MIST_DAILY_ROOT}/logs/neo4j", "/logs", "rw"),
        ]

    def test_llm_mounts_only_the_models_read_only(self, compose):
        ((source, target, mode),) = _mounts(compose["services"][LLM])
        assert source.startswith("${MODELS_DIR:?") and (target, mode) == ("/models", "ro")


class TestPorts:
    @pytest.mark.parametrize("name", SERVICES)
    def test_every_published_port_binds_loopback(self, compose, name):
        for port in compose["services"][name].get("ports", []):
            assert isinstance(port, str) and port.startswith("127.0.0.1:"), f"{name}: {port!r}"

    def test_backend_publishes_the_canonical_websocket_port(self, compose):
        assert compose["services"][BACKEND]["ports"] == [
            "127.0.0.1:${MIST_DAILY_WS_PORT:-8001}:8001"
        ]

    def test_neo4j_publishes_the_canonical_bolt_and_http_ports(self, compose):
        assert compose["services"][NEO4J]["ports"] == [
            "127.0.0.1:${MIST_DAILY_BOLT_PORT:-7687}:7687",
            "127.0.0.1:${MIST_DAILY_NEO4J_HTTP_PORT:-7474}:7474",
        ]

    @pytest.mark.parametrize("name", [LLM, BACKUP])
    def test_publishes_no_port(self, compose, name):
        assert not compose["services"][name].get("ports")


class TestEnvironment:
    def test_backend_literals(self, compose):
        environment = _environment(compose["services"][BACKEND])

        for key, value in BACKEND_LITERALS.items():
            assert environment.get(key) == value, f"{key}={environment.get(key)!r}"

    def test_backend_allowed_origins_default_to_the_tauri_origins(self, compose):
        environment = _environment(compose["services"][BACKEND])
        assert environment["MIST_ALLOWED_ORIGINS"] == f"${{MIST_ALLOWED_ORIGINS:-{TAURI_ORIGINS}}}"

    def test_backend_sets_nothing_beyond_the_contract(self, compose):
        environment = _environment(compose["services"][BACKEND])
        assert set(environment) == set(BACKEND_LITERALS) | {"MIST_ALLOWED_ORIGINS"}

    def test_backup_environment(self, compose):
        environment = _environment(compose["services"][BACKUP])

        for key, value in BACKUP_ENVIRONMENT.items():
            assert environment.get(key) == value, f"{key}={environment.get(key)!r}"
        assert environment["NEO4J_USERNAME"] == "${NEO4J_USERNAME:-neo4j}"
        assert environment["NEO4J_PASSWORD"].startswith("${NEO4J_PASSWORD:?")

    def test_neo4j_auth_requires_a_password(self, compose):
        auth = _environment(compose["services"][NEO4J])["NEO4J_AUTH"]
        assert auth.startswith("${NEO4J_USERNAME:-neo4j}/${NEO4J_PASSWORD:?")

    def test_neo4j_has_explicit_heap_and_page_cache(self, compose):
        environment = _environment(compose["services"][NEO4J])

        assert (
            environment["NEO4J_server_memory_heap_initial__size"] == "${MIST_DAILY_NEO4J_HEAP:-1g}"
        )
        assert environment["NEO4J_server_memory_heap_max__size"] == "${MIST_DAILY_NEO4J_HEAP:-1g}"
        assert environment["NEO4J_server_memory_pagecache_size"] == (
            "${MIST_DAILY_NEO4J_PAGECACHE:-512m}"
        )
        assert environment["NEO4J_PLUGINS"] == '["apoc"]'


class TestDependencies:
    def test_backend_depends_on_neo4j_only(self, compose):
        assert compose["services"][BACKEND]["depends_on"] == {
            NEO4J: {"condition": "service_healthy"}
        }

    def test_backup_depends_on_neo4j_only(self, compose):
        assert compose["services"][BACKUP]["depends_on"] == {
            NEO4J: {"condition": "service_healthy"}
        }

    def test_backup_is_in_the_backup_profile_only(self, compose):
        assert compose["services"][BACKUP]["profiles"] == ["backup"]

    @pytest.mark.parametrize("name", [BACKEND, NEO4J, LLM])
    def test_runs_without_a_profile(self, compose, name):
        assert "profiles" not in compose["services"][name]


class TestNetworksAndVolumes:
    def test_neo4j_data_volume_is_external_with_a_required_name(self, compose):
        volume = compose["volumes"]["neo4j-data"]

        assert volume["external"] is True
        assert volume["name"].startswith("${MIST_DAILY_NEO4J_VOLUME:?")

    def test_cache_volumes_are_project_scoped(self, compose):
        for name in ("mist-hf-cache", "mist-torch-cache"):
            assert not (compose["volumes"][name] or {}).get("external")

    def test_share_network_is_external(self, compose):
        network = compose["networks"][SHARE_NETWORK]

        assert network["external"] is True
        assert network["name"] == "${MIST_LLM_SHARE_NETWORK:-mist-llm-share}"

    def test_llm_joins_the_default_and_share_networks(self, compose):
        assert set(compose["services"][LLM]["networks"]) == {"default", SHARE_NETWORK}

    @pytest.mark.parametrize("name", [BACKEND, NEO4J, BACKUP])
    def test_does_not_join_the_share_network(self, compose, name):
        assert SHARE_NETWORK not in (compose["services"][name].get("networks") or [])


class TestResources:
    @pytest.mark.parametrize("name", SERVICES)
    def test_memory_limit_comes_from_its_variable(self, compose, name):
        assert compose["services"][name]["mem_limit"] == MEM_LIMITS[name]

    @pytest.mark.parametrize("name", SERVICES)
    def test_logs_rotate(self, compose, name):
        assert compose["services"][name]["logging"] == LOGGING

    def test_backup_reserves_no_gpu(self, compose):
        assert "deploy" not in compose["services"][BACKUP]


class TestParityWithTheWorkstationStack:
    @pytest.mark.parametrize("key", ["image", "command", "environment"])
    def test_llm_matches_docker_compose(self, compose, root_compose, key):
        assert compose["services"][LLM][key] == root_compose["services"][LLM][key]

    @pytest.mark.parametrize("name", [BACKEND, NEO4J, LLM])
    def test_healthcheck_matches_docker_compose(self, compose, root_compose, name):
        assert (
            compose["services"][name]["healthcheck"]
            == root_compose["services"][name]["healthcheck"]
        )

    @pytest.mark.parametrize("name", [BACKEND, LLM])
    def test_gpu_reservation_matches_docker_compose(self, compose, root_compose, name):
        assert compose["services"][name]["deploy"] == root_compose["services"][name]["deploy"]

    def test_neo4j_image_matches_docker_compose(self, compose, root_compose):
        assert compose["services"][NEO4J]["image"] == root_compose["services"][NEO4J]["image"]


class TestEnvExample:
    def test_names_every_interpolated_variable(self):
        interpolated = set(INTERPOLATION.findall(_compose_text()))
        named = {name for _commented, name, _value in _example_assignments()}

        assert interpolated - named == set()

    def test_names_the_variables_compose_does_not_interpolate(self):
        named = {name for _commented, name, _value in _example_assignments()}
        assert set(EXAMPLE_EXTRA_VARIABLES) - named == set()

    def test_names_each_variable_once(self):
        names = [name for _commented, name, _value in _example_assignments()]
        assert len(names) == len(set(names))

    def test_every_value_is_a_placeholder_or_the_compose_default(self):
        defaults = dict(DEFAULT_INTERPOLATION.findall(_compose_text()))

        for _commented, name, value in _example_assignments():
            assert PLACEHOLDER.match(value) or value == defaults.get(name), f"{name}={value}"

    def test_every_assignment_has_a_comment_above_it(self):
        previous = None
        for line in ENV_EXAMPLE_PATH.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            match = EXAMPLE_ASSIGNMENT.match(stripped)
            if match:
                assert (
                    previous and previous.startswith("#") and not EXAMPLE_ASSIGNMENT.match(previous)
                ), f"{match.group(2)} has no comment line above it"
            if stripped:
                previous = stripped

    def test_port_variables_are_commented_out_and_rehearsal_only(self):
        lines = ENV_EXAMPLE_PATH.read_text(encoding="utf-8").splitlines()
        by_name = {name: commented for commented, name, _value in _example_assignments()}

        for name in REHEARSAL_PORT_VARIABLES:
            assert by_name[name] is True, f"{name} must be commented out"
            index = next(
                i
                for i, line in enumerate(lines)
                if EXAMPLE_ASSIGNMENT.match(line) and f"{name}=" in line
            )
            assert "rehearsal" in lines[index - 1].lower(), f"{name}: comment must say rehearsal"

    def test_required_and_init_filled_variables_are_live(self):
        required = set(REQUIRED_INTERPOLATION.findall(_compose_text()))
        by_name = {name: commented for commented, name, _value in _example_assignments()}

        for name in required | set(LIVE_EXAMPLE_VARIABLES):
            assert by_name[name] is False, f"{name} must be a live assignment"


class TestReadme:
    def test_names_every_example_variable(self):
        text = README_PATH.read_text(encoding="utf-8")
        missing = [name for _c, name, _v in _example_assignments() if f"`{name}`" not in text]
        assert not missing

    @pytest.mark.parametrize(
        "phrase",
        [
            pytest.param("mist_admin chat", id="no-scripted-traffic"),
            pytest.param("docker stats", id="memory-estimates"),
            pytest.param("docker-compose.yml:137-138", id="workstation-llm-port"),
            pytest.param("release/<tag>/", id="instance-root-layout"),
        ],
    )
    def test_covers(self, phrase):
        assert phrase in README_PATH.read_text(encoding="utf-8")
