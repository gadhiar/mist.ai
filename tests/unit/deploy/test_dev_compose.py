"""Contract of the worktree dev stack, `docker-compose.dev.yml` (plan v1 item 13).

Hermetic apart from one `sh` run: parses the committed YAML and README, and runs the seed
service's shell script against temporary directories with the seed command stubbed; no docker,
no network. The dev stack runs from any worktree, beside the daily instance and other dev
stacks, so each test guards one way an edit could make it reach the daily instance or the
working tree's state: a fixed container or project name that collides, a bind of `./data` or
`.env`, a store path outside its own volume, a graph URI outside the dev allowlist, a port
published beyond loopback, or a second external network.
"""

from __future__ import annotations

import re
import subprocess  # nosec B404 - runs the compose file's own seed script under sh, offline
from pathlib import Path

import pytest
import yaml

from backend.knowledge.eval_isolation import (
    LIVE_NEO4J_ENDPOINTS,
    _allowed_dev_endpoints,
    assert_neo4j_dev_isolated,
)
from backend.origins import DEFAULT_ALLOWED_ORIGINS

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE_PATH = REPO_ROOT / "docker-compose.dev.yml"
ROOT_COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"
DAILY_COMPOSE_PATH = REPO_ROOT / "deploy" / "daily" / "compose.yml"
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "dev-instance"
README_PATH = FIXTURE_DIR / "README.md"

NEO4J = "mist-neo4j-dev"
BACKEND = "mist-backend-dev"
SEED = "mist-dev-seed"
LLM = "mist-llm-dev"
SERVICES = {NEO4J, BACKEND, SEED, LLM}
OWN_LLM_PROFILE = "own-llm"

CODE_BINDS = ("./backend", "./src", "./dependencies", "./scripts", "./tests", "./voice_profiles")
FORBIDDEN_BINDS = ("./data", "./mist-memory", "./dev-state", "./logs", "./models", ".env")
NAMED_VOLUMES = {"dev-state", "neo4j-dev-data", "neo4j-dev-logs"}
DEV_STATE_ROOT = "/app/dev-state"
SHARE_NETWORK_NAME = "${MIST_LLM_SHARE_NETWORK:-mist-llm-share}"
BACKEND_IMAGE_PREFIX = "${MIST_BACKEND_IMAGE:?"
SEED_DIR = "/app/tests/fixtures/dev-instance/seed"
SEED_COMMAND = f"python scripts/mist_admin.py seed --seed-dir {SEED_DIR}"
FIXTURE_VAULT = "/app/tests/fixtures/dev-instance/vault"

# Item 13's variables. Anything else interpolated would be a host-shell variable that some other
# stack may also read, so one export could steer two stacks.
PLAN_VARIABLES = {
    "MIST_BACKEND_IMAGE",
    "MIST_DEV_WS_PORT",
    "MIST_DEV_BOLT_PORT",
    "MIST_DEV_NEO4J_HTTP_PORT",
    "MIST_DEV_LLM_URL",
    "MIST_LLM_SHARE_NETWORK",
    "MIST_DEV_SESSION_ORIGIN",
    "MIST_DEV_NEO4J_PASSWORD",
    "MODELS_DIR",
}
# service -> {port variable: (host default, container port)}
PUBLISHED_PORTS = {
    BACKEND: {"MIST_DEV_WS_PORT": (8002, 8001)},
    NEO4J: {"MIST_DEV_BOLT_PORT": (7690, 7687), "MIST_DEV_NEO4J_HTTP_PORT": (7477, 7474)},
    SEED: {},
    LLM: {},
}
STORE_PATH_KEYS = (
    "EVENT_STORE_DB_PATH",
    "EVENT_STORE_AUDIO_DIR",
    "VECTOR_STORE_DATA_DIR",
    "MIST_VAULT_ROOT",
    "MIST_SIDECAR_DB_PATH",
    "MIST_LOG_DIR",
    "XDG_CACHE_HOME",
)
PATH_KEY = re.compile(r"_(PATH|DIR|ROOT|HOME|FILE)$")
INTERPOLATION = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)")
DEFAULT_INTERPOLATION = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}")
PUBLISHED_PORT = re.compile(r"^127\.0\.0\.1:\$\{([A-Z][A-Z0-9_]*):-(\d+)\}:(\d+)$")


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def compose() -> dict:
    return _load(COMPOSE_PATH)


@pytest.fixture(scope="module")
def services(compose: dict) -> dict:
    return compose["services"]


def _environment(service: dict) -> dict[str, str]:
    raw = service.get("environment") or {}
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    pairs = (str(entry).partition("=") for entry in raw)
    return {key: value for key, _sep, value in pairs}


def _volume_entries(service: dict) -> list[tuple[str, str, str]]:
    """(source, target, mode) for every volume entry, short or long syntax."""
    entries = []
    for entry in service.get("volumes") or []:
        if isinstance(entry, dict):
            mode = "ro" if entry.get("read_only") else "rw"
            entries.append((str(entry["source"]), str(entry["target"]), mode))
            continue
        parts = entry.split(":")
        source, target = parts[0], parts[1]
        mode = parts[2] if len(parts) > 2 else "rw"
        entries.append((source, target, mode))
    return entries


def _binds(service: dict, named: set[str]) -> list[tuple[str, str, str]]:
    return [entry for entry in _volume_entries(service) if entry[0] not in named]


def _resolve_defaults(value: str) -> str:
    return DEFAULT_INTERPOLATION.sub(lambda match: match.group(2), value)


class TestNames:
    def test_no_top_level_name(self, compose):
        assert "name" not in compose, (
            "A top-level `name` makes every worktree the same project, so a second worktree's "
            "`up` recreates the first one's containers and shares its volumes."
        )

    def test_no_container_name(self, services):
        named = sorted(name for name, service in services.items() if "container_name" in service)
        assert not named, (
            f"{named} set container_name; a fixed name collides with the hydration overlay's "
            "containers and with a second dev stack."
        )


class TestServices:
    def test_exactly_the_four_services(self, services):
        assert set(services) == SERVICES

    def test_own_llm_is_behind_its_profile(self, services):
        assert services[LLM].get("profiles") == [OWN_LLM_PROFILE]

    def test_only_the_own_llm_service_has_a_profile(self, services):
        profiled = sorted(name for name, svc in services.items() if svc.get("profiles"))
        assert profiled == [LLM]

    def test_own_llm_publishes_no_port(self, services):
        assert not services[LLM].get("ports")

    def test_backend_and_seed_run_the_chosen_image_without_pulling(self, services):
        for name in (BACKEND, SEED):
            service = services[name]
            assert str(service["image"]).startswith(BACKEND_IMAGE_PREFIX), name
            assert service.get("pull_policy") == "never", name
            assert "build" not in service, name

    def test_backend_and_seed_run_as_the_image_user(self, services):
        for name in (BACKEND, SEED):
            assert services[name].get("user") == "1000:1000", name


class TestMounts:
    def test_named_volumes_are_the_three_and_compose_owned(self, compose):
        volumes = compose.get("volumes") or {}
        assert set(volumes) == NAMED_VOLUMES
        for name, spec in volumes.items():
            assert not (spec or {}).get("external"), f"{name} must belong to the project"

    def test_code_binds_are_read_only_and_from_the_six_directories(self, compose, services):
        named = set(compose.get("volumes") or {})
        for name in (BACKEND, SEED):
            binds = _binds(services[name], named)
            assert {source for source, _target, _mode in binds} == set(CODE_BINDS), name
            for source, target, mode in binds:
                assert mode == "ro", f"{name}: {source} is mounted {mode!r}"
                assert target == "/app/" + source.removeprefix("./"), f"{name}: {source}"

    def test_no_service_binds_state_or_the_env_file(self, compose, services):
        named = set(compose.get("volumes") or {})
        for name, service in services.items():
            for source, _target, _mode in _binds(service, named):
                normalised = source.rstrip("/")
                for forbidden in FORBIDDEN_BINDS:
                    assert normalised != forbidden and not normalised.startswith(
                        forbidden + "/"
                    ), f"{name} binds {source}"

    def test_every_bind_is_a_code_directory_or_the_model_directory(self, compose, services):
        named = set(compose.get("volumes") or {})
        for name, service in services.items():
            for source, _target, mode in _binds(service, named):
                if name == LLM:
                    assert source.startswith("${MODELS_DIR"), f"{name} binds {source}"
                    assert mode == "ro", f"{name}: {source} is mounted {mode!r}"
                else:
                    assert source in CODE_BINDS, f"{name} binds {source}"

    def test_dev_state_is_the_backend_and_seed_state_volume(self, services):
        for name in (BACKEND, SEED):
            entries = _volume_entries(services[name])
            assert ("dev-state", DEV_STATE_ROOT, "rw") in entries, name

    def test_neo4j_keeps_its_data_and_logs_on_its_volumes(self, services):
        entries = _volume_entries(services[NEO4J])
        assert ("neo4j-dev-data", "/data", "rw") in entries
        assert ("neo4j-dev-logs", "/logs", "rw") in entries
        assert len(entries) == 2


class TestEnvironment:
    def test_no_env_file(self, services):
        with_env_file = sorted(name for name, service in services.items() if "env_file" in service)
        assert not with_env_file, f"{with_env_file} read an env file, which can carry live paths"

    @pytest.mark.parametrize("service_name", [BACKEND, SEED])
    def test_every_store_path_lies_under_dev_state(self, services, service_name):
        env = _environment(services[service_name])
        for key in STORE_PATH_KEYS:
            assert key in env, f"{service_name} does not set {key}"
        for key, value in env.items():
            if key in STORE_PATH_KEYS or (PATH_KEY.search(key) and value.startswith("/")):
                assert value.startswith(DEV_STATE_ROOT + "/"), f"{service_name}: {key}={value}"

    @pytest.mark.parametrize("service_name", [BACKEND, SEED])
    def test_neo4j_uri_is_a_dev_endpoint(self, services, service_name, monkeypatch):
        monkeypatch.delenv("MIST_DEV_NEO4J_HOSTS", raising=False)
        uri = _environment(services[service_name])["NEO4J_URI"]
        match = re.fullmatch(r"bolt://([a-z0-9-]+):(\d+)", uri)
        assert match, uri
        endpoint = (match.group(1), int(match.group(2)))
        assert endpoint in _allowed_dev_endpoints(), endpoint
        assert endpoint not in LIVE_NEO4J_ENDPOINTS, endpoint
        assert_neo4j_dev_isolated(uri)
        assert match.group(1) == NEO4J, "the URI must name this file's own Neo4j service"

    def test_default_host_bolt_port_is_a_dev_endpoint(self, monkeypatch):
        monkeypatch.delenv("MIST_DEV_NEO4J_HOSTS", raising=False)
        default_port = PUBLISHED_PORTS[NEO4J]["MIST_DEV_BOLT_PORT"][0]
        assert ("localhost", default_port) in _allowed_dev_endpoints()
        assert ("127.0.0.1", default_port) in _allowed_dev_endpoints()

    def test_neo4j_password_is_the_one_dev_variable(self, services):
        password = "${MIST_DEV_NEO4J_PASSWORD:-"
        auth = _environment(services[NEO4J])["NEO4J_AUTH"]
        assert auth.startswith("neo4j/" + password), auth
        for name in (BACKEND, SEED):
            env = _environment(services[name])
            assert env["NEO4J_USERNAME"] == "neo4j", name
            assert env["NEO4J_PASSWORD"] == auth.removeprefix("neo4j/"), name
        default = _resolve_defaults(auth).removeprefix("neo4j/")
        assert len(default) >= 8, "Neo4j 5 refuses a password shorter than 8 characters"

    def test_session_origin_defaults_to_test(self, services):
        env = _environment(services[BACKEND])
        assert env["MIST_SESSION_ORIGIN"] == "${MIST_DEV_SESSION_ORIGIN:-test}"

    def test_tts_is_off_by_literal(self, services):
        assert _environment(services[BACKEND])["TTS_ENABLED"] == "false"

    def test_origins_are_left_to_the_code_default(self, services):
        assert "MIST_ALLOWED_ORIGINS" not in _environment(services[BACKEND])
        assert "http://localhost:1420" in DEFAULT_ALLOWED_ORIGINS

    def test_no_daily_instance_variables(self, services):
        env = _environment(services[BACKEND])
        assert "MIST_OPS_DIR" not in env, "a dev backend must not read the gaming-mode marker"
        assert "MIST_BACKUP_STATUS_PATH" not in env

    def test_only_the_plan_variables_are_interpolated(self):
        names = set(INTERPOLATION.findall(COMPOSE_PATH.read_text(encoding="utf-8")))
        assert names == PLAN_VARIABLES


class TestPorts:
    def test_every_published_port_binds_loopback_through_its_variable(self, services):
        for name, service in services.items():
            published = {}
            for entry in service.get("ports") or []:
                match = PUBLISHED_PORT.fullmatch(str(entry))
                assert match, f"{name} publishes {entry!r}"
                published[match.group(1)] = (int(match.group(2)), int(match.group(3)))
            assert published == PUBLISHED_PORTS[name], name


class TestLlm:
    def test_backend_llm_url_defaults_to_the_shared_server(self, services):
        env = _environment(services[BACKEND])
        assert env["LLM_SERVER_URL"] == "${MIST_DEV_LLM_URL:-http://mist-llm:8080}"
        assert env["LLM_BACKEND"] == "llamacpp"

    def test_share_network_is_external_and_named_by_its_variable(self, compose):
        externals = {
            key: spec
            for key, spec in (compose.get("networks") or {}).items()
            if (spec or {}).get("external")
        }
        assert len(externals) == 1, externals
        ((_key, spec),) = externals.items()
        assert spec["name"] == SHARE_NETWORK_NAME

    def test_share_network_is_the_one_the_daily_llm_joins(self, compose):
        daily = _load(DAILY_COMPOSE_PATH)
        daily_names = {
            spec["name"] for spec in daily["networks"].values() if (spec or {}).get("external")
        }
        assert SHARE_NETWORK_NAME in daily_names

    def test_only_the_backend_joins_the_share_network(self, compose, services):
        externals = {
            key
            for key, spec in (compose.get("networks") or {}).items()
            if (spec or {}).get("external")
        }
        for name, service in services.items():
            joined = set(service.get("networks") or [])
            if name == BACKEND:
                assert joined & externals == externals, "the backend must reach the shared LLM"
                assert joined - externals <= {"default"}, joined
            else:
                assert not joined & externals, f"{name} joins the share network"

    def test_own_llm_matches_the_workstation_llm(self, services):
        """Same build, model and flags as `docker-compose.yml`'s mist-llm, defaults resolved.

        Literals here rather than the root file's `${LLM_CTX_SIZE:-...}` forms: those names are
        read from the host shell by the workstation stack too.
        """
        root = _load(ROOT_COMPOSE_PATH)["services"]["mist-llm"]
        dev = services[LLM]
        assert dev["image"] == root["image"]
        root_env = {k: _resolve_defaults(v) for k, v in _environment(root).items()}
        assert _environment(dev) == root_env
        assert [str(a) for a in dev["command"]] == [
            _resolve_defaults(str(a)) for a in root["command"]
        ]

    def test_own_llm_model_bind_creates_no_host_directory(self, services):
        """An unset MODELS_DIR must fail the profile's start, not bind a new empty directory."""
        (entry,) = services[LLM]["volumes"]
        assert entry["type"] == "bind"
        assert entry["source"].startswith("${MODELS_DIR:-")
        assert entry["target"] == "/models"
        assert entry["read_only"] is True
        assert entry["bind"]["create_host_path"] is False

    def test_own_llm_reserves_the_gpu(self, services):
        devices = services[LLM]["deploy"]["resources"]["reservations"]["devices"]
        assert devices[0]["capabilities"] == ["gpu"]


class TestSeedService:
    def test_backend_starts_after_the_seed_and_a_healthy_graph(self, services):
        depends = services[BACKEND]["depends_on"]
        assert depends[NEO4J]["condition"] == "service_healthy"
        assert depends[SEED]["condition"] == "service_completed_successfully"
        assert services[SEED]["depends_on"][NEO4J]["condition"] == "service_healthy"

    def test_seed_is_a_one_shot(self, services):
        seed = services[SEED]
        assert seed.get("restart", "no") == "no"
        assert not seed.get("ports")
        assert "healthcheck" not in seed

    def test_seed_reads_the_fixture_seed(self, services):
        script = _seed_script(services)
        assert SEED_COMMAND in script
        assert f"{FIXTURE_VAULT}/." in script


def _seed_script(services: dict) -> str:
    command = services[SEED]["command"]
    assert command[:2] == ["sh", "-c"], command
    assert len(command) == 3, command
    return command[2]


class TestSeedScript:
    """Runs the seed service's script with its paths moved under `tmp_path`.

    Compose turns `$$` into `$` before the shell sees it, so the test does the same.
    """

    @pytest.fixture
    def run(self, services, tmp_path):
        script = _seed_script(services).replace("$$", "$")
        state = tmp_path / "state"
        state.mkdir()
        log = tmp_path / "seed-calls.log"
        stub = tmp_path / "seed-stub.sh"
        stub.write_text(f'echo seeded >> "{log}"\nexit "${{STUB_EXIT:-0}}"\n', encoding="utf-8")
        script = script.replace(SEED_COMMAND, f'sh "{stub}"')
        script = script.replace(FIXTURE_VAULT, (FIXTURE_DIR / "vault").as_posix())
        script = script.replace(DEV_STATE_ROOT, state.as_posix())
        assert "/app/" not in script, "a container path was left in the script under test"

        def _run(stub_exit: int = 0) -> subprocess.CompletedProcess:
            return subprocess.run(  # nosec B603 B607 - fixed argv, test-local script
                ["sh", "-c", script],
                env={"PATH": "/usr/bin:/bin", "STUB_EXIT": str(stub_exit)},
                capture_output=True,
                text=True,
                check=False,
            )

        return _run, state / "vault", log

    def test_fresh_state_gets_the_fixture_vault_and_the_seed(self, run):
        _run, vault, log = run
        result = _run()
        assert result.returncode == 0, result.stderr
        copied = sorted(p.relative_to(vault).as_posix() for p in vault.rglob("*.md"))
        expected = sorted(
            p.relative_to(FIXTURE_DIR / "vault").as_posix()
            for p in (FIXTURE_DIR / "vault").rglob("*.md")
        )
        assert copied == expected and copied
        assert log.read_text(encoding="utf-8").split() == ["seeded"]

    def test_a_vault_with_content_is_not_overwritten(self, run):
        _run, vault, log = run
        vault.mkdir()
        (vault / "MIST.md").write_text("edited in dev\n", encoding="utf-8")
        result = _run()
        assert result.returncode == 0, result.stderr
        assert (vault / "MIST.md").read_text(encoding="utf-8") == "edited in dev\n"
        assert not (vault / "identity").exists()
        assert log.read_text(encoding="utf-8").split() == ["seeded"]

    def test_a_seed_failure_on_fresh_state_fails_the_service(self, run):
        _run, _vault, _log = run
        result = _run(stub_exit=1)
        assert result.returncode != 0

    def test_a_seed_refusal_on_used_state_lets_the_backend_start(self, run):
        """The seed guard refuses a graph holding conversation-derived data, by design.

        On a reused dev-state that is the expected outcome of every `up` after the first
        conversation, so it must not keep the backend down.
        """
        _run, vault, _log = run
        vault.mkdir()
        (vault / "MIST.md").write_text("in use\n", encoding="utf-8")
        result = _run(stub_exit=1)
        assert result.returncode == 0, result.stderr
        assert "refused" in result.stderr


class TestRunbook:
    @pytest.fixture(scope="class")
    def readme(self) -> str:
        return README_PATH.read_text(encoding="utf-8")

    def test_names_the_up_command(self, readme):
        assert "docker compose -f docker-compose.dev.yml -p <name> up -d" in readme

    def test_names_the_image_and_the_share_network(self, readme):
        assert "MIST_BACKEND_IMAGE" in readme
        assert "docker network create mist-llm-share" in readme

    def test_names_every_port_variable_and_the_profile(self, readme):
        for variable in PUBLISHED_PORTS[BACKEND] | PUBLISHED_PORTS[NEO4J]:
            assert variable in readme, variable
        assert f"--profile {OWN_LLM_PROFILE}" in readme
        assert "MIST_DEV_LLM_URL" in readme

    def test_states_the_single_slot_it_shares(self, readme):
        assert "LLAMA_ARG_N_PARALLEL=1" in readme
        root_env = _environment(_load(ROOT_COMPOSE_PATH)["services"]["mist-llm"])
        assert root_env["LLAMA_ARG_N_PARALLEL"] == "1", "the runbook's one-slot claim is stale"
        daily_env = _environment(_load(DAILY_COMPOSE_PATH)["services"]["mist-llm"])
        assert daily_env["LLAMA_ARG_N_PARALLEL"] == "1", "the runbook's one-slot claim is stale"
