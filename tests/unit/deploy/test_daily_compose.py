"""Contract of the mist-daily compose project, `deploy/daily/compose.yml`.

Hermetic: parses the committed YAML and the env example only; no docker, no network. The daily
instance runs a pinned release image with its state in an instance root outside every git work
tree, so each test guards one way an edit could make it reach the working tree, publish a port
beyond loopback, lose a literal that must not depend on the env file, or drift from the
workstation stack's `docker-compose.yml` where the two must agree.

The night extraction services run main's extraction overlay, `docker-compose.extraction.yml`,
with a listed set of deltas, each asserted on its own.

The parity tests fail on a deliberate change to either file until the other follows: that is
their purpose, not a defect.
"""

import re
import shlex
from pathlib import Path

import pytest
import yaml

from backend.realtime.budget import _REMOVED_PER_SLOT
from scripts.daily import instance
from tests.unit.test_compose_extraction import (
    EXPECTED_SERVER_DECLARATIONS,
    declaration_problems,
    llama_server_settings,
    model_path_problems,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DAILY_DIR = REPO_ROOT / "deploy" / "daily"
COMPOSE_PATH = DAILY_DIR / "compose.yml"
ENV_EXAMPLE_PATH = DAILY_DIR / "daily.env.example"
README_PATH = DAILY_DIR / "README.md"
ROOT_COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"
OVERLAY_PATH = REPO_ROOT / "docker-compose.extraction.yml"

BACKEND = "mist-backend"
NEO4J = "mist-neo4j"
LLM = "mist-llm"
BACKUP = "mist-backup"
# The names `scripts/daily/instance.py` starts, stops and waits on, taken from the CLI so a
# rename on either side fails this suite rather than `extraction apply`.
EXTRACTION_LLM = instance.EXTRACTION_LLM_SERVICE
EXTRACTION = instance.EXTRACTION_SERVICE
CHAT_SERVICES = (BACKEND, NEO4J, LLM, BACKUP)
EXTRACTION_SERVICES = instance.EXTRACTION_SERVICES
EXTRACTION_PROFILE = instance.EXTRACTION_PROFILE
SERVICES = (*CHAT_SERVICES, *EXTRACTION_SERVICES)
OVERLAY_LLM = "mist-extraction-llm-local"
OVERLAY_SERVICE = "mist-extraction-local"
OVERLAY_COUNTERPART = {EXTRACTION_LLM: OVERLAY_LLM, EXTRACTION: OVERLAY_SERVICE}
# Top-level keys each daily extraction service drops from, or adds to, its overlay service. A
# key the overlay gains fails the key-set test until it is carried over or listed here.
DROPPED_OVERLAY_KEYS: dict[str, dict[str, str]] = {
    EXTRACTION_LLM: {
        "container_name": "containers stay project-scoped, so a rehearsal can run beside daily",
    },
    EXTRACTION: {
        "container_name": "containers stay project-scoped, so a rehearsal can run beside daily",
        "hostname": "the service is itself named mist-extraction, so its DNS name needs no alias",
        "ports": "nothing outside the project calls it; the backend reaches it by name",
        "build": "the release image is built by `instance.py release-build`, never by compose",
    },
}
ADDED_DAILY_KEYS: dict[str, dict[str, str]] = {
    EXTRACTION_LLM: {
        "networks": "the default network only, stated against mist-llm's share network",
        "mem_limit": "every daily service has a memory limit from daily.env",
        "logging": "every daily service rotates its logs",
    },
    EXTRACTION: {
        "image": "the release's mist-extraction:<tag>, in place of build",
        "pull_policy": "never: a pull of the tag could fetch an unrelated public image",
        "mem_limit": "every daily service has a memory limit from daily.env",
        "logging": "every daily service rotates its logs",
    },
}
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
# The backend's own capability paths. Not in STORE_PATH_LITERALS: mist-backup reads the store
# through LIVE_STORE_FILENAMES under /app/data, and must never see the token files.
# `MIST_HA_TOKEN` is pinned empty so a token pasted into daily.env is ignored: the file under
# `<root>/secrets/` is the token's one source.
CAPABILITY_LITERALS = {
    "MIST_CAPABILITIES_DB_PATH": "/app/data/capabilities.db",
    "MIST_HA_TOKEN": "",
    "MIST_HA_TOKEN_FILE": "/app/secrets/ha_token",
    "MIST_SPOTIFY_TOKEN_PATH": "/app/secrets/spotify_token.json",
}
# The backend calls the night extraction service in this project. The backlog's overdue age is a
# day, since turns wait for the night window; the request timeout holds a slow night job of
# three LLM calls at the service's per-call bound (plan v5, A15).
EXTRACTION_BACKEND_ENTRIES = {
    "MIST_EXTRACTION_INFERENCE": "service",
    "MIST_EXTRACTION_SERVICE_URL": "http://mist-extraction:8090",
    "MIST_HEALTH_BACKLOG_MAX_AGE_S": "${MIST_HEALTH_BACKLOG_MAX_AGE_S:-86400}",
    "MIST_EXTRACTION_REQUEST_TIMEOUT_S": "1800",
}
BACKEND_LITERALS = {
    "NEO4J_URI": "bolt://mist-neo4j:7687",
    "LLM_SERVER_URL": "http://mist-llm:8080",
    "LLM_BACKEND": "llamacpp",
    "MIST_SESSION_ORIGIN": "real",
    "EMBEDDING_DEVICE": "cpu",
    **STORE_PATH_LITERALS,
    "MIST_LOG_DIR": "/app/logs",
    "MIST_OPS_DIR": "/app/ops",
    "MIST_BACKUP_STATUS_PATH": "/app/ops/backup-status.json",
    **CAPABILITY_LITERALS,
    **EXTRACTION_BACKEND_ENTRIES,
}
SECRETS_SOURCE = "${MIST_DAILY_ROOT}/secrets"
# Her voice identities, under the instance root and outside every repository, and the engine
# assets built from them. Both targets are the in-container defaults of
# `backend/realtime/budget.py`'s `PlacementConfig`, so compose sets no variable for either.
VOICES_SOURCE = "${MIST_DAILY_ROOT}/voices"
VOICE_CACHE_VOLUME = "voice-cache"
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
    NEO4J: "${MIST_DAILY_NEO4J_MEM_LIMIT:-3g}",
    LLM: "${MIST_DAILY_LLM_MEM_LIMIT:-9g}",
    BACKUP: "${MIST_DAILY_BACKUP_MEM_LIMIT:-4g}",
    EXTRACTION_LLM: "${MIST_DAILY_EXTRACTION_LLM_MEM_LIMIT:-14g}",
    EXTRACTION: "${MIST_DAILY_EXTRACTION_MEM_LIMIT:-1g}",
}
LOGGING = {"driver": "json-file", "options": {"max-size": "20m", "max-file": "5"}}

# The capability rows of `daily.env.example`: commented out, so capabilities stay off until the
# operator sets them. `MIST_TIMEZONE` also dates the use counts and the uptime ledger.
CAPABILITY_EXAMPLE_VARIABLES = (
    "MIST_CAPABILITIES_ENABLED",
    "MIST_CAPABILITIES",
    "MIST_TIMEZONE",
    "MIST_HA_MCP_URL",
    "MIST_LIGHTS_DEFAULT_TARGET",
    "MIST_SPOTIFY_CLIENT_ID",
    "MIST_SPOTIFY_DEVICE",
    "MIST_RING_FALLBACK_AFTER_S",
)
# The voice rows of `daily.env.example`: commented out, so voice runs on the code's defaults
# until the operator changes one. Each is read by `load_voice_settings`.
VOICE_EXAMPLE_VARIABLES = (
    "MIST_VOICE_STT_CHAIN",
    "MIST_VOICE_TTS_CHAIN",
    "MIST_VOICE_TTS_VOICE",
    "MIST_VOICE_KWS_MODEL",
    "MIST_VOICE_KWS_THRESHOLD",
    "MIST_VOICE_VERIFY_MODEL",
    "MIST_VOICE_WAKE_PHRASES_FILE",
)
# The one example value that is a path rather than a `<placeholder>`: only `/models` is mounted
# for the phrases file, so the example shows where under it the file goes.
EXAMPLE_PATH_VALUES = {"MIST_VOICE_WAKE_PHRASES_FILE": "/models/speech/wake/phrases.json"}
VOICE_SETTINGS_PATH = REPO_ROOT / "backend" / "realtime" / "budget.py"
# Read by the backend from `daily.env` or by `instance.py`, not interpolated by compose.
EXAMPLE_EXTRA_VARIABLES = (
    "MIST_EXTRACTION_WINDOW",
    "MIST_INSTANCE_NAME",
    "MODEL",
    *CAPABILITY_EXAMPLE_VARIABLES,
    *VOICE_EXAMPLE_VARIABLES,
)
# Compose sets these on mist-backend, so a daily.env line for one would be ignored.
COMPOSE_SET_EXTRACTION_VARIABLES = (
    "MIST_EXTRACTION_INFERENCE",
    "MIST_EXTRACTION_SERVICE_URL",
    "MIST_EXTRACTION_REQUEST_TIMEOUT_S",
)

# The night serving settings `daily.env` may change, by the llama-server setting each feeds.
# Every other serving setting is a literal.
DAILY_LLM_VARIABLES = {
    "LLAMA_ARG_MODEL": "MIST_DAILY_EXTRACTION_MODEL_FILE",
    "LLAMA_ARG_CTX_SIZE": "MIST_DAILY_EXTRACTION_CTX_SIZE",
    "-ncmoe": "MIST_DAILY_EXTRACTION_NCMOE",
    "--cache-ram": "MIST_DAILY_EXTRACTION_CACHE_RAM_MIB",
}
# The extraction identity; not required, so a missing value never stops chat:
# `extraction apply` pauses extraction as unconfigured instead.
IDENTITY_VARIABLES = instance.EXTRACTION_IDENTITY_VARIABLES
# Every name the two extraction services read from daily.env, apart from the image tag and the
# models directory, which they share with the chat services.
EXTRACTION_DAILY_VARIABLES = {
    *DAILY_LLM_VARIABLES.values(),
    "MIST_DAILY_EXTRACTION_LLM_MEM_LIMIT",
    "MIST_DAILY_EXTRACTION_MEM_LIMIT",
    *IDENTITY_VARIABLES,
}
EXTRACTION_SHARED_VARIABLES = {"MIST_RELEASE_TAG", "MODELS_DIR"}
# mist-llm's cache and attention settings, which the night server adds to the overlay's.
CACHE_AND_ATTENTION_SETTINGS = (
    "LLAMA_ARG_FLASH_ATTN",
    "LLAMA_ARG_CACHE_TYPE_K",
    "LLAMA_ARG_CACHE_TYPE_V",
)
ADDED_SERVER_DECLARATIONS = {"FLASH_ATTN", "CACHE_TYPE_K", "CACHE_TYPE_V"}
# The service environment entries that differ from the overlay's defaults, each asserted by its
# own test in `TestParityWithTheExtractionOverlay`.
SERVICE_ENVIRONMENT_DELTAS = {
    "EXTRACTION_LLM_BASE_URL",
    "EXTRACTION_MODEL_HASH",
    "EXTRACTION_SERVER_DEVICE",
    "EXTRACTION_SERVER_CTX_SIZE",
    "EXTRACTION_LLM_TIMEOUT_SECONDS",
    "EXTRACTION_SERVER_N_CPU_MOE",
}
# Night context and the overlay's: 65,536 is Raj's ruling of 15 Nov (plan v4 A11, v5 M34).
NIGHT_CTX_SIZE = "65536"
OVERLAY_CTX_SIZE = "16384"
# Night ncmoe and the overlay's: at 24 the night rehearsal decoded 21.76 tokens per second,
# below its bar of 23, and at 21 it passed every bar (plan v6, T8).
NIGHT_NCMOE = "21"
OVERLAY_NCMOE = "24"
# The LLM's floor: about 11 GiB of expert tensors with all 24 layers' experts on the CPU, an
# upper bound at ncmoe 21, the prompt-cache cap, and 1 GiB for the server (plan v4, Interfaces,
# Compose). The night rehearsal measured a peak of 9.716 GiB at ncmoe 21.
EXPERT_TENSORS_ON_CPU_GIB = 11
SERVER_OVERHEAD_GIB = 1
CHAT_TEMPLATE = re.compile(r"chat[-_ ]?templates?", re.I)
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
SIZE = re.compile(r"^(\d+)([kmg]?)$")
SIZE_UNITS = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3}
MIB = 1024**2


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def compose() -> dict:
    return _load(COMPOSE_PATH)


@pytest.fixture(scope="module")
def root_compose() -> dict:
    return _load(ROOT_COMPOSE_PATH)


@pytest.fixture(scope="module")
def overlay() -> dict:
    return _load(OVERLAY_PATH)


def _resolved(value: str) -> str | None:
    """What a compose value comes to with no variable set; None when a variable is required."""
    if REQUIRED_INTERPOLATION.search(value):
        return None
    return DEFAULT_INTERPOLATION.sub(lambda match: match.group(2), value)


def _interpolated(value: object) -> set[str]:
    """Every variable name a compose value, or a whole parsed service, interpolates."""
    return set(INTERPOLATION.findall(yaml.safe_dump(value, width=1 << 20)))


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


def _default_of(value: str) -> str:
    """The default of a single `${NAME:-default}` value."""
    match = DEFAULT_INTERPOLATION.fullmatch(value)
    assert match, f"not a `${{NAME:-default}}` value: {value!r}"
    return match.group(2)


def _size_in_bytes(size: str) -> int:
    """Bytes in a compose or Neo4j size such as `3g` or `512m`, whose units are binary."""
    match = SIZE.match(size.lower())
    assert match, f"not a size: {size!r}"
    return int(match.group(1)) * SIZE_UNITS[match.group(2)]


def _compose_text() -> str:
    """The compose file without its comment lines, whose prose may show `${VAR}` examples."""
    lines = COMPOSE_PATH.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#"))


def _comment_above(name: str) -> str:
    """The comment lines directly above `name`'s line in the example, joined on one line."""
    lines = ENV_EXAMPLE_PATH.read_text(encoding="utf-8").splitlines()
    (index,) = (
        i
        for i, line in enumerate(lines)
        if (match := EXAMPLE_ASSIGNMENT.match(line.strip())) and match.group(2) == name
    )
    comment: list[str] = []
    for line in reversed(lines[:index]):
        if not line.startswith("#") or EXAMPLE_ASSIGNMENT.match(line.strip()):
            break
        comment.insert(0, line.lstrip("# "))
    return " ".join(" ".join(comment).split())


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

    def test_services_are_exactly_the_six(self, compose):
        services = compose["services"]
        in_extraction = {
            name for name in services if EXTRACTION_PROFILE in services[name].get("profiles", [])
        }

        assert set(services) == set(SERVICES)
        assert in_extraction == set(EXTRACTION_SERVICES)

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

    @pytest.mark.parametrize("name", CHAT_SERVICES)
    def test_restarts_unless_stopped(self, compose, name):
        assert compose["services"][name]["restart"] == "unless-stopped"

    @pytest.mark.parametrize("name", EXTRACTION_SERVICES)
    def test_extraction_service_has_no_env_file(self, compose, name):
        # Interpolation is the only way daily.env reaches either container.
        assert "env_file" not in compose["services"][name]


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
            (SECRETS_SOURCE, "/app/secrets", "rw"),
            ("mist-hf-cache", "/home/appuser/.cache/huggingface", "rw"),
            ("mist-torch-cache", "/home/appuser/.cache/torch", "rw"),
            ("${MODELS_DIR:?set MODELS_DIR in daily.env}", "/models", "ro"),
            (VOICES_SOURCE, "/voices", "ro"),
            (VOICE_CACHE_VOLUME, "/voice-cache", "rw"),
        ]

    def test_backend_mounts_the_models_as_mist_llm_does(self, compose):
        # The speech weights sit beside the llama-server weights; the same spec keeps one
        # variable, one required message and read-only for both.
        services = compose["services"]
        (llm_models,) = services[LLM]["volumes"]
        backend_models = [
            spec for spec in services[BACKEND]["volumes"] if spec.endswith(":/models:ro")
        ]

        assert backend_models == [llm_models]

    def test_backend_voices_are_read_only_under_the_instance_root(self, compose):
        source, _target, mode = _mount_at(compose["services"][BACKEND], "/voices")

        assert mode == "ro"
        assert _interpolated(source) == {"MIST_DAILY_ROOT"}
        assert source.startswith("${MIST_DAILY_ROOT}/")

    def test_voice_cache_is_writable_and_project_scoped(self, compose):
        source, _target, mode = _mount_at(compose["services"][BACKEND], "/voice-cache")
        declaration = compose["volumes"][VOICE_CACHE_VOLUME] or {}

        assert (source, mode) == (VOICE_CACHE_VOLUME, "rw")
        # No `name:`, so compose prefixes it with the project: each instance gets its own.
        assert "name" not in declaration
        assert not declaration.get("external")

    def test_no_other_service_mounts_the_voices(self, compose):
        offenders = [
            (name, target)
            for name, service in compose["services"].items()
            if name != BACKEND
            for _source, target, _mode in _mounts(service)
            if target in ("/voices", "/voice-cache")
        ]
        assert not offenders

    def test_no_other_service_mounts_the_secrets(self, compose):
        # The tokens stay out of every backup because mist-backup cannot see them. A mount of
        # the whole root would expose `secrets/` too.
        exposing = {SECRETS_SOURCE, "${MIST_DAILY_ROOT}", "${MIST_DAILY_ROOT}/"}
        offenders = [
            (name, source, target)
            for name, service in compose["services"].items()
            if name != BACKEND
            for source, target, _mode in _mounts(service)
            if source.startswith(SECRETS_SOURCE)
            or source in exposing
            or target.startswith("/app/secrets")
        ]
        assert not offenders

    def test_the_capability_paths_lie_under_the_backend_mounts(self, compose):
        environment = _environment(compose["services"][BACKEND])
        assert environment["MIST_CAPABILITIES_DB_PATH"].startswith(
            _mount_at(compose["services"][BACKEND], "/app/data")[1] + "/"
        )
        secrets_target = _mount_at(compose["services"][BACKEND], "/app/secrets")[1]
        for key in ("MIST_HA_TOKEN_FILE", "MIST_SPOTIFY_TOKEN_PATH"):
            assert environment[key].startswith(secrets_target + "/"), key

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

    @pytest.mark.parametrize("name", [LLM, EXTRACTION_LLM])
    def test_llm_mounts_only_the_models_read_only(self, compose, name):
        ((source, target, mode),) = _mounts(compose["services"][name])
        assert source.startswith("${MODELS_DIR:?") and (target, mode) == ("/models", "ro")

    def test_extraction_llm_mounts_the_models_as_mist_llm_does(self, compose):
        services = compose["services"]
        assert services[EXTRACTION_LLM]["volumes"] == services[LLM]["volumes"]

    def test_extraction_service_mounts_nothing(self, compose):
        assert not compose["services"][EXTRACTION].get("volumes")

    @pytest.mark.parametrize("name", EXTRACTION_SERVICES)
    def test_names_no_chat_template(self, compose, name):
        # llama-server serves the template embedded in the model file, as the overlay runs it.
        text = yaml.safe_dump(compose["services"][name], width=1 << 20)
        assert not CHAT_TEMPLATE.search(text)


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

    @pytest.mark.parametrize("name", [LLM, BACKUP, *EXTRACTION_SERVICES])
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

    def test_backend_leaves_spoken_replies_to_the_code_default(self, compose):
        # `TTS_ENABLED` defaults on in `backend/config.py`; a literal here would pin it off.
        assert "TTS_ENABLED" not in _environment(compose["services"][BACKEND])

    def test_literals_comment_says_voice_runs_on_the_code_defaults(self):
        text = " ".join(COMPOSE_PATH.read_text(encoding="utf-8").split())

        assert "turn TTS on" not in text
        assert "voice runs on the code's defaults" in text

    def test_backend_calls_the_extraction_service_on_its_health_port(self, compose, overlay):
        url = _environment(compose["services"][BACKEND])["MIST_EXTRACTION_SERVICE_URL"]
        health = overlay["services"][OVERLAY_SERVICE]["healthcheck"]["test"][-1]

        assert url == f"http://{EXTRACTION}:8090"
        assert "http://localhost:8090/" in health

    def test_extraction_reads_only_its_own_daily_variables(self, compose):
        # image and volumes take the release tag and models directory the chat services use;
        # everything else, the serving settings above all, reads only its own names.
        read = set()
        for name in EXTRACTION_SERVICES:
            service = dict(compose["services"][name])
            assert _interpolated(service.pop("image")) <= {"MIST_RELEASE_TAG"}
            assert _interpolated(service.pop("volumes", [])) <= {"MODELS_DIR"}
            read |= _interpolated(service)

        assert read == EXTRACTION_DAILY_VARIABLES

    def test_extraction_interpolates_no_overlay_variable(self, compose, overlay):
        overlay_names = set()
        for name in (OVERLAY_LLM, OVERLAY_SERVICE):
            overlay_names |= _interpolated(overlay["services"][name])
        daily_names = set()
        for name in EXTRACTION_SERVICES:
            daily_names |= _interpolated(compose["services"][name])

        shared = {*IDENTITY_VARIABLES, "MODELS_DIR"}
        assert overlay_names - shared, "the overlay interpolates nothing else to keep out"
        assert daily_names & (overlay_names - shared) == set()

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

    def test_extraction_service_waits_for_its_llm(self, compose):
        assert compose["services"][EXTRACTION]["depends_on"] == {
            EXTRACTION_LLM: {"condition": "service_healthy"}
        }

    def test_extraction_llm_depends_on_nothing(self, compose):
        assert "depends_on" not in compose["services"][EXTRACTION_LLM]


class TestNetworksAndVolumes:
    def test_neo4j_data_volume_is_external_with_a_required_name(self, compose):
        volume = compose["volumes"]["neo4j-data"]

        assert volume["external"] is True
        assert volume["name"].startswith("${MIST_DAILY_NEO4J_VOLUME:?")

    def test_cache_volumes_are_project_scoped(self, compose):
        for name in ("mist-hf-cache", "mist-torch-cache", VOICE_CACHE_VOLUME):
            assert not (compose["volumes"][name] or {}).get("external")

    def test_share_network_is_external(self, compose):
        network = compose["networks"][SHARE_NETWORK]

        assert network["external"] is True
        assert network["name"] == "${MIST_LLM_SHARE_NETWORK:-mist-llm-share}"

    def test_llm_joins_the_default_and_share_networks(self, compose):
        assert set(compose["services"][LLM]["networks"]) == {"default", SHARE_NETWORK}

    @pytest.mark.parametrize("name", [BACKEND, NEO4J, BACKUP, *EXTRACTION_SERVICES])
    def test_does_not_join_the_share_network(self, compose, name):
        assert SHARE_NETWORK not in (compose["services"][name].get("networks") or [])

    @pytest.mark.parametrize("name", EXTRACTION_SERVICES)
    def test_extraction_joins_the_default_network_only(self, compose, name):
        assert compose["services"][name].get("networks", ["default"]) == ["default"]


class TestResources:
    @pytest.mark.parametrize("name", SERVICES)
    def test_memory_limit_comes_from_its_variable(self, compose, name):
        assert compose["services"][name]["mem_limit"] == MEM_LIMITS[name]

    def test_neo4j_default_limit_is_above_its_heap_plus_page_cache(self, compose):
        service = compose["services"][NEO4J]
        environment = _environment(service)
        heap = _default_of(environment["NEO4J_server_memory_heap_max__size"])
        page_cache = _default_of(environment["NEO4J_server_memory_pagecache_size"])

        limit = _size_in_bytes(_default_of(service["mem_limit"]))

        assert limit > _size_in_bytes(heap) + _size_in_bytes(page_cache)

    def test_llm_default_limit_is_above_the_prompt_cache_cap(self, compose):
        service = compose["services"][LLM]
        command = service["command"]
        cache_ram_mib = int(_default_of(command[command.index("--cache-ram") + 1]))

        limit = _size_in_bytes(_default_of(service["mem_limit"]))

        assert limit > cache_ram_mib * MIB

    def test_extraction_llm_default_limit_holds_its_floor(self, compose):
        service = compose["services"][EXTRACTION_LLM]
        command = service["command"]
        cache_ram_mib = int(_default_of(command[command.index("--cache-ram") + 1]))
        floor = (EXPERT_TENSORS_ON_CPU_GIB + SERVER_OVERHEAD_GIB) * 1024 * MIB
        floor += cache_ram_mib * MIB

        assert _size_in_bytes(_default_of(service["mem_limit"])) >= floor

    @pytest.mark.parametrize("name", SERVICES)
    def test_logs_rotate(self, compose, name):
        assert compose["services"][name]["logging"] == LOGGING

    @pytest.mark.parametrize("name", [BACKUP, EXTRACTION])
    def test_reserves_no_gpu(self, compose, name):
        assert "deploy" not in compose["services"][name]


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


class TestExtractionLlmFollowsMistLlm:
    """The night server takes mist-llm's image, GPU reservation, healthcheck and KV cache form."""

    @pytest.mark.parametrize("key", ["image", "deploy", "healthcheck"])
    def test_matches_mist_llm(self, compose, key):
        services = compose["services"]
        assert services[EXTRACTION_LLM][key] == services[LLM][key]

    @pytest.mark.parametrize("setting", CACHE_AND_ATTENTION_SETTINGS)
    def test_cache_and_attention_match_mist_llm(self, compose, setting):
        services = compose["services"]
        assert _environment(services[EXTRACTION_LLM])[setting] == (
            _environment(services[LLM])[setting]
        )


class TestParityWithTheExtractionOverlay:
    """Main's `docker-compose.extraction.yml` local services, with each listed delta pinned."""

    @pytest.mark.parametrize("name", EXTRACTION_SERVICES)
    def test_keys_match_the_overlay_but_the_listed_deltas(self, compose, overlay, name):
        daily = set(compose["services"][name])
        local = set(overlay["services"][OVERLAY_COUNTERPART[name]])
        dropped = set(DROPPED_OVERLAY_KEYS.get(name, {}))
        added = set(ADDED_DAILY_KEYS.get(name, {}))

        assert dropped <= local and not added & local, "a listed delta is stale"
        assert daily == (local - dropped) | added

    @pytest.mark.parametrize("name", EXTRACTION_SERVICES)
    @pytest.mark.parametrize("key", ["hostname", "container_name"])
    def test_sets_no_fixed_container_name(self, compose, name, key):
        assert key not in compose["services"][name]

    def test_llm_image_matches_the_overlay(self, compose, overlay):
        assert compose["services"][EXTRACTION_LLM]["image"] == (
            overlay["services"][OVERLAY_LLM]["image"]
        )

    def test_llm_settings_match_the_overlay_but_the_deltas(self, compose, overlay):
        # Compared with no variable set: model path, batch, ubatch, mmap, reasoning, sampling
        # and the cache-ram cap all resolve to the overlay's values.
        daily = llama_server_settings(compose["services"][EXTRACTION_LLM])
        local = llama_server_settings(overlay["services"][OVERLAY_LLM])

        assert set(daily) == set(local) | set(CACHE_AND_ATTENTION_SETTINGS)
        for setting in set(local) - {"LLAMA_ARG_CTX_SIZE", "-ncmoe"}:
            assert _resolved(daily[setting]) == _resolved(local[setting]), setting

    @pytest.mark.parametrize(
        "setting, expected",
        [
            pytest.param("LLAMA_ARG_MODEL", "/models/ggml-org/gpt-oss-20b-MXFP4.gguf", id="model"),
            pytest.param("-b", "2048", id="batch"),
            pytest.param("-ub", "2048", id="ubatch"),
            pytest.param("-lm", "none", id="no-mmap"),
            pytest.param("LLAMA_ARG_REASONING", "on", id="reasoning"),
            pytest.param("LLAMA_ARG_THINK", "deepseek", id="reasoning-format"),
            pytest.param("--temp", "1.0", id="temp"),
            pytest.param("--top-p", "1.0", id="top-p"),
        ],
    )
    def test_llm_setting_equals_the_overlay(self, compose, overlay, setting, expected):
        daily = llama_server_settings(compose["services"][EXTRACTION_LLM])
        local = llama_server_settings(overlay["services"][OVERLAY_LLM])
        assert _resolved(daily[setting]) == _resolved(local[setting]) == expected

    def test_llm_reads_only_the_night_variables(self, compose):
        # A line carried into daily.env from a checkout's .env cannot move the serving identity.
        for setting, value in llama_server_settings(compose["services"][EXTRACTION_LLM]).items():
            expected = {DAILY_LLM_VARIABLES[setting]} if setting in DAILY_LLM_VARIABLES else set()
            assert set(INTERPOLATION.findall(value)) == expected, f"{setting}={value}"

    def test_delta_ctx_size_default(self, compose, overlay):
        daily = llama_server_settings(compose["services"][EXTRACTION_LLM])
        local = llama_server_settings(overlay["services"][OVERLAY_LLM])

        assert _resolved(daily["LLAMA_ARG_CTX_SIZE"]) == NIGHT_CTX_SIZE
        assert _resolved(local["LLAMA_ARG_CTX_SIZE"]) == OVERLAY_CTX_SIZE

    def test_delta_ncmoe_default(self, compose, overlay):
        daily_llm = llama_server_settings(compose["services"][EXTRACTION_LLM])
        local_llm = llama_server_settings(overlay["services"][OVERLAY_LLM])
        daily_service = _environment(compose["services"][EXTRACTION])
        local_service = _environment(overlay["services"][OVERLAY_SERVICE])

        assert _resolved(daily_llm["-ncmoe"]) == NIGHT_NCMOE
        assert _resolved(daily_service["EXTRACTION_SERVER_N_CPU_MOE"]) == NIGHT_NCMOE
        assert _resolved(local_llm["-ncmoe"]) == OVERLAY_NCMOE
        assert _resolved(local_service["EXTRACTION_SERVER_N_CPU_MOE"]) == OVERLAY_NCMOE

    @pytest.mark.parametrize("setting", CACHE_AND_ATTENTION_SETTINGS)
    def test_delta_cache_types_and_flash_attention(self, compose, overlay, setting):
        daily = llama_server_settings(compose["services"][EXTRACTION_LLM])
        local = llama_server_settings(overlay["services"][OVERLAY_LLM])

        assert daily[setting] == ("on" if setting == "LLAMA_ARG_FLASH_ATTN" else "q8_0")
        assert setting not in local

    def test_delta_cache_ram_variable(self, compose, overlay):
        daily = llama_server_settings(compose["services"][EXTRACTION_LLM])
        local = llama_server_settings(overlay["services"][OVERLAY_LLM])

        assert daily["--cache-ram"] == "${MIST_DAILY_EXTRACTION_CACHE_RAM_MIB:-2048}"
        assert local["--cache-ram"] == "${LLM_CACHE_RAM_MIB:-2048}"

    @pytest.mark.parametrize(
        "name, local", [(EXTRACTION_LLM, OVERLAY_LLM), (EXTRACTION, OVERLAY_SERVICE)]
    )
    def test_delta_profile(self, compose, overlay, name, local):
        assert compose["services"][name]["profiles"] == [EXTRACTION_PROFILE]
        assert overlay["services"][local]["profiles"] == ["extraction-local"]

    @pytest.mark.parametrize(
        "name, local", [(EXTRACTION_LLM, OVERLAY_LLM), (EXTRACTION, OVERLAY_SERVICE)]
    )
    def test_delta_restart_no(self, compose, overlay, name, local):
        # Only `extraction apply` starts them; a Docker restart must not run them by day.
        assert compose["services"][name]["restart"] == "no"
        assert overlay["services"][local]["restart"] == "unless-stopped"

    @pytest.mark.parametrize(
        "name, local", [(EXTRACTION_LLM, OVERLAY_LLM), (EXTRACTION, OVERLAY_SERVICE)]
    )
    def test_delta_mem_limit(self, compose, overlay, name, local):
        assert compose["services"][name]["mem_limit"] == MEM_LIMITS[name]
        assert "mem_limit" not in overlay["services"][local]

    def test_delta_no_published_port(self, compose, overlay):
        assert not compose["services"][EXTRACTION].get("ports")
        assert overlay["services"][OVERLAY_SERVICE]["ports"]

    def test_delta_image_in_place_of_build(self, compose, overlay):
        service = compose["services"][EXTRACTION]
        backend_image = compose["services"][BACKEND]["image"]

        assert service["image"] == backend_image.replace(f"{BACKEND}:", f"{EXTRACTION}:", 1)
        assert service["pull_policy"] == "never"
        assert overlay["services"][OVERLAY_SERVICE]["build"]["dockerfile"] == (
            "docker/extraction/Dockerfile"
        )

    def test_delta_llm_url(self, compose, overlay):
        daily = _environment(compose["services"][EXTRACTION])
        local = _environment(overlay["services"][OVERLAY_SERVICE])

        assert daily["EXTRACTION_LLM_BASE_URL"] == f"http://{EXTRACTION_LLM}:8080"
        assert local["EXTRACTION_LLM_BASE_URL"] == f"http://{OVERLAY_LLM}:8080"

    @pytest.mark.parametrize(
        "key, variable",
        [
            ("EXTRACTION_MODEL_HASH", "EXTRACTION_MODEL_HASH"),
            ("EXTRACTION_SERVER_DEVICE", "EXTRACTION_LLM_DEVICE"),
        ],
    )
    def test_delta_identity_variables_are_not_required(self, compose, overlay, key, variable):
        # A missing value never stops chat; `extraction apply` refuses to start instead.
        daily = _environment(compose["services"][EXTRACTION])
        local = _environment(overlay["services"][OVERLAY_SERVICE])

        assert daily[key] == f"${{{variable}:-}}"
        assert local[key].startswith(f"${{{variable}:?")

    def test_delta_per_call_timeout(self, compose, overlay):
        # Plan v5 A15: a full 2,048-token call at a contended decode, bounded at 10 minutes.
        daily = _environment(compose["services"][EXTRACTION])
        local = _environment(overlay["services"][OVERLAY_SERVICE])

        assert daily["EXTRACTION_LLM_TIMEOUT_SECONDS"] == "600"
        assert _resolved(local["EXTRACTION_LLM_TIMEOUT_SECONDS"]) == "120"

    def test_service_environment_matches_the_overlay_but_the_deltas(self, compose, overlay):
        daily = _environment(compose["services"][EXTRACTION])
        local = _environment(overlay["services"][OVERLAY_SERVICE])
        added = {f"EXTRACTION_SERVER_{name}" for name in ADDED_SERVER_DECLARATIONS}

        assert set(daily) == set(local) | added
        for key in set(local) - SERVICE_ENVIRONMENT_DELTAS:
            assert _resolved(daily[key]) == _resolved(local[key]), key

    @pytest.mark.parametrize(
        "key, expected",
        [
            ("EXTRACTION_REASONING_EFFORT", "low"),
            ("EXTRACTION_ADAPTER", "gptoss"),
            ("EXTRACTION_MAX_ATTEMPTS", "2"),
            ("EXTRACTION_LOCATION_LABEL", "local-4070"),
            ("EXTRACTION_CONSTRAINED_MODE", ""),
        ],
    )
    def test_service_setting_is_the_overlay_default_as_a_literal(self, compose, key, expected):
        assert _environment(compose["services"][EXTRACTION])[key] == expected

    def test_every_llm_setting_is_declared_from_the_same_expression(self, compose):
        services = compose["services"]
        llm, service = services[EXTRACTION_LLM], services[EXTRACTION]

        assert declaration_problems(llm, service) == []
        assert model_path_problems(llm, service) == []

    def test_declares_the_overlay_settings_and_the_kv_cache(self, compose):
        environment = _environment(compose["services"][EXTRACTION])
        declared = {
            key.removeprefix("EXTRACTION_SERVER_")
            for key in environment
            if key.startswith("EXTRACTION_SERVER_")
        }
        assert declared == EXPECTED_SERVER_DECLARATIONS | ADDED_SERVER_DECLARATIONS

    def test_service_healthcheck_matches_the_overlay(self, compose, overlay):
        assert compose["services"][EXTRACTION]["healthcheck"] == (
            overlay["services"][OVERLAY_SERVICE]["healthcheck"]
        )


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

    def test_extra_variables_name_each_variable_once(self):
        assert len(EXAMPLE_EXTRA_VARIABLES) == len(set(EXAMPLE_EXTRA_VARIABLES))

    def test_capability_variables_are_commented_out_placeholders(self):
        by_name = {name: (commented, value) for commented, name, value in _example_assignments()}

        for name in CAPABILITY_EXAMPLE_VARIABLES:
            commented, value = by_name[name]
            assert commented is True, f"{name} must be commented out"
            assert PLACEHOLDER.match(value), f"{name}={value}"

    def test_does_not_offer_the_capability_literals(self):
        # Compose sets these four itself; a daily.env line for one would be ignored.
        named = {name for _commented, name, _value in _example_assignments()}
        assert named & set(CAPABILITY_LITERALS) == set()

    def test_does_not_offer_the_extraction_entries_compose_sets(self):
        named = {name for _commented, name, _value in _example_assignments()}
        assert named & set(COMPOSE_SET_EXTRACTION_VARIABLES) == set()

    def test_says_compose_sets_the_extraction_mode_and_url(self):
        comments = " ".join(
            line.lstrip("# ").strip()
            for line in ENV_EXAMPLE_PATH.read_text(encoding="utf-8").splitlines()
            if line.startswith("#")
        )
        phrase = "compose sets MIST_EXTRACTION_INFERENCE and MIST_EXTRACTION_SERVICE_URL"
        assert phrase.lower() in comments.lower()

    def test_extraction_variables_are_commented_out(self):
        by_name = {name: commented for commented, name, _value in _example_assignments()}
        names = EXTRACTION_DAILY_VARIABLES | {"MIST_HEALTH_BACKLOG_MAX_AGE_S"}

        for name in names:
            assert by_name[name] is True, f"{name} must be commented out"

    def test_extraction_window_is_a_commented_placeholder_with_its_format(self):
        lines = ENV_EXAMPLE_PATH.read_text(encoding="utf-8").splitlines()
        index = lines.index("# MIST_EXTRACTION_WINDOW=<hh-mm-hh-mm>")
        comment = []
        for line in reversed(lines[:index]):
            if not line.startswith("#") or EXAMPLE_ASSIGNMENT.match(line):
                break
            comment.insert(0, line.lstrip("# "))
        text = " ".join(comment)

        for phrase in ("HH:MM-HH:MM", "host local time", "23:59-10:00"):
            assert phrase in text, phrase

    def test_every_value_is_a_placeholder_or_the_compose_default(self):
        defaults = dict(DEFAULT_INTERPOLATION.findall(_compose_text()))

        for _commented, name, value in _example_assignments():
            assert (
                PLACEHOLDER.match(value)
                or value == defaults.get(name)
                or value == EXAMPLE_PATH_VALUES.get(name)
            ), f"{name}={value}"

    def test_voice_variables_are_commented_out_placeholders(self):
        by_name = {name: (commented, value) for commented, name, value in _example_assignments()}

        for name in VOICE_EXAMPLE_VARIABLES:
            commented, value = by_name[name]
            assert commented is True, f"{name} must be commented out"
            assert PLACEHOLDER.match(value) or value == EXAMPLE_PATH_VALUES.get(name), name

    def test_voice_variables_are_ones_the_backend_reads(self):
        source = VOICE_SETTINGS_PATH.read_text(encoding="utf-8")
        unread = [name for name in VOICE_EXAMPLE_VARIABLES if f'"{name}"' not in source]
        assert not unread

    def test_does_not_offer_the_removed_per_slot_voice_variables(self):
        # Set at all, each one makes the backend refuse to start.
        named = {name for _commented, name, _value in _example_assignments()}
        assert named & set(_REMOVED_PER_SLOT) == set()

    def test_voice_comment_says_off_and_where_the_identities_live(self):
        comment = _comment_above("MIST_VOICE_STT_CHAIN")

        for phrase in ("`off`", "<MIST_DAILY_ROOT>/voices/", "outside every repository"):
            assert phrase in comment, phrase

    def test_phrases_file_comment_says_it_lives_under_the_models_directory(self):
        comment = _comment_above("MIST_VOICE_WAKE_PHRASES_FILE")

        for phrase in ("MODELS_DIR", "only `/models` is mounted"):
            assert phrase in comment, phrase

    def test_verify_model_comment_says_where_it_lives_and_that_wake_needs_it(self):
        comment = _comment_above("MIST_VOICE_VERIFY_MODEL")

        for phrase in ("speech/wake/<id>/", "MODELS_DIR", "no wake is accepted"):
            assert phrase in comment, phrase

    def test_header_no_longer_says_tts_stays_off(self):
        assert "turn TTS on" not in " ".join(ENV_EXAMPLE_PATH.read_text(encoding="utf-8").split())

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

    @pytest.mark.parametrize("name", ["MIST_EXTRACTION_INFERENCE", "MIST_EXTRACTION_SERVICE_URL"])
    def test_variable_row_says_compose_sets_it(self, name):
        rows = [
            line
            for line in README_PATH.read_text(encoding="utf-8").splitlines()
            if line.startswith(f"| `{name}` |")
        ]
        assert len(rows) == 1 and "compose sets it" in rows[0], rows

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

    def test_instance_root_layout_lists_the_secrets_directory(self):
        layout = _readme_section("Instance root layout")
        entries = [
            line.split()[0]
            for line in layout.splitlines()
            if line.startswith("  ") and line.strip()
        ]
        assert "secrets/" in entries

    @pytest.mark.parametrize(
        "phrase",
        [
            pytest.param("`<root>/secrets/`", id="secrets-folder"),
            pytest.param("`ha_token`", id="ha-token-file"),
            pytest.param("`spotify_token.json`", id="spotify-token-file"),
            pytest.param("python -m backend.capabilities.music.authorize", id="spotify-auth"),
            pytest.param("not backed up", id="secrets-not-backed-up"),
            pytest.param("`up`", id="activation-up"),
            pytest.param("`switch", id="activation-switch"),
            pytest.param("gaming mode", id="off-level"),
            pytest.param("`MIST_CAPABILITIES_ENABLED=false`", id="behaviour-rollback"),
            pytest.param("`capabilities.db`", id="store"),
            pytest.param("`backup_suspect`", id="data-rollback-suspect"),
            pytest.param("scripts/backup/scheduler.py", id="data-rollback-rule"),
            pytest.param("`accept-backup-baseline --label <label>`", id="data-rollback-baseline"),
        ],
    )
    def test_capabilities_section_covers(self, phrase):
        text = " ".join(_readme_section("Capabilities").split())
        assert phrase in text

    def test_capabilities_section_names_every_capability_variable(self):
        text = _readme_section("Capabilities")
        missing = [name for name in CAPABILITY_EXAMPLE_VARIABLES if f"`{name}`" not in text]
        assert not missing


class TestReadmeVoice:
    """What the README tells the operator about voice on this instance."""

    def test_no_tts_literal_in_the_contract(self):
        assert "TTS_ENABLED=false" not in README_PATH.read_text(encoding="utf-8")

    def test_instance_root_layout_lists_the_voices_directory(self):
        layout = _readme_section("Instance root layout")
        entries = [
            line.split()[0]
            for line in layout.splitlines()
            if line.startswith("  ") and line.strip()
        ]
        assert "voices/" in entries

    @pytest.mark.parametrize(
        "phrase",
        [
            pytest.param("speech weights from `MODELS_DIR`", id="speech-weights"),
            pytest.param("`<MIST_DAILY_ROOT>/voices/`", id="voice-identities"),
            pytest.param("`mist-<instance>_voice-cache`", id="cache-volume"),
            pytest.param("deleted while the backend is stopped", id="cache-deletable"),
            pytest.param("`<MODELS_DIR>/speech/wake/<model id>/`", id="wake-model-staging"),
            pytest.param("`<MODELS_DIR>/speech/wake/phrases.json`", id="phrases-staging"),
            pytest.param("personal use only", id="wake-model-licence"),
            pytest.param("`<MODELS_DIR>/speech/wake/<verifier id>/`", id="verifier-staging"),
            pytest.param("`MIST_VOICE_VERIFY_THREADS`", id="verifier-threads"),
            pytest.param("never committed", id="verifier-never-committed"),
            pytest.param("no wake is accepted", id="verifier-fail-closed"),
        ],
    )
    def test_voice_section_covers(self, phrase):
        assert phrase in " ".join(_readme_section("Voice").split())

    def test_voice_section_names_every_voice_variable(self):
        text = _readme_section("Voice")
        missing = [name for name in VOICE_EXAMPLE_VARIABLES if f"`{name}`" not in text]
        assert not missing

    @pytest.mark.parametrize(
        "phrase",
        [
            pytest.param("releases every speech engine on the GPU", id="release"),
            pytest.param("reloads the speech engines", id="reload"),
            pytest.param("`reload_failed`", id="reload-failed"),
            pytest.param("`/health`", id="health"),
            pytest.param("typed turns keep working", id="typed-turns"),
        ],
    )
    def test_gaming_mode_covers_the_speech_engines(self, phrase):
        assert phrase in " ".join(_readme_section("Gaming mode").split())

    def test_gaming_mode_no_longer_says_tts_is_off(self):
        assert "TTS is off" not in " ".join(_readme_section("Gaming mode").split())


def _readme_section(title: str) -> str:
    text = README_PATH.read_text(encoding="utf-8")
    match = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    assert match is not None, f"README.md has no '## {title}' section"
    return match.group(1)
