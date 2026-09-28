"""Tests for the standalone GTX 1070 host deploy file (MIS-171 compose-split,
goal mist-two-loop).

Hermetic: parses `docker/extraction/compose.host.yml` and
`backend/extraction_service/settings.py` as text/YAML/AST only. No docker, no
network, no import of `backend.extraction_service` itself.

`docker/extraction/compose.host.yml` holds ONLY the three services the remote
GTX 1070 host runs: its own llama-server (`mist-extraction-llm-host`), the
stateless extraction service (`mist-extraction-host`), and the Tailscale
sidecar (`mist-extraction-ts`) that exposes it over the tailnet. It is
STANDALONE -- it needs no other compose file to resolve, and in particular it
must never start the repo root docker-compose.yml's main stack (mist-llm,
mist-neo4j, mist-backend). That was defect 2 the compose-split fixed: the
previously documented host command
(`-f docker-compose.yml -f docker-compose.extraction.yml --profile
extraction-host`) started the whole main stack on the remote 1070 machine as
well as the extraction services.

The LOCAL overlay (`docker-compose.extraction.yml`, mist-extraction-llm-local
/ mist-extraction-local) is tested separately by
`tests/unit/test_compose_extraction.py`; that file's own tests pin defect 1
(the local profile must never need TS_AUTHKEY).

Reuses `test_compose_pins.py`'s pinned-image constant rather than restating
the digest, so a future re-pin only needs to change one file to keep every
compose test module in sync.
"""

import ast
import re
from pathlib import Path

import yaml

from tests.unit.test_compose_pins import DIGEST_PATTERN, PINNED_IMAGE

REPO_ROOT = Path(__file__).resolve().parents[2]
HOST_COMPOSE_PATH = REPO_ROOT / "docker" / "extraction" / "compose.host.yml"
SETTINGS_PATH = REPO_ROOT / "backend" / "extraction_service" / "settings.py"

# Resolved on the host by the lead on 2026-09-26 (tag kept in the file comments).
TAILSCALE_PINNED_IMAGE = (
    "tailscale/tailscale@sha256:c507f3a2a6ab1cabd8d809b98edeb41edbd5c3fb6ad9632ffd098b4c7d0b4065"
)
DIGEST_REF_PATTERN = re.compile(r"^[a-z0-9./_-]+@sha256:[0-9a-f]{64}$")

EXPECTED_SERVICES = {"mist-extraction-ts", "mist-extraction-llm-host", "mist-extraction-host"}

MAIN_STACK_SERVICE_NAMES = {"mist-llm", "mist-neo4j", "mist-backend"}


def _compose() -> dict:
    with HOST_COMPOSE_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _services() -> dict:
    return _compose()["services"]


def _env_dict(service: dict) -> dict[str, str]:
    """Flatten a service's list-form `environment:` into a key -> raw-value dict.

    Values are the raw, un-interpolated compose strings (e.g.
    `"${TS_AUTHKEY:?msg}"`) -- `yaml.safe_load` never evaluates `${...}`
    interpolation, so the required-vs-default form is still inspectable.
    """
    raw = service.get("environment", [])
    out: dict[str, str] = {}
    for entry in raw:
        key, _, value = str(entry).partition("=")
        out[key] = value
    return out


def _command(service: dict) -> list[str]:
    return [str(arg) for arg in service.get("command", [])]


def _sidecar() -> dict:
    return next(svc for svc in _services().values() if "tailscale" in svc.get("image", ""))


def _sidecar_name() -> str:
    return next(name for name, svc in _services().items() if "tailscale" in svc.get("image", ""))


def _host_llm() -> dict:
    return _services()["mist-extraction-llm-host"]


def _host_extraction() -> dict:
    return _services()["mist-extraction-host"]


class TestExactlyTheThreeHostServices:
    """Pins defect 2: the host file starts ONLY its three own services."""

    def test_exactly_three_services(self) -> None:
        assert set(_services()) == EXPECTED_SERVICES

    def test_no_main_stack_service_names(self) -> None:
        assert MAIN_STACK_SERVICE_NAMES.isdisjoint(_services())


class TestNoOtherComposeFileNeeded:
    """Every depends_on / network_mode target must resolve within this file alone."""

    def test_every_depends_on_target_is_defined_here(self) -> None:
        names = set(_services())
        for name, service in _services().items():
            for target in service.get("depends_on", {}):
                assert target in names, f"{name} depends_on undefined service {target!r}"

    def test_network_mode_service_target_is_defined_here(self) -> None:
        names = set(_services())
        for name, service in _services().items():
            network_mode = service.get("network_mode", "")
            if network_mode.startswith("service:"):
                target = network_mode.split(":", 1)[1]
                assert target in names, f"{name} network_mode targets undefined service {target!r}"

    def test_top_level_volumes_used_by_this_file_are_declared_here(self) -> None:
        top_level_volumes = set(_compose().get("volumes") or {})
        assert top_level_volumes == {"mist-extraction-ts-state", "mist-extraction-cuda-cache"}


class TestBuildContextResolvesToRepoRoot:
    def test_context_resolves_to_repo_root(self) -> None:
        context = _host_extraction()["build"]["context"]
        resolved = (HOST_COMPOSE_PATH.parent / context).resolve()
        assert resolved == REPO_ROOT

    def test_dockerfile_path_is_repo_relative(self) -> None:
        assert _host_extraction()["build"]["dockerfile"] == "docker/extraction/Dockerfile"


class TestModelsVolumeIsParameterised:
    def test_models_mount_uses_an_env_var(self) -> None:
        volumes = _host_llm().get("volumes", [])
        models_mount = next(entry for entry in volumes if entry.endswith(":/models:ro"))
        assert (
            "${MODELS_DIR" in models_mount
        ), f"models mount is not parameterised: {models_mount!r}"


class TestLlamaServerPins:
    def test_host_llm_uses_the_mist_llm_digest(self) -> None:
        assert _host_llm()["image"] == PINNED_IMAGE

    def test_host_llm_matches_the_digest_pattern(self) -> None:
        assert DIGEST_PATTERN.match(_host_llm()["image"])

    def test_host_llm_does_not_pass_reasoning_budget(self) -> None:
        assert "--reasoning-budget" not in _command(_host_llm()), (
            "mist-extraction-llm-host must not pass --reasoning-budget: the service "
            "sends a per-request budget, and the server's own default (-1, "
            "unrestricted) already matches 'no cap'"
        )

    def test_host_llm_passes_cache_ram(self) -> None:
        command = _command(_host_llm())
        assert "--cache-ram" in command
        value = command[command.index("--cache-ram") + 1]
        assert value, "mist-extraction-llm-host's --cache-ram has no value"

    def test_ncmoe_is_parameterized(self) -> None:
        command = _command(_host_llm())
        assert "-ncmoe" in command
        assert "EXTRACTION_HOST_NCMOE" in command[command.index("-ncmoe") + 1]

    def test_ncmoe_default_is_24(self) -> None:
        # gpt-oss-20b's block_count is verified as 24 via GGUF metadata (the
        # host build's finding); 24 means "all experts on CPU", replacing the
        # previous unverified 999 sentinel. The real host fit (ncmoe=12) was
        # measured at ctx 8192 and is superseded by the ctx-16384 default
        # below -- it needs re-measuring, so 24 (not 12) stays the default.
        command = _command(_host_llm())
        value = command[command.index("-ncmoe") + 1]
        assert value == "${EXTRACTION_HOST_NCMOE:-24}", value

    def test_ctx_size_default_is_16384(self) -> None:
        # A real /v1/extract probe needed 9162 prompt tokens plus
        # max_tokens=2048 from the extraction engine, exceeding the old 8192
        # default.
        env = _env_dict(_host_llm())
        assert env["LLAMA_ARG_CTX_SIZE"] == "${EXTRACTION_LLM_CTX_SIZE:-16384}", env[
            "LLAMA_ARG_CTX_SIZE"
        ]


class TestServiceEnvPassthrough:
    """Pins the compose passthrough for three vars that were previously
    silently unreachable inside the container (compose never forwarded
    them, even though README.md documented them as operator-settable).
    """

    def test_llm_timeout_seconds_is_forwarded_with_120_default(self) -> None:
        env = _env_dict(_host_extraction())
        assert (
            env["EXTRACTION_LLM_TIMEOUT_SECONDS"] == "${EXTRACTION_LLM_TIMEOUT_SECONDS:-120}"
        ), env["EXTRACTION_LLM_TIMEOUT_SECONDS"]

    def test_max_attempts_is_forwarded_with_2_default(self) -> None:
        env = _env_dict(_host_extraction())
        assert env["EXTRACTION_MAX_ATTEMPTS"] == "${EXTRACTION_MAX_ATTEMPTS:-2}", env[
            "EXTRACTION_MAX_ATTEMPTS"
        ]

    def test_constrained_mode_is_forwarded_with_empty_default(self) -> None:
        # Empty default, not a literal mode name -- an empty string is
        # falsy, so `constrained_mode or adapter.default_constrained_mode`
        # (engine.py) falls through to the adapter's own default exactly
        # like the unset/None case did before this var was reachable.
        env = _env_dict(_host_extraction())
        assert env["EXTRACTION_CONSTRAINED_MODE"] == "${EXTRACTION_CONSTRAINED_MODE:-}", env[
            "EXTRACTION_CONSTRAINED_MODE"
        ]


class TestHostPublishesNoPorts:
    def test_no_service_publishes_ports(self) -> None:
        for name, service in _services().items():
            assert "ports" not in service, f"{name} (host deploy) must not publish ports"

    def test_extraction_service_shares_the_tailscale_sidecars_network(self) -> None:
        assert _host_extraction().get("network_mode") == f"service:{_sidecar_name()}"


class TestTailscaleAuthKey:
    def test_ts_authkey_has_no_default(self) -> None:
        value = _env_dict(_sidecar())["TS_AUTHKEY"]
        assert re.search(
            r"\$\{TS_AUTHKEY:\?", value
        ), f"TS_AUTHKEY must use the required ${{VAR:?msg}} form, got {value!r}"
        assert ":-" not in value, f"TS_AUTHKEY must not have a default, got {value!r}"

    def test_tailscale_image_is_pinned_by_digest(self) -> None:
        """The sidecar image is the digest the lead resolved on the host, not a moving tag."""
        assert _sidecar()["image"] == TAILSCALE_PINNED_IMAGE
        assert DIGEST_REF_PATTERN.match(_sidecar()["image"])


class TestModelHashRequired:
    def test_model_hash_uses_the_required_form(self) -> None:
        value = _env_dict(_host_extraction())["EXTRACTION_MODEL_HASH"]
        assert re.search(
            r"\$\{EXTRACTION_MODEL_HASH:\?", value
        ), f"EXTRACTION_MODEL_HASH must use ${{VAR:?msg}}, got {value!r}"
        assert ":-" not in value, "EXTRACTION_MODEL_HASH must not have a default"


class TestHostCudaCache:
    def test_host_llm_sets_cuda_cache_path_on_a_named_volume(self) -> None:
        top_level_volumes = set(_compose().get("volumes") or {})
        cache_path = _env_dict(_host_llm()).get("CUDA_CACHE_PATH")
        assert cache_path, "mist-extraction-llm-host must set CUDA_CACHE_PATH"

        mounted_on_named_volume = any(
            entry.split(":", 1)[1] == cache_path and entry.split(":", 1)[0] in top_level_volumes
            for entry in _host_llm().get("volumes", [])
            if ":" in entry
        )
        assert mounted_on_named_volume, (
            f"CUDA_CACHE_PATH={cache_path!r} must be mounted on one of the compose file's "
            f"top-level named volumes {top_level_volumes!r}"
        )


def _service_settings_fields_without_defaults() -> list[str]:
    """Parse `ServiceSettings`'s dataclass fields that carry no class-body default.

    AST-based per the T1b brief, rather than importing the module -- see
    test_compose_extraction.py's identical helper for the full rationale.
    """
    tree = ast.parse(SETTINGS_PATH.read_text(encoding="utf-8"))
    class_def = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "ServiceSettings"
    )
    fields = []
    for node in class_def.body:
        if isinstance(node, ast.AnnAssign) and node.value is None:
            assert isinstance(node.target, ast.Name)
            fields.append(node.target.id)
    return fields


class TestRequiredSettingsAreSetInCompose:
    def test_fields_without_defaults_are_pinpointed_correctly(self) -> None:
        fields = _service_settings_fields_without_defaults()
        assert fields == ["llm_base_url", "model_hash", "model_file"]

    def test_every_required_field_has_its_env_var_set(self) -> None:
        fields = _service_settings_fields_without_defaults()
        required_env_vars = [f"EXTRACTION_{field.upper()}" for field in fields]
        env = _env_dict(_host_extraction())
        for var in required_env_vars:
            assert var in env, f"mist-extraction-host does not set {var}"
