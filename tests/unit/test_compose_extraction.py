"""Tests for the LOCAL extraction service deploy artifacts (T1b, goal mist-two-loop /
MIS-171; MIS-171 compose-split).

Hermetic: parses `docker-compose.extraction.yml`, `docker/extraction/Dockerfile`,
`docker/extraction/requirements.txt`, and `backend/extraction_service/settings.py`
as text/YAML/AST only. No docker, no network, no import of `backend.extraction_service`
itself (its own import closure pulls in torch-adjacent packages this test tier does not
need to require).

`docker-compose.extraction.yml` is the LOCAL overlay only (mist-extraction-llm-local,
mist-extraction-local) -- it joins docker-compose.yml's network and never needs a
Tailscale key. The GTX 1070 host's three services (its own llama-server, the
extraction service, and the Tailscale sidecar) live in the standalone
`docker/extraction/compose.host.yml`, tested separately by
`tests/unit/test_compose_extraction_host.py`. Splitting the files fixed two defects:
compose interpolates every service in a file regardless of the active profile, so the
host sidecar's required `TS_AUTHKEY` broke `docker compose config` for the local
profile; and the documented host command started docker-compose.yml's whole main
stack (mist-llm, mist-neo4j, mist-backend) on the remote machine.

Reuses `test_compose_pins.py`'s pinned-image constant rather than restating the digest,
so a future re-pin only needs to change one file to keep both test modules in sync.
"""

import ast
import re
from pathlib import Path

import yaml

from tests.unit.test_compose_pins import DIGEST_PATTERN, PINNED_IMAGE

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATH = REPO_ROOT / "docker-compose.extraction.yml"
DOCKERFILE_PATH = REPO_ROOT / "docker" / "extraction" / "Dockerfile"
REQUIREMENTS_PATH = REPO_ROOT / "docker" / "extraction" / "requirements.txt"
SETTINGS_PATH = REPO_ROOT / "backend" / "extraction_service" / "settings.py"

PYTHON_PINNED_IMAGE = (
    "python@sha256:e41613d42d4891e4930f79523f93f81bbc7632584ec65e36ab055f41a800b41e"
)
DIGEST_REF_PATTERN = re.compile(r"^[a-z0-9./_-]+@sha256:[0-9a-f]{64}$")

LOCAL_PROFILE = "extraction-local"

# The ONLY required (`${VAR:?...}`) env vars this file's local services may declare.
# A Tailscale key belongs to the host file only -- this allowlist is what pins that:
# if a future edit adds another required var here (in particular TS_AUTHKEY), this
# test fails rather than silently reintroducing the interpolation defect that made
# `--profile extraction-local` fail with "required variable TS_AUTHKEY is missing".
# EXTRACTION_LLM_DEVICE joins EXTRACTION_MODEL_HASH for the same reason it is
# required: it keys the serving identity, and a deploy must name it (MIS-196).
REQUIRED_VAR_ALLOWLIST = {"EXTRACTION_MODEL_HASH", "EXTRACTION_LLM_DEVICE"}

REQUIRED_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+):\?")

SERVING_IDENTITY_PATH = REPO_ROOT / "backend" / "knowledge" / "extraction" / "serving_identity.py"

# Declared llama-server settings (MIS-196). The extraction service cannot see its
# llama-server's launch flags, so compose declares each one to it as
# `EXTRACTION_SERVER_<NAME>`, from the same expression the flag uses, and the
# service folds them into its serving identity. Both compose files are checked
# against the tables below by `declaration_problems`; the host test reuses them.
SERVER_ENV_PREFIX = "EXTRACTION_SERVER_"
LLAMA_ARG_PREFIX = "LLAMA_ARG_"

# Every llama-server env var or command flag that is NOT declared, and why. A
# setting is declared unless it provably cannot change the bytes the model
# returns (plan mist-extraction-epoch-identity v1, "Setting classification").
SERVER_SETTING_EXCLUSIONS: dict[str, str] = {
    "LLAMA_ARG_HOST": "Transport: the address llama-server binds to.",
    "LLAMA_ARG_PORT": "Transport: the port llama-server listens on.",
    "LLAMA_ARG_METRICS": "Exposes the /metrics endpoint; no completion reads it.",
    "--no-webui": "UI: turns off the bundled web UI.",
    "-lm": (
        "Loads the same weight bytes without mmap; the bytes in memory are identical, "
        "so the output cannot change."
    ),
    "CUDA_CACHE_PATH": (
        "Caches the driver's compile of the same PTX; the device it compiles for is "
        "declared as EXTRACTION_SERVER_DEVICE."
    ),
    "CUDA_CACHE_MAXSIZE": "Caps that compile cache; the device is declared separately.",
    "LLAMA_ARG_MODEL": (
        "Keyed through EXTRACTION_MODEL_FILE: one expression feeds both, and "
        "`model_path_problems` pins them equal."
    ),
}

# Command flags -> the `EXTRACTION_SERVER_<NAME>` each is declared as. An env var
# `LLAMA_ARG_<NAME>` is declared as `EXTRACTION_SERVER_<NAME>`, so it needs no entry.
FLAG_DECLARATIONS: dict[str, str] = {
    "-ncmoe": "N_CPU_MOE",
    "-b": "BATCH",
    "-ub": "UBATCH",
    "--cache-ram": "CACHE_RAM_MIB",
    "--temp": "TEMP",
    "--top-p": "TOP_P",
}

# Declarations that name no llama-server flag: the image string, which pins every
# flag left at its build default, and the GPU, which no flag names.
STANDALONE_DECLARATIONS = {"IMAGE", "DEVICE"}

# The keyed set the plan names for both files, so a dropped declaration fails even
# when its flag is dropped with it.
EXPECTED_SERVER_DECLARATIONS = {
    "IMAGE",
    "DEVICE",
    "N_GPU_LAYERS",
    "N_CPU_MOE",
    "CTX_SIZE",
    "BATCH",
    "UBATCH",
    "N_PARALLEL",
    "CACHE_RAM_MIB",
    "JINJA",
    "REASONING",
    "THINK",
    "TEMP",
    "TOP_P",
}

_NUMBER_PATTERN = re.compile(r"^-?\d+(\.\d+)?$")


def _compose() -> dict:
    with COMPOSE_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _services() -> dict:
    return _compose()["services"]


def _env_dict(service: dict) -> dict[str, str]:
    """Flatten a service's list-form `environment:` into a key -> raw-value dict.

    Values are the raw, un-interpolated compose strings (e.g.
    `"${EXTRACTION_MODEL_HASH:?msg}"`) -- `yaml.safe_load` never evaluates `${...}`
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


def _local_llm() -> dict:
    return _services()["mist-extraction-llm-local"]


def _local_extraction() -> dict:
    return _services()["mist-extraction-local"]


def llama_server_settings(llm: dict) -> dict[str, str]:
    """Every env var and command flag of a llama-server service, as name -> raw value.

    A flag's value is the token after it unless that token is itself a flag; a
    flag with no value (`--no-webui`) maps to "". A token that is neither a flag
    nor a flag's value is reported as a setting of its own, so it cannot hide.
    """
    settings: dict[str, str] = {}
    for entry in llm.get("environment", []):
        key, _, value = str(entry).partition("=")
        settings[key] = value
    tokens = _command(llm)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        nxt = tokens[index + 1] if index + 1 < len(tokens) else None
        if not token.startswith("-") or _NUMBER_PATTERN.match(token):
            settings[f"<positional {token}>"] = token
            index += 1
            continue
        assert token not in settings, f"command flag {token} is passed twice"
        if nxt is not None and (not nxt.startswith("-") or _NUMBER_PATTERN.match(nxt)):
            settings[token] = nxt
            index += 2
        else:
            settings[token] = ""
            index += 1
    return settings


def _declared_name(setting: str) -> str | None:
    if setting.startswith(LLAMA_ARG_PREFIX):
        return setting[len(LLAMA_ARG_PREFIX) :]
    return FLAG_DECLARATIONS.get(setting)


def _server_declarations(extraction: dict) -> dict[str, str]:
    return {
        key[len(SERVER_ENV_PREFIX) :]: value
        for key, value in _env_dict(extraction).items()
        if key.startswith(SERVER_ENV_PREFIX)
    }


def declaration_problems(llm: dict, extraction: dict) -> list[str]:
    """What the extraction service fails to declare about its llama-server.

    Every llama-server setting is either in `SERVER_SETTING_EXCLUSIONS` or declared
    as `EXTRACTION_SERVER_<NAME>` with the identical raw compose string, so an
    operator override reaches both containers through one `${...}` expression.
    Every declaration names a setting the server has, so none goes stale.
    """
    declared = _server_declarations(extraction)
    problems: list[str] = []
    named: set[str] = set()
    for setting, value in llama_server_settings(llm).items():
        if setting in SERVER_SETTING_EXCLUSIONS:
            continue
        name = _declared_name(setting)
        if name is None:
            problems.append(
                f"{setting} is neither declared to the extraction service nor in "
                "SERVER_SETTING_EXCLUSIONS"
            )
            continue
        named.add(name)
        if name not in declared:
            problems.append(f"{setting}={value!r} is not declared as {SERVER_ENV_PREFIX}{name}")
        elif declared[name] != value:
            problems.append(
                f"{SERVER_ENV_PREFIX}{name}={declared[name]!r} differs from {setting}={value!r}"
            )
    if declared.get("IMAGE") != llm.get("image"):
        problems.append(
            f"{SERVER_ENV_PREFIX}IMAGE={declared.get('IMAGE')!r} differs from the "
            f"llama-server image {llm.get('image')!r}"
        )
    for name in sorted(set(declared) - named - STANDALONE_DECLARATIONS):
        problems.append(f"{SERVER_ENV_PREFIX}{name} names no setting of the llama-server")
    return problems


def model_path_problems(llm: dict, extraction: dict) -> list[str]:
    """`LLAMA_ARG_MODEL` must load the file the service reports as `EXTRACTION_MODEL_FILE`."""
    model = _env_dict(llm).get("LLAMA_ARG_MODEL")
    model_file = _env_dict(extraction).get("EXTRACTION_MODEL_FILE")
    if model_file is None or model != f"/models/{model_file}":
        return [f"LLAMA_ARG_MODEL={model!r} is not /models/ + EXTRACTION_MODEL_FILE={model_file!r}"]
    return []


def device_problems(extraction: dict) -> list[str]:
    """The device is required, with no default, from `EXTRACTION_LLM_DEVICE`."""
    value = _server_declarations(extraction).get("DEVICE", "")
    if not re.fullmatch(r"\$\{EXTRACTION_LLM_DEVICE:\?[^}]+\}", value):
        return [f"{SERVER_ENV_PREFIX}DEVICE must be ${{EXTRACTION_LLM_DEVICE:?msg}}, got {value!r}"]
    return []


def required_server_keys() -> tuple[str, ...]:
    """`REQUIRED_SERVER_KEYS` as `serving_identity.py` defines it, read by AST."""
    tree = ast.parse(SERVING_IDENTITY_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "REQUIRED_SERVER_KEYS"
        ):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError("REQUIRED_SERVER_KEYS not found in serving_identity.py")


class TestLocalFileHasNoTailscale:
    """Pins defect 1: the local profile must never need a Tailscale key."""

    def test_no_service_is_named_or_images_tailscale(self) -> None:
        for name, service in _services().items():
            assert "tailscale" not in name, f"{name} must not be a Tailscale service"
            assert "tailscale" not in service.get(
                "image", ""
            ), f"{name} must not use a tailscale image"

    def test_no_service_requires_ts_authkey(self) -> None:
        for name, service in _services().items():
            env = _env_dict(service)
            assert "TS_AUTHKEY" not in env, f"{name} must not reference TS_AUTHKEY"

    def test_every_required_var_is_on_the_allowlist(self) -> None:
        """Every `${VAR:?...}` in the file is EXTRACTION_MODEL_HASH -- nothing else.

        This is the direct pin for defect 1: compose interpolates every service in a
        file whatever the active profile, so ANY required var anywhere in this file
        (not just on an active service) would fail `docker compose config` for
        `--profile extraction-local`. EXTRACTION_MODEL_HASH failing fast is correct
        and stays; nothing else may be required here.
        """
        found: set[str] = set()
        for service in _services().values():
            for value in _env_dict(service).values():
                found.update(REQUIRED_VAR_PATTERN.findall(value))
        assert found == REQUIRED_VAR_ALLOWLIST, (
            f"required vars in docker-compose.extraction.yml are {found!r}, "
            f"expected exactly {REQUIRED_VAR_ALLOWLIST!r}"
        )


class TestLocalFileHasOnlyLocalServices:
    def test_exactly_two_services(self) -> None:
        assert set(_services()) == {"mist-extraction-llm-local", "mist-extraction-local"}

    def test_no_host_or_main_stack_service_names(self) -> None:
        forbidden = {
            "mist-extraction-llm-host",
            "mist-extraction-host",
            "mist-extraction-ts",
            "mist-llm",
            "mist-neo4j",
            "mist-backend",
        }
        assert forbidden.isdisjoint(_services())


class TestLlamaServerPins:
    def test_local_llm_uses_the_mist_llm_digest(self) -> None:
        assert _local_llm()["image"] == PINNED_IMAGE

    def test_local_llm_matches_the_digest_pattern(self) -> None:
        assert DIGEST_PATTERN.match(_local_llm()["image"])

    def test_local_llm_does_not_pass_reasoning_budget(self) -> None:
        assert "--reasoning-budget" not in _command(_local_llm()), (
            "mist-extraction-llm-local must not pass --reasoning-budget: the service "
            "sends a per-request budget, and the server's own default (-1, "
            "unrestricted) already matches 'no cap'"
        )

    def test_local_llm_passes_cache_ram(self) -> None:
        command = _command(_local_llm())
        assert "--cache-ram" in command
        value = command[command.index("--cache-ram") + 1]
        assert value, "mist-extraction-llm-local's --cache-ram has no value"

    def test_ncmoe_is_parameterized(self) -> None:
        command = _command(_local_llm())
        assert "-ncmoe" in command
        assert "EXTRACTION_LOCAL_NCMOE" in command[command.index("-ncmoe") + 1]

    def test_ncmoe_default_is_24(self) -> None:
        # gpt-oss-20b's block_count is verified as 24 via GGUF metadata (the
        # host build's finding); 24 means "all experts on CPU", replacing the
        # previous unverified 999 sentinel.
        command = _command(_local_llm())
        value = command[command.index("-ncmoe") + 1]
        assert value == "${EXTRACTION_LOCAL_NCMOE:-24}", value

    def test_ctx_size_default_is_16384(self) -> None:
        # A real /v1/extract probe needed 9162 prompt tokens plus
        # max_tokens=2048 from the extraction engine, exceeding the old 8192
        # default.
        env = _env_dict(_local_llm())
        assert env["LLAMA_ARG_CTX_SIZE"] == "${EXTRACTION_LLM_CTX_SIZE:-16384}", env[
            "LLAMA_ARG_CTX_SIZE"
        ]


class TestServiceEnvPassthrough:
    """Pins the compose passthrough for three vars that were previously
    silently unreachable inside the container (compose never forwarded
    them, even though README.md documented them as operator-settable).
    """

    def test_llm_timeout_seconds_is_forwarded_with_120_default(self) -> None:
        env = _env_dict(_services()["mist-extraction-local"])
        assert (
            env["EXTRACTION_LLM_TIMEOUT_SECONDS"] == "${EXTRACTION_LLM_TIMEOUT_SECONDS:-120}"
        ), env["EXTRACTION_LLM_TIMEOUT_SECONDS"]

    def test_max_attempts_is_forwarded_with_2_default(self) -> None:
        env = _env_dict(_services()["mist-extraction-local"])
        assert env["EXTRACTION_MAX_ATTEMPTS"] == "${EXTRACTION_MAX_ATTEMPTS:-2}", env[
            "EXTRACTION_MAX_ATTEMPTS"
        ]

    def test_constrained_mode_is_forwarded_with_empty_default(self) -> None:
        # Empty default, not a literal mode name -- an empty string is
        # falsy, so `constrained_mode or adapter.default_constrained_mode`
        # (engine.py) falls through to the adapter's own default exactly
        # like the unset/None case did before this var was reachable.
        env = _env_dict(_services()["mist-extraction-local"])
        assert env["EXTRACTION_CONSTRAINED_MODE"] == "${EXTRACTION_CONSTRAINED_MODE:-}", env[
            "EXTRACTION_CONSTRAINED_MODE"
        ]


class TestProfile:
    def test_every_service_has_exactly_the_local_profile(self) -> None:
        for name, service in _services().items():
            profiles = service.get("profiles")
            assert profiles == [LOCAL_PROFILE], f"{name} must be in [{LOCAL_PROFILE!r}]"


class TestNoHostPortsBeyondTheDebugLoopback:
    def test_local_llm_publishes_no_ports(self) -> None:
        assert "ports" not in _local_llm()

    def test_extraction_service_only_publishes_the_loopback_debug_port(self) -> None:
        ports = _services()["mist-extraction-local"].get("ports", [])
        for entry in ports:
            assert str(entry).startswith("127.0.0.1:"), f"non-loopback port published: {entry!r}"


class TestModelHashRequired:
    def test_model_hash_uses_the_required_form(self) -> None:
        value = _env_dict(_services()["mist-extraction-local"])["EXTRACTION_MODEL_HASH"]
        assert re.search(
            r"\$\{EXTRACTION_MODEL_HASH:\?", value
        ), f"EXTRACTION_MODEL_HASH must use ${{VAR:?msg}}, got {value!r}"
        assert ":-" not in value, "EXTRACTION_MODEL_HASH must not have a default"


class TestDeclaredServerSettings:
    """The local service declares every keyed setting of its llama-server (MIS-196)."""

    def test_every_llama_server_setting_is_declared_or_excluded(self) -> None:
        assert declaration_problems(_local_llm(), _local_extraction()) == []

    def test_the_declared_names_are_the_keyed_set(self) -> None:
        assert set(_server_declarations(_local_extraction())) == EXPECTED_SERVER_DECLARATIONS

    def test_the_image_repeats_the_llama_server_image(self) -> None:
        declared = _server_declarations(_local_extraction())
        assert declared["IMAGE"] == _local_llm()["image"] == PINNED_IMAGE

    def test_ctx_size_is_the_expression_that_feeds_llama_server(self) -> None:
        declared = _server_declarations(_local_extraction())
        assert declared["CTX_SIZE"] == _env_dict(_local_llm())["LLAMA_ARG_CTX_SIZE"]

    def test_the_device_is_required_with_no_default(self) -> None:
        assert device_problems(_local_extraction()) == []

    def test_the_model_path_is_pinned_to_the_model_file(self) -> None:
        assert model_path_problems(_local_llm(), _local_extraction()) == []

    def test_every_required_server_key_is_declared(self) -> None:
        declared = _server_declarations(_local_extraction())
        for key in required_server_keys():
            assert declared.get(key.upper()), f"{SERVER_ENV_PREFIX}{key.upper()} is empty"


class TestTheDeclarationCheckCanFail:
    """Guards the checker itself: each kind of drift is reported, not passed."""

    def _pair(self) -> tuple[dict, dict]:
        return _local_llm(), _local_extraction()

    def test_an_undeclared_flag_is_reported(self) -> None:
        llm, extraction = self._pair()
        llm["command"] = [*llm["command"], "--min-p", "0.05"]
        problems = declaration_problems(llm, extraction)
        assert any("--min-p" in problem for problem in problems), problems

    def test_an_undeclared_env_var_is_reported(self) -> None:
        llm, extraction = self._pair()
        llm["environment"] = [*llm["environment"], "LLAMA_ARG_FLASH_ATTN=on"]
        problems = declaration_problems(llm, extraction)
        assert any("LLAMA_ARG_FLASH_ATTN" in problem for problem in problems), problems

    def test_a_non_llama_arg_env_var_is_reported(self) -> None:
        llm, extraction = self._pair()
        llm["environment"] = [*llm["environment"], "GGML_CUDA_FORCE_MMQ=1"]
        problems = declaration_problems(llm, extraction)
        assert any("GGML_CUDA_FORCE_MMQ" in problem for problem in problems), problems

    def test_a_differing_value_is_reported(self) -> None:
        llm, extraction = self._pair()
        llm["environment"] = [
            "LLAMA_ARG_CTX_SIZE=32768" if str(entry).startswith("LLAMA_ARG_CTX_SIZE=") else entry
            for entry in llm["environment"]
        ]
        problems = declaration_problems(llm, extraction)
        assert any("EXTRACTION_SERVER_CTX_SIZE" in problem for problem in problems), problems

    def test_a_stale_declaration_is_reported(self) -> None:
        llm, extraction = self._pair()
        extraction["environment"] = [*extraction["environment"], "EXTRACTION_SERVER_MIN_P=0.05"]
        problems = declaration_problems(llm, extraction)
        assert any("EXTRACTION_SERVER_MIN_P" in problem for problem in problems), problems

    def test_a_differing_image_is_reported(self) -> None:
        llm, extraction = self._pair()
        llm["image"] = "ghcr.io/ggml-org/llama.cpp:server-cuda"
        problems = declaration_problems(llm, extraction)
        assert any("EXTRACTION_SERVER_IMAGE" in problem for problem in problems), problems

    def test_a_device_default_is_reported(self) -> None:
        _, extraction = self._pair()
        extraction["environment"] = [
            (
                "EXTRACTION_SERVER_DEVICE=${EXTRACTION_LLM_DEVICE:-rtx-4070-super}"
                if str(entry).startswith("EXTRACTION_SERVER_DEVICE=")
                else entry
            )
            for entry in extraction["environment"]
        ]
        assert device_problems(extraction)

    def test_a_drifted_model_path_is_reported(self) -> None:
        llm, extraction = self._pair()
        llm["environment"] = [
            (
                "LLAMA_ARG_MODEL=/models/other.gguf"
                if str(entry).startswith("LLAMA_ARG_MODEL=")
                else entry
            )
            for entry in llm["environment"]
        ]
        assert model_path_problems(llm, extraction)


class TestServerSettingExclusions:
    def test_every_exclusion_carries_a_reason(self) -> None:
        for setting, reason in SERVER_SETTING_EXCLUSIONS.items():
            assert reason.strip(), f"{setting} is excluded with no reason"

    def test_every_exclusion_names_a_setting_one_of_the_files_uses(self) -> None:
        with (REPO_ROOT / "docker" / "extraction" / "compose.host.yml").open(
            encoding="utf-8"
        ) as fh:
            host_llm = yaml.safe_load(fh)["services"]["mist-extraction-llm-host"]
        used = set(llama_server_settings(_local_llm())) | set(llama_server_settings(host_llm))
        assert set(SERVER_SETTING_EXCLUSIONS) <= used, set(SERVER_SETTING_EXCLUSIONS) - used

    def test_no_setting_is_both_excluded_and_declared(self) -> None:
        excluded_names = {_declared_name(setting) for setting in SERVER_SETTING_EXCLUSIONS}
        assert excluded_names.isdisjoint(EXPECTED_SERVER_DECLARATIONS)

    def test_the_required_keys_are_in_the_keyed_set(self) -> None:
        names = {key.upper() for key in required_server_keys()}
        assert names == {"IMAGE", "DEVICE", "CTX_SIZE"}
        assert names <= EXPECTED_SERVER_DECLARATIONS


class TestLocalLlmHasNoCudaCache:
    def test_local_llm_does_not_need_a_cuda_cache(self) -> None:
        # sm_89 (4070 SUPER) ships native SASS in this cuda12 image; only the
        # Pascal (sm_61) host needs the PTX-JIT cache volume (see
        # test_compose_extraction_host.py).
        assert "CUDA_CACHE_PATH" not in _env_dict(_local_llm())

    def test_no_top_level_volumes_are_declared(self) -> None:
        # The host file's named volumes (Tailscale state, CUDA JIT cache) belong to
        # the host deployment only -- this local overlay needs none of its own.
        assert not _compose().get("volumes")


def _service_settings_fields_without_defaults() -> list[str]:
    """Parse `ServiceSettings`'s dataclass fields that carry no class-body default.

    AST-based per the T1b brief, rather than importing the module (importing
    `backend.extraction_service.settings` is cheap on its own, but this test
    module stays consistent with the rest of the file's "parse text only"
    discipline and needs no live Python object to answer "does this field
    have a default").
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
        # Guards the AST walk itself: if settings.py changes shape, this
        # fails loudly instead of the real test below silently checking an
        # empty list.
        fields = _service_settings_fields_without_defaults()
        assert fields == ["llm_base_url", "model_hash", "model_file"]

    def test_every_required_field_has_its_env_var_set(self) -> None:
        fields = _service_settings_fields_without_defaults()
        required_env_vars = [f"EXTRACTION_{field.upper()}" for field in fields]
        env = _env_dict(_services()["mist-extraction-local"])
        for var in required_env_vars:
            assert var in env, f"mist-extraction-local does not set {var}"


class TestDockerfile:
    def _text(self) -> str:
        return DOCKERFILE_PATH.read_text(encoding="utf-8")

    def test_declares_first_run_memory_label(self) -> None:
        assert re.search(r"^LABEL\s+claude-worker\.first-run-memory=\S+", self._text(), re.M)

    def test_runs_as_a_non_root_user(self) -> None:
        text = self._text()
        assert re.search(r"^USER\s+appuser\s*$", text, re.M)
        assert "root" not in re.findall(r"^USER\s+(\S+)\s*$", text, re.M)

    def test_entrypoint_runs_the_extraction_service_module(self) -> None:
        assert 'ENTRYPOINT ["python", "-m", "backend.extraction_service"]' in self._text()

    def test_healthcheck_uses_urllib_not_curl(self) -> None:
        text = self._text()
        start = text.index("HEALTHCHECK")
        end = text.index("ENTRYPOINT")
        healthcheck_block = text[start:end]
        assert "urllib" in healthcheck_block
        assert "curl" not in healthcheck_block

    def test_copies_only_backend(self) -> None:
        copy_lines = re.findall(r"^COPY\s+(\S+)\s+", self._text(), re.M)
        # requirements.txt is copied by path (not a directory copy) --
        # everything else copied wholesale must be backend/.
        directory_copies = [line for line in copy_lines if line.endswith("/")]
        assert directory_copies == ["backend/"]


class TestPythonBaseImagePin:
    def test_dockerfile_base_is_pinned_by_digest(self) -> None:
        from_lines = [
            line.split(None, 1)[1].strip()
            for line in DOCKERFILE_PATH.read_text(encoding="utf-8").splitlines()
            if line.upper().startswith("FROM ")
        ]
        assert from_lines == [PYTHON_PINNED_IMAGE]

    def test_digest_ref_pattern_rejects_a_moving_tag(self) -> None:
        assert not DIGEST_REF_PATTERN.match("tailscale/tailscale:stable")
        assert not DIGEST_REF_PATTERN.match("python:3.11-slim")
        assert not DIGEST_REF_PATTERN.match("python@sha256:abc")


# Module (as actually imported by `backend.extraction_service.app` at process
# start, per T1a) -> distribution (as pinned in requirements.txt). Verified
# 2026-09-26 by hooking `builtins.__import__` and importing
# `backend.extraction_service.app` inside the worker container.
#
# neo4j/pandas/pyarrow/numpy/pytz/dateutil/six dropped out of this closure
# 2026-09-26 (goal mist-two-loop v2-lazy-imports, MIS-171) when
# `backend/knowledge/extraction/__init__.py` and
# `backend/knowledge/storage/__init__.py` became PEP 562 lazy -- see
# requirements.txt's own header comment for the full chain that used to pull
# them in despite no extraction code path using them.
MODULE_TO_DISTRIBUTION = {
    "annotated_doc": "annotated-doc",
    "annotated_types": "annotated-types",
    "anyio": "anyio",
    "brotli": "brotli",
    "click": "click",
    "distro": "distro",
    "dotenv": "python-dotenv",
    "fastapi": "fastapi",
    "httpx": "httpx",
    "idna": "idna",
    "openai": "openai",
    "orjson": "orjson",
    "pydantic": "pydantic",
    "pydantic_core": "pydantic-core",
    "pygments": "pygments",
    "python_multipart": "python-multipart",
    "rich": "rich",
    "sniffio": "sniffio",
    "starlette": "starlette",
    "typing_extensions": "typing-extensions",
    "typing_inspection": "typing-inspection",
}

# Distributions that must NEVER be pinned in this file -- their absence from
# the deploy image is the point of the v2-lazy-imports change (see
# requirements.txt's header comment for the full trace of how they used to
# get pulled in as an __init__ side effect, and
# tests/unit/extraction_service/test_import_closure.py for the corresponding
# runtime assertion against sys.modules).
FORBIDDEN_DISTRIBUTIONS = {"neo4j", "pandas", "pyarrow"}


class TestRequirementsCoverTheMeasuredClosure:
    def _lines(self) -> list[str]:
        return REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()

    def _pinned_distributions(self) -> set[str]:
        pinned = set()
        for line in self._lines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            name = re.split(r"[=<>!~]", stripped, maxsplit=1)[0].strip()
            pinned.add(name.lower())
        return pinned

    def test_every_closure_module_has_its_distribution_pinned(self) -> None:
        pinned = self._pinned_distributions()
        for module, distribution in MODULE_TO_DISTRIBUTION.items():
            assert distribution.lower() in pinned, (
                f"module {module!r} (distribution {distribution!r}) is not pinned in "
                f"docker/extraction/requirements.txt"
            )

    def test_every_pinned_distribution_is_exactly_pinned(self) -> None:
        for line in self._lines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            assert "==" in stripped, f"{stripped!r} is not pinned with =="

    def test_uvicorn_is_present(self) -> None:
        assert "uvicorn" in self._pinned_distributions()

    def test_forbidden_distributions_are_absent(self) -> None:
        pinned = self._pinned_distributions()
        present = FORBIDDEN_DISTRIBUTIONS & pinned
        assert not present, (
            f"{present} must not be pinned in docker/extraction/requirements.txt -- "
            "the lazy __init__ change (MIS-171 v2-lazy-imports) removed the only "
            "code path that pulled them in"
        )
