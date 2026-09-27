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

# The ONLY required (`${VAR:?...}`) env var this file's local services may declare.
# A Tailscale key belongs to the host file only -- this allowlist is what pins that:
# if a future edit adds another required var here (in particular TS_AUTHKEY), this
# test fails rather than silently reintroducing the interpolation defect that made
# `--profile extraction-local` fail with "required variable TS_AUTHKEY is missing".
REQUIRED_VAR_ALLOWLIST = {"EXTRACTION_MODEL_HASH"}

REQUIRED_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+):\?")


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
