"""Tests for the extraction service deploy artifacts (T1b, goal mist-two-loop / MIS-171).

Hermetic: parses `docker-compose.extraction.yml`, `docker/extraction/Dockerfile`,
`docker/extraction/requirements.txt`, and `backend/extraction_service/settings.py`
as text/YAML/AST only. No docker, no network, no import of `backend.extraction_service`
itself (its own import closure pulls in torch-adjacent packages this test tier does not
need to require).

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

# Resolved on the host by the lead on 2026-09-26 (tags kept in the file comments).
TAILSCALE_PINNED_IMAGE = (
    "tailscale/tailscale@sha256:c507f3a2a6ab1cabd8d809b98edeb41edbd5c3fb6ad9632ffd098b4c7d0b4065"
)
PYTHON_PINNED_IMAGE = (
    "python@sha256:e41613d42d4891e4930f79523f93f81bbc7632584ec65e36ab055f41a800b41e"
)
DIGEST_REF_PATTERN = re.compile(r"^[a-z0-9./_-]+@sha256:[0-9a-f]{64}$")

LOCAL_PROFILE = "extraction-local"
HOST_PROFILE = "extraction-host"
VALID_PROFILES = {LOCAL_PROFILE, HOST_PROFILE}


def _compose() -> dict:
    with COMPOSE_PATH.open(encoding="utf-8") as fh:
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


def _llm_services() -> dict[str, dict]:
    services = _services()
    return {
        "mist-extraction-llm-local": services["mist-extraction-llm-local"],
        "mist-extraction-llm-host": services["mist-extraction-llm-host"],
    }


class TestLlamaServerPins:
    def test_both_llm_services_use_the_mist_llm_digest(self) -> None:
        for name, service in _llm_services().items():
            assert service["image"] == PINNED_IMAGE, f"{name} image is not the pinned build"

    def test_both_llm_services_match_the_digest_pattern(self) -> None:
        for name, service in _llm_services().items():
            assert DIGEST_PATTERN.match(service["image"]), f"{name} image is not digest-pinned"

    def test_neither_llm_service_passes_reasoning_budget(self) -> None:
        for name, service in _llm_services().items():
            assert "--reasoning-budget" not in _command(service), (
                f"{name} must not pass --reasoning-budget: the service sends a "
                "per-request budget, and the server's own default (-1, unrestricted) "
                "already matches 'no cap'"
            )

    def test_both_llm_services_pass_cache_ram(self) -> None:
        for name, service in _llm_services().items():
            command = _command(service)
            assert "--cache-ram" in command, f"{name} is missing --cache-ram"
            value = command[command.index("--cache-ram") + 1]
            assert value, f"{name}'s --cache-ram has no value"

    def test_ncmoe_is_parameterized_per_profile(self) -> None:
        local = _command(_services()["mist-extraction-llm-local"])
        host = _command(_services()["mist-extraction-llm-host"])
        assert "-ncmoe" in local
        assert "EXTRACTION_LOCAL_NCMOE" in local[local.index("-ncmoe") + 1]
        assert "-ncmoe" in host
        assert "EXTRACTION_HOST_NCMOE" in host[host.index("-ncmoe") + 1]


class TestProfiles:
    def test_every_service_has_exactly_one_known_profile(self) -> None:
        for name, service in _services().items():
            profiles = service.get("profiles")
            assert profiles is not None, f"{name} has no `profiles` key"
            assert len(profiles) == 1, f"{name} must be in exactly one profile, got {profiles!r}"
            assert profiles[0] in VALID_PROFILES, f"{name} has an unknown profile {profiles!r}"

    def test_both_profiles_are_used_by_at_least_one_service(self) -> None:
        used = {service["profiles"][0] for service in _services().values()}
        assert used == VALID_PROFILES


class TestHostProfilePublishesNoPorts:
    def test_no_host_profile_service_publishes_ports(self) -> None:
        for name, service in _services().items():
            if service.get("profiles") == [HOST_PROFILE]:
                assert "ports" not in service, f"{name} (host profile) must not publish ports"

    def test_extraction_service_shares_the_tailscale_sidecars_network(self) -> None:
        sidecar_name = next(
            name for name, svc in _services().items() if "tailscale" in svc.get("image", "")
        )
        host_extraction = _services()["mist-extraction-host"]
        assert host_extraction.get("network_mode") == f"service:{sidecar_name}"


class TestTailscaleAuthKey:
    def test_ts_authkey_has_no_default(self) -> None:
        sidecar = next(svc for svc in _services().values() if "tailscale" in svc.get("image", ""))
        value = _env_dict(sidecar)["TS_AUTHKEY"]
        assert re.search(
            r"\$\{TS_AUTHKEY:\?", value
        ), f"TS_AUTHKEY must use the required ${{VAR:?msg}} form, got {value!r}"
        assert ":-" not in value, f"TS_AUTHKEY must not have a default, got {value!r}"

    def test_tailscale_image_is_pinned_by_digest(self) -> None:
        """The sidecar image is the digest the lead resolved on the host, not a moving tag."""
        sidecar = next(svc for svc in _services().values() if "tailscale" in svc.get("image", ""))
        assert sidecar["image"] == TAILSCALE_PINNED_IMAGE
        assert DIGEST_REF_PATTERN.match(sidecar["image"])


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


class TestModelHashRequired:
    def test_model_hash_uses_the_required_form_in_both_profiles(self) -> None:
        for name in ("mist-extraction-local", "mist-extraction-host"):
            value = _env_dict(_services()[name])["EXTRACTION_MODEL_HASH"]
            assert re.search(
                r"\$\{EXTRACTION_MODEL_HASH:\?", value
            ), f"{name}'s EXTRACTION_MODEL_HASH must use ${{VAR:?msg}}, got {value!r}"
            assert ":-" not in value, f"{name}'s EXTRACTION_MODEL_HASH must not have a default"


class TestHostCudaCache:
    def test_host_llm_sets_cuda_cache_path_on_a_named_volume(self) -> None:
        compose = _compose()
        top_level_volumes = set(compose.get("volumes") or {})
        host_llm = compose["services"]["mist-extraction-llm-host"]
        cache_path = _env_dict(host_llm).get("CUDA_CACHE_PATH")
        assert cache_path, "mist-extraction-llm-host must set CUDA_CACHE_PATH"

        mounted_on_named_volume = any(
            entry.split(":", 1)[1] == cache_path and entry.split(":", 1)[0] in top_level_volumes
            for entry in host_llm.get("volumes", [])
            if ":" in entry
        )
        assert mounted_on_named_volume, (
            f"CUDA_CACHE_PATH={cache_path!r} must be mounted on one of the compose file's "
            f"top-level named volumes {top_level_volumes!r}"
        )

    def test_local_llm_does_not_need_a_cuda_cache(self) -> None:
        # sm_89 (4070 SUPER) ships native SASS in this cuda12 image; only the
        # Pascal (sm_61) host needs the PTX-JIT cache volume.
        local_llm = _services()["mist-extraction-llm-local"]
        assert "CUDA_CACHE_PATH" not in _env_dict(local_llm)


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

    def test_every_required_field_has_its_env_var_set_in_both_profiles(self) -> None:
        fields = _service_settings_fields_without_defaults()
        required_env_vars = [f"EXTRACTION_{field.upper()}" for field in fields]
        for service_name in ("mist-extraction-local", "mist-extraction-host"):
            env = _env_dict(_services()[service_name])
            for var in required_env_vars:
                assert var in env, f"{service_name} does not set {var}"


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
