"""Containment assertions for `docker-compose.live-path-smoke.yml`.

The live-path smoke stack runs ONE real conversation through the real backend
so the per-turn memory pipeline can be observed end to end. It is authorised
only because it cannot reach Raj's live graph or live vault. That guarantee is
"environment configuration plus mount discipline", which is a property of a
YAML file and therefore drifts silently -- one added bind mount, one
`${VAR:-default}` restored, and the stack is pointed at live state with nothing
to say so.

These tests parse that YAML and re-check the guarantee on every unit run.
`docker compose config` is not available in the test container, so this file is
the standing validation of the compose file, not a supplement to one.

Failure messages here name the HAZARD, not just the mismatch: whoever trips one
in six months will not have the design conversation in front of them.

Note on `tests/unit/conftest.py`: its autouse fixture sets MIST_EVAL_ISOLATION=1
and unsets MIST_EVAL_NEO4J_HOSTS for every unit test. That is a live-write guard
for tests that touch Neo4j; it has no bearing on parsing YAML. The one test here
that sets MIST_EVAL_NEO4J_HOSTS does so through `monkeypatch` on purpose, to run
the compose file's value through the SAME parser the runtime uses.
"""

import re
from pathlib import Path

import pytest
import yaml

from backend.knowledge.eval_isolation import LIVE_NEO4J_ENDPOINTS, _allowed_endpoints

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE_PATH = REPO_ROOT / "docker-compose.live-path-smoke.yml"

BACKEND_SERVICE = "mist-backend-smoke"
NEO4J_SERVICE = "mist-neo4j-smoke"

SCRATCH_CONTAINER_ROOT = "/app/smoke-state"
SCRATCH_HOST_DIR = "./smoke-state"

#: The five store paths the backend writes per turn. Every one must land under
#: the scratch root; any one left pointing at /app/data or /app/mist-memory
#: would write into live state the moment someone also mounted that directory.
STORE_PATH_VARS = (
    "EVENT_STORE_DB_PATH",
    "EVENT_STORE_AUDIO_DIR",
    "VECTOR_STORE_DATA_DIR",
    "MIST_VAULT_ROOT",
    "MIST_SIDECAR_DB_PATH",
)

#: Host directories that must never be bind-mounted into the smoke backend.
#: ./data and ./mist-memory are live state. ./dev-state is not live, but it
#: holds the hydration fixture that `docker-compose.dev-hydration.yml:21-23`
#: records as costing 87 LLM turns to reproduce.
FORBIDDEN_HOST_MOUNTS = {
    "./data": "the LIVE event store, vector store and audio directory",
    "./mist-memory": "the LIVE vault",
    "./dev-state": "the hydration fixture (87 LLM turns to reproduce)",
}

#: Named volumes belonging to the live Neo4j (`docker-compose.yml:180-181`).
FORBIDDEN_VOLUME_NAMES = {"mist-neo4j-data", "mist-neo4j-logs"}

#: Compose files whose published host ports the smoke stack must not reuse.
#: Read from disk rather than restated, so a port change anywhere re-checks here.
SIBLING_COMPOSE_FILES = (
    "docker-compose.yml",
    "docker-compose.eval-neo4j.yml",
    "docker-compose.staging-neo4j.yml",
    "docker-compose.dev-hydration.yml",
)


def _load(path: Path) -> dict:
    """Parse a compose file into a plain dict."""
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def compose() -> dict:
    """The parsed smoke compose file."""
    assert COMPOSE_PATH.exists(), (
        f"{COMPOSE_PATH} is missing. The smoke stack's isolation lives in that "
        "file; without it there is nothing standing between a live-path "
        "experiment and the live graph and vault."
    )
    return _load(COMPOSE_PATH)


@pytest.fixture(scope="module")
def backend_env(compose: dict) -> dict[str, str]:
    """The backend service's `environment` block as a dict.

    Compose accepts a list of `KEY=VALUE` strings or a mapping; this file uses
    the list form, and the helper handles both so a reformat does not silently
    empty every assertion below.
    """
    return _environment_of(compose, BACKEND_SERVICE)


def _environment_of(compose: dict, service: str) -> dict[str, str]:
    raw = compose["services"][service].get("environment", [])
    if isinstance(raw, dict):
        return {str(k): "" if v is None else str(v) for k, v in raw.items()}
    env: dict[str, str] = {}
    for entry in raw:
        key, _, value = str(entry).partition("=")
        env[key] = value
    return env


#: `C:/path` or `D:\path` -- a Windows drive letter, which is a COLON that is
#: not a field separator. This is the native path spelling on the machine this
#: stack runs on, so it is the likely accidental spelling, not an exotic one.
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")


def _require_short_syntax(entry: object, service: str) -> str:
    """Refuse a volume entry these helpers cannot classify, rather than guess.

    Two refusals, each closing a way a live path could be mounted while the
    assertions below still passed.

    NON-STRING: compose's long mapping syntax (`- type: bind / source: ./data /
    target: /app/data`). `str()` of that dict is `{'type': 'bind', ...}`, which
    `_split_volume_spec` cannot parse into a host path.

    INTERPOLATED: a `${VAR}` anywhere in the entry. The host side cannot be
    resolved without the environment the operator will actually run under, so
    no static check can show it misses `./data`. `${MIST_DATA:-./data}` is the
    concrete case: it looks like a named volume to any leading-character test.

    Both fail closed. Legitimately introducing either form is fine; doing it
    without teaching these helpers is what is refused.
    """
    if not isinstance(entry, str):
        raise AssertionError(
            f"{service} has a non-string volume entry {entry!r}. These helpers "
            "parse compose SHORT syntax only; the long mapping form cannot be "
            "resolved to a host path here and would escape the forbidden-mount "
            "assertions. Extend the helpers before switching."
        )
    if "${" in entry:
        raise AssertionError(
            f"{service} has an interpolated volume entry {entry!r}. Its host "
            "side depends on the operator's environment, so this module cannot "
            "show it does not resolve to ./data, ./mist-memory or ./dev-state. "
            "Use a literal path in this file."
        )
    return entry


def _split_volume_spec(spec: str) -> tuple[str, str, str]:
    """Split `host:container[:mode]`, keeping a Windows drive letter attached.

    `"C:/Users/rajga/mist.ai/data:/app/data".split(":")` yields `["C", ...]`,
    which silently renames the live event store to a one-character host. The
    drive prefix is consumed first so the host side survives intact.
    """
    if _WINDOWS_DRIVE.match(spec):
        drive, rest = spec[:2], spec[2:]
        parts = rest.split(":")
        parts[0] = drive + parts[0]
    else:
        parts = spec.split(":")
    host = parts[0]
    container = parts[1] if len(parts) > 1 else ""
    mode = parts[2] if len(parts) > 2 else ""
    return host, container, mode


#: Compose's named-volume grammar: `[a-zA-Z0-9]` then word characters, dots
#: and dashes. Crucially NO `/`, `\`, `:` or `~`. A host side that does not
#: match this is a PATH, so the entry is a bind mount.
#:
#: This is the discriminator because it is a property of THE STRING, needing
#: neither the base compose file nor an environment. Two alternatives were
#: tried and are both wrong:
#:
#:   - LEADING `.` OR `/`: classifies `C:/Users/rajga/mist.ai/data`,
#:     `~/mist.ai/data` and `/c/Users/...` as NAMED VOLUMES. All three bind the
#:     live event store, and all three passed every forbidden-mount assertion
#:     in this module. The Windows drive-letter form is this machine's native
#:     path spelling, so it is the likely accidental input, not an exotic one.
#:
#:   - MEMBERSHIP OF THE TOP-LEVEL `volumes:` KEY: correct for a whole compose
#:     PROJECT, wrong for one overlay file. `mist-hf-cache` and
#:     `mist-torch-cache` are declared at `docker-compose.yml:182-183` and used
#:     by this stack's backend, so a test parsing only this overlay reports
#:     both as binds. Measured, not predicted: it turned
#:     `test_bind_mount_set_is_exactly_the_allowlist`,
#:     `test_scratch_is_the_only_writable_bind_mount` and
#:     `test_named_volumes_are_not_reported_as_binds` red. Do not
#:     reach for it again.
_VOLUME_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")


def _declared_volume_names(compose: dict) -> set[str]:
    """The top-level `volumes:` keys of THIS file.

    Deliberately NOT the classifier -- see `_VOLUME_NAME` for why one overlay
    cannot classify by declaration. Used only to assert that this file
    declares the volumes it is itself responsible for.
    """
    return set((compose.get("volumes") or {}).keys())


def _bind_mounts(compose: dict, service: str) -> list[tuple[str, str, str]]:
    """Return (host, container, mode) for every BIND mount on `service`.

    Fail-safe direction: anything that is not a syntactically valid volume
    NAME is reported as a bind, so an unrecognised entry must then appear in
    the allowlist to pass rather than vanishing from the assertion.

    Agrees with compose's own semantics including the case that looks odd: a
    bare `data:/app/data` IS a named volume to compose, and matches
    `_VOLUME_NAME`, so it is reported as one here too.
    """
    mounts = []
    for entry in compose["services"][service].get("volumes", []):
        spec = _require_short_syntax(entry, service)
        host, container, mode = _split_volume_spec(spec)
        if not _VOLUME_NAME.match(host):
            mounts.append((host, container, mode))
    return mounts


def _volume_names(compose: dict, service: str) -> list[str]:
    """Return the named-volume side of every non-bind mount on `service`.

    Exact complement of `_bind_mounts`: matches `_VOLUME_NAME`.
    """
    names = []
    for entry in compose["services"][service].get("volumes", []):
        host, _, _ = _split_volume_spec(_require_short_syntax(entry, service))
        if _VOLUME_NAME.match(host):
            names.append(host)
    return names


def _published_host_ports(compose: dict) -> set[int]:
    """Every host-side port published by any service in `compose`."""
    ports: set[int] = set()
    for service in (compose.get("services") or {}).values():
        for entry in service.get("ports", []) or []:
            host_side = str(entry).split(":")[0]
            if host_side.isdigit():
                ports.add(int(host_side))
    return ports


class TestStorePathsAreScratchOnly:
    """Every per-turn store writes under the scratch root, and nowhere else."""

    @pytest.mark.parametrize("var", STORE_PATH_VARS)
    def test_store_path_resolves_under_scratch_root(self, backend_env, var):
        value = backend_env.get(var)
        assert value is not None, (
            f"{var} is absent from {BACKEND_SERVICE}'s environment. Absent means "
            f"the backend falls back to its live default -- docker-compose.yml "
            f"maps this variable to /app/data or /app/mist-memory. Set it to a "
            f"path under {SCRATCH_CONTAINER_ROOT}."
        )
        assert value.startswith(SCRATCH_CONTAINER_ROOT + "/"), (
            f"{var}={value!r} is outside {SCRATCH_CONTAINER_ROOT}. This is the "
            f"per-turn write target for a REAL conversation. Outside the scratch "
            f"root it is one stray bind mount away from Raj's live store, and the "
            f"experiment only exists on a stack he can reset."
        )

    @pytest.mark.parametrize("var", STORE_PATH_VARS)
    def test_store_path_is_a_literal_not_an_interpolation(self, backend_env, var):
        value = backend_env[var]
        assert "${" not in value, (
            f"{var}={value!r} interpolates a host-shell variable. The base compose "
            f"reads EVENT_STORE_DB_PATH, MIST_VAULT_ROOT and MIST_SIDECAR_DB_PATH "
            f"from the host shell (docker-compose.yml:39,45,46), so an operator "
            f"exporting one to steer the smoke stack would silently RETARGET THE "
            f"LIVE BACKEND on its next recreate. The rule covers all five paths "
            f"rather than only those three, because which ones are interpolated in "
            f"the base file is not a property this stack should depend on. "
            f"Hardcode the container path."
        )


class TestNeo4jIsolation:
    """The smoke backend must be structurally unable to open the live graph."""

    def test_neo4j_uri_names_the_smoke_instance(self, backend_env):
        uri = backend_env.get("NEO4J_URI")
        assert uri == f"bolt://{NEO4J_SERVICE}:7687", (
            f"NEO4J_URI={uri!r} does not name the disposable smoke instance "
            f"({NEO4J_SERVICE}). docker-compose.yml:23 points the live backend at "
            f"bolt://mist-neo4j:7687, and the smoke stack shares the project "
            f"network, so an unset or inherited URI lands on the canonical graph."
        )

    def test_eval_isolation_is_active(self, backend_env):
        value = backend_env.get("MIST_EVAL_ISOLATION")
        assert value == "1", (
            f"MIST_EVAL_ISOLATION={value!r}. `assert_neo4j_isolated` NO-OPS unless "
            f"this is truthy (backend/knowledge/eval_isolation.py:377-378), and it "
            f"is the first statement of `Neo4jConnection.connect()` "
            f"(backend/knowledge/storage/neo4j_connection.py:40). Without it the "
            f"only thing keeping this backend off the live graph is NEO4J_URI "
            f"being correct -- configuration, not structure."
        )

    def test_eval_allowlist_admits_no_live_endpoint(self, backend_env, monkeypatch):
        """The allowlist REPLACES the default, so it can widen onto live."""
        hosts = backend_env.get("MIST_EVAL_NEO4J_HOSTS")
        assert hosts, (
            "MIST_EVAL_NEO4J_HOSTS is absent. Without it the allowlist falls back "
            "to DEFAULT_EVAL_NEO4J_ENDPOINTS (the EVAL instance, 7688), so the "
            "smoke backend's own Neo4j would be refused and the run would fail "
            "closed rather than run isolated."
        )
        monkeypatch.setenv("MIST_EVAL_NEO4J_HOSTS", hosts)
        allowed = _allowed_endpoints()
        overlap = allowed & LIVE_NEO4J_ENDPOINTS
        assert not overlap, (
            f"MIST_EVAL_NEO4J_HOSTS={hosts!r} admits {sorted(overlap)}, which is in "
            f"LIVE_NEO4J_ENDPOINTS. The live bolt port is host-published, so "
            f"'localhost:7687' and 'mist-neo4j:7687' are THE SAME DATABASE. This "
            f"variable REPLACES the allowlist rather than extending it, so an entry "
            f"here is the one way to hand the smoke backend the canonical graph."
        )

    def test_eval_allowlist_contains_only_smoke_endpoints(self, backend_env, monkeypatch):
        monkeypatch.setenv("MIST_EVAL_NEO4J_HOSTS", backend_env["MIST_EVAL_NEO4J_HOSTS"])
        allowed = _allowed_endpoints()
        expected = {(NEO4J_SERVICE, 7687), ("localhost", 7691), ("127.0.0.1", 7691)}
        assert allowed == expected, (
            f"MIST_EVAL_NEO4J_HOSTS resolves to {sorted(allowed)}, expected "
            f"{sorted(expected)}. Any extra endpoint is a graph this run may write "
            f"to; the eval (7688), staging (7689) and dev (7690) instances all hold "
            f"state someone else depends on."
        )

    def test_smoke_neo4j_publishes_its_own_bolt_port(self, compose):
        ports = [str(p) for p in compose["services"][NEO4J_SERVICE]["ports"]]
        assert "7691:7687" in ports, (
            f"{NEO4J_SERVICE} does not publish bolt on host 7691, but "
            f"MIST_EVAL_NEO4J_HOSTS allows localhost:7691. A mismatch means the "
            f"allowlist entry names a port that is either nothing or, worse, some "
            f"other stack's database."
        )


class TestHydrationVarsAbsent:
    """Neither hydration flag may appear -- each one silently breaks the run."""

    def test_hydration_isolation_absent(self, backend_env):
        assert "MIST_HYDRATION_ISOLATION" not in backend_env, (
            "MIST_HYDRATION_ISOLATION is set on the smoke backend. "
            "`CurationScheduler.start()` returns False whenever it is set "
            "(backend/knowledge/curation/scheduler.py:260), and that check runs "
            "AHEAD of the enabled-flag check at scheduler.py:271 -- so the "
            "scheduler never starts and the curation run with examined > 0 that "
            "this experiment must observe cannot happen. It also forces "
            "`include_internal_derivation` to False (backend/factories.py:364-365). "
            "It buys no store isolation: none of its five readers rewrites a store "
            "path. The store isolation is the literal paths plus the absent mounts."
        )

    def test_hydration_clock_absent(self, backend_env):
        assert "MIST_HYDRATION_CLOCK" not in backend_env, (
            "MIST_HYDRATION_CLOCK is set on the smoke backend. It makes "
            "`_record_turn_event` stamp turns with the corpus's AUTHORED "
            "timestamps (2025-09..2026-07) instead of the wall clock. "
            "`SelfReflectionJob` looks back 24 hours from now "
            "(backend/knowledge/curation/self_reflection.py:60-61) through "
            "`get_turns_since`, which filters `WHERE timestamp >= ?` "
            "(backend/event_store/store.py:303-322). Authored-stamped turns fall "
            "outside that window, so the job would report examined = 0 ON A "
            "WORKING PIPELINE -- the exact symptom this experiment is "
            "investigating, manufactured. (It also raises HydrationClockError "
            "unless MIST_HYDRATION_ISOLATION is set too, backend/factories.py:120-126.)"
        )


class TestMountDiscipline:
    """Isolation is the absent mounts first. These are the absences."""

    @pytest.mark.parametrize("forbidden,what", sorted(FORBIDDEN_HOST_MOUNTS.items()))
    def test_forbidden_host_directory_is_not_mounted(self, compose, forbidden, what):
        for service in (BACKEND_SERVICE, NEO4J_SERVICE):
            hosts = {host for host, _, _ in _bind_mounts(compose, service)}
            offending = {
                h for h in hosts if h == forbidden or h.startswith(forbidden.rstrip("/") + "/")
            }
            assert not offending, (
                f"{service} bind-mounts {sorted(offending)}, under {forbidden} -- "
                f"{what}. The smoke stack's containment is that these directories "
                f"are NOT VISIBLE to it, so a misconfigured store path writes into "
                f"the container's throwaway layer instead of onto real state. "
                f"Mounting a subdirectory does not preserve that; it removes it for "
                f"whatever is inside."
            )

    def test_bind_mount_set_is_exactly_the_allowlist(self, compose):
        """Closes the spelling gap the per-directory tests leave open.

        `test_forbidden_host_directory_is_not_mounted` matches host sides as
        strings, so an ABSOLUTE spelling of the same directory
        (/home/raj/mist.ai/data rather than ./data) would pass it. An exhaustive
        allowlist does not care how a new mount is spelled.
        """
        expected = {
            SCRATCH_HOST_DIR,
            "./backend",
            "./src",
            "./dependencies",
            "./scripts",
            "./tests",
            "./voice_profiles",
        }
        actual = {host for host, _, _ in _bind_mounts(compose, BACKEND_SERVICE)}
        assert actual == expected, (
            f"{BACKEND_SERVICE}'s bind mounts are {sorted(actual)}; the allowlist "
            f"is {sorted(expected)}. Every host directory mounted here is one the "
            f"smoke run can see, and the whole containment argument is that the "
            f"live stores and the hydration fixture are NOT among them. Adding a "
            f"mount is a containment decision -- make it here, deliberately, "
            f"rather than by editing the compose file alone."
        )

    def test_no_live_neo4j_volume_is_referenced(self, compose):
        referenced = set()
        for service in compose.get("services", {}):
            referenced.update(_volume_names(compose, service))
        referenced.update((compose.get("volumes") or {}).keys())
        overlap = referenced & FORBIDDEN_VOLUME_NAMES
        assert not overlap, (
            f"This compose file references {sorted(overlap)}, the LIVE Neo4j's "
            f"named volumes (docker-compose.yml:180-181). Attaching one would put "
            f"the smoke Neo4j's writes into the canonical graph's data directory "
            f"-- two Neo4j instances over one store, which is corruption rather "
            f"than sharing."
        )

    def test_scratch_is_the_only_writable_bind_mount(self, compose):
        writable = [
            (host, container)
            for host, container, mode in _bind_mounts(compose, BACKEND_SERVICE)
            if mode != "ro"
        ]
        assert writable == [(SCRATCH_HOST_DIR, SCRATCH_CONTAINER_ROOT)], (
            f"{BACKEND_SERVICE}'s writable bind mounts are {writable}; the only "
            f"permitted one is {(SCRATCH_HOST_DIR, SCRATCH_CONTAINER_ROOT)}. Each "
            f"extra writable mount is another host directory a real conversation "
            f"can write into, and the blast-radius argument that authorised this "
            f"experiment is exactly 'there is only one, and `rm -rf smoke-state` "
            f"undoes it'."
        )

    def test_code_mounts_are_read_only(self, compose):
        expected_ro = {
            "./backend",
            "./src",
            "./dependencies",
            "./scripts",
            "./tests",
            "./voice_profiles",
        }
        modes = {host: mode for host, _, mode in _bind_mounts(compose, BACKEND_SERVICE)}
        missing = sorted(expected_ro - modes.keys())
        assert not missing, (
            f"{BACKEND_SERVICE} does not mount {missing}. The backend runs from "
            f"the image copy without them, so the smoke run would not exercise the "
            f"working tree the experiment is about."
        )
        offenders = sorted(h for h in expected_ro if modes[h] != "ro")
        assert not offenders, (
            f"{BACKEND_SERVICE} mounts {offenders} writable. The live stack runs "
            f"from this same tree (docker-compose.override.yml:7-12), so a "
            f"writable code mount lets a throwaway container edit the source the "
            f"live backend is executing. Append ':ro'."
        )

    def test_scratch_root_is_not_live_state(self):
        """The scratch root must satisfy the same guard hydration roots do."""
        from backend.knowledge.eval_isolation import assert_isolated_root

        assert_isolated_root(REPO_ROOT / "smoke-state", purpose="live-path smoke")


class TestKnowledgeIntegrationOn:
    """Without knowledge integration the conversation records no turn at all."""

    def test_knowledge_integration_enabled(self, backend_env):
        value = backend_env.get("ENABLE_KNOWLEDGE_INTEGRATION")
        assert value == "true", (
            f"ENABLE_KNOWLEDGE_INTEGRATION={value!r}. At "
            f"backend/voice_models/model_manager.py:494 `generate_llm_response` "
            f"takes the knowledge-augmented path only when knowledge is enabled; "
            f"the else branch (model_manager.py:501-530) calls the LLM provider "
            f"directly. Turn recording lives only on the knowledge path "
            f"(`_record_turn_event`, backend/chat/conversation_handler.py:2279). "
            f"With this off the smoke run produces a reply, no turn row, and four "
            f"absent artifacts that look like a broken pipeline."
        )


class TestSchedulerTwoPhaseWiring:
    """The one interpolated variable, and the ordering it exists to serve."""

    def test_scheduler_is_driven_by_the_dedicated_smoke_variable(self, backend_env):
        value = backend_env.get("MIST_CURATION_SCHEDULER_ENABLED")
        assert value == "${MIST_SMOKE_SCHEDULER:-0}", (
            f"MIST_CURATION_SCHEDULER_ENABLED={value!r}. The two-phase protocol "
            f"needs the scheduler OFF during the conversation and ON afterwards: "
            f"`last_run.get(name, 0.0)` makes every enabled job due on the loop's "
            f"first pass (backend/knowledge/curation/scheduler.py:305-311), and "
            f"`self_reflection` is registered with interval_seconds=86400 "
            f"(backend/factories.py:751), so a scheduler that starts before any "
            f"turn exists spends its single daily run on an empty table. The "
            f"default must be the string '0', not empty: "
            f"`curation_scheduler_enabled()` treats '' as UNSET and returns True "
            f"(scheduler.py:85-91)."
        )

    def test_no_other_service_reads_the_smoke_scheduler_variable(self):
        """`MIST_SMOKE_SCHEDULER` is new, which is what makes interpolating it safe.

        The literals rule exists because the LIVE backend reads the same variable
        names from the host shell. A name nothing else reads carries no such
        hazard -- but only while that stays true.

        Scope: the repo-root `docker-compose*.yml` files, which are where every
        service definition in this repository lives.
        """
        hits = []
        for path in REPO_ROOT.glob("docker-compose*.yml"):
            if "MIST_SMOKE_SCHEDULER" in path.read_text(encoding="utf-8"):
                hits.append(path.name)
        assert hits == [COMPOSE_PATH.name], (
            f"MIST_SMOKE_SCHEDULER appears in {sorted(hits)}. It is interpolated "
            f"from the host shell as the single exception to this file's "
            f"literals-only rule, and that exception is only safe because NO OTHER "
            f"service reads the name. A second reader reintroduces exactly the "
            f"hazard the rule prevents: one export steering two stacks."
        )


class TestPortsDoNotCollide:
    """Five instances on one host; a shared port is a shared database."""

    def test_smoke_ports_are_free_against_every_other_stack(self, compose):
        smoke_ports = _published_host_ports(compose)
        assert smoke_ports == {7478, 7691, 8003}, (
            f"Smoke publishes {sorted(smoke_ports)}; the allocated set is "
            f"{{7478, 7691, 8003}}. MIST_EVAL_NEO4J_HOSTS and the client's "
            f"ws://localhost:8003/ws are written against those numbers."
        )
        for name in SIBLING_COMPOSE_FILES:
            path = REPO_ROOT / name
            if not path.exists():
                continue
            other = _published_host_ports(_load(path))
            overlap = smoke_ports & other
            assert not overlap, (
                f"Smoke publishes {sorted(overlap)}, already published by {name}. "
                f"A shared host port means a host-side client -- the smoke "
                f"conversation driver, or an operator running cypher-shell -- "
                f"cannot tell the two stacks apart, and whichever container bound "
                f"first silently receives the traffic. That is how a smoke run "
                f"reaches a database it was never meant to see."
            )


class TestHarnessShape:
    """Properties that keep the stack throwaway rather than service-shaped."""

    def test_both_services_are_profile_gated(self, compose):
        for service in (BACKEND_SERVICE, NEO4J_SERVICE):
            profiles = compose["services"][service].get("profiles")
            assert profiles == ["smoke"], (
                f"{service} has profiles={profiles!r}, expected ['smoke']. Without "
                f"the gate a bare `docker compose up -d` starts this stack "
                f"alongside live, competing for 12GB of VRAM and the model."
            )

    def test_neither_service_restarts(self, compose):
        for service in (BACKEND_SERVICE, NEO4J_SERVICE):
            assert compose["services"][service].get("restart") == "no", (
                f"{service} is not `restart: \"no\"`. A run that dies "
                f"mid-conversation must stay dead: a silent restart leaves a "
                f"half-recorded turn sequence behind assertions that then measure "
                f"an artifact count nobody can interpret."
            )

    def test_backend_reuses_the_live_image_and_declares_no_build(self, compose):
        service = compose["services"][BACKEND_SERVICE]
        assert "build" not in service, (
            f"{BACKEND_SERVICE} declares a build. The point of this run is to "
            f"exercise the binary production runs; a separately-built image is a "
            f"second artifact that can drift from it, and the image is ~62GB."
        )
        assert service.get("image") == "${MIST_BACKEND_IMAGE:-mistai-mist-backend:latest}", (
            f"{BACKEND_SERVICE} image is {service.get('image')!r}; it must reuse "
            f"the live backend tag so the smoke run and production share one "
            f"binary by construction."
        )

    def test_backend_waits_for_its_own_neo4j_to_be_healthy(self, compose):
        depends = compose["services"][BACKEND_SERVICE].get("depends_on", {})
        assert depends.get(NEO4J_SERVICE, {}).get("condition") == "service_healthy", (
            f"{BACKEND_SERVICE} does not wait for {NEO4J_SERVICE} to be healthy. "
            f"A backend that starts first fails its first graph write, and a smoke "
            f"run whose artifacts are missing for a startup-race reason is "
            f"indistinguishable from the pipeline defect under investigation."
        )

    def test_both_services_have_healthchecks(self, compose):
        for service in (BACKEND_SERVICE, NEO4J_SERVICE):
            assert "healthcheck" in compose["services"][service], (
                f"{service} has no healthcheck, so `condition: service_healthy` "
                f"cannot be satisfied and the phase-2 recreate has nothing to "
                f"gate on."
            )

    def test_backend_runs_as_uid_1000(self, compose):
        user = compose["services"][BACKEND_SERVICE].get("user")
        assert user == "1000:1000", (
            f"{BACKEND_SERVICE} runs as {user!r}. docker/backend/Dockerfile:114 "
            f"chowns /app to appuser (uid 1000); any other uid cannot write "
            f"./smoke-state through the bind mount."
        )

    def test_unparseable_mount_forms_are_refused(self):
        """The two forms the helpers cannot resolve fail closed.

        Long mapping syntax and `${VAR}` interpolation both bind a real host
        path that no static check here can pin down.
        """
        cases = [
            ({"type": "bind", "source": "./data", "target": "/app/data"}, "non-string"),
            ("${MIST_DATA:-./data}:/app/data", "interpolated"),
        ]
        for entry, expected in cases:
            smuggled = {
                "volumes": {"mist-hf-cache": None},
                "services": {BACKEND_SERVICE: {"volumes": [entry]}},
            }
            for helper in (_bind_mounts, _volume_names):
                with pytest.raises(AssertionError, match=expected):
                    helper(smuggled, BACKEND_SERVICE)

    def test_live_paths_that_defeat_a_spelling_test_are_classified_as_binds(self):
        """Spellings of the LIVE event store that must register as BINDS.

        Each is a plain string, so `_require_short_syntax` passes it; only the
        classifier stands between it and a silently-mounted live event store.

        MEASURED against the pre-fix helper at `14ca49d`, by importing it from
        git rather than reasoning about it -- the first two ESCAPED, the third
        did not:

            'C:/Users/rajga/mist.ai/data'   OLD binds=[]  ESCAPED
            '~/mist.ai/data'                OLD binds=[]  ESCAPED
            '/c/Users/rajga/mist.ai/data'   OLD           CAUGHT

        So this is a regression test for two real escapes plus one
        non-regression case, NOT three escapes. The Windows drive-letter form
        is this machine's native spelling and is the one that matters; `/c/...`
        is kept because it is the Git-Bash rewriting of the same path and must
        not break when the discriminator changes.
        """
        smuggled_hosts = [
            "C:/Users/rajga/mist.ai/data",
            "~/mist.ai/data",
            "/c/Users/rajga/mist.ai/data",
        ]
        for host in smuggled_hosts:
            compose = {
                "volumes": {"mist-hf-cache": None},
                "services": {
                    BACKEND_SERVICE: {
                        "volumes": [f"{host}:/app/data", "mist-hf-cache:/cache"]
                    }
                },
            }
            binds = _bind_mounts(compose, BACKEND_SERVICE)
            names = _volume_names(compose, BACKEND_SERVICE)
            assert (host, "/app/data", "") in binds, (
                f"{host!r} binds the live event store but was not reported as a "
                f"bind mount; got {binds!r}. It would escape every "
                f"forbidden-mount assertion in this module."
            )
            assert host not in names, (
                f"{host!r} was counted as a NAMED VOLUME; got {names!r}."
            )
            assert names == ["mist-hf-cache"], (
                f"the genuinely named volume was misclassified; got {names!r}"
            )

    def test_named_volumes_are_not_reported_as_binds(self, compose):
        """The complement holds on the shipped file: no false positives.

        Without this, a discriminator that called EVERYTHING a bind would pass
        the test above while making the allowlist assertion fail for the wrong
        reason -- which is exactly what a declaration-based attempt did.

        `mist-hf-cache` is the load-bearing case: it is declared in the BASE
        file (`docker-compose.yml:182`), not in this overlay, so any classifier
        that consults only this file's top-level `volumes:` key misreports it.
        """
        names = _volume_names(compose, BACKEND_SERVICE)
        assert "mist-hf-cache" in names, (
            f"the shared HF cache is a named volume declared in the base "
            f"compose file, but was reported as a bind; got {names!r}"
        )
        bind_hosts = [h for h, _, _ in _bind_mounts(compose, BACKEND_SERVICE)]
        assert not set(bind_hosts) & set(names), (
            f"these hosts were reported as BOTH bind and named: "
            f"{sorted(set(bind_hosts) & set(names))}"
        )

    def test_this_overlay_declares_the_volumes_it_owns(self, compose):
        """The smoke Neo4j volumes are this file's responsibility to declare."""
        declared = _declared_volume_names(compose)
        for name in ("mist-neo4j-smoke-data", "mist-neo4j-smoke-logs"):
            assert name in declared, (
                f"{name} is used by this stack but not declared in its own "
                f"top-level volumes:; got {sorted(declared)}"
            )

    def test_no_named_volume_is_bind_backed_via_driver_opts(self, compose):
        """A named volume can BE a bind, and the name grammar cannot see it.

        `_VOLUME_NAME` classifies by the host side of the mount entry, so a
        declaration like

            volumes:
              innocent-name:
                driver_opts: {type: none, device: ./data, o: bind}

        is reported as a NAMED VOLUME by every helper in this module while
        docker bind-mounts the live event store into the container. The mount
        entry reads `innocent-name:/app/data` and is indistinguishable from a
        real named volume at the point the other tests look.

        This is the last known member of the class the grammar fix closed --
        found by review, not by the fix -- so it is checked where it is
        actually visible: the top-level declaration, not the mount entry.

        Any `driver_opts` at all is refused rather than just `o: bind`. A
        `device:` with `type: none` is the bind spelling, but this stack has no
        legitimate use for driver_opts of any kind, and an allowlist of safe
        options is a thing to get wrong later.
        """
        volumes = compose.get("volumes") or {}
        for name, spec in volumes.items():
            if not isinstance(spec, dict):
                continue
            assert "driver_opts" not in spec, (
                f"top-level volume {name!r} declares driver_opts "
                f"{spec.get('driver_opts')!r}. A driver_opts volume can be "
                f"bind-backed, so it would mount a host path while every "
                f"mount-entry assertion in this module still passes. Nothing "
                f"in this stack needs driver_opts."
            )

    def test_the_driver_opts_guard_is_not_vacuous(self):
        """The guard above fires on the exact evasion it exists to stop."""
        smuggled = {
            "volumes": {
                "innocent-name": {
                    "driver_opts": {"type": "none", "device": "./data", "o": "bind"}
                }
            },
            "services": {
                BACKEND_SERVICE: {"volumes": ["innocent-name:/app/data"]}
            },
        }
        # It passes the mount-entry helpers, which is the whole problem.
        assert _volume_names(smuggled, BACKEND_SERVICE) == ["innocent-name"]
        assert _bind_mounts(smuggled, BACKEND_SERVICE) == []
        # The declaration check is what catches it.
        with pytest.raises(AssertionError, match="driver_opts"):
            TestHarnessShape().test_no_named_volume_is_bind_backed_via_driver_opts(
                smuggled
            )
