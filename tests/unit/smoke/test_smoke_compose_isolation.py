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


def _require_short_syntax(entry: object, service: str) -> str:
    """Refuse a non-string volume entry rather than misclassifying it.

    Compose accepts a long mapping syntax (`- type: bind / source: ./data /
    target: /app/data`). `str()` of that dict begins with `{'type'`, which
    starts with neither `.` nor `/`, so the leading-character discriminator
    below would treat it as a NAMED VOLUME: invisible to `_bind_mounts` and
    counted by `_volume_names`. A long-syntax `./data` mount would then pass
    `test_bind_mount_set_is_exactly_the_allowlist` while actually mounting the
    live event store. That is the one known way this module can report a
    containment property it is not checking, so it fails closed here instead.

    Converting the compose file to long syntax is legitimate; doing so
    silently is not. Teach both helpers the mapping form in the same change.
    """
    if not isinstance(entry, str):
        raise AssertionError(
            f"{service} has a non-string volume entry {entry!r}. These helpers "
            "only understand compose SHORT syntax; the long mapping form would "
            "be silently misclassified as a named volume and escape the "
            "forbidden-mount assertions. Extend the helpers before switching."
        )
    return entry


def _bind_mounts(compose: dict, service: str) -> list[tuple[str, str, str]]:
    """Return (host, container, mode) for every BIND mount on `service`.

    Named volumes are excluded: their host side is an identifier, not a path.
    The discriminator is the leading `.` or `/`, which is also how compose
    itself tells the two apart. Non-string entries are refused by
    `_require_short_syntax` rather than falling through that discriminator.
    """
    mounts = []
    for entry in compose["services"][service].get("volumes", []):
        spec = _require_short_syntax(entry, service)
        parts = spec.split(":")
        host = parts[0]
        container = parts[1] if len(parts) > 1 else ""
        mode = parts[2] if len(parts) > 2 else ""
        if host.startswith(".") or host.startswith("/"):
            mounts.append((host, container, mode))
    return mounts


def _volume_names(compose: dict, service: str) -> list[str]:
    """Return the named-volume side of every non-bind mount on `service`."""
    names = []
    for entry in compose["services"][service].get("volumes", []):
        host = _require_short_syntax(entry, service).split(":")[0]
        if not host.startswith(".") and not host.startswith("/"):
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

    def test_long_syntax_mount_is_refused_not_misclassified(self):
        """The helpers fail closed on compose long syntax.

        Without `_require_short_syntax`, a long-syntax bind of the LIVE event
        store stringifies to `{'type': 'bind', ...}`, whose first character is
        neither `.` nor `/`. `_bind_mounts` would skip it and `_volume_names`
        would count it as a named volume, so the forbidden-mount assertions
        would pass with ./data mounted. This test is the standing proof that
        the guard fires; it is the only reason the eight containment
        assertions can be trusted against a future syntax change.
        """
        smuggled = {
            "services": {
                BACKEND_SERVICE: {
                    "volumes": [
                        {"type": "bind", "source": "./data", "target": "/app/data"}
                    ]
                }
            }
        }
        for helper in (_bind_mounts, _volume_names):
            with pytest.raises(AssertionError, match="non-string volume entry"):
                helper(smuggled, BACKEND_SERVICE)

    def test_real_compose_uses_only_short_syntax(self, compose):
        """The guard above is not vacuous on the file actually shipped."""
        entries = compose["services"][BACKEND_SERVICE].get("volumes", [])
        assert entries, "backend service declares no volumes at all"
        assert all(isinstance(e, str) for e in entries), (
            "The shipped compose file now uses long-syntax mounts. Extend "
            "_bind_mounts and _volume_names to parse the mapping form before "
            "relaxing this."
        )
