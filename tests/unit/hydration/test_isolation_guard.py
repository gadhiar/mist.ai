"""R1.4.6 T1: the dev/hydration isolation guards refuse live targets.

Two guards, two shapes, and the asymmetry is the point (see the
`eval_isolation` module docstring): a filesystem path has a canonical form, so
a DENYLIST over resolved paths cannot be evaded by respelling; a bolt URI does
not, so only an ALLOWLIST closes it.

The containment case is the one worth reading. `scripts/golden_log/generate.py`
carries a narrower twin of `assert_isolated_root` that checks only "root IS
live" and "root UNDER live". That is sufficient for a generator which creates
two files, and insufficient here, because `snapshot.restore` CLEARS its target
before writing -- a root that CONTAINS `data/` would delete the live event
store on the way in.
"""

from pathlib import Path

import pytest

from backend.knowledge.eval_isolation import (
    EvalIsolationError,
    IsolatedRootError,
    assert_isolated_root,
    assert_neo4j_dev_isolated,
    live_state_roots,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


class TestLiveStateRoots:
    def test_covers_both_host_and_container_spellings(self):
        # The same guard runs on the host and inside mist-backend-dev, and each
        # sees only its own spelling of the stores.
        roots = {str(p) for p in live_state_roots()}
        assert str((REPO_ROOT / "data").resolve()) in roots
        assert str((REPO_ROOT / "mist-memory").resolve()) in roots
        assert str(Path("/app/data").resolve()) in roots
        assert str(Path("/app/mist-memory").resolve()) in roots
        assert str((Path.home() / ".mist").resolve()) in roots

    def test_does_not_filter_on_existence(self):
        # generate.py's twin drops candidates that do not exist, which fails
        # OPEN for a live root absent from the current filesystem view. At most
        # one of the host and container spellings of `data/` exists at a time,
        # so a presence filter could never return all five -- which is what
        # makes the count a real assertion about the filter rather than a
        # restatement of the list.
        assert len(live_state_roots()) == 5


class TestAssertIsolatedRootRefusesLive:
    @pytest.mark.parametrize(
        "candidate",
        [
            pytest.param(REPO_ROOT / "data", id="is-repo-data-dir"),
            pytest.param(REPO_ROOT / "mist-memory", id="is-repo-vault"),
            pytest.param(Path("/app/data"), id="is-container-data-dir"),
            pytest.param(Path.home() / ".mist", id="is-user-store"),
        ],
    )
    def test_refuses_a_root_that_is_a_live_directory(self, candidate):
        with pytest.raises(IsolatedRootError, match="IS the live state directory"):
            assert_isolated_root(candidate)

    @pytest.mark.parametrize(
        "candidate",
        [
            pytest.param(REPO_ROOT / "data" / "dev-state", id="under-repo-data-dir"),
            pytest.param(REPO_ROOT / "mist-memory" / "sessions", id="under-repo-vault"),
            pytest.param(Path("/app/data") / "hydration", id="under-container-data-dir"),
        ],
    )
    def test_refuses_a_root_that_sits_under_a_live_directory(self, candidate):
        with pytest.raises(IsolatedRootError, match="sits under the live state directory"):
            assert_isolated_root(candidate)

    def test_refuses_a_root_that_contains_a_live_directory(self):
        # The arm the golden-log precedent lacks. Restore would delete data/
        # and mist-memory/ on its way in.
        with pytest.raises(IsolatedRootError, match="CONTAINS the live state directory"):
            assert_isolated_root(REPO_ROOT)

    def test_refuses_a_relative_spelling_of_a_live_directory(self):
        # Canonicalization is what lets a denylist work at all: `dev-state/../data`
        # and `data` are the same directory, and `resolve()` collapses them.
        with pytest.raises(IsolatedRootError):
            assert_isolated_root(REPO_ROOT / "dev-state" / ".." / "data")

    def test_refuses_the_home_directory_itself(self):
        with pytest.raises(IsolatedRootError, match="home directory"):
            assert_isolated_root(Path.home())

    def test_refuses_a_filesystem_root(self):
        anchor = Path(REPO_ROOT.anchor)
        with pytest.raises(IsolatedRootError, match="filesystem/drive root"):
            assert_isolated_root(anchor)

    def test_refusal_names_the_purpose_so_the_operator_knows_what_refused(self):
        with pytest.raises(IsolatedRootError, match="hydration restore target"):
            assert_isolated_root(REPO_ROOT / "data", purpose="hydration restore target")


class TestAssertIsolatedRootAcceptsIsolated:
    def test_accepts_the_dev_profile_root(self):
        assert_isolated_root(REPO_ROOT / "dev-state")

    def test_accepts_a_pytest_tmp_path(self, tmp_path):
        assert_isolated_root(tmp_path)

    def test_accepts_a_nested_root_beside_the_live_directories(self):
        # Adjacency is not containment; only the resolved-path relation matters.
        assert_isolated_root(REPO_ROOT / "dev-state" / "alt")


class TestAssertNeo4jDevIsolated:
    @pytest.mark.parametrize(
        "uri",
        [
            pytest.param("bolt://mist-neo4j:7687", id="live-service-name"),
            pytest.param("bolt://localhost:7687", id="live-host-published-port"),
            pytest.param("bolt://127.0.0.1:7687", id="live-loopback"),
            pytest.param("bolt://mist-neo4j-eval:7687", id="eval-instance-is-not-dev"),
            pytest.param("bolt://localhost:7689", id="staging-host-published-port"),
            pytest.param("bolt://mist-neo4j-dev", id="dev-host-without-port"),
            pytest.param("mist-neo4j-dev:7687", id="no-scheme"),
        ],
    )
    def test_refuses_everything_outside_the_dev_allowlist(self, uri):
        with pytest.raises(EvalIsolationError):
            assert_neo4j_dev_isolated(uri)

    @pytest.mark.parametrize(
        "uri",
        [
            pytest.param("bolt://mist-neo4j-dev:7687", id="dev-in-network"),
            pytest.param("bolt://localhost:7690", id="dev-host-published"),
            pytest.param("bolt://127.0.0.1:7690", id="dev-loopback"),
        ],
    )
    def test_allows_dev_endpoints(self, uri):
        assert_neo4j_dev_isolated(uri)

    def test_has_no_activation_flag_to_forget(self, monkeypatch):
        # `assert_neo4j_isolated` no-ops unless MIST_EVAL_ISOLATION is set,
        # because live runtime shares its call site. Nothing legitimately runs
        # hydration tooling against live, so this guard has no off switch --
        # setting or clearing every isolation flag must not change its answer.
        for var in ("MIST_EVAL_ISOLATION", "MIST_HYDRATION_ISOLATION"):
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(EvalIsolationError):
            assert_neo4j_dev_isolated("bolt://mist-neo4j:7687")

        monkeypatch.setenv("MIST_EVAL_ISOLATION", "0")
        monkeypatch.setenv("MIST_HYDRATION_ISOLATION", "0")
        with pytest.raises(EvalIsolationError):
            assert_neo4j_dev_isolated("bolt://mist-neo4j:7687")

    def test_allowlist_is_env_overridable(self, monkeypatch):
        monkeypatch.setenv("MIST_DEV_NEO4J_HOSTS", "ci-neo4j:9999")
        assert_neo4j_dev_isolated("bolt://ci-neo4j:9999")
        with pytest.raises(EvalIsolationError):
            assert_neo4j_dev_isolated("bolt://mist-neo4j-dev:7687")

    @pytest.mark.parametrize(
        "uri",
        [
            pytest.param("bolt://mist-neo4j:7687", id="live-service-name"),
            pytest.param("bolt://localhost:7687", id="live-host-published-port"),
            pytest.param("bolt://127.0.0.1:7687", id="live-loopback"),
        ],
    )
    def test_env_override_cannot_admit_a_live_endpoint(self, monkeypatch, uri):
        """The override REPLACES the allowlist, so it can admit anything -- except live.

        `_parse_endpoint_allowlist` does `os.getenv(env_var, default)`: the
        override replaces the allowlist wholesale rather than extending it, and
        `test_allowlist_is_env_overridable` above pins that. Before the live
        denylist, that made one env var sufficient to point this guard's callers
        at the canonical graph -- and its caller set includes `snapshot restore`,
        which runs `MATCH (n) WITH n LIMIT 10000 DETACH DELETE n`.

        `assert_rebuild_target_not_live` already had a second arm for exactly
        this, and documents why: "an operator who widens
        MIST_REBUILD_NEO4J_HOSTS to include a live endpoint is caught by the
        second arm rather than being handed the wipe by their own override."
        The dev guard gates the MORE destructive tool and had only one arm.
        """
        host_port = uri.removeprefix("bolt://")
        monkeypatch.setenv("MIST_DEV_NEO4J_HOSTS", host_port)

        with pytest.raises(EvalIsolationError, match="live"):
            assert_neo4j_dev_isolated(uri)

    def test_live_denylist_is_not_env_overridable(self, monkeypatch):
        """No environment variable may widen the denylist -- that is its whole point.

        Sweeps every isolation-related variable in the module. A denylist that
        an override can empty is an allowlist wearing a different name.
        """
        for var in (
            "MIST_DEV_NEO4J_HOSTS",
            "MIST_EVAL_NEO4J_HOSTS",
            "MIST_REBUILD_NEO4J_HOSTS",
            "MIST_EVAL_ISOLATION",
            "MIST_HYDRATION_ISOLATION",
        ):
            monkeypatch.setenv(var, "mist-neo4j:7687")

        with pytest.raises(EvalIsolationError, match="live"):
            assert_neo4j_dev_isolated("bolt://mist-neo4j:7687")

    def test_the_dev_endpoint_still_passes_with_the_denylist_in_place(self):
        """Pairing guard: the denylist must not refuse the endpoint this guard exists to admit."""
        assert_neo4j_dev_isolated("bolt://mist-neo4j-dev:7687")

    def test_empty_allowlist_refuses_rather_than_allowing_everything(self, monkeypatch):
        monkeypatch.setenv("MIST_DEV_NEO4J_HOSTS", " , ")
        with pytest.raises(EvalIsolationError, match="empty allowlist"):
            assert_neo4j_dev_isolated("bolt://mist-neo4j-dev:7687")

    def test_malformed_allowlist_entry_refuses(self, monkeypatch):
        monkeypatch.setenv("MIST_DEV_NEO4J_HOSTS", "no-port-here")
        with pytest.raises(EvalIsolationError, match="Malformed MIST_DEV_NEO4J_HOSTS"):
            assert_neo4j_dev_isolated("bolt://mist-neo4j-dev:7687")


class TestEvalGuardStillIntact:
    """The shared allowlist parser must not have changed the eval guard."""

    def test_eval_allowlist_error_still_names_its_own_env_var(self, monkeypatch):
        from backend.knowledge.config import Neo4jConfig
        from backend.knowledge.eval_isolation import assert_neo4j_isolated

        monkeypatch.setenv("MIST_EVAL_ISOLATION", "1")
        monkeypatch.setenv("MIST_EVAL_NEO4J_HOSTS", "no-port-here")
        with pytest.raises(EvalIsolationError, match="Malformed MIST_EVAL_NEO4J_HOSTS"):
            assert_neo4j_isolated(Neo4jConfig(uri="bolt://mist-neo4j-eval:7687"))

    def test_dev_and_eval_allowlists_are_disjoint(self, monkeypatch):
        from backend.knowledge.config import Neo4jConfig
        from backend.knowledge.eval_isolation import assert_neo4j_isolated

        # An eval run must not accept the dev instance and vice versa, or a
        # hydration run and a gauntlet could land in the same graph.
        monkeypatch.setenv("MIST_EVAL_ISOLATION", "1")
        with pytest.raises(EvalIsolationError):
            assert_neo4j_isolated(Neo4jConfig(uri="bolt://mist-neo4j-dev:7687"))
        with pytest.raises(EvalIsolationError):
            assert_neo4j_dev_isolated("bolt://mist-neo4j-eval:7687")
