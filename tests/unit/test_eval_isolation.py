"""Fail-closed Neo4j eval-isolation guard (F1, hardened by the deep review).

foundation-f123-1: the original guard was a single-hostname DENYLIST -- every
alternate spelling of the live endpoint (localhost:7687 via the host-published
bolt port, 127.0.0.1, container IP) passed. The guard is now an ALLOWLIST of
disposable eval endpoints compared on (host, port).

foundation-f123-2: unrecognized MIST_EVAL_ISOLATION values raise instead of
silently deactivating the guard.

mist-daily-instance-build T4: the filesystem guard also refuses a daily
instance's state, which lives outside the repository. Every test here builds a
synthetic instance under `tmp_path` and points `MIST_LIVE_INSTANCES_FILE` at a
registry it writes itself, so nothing reads the host's registry or real home.
"""

from pathlib import Path

import pytest

from backend.knowledge.config import Neo4jConfig
from backend.knowledge.eval_isolation import (
    LIVE_INSTANCE_MARKER,
    LIVE_INSTANCES_FILE_ENV,
    EvalIsolationError,
    IsolatedRootError,
    assert_isolated_root,
    assert_neo4j_isolated,
    assert_not_live_instance,
    is_eval_isolation_active,
    live_state_roots,
)

# The five roots `live_state_roots()` lists with an empty registry. Restated
# here only as a count, so a test can assert what the registry ADDED.
BUILTIN_LIVE_ROOT_COUNT = 5


def write_registry(path: Path, *lines: str) -> Path:
    """Write a live-instance registry holding `lines`, one per line."""
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def make_marked_root(path: Path) -> Path:
    """Create a directory carrying the live-instance marker, as `instance.py init` does."""
    path.mkdir(parents=True)
    (path / LIVE_INSTANCE_MARKER).write_text(
        '{"v": 1, "instance": "daily", "created_at": "2026-10-23T00:00:00Z"}',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def registered_root(tmp_path, monkeypatch) -> Path:
    """An instance root listed in a tmp registry, with NO marker on disk.

    Unmarked on purpose: the tests using it prove the registry alone refuses,
    so a marker cannot be what made them pass.
    """
    root = tmp_path / "instances" / "daily"
    root.mkdir(parents=True)
    registry = write_registry(tmp_path / "live-instances", str(root))
    monkeypatch.setenv(LIVE_INSTANCES_FILE_ENV, str(registry))
    return root


@pytest.fixture
def marked_root(tmp_path, monkeypatch) -> Path:
    """An instance root carrying the marker and absent from the (empty) registry."""
    monkeypatch.setenv(LIVE_INSTANCES_FILE_ENV, str(write_registry(tmp_path / "live-instances")))
    return make_marked_root(tmp_path / "instances" / "daily")


class TestIsEvalIsolationActive:
    def test_inactive_when_unset(self, monkeypatch):
        monkeypatch.delenv("MIST_EVAL_ISOLATION", raising=False)
        assert is_eval_isolation_active() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_active_for_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("MIST_EVAL_ISOLATION", value)
        assert is_eval_isolation_active() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
    def test_inactive_for_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv("MIST_EVAL_ISOLATION", value)
        assert is_eval_isolation_active() is False

    @pytest.mark.parametrize("value", ["on1", "enabled", "Y", "TRUE=1"])
    def test_unrecognized_values_refuse_instead_of_deactivating(self, monkeypatch, value):
        # A fail-closed guard must not treat an operator typo as "off".
        monkeypatch.setenv("MIST_EVAL_ISOLATION", value)
        with pytest.raises(EvalIsolationError, match="Unrecognized"):
            is_eval_isolation_active()


class TestAssertNeo4jIsolated:
    def test_noop_when_isolation_inactive_even_if_live(self, monkeypatch):
        monkeypatch.delenv("MIST_EVAL_ISOLATION", raising=False)
        cfg = Neo4jConfig(uri="bolt://mist-neo4j:7687")
        # Normal live use is unaffected -- must not raise.
        assert_neo4j_isolated(cfg)

    @pytest.mark.parametrize(
        "uri",
        [
            pytest.param("bolt://mist-neo4j:7687", id="live-service-name"),
            pytest.param("bolt://localhost:7687", id="live-host-published-port"),
            pytest.param("bolt://127.0.0.1:7687", id="live-loopback-ip"),
            pytest.param("bolt://some-dns-alias:7687", id="unknown-alias"),
            pytest.param("bolt://mist-neo4j-eval", id="eval-host-without-port"),
            pytest.param("localhost:7687", id="no-scheme"),
        ],
    )
    def test_refuses_everything_outside_the_allowlist(self, monkeypatch, uri):
        monkeypatch.setenv("MIST_EVAL_ISOLATION", "1")
        with pytest.raises(EvalIsolationError):
            assert_neo4j_isolated(Neo4jConfig(uri=uri))

    @pytest.mark.parametrize(
        "uri",
        [
            pytest.param("bolt://mist-neo4j-eval:7687", id="eval-in-network"),
            pytest.param("bolt://localhost:7688", id="eval-host-published"),
            pytest.param("bolt://127.0.0.1:7688", id="eval-loopback"),
        ],
    )
    def test_allows_disposable_eval_endpoints(self, monkeypatch, uri):
        monkeypatch.setenv("MIST_EVAL_ISOLATION", "1")
        assert_neo4j_isolated(Neo4jConfig(uri=uri))

    def test_allowlist_is_env_overridable(self, monkeypatch):
        monkeypatch.setenv("MIST_EVAL_ISOLATION", "1")
        monkeypatch.setenv("MIST_EVAL_NEO4J_HOSTS", "ci-neo4j:9999")
        assert_neo4j_isolated(Neo4jConfig(uri="bolt://ci-neo4j:9999"))
        with pytest.raises(EvalIsolationError):
            assert_neo4j_isolated(Neo4jConfig(uri="bolt://mist-neo4j-eval:7687"))

    def test_malformed_allowlist_entry_refuses(self, monkeypatch):
        monkeypatch.setenv("MIST_EVAL_ISOLATION", "1")
        monkeypatch.setenv("MIST_EVAL_NEO4J_HOSTS", "no-port-here")
        with pytest.raises(EvalIsolationError, match="Malformed"):
            assert_neo4j_isolated(Neo4jConfig(uri="bolt://mist-neo4j-eval:7687"))


class TestConnectGuard:
    def test_connect_refuses_live_in_eval_mode_before_driver(self, monkeypatch):
        # Eval mode + live URI -> refuse at the guard, before any real driver.
        monkeypatch.setenv("MIST_EVAL_ISOLATION", "1")
        from backend.knowledge.storage.neo4j_connection import Neo4jConnection

        conn = Neo4jConnection(Neo4jConfig(uri="bolt://mist-neo4j:7687"))
        with pytest.raises(EvalIsolationError):
            conn.connect()

    def test_connect_refuses_unrecognized_isolation_value(self, monkeypatch):
        monkeypatch.setenv("MIST_EVAL_ISOLATION", "on1")
        from backend.knowledge.storage.neo4j_connection import Neo4jConnection

        conn = Neo4jConnection(Neo4jConfig(uri="bolt://mist-neo4j:7687"))
        with pytest.raises(EvalIsolationError):
            conn.connect()


class TestTheSpellingsSharedWithTheInstanceCli:
    def test_the_marker_and_registry_variable_names_are_pinned(self):
        # `scripts/daily/instance.py` (T2) is stdlib-only and cannot import
        # this module, so it spells both strings itself. A rename here that it
        # does not repeat would leave every marker and registry it writes
        # unread by this guard, which then passes paths it exists to refuse.
        assert LIVE_INSTANCE_MARKER == ".mist-live-instance"
        assert LIVE_INSTANCES_FILE_ENV == "MIST_LIVE_INSTANCES_FILE"


class TestLiveStateRootsReadsTheRegistry:
    def test_a_missing_registry_adds_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setenv(LIVE_INSTANCES_FILE_ENV, str(tmp_path / "never-written"))

        roots = live_state_roots()

        assert len(roots) == BUILTIN_LIVE_ROOT_COUNT

    def test_every_registered_root_is_added_resolved(self, tmp_path, monkeypatch):
        first = tmp_path / "daily"
        second = tmp_path / "rehearsal"
        registry = write_registry(
            tmp_path / "live-instances",
            str(first),
            "",
            "   ",
            str(tmp_path / "elsewhere" / ".." / "rehearsal"),
        )
        monkeypatch.setenv(LIVE_INSTANCES_FILE_ENV, str(registry))

        roots = live_state_roots()

        assert roots[BUILTIN_LIVE_ROOT_COUNT:] == [first.resolve(), second.resolve()]

    def test_the_unset_variable_falls_back_to_the_registry_in_home(self, tmp_path, monkeypatch):
        # HOME is redirected, so "the registry in home" is a tmp file, never the
        # real one. USERPROFILE too, which is what `Path.home()` reads on Windows.
        monkeypatch.delenv(LIVE_INSTANCES_FILE_ENV, raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        (tmp_path / ".mist").mkdir()
        write_registry(tmp_path / ".mist" / "live-instances", str(tmp_path / "daily"))

        roots = live_state_roots()

        assert (tmp_path / "daily").resolve() in roots

    def test_an_empty_variable_falls_back_to_the_registry_in_home(self, tmp_path, monkeypatch):
        # `MIST_LIVE_INSTANCES_FILE=` in a profile yields "", not absence. It
        # must mean "the default registry", never `Path("")`, the current
        # directory. HOME is a tmp directory, never the real one.
        monkeypatch.setenv(LIVE_INSTANCES_FILE_ENV, "")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        (tmp_path / ".mist").mkdir()
        write_registry(tmp_path / ".mist" / "live-instances", str(tmp_path / "daily"))

        roots = live_state_roots()

        assert (tmp_path / "daily").resolve() in roots

    def test_a_relative_line_refuses_and_names_the_file_and_line(self, tmp_path, monkeypatch):
        # Only `instance.py init` and `retire` write the registry, and they
        # write absolute paths. A relative line is a corrupt registry, and
        # resolving it against whatever the current directory happens to be
        # would protect the wrong directory.
        registry = write_registry(
            tmp_path / "live-instances", str(tmp_path / "daily"), "", "relative/daily"
        )
        monkeypatch.setenv(LIVE_INSTANCES_FILE_ENV, str(registry))

        with pytest.raises(IsolatedRootError) as refusal:
            live_state_roots()

        message = str(refusal.value)
        assert str(registry) in message
        assert "line 3" in message
        assert "relative/daily" in message

    def test_an_unreadable_registry_refuses_and_names_the_file(self, tmp_path, monkeypatch):
        # A directory where the file should be: reading it raises an OSError
        # that is not "missing". Fail closed -- a registry that cannot be read
        # may be hiding the live root a caller is about to clear.
        registry = tmp_path / "live-instances"
        registry.mkdir()
        monkeypatch.setenv(LIVE_INSTANCES_FILE_ENV, str(registry))

        with pytest.raises(IsolatedRootError) as refusal:
            live_state_roots()

        assert str(registry) in str(refusal.value)

    def test_an_undecodable_registry_refuses_and_names_the_file(self, tmp_path, monkeypatch):
        registry = tmp_path / "live-instances"
        registry.write_bytes(b"/srv/mist-daily\n\xff\xfe\n")
        monkeypatch.setenv(LIVE_INSTANCES_FILE_ENV, str(registry))

        with pytest.raises(IsolatedRootError) as refusal:
            live_state_roots()

        assert str(registry) in str(refusal.value)

    def test_an_unreadable_registry_makes_the_isolation_guard_refuse_too(
        self, tmp_path, monkeypatch
    ):
        registry = tmp_path / "live-instances"
        registry.mkdir()
        monkeypatch.setenv(LIVE_INSTANCES_FILE_ENV, str(registry))

        with pytest.raises(IsolatedRootError, match="live-instances"):
            assert_isolated_root(tmp_path / "dev-state")


class TestAssertNotLiveInstance:
    @pytest.mark.parametrize(
        "relative",
        [
            pytest.param(".", id="is-registered-root"),
            pytest.param("state/events", id="under-registered-root"),
        ],
    )
    def test_refuses_a_path_at_or_under_a_registered_root(self, registered_root, relative):
        with pytest.raises(IsolatedRootError, match="registered"):
            assert_not_live_instance(registered_root / relative, purpose="test")

    @pytest.mark.parametrize(
        "relative",
        [
            pytest.param(".", id="is-marked-root"),
            pytest.param("mist-memory/sessions", id="under-marked-root"),
        ],
    )
    def test_refuses_a_path_at_or_under_a_marked_directory(self, marked_root, relative):
        with pytest.raises(IsolatedRootError, match=LIVE_INSTANCE_MARKER):
            assert_not_live_instance(marked_root / relative, purpose="test")

    def test_accepts_a_sibling_of_a_marked_directory(self, marked_root):
        assert_not_live_instance(marked_root.parent / "dev-state", purpose="test")

    def test_cannot_see_an_unregistered_marked_root_from_above(self, marked_root):
        # Why the registry exists at all: the marker sits INSIDE the root, so a
        # walk upward from the root's parent never meets it. Pinned so the
        # module docstring's reason stays true.
        assert_not_live_instance(marked_root.parent, purpose="test")

    def test_the_refusal_names_the_purpose(self, marked_root):
        with pytest.raises(IsolatedRootError, match="golden log store"):
            assert_not_live_instance(marked_root, purpose="golden log store")


class TestAssertIsolatedRootRefusesLiveInstances:
    def test_refuses_a_registered_root_as_is(self, registered_root):
        with pytest.raises(IsolatedRootError, match="IS the live state directory"):
            assert_isolated_root(registered_root)

    def test_refuses_a_path_under_a_registered_root(self, registered_root):
        with pytest.raises(IsolatedRootError, match="sits under the live state directory"):
            assert_isolated_root(registered_root / "state")

    def test_refuses_a_path_that_contains_a_registered_root(self, registered_root):
        # The arm only the registry can supply: nothing under the candidate is
        # read, so a marker inside the instance would never be seen from here.
        with pytest.raises(IsolatedRootError, match="CONTAINS the live state directory"):
            assert_isolated_root(registered_root.parent)

    def test_refuses_an_unregistered_marked_root_as_is(self, marked_root):
        with pytest.raises(IsolatedRootError, match=LIVE_INSTANCE_MARKER):
            assert_isolated_root(marked_root)

    def test_refuses_a_path_under_an_unregistered_marked_root(self, marked_root):
        with pytest.raises(IsolatedRootError, match=LIVE_INSTANCE_MARKER):
            assert_isolated_root(marked_root / "state")

    def test_accepts_a_sibling_of_a_marked_root(self, marked_root):
        assert_isolated_root(marked_root.parent / "dev-state")

    def test_accepts_a_sibling_of_a_registered_root(self, registered_root):
        assert_isolated_root(registered_root.parent / "dev-state")
