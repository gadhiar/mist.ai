"""Inv-A1 guard: a user vault edit performs no graph write.

R1.3 retired GraphRegenerator (Task 6), the class that used to sit between
the sidecar reindex and the read-path cache invalidation on VaultFilewatcher's
vault-edit sequence. Seven `rebuild_*` tests in the deleted
tests/unit/knowledge/curation/test_graph_regenerator.py collectively proved
the guarantee this file now carries forward onto its new subject,
`VaultFilewatcher._do_reindex`: a vault edit performs no graph write.

Fix round 1 (team-lead review): the first version of this file used a
pre-seeded FakeGraphStore that was constructed but never wired into the
watcher, and a signature check that only pinned the name "regenerator". Both
were proven vacuous by mutation. Replaced with an exhaustive constructor
parameter whitelist plus a source-text layering check.

Fix round 2 (team-lead review): the source-text layering check was itself
proven permeable by mutation -- `from backend.knowledge import storage`
contains no contiguous "knowledge.storage" substring, so it passed silently,
and a write reached through `backend.factories.build_graph_store` (this
codebase's own DI entry point, and exactly how `build_filewatcher` used to
reach GraphRegenerator) was invisible to any check that never executes
`_do_reindex` at all. Three checks now:

1. The exhaustive parameter whitelist (unchanged from round 1) -- catches a
   graph-store dependency added directly to the constructor.
2. An AST-based import check (replaces the substring check) -- walks the
   parsed module for Import/ImportFrom nodes and asserts no resulting dotted
   name starts with the two packages that own graph-write machinery. Immune
   to the substring check's false positive (a comment mentioning
   "knowledge.storage" would have failed it) and its false negative (`from
   backend.knowledge import storage` evades a substring match).
3. A parametrized behavioral trap that actually runs `_do_reindex` for a
   users/, sessions/, identity/, and decisions/ path, with
   `backend.knowledge.storage.graph_store.GraphStore` and
   `backend.factories.build_graph_store` monkeypatched to a RECORDING
   double. Recording, not raising: `_do_reindex` wraps its vault-edit
   sequence in `except Exception` (filewatcher.py, watcher-must-survive
   guard), which silently swallows a raising double's exception -- proven by
   the reviewer's fifth mutation, which placed a factories-based write after
   the publish() call and got a fully green suite from a raising trap. A
   recording double's `.append()` never raises, so its effect survives that
   swallow and is observable in the assertion afterward.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path

import pytest

import backend.vault.filewatcher as filewatcher_module
from backend.knowledge.config import FilewatcherConfig
from backend.vault.filewatcher import VaultFilewatcher

# ---------------------------------------------------------------------------
# Platform-availability marker (mirrors tests/unit/test_factories_phase3.py)
# ---------------------------------------------------------------------------
#
# Only the behavioral trap below needs this: monkeypatching
# backend.factories.build_graph_store requires importing backend.factories,
# which eagerly imports EmbeddingGenerator (sentence_transformers,
# Linux/container only).

_SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    import sentence_transformers as _st  # noqa: F401

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

requires_sentence_transformers = pytest.mark.skipif(
    not _SENTENCE_TRANSFORMERS_AVAILABLE,
    reason="sentence_transformers not available on this platform",
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeSidecarIndex:
    """Minimal SidecarIndexProtocol double; only upsert_file is exercised here."""

    def initialize(self) -> None:
        pass

    def close(self) -> None:
        pass

    def upsert_file(
        self,
        path: str,
        content: str,
        mtime: int,
        frontmatter: dict | None = None,
    ) -> int:
        return 1

    def delete_path(self, path: str) -> int:
        return 1

    def query_vector(self, embedding: list[float], k: int = 10) -> list[dict]:
        return []

    def query_fts(self, text: str, k: int = 10) -> list[dict]:
        return []

    def query_hybrid(
        self,
        embedding: list[float],
        text: str,
        k: int = 10,
        rrf_k: int = 60,
    ) -> list[dict]:
        return []

    def chunk_count(self) -> int:
        return 0

    def health_check(self) -> bool:
        return True


class _FakeVaultWriter:
    """Records authored_by writeback calls -- the first vault-edit step."""

    def __init__(self) -> None:
        self.authored_by_calls: list[Path] = []

    async def mark_authored_by_user_edit(self, path: Path) -> None:
        self.authored_by_calls.append(path)


class _FakeInvalidationBus:
    """Records publish calls -- the second and final vault-edit step."""

    def __init__(self) -> None:
        self.published: list[object] = []

    async def publish(self, event: object) -> None:
        self.published.append(event)


class _RecordingGraphStore:
    """Records any attribute access instead of performing a real graph write.

    Any attribute is a bound no-op that appends its own name to `uses` when
    called. Used as the monkeypatched stand-in for both `GraphStore` (the
    class) and `build_graph_store`'s return value, so that however a
    regression reaches a graph store, calling any method on it leaves a
    trace in `uses` -- without ever raising, which is the property that
    matters (see module docstring, fix round 2).
    """

    def __init__(self, uses: list[str]) -> None:
        self._uses = uses

    def __getattr__(self, name: str):
        def _record(*_args: object, **_kwargs: object) -> None:
            self._uses.append(name)

        return _record


def _make_watcher(
    vault_root: Path, writer: _FakeVaultWriter, bus: _FakeInvalidationBus
) -> VaultFilewatcher:
    config = FilewatcherConfig(
        enabled=True,
        observer_type="polling",
        debounce_ms=100,
        staleness_slo_seconds=5,
        audit_interval_seconds=3600,
    )
    return VaultFilewatcher(
        config,
        vault_root,
        _FakeSidecarIndex(),
        invalidation_bus=bus,
        writer=writer,
    )


# ---------------------------------------------------------------------------
# Check 1: exhaustive constructor whitelist
# ---------------------------------------------------------------------------


def test_filewatcher_constructor_accepts_exactly_the_known_dependencies() -> None:
    """No graph-store dependency, of any name, can be added silently.

    An exhaustive whitelist (not a single-name exclusion) so that adding ANY
    new constructor parameter -- `graph_store`, `graph_executor`, `curator`,
    `rebuilder`, or anything else -- fails this test and forces a deliberate
    review, rather than only catching a regression that happens to reuse the
    name `regenerator`.
    """
    params = list(inspect.signature(VaultFilewatcher.__init__).parameters)
    assert params == [
        "self",
        "config",
        "vault_root",
        "sidecar_index",
        "invalidation_bus",
        "writer",
    ]


# ---------------------------------------------------------------------------
# Check 2: AST-based import layering guard
# ---------------------------------------------------------------------------

_FORBIDDEN_IMPORT_PREFIXES = ("backend.knowledge.storage", "backend.knowledge.curation")


def _imported_dotted_names(source: str) -> list[str]:
    """Return every dotted name an import statement in `source` binds to.

    `import X.Y.Z` contributes `"X.Y.Z"`. `from X.Y import Z` contributes
    `"X.Y.Z"` (the import's real target, not whatever local alias it is
    bound to via `as`) so a regression cannot dodge detection by renaming
    what it imports. Walks the full AST, including inside function and
    method bodies, so a lazily-constructed import inside `_do_reindex`
    itself is caught, not just module-level imports.
    """
    tree = ast.parse(source)
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_filewatcher_module_does_not_import_graph_write_layers() -> None:
    """The filewatcher module cannot reach graph-write machinery even indirectly.

    A regression need not go through the constructor: a lazily-constructed
    store inside `_do_reindex` (mirroring how backend/factories.py used to
    lazily construct GraphRegenerator) would pass the exhaustive-parameter
    check above while still writing to the graph. This closes that gap via
    an AST walk rather than a substring search: `from backend.knowledge
    import storage` reaches knowledge.storage just as directly as `import
    backend.knowledge.storage`, but contains no contiguous "knowledge.storage"
    substring, so a text search would miss it (proven by mutation in fix
    round 2) while this dotted-name comparison does not.
    """
    source = inspect.getsource(filewatcher_module)
    for dotted in _imported_dotted_names(source):
        assert not dotted.startswith(_FORBIDDEN_IMPORT_PREFIXES), (
            f"filewatcher.py imports {dotted!r}, which reaches graph-write "
            "machinery; VaultFilewatcher must have no path to a graph write"
        )


# ---------------------------------------------------------------------------
# Check 3: behavioral trap -- actually runs _do_reindex
# ---------------------------------------------------------------------------


@requires_sentence_transformers
@pytest.mark.parametrize(
    "subdir, filename",
    [
        pytest.param("users", "raj.md", id="users-path"),
        pytest.param("sessions", "2026-07-30-test.md", id="sessions-path"),
        pytest.param("identity", "mist.md", id="identity-path"),
        pytest.param("decisions", "ADR-001.md", id="decisions-path"),
    ],
)
def test_do_reindex_never_reaches_a_graph_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, subdir: str, filename: str
) -> None:
    """A full _do_reindex run never touches a graph store, however it might be reached.

    Monkeypatches the two entry points a regression could use to reach a
    real graph write -- direct construction
    (`backend.knowledge.storage.graph_store.GraphStore`) and the DI factory
    (`backend.factories.build_graph_store`, exactly how `build_filewatcher`
    used to reach `GraphRegenerator` before Task 6: `CurationGraphRegenerator
    (graph_store=build_graph_store(config))`) -- with a recording double,
    then runs the real `_do_reindex` for a user-edit path under each of the
    four vault subdirectories the retired GraphRegenerator once dispatched
    on differently.
    """
    # Arrange
    # Import backend.factories BEFORE patching GraphStore: factories.py does
    # `from backend.knowledge.storage.graph_store import GraphStore` and uses
    # `GraphStore | None` as a live type annotation on several factory
    # functions, evaluated once at module-import time. Patching GraphStore
    # to a plain callable first (this test's first import of backend.factories
    # in the process, since it is otherwise unrelated to filewatcher tests)
    # would corrupt that evaluation with a TypeError, breaking every other
    # factory function in the module, not just the one this test cares about.
    import backend.factories  # noqa: F401

    uses: list[str] = []
    recording_store = _RecordingGraphStore(uses)
    monkeypatch.setattr(
        "backend.knowledge.storage.graph_store.GraphStore",
        lambda *args, **kwargs: recording_store,
    )
    monkeypatch.setattr(
        "backend.factories.build_graph_store",
        lambda *args, **kwargs: recording_store,
    )

    vault_root = tmp_path / "vault"
    target_dir = vault_root / subdir
    target_dir.mkdir(parents=True)
    p = target_dir / filename
    p.write_text("---\ntype: mist-session\n---\n\nbody\n", encoding="utf-8")

    writer = _FakeVaultWriter()
    bus = _FakeInvalidationBus()
    fw = _make_watcher(vault_root, writer, bus)

    # Act
    asyncio.run(fw._do_reindex(str(p), is_mist_write=False))

    # Assert
    assert writer.authored_by_calls == [p]
    assert len(bus.published) == 1
    assert uses == []
