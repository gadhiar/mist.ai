"""Inv-A1 guard: a user vault edit performs no graph write.

R1.3 retired GraphRegenerator (Task 6), the class that used to sit between
the sidecar reindex and the read-path cache invalidation on VaultFilewatcher's
vault-edit sequence. Seven `rebuild_*` tests in the deleted
tests/unit/knowledge/curation/test_graph_regenerator.py collectively proved
the guarantee this file now carries forward onto its new subject,
`VaultFilewatcher._do_reindex`: a vault edit performs no graph write.

Fix round 1: replaced a vacuous FakeGraphStore-never-wired-in test and a
single-name ("regenerator") signature check with an exhaustive constructor
parameter whitelist plus a source-text layering check.

Fix round 2: the substring layering check missed `from backend.knowledge
import storage` (no contiguous "knowledge.storage" substring) and, because
round 1 deleted the only tests that ran `_do_reindex` at all, missed a write
reached through `backend.factories.build_graph_store`. Replaced the
substring check with an AST-based dotted-import-name walk and added a
parametrized behavioral trap that runs the real `_do_reindex`.

Fix round 3 (team-lead review, 10 mutation forms, 2 still open): two more
holes.

- C6: `from backend.factories import GraphStore` -- `factories.py` re-exports
  the class at import time. Neither the AST check (the dotted name
  `backend.factories.GraphStore` matches no forbidden prefix) nor the round-2
  trap (which patched a *module attribute binding*, not the class itself)
  caught it, because `backend.factories.GraphStore` was bound to the real
  class object before any patch ran. Fix: patch `GraphStore.__init__` on the
  class OBJECT rather than replacing a name in one module's namespace. Since
  `from X import Y` binds the SAME object `Y` is (`backend.factories.GraphStore
  is backend.knowledge.storage.graph_store.GraphStore` is True -- confirmed
  directly, not assumed), patching the object is import-route-independent:
  static import, `from backend.knowledge import storage`, the factories
  re-export, `importlib.import_module`, and a relative import all resolve to
  the identical class, so patching its `__init__` requires enumerating no
  import spellings at all. It also removes the round-2 import-order hazard as
  a side effect: patching `__init__` leaves the class itself intact, so
  `GraphStore | None` type annotations elsewhere keep evaluating regardless
  of when `backend.factories` gets imported relative to the patch.
- C4: `_do_reindex` wraps its vault-edit sequence in `except Exception` (the
  "watcher must survive" guard) -- every prior case only drove the try
  block's happy path, so a write reachable only via the except handler had no
  coverage. Fix: a new case with a raising writer, asserting the graph store
  is untouched on the failure path too.

The AST check also now resolves relative imports (`node.level`) against the
importing module's own package, so `from ..knowledge.storage import x` (from
`backend.vault.filewatcher`, package `backend.vault`, resolves to
`backend.knowledge.storage.x`) is caught structurally rather than relying on
the behavioral trap alone.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path

import pytest

import backend.knowledge.storage.graph_store as gs_mod
import backend.vault.filewatcher as filewatcher_module
from backend.knowledge.config import FilewatcherConfig
from backend.vault.filewatcher import VaultFilewatcher

# ---------------------------------------------------------------------------
# Platform-availability marker (mirrors tests/unit/test_factories_phase3.py)
# ---------------------------------------------------------------------------
#
# Only the behavioral trap tests below need this: patching
# backend.factories.build_graph_store requires importing backend.factories,
# which eagerly imports EmbeddingGenerator (sentence_transformers,
# Linux/container only).
#
# Deliberate trade-off, not an oversight: on a platform without
# sentence_transformers, all five behavioral cases below skip (4 parametrized
# + 1 error-path) and this file degrades to the whitelist + AST check --
# round-2 strength, which fix round 2 itself proved permeable to two mutation
# forms. This does not bite in
# MIST.AI's actual verification environment (tests run in-container only,
# per tests/CLAUDE.md and every task brief in this plan; sentence_transformers
# is always present there). The alternative -- restructuring to avoid the
# backend.factories dependency -- would mean not exercising the real
# build_graph_store/GraphStore integration points this guard exists to watch,
# which defeats its purpose. Kept as a conscious choice, documented here
# rather than left implicit.

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


class _RaisingVaultWriter:
    """Fails the first vault-edit step, driving _do_reindex's except path."""

    async def mark_authored_by_user_edit(self, path: Path) -> None:
        raise RuntimeError("disk hiccup")


class _FakeInvalidationBus:
    """Records publish calls -- the second and final vault-edit step."""

    def __init__(self) -> None:
        self.published: list[object] = []

    async def publish(self, event: object) -> None:
        self.published.append(event)


def _make_watcher(vault_root: Path, writer: object, bus: _FakeInvalidationBus) -> VaultFilewatcher:
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


def _patch_graph_store_construction(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Patch GraphStore construction, route-independently, and return the trace.

    Patches `GraphStore.__init__` on the class OBJECT (not a name binding in
    any one module's namespace), so however a regression obtains "the
    GraphStore class" -- direct import, `from backend.knowledge import
    storage`, the `backend.factories` re-export, `importlib.import_module`,
    a relative import -- it is the same object, and constructing it here is
    recorded regardless of route (fix round 3, closing C6).

    Also patches `backend.factories.build_graph_store` so a write reached
    through it does not first crash inside the REAL `build_neo4j_connection`
    (which `build_graph_store` calls before ever touching `GraphStore`, and
    which cannot succeed against this test's fake config) -- without this,
    that mutation form would be masked by an unrelated crash rather than
    caught by a clean trap record. The patched version still routes through
    the (already-patched) real `GraphStore`, so a single `uses` list captures
    every path.

    `backend.factories` is imported before either patch is applied. This is
    no longer strictly required for correctness -- patching `__init__`
    leaves `GraphStore` itself a class, so `GraphStore | None` annotations
    elsewhere keep evaluating regardless of import order -- but it is kept
    for clarity and as a safety margin around the `build_graph_store` name
    patch, which IS a module-attribute rebinding.
    """
    import backend.factories  # noqa: F401

    uses: list[str] = []

    def _record_init(self: object, *args: object, **kwargs: object) -> None:
        uses.append("GraphStore.__init__")

    monkeypatch.setattr(gs_mod.GraphStore, "__init__", _record_init)

    def _fake_build_graph_store(*args: object, **kwargs: object) -> object:
        uses.append("build_graph_store")
        return gs_mod.GraphStore(None, None)  # still routes through the patched __init__

    monkeypatch.setattr("backend.factories.build_graph_store", _fake_build_graph_store)
    return uses


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


def _imported_dotted_names(source: str, package: str) -> list[str]:
    """Return every dotted name an import statement in `source` binds to.

    `import X.Y.Z` contributes `"X.Y.Z"`. `from X.Y import Z` contributes
    `"X.Y.Z"` (the import's real target, not whatever local alias it is
    bound to via `as`) so a regression cannot dodge detection by renaming
    what it imports. Walks the full AST, including inside function and
    method bodies, so a lazily-constructed import inside `_do_reindex`
    itself is caught, not just module-level imports.

    `package` is the importing module's own `__package__` (e.g.
    "backend.vault" for backend/vault/filewatcher.py), used to resolve
    relative imports to the same absolute dotted form absolute imports
    produce: `from ..knowledge.storage import x` reaches
    `backend.knowledge.storage.x` exactly as directly as `from
    backend.knowledge.storage import x` does, and must not evade detection
    just by spelling the path with dots.
    """
    tree = ast.parse(source)
    package_parts = package.split(".") if package else []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module
            else:
                base_parts = package_parts[: len(package_parts) - node.level + 1]
                base = ".".join(base_parts + ([node.module] if node.module else []))
            if base:
                names.extend(f"{base}.{alias.name}" for alias in node.names)
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
    round 2) while this dotted-name comparison does not. Also resolves
    relative imports (fix round 3) via `_imported_dotted_names`'s `package`
    argument.
    """
    source = inspect.getsource(filewatcher_module)
    package = filewatcher_module.__package__
    for dotted in _imported_dotted_names(source, package):
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

    Patches GraphStore.__init__ on the class object (route-independent --
    see `_patch_graph_store_construction`) plus `backend.factories.build_graph_store`
    (this codebase's own DI entry point, and exactly how `build_filewatcher`
    used to reach `GraphRegenerator` before Task 6: `CurationGraphRegenerator
    (graph_store=build_graph_store(config))`), then runs the real
    `_do_reindex` for a user-edit path under each of the four vault
    subdirectories the retired GraphRegenerator once dispatched on
    differently.
    """
    # Arrange
    uses = _patch_graph_store_construction(monkeypatch)

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


@requires_sentence_transformers
def test_do_reindex_error_path_never_reaches_a_graph_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The except-Exception failure path is Inv-A1-safe too (fix round 3, C4).

    All four happy-path cases above only exercise `_do_reindex`'s try block;
    none drives the `except Exception` recovery handler ("watcher must
    survive" guard), so a graph write reachable only from inside that handler
    had no coverage. A writer whose `mark_authored_by_user_edit` raises
    forces that path. Also pins the handler's documented recovery behavior:
    the mtime is dropped so the next audit pass retries the edit, and no
    cache-invalidation event is published for a sequence that never
    completed.
    """
    # Arrange
    uses = _patch_graph_store_construction(monkeypatch)

    vault_root = tmp_path / "vault"
    sessions_dir = vault_root / "sessions"
    sessions_dir.mkdir(parents=True)
    p = sessions_dir / "2026-07-30-test.md"
    p.write_text("---\ntype: mist-session\n---\n\nbody\n", encoding="utf-8")

    writer = _RaisingVaultWriter()
    bus = _FakeInvalidationBus()
    fw = _make_watcher(vault_root, writer, bus)

    # Act
    asyncio.run(fw._do_reindex(str(p), is_mist_write=False))

    # Assert
    assert bus.published == []
    assert str(p) not in fw._known_mtimes
    assert uses == []
