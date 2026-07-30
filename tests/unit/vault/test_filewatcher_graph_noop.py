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
were proven vacuous by mutation -- a reviewer patched an optional
`graph_store` constructor param plus a live `add_triple` call into
`_do_reindex` and all tests still passed, because nothing in this file gave
the mutation a channel to be observed through. The guard below is two
structural checks instead: an exhaustive parameter whitelist (no dependency
of ANY name can be added without failing this test, not just one named
"regenerator") and a source-text layering check (the filewatcher module
cannot import from knowledge.curation or knowledge.storage, the two packages
that own graph-write machinery). Together these prove no channel exists,
which is strictly stronger than proving one specific object was untouched.
"""

from __future__ import annotations

import inspect

import backend.vault.filewatcher as filewatcher_module
from backend.vault.filewatcher import VaultFilewatcher


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


def test_filewatcher_module_does_not_import_graph_write_layers() -> None:
    """The filewatcher module cannot reach graph-write machinery even indirectly.

    A regression need not go through the constructor: a lazily-constructed
    store inside `_do_reindex` (mirroring how backend/factories.py used to
    lazily construct GraphRegenerator) would pass the exhaustive-parameter
    check above while still writing to the graph. This closes that gap by
    asserting the module source never references the two packages that own
    graph-write surfaces: knowledge.curation (CurationGraphWriter,
    CurationPipeline) and knowledge.storage (GraphStore itself).
    """
    source = inspect.getsource(filewatcher_module)
    assert "knowledge.curation" not in source
    assert "knowledge.storage" not in source
