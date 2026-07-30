"""Vault-edit graph no-op reporter (R1.3 transitional).

Under R1 Inv-A1 the vault is not a fact source: vault edits change the prose
MIST reads, never what the graph asserts. Every re-derivation bucket has
retired -- Bucket 1 (deterministic identity/user parse) and Bucket 2/3 (LLM
re-extraction of session and decision notes). What remains reports the edit so
the filewatcher can publish a read-path cache-invalidation event.

This module is deleted in the next task, which moves that event onto the
filewatcher directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


class _OntologyVersionSource(Protocol):
    """The only graph-store surface this transitional shim still needs."""

    def current_ontology_version(self) -> str: ...


@dataclass(frozen=True)
class RebuildResult:
    """Result of a single GraphRegenerator.rebuild_from_path call."""

    path: Path
    bucket: str  # always "ignored" post-R1.3 (all buckets retired)
    orphaned_triple_count: int  # always 0 post-R1.3
    new_triple_count: int  # always 0 post-R1.3
    ontology_version: str
    deferred: bool  # always False post-R1.3


class GraphRegenerator:
    """Reports vault edits as graph no-ops (R1.3 transitional).

    Wired into VaultFilewatcher._do_reindex after sidecar indexing completes.
    Every call to `rebuild_from_path` returns immediately with bucket
    "ignored" and zeroed counts -- no orphan-marking, no re-derivation, no
    graph write of any kind. The vault is not a fact source under Inv-A1, so
    there is nothing left for this class to rebuild.
    """

    def __init__(
        self,
        graph_store: _OntologyVersionSource,
    ) -> None:
        self._graph_store = graph_store

    async def rebuild_from_path(self, path: Path) -> RebuildResult:
        """Report a vault edit as a graph no-op.

        R1.3 (Inv-A1): the vault is not a fact source. Vault edits change the
        prose MIST reads; facts enter the graph only through the utterance
        log. Every bucket has retired -- Bucket 1 (deterministic user/
        identity parse) and Bucket 2/3 (LLM re-extraction of session and
        decision notes) alike -- so this returns without touching the graph.

        The method survives its buckets because the filewatcher's invariant-5
        chain still needs a rebuild event to publish for read-path cache
        eviction. Task 6 removes this class entirely and moves that event onto
        the filewatcher.

        Args:
            path: Absolute Path to the vault file that was user-edited.

        Returns:
            RebuildResult with zeroed counts and bucket "ignored".
        """
        logger.info(
            "GraphRegenerator: vault edit at %s is a graph no-op under R1.3 "
            "(facts enter via the utterance log, not the vault)",
            path,
        )
        return RebuildResult(
            path=path,
            bucket="ignored",
            orphaned_triple_count=0,
            new_triple_count=0,
            ontology_version=self._graph_store.current_ontology_version(),
            deferred=False,
        )
