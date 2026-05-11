"""Graph subgraph regenerator for vault user-edit handling.

Per ADR-010 invariants 5 + 6: vault is canonical; graph is rebuildable
from vault alone. On user-edit detected by VaultFilewatcher, triples
WHERE DERIVED_FROM.path == edited_path are marked status='orphaned'
(preserved per ADR-010, not hard-deleted) and re-derived from the
updated file content.

Bucket dispatch (per ADR-011):
- Bucket 1 (identity/, users/): deterministic parse + write (no LLM call)
- Bucket 2/3 (sessions/, decisions/): queue async LLM re-extraction via
  the existing extraction pipeline at current ontology_version
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from backend.interfaces import ExtractionPipelineProtocol, GraphStoreProtocol
from backend.knowledge.curation.bucket1_reader import (
    parse_identity_file,
    parse_user_file,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RebuildResult:
    """Result of a single GraphRegenerator.rebuild_from_path call."""

    path: Path
    bucket: str  # "1" | "2" | "3"
    orphaned_triple_count: int
    new_triple_count: int  # 0 if async-deferred
    ontology_version: str
    deferred: bool


class GraphRegenerator:
    """Rebuilds graph subgraph from a vault file on user-edit detection.

    Wired into VaultFilewatcher._do_reindex after sidecar indexing
    completes (Task 19). On each call:

    1. Orphan-marks existing DERIVED_FROM-scoped triples for the edited
       path (atomic, sync via graph store).
    2. Dispatches bucket-specific re-derivation:
       - Bucket 1 (identity/, users/): deterministic parse, no LLM.
       - Bucket 2/3 (sessions/, decisions/): async LLM extraction queued
         via asyncio.create_task; caller receives deferred=True.
    """

    def __init__(
        self,
        graph_store: GraphStoreProtocol,
        extraction_pipeline: ExtractionPipelineProtocol,
    ) -> None:
        self._graph_store = graph_store
        self._extraction_pipeline = extraction_pipeline

    def _classify_bucket(self, path: Path) -> str:
        """Return bucket string based on vault path segments.

        Returns:
            "1" for identity/ or users/ paths (deterministic parse).
            "2" for sessions/ paths (async LLM extraction).
            "3" for decisions/ paths (async LLM extraction).
            "2" as conservative default for unrecognised paths.
        """
        parts = path.parts
        if "identity" in parts or "users" in parts:
            return "1"
        if "sessions" in parts:
            return "2"
        if "decisions" in parts:
            return "3"
        return "2"

    async def rebuild_from_path(self, path: Path) -> RebuildResult:
        """Rebuild the graph subgraph derived from the given vault file.

        Steps:
        1. Orphan-mark all triples with DERIVED_FROM.path == path.
        2. Re-derive based on bucket classification.
        3. Return RebuildResult (deferred=True for Bucket 2/3).

        Args:
            path: Absolute Path to the vault file that was user-edited.

        Returns:
            RebuildResult with counts and metadata.
        """
        bucket = self._classify_bucket(path)

        # Step 1: orphan-mark existing DERIVED_FROM-scoped triples
        orphaned = await self._graph_store.mark_orphaned_by_provenance_path(str(path))
        ontology_version = self._graph_store.current_ontology_version()

        if bucket == "1":
            # Deterministic re-derivation; no LLM call
            new_count = await self._rebuild_bucket1(path)
            return RebuildResult(
                path=path,
                bucket=bucket,
                orphaned_triple_count=orphaned,
                new_triple_count=new_count,
                ontology_version=ontology_version,
                deferred=False,
            )

        # Bucket 2/3: queue async LLM re-extraction
        asyncio.create_task(self._rebuild_async_extraction(path, ontology_version))
        return RebuildResult(
            path=path,
            bucket=bucket,
            orphaned_triple_count=orphaned,
            new_triple_count=0,
            ontology_version=ontology_version,
            deferred=True,
        )

    async def _rebuild_bucket1(self, path: Path) -> int:
        """Deterministic Bucket 1 re-derivation.

        Parses identity/mist.md or users/<user>.md and upserts the
        resulting attributes/edges into the graph store. No LLM calls.

        Returns:
            Count of new triples written.
        """
        if path.name == "mist.md":
            parsed_identity = parse_identity_file(path)
            return await self._graph_store.upsert_identity(
                parsed_identity, derived_from_path=str(path)
            )

        # Check parent directory name for users/ classification
        if len(path.parts) >= 2 and path.parts[-2] == "users":
            parsed_user = parse_user_file(path)
            return await self._graph_store.upsert_user(parsed_user, derived_from_path=str(path))

        logger.warning(
            "GraphRegenerator: bucket-1 path not recognised as identity or user file: %s",
            path,
        )
        return 0

    async def retry_orphaned(self) -> None:
        """Retry orphaned triples by re-running async re-extraction.

        Queries the graph for triples with status='orphaned', groups by
        DERIVED_FROM.path, and re-runs re-extraction for each path that
        still exists on disk. Paths that no longer exist are skipped (vault
        file was deleted after the orphan was created).

        Called by `mist_admin vault-rebuild --retry-orphaned` as an operator
        handle for cases where the filewatcher missed events or async rebuild
        previously failed.
        """
        paths = await self._graph_store.get_orphaned_provenance_paths()
        ontology_version = self._graph_store.current_ontology_version()
        for path_str in paths:
            path = Path(path_str)
            if path.exists():
                await self._rebuild_async_extraction(path, ontology_version)
            else:
                logger.warning(
                    "GraphRegenerator.retry_orphaned: path no longer exists on disk, skipping: %s",
                    path,
                )

    async def _rebuild_async_extraction(self, path: Path, ontology_version: str) -> None:
        """Bucket 2/3 async re-extraction via existing extraction pipeline.

        Reads the vault file content and passes it to the extraction
        pipeline's extract_from_file method. On failure, logs and returns;
        the orphan-marked triples remain marked until a manual retry via
        `mist_admin vault-rebuild --retry-orphaned`.

        Args:
            path: Absolute Path to the vault file.
            ontology_version: Ontology version string to stamp on triples.
        """
        try:
            content = path.read_text(encoding="utf-8")
            await self._extraction_pipeline.extract_from_file(
                content=content,
                vault_note_path=str(path),
                ontology_version=ontology_version,
            )
        except Exception:
            logger.exception(
                "GraphRegenerator async re-extraction failed for %s; "
                "orphaned triples remain marked. Use mist_admin "
                "vault-rebuild --retry-orphaned to retry.",
                path,
            )
