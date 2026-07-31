"""Graph Regeneration Script - QUARANTINED (ADR-010, retired further by R1.3).

This CLI previously drove the legacy utterance-based GraphRegenerator
(backend/knowledge/regeneration/graph_regenerator.py), which rebuilt the
knowledge graph by replaying event-store utterances.

ADR-010 first superseded that model with a vault-derived rebuild; R1.3
(Inv-A1) retired that path in turn -- vault markdown is prose MIST reads,
not a fact source, and a vault edit writes nothing to the graph. Re-running
utterance-based regeneration through THIS script would still re-introduce
synthetic eval pollution. The script is quarantined: invoking it prints
the superseded message and exits 1 without constructing or running any
regenerator.

Replacement:
    mist_admin graph-rebuild-from-log --dry-run   (graph rebuild, R1.2)
    mist_admin vault-rebuild --confirm             (vault sidecar reindex only)

The module remains importable so existing references do not break.
"""

from __future__ import annotations

import asyncio
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

_SUPERSEDED_MESSAGE = (
    "Legacy utterance-based regeneration is superseded by ADR-010's "
    "vault-derived rebuild, itself retired by R1.3 (Inv-A1). Re-deriving "
    "the graph from event-store utterances would re-introduce eval "
    "pollution. Use `mist_admin graph-rebuild-from-log --dry-run` for a "
    "graph rebuild, or `mist_admin vault-rebuild --confirm` to reindex "
    "the vault sidecar."
)


async def main() -> None:
    """Main entry point.

    Quarantined per ADR-010: prints the superseded message and exits 1
    instead of constructing or running the legacy regenerator. See
    `_SUPERSEDED_MESSAGE` for the current replacement commands.
    """
    logger.error(_SUPERSEDED_MESSAGE)
    print(_SUPERSEDED_MESSAGE)
    sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
