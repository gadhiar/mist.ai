"""Graph Regeneration Script - QUARANTINED (ADR-010).

This CLI previously drove the legacy utterance-based GraphRegenerator
(backend/knowledge/regeneration/graph_regenerator.py), which rebuilt the
knowledge graph by replaying event-store utterances.

Per ADR-010, vault markdown is the source of truth and the graph is
re-derived from the curated vault. Re-running utterance-based
regeneration would re-introduce synthetic eval pollution. The script is
quarantined: invoking it prints the superseded message and exits 1
without constructing or running any regenerator.

Replacement:
    mist_admin vault-rebuild --scope all

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
    "Legacy utterance-based regeneration is superseded by ADR-010 "
    "vault-rebuild. Re-deriving the graph from event-store utterances "
    "would re-introduce eval pollution. Use `mist_admin vault-rebuild "
    "--scope all` instead."
)


async def main() -> None:
    """Main entry point.

    Quarantined per ADR-010: prints the superseded message and exits 1
    instead of constructing or running the legacy regenerator. Use
    `mist_admin vault-rebuild --scope all`.
    """
    logger.error(_SUPERSEDED_MESSAGE)
    print(_SUPERSEDED_MESSAGE)
    sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
