"""Quarantine guard for the legacy utterance-based GraphRegenerator.

Per ADR-010, vault markdown is the source of truth and the graph is
re-derived from the curated vault (see backend/knowledge/curation/
graph_regenerator.py). The legacy utterance-based regenerator in
backend/knowledge/regeneration/ derived the graph by replaying
event-store utterances, which would re-introduce synthetic eval
pollution. Both of its async entry points are quarantined: they raise
NotImplementedError immediately, pointing callers at the vault rebuild.

These tests lock in that quarantine so the legacy path cannot silently
come back to life.
"""

import asyncio

import pytest

from backend.knowledge.regeneration.graph_regenerator import GraphRegenerator


def test_regenerate_all_raises_quarantine():
    reg = GraphRegenerator.__new__(GraphRegenerator)
    with pytest.raises(NotImplementedError, match="ADR-010"):
        asyncio.run(reg.regenerate_all())


def test_regenerate_conversation_raises_quarantine():
    reg = GraphRegenerator.__new__(GraphRegenerator)
    with pytest.raises(NotImplementedError, match="ADR-010"):
        asyncio.run(reg.regenerate_conversation("some-id"))
