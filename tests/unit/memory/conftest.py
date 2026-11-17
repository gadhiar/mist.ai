"""Fixtures for the memory read-path tests: the shared fakes from tests/fakes/memory_fakes.py."""

import pytest

from tests.fakes.memory_fakes import FakeClock, FakeEmbedder, FakeReranker, FakeTokenCounter


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    """A FakeEmbedder producing deterministic 384-d unit vectors."""
    return FakeEmbedder()


@pytest.fixture
def fake_reranker() -> FakeReranker:
    """A FakeReranker scoring by query-token overlap."""
    return FakeReranker()


@pytest.fixture
def fake_clock() -> FakeClock:
    """A FakeClock pinned at its default start instant."""
    return FakeClock()


@pytest.fixture
def fake_token_counter() -> FakeTokenCounter:
    """A FakeTokenCounter counting whitespace-separated words."""
    return FakeTokenCounter()
