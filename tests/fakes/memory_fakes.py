"""Fakes for the memory read path: embedder, reranker, clock and token counter.

Every read-path task tests against these instead of real models or the wall clock, so the
tests need no weights, no GPU and no tz database, and each run produces the same numbers.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2, the embedder the sidecar and the index use

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class FakeEmbedder:
    """Deterministic 384-d unit vectors with the EmbeddingProvider shape.

    Each lowercase token is hashed to one dimension and one sign, and the vector is the
    normalised sum. Texts that share words therefore have a positive cosine similarity, which
    lets vector-pool tests rank a related text above an unrelated one, while the same text
    always maps to the same vector. A text with no tokens maps to the first basis vector, so
    every output is a unit vector.
    """

    def __init__(self, *, dimension: int = EMBEDDING_DIMENSION) -> None:
        self.dimension = dimension
        self.calls: list[str] = []

    def generate_embedding(self, text: str) -> list[float]:
        """Embed one text and record the call."""
        self.calls.append(text)
        vec = [0.0] * self.dimension
        for token in _tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            vec[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Embed several texts in order."""
        return [self.generate_embedding(text) for text in texts]


class FakeReranker:
    """Token-overlap scores with the Reranker shape (`score`, `pairs_within`).

    A text scores the share of the query's distinct tokens it contains, so the score is in
    [0, 1] and an exact paraphrase outranks a partial one. `pairs_per_ms` sets how many pairs
    `pairs_within` says fit a budget, which lets fetcher tests cut the rerank short without a
    real cost measurement.
    """

    def __init__(self, *, pairs_per_ms: float = 1.0) -> None:
        self.pairs_per_ms = pairs_per_ms
        self.calls: list[tuple[str, tuple[str, ...], float | None]] = []

    def score(
        self, query: str, texts: Sequence[str], budget_ms: float | None = None
    ) -> list[float]:
        """Score each text against the query and record the call."""
        self.calls.append((query, tuple(texts), budget_ms))
        query_tokens = set(_tokens(query))
        if not query_tokens:
            return [0.0 for _ in texts]
        return [len(query_tokens & set(_tokens(text))) / len(query_tokens) for text in texts]

    def pairs_within(self, budget_ms: float) -> int:
        """How many pairs fit `budget_ms` at the configured rate; never negative."""
        if budget_ms <= 0:
            return 0
        return int(budget_ms * self.pairs_per_ms)


class FakeClock:
    """A clock that moves only when a test advances it.

    `now()` gives an aware wall-clock instant and `monotonic()` a float in seconds; both move
    together under `advance`, so deadline and recency code sees one consistent time.
    """

    def __init__(self, start: datetime | None = None, *, monotonic_start: float = 1000.0) -> None:
        self._now = start if start is not None else datetime(2026, 10, 22, 18, 0, tzinfo=UTC)
        if self._now.tzinfo is None:
            raise ValueError("FakeClock needs an aware start instant")
        self._monotonic = monotonic_start

    def now(self) -> datetime:
        """The current wall-clock instant."""
        return self._now

    def monotonic(self) -> float:
        """The current monotonic reading in seconds."""
        return self._monotonic

    def advance(self, *, seconds: float = 0.0, ms: float = 0.0) -> None:
        """Move both readings forward by `seconds` plus `ms` milliseconds."""
        delta = seconds + ms / 1000.0
        if delta < 0:
            raise ValueError("FakeClock cannot move backwards")
        self._monotonic += delta
        self._now += timedelta(seconds=delta)


class FakeTokenCounter:
    """Counts whitespace-separated words, with the TokenCounter shape.

    A word count lets a block-budget test compute the expected total by hand, which the
    char/3.6 heuristic of ApproximateTokenCounter does not.
    """

    def __init__(self, *, per_message_overhead: int = 4) -> None:
        self.per_message_overhead = per_message_overhead

    def count(self, text: str) -> int:
        """Number of whitespace-separated words in `text`."""
        return len(text.split()) if text else 0

    def count_message(self, message: dict[str, object]) -> int:
        """Words in the message content plus the fixed per-message overhead."""
        return self.count(str(message.get("content") or "")) + self.per_message_overhead

    def count_messages(self, messages: list[dict[str, object]]) -> int:
        """Sum of `count_message` over the list."""
        return sum(self.count_message(message) for message in messages)
