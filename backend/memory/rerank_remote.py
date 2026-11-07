"""Remote placement of the rerank step: HttpReranker scores pairs over an HTTP rerank endpoint.

`MIST_MEMORY_RERANK_BACKEND=remote` puts the cross-encoder on another host (Design point 17),
for example a llama-server started with `--reranking`. The endpoint's path and its request and
reply shapes are as known to the planner and unverified: H4 probes the server before anyone sets
`remote`. They live in `RERANK_PATH`, `build_rerank_request` and `parse_rerank_reply` only, so a
correction from the probe touches those three and nothing else.

Every fault (a timeout, a transport error, an error status or a malformed reply) raises
MemoryRerankError, which the fetcher turns into RRF scores, so a remote fault costs ranking
quality and never the turn. Messages name the endpoint and the cause, never the query or the
documents, which are the user's memory text.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import httpx

from backend.errors import MemoryRerankError

from .config import RemoteScoreKind
from .rerank import _sigmoid

# Path of the rerank call, joined to the base URL from `MIST_MEMORY_RERANK_URL`.
RERANK_PATH = "/v1/rerank"

_SCORE_KINDS: tuple[str, ...] = ("logit", "probability")
_URL_SCHEMES: tuple[str, ...] = ("http", "https")

# Below this budget `pairs_within` reports nothing fits: a round trip to another host costs more
# than a few milliseconds, so a call started with less would only time out.
_MIN_BUDGET_MS = 5.0


class RerankReplyError(ValueError):
    """The rerank server's reply does not have the expected shape.

    `parse_rerank_reply` raises it without the endpoint, which it does not know; HttpReranker
    wraps it in MemoryRerankError naming the endpoint.
    """


def build_rerank_request(model: str | None, query: str, texts: Sequence[str]) -> dict[str, object]:
    """The JSON body of one rerank call for `query` against `texts`, in order.

    `top_n` is the whole batch, so the server scores and returns every document. `model` is
    left out when None, for a server that serves one model and takes no name.
    """
    body: dict[str, object] = {"query": query, "documents": list(texts), "top_n": len(texts)}
    if model is not None:
        body = {"model": model, **body}
    return body


def parse_rerank_reply(body: object, count: int) -> list[float]:
    """The raw relevance scores of a rerank reply, in input order.

    The server may return its results sorted by score, so each result's `index` places its
    `relevance_score` back at the document it scores.

    Args:
        body: The decoded JSON reply.
        count: How many documents the request sent.

    Raises:
        RerankReplyError: The body is not an object with a `results` list; a result is not an
            object, has no integer `index`, an index outside the batch or a repeated one, or
            has no finite numeric `relevance_score`; or an index of the batch is missing.
    """
    if not isinstance(body, dict):
        raise RerankReplyError(f"the reply is not a JSON object but {type(body).__name__}")
    results = body.get("results")
    if not isinstance(results, list):
        raise RerankReplyError("the reply has no `results` list")
    scores: list[float | None] = [None] * count
    for position, result in enumerate(results):
        if not isinstance(result, dict):
            raise RerankReplyError(f"result {position} is not a JSON object")
        index = result.get("index")
        # bool is an int subclass, and JSON `true` is neither an index nor a score.
        if isinstance(index, bool) or not isinstance(index, int):
            raise RerankReplyError(f"result {position} has no integer `index`")
        if not 0 <= index < count:
            raise RerankReplyError(f"result {position} has index {index}, out of range for {count}")
        if scores[index] is not None:
            raise RerankReplyError(f"result {position} repeated index {index}")
        score = result.get("relevance_score")
        if isinstance(score, bool) or not isinstance(score, int | float):
            raise RerankReplyError(f"result {position} has no numeric `relevance_score`")
        if not math.isfinite(score):
            raise RerankReplyError(f"result {position} has a relevance_score that is not finite")
        scores[index] = float(score)
    missing = [index for index, score in enumerate(scores) if score is None]
    if missing:
        raise RerankReplyError(f"the reply is missing index {missing[0]} of {count}")
    return [score for score in scores if score is not None]


class HttpReranker:
    """Scores query-text pairs on a remote rerank server, one request per `score` call.

    The client is injected so the factory owns its lifetime and connection pool, and tests pass
    one over `httpx.MockTransport`. The caller restricts the URL to a private host
    (`MemoryConfig` does, for `MIST_MEMORY_RERANK_URL`); this class sends memory text wherever
    its URL points.
    """

    def __init__(
        self,
        client: httpx.Client,
        *,
        base_url: str,
        model: str | None,
        timeout_ms: float,
        score_kind: RemoteScoreKind,
        max_pairs: int,
    ) -> None:
        """Bind the reranker to one endpoint.

        Args:
            client: A synchronous client; the fetcher calls `score` from a worker thread.
            base_url: The server's base URL (`MIST_MEMORY_RERANK_URL`); `RERANK_PATH` is
                joined to it.
            model: Model name sent with each request (`MIST_MEMORY_RERANK_REMOTE_MODEL`), or
                None to send none.
            timeout_ms: Configured timeout of one call (`MIST_MEMORY_RERANK_TIMEOUT_MS`);
                `score` lowers it to the budget left.
            score_kind: `logit` when the server returns raw logits, which a sigmoid maps into
                [0, 1]; `probability` when it returns scores already in [0, 1], which are
                clamped (`MIST_MEMORY_RERANK_REMOTE_SCORES`).
            max_pairs: Ceiling on the pairs one call scores (`MIST_MEMORY_RERANK_TOP_N`).

        Raises:
            MemoryRerankError: `base_url` is not an http or https URL with a host,
                `timeout_ms` is not a positive finite number, `score_kind` is not `logit` or
                `probability`, or `max_pairs` is not positive.
        """
        try:
            base = httpx.URL(base_url)
        except httpx.InvalidURL as exc:
            raise MemoryRerankError(f"remote reranker base_url is not a valid URL: {exc}") from exc
        if base.scheme not in _URL_SCHEMES or not base.host:
            raise MemoryRerankError(
                f"remote reranker base_url must be an http or https URL with a host, "
                f"got scheme {base.scheme!r} and host {base.host!r}"
            )
        self._url = base.join(base.path.rstrip("/") + RERANK_PATH)
        # The URL as messages show it: without any credentials the base URL carried.
        self._shown_url = f"{self._url.scheme}://{self._url.netloc.decode('ascii')}{self._url.path}"
        if not (math.isfinite(timeout_ms) and timeout_ms > 0):
            raise MemoryRerankError(
                f"remote reranker timeout_ms must be positive and finite, got {timeout_ms} "
                f"for {self._shown_url}"
            )
        if score_kind not in _SCORE_KINDS:
            raise MemoryRerankError(
                f"remote reranker score_kind must be one of {_SCORE_KINDS}, got {score_kind!r} "
                f"for {self._shown_url}"
            )
        if max_pairs <= 0:
            raise MemoryRerankError(
                f"remote reranker max_pairs must be positive, got {max_pairs} "
                f"for {self._shown_url}"
            )
        self._client = client
        self._model = model
        self._timeout_ms = float(timeout_ms)
        self._score_kind = score_kind
        self._max_pairs = max_pairs

    def score(
        self, query: str, texts: Sequence[str], budget_ms: float | None = None
    ) -> list[float]:
        """Score each text against `query` in one request.

        The request's timeout is the configured one or `budget_ms`, whichever is smaller. httpx
        applies it to each phase of the call (connect, write, read and pool wait), not to the
        call as a whole, so a server that trickles its reply can overrun it; the fetcher's own
        deadline still bounds the turn.

        Raises:
            MemoryRerankError: No budget is left, the call times out or fails in transport,
                the server answers with an error status, or the reply is not JSON in the shape
                `parse_rerank_reply` accepts. The message names the endpoint and the cause.
        """
        if not texts:
            return []
        timeout_ms = self._timeout_ms
        if budget_ms is not None:
            if not budget_ms > 0:
                raise MemoryRerankError(
                    f"remote reranker at {self._shown_url} has no budget left ({budget_ms} ms) "
                    f"for {len(texts)} pairs"
                )
            timeout_ms = min(timeout_ms, budget_ms)
        body = self._post(build_rerank_request(self._model, query, texts), timeout_ms)
        try:
            raw = parse_rerank_reply(body, len(texts))
        except RerankReplyError as exc:
            raise MemoryRerankError(
                f"remote reranker at {self._shown_url} sent a malformed reply: {exc}"
            ) from exc
        if self._score_kind == "logit":
            return [_sigmoid(value) for value in raw]
        return [min(max(value, 0.0), 1.0) for value in raw]

    def pairs_within(self, budget_ms: float) -> int:
        """The configured ceiling when at least 5 ms are left, else zero.

        One request scores every pair, so the cost hardly grows with the batch; what a tight
        budget cannot afford is the round trip itself.
        """
        return self._max_pairs if budget_ms >= _MIN_BUDGET_MS else 0

    def _post(self, payload: dict[str, object], timeout_ms: float) -> object:
        """Send one rerank request and return its decoded JSON body.

        Only exception types and the status code reach the messages: an httpx message can
        quote the request, and the request carries the user's text.
        """
        where = f"remote reranker at {self._shown_url}"
        try:
            response = self._client.post(
                self._url, json=payload, timeout=httpx.Timeout(timeout_ms / 1000.0)
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise MemoryRerankError(
                f"{where} timed out after {timeout_ms:g} ms: {type(exc).__name__}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise MemoryRerankError(f"{where} returned HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise MemoryRerankError(f"{where} failed: {type(exc).__name__}") from exc
        try:
            return response.json()
        except ValueError as exc:
            # json.JSONDecodeError and UnicodeDecodeError both derive from ValueError.
            raise MemoryRerankError(
                f"{where} sent a body that is not JSON: {type(exc).__name__}"
            ) from exc
