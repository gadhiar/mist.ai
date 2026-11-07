"""Tests for backend.memory.rerank_remote: HttpReranker over an HTTP rerank endpoint.

Every test answers through `httpx.MockTransport`, in-process: no network is used and no server
is started. The handler records each request, so a test can check what was sent and that an
empty batch sends nothing.
"""

from __future__ import annotations

import inspect
import json
import math
from collections.abc import Callable

import httpx
import pytest

from backend.errors import MemoryRerankError
from backend.memory.rerank import Reranker
from backend.memory.rerank_remote import (
    RERANK_PATH,
    HttpReranker,
    RerankReplyError,
    build_rerank_request,
    parse_rerank_reply,
)

BASE_URL = "http://rerank.lan:8080"
ENDPOINT = f"{BASE_URL}{RERANK_PATH}"
SECRET = "the user's private note about the clinic appointment"

Handler = Callable[[httpx.Request], httpx.Response]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Recorder:
    """A MockTransport handler that records each request and answers with a scripted reply."""

    def __init__(self, reply: Handler) -> None:
        self.requests: list[httpx.Request] = []
        self._reply = reply

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._reply(request)


def _results(*pairs: tuple[int, float]) -> Handler:
    body = {"results": [{"index": i, "relevance_score": s} for i, s in pairs]}
    return lambda request: httpx.Response(200, json=body)


def _reranker(
    handler: Handler,
    *,
    score_kind: str = "logit",
    timeout_ms: float = 40.0,
    model: str | None = "bge-reranker",
    max_pairs: int = 20,
    base_url: str = BASE_URL,
) -> HttpReranker:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return HttpReranker(
        client,
        base_url=base_url,
        model=model,
        timeout_ms=timeout_ms,
        score_kind=score_kind,  # type: ignore[arg-type]
        max_pairs=max_pairs,
    )


def _needs_reranker(reranker: Reranker) -> Reranker:
    return reranker


# -- protocol and constructor -----------------------------------------------------------------


def test_implements_the_reranker_protocol_signatures() -> None:
    for name in ("score", "pairs_within"):
        expected = inspect.signature(getattr(Reranker, name))
        actual = inspect.signature(getattr(HttpReranker, name))
        assert list(actual.parameters) == list(expected.parameters)
        for param_name, param in expected.parameters.items():
            assert actual.parameters[param_name].default == param.default
    assert isinstance(_needs_reranker(_reranker(_results())), HttpReranker)


def test_every_constructor_argument_is_required() -> None:
    params = inspect.signature(HttpReranker.__init__).parameters
    for name in ("client", "base_url", "model", "timeout_ms", "score_kind", "max_pairs"):
        assert params[name].default is inspect.Parameter.empty, name


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"score_kind": "raw"}, "score_kind"),
        ({"timeout_ms": 0}, "timeout_ms"),
        ({"timeout_ms": -5}, "timeout_ms"),
        ({"timeout_ms": math.nan}, "timeout_ms"),
        ({"max_pairs": 0}, "max_pairs"),
        ({"base_url": "not a url ::"}, "base_url"),
        ({"base_url": "ftp://rerank.lan"}, "base_url"),
    ],
)
def test_constructor_rejects_values_it_cannot_run_with(overrides, fragment) -> None:
    with pytest.raises(MemoryRerankError, match=fragment):
        _reranker(_results(), **overrides)


# -- the request ------------------------------------------------------------------------------


def test_sends_one_post_with_the_query_and_documents_in_order() -> None:
    recorder = Recorder(_results((0, 0.0), (1, 0.0), (2, 0.0)))
    texts = ["first", "second", "third"]

    _reranker(recorder).score("what did we decide", texts, budget_ms=30.0)

    assert len(recorder.requests) == 1
    request = recorder.requests[0]
    assert request.method == "POST"
    assert str(request.url) == ENDPOINT
    assert json.loads(request.content) == {
        "model": "bge-reranker",
        "query": "what did we decide",
        "documents": texts,
        "top_n": 3,
    }


def test_a_base_url_with_a_trailing_slash_joins_cleanly() -> None:
    recorder = Recorder(_results((0, 0.0)))

    _reranker(recorder, base_url=f"{BASE_URL}/").score("q", ["a"], budget_ms=30.0)

    assert str(recorder.requests[0].url) == ENDPOINT


def test_no_model_name_leaves_the_field_out() -> None:
    assert "model" not in build_rerank_request(None, "q", ["a"])
    assert build_rerank_request("m", "q", ["a", "b"]) == {
        "model": "m",
        "query": "q",
        "documents": ["a", "b"],
        "top_n": 2,
    }


@pytest.mark.parametrize(
    ("timeout_ms", "budget_ms", "expected_s"),
    [
        (40.0, 25.0, 0.025),
        (40.0, 90.0, 0.040),
        (40.0, None, 0.040),
        (40.0, math.inf, 0.040),
    ],
)
def test_timeout_is_the_smaller_of_the_configured_one_and_the_budget(
    timeout_ms, budget_ms, expected_s
) -> None:
    recorder = Recorder(_results((0, 0.0)))

    _reranker(recorder, timeout_ms=timeout_ms).score("q", ["a"], budget_ms=budget_ms)

    timeouts = recorder.requests[0].extensions["timeout"]
    assert set(timeouts) == {"connect", "read", "write", "pool"}
    assert all(value == pytest.approx(expected_s) for value in timeouts.values())


@pytest.mark.parametrize("budget_ms", [0.0, -1.0, math.nan])
def test_no_budget_left_raises_without_a_request(budget_ms) -> None:
    recorder = Recorder(_results((0, 0.0)))

    with pytest.raises(MemoryRerankError, match="rerank.lan"):
        _reranker(recorder).score("q", ["a"], budget_ms=budget_ms)

    assert recorder.requests == []


def test_an_empty_list_sends_no_request() -> None:
    recorder = Recorder(_results())

    assert _reranker(recorder).score("q", [], budget_ms=30.0) == []
    assert _reranker(recorder).score("q", [], budget_ms=0.0) == []
    assert recorder.requests == []


# -- the reply --------------------------------------------------------------------------------


def test_results_out_of_order_come_back_in_input_order() -> None:
    recorder = Recorder(_results((2, 3.0), (0, -1.0), (1, 0.5)))

    scores = _reranker(recorder).score("q", ["a", "b", "c"], budget_ms=30.0)

    assert scores == pytest.approx([_sigmoid(-1.0), _sigmoid(0.5), _sigmoid(3.0)])


def test_logit_scores_pass_through_a_sigmoid_that_does_not_overflow() -> None:
    recorder = Recorder(_results((0, 0.0), (1, 1000.0), (2, -1000.0)))

    scores = _reranker(recorder, score_kind="logit").score("q", ["a", "b", "c"], budget_ms=30.0)

    assert scores == pytest.approx([0.5, 1.0, 0.0])
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_probability_scores_are_clamped_to_the_unit_interval() -> None:
    recorder = Recorder(_results((0, 0.4), (1, 1.2), (2, -0.1)))

    scores = _reranker(recorder, score_kind="probability").score(
        "q", ["a", "b", "c"], budget_ms=30.0
    )

    assert scores == [0.4, 1.0, 0.0]


def test_parser_returns_raw_scores_in_input_order() -> None:
    body = {"results": [{"index": 1, "relevance_score": 2}, {"index": 0, "relevance_score": 0.5}]}

    assert parse_rerank_reply(body, 2) == [0.5, 2.0]


# -- faults -----------------------------------------------------------------------------------


def _assert_names_url_not_text(exc: pytest.ExceptionInfo[MemoryRerankError]) -> None:
    message = str(exc.value)
    assert ENDPOINT in message
    assert SECRET not in message
    assert "clinic" not in message


def test_a_500_raises_naming_the_url_and_the_status() -> None:
    reranker = _reranker(lambda request: httpx.Response(500, text=f"echo: {SECRET}"))

    with pytest.raises(MemoryRerankError, match="500") as exc:
        reranker.score(SECRET, [SECRET], budget_ms=30.0)

    _assert_names_url_not_text(exc)
    assert isinstance(exc.value.__cause__, httpx.HTTPStatusError)


def test_a_timeout_raises_naming_the_url_and_the_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(f"timed out reading {SECRET}", request=request)

    with pytest.raises(MemoryRerankError, match="timed out") as exc:
        _reranker(handler).score(SECRET, [SECRET], budget_ms=30.0)

    _assert_names_url_not_text(exc)


def test_a_connection_failure_raises_naming_the_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(MemoryRerankError, match="ConnectError") as exc:
        _reranker(handler).score(SECRET, [SECRET], budget_ms=30.0)

    _assert_names_url_not_text(exc)


def test_a_body_that_is_not_json_raises() -> None:
    reranker = _reranker(lambda request: httpx.Response(200, text=f"<html>{SECRET}"))

    with pytest.raises(MemoryRerankError, match="JSON") as exc:
        reranker.score(SECRET, [SECRET], budget_ms=30.0)

    _assert_names_url_not_text(exc)


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ([], "object"),
        ({}, "results"),
        ({"results": {"index": 0}}, "results"),
        ({"results": ["x"]}, "object"),
        ({"results": [{"relevance_score": 0.5}]}, "index"),
        ({"results": [{"index": "0", "relevance_score": 0.5}]}, "index"),
        ({"results": [{"index": True, "relevance_score": 0.5}]}, "index"),
        ({"results": [{"index": 1, "relevance_score": 0.5}]}, "out of range"),
        ({"results": [{"index": -1, "relevance_score": 0.5}]}, "out of range"),
        ({"results": [{"index": 0}]}, "relevance_score"),
        ({"results": [{"index": 0, "relevance_score": "high"}]}, "relevance_score"),
        ({"results": [{"index": 0, "relevance_score": False}]}, "relevance_score"),
        ({"results": [{"index": 0, "relevance_score": None}]}, "relevance_score"),
    ],
)
def test_parser_rejects_a_malformed_body(body, fragment) -> None:
    with pytest.raises(RerankReplyError, match=fragment):
        parse_rerank_reply(body, 1)


def test_parser_rejects_a_non_finite_score() -> None:
    body = {"results": [{"index": 0, "relevance_score": math.inf}]}

    with pytest.raises(RerankReplyError, match="finite"):
        parse_rerank_reply(body, 1)


def test_parser_rejects_a_missing_index() -> None:
    body = {"results": [{"index": 0, "relevance_score": 0.5}]}

    with pytest.raises(RerankReplyError, match="missing index 1"):
        parse_rerank_reply(body, 2)


def test_parser_rejects_a_repeated_index() -> None:
    body = {"results": [{"index": 0, "relevance_score": 0.5}, {"index": 0, "relevance_score": 1}]}

    with pytest.raises(RerankReplyError, match="repeated"):
        parse_rerank_reply(body, 2)


@pytest.mark.parametrize(
    "results",
    [
        [{"index": 0, "relevance_score": 0.5}],
        [{"index": 0, "relevance_score": 0.5}, {"index": 5, "relevance_score": 0.1}],
        [{"index": 0, "relevance_score": "x"}, {"index": 1, "relevance_score": 0.1}],
    ],
)
def test_a_malformed_reply_raises_naming_the_url_not_the_text(results) -> None:
    reranker = _reranker(lambda request: httpx.Response(200, json={"results": results}))

    with pytest.raises(MemoryRerankError, match="malformed") as exc:
        reranker.score(SECRET, [SECRET, SECRET], budget_ms=30.0)

    _assert_names_url_not_text(exc)
    assert isinstance(exc.value.__cause__, RerankReplyError)


def test_credentials_in_the_base_url_stay_out_of_errors() -> None:
    reranker = _reranker(
        lambda request: httpx.Response(503), base_url="http://user:hunter2@rerank.lan:8080"
    )

    with pytest.raises(MemoryRerankError, match="503") as exc:
        reranker.score("q", ["a"], budget_ms=30.0)

    assert "hunter2" not in str(exc.value)
    assert ENDPOINT in str(exc.value)


# -- pairs_within -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("budget_ms", "expected"),
    [(5.0, 12), (50.0, 12), (math.inf, 12), (4.999, 0), (0.0, 0), (-3.0, 0), (math.nan, 0)],
)
def test_pairs_within_is_the_ceiling_from_five_ms(budget_ms, expected) -> None:
    assert _reranker(_results(), max_pairs=12).pairs_within(budget_ms) == expected
