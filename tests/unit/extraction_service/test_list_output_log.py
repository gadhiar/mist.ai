"""The list output log: opt-in, one JSON line per list call attempt, keyed by job id.

`EXTRACTION_LOG_LIST_OUTPUTS` is off by default. Off, the engine passes no callback
and makes no call at any level to the `backend.extraction_service.list_outputs`
logger. On, every frame and item call attempt that returns a response, a cut-short
one included, writes one line there at INFO: the request's job id, the stage, the
asked items, the attempt, the finish reason, the final text and the parser's refusal
or null. The r6 path never writes one. `__main__.main()` gives that logger a stderr
handler and the INFO level only when the setting is on, because under uvicorn's
default configuration an INFO record is otherwise dropped.

Every name in these turns is invented.
"""

from __future__ import annotations

import json
import logging
import sys

import pytest

from backend.extraction_service import __main__ as service_main
from backend.extraction_service import engine
from backend.extraction_service.adapters import get_adapter
from backend.extraction_service.engine import ExtractionEngine
from backend.extraction_service.settings import ServiceSettings
from backend.knowledge.extraction.list_frames import ListOutputError, parse_frame_output
from backend.llm.models import LLMRequest, LLMResponse
from tests.unit.extraction_service.test_list_pass import make_request
from tests.unit.extraction_service.test_list_path import (
    _CUT_SHORT,
    ListLLM,
    always,
    asked_items,
    fail_first,
    frame_ok,
    items_ok,
    tool_list,
    turn_groups,
)

_LOGGER = "backend.extraction_service.list_outputs"
_KEYS = ["event", "job_id", "stage", "items", "attempt", "finish_reason", "text", "refusal"]


def make_engine(llm: ListLLM, *, log: bool | None) -> ExtractionEngine:
    """An engine whose settings turn the log on or off, or leave it at its default (None)."""
    flag = {} if log is None else {"log_list_outputs": log}
    settings = ServiceSettings(
        llm_base_url="http://unused",
        model_hash="test-model-hash",
        model_file="test-model.gguf",
        max_attempts=2,
        llm_timeout_seconds=10.0,
        **flag,
    )
    return ExtractionEngine(llm=llm, adapter=get_adapter("gptoss"), settings=settings)


class Recorder:
    """Stands in for the log's logger and records every method called on it."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name: str):
        def record(*args, **kwargs) -> None:
            self.calls.append(name)

        return record


def lines(caplog) -> list[dict]:
    return [json.loads(r.getMessage()) for r in caplog.records if r.name == _LOGGER]


# ---------------------------------------------------------------------------
# Off
# ---------------------------------------------------------------------------


def test_the_setting_is_off_by_default():
    assert make_engine(ListLLM(), log=None)._settings.log_list_outputs is False


@pytest.mark.asyncio
@pytest.mark.parametrize("log", [None, False], ids=["default", "off"])
async def test_off_makes_no_call_to_the_logger_at_any_level(monkeypatch, caplog, log):
    recorder = Recorder()
    monkeypatch.setattr(engine, "_list_outputs_log", recorder)
    llm = ListLLM(frame=fail_first(frame_ok))

    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        await make_engine(llm, log=log).run(make_request(tool_list(5)))

    assert len(llm.of("frame")) == 2
    assert recorder.calls == []
    assert lines(caplog) == []


@pytest.mark.asyncio
async def test_on_calls_the_same_logger_the_off_test_watches(monkeypatch):
    recorder = Recorder()
    monkeypatch.setattr(engine, "_list_outputs_log", recorder)

    await make_engine(ListLLM(), log=True).run(make_request(tool_list(5)))

    assert recorder.calls == ["info", "info", "info"]


# ---------------------------------------------------------------------------
# On
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_writes_one_line_per_attempt_with_every_field(caplog):
    text = tool_list(5)
    llm = ListLLM(frame=fail_first(frame_ok))
    req = make_request(text, job_id="job-list-log-1")

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        await make_engine(llm, log=True).run(req)

    records = [r for r in caplog.records if r.name == _LOGGER]
    assert {r.levelno for r in records} == {logging.INFO}
    with pytest.raises(ListOutputError) as refused:
        parse_frame_output("not json", turn_groups(text))
    frame_calls, item_calls = llm.of("frame"), llm.of("items")
    assert lines(caplog) == [
        {
            "event": "list_output",
            "job_id": "job-list-log-1",
            "stage": "frame",
            "items": [],
            "attempt": 1,
            "finish_reason": "stop",
            "text": "not json",
            "refusal": str(refused.value),
        },
        {
            "event": "list_output",
            "job_id": "job-list-log-1",
            "stage": "frame",
            "items": [],
            "attempt": 2,
            "finish_reason": "stop",
            "text": frame_ok(frame_calls[1], 2).content,
            "refusal": None,
        },
        *(
            {
                "event": "list_output",
                "job_id": "job-list-log-1",
                "stage": "items",
                "items": asked_items(request),
                "attempt": 1,
                "finish_reason": "stop",
                "text": items_ok(request, n).content,
                "refusal": None,
            }
            for n, request in enumerate(item_calls, start=1)
        ),
    ]
    assert [list(line) for line in lines(caplog)] == [_KEYS] * 4


@pytest.mark.asyncio
async def test_a_cut_short_output_is_logged_and_the_r6_fallback_is_not(caplog):
    llm = ListLLM(frame=always(_CUT_SHORT))
    req = make_request(tool_list(3), job_id="job-list-log-2")

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        response = await make_engine(llm, log=True).run(req)

    assert response.warnings == ["list_frame_fallback"]
    assert len(llm.of("r6")) == 1
    (line,) = lines(caplog)
    assert line["stage"] == "frame"
    assert (line["attempt"], line["finish_reason"]) == (1, "length")
    assert (line["text"], line["refusal"]) == (_CUT_SHORT.content, None)
    assert line["job_id"] == "job-list-log-2"


@pytest.mark.asyncio
async def test_a_prose_turn_writes_no_line_when_on(caplog):
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        await make_engine(ListLLM(), log=True).run(make_request("I use Kestrin at work"))

    assert lines(caplog) == []


@pytest.mark.asyncio
async def test_a_line_keeps_non_ascii_text_as_written(caplog):
    def items(request: LLMRequest, call: int) -> LLMResponse:
        body = json.loads(items_ok(request, call).content)
        return LLMResponse(content=json.dumps(body, ensure_ascii=False), finish_reason="stop")

    llm = ListLLM(items=items)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        await make_engine(llm, log=True).run(make_request("Tools I use:\n- Zoëtrope"))

    item_records = [r for r in caplog.records if r.name == _LOGGER][1:]
    (record,) = item_records
    assert "Zoëtrope" in record.getMessage()
    assert "\\u00eb" not in record.getMessage()


# ---------------------------------------------------------------------------
# The handler __main__ attaches
# ---------------------------------------------------------------------------


def _run_main(monkeypatch, value: str | None) -> tuple[tuple, tuple]:
    """Run `main()` with uvicorn stubbed; the logger's state at uvicorn.run and after."""
    logger = logging.getLogger(_LOGGER)
    saved = (logger.level, list(logger.handlers), logger.propagate)
    if value is None:
        monkeypatch.delenv("EXTRACTION_LOG_LIST_OUTPUTS", raising=False)
    else:
        monkeypatch.setenv("EXTRACTION_LOG_LIST_OUTPUTS", value)
    seen: list[tuple] = []

    def run(app, **kwargs) -> None:
        seen.append((logger.level, list(logger.handlers), logger.propagate))

    monkeypatch.setattr(service_main, "build_app_from_env", lambda: object())
    monkeypatch.setattr(service_main.uvicorn, "run", run)
    try:
        service_main.main()
        return seen[0], (logger.level, list(logger.handlers), logger.propagate)
    finally:
        logger.setLevel(saved[0])
        logger.handlers[:] = saved[1]
        logger.propagate = saved[2]


@pytest.mark.parametrize("value", [None, "false"], ids=["unset", "false"])
def test_main_attaches_no_handler_when_off(monkeypatch, value):
    before = logging.getLogger(_LOGGER)
    state = (before.level, list(before.handlers), before.propagate)

    at_run, after = _run_main(monkeypatch, value)

    assert at_run == after == state


def test_main_attaches_a_stderr_handler_at_info_without_propagation_when_on(monkeypatch):
    before = list(logging.getLogger(_LOGGER).handlers)

    at_run, after = _run_main(monkeypatch, "true")

    assert at_run == after
    level, handlers, propagate = after
    (added,) = (h for h in handlers if h not in before)
    assert isinstance(added, logging.StreamHandler)
    assert added.stream is sys.stderr
    assert (level, propagate) == (logging.INFO, False)
