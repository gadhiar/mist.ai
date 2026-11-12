"""Cancelling a running job: the cancel route, and a client that disconnects.

A cancel is the job's: every request waiting on the run is answered
`cancelled`, nothing is stored, and the same `job_id` runs when it is sent
again. A disconnect is one caller's: the run is cancelled only when no other
caller waits on it.

Every job here is held inside its first LLM call by the fake llama-server's
gate, so each test decides when the call ends and reads from the fake
whether the service left it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json

import pytest

from backend.extraction_contract.models import ExtractRequest
from tests.unit.extraction_service.conftest import make_extract_request

_SCOPE_THEN_EXTRACT = [
    '{"scope": "unknown", "confidence": 0.0}',
    '{"entities": [], "relationships": []}',
]


async def _until(condition, *, ticks: int = 2000) -> None:
    """Yield to the event loop until `condition()` holds, within `ticks` yields."""
    for _ in range(ticks):
        if condition():
            return
        await asyncio.sleep(0)
    raise AssertionError("the condition never held")


async def _settle(ticks: int = 50) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


async def _joined(fake_llama_state, callers: int) -> None:
    """Wait until `callers` extract requests have reached the shared run.

    Each extract request reads `/props` once, just before it joins or starts
    its job's run, so the fake's count of those reads places the callers.
    """
    await _until(lambda: fake_llama_state.props_requests == callers)
    await _settle()


async def _held_job(client, fake_llama_state, body: dict, *, callers: int = 1) -> list:
    """Send `body` from `callers` more clients and wait until its run is held in the fake."""
    already = fake_llama_state.props_requests
    tasks = []
    for n in range(1, callers + 1):
        tasks.append(asyncio.create_task(client.post("/v1/extract", json=body)))
        await _joined(fake_llama_state, already + n)
    await _until(lambda: len(fake_llama_state.chat_requests) >= 1)
    return tasks


async def _post_until_disconnect(app, body: dict, gone: asyncio.Event) -> list[dict]:
    """POST `body` to `/v1/extract` as a client that disconnects when `gone` is set.

    `httpx.ASGITransport` cancels the handler when its client is cancelled.
    A real server does not: the handler keeps running and the next ASGI
    `receive` answers `http.disconnect`. This is that shape.
    """
    payload = json.dumps(body).encode()
    sent = False

    async def receive() -> dict:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await gone.wait()
        return {"type": "http.disconnect"}

    messages: list[dict] = []

    async def send(message: dict) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/extract",
        "raw_path": b"/v1/extract",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"service"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(payload)).encode()),
        ],
        "client": ("test-client", 1),
        "server": ("service", 80),
    }
    await app(scope, receive, send)
    return messages


@pytest.mark.asyncio
class TestCancelRoute:
    async def test_a_running_job_is_cancelled_and_its_request_answered_cancelled(
        self, client, fake_llama_state, blocked_chat
    ):
        # Arrange
        (waiting,) = await _held_job(client, fake_llama_state, make_extract_request(job_id="j-1"))

        # Act
        cancel = await client.post("/v1/jobs/j-1/cancel")
        response = await asyncio.wait_for(waiting, timeout=2.0)

        # Assert
        assert cancel.status_code == 200
        assert cancel.json() == {"job_id": "j-1", "status": "cancelled"}
        assert response.status_code == 499
        error = response.json()["error"]
        assert error["code"] == "cancelled"
        assert error["retryable"] is True
        assert "j-1" in error["message"]

    async def test_the_cancelled_job_leaves_its_llm_call(
        self, client, fake_llama_state, blocked_chat
    ):
        (waiting,) = await _held_job(client, fake_llama_state, make_extract_request(job_id="j-1"))

        await client.post("/v1/jobs/j-1/cancel")
        await asyncio.wait_for(waiting, timeout=2.0)

        assert fake_llama_state.chat_aborted == 1
        assert len(fake_llama_state.chat_requests) == 1

    async def test_nothing_is_stored_and_the_same_job_id_runs_again(
        self, client, fake_llama_state, blocked_chat
    ):
        # Arrange: a job cancelled inside its first call.
        fake_llama_state.chat_responses = [*_SCOPE_THEN_EXTRACT[:1], *_SCOPE_THEN_EXTRACT]
        body = make_extract_request(job_id="j-1")
        (waiting,) = await _held_job(client, fake_llama_state, body)
        await client.post("/v1/jobs/j-1/cancel")
        await asyncio.wait_for(waiting, timeout=2.0)

        # Act: the model answers again, and the job is sent again.
        blocked_chat.set()
        again = await client.post("/v1/extract", json=body)

        # Assert: it ran from its first call, not from a stored result.
        assert again.status_code == 200, again.text
        assert again.json()["job_id"] == "j-1"
        assert len(fake_llama_state.chat_requests) == 3

    async def test_every_caller_sharing_the_run_is_answered_cancelled(
        self, client, fake_llama_state, blocked_chat
    ):
        callers = await _held_job(
            client, fake_llama_state, make_extract_request(job_id="j-1"), callers=3
        )

        cancel = await client.post("/v1/jobs/j-1/cancel")
        responses = await asyncio.wait_for(asyncio.gather(*callers), timeout=2.0)

        assert cancel.json()["status"] == "cancelled"
        assert [r.status_code for r in responses] == [499, 499, 499]
        assert {r.json()["error"]["code"] for r in responses} == {"cancelled"}
        assert fake_llama_state.chat_aborted == 1

    async def test_the_reply_is_the_contract_model(self):
        from backend.extraction_contract import models as contract
        from backend.extraction_service import app

        assert app.CancelResponse is contract.CancelResponse
        assert app.CancelOutcome is contract.CancelOutcome
        assert app.CancelResponse.__module__ == "backend.extraction_contract.models"

    async def test_an_unknown_job_is_reported_unknown(self, client, fake_llama_state):
        cancel = await client.post("/v1/jobs/never-sent/cancel")

        assert cancel.status_code == 200
        assert cancel.json() == {"job_id": "never-sent", "status": "unknown"}
        assert fake_llama_state.chat_requests == []

    async def test_a_finished_job_is_reported_finished_and_keeps_its_result(
        self, client, fake_llama_state
    ):
        # Arrange
        fake_llama_state.chat_responses = list(_SCOPE_THEN_EXTRACT)
        body = make_extract_request(job_id="j-done")
        first = await client.post("/v1/extract", json=body)
        assert first.status_code == 200

        # Act
        cancel = await client.post("/v1/jobs/j-done/cancel")
        replay = await client.post("/v1/extract", json=body)

        # Assert: the stored result still answers the job; nothing ran again.
        assert cancel.status_code == 200
        assert cancel.json() == {"job_id": "j-done", "status": "finished"}
        assert replay.json() == first.json()
        assert len(fake_llama_state.chat_requests) == 2

    async def test_a_second_cancel_of_a_cancelled_job_finds_nothing(
        self, client, fake_llama_state, blocked_chat
    ):
        (waiting,) = await _held_job(client, fake_llama_state, make_extract_request(job_id="j-1"))
        await client.post("/v1/jobs/j-1/cancel")
        await asyncio.wait_for(waiting, timeout=2.0)

        second = await client.post("/v1/jobs/j-1/cancel")

        assert second.json() == {"job_id": "j-1", "status": "unknown"}

    async def test_cancelling_one_job_leaves_another_running(
        self, client, fake_llama_state, blocked_chat
    ):
        # Arrange: two jobs, each held in its first call.
        fake_llama_state.chat_responses = [
            _SCOPE_THEN_EXTRACT[0],
            _SCOPE_THEN_EXTRACT[0],
            _SCOPE_THEN_EXTRACT[1],
        ]
        (first,) = await _held_job(client, fake_llama_state, make_extract_request(job_id="j-1"))
        (second,) = await _held_job(client, fake_llama_state, make_extract_request(job_id="j-2"))
        await _until(lambda: len(fake_llama_state.chat_requests) == 2)

        # Act
        await client.post("/v1/jobs/j-1/cancel")
        cancelled = await asyncio.wait_for(first, timeout=2.0)
        blocked_chat.set()
        kept = await asyncio.wait_for(second, timeout=2.0)

        # Assert
        assert cancelled.status_code == 499
        assert kept.status_code == 200, kept.text
        assert kept.json()["job_id"] == "j-2"
        assert fake_llama_state.chat_aborted == 1


@pytest.mark.asyncio
class TestACancelAsACallReturns:
    async def test_a_cancel_that_lands_as_an_llm_call_returns_ends_the_run(
        self, engine, wired_llm, fake_llama_state, blocked_chat, monkeypatch
    ):
        """A cancel that lands in the turn the scope call returns ends the run.

        On Python 3.11 `asyncio.wait_for` runs the call in a task of its own,
        and when that task has finished by the time a cancel wakes the caller,
        it returns the result and drops the cancel: the run goes on to its
        extract call and finishes. The fake holds the scope reply until the
        test sets its gate, and the test cancels the run's task in the first
        turn it sees the call return, which is inside that window, so the two
        land together on every run, not by chance. `_IdempotencyCache.cancel`
        cancels a job's run task the same way.

        Under `asyncio.timeout` the call runs in the run's own task, so the
        cancel stays pending and is raised at the run's next await. In this
        test that await is inside the extract call's request, which the run
        leaves before reading any of its reply.
        """
        # Arrange
        fake_llama_state.chat_responses = list(_SCOPE_THEN_EXTRACT)
        returned: list[object] = []
        streamed: list[object] = []
        real_invoke = wired_llm.invoke
        real_generate = wired_llm.generate

        async def invoke_and_note(request):
            response = await real_invoke(request)
            returned.append(response)
            return response

        async def generate_and_note(request, *, stream=False):
            # `invoke` reads the scope call through `generate` too, unstreamed.
            async with contextlib.aclosing(real_generate(request, stream=stream)) as chunks:
                async for chunk in chunks:
                    if stream:
                        streamed.append(chunk)
                    yield chunk

        monkeypatch.setattr(wired_llm, "invoke", invoke_and_note)
        monkeypatch.setattr(wired_llm, "generate", generate_and_note)
        request = ExtractRequest(**make_extract_request(job_id="j-1"))
        run = asyncio.create_task(engine.run(request))
        await _until(lambda: len(fake_llama_state.chat_requests) == 1)

        # Act: release the scope reply, and cancel the run as the call returns.
        blocked_chat.set()
        await _until(lambda: bool(returned))
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run, timeout=2.0)
        await _settle()

        # Assert: the scope call returned, the run ended cancelled, and no
        # reply of a later call was read.
        assert len(returned) == 1
        assert run.cancelled()
        assert streamed == []


@pytest.mark.asyncio
class TestClientDisconnect:
    async def test_the_only_caller_disconnecting_cancels_the_run(
        self, service_app, client, fake_llama_state, blocked_chat
    ):
        # Arrange: one caller, its job held in the first call.
        fake_llama_state.chat_responses = [*_SCOPE_THEN_EXTRACT[:1], *_SCOPE_THEN_EXTRACT]
        body = make_extract_request(job_id="j-1")
        gone = asyncio.Event()
        caller = asyncio.create_task(_post_until_disconnect(service_app, body, gone))
        await _joined(fake_llama_state, 1)
        await _until(lambda: len(fake_llama_state.chat_requests) == 1)

        # Act
        gone.set()
        await asyncio.wait_for(caller, timeout=2.0)
        await _until(lambda: fake_llama_state.chat_aborted == 1)

        # Assert: nothing is stored either, so the job runs when sent again.
        blocked_chat.set()
        again = await client.post("/v1/extract", json=body)
        assert again.status_code == 200, again.text
        assert len(fake_llama_state.chat_requests) == 3

    async def test_a_cancelled_only_caller_cancels_the_run(
        self, client, fake_llama_state, blocked_chat
    ):
        """The same, for a server that cancels the handler when its client leaves."""
        (caller,) = await _held_job(client, fake_llama_state, make_extract_request(job_id="j-1"))

        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        await _until(lambda: fake_llama_state.chat_aborted == 1)

        assert (await client.post("/v1/jobs/j-1/cancel")).json()["status"] == "unknown"

    async def test_a_caller_disconnecting_leaves_the_run_to_the_callers_still_waiting(
        self, service_app, client, fake_llama_state, blocked_chat
    ):
        # Arrange: two callers on one run; the first will disconnect.
        fake_llama_state.chat_responses = list(_SCOPE_THEN_EXTRACT)
        body = make_extract_request(job_id="j-1")
        gone = asyncio.Event()
        leaver = asyncio.create_task(_post_until_disconnect(service_app, body, gone))
        await _joined(fake_llama_state, 1)
        (stayer,) = await _held_job(client, fake_llama_state, body)

        # Act: the first disconnects; then the model answers.
        gone.set()
        await asyncio.wait_for(leaver, timeout=2.0)
        await _settle()
        aborted_while_held = fake_llama_state.chat_aborted
        blocked_chat.set()
        response = await asyncio.wait_for(stayer, timeout=2.0)

        # Assert: one run, never left, answered the caller that stayed and
        # was stored for the job.
        assert aborted_while_held == 0
        assert response.status_code == 200, response.text
        assert len(fake_llama_state.chat_requests) == 2
        replay = await client.post("/v1/extract", json=body)
        assert replay.json() == response.json()
        assert len(fake_llama_state.chat_requests) == 2

    async def test_the_last_of_several_callers_disconnecting_cancels_the_run(
        self, service_app, fake_llama_state, blocked_chat
    ):
        # Arrange
        body = make_extract_request(job_id="j-1")
        first_gone, second_gone = asyncio.Event(), asyncio.Event()
        first = asyncio.create_task(_post_until_disconnect(service_app, body, first_gone))
        await _joined(fake_llama_state, 1)
        second = asyncio.create_task(_post_until_disconnect(service_app, body, second_gone))
        await _joined(fake_llama_state, 2)
        await _until(lambda: len(fake_llama_state.chat_requests) == 1)

        # Act
        first_gone.set()
        await asyncio.wait_for(first, timeout=2.0)
        await _settle()
        aborted_with_one_left = fake_llama_state.chat_aborted
        second_gone.set()
        await asyncio.wait_for(second, timeout=2.0)
        await _until(lambda: fake_llama_state.chat_aborted == 1)

        # Assert
        assert aborted_with_one_left == 0
        assert len(fake_llama_state.chat_requests) == 1

    async def test_a_caller_that_stays_connected_is_answered_as_before(
        self, service_app, fake_llama_state
    ):
        fake_llama_state.chat_responses = list(_SCOPE_THEN_EXTRACT)
        body = make_extract_request(job_id="j-1")

        messages = await asyncio.wait_for(
            _post_until_disconnect(service_app, body, asyncio.Event()), timeout=2.0
        )

        assert messages[0]["type"] == "http.response.start"
        assert messages[0]["status"] == 200
        assert json.loads(messages[1]["body"])["job_id"] == "j-1"
