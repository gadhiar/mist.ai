"""Regression test for the TTS consumer sentinel-coalescing bug.

The bug (surfaced by the 2026-05-25 smoke walk): when the LLM produces
short sentences that trip the per-sentence coalescing threshold, the
inner coalescing loop pulls the None end-of-stream sentinel off the
sentence queue and breaks. Pre-fix, the outer loop never sees the
sentinel and hangs on `get(timeout=1.0)` forever -- state_cycle("idle")
never emits, FE stuck in SPEAK state.

The fix (`backend/voice_processor.py:329-339`): put the sentinel back on
the queue when the coalescing branch consumes it, so the outer loop
observes it on the next iteration.

This test runs `_tts_consumer_loop` against a sentence queue containing
two short sentences plus the sentinel. With the fix, the loop returns
within the test timeout. Without the fix, the loop hangs (the test
runner kills it via thread join timeout, asserting the regression).
"""

import asyncio
import queue
import threading
import time
import types

from backend.voice_processor import VoiceProcessor


def _make_voice_processor() -> VoiceProcessor:
    """Construct a VoiceProcessor with only the fields _tts_consumer_loop needs.

    Skips `initialize()` -- the loop does not touch model loading. The
    fields exercised by `_tts_consumer_loop` are: `interrupt_flag`,
    `models.generate_tts_audio`, `message_queue`, `loop`,
    `config.tts_enabled` (implicitly via caller guard, not the loop itself).
    """
    config = types.SimpleNamespace(tts_enabled=True)
    message_queue: asyncio.Queue = asyncio.Queue()
    vp = VoiceProcessor(config=config, message_queue=message_queue)

    # Stub the model manager so generate_tts_audio yields nothing (no audio
    # frames, no MSG_AUDIO_CHUNK emission). The loop still completes its
    # per-sentence iteration cleanly because the per-chunk for-loop is
    # empty.
    vp.models = types.SimpleNamespace(generate_tts_audio=lambda _text: iter([]))

    # Provide a real event loop so run_coroutine_threadsafe targets succeed.
    # The loop does not need to run; we just need a valid reference.
    vp.loop = asyncio.new_event_loop()
    return vp


class TestSentinelPropagation:
    """Regression: sentinel consumed during coalescing must be put back."""

    def test_consumer_loop_exits_when_coalescing_consumes_sentinel(self) -> None:
        """Two short sentences followed by the sentinel must terminate the loop.

        With the bug, the consumer pulls sentence 1 (first_sentence=True so
        skips coalescing), then sentence 2 (first_sentence=False, enters
        coalescing because under min_tts_chars), then pulls the sentinel
        during coalescing and breaks. Pre-fix the outer loop hangs.
        """
        vp = _make_voice_processor()
        sentence_queue: queue.Queue = queue.Queue()
        sentence_queue.put("short.")
        sentence_queue.put("short.")
        sentence_queue.put(None)

        thread = threading.Thread(
            target=vp._tts_consumer_loop,
            kwargs={
                "sentence_queue": sentence_queue,
                "tts_start_time": time.time(),
                "chunk_count": 0,
                "chunk_seq": 0,
                "first_chunk_time": None,
                "first_sentence_time": None,
                "first_sentence": True,
                "min_tts_chars": 64,  # both sentences under -> coalescing fires
            },
            daemon=True,
        )
        thread.start()
        thread.join(timeout=5.0)

        assert not thread.is_alive(), (
            "TTS consumer loop hung -- sentinel was consumed during coalescing "
            "and never reached the outer loop. Fix at "
            "voice_processor.py:329-339 puts the sentinel back."
        )

    def test_consumer_loop_exits_when_sentinel_arrives_at_outer_loop(self) -> None:
        """Control: a single sentence followed by the sentinel terminates.

        first_sentence=True skips coalescing entirely, so the sentinel
        reaches the outer loop directly. This path was never broken; the
        test pins it as the working comparison case.
        """
        vp = _make_voice_processor()
        sentence_queue: queue.Queue = queue.Queue()
        sentence_queue.put("only one sentence.")
        sentence_queue.put(None)

        thread = threading.Thread(
            target=vp._tts_consumer_loop,
            kwargs={
                "sentence_queue": sentence_queue,
                "tts_start_time": time.time(),
                "chunk_count": 0,
                "chunk_seq": 0,
                "first_chunk_time": None,
                "first_sentence_time": None,
                "first_sentence": True,
                "min_tts_chars": 64,
            },
            daemon=True,
        )
        thread.start()
        thread.join(timeout=5.0)

        assert not thread.is_alive(), "Single-sentence path should terminate cleanly"

    def test_consumer_loop_exits_on_interrupt_during_get(self) -> None:
        """Control: interrupt during outer-loop get also terminates."""
        vp = _make_voice_processor()
        sentence_queue: queue.Queue = queue.Queue()
        # No sentences and no sentinel -- consumer will get(timeout=1.0) -> Empty
        # then check interrupt_flag and break.
        vp.interrupt_flag.set()

        thread = threading.Thread(
            target=vp._tts_consumer_loop,
            kwargs={
                "sentence_queue": sentence_queue,
                "tts_start_time": time.time(),
                "chunk_count": 0,
                "chunk_seq": 0,
                "first_chunk_time": None,
                "first_sentence_time": None,
                "first_sentence": True,
                "min_tts_chars": 64,
            },
            daemon=True,
        )
        thread.start()
        thread.join(timeout=5.0)

        assert not thread.is_alive(), "Interrupt during get should break out"


class TestSentinelOnProducerException:
    """Deep review febe-observability-2: the LLM producer raising must not
    leave the TTS consumer spinning forever on an empty queue while holding
    tts_render_lock (voice output dead until restart).
    """

    def test_consumer_released_when_llm_producer_raises(self) -> None:
        vp = _make_voice_processor()

        def _raising_gen(_text):
            raise RuntimeError("llama down")
            yield  # pragma: no cover -- makes this a generator function

        vp.models = types.SimpleNamespace(
            generate_llm_response=_raising_gen,
            trim_to_last_sentence=lambda s: s,
            generate_tts_audio=lambda _t: iter([]),
        )

        vp._process_conversation_turn("hello")

        # The finally-path sentinel lets the consumer drain and release the
        # render lock; pre-fix this acquire timed out.
        acquired = vp.tts_render_lock.acquire(timeout=5.0)
        assert acquired, "TTS consumer still holds tts_render_lock -- sentinel never arrived"
        vp.tts_render_lock.release()


class TestStaleIdleSuppression:
    """Deep review febe-observability-3: a previous turn's late TTS-consumer
    idle emit must not clobber the next turn's think/speak state.
    """

    def _run_loop(self, vp, turn_epoch):
        emitted: list[str] = []
        vp._emit_state_cycle = emitted.append
        q: queue.Queue = queue.Queue()
        q.put(None)
        vp._tts_consumer_loop(
            sentence_queue=q,
            tts_start_time=time.time(),
            chunk_count=0,
            chunk_seq=0,
            first_chunk_time=None,
            first_sentence_time=None,
            first_sentence=True,
            min_tts_chars=40,
            turn_epoch=turn_epoch,
        )
        return emitted

    def test_stale_epoch_skips_idle(self) -> None:
        vp = _make_voice_processor()
        vp._turn_epoch = 2  # a newer turn already started

        emitted = self._run_loop(vp, turn_epoch=1)

        assert "idle" not in emitted

    def test_current_epoch_emits_idle(self) -> None:
        vp = _make_voice_processor()
        vp._turn_epoch = 1

        emitted = self._run_loop(vp, turn_epoch=1)

        assert emitted == ["idle"]
