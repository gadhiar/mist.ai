# Voice Pipeline Streaming Optimization -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce time-to-first-audio from 12-19s to ~3.5s by overlapping LLM generation, TTS synthesis, and audio delivery via pipeline parallelism.

**Architecture:** Stream LLM tokens through a sentence boundary detector, feed complete sentences to TTS as they arrive, send audio to the client while later sentences are still generating. The LLM, TTS, and audio delivery stages run concurrently on different data.

**Tech Stack:** Python 3.11, asyncio, threading, Ollama streaming API, Chatterbox Turbo, WebSocket

**Spec:** `docs/superpowers/specs/2026-03-26-voice-pipeline-optimization-design.md`

---

## File Map

### New Files

| File | Responsibility |
|------|---------------|
| `backend/sentence_detector.py` | Sentence boundary detection in token streams |
| `tests/unit/test_sentence_detector.py` | Tests for boundary detection |
| `tests/unit/test_knowledge_streaming.py` | Tests for true LLM token streaming |

### Modified Files

| File | Change |
|------|--------|
| `backend/chat/knowledge_integration.py` | Add `generate_response_streaming_tokens()` for true token-level streaming |
| `backend/voice_processor.py` | Rewrite `_process_conversation_turn()` with pipeline parallelism |
| `backend/voice_models/model_manager.py` | Simplify `_generate_tts_chatterbox()` for single-sentence generation |
| `backend/knowledge/embeddings/embedding_generator.py` | Add eager model loading method |

---

## Task 1: Sentence Boundary Detector

**Files:**
- Create: `backend/sentence_detector.py`
- Create: `tests/unit/test_sentence_detector.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_sentence_detector.py
"""Tests for SentenceBoundaryDetector."""
import pytest
from backend.sentence_detector import SentenceBoundaryDetector


class TestSentenceBoundaryDetector:
    """Tests for sentence boundary detection in token streams."""

    def setup_method(self):
        self.detector = SentenceBoundaryDetector()

    def test_simple_sentence(self):
        """Period followed by space triggers boundary."""
        result = self.detector.feed("Hello there. ")
        assert result == ["Hello there."]

    def test_multiple_sentences(self):
        """Multiple sentences in one feed."""
        result = self.detector.feed("First sentence. Second sentence. ")
        assert result == ["First sentence.", "Second sentence."]

    def test_exclamation_mark(self):
        """Exclamation triggers boundary."""
        result = self.detector.feed("Wow! ")
        # Too short (<20 chars), stays in buffer
        assert result == []
        result = self.detector.feed("That is really great! ")
        # Buffer had "Wow! ", now "Wow! That is really great!"
        assert len(result) == 1
        assert "Wow" in result[0]

    def test_question_mark(self):
        """Question mark triggers boundary."""
        result = self.detector.feed("How are you doing today? ")
        assert result == ["How are you doing today?"]

    def test_abbreviation_no_split(self):
        """Common abbreviations do not trigger boundary."""
        result = self.detector.feed("Dr. Smith went to the store. ")
        assert result == ["Dr. Smith went to the store."]

    def test_decimal_no_split(self):
        """Decimal numbers do not trigger boundary."""
        result = self.detector.feed("The value is 3.14 meters. ")
        assert result == ["The value is 3.14 meters."]

    def test_ellipsis(self):
        """Ellipsis treated as single boundary."""
        result = self.detector.feed("Well... I suppose that works. ")
        assert len(result) >= 1

    def test_accumulation_across_feeds(self):
        """Tokens accumulate across multiple feed calls."""
        assert self.detector.feed("Hello") == []
        assert self.detector.feed(" there") == []
        assert self.detector.feed(". ") == ["Hello there."]

    def test_flush_remaining(self):
        """Flush emits remaining buffer."""
        self.detector.feed("Partial sentence without")
        result = self.detector.flush()
        assert result == ["Partial sentence without"]

    def test_flush_empty(self):
        """Flush on empty buffer returns nothing."""
        assert self.detector.flush() == []

    def test_min_length_merging(self):
        """Short sentences merge with previous."""
        result = self.detector.feed("This is a full sentence. Yes. And another sentence here. ")
        # "Yes." is <20 chars, should merge
        assert len(result) == 2

    def test_closing_quote(self):
        """Closing quote after terminal punctuation."""
        tokens = 'She said "hello there." Then she left. '
        result = self.detector.feed(tokens)
        assert len(result) == 2

    def test_list_marker_no_split(self):
        """Period after single digit (list marker) does not split."""
        result = self.detector.feed("Here are the steps: 1. First do this thing. ")
        assert "1." in result[0]

    def test_end_of_stream(self):
        """Feed tokens then flush gets remaining buffer."""
        result1 = self.detector.feed("First sentence. ")
        assert result1 == ["First sentence."]
        self.detector.feed("Second without ending")
        sentences = self.detector.flush()
        assert sentences == ["Second without ending"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose run --rm --no-deps mist-backend pytest tests/unit/test_sentence_detector.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'backend.sentence_detector')

- [ ] **Step 3: Implement SentenceBoundaryDetector**

```python
# backend/sentence_detector.py
"""Sentence boundary detection for streaming LLM token output.

Accumulates tokens and emits complete sentences as soon as a boundary is
detected.  Handles abbreviations, decimals, ellipses, and list markers.
"""

import re

# Common abbreviations that should NOT trigger sentence splits
_ABBREVIATIONS = frozenset({
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "vs", "etc",
    "inc", "ltd", "dept", "est", "approx", "vol", "no", "fig",
    "eq", "st", "ave", "blvd", "govt", "gen", "sgt", "cpl",
    "pvt", "rev", "hon", "corp",
})

# Regex: terminal punctuation (.!?) optionally followed by closing quote/bracket,
# then whitespace or end-of-string
_BOUNDARY_RE = re.compile(
    r'([.!?]["\')}\]]?)'  # terminal punct + optional close
    r'(\s+)',              # followed by whitespace
)


class SentenceBoundaryDetector:
    """Detects sentence boundaries in a streaming token sequence.

    Feed tokens as they arrive from the LLM. The detector returns
    a list of complete sentences (0 or more) after each feed call.
    Call flush() at end-of-stream to emit any remaining text.
    """

    MIN_SENTENCE_LENGTH = 20

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, token: str) -> list[str]:
        """Feed a token, return list of complete sentences (0 or more).

        Args:
            token: Next token from the LLM stream.

        Returns:
            List of complete sentences detected. May be empty.
        """
        self._buffer += token
        sentences: list[str] = []

        while True:
            boundary = self._find_boundary()
            if boundary is None:
                break

            sentence = self._buffer[:boundary].strip()
            self._buffer = self._buffer[boundary:].lstrip()

            if len(sentence) < self.MIN_SENTENCE_LENGTH:
                if sentences:
                    # Merge short sentence with previous
                    sentences[-1] += " " + sentence
                else:
                    # Too short and nothing to merge -- put back
                    self._buffer = sentence + " " + self._buffer
                    break
            else:
                sentences.append(sentence)

        return sentences

    def flush(self) -> list[str]:
        """Flush remaining buffer as final sentence.

        Returns:
            List containing the remaining text, or empty if buffer is empty.
        """
        remaining = self._buffer.strip()
        self._buffer = ""
        if remaining:
            return [remaining]
        return []

    def _find_boundary(self) -> int | None:
        """Find first sentence boundary position in buffer.

        Returns:
            Character index after the boundary (start of next sentence),
            or None if no boundary found.
        """
        for match in _BOUNDARY_RE.finditer(self._buffer):
            pos = match.start()
            end = match.end()

            # Check: is this an abbreviation?
            if self._buffer[match.start(1)] == ".":
                if self._is_abbreviation(pos):
                    continue
                if self._is_decimal(pos):
                    continue
                if self._is_list_marker(pos):
                    continue
                if self._is_ellipsis(pos):
                    continue

            # Valid boundary -- return position after the punctuation
            return end

        return None

    def _is_abbreviation(self, dot_pos: int) -> bool:
        """Check if period at dot_pos is part of an abbreviation."""
        # Find the word before the dot
        start = dot_pos
        while start > 0 and self._buffer[start - 1].isalpha():
            start -= 1
        word = self._buffer[start:dot_pos].lower()
        return word in _ABBREVIATIONS

    def _is_decimal(self, dot_pos: int) -> bool:
        """Check if period at dot_pos is a decimal point between digits."""
        if dot_pos == 0 or dot_pos >= len(self._buffer) - 1:
            return False
        return self._buffer[dot_pos - 1].isdigit() and self._buffer[dot_pos + 1].isdigit()

    def _is_list_marker(self, dot_pos: int) -> bool:
        """Check if period is after a single digit (list marker like '1.')."""
        if dot_pos < 1:
            return False
        # "1." at start or after whitespace
        if self._buffer[dot_pos - 1].isdigit():
            if dot_pos < 2 or not self._buffer[dot_pos - 2].isalpha():
                return True
        return False

    def _is_ellipsis(self, dot_pos: int) -> bool:
        """Check if period is part of an ellipsis (...)."""
        if dot_pos >= 2 and self._buffer[dot_pos - 2 : dot_pos + 1] == "...":
            return True
        if (
            dot_pos < len(self._buffer) - 2
            and self._buffer[dot_pos : dot_pos + 3] == "..."
        ):
            return True
        return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm --no-deps mist-backend pytest tests/unit/test_sentence_detector.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/sentence_detector.py tests/unit/test_sentence_detector.py
git commit -m "feat(voice): add SentenceBoundaryDetector for streaming pipeline"
```

---

## Task 2: True LLM Token Streaming in KnowledgeIntegration

**Files:**
- Modify: `backend/chat/knowledge_integration.py`
- Create: `tests/unit/test_knowledge_streaming.py`

This is the prerequisite for pipeline parallelism. The current `generate_response_streaming()` blocks on the full response. We add a new method `generate_tokens_streaming()` that yields actual tokens.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_knowledge_streaming.py
"""Tests for true token streaming in KnowledgeIntegration."""
from unittest.mock import MagicMock, patch


class TestKnowledgeStreamingTokens:
    """Test that generate_tokens_streaming yields individual tokens."""

    def test_yields_multiple_tokens(self):
        """Should yield individual tokens, not a single blob."""
        from backend.chat.knowledge_integration import KnowledgeIntegration

        ki = KnowledgeIntegration.__new__(KnowledgeIntegration)
        ki.enabled = True
        ki.current_session_id = "test"

        # Mock conversation handler with realistic attributes
        handler = MagicMock()
        handler.config.auto_inject_docs = False  # Skip RAG for this test
        handler.config.model = "qwen2.5:7b-instruct"
        session = MagicMock()
        session.messages = []
        handler.get_or_create_session.return_value = session
        handler.event_store = None
        ki.conversation_handler = handler

        mock_chunks = [
            {"message": {"content": "Hello"}},
            {"message": {"content": " there"}},
            {"message": {"content": "."}},
        ]

        with patch("backend.chat.knowledge_integration.ollama") as mock_ollama:
            mock_ollama.chat.return_value = iter(mock_chunks)

            tokens = list(ki.generate_tokens_streaming(
                "hello how are you doing today", event_loop=MagicMock()
            ))

        assert len(tokens) == 3
        assert tokens[0] == "Hello"
        assert tokens[1] == " there"
        assert tokens[2] == "."
        session.add_message.assert_any_call("user", "hello how are you doing today")
        session.add_message.assert_any_call("assistant", "Hello there.")

    def test_rag_context_included_when_available(self):
        """RAG results should be included in messages sent to Ollama."""
        from backend.chat.knowledge_integration import KnowledgeIntegration

        ki = KnowledgeIntegration.__new__(KnowledgeIntegration)
        ki.enabled = True
        ki.current_session_id = "test"

        handler = MagicMock()
        handler.config.auto_inject_docs = True
        handler.config.auto_inject_limit = 3
        handler.config.auto_inject_threshold = 0.4
        handler.config.model = "qwen2.5:7b-instruct"
        session = MagicMock()
        session.messages = []
        handler.get_or_create_session.return_value = session
        handler.event_store = None

        retrieval_result = MagicMock()
        retrieval_result.total_facts = 2
        retrieval_result.formatted_context = "Known facts: User likes Python."
        handler.retriever.retrieve.return_value = retrieval_result
        ki.conversation_handler = handler

        mock_loop = MagicMock()

        mock_chunks = [{"message": {"content": "Sure!"}}]

        with patch("backend.chat.knowledge_integration.ollama") as mock_ollama, \
             patch("asyncio.run_coroutine_threadsafe") as mock_rcts:
            # Make the future return the retrieval result
            mock_future = MagicMock()
            mock_future.result.return_value = retrieval_result
            mock_rcts.return_value = mock_future

            mock_ollama.chat.return_value = iter(mock_chunks)

            tokens = list(ki.generate_tokens_streaming(
                "what programming languages do I know",
                event_loop=mock_loop,
            ))

        assert tokens == ["Sure!"]
        # Verify RAG context was included in Ollama call
        call_args = mock_ollama.chat.call_args
        messages = call_args.kwargs["messages"]
        context_messages = [m for m in messages if "Known facts" in m.get("content", "")]
        assert len(context_messages) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm --no-deps mist-backend pytest tests/unit/test_knowledge_streaming.py -v`
Expected: FAIL (AttributeError: 'KnowledgeIntegration' has no attribute 'generate_tokens_streaming')

- [ ] **Step 3: Implement generate_tokens_streaming()**

Add to `backend/chat/knowledge_integration.py`:

```python
def generate_tokens_streaming(
    self,
    user_text: str,
    session_id: str | None = None,
    event_loop: asyncio.AbstractEventLoop | None = None,
) -> Generator[str, None, None]:
    """Generate LLM response with true token-level streaming.

    Runs RAG retrieval, builds a voice-optimized prompt, then streams
    tokens directly from Ollama. Bypasses ConversationHandler's tool-calling
    chain for lower latency.

    Known limitations vs handle_message():
    - No tool calling (voice mode does not need mid-stream tools)
    - EventStore recording added after stream completes
    - Background extraction skipped (TODO: add in future)

    Args:
        user_text: User's message.
        session_id: Optional session ID.
        event_loop: Event loop for async RAG retrieval.

    Yields:
        Individual tokens from the LLM stream.
    """
    if not self.enabled or not self.conversation_handler:
        yield "I'm sorry, the knowledge system is not available right now."
        return

    import ollama

    try:
        sid = session_id or self.current_session_id
        handler = self.conversation_handler

        # Step 1: RAG retrieval
        retrieval_result = None
        if (
            handler.retriever
            and handler.config.auto_inject_docs
            and len(user_text.split()) >= 3
            and event_loop is not None
        ):
            try:
                future = asyncio.run_coroutine_threadsafe(
                    handler.retriever.retrieve(
                        query=user_text,
                        user_id="User",
                        limit=handler.config.auto_inject_limit,
                        similarity_threshold=handler.config.auto_inject_threshold,
                    ),
                    event_loop,
                )
                retrieval_result = future.result(timeout=30)
                if retrieval_result and retrieval_result.total_facts > 0:
                    logger.info(
                        "RAG retrieved %d facts for voice streaming",
                        retrieval_result.total_facts,
                    )
            except Exception as e:
                logger.warning("RAG retrieval failed in voice streaming: %s", e)

        # Step 2: Build voice-optimized messages
        session = handler.get_or_create_session(sid, "User")
        session.add_message("user", user_text)

        system_prompt = (
            "You are M.I.S.T, a helpful voice assistant and friend to your "
            "creator, Raj Gadhia.\n\n"
            "Response Guidelines:\n"
            "- For simple questions or greetings: 1-3 sentences\n"
            "- For detailed requests: provide complete, thorough responses\n"
            "- Use a warm, friendly tone suitable for spoken conversation\n"
            "- Prioritize correctness, accuracy, and thoroughness\n"
            "- Don't artificially truncate content the user explicitly requested"
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Add RAG context if available
        if retrieval_result and retrieval_result.total_facts > 0:
            messages.append({
                "role": "system",
                "content": retrieval_result.formatted_context,
            })

        # Add conversation history (last 10 turns)
        for msg in session.messages[-20:]:  # 20 entries = 10 turns
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Step 3: Stream tokens directly from Ollama
        response = ollama.chat(
            model=handler.config.model,
            messages=messages,
            stream=True,
            options={"num_predict": 400, "temperature": 0.7, "top_p": 0.9},
        )

        full_response = ""
        for chunk in response:
            if "message" in chunk and "content" in chunk["message"]:
                token = chunk["message"]["content"]
                full_response += token
                yield token

        # Step 4: Record to session history
        session.add_message("assistant", full_response)

        # Step 5: Record to EventStore (Layer 1 audit trail)
        if handler.event_store:
            try:
                handler._record_turn_event(
                    session_id=sid,
                    user_message=user_text,
                    assistant_message=full_response,
                )
            except Exception as e:
                logger.warning("EventStore recording failed: %s", e)

    except Exception as e:
        logger.error("Error in streaming knowledge integration: %s", e, exc_info=True)
        yield f"I encountered an error: {e!s}"
```

Note: This bypasses `ConversationHandler.handle_message()` and the LangChain tool-calling chain for the voice path. Voice mode does not need tool-calling mid-stream. The full `handle_message()` path is preserved for non-voice use. The `import ollama` at the top of the method body is a placeholder -- see Step 4 below for the module-level import note.

- [ ] **Step 4: Add module-level `import ollama` to knowledge_integration.py**

The `import ollama` statement must be added at the module level of `backend/chat/knowledge_integration.py` (alongside other top-level imports), not inside the method body. The inline import shown in Step 3 above is only for plan readability -- when implementing, move it to the module's import section.

- [ ] **Step 5: Run tests**

Run: `docker compose run --rm --no-deps mist-backend pytest tests/unit/test_knowledge_streaming.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/chat/knowledge_integration.py tests/unit/test_knowledge_streaming.py
git commit -m "feat(voice): add true token streaming to KnowledgeIntegration"
```

---

## Task 3: Wire Token Streaming into ModelManager

**Files:**
- Modify: `backend/voice_models/model_manager.py`

Update `generate_llm_response()` to use the new token-level streaming when knowledge integration is available.

- [ ] **Step 1: Update generate_llm_response() to use generate_tokens_streaming()**

In `model_manager.py`, the `generate_llm_response()` method currently calls `self.knowledge.generate_response_streaming()`. Change the knowledge path to call `self.knowledge.generate_tokens_streaming()` instead:

```python
def generate_llm_response(self, user_text):
    if self.knowledge and self.knowledge.is_enabled():
        logger.info("Using knowledge-augmented LLM response (streaming tokens)")
        yield from self.knowledge.generate_tokens_streaming(
            user_text, event_loop=self.event_loop
        )
    else:
        # Fallback to standard Ollama (already streams tokens)
        ...  # existing code unchanged
```

- [ ] **Step 2: Verify existing tests still pass**

Run: `docker compose run --rm --no-deps mist-backend pytest tests/unit/ -x -q`
Expected: All existing tests PASS

- [ ] **Step 3: Commit**

```bash
git add backend/voice_models/model_manager.py
git commit -m "feat(voice): wire token streaming into LLM response generator"
```

---

## Task 4: Pipeline Parallelism in VoiceProcessor

**Files:**
- Modify: `backend/voice_processor.py`

This is the core orchestration change. Rewrite `_process_conversation_turn()` to run LLM and TTS concurrently via a sentence queue.

- [ ] **Step 1: Add the TTS consumer method**

Add `_tts_consumer()` to VoiceProcessor:

```python
def _tts_consumer(self, sentence_queue: queue.Queue, tts_start_time: float) -> None:
    """Consume sentences from queue, generate TTS, send audio to client.

    Runs in a dedicated thread. Processes sentences as they arrive
    from the LLM producer, generating and sending audio for each.

    Args:
        sentence_queue: Queue of sentences to synthesize. None = stop signal.
        tts_start_time: Timestamp when TTS phase started (for logging).
    """
    chunk_count = 0
    first_chunk_time = None

    while True:
        sentence = sentence_queue.get()
        if sentence is None:
            break

        if self.interrupt_flag.is_set():
            break

        log_timestamp(f"TTS: Generating sentence ({len(sentence)} chars)")

        for audio_chunk in self.models.generate_tts_audio(sentence):
            if self.interrupt_flag.is_set():
                break

            chunk_count += 1
            if first_chunk_time is None:
                first_chunk_time = time.time() - tts_start_time
                log_timestamp(
                    f"TTS: First audio chunk ({first_chunk_time:.2f}s from TTS start)"
                )

            if isinstance(audio_chunk, torch.Tensor):
                audio_np = audio_chunk.cpu().numpy().astype(np.float32)
            else:
                audio_np = audio_chunk.astype(np.float32)

            asyncio.run_coroutine_threadsafe(
                self.message_queue.put({
                    "type": "audio_chunk",
                    "audio": audio_np.tolist(),
                    "sample_rate": 24000,
                    "chunk_num": chunk_count,
                }),
                self.loop,
            )

    tts_total = time.time() - tts_start_time
    log_timestamp(f"TTS consumer done ({tts_total:.2f}s, {chunk_count} chunks)")
```

- [ ] **Step 2: Rewrite _process_conversation_turn() with pipeline parallelism**

Replace the sequential LLM-then-TTS flow with concurrent pipeline:

```python
def _process_conversation_turn(self, user_text):
    """Process one conversation turn with LLM-TTS pipeline parallelism."""
    if not self.generation_lock.acquire(blocking=False):
        log_timestamp("Generation already in progress, skipping")
        return

    try:
        from backend.sentence_detector import SentenceBoundaryDetector
        from request_context import new_request_id, spawn_with_context

        new_request_id()
        log_timestamp(f"Starting conversation turn for: '{user_text}'")

        self.interrupt_flag.clear()
        self.is_speaking = True

        # === LLM + TTS Pipeline ===
        sentence_detector = SentenceBoundaryDetector()
        sentence_queue = queue.Queue()
        tts_start_time = time.time()

        # Start TTS consumer thread (reads sentences, generates audio)
        tts_thread = spawn_with_context(
            self._tts_consumer, sentence_queue, tts_start_time
        )

        # LLM producer: stream tokens, detect sentences, feed TTS
        log_timestamp("LLM: Generating response (streaming)...")
        llm_start = time.time()
        full_response = ""

        for token in self.models.generate_llm_response(user_text):
            if self.interrupt_flag.is_set():
                log_timestamp("LLM generation interrupted")
                break
            full_response += token

            # Send token to client for real-time text display
            asyncio.run_coroutine_threadsafe(
                self.message_queue.put({"type": "llm_token", "token": token}),
                self.loop,
            )

            # Detect sentence boundaries and feed TTS
            if self.config.tts_enabled:
                sentences = sentence_detector.feed(token)
                for sentence in sentences:
                    sentence_queue.put(sentence)

        llm_time = time.time() - llm_start

        # Trim to last complete sentence
        full_response = self.models.trim_to_last_sentence(full_response)
        log_timestamp(f"LLM complete ({llm_time:.2f}s, {len(full_response)} chars)")

        # Send full response
        asyncio.run_coroutine_threadsafe(
            self.message_queue.put({"type": "llm_response", "text": full_response}),
            self.loop,
        )

        # Flush remaining text to TTS
        if self.config.tts_enabled:
            for sentence in sentence_detector.flush():
                sentence_queue.put(sentence)
            sentence_queue.put(None)  # Signal end
            tts_thread.join(timeout=120)
            if tts_thread.is_alive():
                logger.warning("TTS consumer thread did not finish within 120s")
        else:
            log_timestamp("TTS: Disabled (text-only mode)")

        # Send completion signal
        asyncio.run_coroutine_threadsafe(
            self.message_queue.put({"type": "audio_complete"}),
            self.loop,
        )

    except Exception as e:
        logger.error("Error in conversation turn: %s", e, exc_info=True)
        asyncio.run_coroutine_threadsafe(
            self.message_queue.put({"type": "error", "message": f"Generation error: {e}"}),
            self.loop,
        )

    finally:
        self.is_speaking = False
        self.generation_lock.release()

        # Check for pending input
        with self.input_lock:
            if self.latest_user_input and not self.interrupt_flag.is_set():
                pending_input = self.latest_user_input
                self.latest_user_input = None
                log_timestamp(f"Processing pending input: '{pending_input}'")
                spawn_with_context(self._process_conversation_turn, pending_input)
```

- [ ] **Step 3: Run full backend test suite**

Run: `docker compose run --rm --no-deps mist-backend pytest tests/unit/ -x -q`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add backend/voice_processor.py
git commit -m "feat(voice): pipeline parallelism in conversation turn processing"
```

---

## Task 5: Simplify TTS for Single-Sentence Generation

**Files:**
- Modify: `backend/voice_models/model_manager.py`

The current `_generate_tts_chatterbox()` groups sentences into 50-word chunks. With pipeline parallelism, each sentence arrives individually. Simplify the method to process single sentences directly.

- [ ] **Step 1: Simplify _generate_tts_chatterbox()**

Replace the chunking logic with direct single-text generation:

```python
def _generate_tts_chatterbox(self, text):
    """Generate TTS audio using Chatterbox engine for a single sentence.

    With pipeline parallelism, text arrives as individual sentences.
    No re-chunking needed -- each call generates audio for one sentence.
    """
    preprocessed_text = preprocess_text_for_tts(text)

    with self.tts_lock:
        self.tts_generation_id += 1
        current_gen_id = self.tts_generation_id

        while not self.tts_result_queue.empty():
            try:
                self.tts_result_queue.get_nowait()
            except queue.Empty:
                break

    logger.info("Chatterbox: generating '%s...' (%d chars)",
                preprocessed_text[:50], len(preprocessed_text))

    self.tts_request_queue.put((current_gen_id, preprocessed_text))

    TIMEOUT = 120.0
    while True:
        try:
            msg_type, gen_id, data = self.tts_result_queue.get(timeout=TIMEOUT)
        except queue.Empty:
            raise RuntimeError(
                f"TTS generation timeout after {TIMEOUT}s"
            ) from None

        if gen_id != current_gen_id:
            continue

        if msg_type == "chunk":
            yield data
        elif msg_type == "complete":
            logger.info("Chatterbox: generation complete")
            break
        elif msg_type == "error":
            raise RuntimeError(f"TTS generation failed: {data}")
```

- [ ] **Step 2: Remove CUDA sync and empty_cache between sentences**

In `_tts_worker_chatterbox()`, remove the `torch.cuda.synchronize()` calls before and after generation. Keep only the synchronize in the error path. Remove `torch.cuda.empty_cache()` from the caller (now in voice_processor's `_tts_consumer`).

- [ ] **Step 3: Run tests**

Run: `docker compose run --rm --no-deps mist-backend pytest tests/unit/ -x -q`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add backend/voice_models/model_manager.py
git commit -m "refactor(voice): simplify TTS for single-sentence generation, remove CUDA syncs"
```

---

## Task 6: Eager Embedding Model Loading

**Files:**
- Modify: `backend/knowledge/embeddings/embedding_generator.py`
- Modify: `backend/voice_models/model_manager.py`

- [ ] **Step 1: Add eager loading to EmbeddingGenerator**

Read `backend/knowledge/embeddings/embedding_generator.py` and add a `warmup()` or `load_model()` method that forces the sentence-transformers model to load immediately rather than on first encode.

- [ ] **Step 2: Call warmup during model_manager.load_all_models()**

After loading Whisper and starting TTS worker, call the embedding generator's warmup method so the model is ready before the first conversation turn.

- [ ] **Step 3: Run tests**

Run: `docker compose run --rm --no-deps mist-backend pytest tests/unit/ -x -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add backend/knowledge/embeddings/embedding_generator.py backend/voice_models/model_manager.py
git commit -m "perf(voice): eager embedding model loading at startup"
```

---

## Task 7: Integration Test

- [ ] **Step 1: Start the stack**

```bash
python scripts/start_dev.py --deps-only
```

- [ ] **Step 2: Run a multi-turn voice conversation**

Test 3-4 turns including short, medium, and long responses. Observe:
- Time-to-first-audio (should be ~3.5-4.5s regardless of response length)
- Audio continuity (no long gaps between sentences)
- Text token streaming (should appear word-by-word in Flutter)
- Interrupt behavior (speak during TTS -- should stop generation)

- [ ] **Step 3: Analyze logs**

```bash
grep -E "Starting conversation|LLM complete|TTS: First audio|TTS consumer done" logs/mist-backend.log
```

Compare time-to-first-audio against the 12-19s baseline.

- [ ] **Step 4: Commit log analysis**

Document the before/after comparison.
