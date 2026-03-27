# Voice Pipeline Latency Optimization -- Comprehensive Design Spec

**Date:** 2026-03-26
**Status:** Phase 1 + 3 Implemented (Phase 2 pending: binary WebSocket audio transport)
**Project:** MIST.AI -- Voice Pipeline
**Author:** Pipeline Performance Analysis

---

## 1. Current State Analysis

### 1.1 Hardware Profile

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 4070 SUPER, 12 GB VRAM |
| CPU | AMD Ryzen 7 7800X3D, 8 cores / 16 threads, 4.2 GHz |
| RAM | 32 GB |
| CUDA | 12.4 (container), driver 591.74 |
| Runtime | Docker (nvidia/cuda:12.4.0-devel-ubuntu22.04) |

### 1.2 Model VRAM Budget (Estimated)

| Model | VRAM (approx) | Residency |
|-------|---------------|-----------|
| Chatterbox Turbo | ~3.9 GB | Persistent (TTS worker thread) |
| Whisper base | ~0.3 GB | Persistent (STT) |
| all-MiniLM-L6-v2 | ~0.1 GB | Lazy-loaded, evicted between calls |
| Qwen 2.5 7B (Ollama) | ~4.5 GB (Q4) | Separate container, own VRAM allocation |
| PyTorch overhead | ~0.5 GB | CUDA context, allocator |
| **Total** | **~9.3 GB** | **Headroom: ~2.7 GB** |

Qwen runs in a separate Ollama container and manages its own VRAM. The backend
container holds Whisper + Chatterbox + embeddings. No GPU contention between
Qwen and the backend models because they are in separate processes with
separate CUDA contexts. However, Whisper, Chatterbox, and the embedding model
share a single CUDA context inside the backend container.

### 1.3 Log-Derived Pipeline Timings

The log contains two server sessions. Session 2 (05:00 series, FRIDAY voice
profile, 46s reference) provides the cleanest data because LLM was already
warm. All timings below are from session 2 unless noted.

#### Conversation Turn 1: "hi mist how are you doing" (61 chars response)

```
05:01:18.529  Conversation turn start
05:01:18.530  LLM generation start
05:01:18.532  RAG retrieval start (embedding model lazy-load)
05:01:19.107  Embedding model loaded                           +575ms
05:01:19.137  RAG retrieval complete (606.2ms total, 0 facts)
05:01:19.138  LLM inference to Ollama begins
05:01:21.361  Ollama HTTP response complete
05:01:21.734  LLM generation complete                          3.20s total
05:01:21.734  TTS generation start
05:01:21.736  TTS chunk submitted (61 chars, 1 chunk)
05:01:23.880  Chatterbox EOS at step 86
05:01:24.578  TTS audio ready (3.40s audio)                    2.84s TTS
05:01:24.635  TTS generation complete (2.90s)

TOTAL: speech-end to audio-sent = 6.10s
  RAG:  0.61s  (embedding model cold-load: 0.58s)
  LLM:  3.20s  (includes RAG; net LLM inference ~2.60s)
  TTS:  2.90s  (generating 3.40s audio)
```

#### Conversation Turn 2: "Give me samples to test your voice generation" (298 chars, 2 chunks)

```
05:01:46.709  Conversation turn start
05:01:46.710  LLM generation start
05:01:46.727  RAG retrieval complete (15.7ms, cached embeddings)
05:01:48.961  Ollama HTTP response complete
05:01:50.378  LLM generation complete                          3.67s total
05:01:50.379  TTS generation start
05:01:50.380  TTS chunk 1/2 submitted (222 chars)
05:01:58.226  Chatterbox EOS at step 302
05:01:59.035  TTS chunk 1 audio ready (12.04s audio)           8.66s TTS
05:01:59.168  TTS chunk 2/2 submitted (73 chars)
05:02:01.421  Chatterbox EOS at step 96
05:02:02.072  TTS chunk 2 audio ready (3.80s audio)            2.90s TTS
05:02:02.122  TTS generation complete (11.74s total)

TOTAL: speech-end to first-audio-sent  = 12.33s
TOTAL: speech-end to last-audio-sent   = 15.41s
  RAG:  0.02s  (cached)
  LLM:  3.67s
  TTS:  11.74s (chunk 1: 8.66s, chunk 2: 2.90s)
```

#### Session 1 Data Points (Jarvis voice, 25s reference)

| Turn | Input | Response | LLM | TTS | TTS Chunks | First Audio | Total |
|------|-------|----------|-----|-----|------------|-------------|-------|
| 1 | "hi mist how are you doing" | 154 chars | 4.90s | 8.01s | 1 (154c) | 12.91s | 12.91s |
| 2 | "im trying to test..." | 188 chars | 3.08s | 9.44s | 1 (188c) | 12.52s | 12.52s |
| 3 | "Try and test longer..." | 423 chars | 3.14s | 25.59s | 2 (236c+184c) | 19.18s | 28.73s |

### 1.4 Current Pipeline Flow (Sequential)

```
USER SPEAKS          VAD/STT              LLM                  TTS                  CLIENT
    |                  |                   |                    |                     |
    |--speech-end----->|                   |                    |                     |
    |                  |--transcribe------>|                    |                     |
    |                  |   (Whisper)       |                    |                     |
    |                  |                   |                    |                     |
    |                  |<--text------------|                    |                     |
    |                  |                   |                    |                     |
    |                  |---RAG retrieval-->|                    |                     |
    |                  |   (embedding +    |                    |                     |
    |                  |    Neo4j query)   |                    |                     |
    |                  |                   |                    |                     |
    |                  |---Ollama stream-->|                    |                     |
    |                  |   (tokens)        |                    |                     |
    |                  |   ...             |                    |                     |
    |                  |<--full response---|                    |                     |
    |                  |                   |                    |                     |
    |                  |                   |                    |                     |
    |                  |                   |---chunk text 1---->|                     |
    |                  |                   |                    |---generate--------->|
    |                  |                   |                    |   (Chatterbox)      |
    |                  |                   |                    |<--audio chunk 1-----|
    |                  |                   |                    |-----send (JSON)---->|
    |                  |                   |                    |                     |
    |                  |                   |---chunk text 2---->|                     |
    |                  |                   |                    |---generate--------->|
    |                  |                   |                    |<--audio chunk 2-----|
    |                  |                   |                    |-----send (JSON)---->|
    |                  |                   |                    |                     |
    |                  |                   |                    |--audio_complete---->|
    |                  |                   |                    |                     |
    |<---------------------------------------------------------|-----playback------->|
```

**Critical observation:** Every stage is fully sequential. TTS does not start
until the entire LLM response is complete. Audio is not sent until each TTS
chunk finishes generating. The client does not begin playback until the first
complete audio chunk arrives.

---

## 2. Bottleneck Ranking (by Impact on Perceived Latency)

### Rank 1: LLM response blocks TTS entirely (CRITICAL)

**Measured impact:** 3.0--5.0 seconds of dead silence

The `generate_llm_response()` generator in `voice_processor.py` (line 175)
iterates ALL tokens before `generate_tts_audio()` begins (line 213). The LLM
streams tokens via Ollama but the voice processor accumulates the full response
before passing it to TTS.

Worse: `KnowledgeIntegration.generate_response_streaming()` is not actually
streaming. Line 96 of `knowledge_integration.py` yields the entire response
as a single string after `handle_message()` completes:

```python
# knowledge_integration.py:94-96
# Yield complete response
# TODO: Future enhancement - stream tokens as they're generated
yield response
```

This means the "streaming" LLM call is actually a blocking call that waits
for the full Ollama response + RAG retrieval + tool execution before yielding
a single blob. The token-level streaming visible in `voice_processor.py`
line 175 is illusory when knowledge integration is enabled.

### Rank 2: TTS generates full chunk before audio is sent (CRITICAL)

**Measured impact:** 2.8--16.0 seconds per chunk (scales with text length)

Chatterbox Turbo is a non-streaming model. `ChatterboxTTS.generate()` returns
a complete audio tensor only after the entire generation finishes. For a 222-char
chunk, this takes 8.66 seconds. For 236 chars, 16.04 seconds.

The TTS worker thread (model_manager.py line 158) puts the entire audio result
into the result queue as a single "chunk" message. There is no sub-chunk
streaming.

### Rank 3: Audio serialization as JSON float arrays (HIGH)

**Measured impact:** Network bandwidth waste ~10x, serialization CPU overhead

Audio is sent via `audio_np.tolist()` (voice_processor.py line 235) which
converts float32 PCM to a JSON array of numbers. A 10-second audio clip at
24kHz = 240,000 float32 samples = ~960KB as binary PCM16, but ~3.8MB as JSON
text (each float like "0.0234375" is ~10 bytes + comma). This is addressed by
the existing binary WebSocket protocol design (2026-03-24-voice-streaming-pipeline-design.md)
but has not been implemented yet.

### Rank 4: Embedding model lazy-loading on first query (MEDIUM)

**Measured impact:** 575ms on first query per server restart

The sentence-transformers embedding model (`all-MiniLM-L6-v2`) is loaded lazily
on first RAG retrieval. Subsequent queries are fast (15ms). This adds 0.6s
to the first conversation turn only.

### Rank 5: RAG retrieval on every turn (LOW-MEDIUM)

**Measured impact:** 16--606ms

After the embedding model is cached, RAG retrieval takes ~16ms, which is
negligible. The first-query 606ms is mostly embedding model loading (Rank 4).
However, RAG runs synchronously inside the LLM generation path, blocking
the first Ollama token from being sent.

### Rank 6: Thread spawning and queue overhead (LOW)

**Measured impact:** <10ms

`spawn_with_context()` and the threading.Thread creation for each conversation
turn adds minimal overhead. The `queue.Queue` polling in the TTS worker (100ms
timeout) adds up to 100ms worst-case latency on message pickup, though
typically much less.

### Rank 7: CUDA synchronization calls (LOW)

**Measured impact:** <50ms per call

`torch.cuda.synchronize()` is called before and after each TTS generation
(model_manager.py lines 155, 166). These force CPU-GPU sync and add small
but measurable overhead. The `torch.cuda.empty_cache()` between chunks
(line 819) also triggers a sync.

---

## 3. Proposed Architecture: Streaming Sentence-Level Pipeline

### 3.1 Core Principle: Pipeline Parallelism

The fundamental change is to overlap LLM generation, TTS generation, and audio
delivery so they run concurrently on different data:

```
TIME ----->

LLM:   [---sentence 1---][---sentence 2---][---sentence 3---]
TTS:                      [---gen sent 1---][---gen sent 2---][---gen sent 3---]
SEND:                                       [--send sent 1--][--send sent 2--][--send sent 3--]
PLAY:                                       [==play sent 1==][==play sent 2==][==play sent 3==]
```

While the LLM generates sentence 2, TTS processes sentence 1. While TTS
generates sentence 2's audio, the client plays sentence 1's audio. This is
classic pipeline parallelism.

### 3.2 Proposed Pipeline Flow

```
USER SPEAKS          STT           SENTENCE         TTS              CLIENT
    |                 |            DETECTOR          |                 |
    |--speech-end---->|              |               |                 |
    |                 |--transcribe->|               |                 |
    |                 |              |               |                 |
    |                 |    LLM tokens streaming...   |                 |
    |                 |    "Hello"   |               |                 |
    |                 |    " there"  |               |                 |
    |                 |    "."  <----+-- BOUNDARY    |                 |
    |                 |              |               |                 |
    |                 |              |--"Hello       |                 |
    |                 |              |  there."----->|                 |
    |                 |              |               |--generate------>|
    |                 |    " I'm"    |               |                 |
    |                 |    " doing"  |               |                 |
    |                 |    " well"   |               |                 |
    |                 |    "!"  <----+-- BOUNDARY    |                 |
    |                 |              |               |<--audio 1-------|
    |                 |              |               |---send binary-->|
    |                 |              |               |                 |--play-->
    |                 |              |--"I'm doing   |                 |
    |                 |              |  well!"------>|                 |
    |                 |              |               |--generate------>|
    |                 |              |               |                 |
    |                 |              |               |<--audio 2-------|
    |                 |              |               |---send binary-->|
    |                 |              |               |                 |--play-->
```

### 3.3 Sentence Boundary Detection in LLM Token Stream

The sentence boundary detector accumulates tokens from the Ollama stream and
emits complete sentences as soon as a boundary is detected.

**Detection rules (ordered by priority):**

1. Terminal punctuation followed by space or end-of-stream: `.` `!` `?`
   followed by ` ` or stream-end
2. Closing quote after terminal: `."` `!'` `?"` followed by space or end
3. Minimum sentence length: do not emit sentences shorter than 20 characters
   (prevents splitting abbreviations like "Dr. Smith" or "U.S.A.")
4. On stream end: flush any remaining buffer as final sentence

**Edge cases to handle:**
- Abbreviations: "Dr.", "Mr.", "U.S.", "e.g." -- maintain a set of common
  abbreviations and do not split on their periods
- Decimal numbers: "3.14" -- do not split on period between digits
- Ellipses: "..." -- treat as single boundary, do not emit three times
- Lists: "1. First item" -- period after single digit is likely a list marker

**Implementation sketch:**

```python
class SentenceBoundaryDetector:
    """Detects sentence boundaries in a streaming token sequence."""

    ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "prof", "sr", "jr",
                     "vs", "etc", "inc", "ltd", "dept", "est",
                     "approx", "vol", "no", "fig", "eq"}
    MIN_SENTENCE_LENGTH = 20

    def __init__(self):
        self.buffer = ""

    def feed(self, token: str) -> list[str]:
        """Feed a token, return list of complete sentences (0 or more)."""
        self.buffer += token
        sentences = []

        while True:
            boundary = self._find_boundary()
            if boundary is None:
                break
            sentence = self.buffer[:boundary].strip()
            self.buffer = self.buffer[boundary:].lstrip()
            if len(sentence) >= self.MIN_SENTENCE_LENGTH:
                sentences.append(sentence)
            elif sentences:
                # Too short -- merge with previous
                sentences[-1] += " " + sentence
            else:
                # Too short and nothing to merge with -- keep in buffer
                self.buffer = sentence + " " + self.buffer
                break

        return sentences

    def flush(self) -> list[str]:
        """Flush remaining buffer as final sentence."""
        if self.buffer.strip():
            return [self.buffer.strip()]
        return []

    def _find_boundary(self) -> int | None:
        """Find first sentence boundary position in buffer."""
        # Implementation: scan for .!? followed by space,
        # excluding abbreviations and decimals
        ...
```

### 3.4 Component Changes Required

#### A. KnowledgeIntegration -- True Token Streaming

Replace the blocking `handle_message()` call with actual token-level streaming
from Ollama. The current code yields a single blob. The new code must:

1. Run RAG retrieval synchronously (it is fast after warm-up)
2. Construct the augmented prompt with RAG context
3. Call `ollama.chat(..., stream=True)` directly
4. Yield each token as it arrives

This requires either:
- Refactoring `ConversationHandler.handle_message()` to support streaming, or
- Bypassing ConversationHandler for the voice path and constructing the
  augmented Ollama call directly in KnowledgeIntegration

The second approach is simpler and avoids disturbing the tested conversation
handler. The voice path does not need tool-calling support (the LLM does not
call tools mid-stream in voice mode).

#### B. VoiceProcessor -- Sentence-Level Pipeline Orchestration

Replace the sequential LLM-then-TTS flow in `_process_conversation_turn()`
with a concurrent pipeline:

```python
def _process_conversation_turn(self, user_text):
    """Process one conversation turn with pipeline parallelism."""
    # ... setup, lock acquisition ...

    sentence_detector = SentenceBoundaryDetector()
    sentence_queue = queue.Queue()  # Sentences ready for TTS
    tts_done = threading.Event()

    # Start TTS consumer thread
    tts_thread = threading.Thread(
        target=self._tts_consumer,
        args=(sentence_queue, tts_done),
        daemon=True,
    )
    tts_thread.start()

    # LLM producer: stream tokens, detect sentences, feed TTS queue
    full_response = ""
    for token in self.models.generate_llm_response(user_text):
        full_response += token

        # Send token to client for real-time text display
        asyncio.run_coroutine_threadsafe(
            self.message_queue.put({"type": "llm_token", "token": token}),
            self.loop,
        )

        # Detect sentence boundaries
        sentences = sentence_detector.feed(token)
        for sentence in sentences:
            sentence_queue.put(sentence)

    # Flush remaining text
    for sentence in sentence_detector.flush():
        sentence_queue.put(sentence)

    # Signal no more sentences
    sentence_queue.put(None)

    # Wait for TTS to finish all sentences
    tts_thread.join()
```

#### C. TTS Consumer Thread

```python
def _tts_consumer(self, sentence_queue, tts_done):
    """Consume sentences from queue, generate TTS, send audio."""
    chunk_count = 0
    first_chunk_time = None
    tts_start = time.time()

    while True:
        sentence = sentence_queue.get()
        if sentence is None:
            break

        if self.interrupt_flag.is_set():
            break

        # Generate TTS for this sentence
        for audio_chunk in self.models.generate_tts_audio(sentence):
            if self.interrupt_flag.is_set():
                break

            chunk_count += 1
            if first_chunk_time is None:
                first_chunk_time = time.time() - tts_start

            # Send audio to client immediately
            # (use binary protocol per 2026-03-24 spec)
            audio_np = audio_chunk.cpu().numpy().astype(np.float32)
            asyncio.run_coroutine_threadsafe(
                self.message_queue.put({
                    "type": "audio_chunk",
                    "audio": audio_np.tolist(),  # Replace with binary
                    "sample_rate": 24000,
                    "chunk_num": chunk_count,
                }),
                self.loop,
            )

    tts_done.set()
```

#### D. Chatterbox Chunk Sizing

Currently, `_generate_tts_chatterbox()` groups sentences into chunks of up to
50 words. With sentence-level streaming, each sentence arrives individually.
The TTS consumer should process each sentence as its own generation call
(no re-chunking needed) since:

- Average sentence is 10-20 words (well within Chatterbox's quality range)
- Sentence-level generation avoids the 8-16 second mega-chunks seen in logs
- Shorter generations have better RTF (see analysis below)

---

## 4. Specific Changes -- Ranked by Impact

### Change 1: LLM-TTS Pipeline Parallelism (CRITICAL)

**What:** Stream LLM tokens through sentence boundary detector, feed sentences
to TTS as they complete, overlap LLM and TTS execution.

**Files changed:**
- `backend/voice_processor.py` -- New `_process_conversation_turn()` with
  pipeline orchestration
- `backend/chat/knowledge_integration.py` -- True token-level streaming
- New file: `backend/sentence_detector.py` -- Sentence boundary detection

**Estimated latency improvement:**

Current time-to-first-audio (Turn 2, session 2):
```
LLM total:  3.67s (sequential)
TTS chunk1: 8.66s (sequential)
Current:    3.67 + 8.66 = 12.33s to first audio
```

With pipeline parallelism, TTS starts after the first sentence completes
(~1.0--1.5s into LLM generation, based on typical first-sentence timing at
~40 tokens/s):

```
LLM first sentence:  ~1.2s
TTS first sentence:   ~2.5s (short sentence, ~60 chars)
Proposed:             ~3.7s to first audio
```

**Savings: ~8.6 seconds on first audio delivery (70% reduction)**

This is the single highest-impact change.

### Change 2: Sentence-Level TTS Chunking (HIGH)

**What:** Generate TTS per-sentence instead of per-50-word-chunk.

**Files changed:**
- `backend/voice_models/model_manager.py` -- Simplify `_generate_tts_chatterbox()`
  to process single sentences directly

**Estimated latency improvement:**

TTS RTF analysis from logs (Chatterbox Turbo, FRIDAY voice):

| Chars | Audio Duration | Gen Time | RTF | Steps |
|-------|---------------|----------|-----|-------|
| 61 | 3.40s | 2.84s | 0.84x | 86 |
| 73 | 3.80s | 2.90s | 0.76x | 96 |
| 154 | 10.44s | 7.90s | 0.76x | 262 |
| 184 | 12.24s | 9.25s | 0.76x | 307 |
| 188 | 12.52s | 9.31s | 0.74x | 314 |
| 222 | 12.04s | 8.66s | 0.72x | 302 |
| 236 | 16.64s | 16.04s | 0.96x | 417 |

**RTF analysis:** Chatterbox maintains ~0.74--0.84x RTF for inputs under
200 chars. At 236 chars, RTF degrades to 0.96x. The relationship between
generation steps and chars is roughly linear (~1.3 steps/char), but generation
time per step appears to increase slightly for longer sequences (likely due
to attention cost scaling quadratically with sequence length in the T3
transformer).

**Optimal chunk size:** Under 150 chars per sentence. Most natural sentences
fall in the 40-120 char range, which is the sweet spot for Chatterbox.

By processing individual sentences (~60-100 chars each) instead of 50-word
chunks (~200-250 chars), each TTS call completes in ~2.5-4.0s instead of
8-16s. Combined with pipeline parallelism, the user hears audio from the
first sentence while later sentences are still generating.

**Savings: Perceived latency reduction of 5-10 seconds on multi-sentence responses**

### Change 3: Binary WebSocket Audio Transport (HIGH)

**What:** Implement the binary frame protocol from the 2026-03-24 spec.
Replace `audio_np.tolist()` JSON serialization with PCM16 binary frames.

**Files changed:**
- `backend/voice_processor.py` -- Send binary frames instead of JSON
- `backend/server.py` -- `broadcast_messages()` handles binary frame routing
- Flutter client -- Binary frame parsing and PCM16 playback

**Estimated improvement:**

A 3.40s audio clip at 24kHz = 81,600 samples.
- As float32 JSON array: ~1.2 MB (each number ~15 bytes with comma)
- As PCM16 binary: ~163 KB + 16 byte header

**Savings: ~7x bandwidth reduction, ~50-100ms serialization time per chunk**

This also enables gapless playback on the client because binary frames arrive
faster and can be fed directly into an audio ring buffer.

### Change 4: Eager Embedding Model Loading (MEDIUM)

**What:** Load the sentence-transformers embedding model during server startup
instead of lazily on first RAG query.

**Files changed:**
- `backend/knowledge/embeddings/embedding_generator.py` -- Load model in
  `__init__()` or add explicit `load()` method
- `backend/voice_models/model_manager.py` -- Call embedding warmup during
  `load_all_models()`

**Estimated improvement:**

First-turn RAG retrieval drops from 606ms to ~16ms (matching subsequent turns).

**Savings: 590ms on first conversation turn only**

### Change 5: True Streaming in KnowledgeIntegration (MEDIUM)

**What:** Replace the blocking `handle_message()` call with direct Ollama
streaming that yields tokens.

**Files changed:**
- `backend/chat/knowledge_integration.py` -- Rewrite
  `generate_response_streaming()` to actually stream

**Current behavior:** Despite the method name, it blocks until the full
response is ready, then yields it as one string. This is confirmed by the
comment on line 54: "Current implementation returns complete response, not
streaming."

**Impact:** This change is a prerequisite for Change 1. Without true token
streaming from the knowledge layer, pipeline parallelism cannot work because
the LLM "stream" yields zero tokens for 3-5 seconds, then dumps everything
at once.

**Savings: Prerequisite for Change 1 savings (~8.6 seconds)**

### Change 6: Remove Unnecessary CUDA Synchronization (LOW)

**What:** Remove `torch.cuda.synchronize()` calls before and after TTS
generation in the Chatterbox worker path.

**Files changed:**
- `backend/voice_models/model_manager.py` lines 155-156, 166-167

**Rationale:** CUDA synchronization forces the CPU to wait for all pending GPU
operations to complete. This is useful for timing measurements but adds latency
in production. The `tts_result_queue.put()` call that follows will naturally
block until the tensor data is available. The pre-generation sync is entirely
unnecessary -- the generate() call itself will synchronize as needed.

**Savings: ~20-50ms per TTS generation call**

### Change 7: Pre-warm Next TTS Generation (LOW)

**What:** While audio is being sent to the client, begin the next TTS
generation immediately rather than waiting for the send to complete.

**Current flow:**
```
generate chunk 1 -> put in result queue -> voice_processor reads ->
converts to numpy -> puts in message_queue -> broadcast to WebSocket ->
THEN submit chunk 2 to TTS worker
```

**With pipeline parallelism (Change 1), this is largely solved.** The TTS
consumer thread processes sentences from a queue independently, so there is no
wait between TTS generations. This change is subsumed by Change 1.

---

## 5. Hardware Constraints vs Architecture Constraints

### Hardware-Limited (Cannot Be Optimized Away)

| Constraint | Limit | Impact |
|------------|-------|--------|
| Chatterbox inference speed | ~0.74-0.84x RTF | Each second of audio takes 0.74-0.84s to generate |
| Ollama/Qwen first-token latency | ~1.0-1.5s | Minimum wait before any LLM text appears |
| Ollama/Qwen generation speed | ~35-50 tokens/s | Based on 3.2s for 61-char response (~100 tokens) |
| VRAM capacity | 12 GB total | Cannot run larger/faster models |
| GPU compute (4070 SUPER) | 35.48 TFLOPS FP16 | Fixed throughput ceiling |

### Architecture-Limited (Can Be Optimized)

| Constraint | Current Impact | Optimization |
|------------|---------------|-------------|
| LLM blocks TTS | +3-5s dead time | Pipeline parallelism (Change 1) |
| Fake streaming in KnowledgeIntegration | Full LLM wait | True streaming (Change 5) |
| Large TTS chunks | 8-16s per chunk | Sentence-level chunks (Change 2) |
| JSON audio serialization | 7x bandwidth waste | Binary protocol (Change 3) |
| Lazy embedding load | +590ms first turn | Eager loading (Change 4) |
| Unnecessary CUDA syncs | +20-50ms per gen | Remove syncs (Change 6) |

---

## 6. VRAM Budget Impact

### Current Usage

```
Chatterbox Turbo:     ~3.9 GB (persistent)
Whisper base:         ~0.3 GB (persistent)
Embeddings:           ~0.1 GB (lazy, evicted)
PyTorch overhead:     ~0.5 GB
                      ---------
Total in backend:     ~4.8 GB
Qwen 2.5 7B (Ollama): ~4.5 GB (separate container)
                      ---------
Grand total:          ~9.3 GB / 12.0 GB
```

### Impact of Proposed Changes

None of the proposed changes increase VRAM usage. The changes are purely
architectural (pipeline orchestration, serialization format, streaming).

If eager embedding model loading (Change 4) keeps the model resident instead
of evicting it, that adds ~100 MB persistently, bringing the backend container
to ~4.9 GB. This is well within the headroom.

**Headroom after changes: ~2.6 GB** -- sufficient for future model upgrades
or adding a second small model.

---

## 7. Risk Assessment

### Risk 1: Audio Gaps Between Sentences (MEDIUM)

**Cause:** If TTS generation for sentence N+1 takes longer than the playback
duration of sentence N's audio, the client runs out of audio and there is
silence.

**Analysis:** With Chatterbox at 0.74-0.84x RTF, generation is faster than
real-time. A 3-second sentence generates in ~2.5 seconds. As long as the LLM
delivers the next sentence before the current one finishes playing, there will
be no gap.

**Worst case:** A very short sentence (e.g., "Yes.") followed by a very long
sentence. The short sentence plays in ~0.5s, but TTS for the long sentence
might take 4s. This creates a 3.5s gap.

**Mitigation:**
- Merge very short sentences (<20 chars) with the next sentence before
  submitting to TTS
- Client-side: implement a small audio pre-buffer (buffer 1 sentence ahead
  before starting playback)
- Client-side: if buffer runs dry, show a "thinking" indicator rather than
  dead silence

### Risk 2: Sentence Boundary Detection Errors (LOW-MEDIUM)

**Cause:** Abbreviations, decimal numbers, URLs, or unusual punctuation
patterns cause premature sentence splits.

**Impact:** A sentence split mid-thought results in unnatural TTS output
(wrong prosody, abrupt ending).

**Mitigation:**
- Maintain abbreviation allowlist
- Require space or end-of-stream after terminal punctuation
- Minimum sentence length threshold (20 chars)
- Decimal number detection (period between digits)
- Test with diverse LLM outputs including technical content

### Risk 3: GPU Contention Between Whisper and Chatterbox (LOW)

**Cause:** If the user interrupts with speech while TTS is generating, Whisper
STT and Chatterbox could compete for GPU cycles.

**Current state:** This already happens in the current architecture (interrupt
flag exists). The interrupt flag stops TTS generation, so the contention
window is brief.

**With pipeline parallelism:** Same behavior. The interrupt flag is checked
in the TTS consumer loop. When set, TTS stops and Whisper can proceed.

**Mitigation:** No additional mitigation needed. The interrupt mechanism
already handles this correctly.

### Risk 4: Knowledge Extraction Competing with TTS on GPU (LOW)

**Cause:** The background knowledge extraction pipeline
(`ontology_extractor`) calls Ollama for entity extraction concurrently with
TTS generation. Ollama runs in a separate container with its own CUDA context,
so this does not directly contend with Chatterbox. However, the embedding
model (`all-MiniLM-L6-v2`) runs in the same container as Chatterbox.

**Observed in logs:** Embedding model loading occurs during TTS generation
(lines 231-238 in session 1 overlap with TTS generation). Since the embedding
model is small (~100MB) and the overlap is brief, the impact is minimal.

**Mitigation:** If contention becomes measurable, pin the embedding model
to CPU (`device='cpu'`) since it is small and fast enough on CPU for the
retrieval use case.

### Risk 5: Thread Safety in Pipeline Orchestration (MEDIUM)

**Cause:** The proposed architecture adds a sentence queue and a TTS consumer
thread that run concurrently with the LLM producer thread. The existing
`generation_lock`, `interrupt_flag`, and `is_speaking` state must remain
consistent.

**Mitigation:**
- The sentence queue (`queue.Queue`) is thread-safe by design
- The interrupt flag (`threading.Event`) is already thread-safe
- The generation lock remains the outer guard -- only one conversation turn
  runs at a time
- The TTS consumer thread is scoped to a single conversation turn and joins
  before the turn completes

### Risk 6: Ordering Guarantees in Async Message Queue (LOW)

**Cause:** `asyncio.run_coroutine_threadsafe()` submits messages to the async
queue from multiple threads (LLM tokens from one thread, audio chunks from
the TTS consumer thread). If both produce messages simultaneously, ordering
between text tokens and audio could interleave.

**Impact:** The client might receive an audio chunk before the corresponding
LLM text tokens finish streaming.

**Mitigation:** This is acceptable behavior. The client should handle audio
and text independently. Text tokens update the transcript display; audio
chunks feed the playback buffer. They do not need strict ordering relative
to each other.

---

## 8. Implementation Priority

### Phase 1: Foundation (Highest Impact, Do First)

**Estimated effort: 2-3 days**

1. **Change 5: True streaming in KnowledgeIntegration** -- This is the
   prerequisite. Without real token streaming, pipeline parallelism is
   impossible. Implement a voice-path streaming method that runs RAG, builds
   the augmented prompt, and calls `ollama.chat(stream=True)` directly.

2. **Change 1: LLM-TTS Pipeline Parallelism** -- Implement sentence boundary
   detector, sentence queue, TTS consumer thread, and the new
   `_process_conversation_turn()` orchestration.

3. **Change 2: Sentence-Level TTS Chunking** -- Simplify TTS generation to
   process individual sentences. Remove the 50-word chunking logic from
   `_generate_tts_chatterbox()`.

**Expected result after Phase 1:**
- Time-to-first-audio drops from 12-19s to ~3.5-4.5s
- Multi-sentence responses feel conversational (audio starts before LLM finishes)
- Total end-to-end latency drops by 50-70%

### Phase 2: Transport Optimization (Already Designed)

**Estimated effort: 2-3 days**

4. **Change 3: Binary WebSocket Audio Transport** -- Implement the binary
   frame protocol from 2026-03-24-voice-streaming-pipeline-design.md. This
   enables gapless playback and reduces bandwidth by 7x.

**Expected result after Phase 2:**
- Gapless audio playback between sentences
- Dramatically reduced network overhead
- Client can buffer ahead for smooth playback

### Phase 3: Polish (Diminishing Returns)

**Estimated effort: 0.5 days**

5. **Change 4: Eager Embedding Model Loading** -- Pre-load embedding model
   at startup to eliminate 590ms first-turn penalty.

6. **Change 6: Remove Unnecessary CUDA Syncs** -- Small but free improvement.

---

## 9. Target Metrics

### Current Measured Performance

| Metric | Short Response (61 chars) | Medium (298 chars) | Long (423 chars) |
|--------|--------------------------|--------------------|--------------------|
| Time to first audio | 6.10s | 12.33s | 19.18s |
| Total latency (last audio) | 6.10s | 15.41s | 28.73s |
| LLM time | 3.20s | 3.67s | 3.14s |
| TTS time | 2.90s | 11.74s | 25.59s |

### Target Performance After Phase 1

| Metric | Short (61 chars) | Medium (298 chars) | Long (423 chars) |
|--------|-----------------|--------------------|--------------------|
| Time to first audio | **~3.5s** | **~3.7s** | **~3.7s** |
| Total latency (last audio) | **~3.5s** | **~8.5s** | **~12.0s** |

**Calculation for medium response (298 chars, ~5 sentences):**

```
Current:
  LLM:           3.67s (all sequential)
  TTS chunk 1:   8.66s (222 chars)
  TTS chunk 2:   2.90s (73 chars)
  Total:         3.67 + 8.66 + 2.90 = 15.23s

Proposed (pipeline):
  LLM sentence 1 ready:    ~1.2s  (first ~60 chars at 40 tok/s)
  TTS sentence 1:          ~2.5s  (60 chars, RTF ~0.8x)
  First audio at:          ~3.7s  (1.2 + 2.5)

  While TTS does sentence 1, LLM generates sentences 2-5.
  TTS pipeline processes remaining 4 sentences:
    ~4 x 2.5s = 10.0s TTS, but overlapped with playback

  LLM total:               ~3.5s
  TTS sentence 1 done at:  ~3.7s (user hears audio)
  TTS sentence 5 done at:  ~3.7 + (4 x 2.5) = ~13.7s (worst case serial)

  But TTS runs concurrent with playback of earlier sentences.
  5 sentences x ~3.5s audio each = 17.5s playback.
  Last TTS generation completes before its playback slot:
    Sentence 5 TTS starts at: ~3.7 + 3 x 2.5 = ~11.2s
    Sentence 5 TTS done at:   ~13.7s
    Sentence 5 playback starts at: ~3.7 + 4 x 3.5 = ~17.7s (audio queue)
    No gap -- TTS is ahead of playback.

  Last audio SENT: ~13.7s
  Last audio PLAYED: ~17.7 + 3.5 = ~21.2s (but user started hearing at 3.7s)

  Perceived latency: ~3.7s (vs 12.33s currently)
```

**Calculation for long response (423 chars, ~7 sentences):**

```
Current:
  LLM:           3.14s
  TTS chunk 1:   16.04s (236 chars)
  TTS chunk 2:   9.25s  (184 chars)
  Total:         28.73s (user waits 19.18s for first audio)

Proposed:
  LLM sentence 1 ready:    ~1.2s
  TTS sentence 1:          ~2.5s
  First audio at:           ~3.7s

  Remaining 6 sentences generate concurrently with playback.
  Total TTS time: ~7 x 2.5s = ~17.5s
  But pipelined with LLM (overlap ~2s) and with playback.

  Last TTS done at: ~3.7 + 6 x 2.5 = ~18.7s
  Last audio played at: ~3.7 + 6 x 3.0 = ~21.7s

  User perceived: 3.7s to first audio (vs 19.18s currently)
```

### Target Performance After Phase 2 (Binary Transport)

Same first-audio times. Improvement is in playback quality:
- Zero gaps between sentences (ring buffer with pre-buffering)
- ~100ms less per chunk for serialization
- Smoother overall experience

### Theoretical Minimum Latency (Hardware-Limited)

```
Theoretical minimum with this hardware:
  LLM first sentence:   ~1.0s (Ollama/Qwen first-token + 20 tokens)
  TTS for one sentence:  ~2.0s (short sentence, 50 chars, 0.8x RTF)
  Network/serialize:     ~0.05s (binary, local network)
  Client buffer delay:   ~0.1s (pre-buffer before playback)
  -----------------------------------------------
  Theoretical minimum:   ~3.15s time-to-first-audio
```

This is a hard floor set by the Qwen 7B first-token latency (~1.0s) and
Chatterbox Turbo inference speed (~0.74x RTF). To go below 3 seconds would
require a faster LLM (smaller model or speculative decoding) or a faster
TTS model.

### Summary of Expected Improvements

| Metric | Current | Phase 1 Target | Improvement |
|--------|---------|---------------|-------------|
| Time-to-first-audio (short) | 6.1s | 3.5s | 43% faster |
| Time-to-first-audio (medium) | 12.3s | 3.7s | 70% faster |
| Time-to-first-audio (long) | 19.2s | 3.7s | 81% faster |
| Total latency (medium) | 15.4s | 13.7s (sent) | 11% faster |
| Total latency (long) | 28.7s | 18.7s (sent) | 35% faster |
| Audio bandwidth | ~3.8 MB/10s | ~163 KB/10s (Phase 2) | 96% reduction |

The dominant win is in perceived latency (time-to-first-audio), which becomes
nearly constant regardless of response length. Total pipeline time also
decreases because sentence-level TTS chunks are more efficient than large
chunks (better RTF at shorter lengths).

---

## Appendix A: LLM Token Rate Estimation

From the logs, LLM generation times and response lengths:

| Response | Chars | Est. Tokens | Time | Tokens/sec |
|----------|-------|-------------|------|------------|
| Turn 1 S1 | 154 | ~40 | 4.90s | ~8.2 |
| Turn 2 S1 | 188 | ~50 | 3.08s | ~16.2 |
| Turn 3 S1 | 423 | ~110 | 3.14s | ~35.0 |
| Turn 1 S2 | 61 | ~16 | 3.20s | ~5.0 |
| Turn 2 S2 | 298 | ~78 | 3.67s | ~21.3 |

The low tokens/sec on short responses is explained by high first-token latency
dominating small outputs. For Turn 1 S2 (61 chars, 3.20s), the actual
generation was probably ~1.5s of first-token latency + ~1.7s generation.
Turn 3 S1 (423 chars, 3.14s) shows the model running at ~35 tok/s once warm,
which is consistent with Qwen 2.5 7B Q4 on an RTX 4070 SUPER.

**Effective first-token latency:** ~1.0-1.5s (estimated from the pattern of
diminishing returns on short inputs).

**Sustained generation speed:** ~35-50 tokens/sec.

**Note:** The knowledge-augmented path adds RAG retrieval overhead before the
first Ollama call. With cached embeddings this is ~16ms (negligible). The
blocking `handle_message()` call also adds async scheduling overhead.

## Appendix B: Chatterbox Generation Step Analysis

Chatterbox Turbo reports "EOS token detected at step N" for each generation.
The step count correlates with output audio duration:

| Steps | Audio Duration | Duration/Step |
|-------|---------------|---------------|
| 46 | ~1.8s (warmup) | ~39ms |
| 49 | ~1.9s (warmup) | ~39ms |
| 63 | ~2.5s (warmup) | ~40ms |
| 86 | 3.40s | ~40ms |
| 96 | 3.80s | ~40ms |
| 262 | 10.44s | ~40ms |
| 302 | 12.04s | ~40ms |
| 307 | 12.24s | ~40ms |
| 314 | 12.52s | ~40ms |
| 417 | 16.64s | ~40ms |

**Consistent 40ms per generation step.** This is the codec frame rate of
Chatterbox Turbo -- each autoregressive step produces ~40ms of audio (960
samples at 24kHz).

**Time per step (wall clock):**

| Chars | Steps | Wall Time | Time/Step |
|-------|-------|-----------|-----------|
| 61 | 86 | 2.14s | 24.9ms |
| 73 | 96 | 2.25s | 23.4ms |
| 222 | 302 | 7.85s | 26.0ms |
| 236 | 417 | 15.12s | 36.3ms |

Wall-clock time per step increases slightly for longer sequences (24.9ms at
86 steps vs 36.3ms at 417 steps), consistent with O(n^2) attention cost.
This reinforces the case for sentence-level chunking: shorter generations
maintain lower per-step latency.

## Appendix C: Overhead of torch.cuda.synchronize() + torch.cuda.empty_cache()

Between TTS chunks, the current code calls:
```python
torch.cuda.synchronize()   # Before generation
# ... generate ...
torch.cuda.synchronize()   # After generation
torch.cuda.empty_cache()   # Between chunks
```

`empty_cache()` releases all unused cached CUDA memory back to the driver.
This triggers an implicit synchronization and a heap walk. On a 12GB card with
~5GB in use, this typically takes 5-20ms. The explicit `synchronize()` calls
add another 1-5ms each.

Total overhead per TTS generation: ~10-30ms. Across 5-7 sentences in a
response, this adds ~50-200ms. Low priority but free to remove.
