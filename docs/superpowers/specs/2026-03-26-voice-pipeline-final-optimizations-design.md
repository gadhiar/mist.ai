# Voice Pipeline Final Software Optimizations -- Design Spec

**Date:** 2026-03-26
**Status:** Approved
**Project:** MIST.AI -- Voice Pipeline
**Prerequisite:** 2026-03-26-voice-pipeline-optimization-design.md (Phase 1+3 implemented)
**Related:** 2026-03-24-voice-streaming-pipeline-design.md (binary transport protocol spec)

---

## 1. Context

Phase 1 (pipeline parallelism, true token streaming, sentence-level TTS) and
Phase 3 (eager embedding loading, CUDA sync removal) from the voice pipeline
optimization spec are complete. Time-to-first-audio improved from 12-19s to
~4-5s. Theoretical hardware minimum is ~3.15s.

This spec covers the remaining software optimizations. After completion, only
hardware changes (second GPU, faster GPU, model swap) can further improve the
pipeline.

### Current Performance

| Metric | Value |
|--------|-------|
| Time-to-first-audio (short response) | ~3.5-4.5s |
| Time-to-first-audio (medium response) | ~4-5s |
| LLM first-token latency | ~1.0-1.5s |
| LLM generation speed | ~35-50 tok/s |
| Chatterbox RTF | ~0.74-0.84x |
| Audio transport | JSON float32 (~3.8MB per 10s) |
| Client pre-buffer | 6s duration-based (backend-side) |
| Theoretical minimum TTFA | ~3.15s |

### Existing Assets

- `feat/voice-streaming-pipeline` branch: 7 commits implementing binary
  WebSocket transport (97 tests). Has 2 merge conflicts with main
  (`voice_processor.py`, `audio_playback_service.dart`).
- `2026-03-24-voice-streaming-pipeline-design.md`: Full protocol spec
  (binary frame format, playback state machine, jitter buffer, testing).
- MIS-76: Linear ticket tracking binary transport E2E validation.

---

## 2. Scope

| # | Item | Type | Expected Impact |
|---|------|------|-----------------|
| 1 | Binary WebSocket audio transport | Implementation | 7x bandwidth reduction, gapless playback, ~50-100ms serialization savings |
| 2 | Adaptive pre-buffering | Implementation | Remove 6s backend pre-buffer latency penalty |
| 3 | Personality profiles + first-utterance priming | Implementation | ~0.5-1.0s TTFA reduction, personality system established |
| 4 | Streaming TTS model evaluation | Research | Informs future TTS direction |

**Deferred:** Speculative decoding (Ollama draft model) -- deferred to
planned LLM model research session.

---

## 3. Binary WebSocket Audio Transport

### State

Already implemented on `feat/voice-streaming-pipeline`. Full protocol spec
exists at `2026-03-24-voice-streaming-pipeline-design.md`. MIS-76 tracks it.

### Work Required

**Merge conflict resolution:**

The branch predates 24 commits on main including pipeline parallelism. Two
conflicts:

1. `backend/voice_processor.py` -- The branch's binary send logic targets
   the old sequential `_process_conversation_turn()`. Must be rewired into
   the new `_tts_consumer()` pipeline architecture. Specifically:
   - Replace `audio_np.tolist()` JSON serialization in `_tts_consumer()`
     with binary frame construction from the branch's `audio_protocol.py`
   - Wire RMS normalization and PCM16 conversion into the TTS consumer loop
   - Wire interrupt fade-out into the interrupt flag check

2. `mist_desktop/lib/services/audio_playback_service.dart` -- The branch
   rewrites this file for PCM streaming via `flutter_soloud`. Main added
   6s duration-based pre-buffering. Resolution: take the branch's rewrite
   (PCM streaming model) and discard the 6s pre-buffer (see Section 4).

**E2E validation (per MIS-76 checklist):**

- Gapless audio playback across multi-turn conversation
- Volume consistency (RMS normalization at -20 dBFS)
- Clean fade-out on user interrupt (no pop/click)
- Jitter buffer handles variable chunk timing
- Binary/text frame discrimination works correctly
- Fallback: JSON audio path removed after validation

### Protocol Summary (from 2026-03-24 spec)

Binary frame format: 16-byte header + PCM16 payload.

| Field | Size | Value |
|-------|------|-------|
| magic | uint32 | 0x4D495354 ("MIST") |
| message_type | uint16 | 0x01 audio_chunk, 0x02 audio_complete, 0x03 interrupt_fade |
| session_id | uint16 | 0 (multi-tenant affordance) |
| chunk_seq | uint32 | Monotonically increasing |
| sample_rate | uint32 | 24000 |
| payload | var | Raw PCM16 LE int16 samples |

A 1-second audio chunk: 16 header + 48,000 payload = ~48KB (vs ~500KB JSON).

### Files Affected

**Backend (from branch, adapted for pipeline parallelism):**
- `backend/audio_protocol.py` -- Binary frame builder, RMS normalization, fade-out
- `backend/voice_processor.py` -- Binary frames in `_tts_consumer()`, remove `audio_np.tolist()`
- `backend/server.py` -- Mixed text/binary WebSocket send in broadcast loop

**Frontend (from branch):**
- `mist_desktop/lib/services/audio_playback_service.dart` -- Full rewrite for PCM streaming
- `mist_desktop/lib/services/websocket_service.dart` -- Binary/text frame discrimination
- `mist_desktop/lib/providers/voice_provider.dart` -- Subscribe to audio frame stream
- `mist_desktop/lib/models/binary_audio_frame.dart` -- New frame parser
- `mist_desktop/pubspec.yaml` -- Remove `audioplayers`, add `flutter_soloud`

### Testing

Backend unit tests from the branch (binary frame construction, RMS normalization,
fade-out generation) carry over. Integration test: verify binary frames parse
correctly end-to-end. Manual E2E test per MIS-76 checklist.

---

## 4. Adaptive Pre-Buffering

### Problem

Two competing buffering mechanisms exist after merge:

1. **Main branch (client-side):** 6s duration-based pre-buffer in
   `audio_playback_service.dart`. Accumulates 6s of audio before starting
   playback. Adds directly to perceived TTFA.

2. **Voice-streaming branch (client-side):** 150ms jitter buffer in the
   rewritten `audio_playback_service.dart`. Accumulates ~7,200 bytes of
   PCM16 before starting playback. Absorbs network variance and TTS
   generation jitter.

### Design

Remove the 6s backend pre-buffer entirely. It was a workaround for the old
WAV-queue model. With binary PCM streaming, the backend sends audio frames
immediately as TTS generates them.

The client-side jitter buffer becomes the sole buffering mechanism with
adaptive behavior:

**First chunk:** Start playback immediately. The first chunk from pipeline
parallelism is a full sentence (~2-4s of audio), far exceeding any jitter
buffer threshold. No reason to wait.

**Subsequent chunks:** The 150ms jitter buffer from the voice-streaming
branch handles inter-chunk timing. If buffer drops below 100ms between
sentences, insert silence (zero samples) rather than stutter or loop.

**Underrun tracking:** Count buffer underruns per response. If underruns
exceed 3, log average chunk arrival interval vs. playback consumption rate.
This data informs future tuning without guesswork.

### Changes

- `backend/voice_processor.py`: Remove 6s pre-buffer accumulation logic
- `mist_desktop/lib/services/audio_playback_service.dart`: First-chunk
  immediate playback (no minimum threshold for first chunk). Subsequent
  chunks use 150ms jitter buffer. Underrun counter with diagnostic logging.

### Expected Impact

Eliminates the 6s perceived latency penalty. Combined with binary transport,
first audio reaches the user's speakers as soon as TTS finishes generating
the first sentence.

---

## 5. Personality Profiles + First-Utterance Priming

### Problem

The voice system prompt in `generate_tokens_streaming()` doesn't instruct the
LLM on response structure or personality. The first sentence is typically 60+
chars, meaning TTS needs ~2.5s to generate it. A shorter first utterance
(10-20 chars) would generate in ~0.8-1.0s.

Additionally, the 40-char sentence coalescing minimum in the TTS consumer
swallows short openers and merges them with the next sentence, defeating
the purpose.

### Design

#### Personality Config Structure

Each voice profile gets a companion personality config:

```
voice_profiles/
  friday/
    profile.yaml          # Existing: TTS params (reference audio, exaggeration, etc.)
    personality.yaml      # New: speaking style, openers, mannerisms
  jarvis/
    profile.yaml
    personality.yaml
  cortana/
    profile.yaml
    personality.yaml
```

Personality config format:

```yaml
name: FRIDAY
address_form: "Boss"
characteristic_openers:
  - "Sure thing, Boss."
  - "Got it."
  - "Right away."
  - "On it, Boss."
  - "Here's what I found."
  - "One moment."
speaking_style: |
  Concise, professional, slightly warm. Military-assistant cadence.
  Favor short declarative sentences. No filler or hedging.
  Address the user as "Boss" occasionally but not every response.
  Lead with action or acknowledgment, then substance.
mannerisms:
  - Uses "Boss" as form of address (2-3x per conversation, not every turn)
  - Dry humor when appropriate
  - Never says "I think" -- states things directly
  - Avoids unnecessary qualifiers
```

#### System Prompt Templating

`generate_tokens_streaming()` loads the personality config for the active
voice profile and templates it into the voice system prompt:

```
You are {name}, a personal AI assistant.

Speaking style: {speaking_style}

When responding, begin with a brief acknowledgment or opening phrase before
your main answer. Examples of characteristic openers you use:
{characteristic_openers as bulleted list}

{mannerisms as additional instructions}
```

The personality config is loaded once at `KnowledgeIntegration.__init__()` and
cached. Profile switches reload the config.

#### Coalescing Bypass

The 40-char sentence coalescing minimum in the TTS consumer must skip the
first emitted sentence in each turn. Implementation: add a `first_sentence`
flag to the TTS consumer loop, set True at turn start, cleared after the
first sentence is sent to TTS. When `first_sentence` is True, send to TTS
regardless of length.

### Changes

- New: `voice_profiles/friday/personality.yaml`
- New: `voice_profiles/jarvis/personality.yaml` (stub)
- New: `voice_profiles/cortana/personality.yaml` (stub)
- `backend/chat/knowledge_integration.py`: Load personality config, template
  into voice system prompt
- `backend/voice_processor.py`: First-sentence coalescing bypass in
  `_tts_consumer()`

### Expected Impact

TTFA drops ~0.5-1.0s. First TTS input shrinks from ~60 chars to ~10-20 chars,
cutting first-chunk TTS generation from ~2.5s to ~0.8-1.0s. Personality system
established for all future voice profile work.

### Separate Tickets (not implemented in this spec)

- Flutter "Profiles" screen: Replace Voice Profiles stub with unified profile
  management UI combining TTS voice settings + personality config
- FRIDAY personality review: Dedicated pass to tune openers, mannerisms, and
  speaking style for quality and naturalness

---

## 6. Streaming TTS Model Evaluation (Research)

### Objective

Determine whether a streaming TTS model can replace Chatterbox Turbo for
lower time-to-first-audio. Research only -- no implementation.

### Direction A: Streaming TTS Model Alternatives

Evaluate models that natively emit audio frames during generation rather than
after full sequence completion.

**Evaluation criteria:**

| Criterion | Requirement |
|-----------|-------------|
| Output mode | Streaming/chunked (frames emitted mid-generation) |
| Voice cloning | Zero-shot or few-shot from reference audio |
| RTF | Competitive with Chatterbox (~0.74x or better) |
| VRAM | Fits within ~4GB (current Chatterbox allocation) |
| License | MIT, Apache 2.0, or similar |
| Quality | Comparable to Chatterbox Turbo |

**Candidates (starting from MIS-75's 18-model evaluation, filtering for
streaming capability):**

- Kokoro (82M params) -- fast, but verify streaming support
- F5-TTS (flow matching architecture) -- inherently supports chunked generation?
- MaskGCT -- masked generative codec transformer
- XTTS v2 (Coqui) -- known streaming support
- Any models released since MIS-75 evaluation (2026-03-25)

### Direction B: Chatterbox Sub-Chunk Streaming

Investigate whether Chatterbox's internal architecture allows emitting partial
audio before EOS:

- Chatterbox generates codec tokens autoregressively at ~40ms per step
- Can partial codec token sequences be decoded into playable audio mid-generation?
- What is the minimum number of codec tokens needed for coherent audio output?
- Quality impact of partial decoding vs full-sequence decoding
- Does the Chatterbox codebase expose hooks for mid-generation audio extraction?

### Deliverable

Research note in the vault (`04-Research/`) documenting:
- Each candidate model evaluated with scores against criteria
- Chatterbox sub-chunk streaming feasibility assessment
- Recommendation: swap to streaming model, modify Chatterbox, or stay with
  current architecture
- If swap recommended: migration path and voice profile compatibility

This research feeds into the planned LLM model research session.

---

## 7. Linear Ticket Plan

**Pre-implementation gate:** All tickets must be created before any code is
written.

### New Tickets

| Title | Type | Priority | Project |
|-------|------|----------|---------|
| Adaptive pre-buffering -- remove 6s buffer, client jitter only | Improvement | High | Voice |
| Personality profile system + first-utterance priming | Feature | High | Voice |
| Flutter Profiles screen (voice + personality management) | Feature | Medium | Frontend |
| FRIDAY personality review and finalization | Improvement | Medium | Voice |
| Research: Streaming TTS model evaluation | Research | Medium | Voice |

### Ticket Updates

| Ticket | Current Status | Action |
|--------|---------------|--------|
| MIS-76 (Voice Streaming E2E) | Backlog | Update description: merge conflict resolution needed against pipeline parallelism on main. Covers binary transport implementation (no new ticket needed). |
| MIS-45 (FRIDAY Voice Research) | In Progress | Mark Done -- FRIDAY voice profile created and shipped |
| MIS-44 (Voice Pipeline Review) | Backlog | Update -- pipeline parallelism addressed architecture rework |
| MIS-50 (Voice Pipeline Refactoring Phase 1) | Backlog | Review scope overlap with completed pipeline parallelism work |

---

## 8. Sequencing

```
Phase A (tightly coupled):
  1. Merge feat/voice-streaming-pipeline -> resolve conflicts -> E2E validate
  2. Remove 6s backend pre-buffer, validate adaptive client jitter buffer

Phase B (independent):
  3. Personality config system + FRIDAY personality + system prompt templating
  4. First-sentence coalescing bypass

Phase C (parallel research):
  5. Streaming TTS model evaluation (Direction A + B)
  6. Document findings in vault research note

Linear tickets created BEFORE Phase A begins.
```

Phases A and B can run in parallel (different files, no overlap). Phase C
runs independently of A and B.

---

## 9. Post-Completion State

After all items are implemented:

| Optimization | Status |
|-------------|--------|
| Pipeline parallelism (LLM-TTS overlap) | [DONE] Phase 1 |
| True token streaming from LLM | [DONE] Phase 1 |
| Sentence-level TTS chunking | [DONE] Phase 1 |
| Eager embedding model loading | [DONE] Phase 3 |
| CUDA sync removal | [DONE] Phase 3 |
| Binary WebSocket audio transport | [DONE] This spec |
| Adaptive pre-buffering (client jitter) | [DONE] This spec |
| First-utterance priming via personality | [DONE] This spec |
| Streaming TTS evaluation | [DONE] This spec (research) |
| Speculative decoding | Deferred to LLM research session |

**Remaining improvement vectors (all hardware-dependent):**
- Second GPU for dedicated voice processing (RTX 3060 12GB)
- Faster primary GPU for better LLM tok/s and TTS RTF
- LLM model swap (Qwen3, Llama 4, etc.) -- separate research session
- TTS model swap -- informed by this spec's streaming TTS research

**Expected TTFA after this spec:** ~2.5-3.5s (down from current ~4-5s).
Approaches the ~3.15s theoretical hardware minimum.
