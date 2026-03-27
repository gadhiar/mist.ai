# Voice Pipeline Final Software Optimizations -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exhaust all remaining software-side voice pipeline optimizations so only hardware changes can further improve performance.

**Architecture:** Merge the existing binary WebSocket transport branch (resolving conflicts against pipeline parallelism), replace the 6s client pre-buffer with adaptive jitter buffering, add a personality profile system that provides first-utterance priming for faster TTS, and research streaming TTS alternatives.

**Tech Stack:** Python 3.11, FastAPI WebSocket, Flutter/Dart, Riverpod, flutter_soloud, Ollama, Chatterbox TTS

**Spec:** `docs/superpowers/specs/2026-03-26-voice-pipeline-final-optimizations-design.md`

---

## File Map

### Backend -- New Files
| File | Responsibility |
|------|---------------|
| `backend/audio_protocol.py` | Binary frame builder, RMS normalization, fade-out generation, PCM16 conversion. Taken from `feat/voice-streaming-pipeline` branch unchanged. |
| `voice_profiles/friday/personality.yaml` | FRIDAY personality config (openers, speaking style, mannerisms) |
| `voice_profiles/jarvis/personality.yaml` | Jarvis personality config (stub) |
| `voice_profiles/cortana/personality.yaml` | Cortana personality config (stub) |

### Backend -- Modified Files
| File | Changes |
|------|---------|
| `backend/voice_processor.py` | Binary frames in `_tts_consumer()`, interrupt fade-out, first-sentence coalescing bypass, remove `audio_np.tolist()` |
| `backend/server.py` | Queue type `str \| bytes`, type-checked dispatch in `broadcast_messages()` |
| `backend/chat/knowledge_integration.py` | Load personality config, template voice system prompt |

### Frontend -- New Files
| File | Responsibility |
|------|---------------|
| `mist_desktop/lib/models/binary_audio_frame.dart` | Binary frame parser (magic validation, header extraction). Taken from branch. |

### Frontend -- Modified/Rewritten Files
| File | Changes |
|------|---------|
| `mist_desktop/lib/services/audio_playback_service.dart` | Full rewrite: flutter_soloud PCM streaming, state machine, jitter buffer, first-chunk immediate playback. Based on branch rewrite with adaptive buffering. |
| `mist_desktop/lib/services/websocket_service.dart` | Binary/text frame discrimination, new `audioFrameStream`. |
| `mist_desktop/lib/providers/voice_provider.dart` | Subscribe to `audioFrameStream`, route binary frames to playback service. |
| `mist_desktop/pubspec.yaml` | Remove `audioplayers`, add `flutter_soloud`. |

### Tests
| File | What It Tests |
|------|--------------|
| `tests/unit/test_audio_protocol.py` | Binary frame construction, RMS normalization, fade-out generation, PCM16 conversion |
| `tests/unit/test_personality_config.py` | Personality YAML loading, system prompt templating |
| `mist_desktop/test/unit/services/audio_playback_service_test.dart` | Playback state machine transitions (rewritten to match new service) |
| `mist_desktop/test/unit/models/binary_audio_frame_test.dart` | Frame parsing, magic validation, malformed frame handling |

---

## Task 0: Create Linear Tickets (Pre-Implementation Gate)

**No code is written until all tickets exist.**

- [ ] **Step 1: Update MIS-45 (FRIDAY Voice Research) to Done**

Mark as Done with comment: "FRIDAY voice profile created (46.2s reference WAV), all 3 profiles migrated to Chatterbox, default profile switched to friday."

- [ ] **Step 2: Update MIS-76 (Voice Streaming E2E) description**

Update description to note merge conflicts with pipeline parallelism on main. Add: "Branch predates 24 commits on main. Two merge conflicts: voice_processor.py (binary send targets old sequential architecture, must rewire to _tts_consumer pipeline) and audio_playback_service.dart (branch rewrite vs main's 6s pre-buffer). Resolution: manually integrate binary transport into pipeline architecture."

- [ ] **Step 3: Create ticket -- Adaptive pre-buffering**

Title: "Adaptive pre-buffering -- remove 6s client buffer, first-chunk immediate playback"
Project: Voice, Priority: High, Label: Improvement
Description: "Remove 6s duration-based pre-buffer from audio_playback_service.dart. Replace with adaptive strategy: start playback immediately on first chunk (full sentence, 2-4s of audio), use 150ms jitter buffer for subsequent chunks, insert silence on underrun. Track underrun count per response for diagnostics."

- [ ] **Step 4: Create ticket -- Personality profile system**

Title: "Personality profile system + first-utterance priming"
Project: Voice, Priority: High, Label: Feature
Description: "Create personality config YAML per voice profile (openers, speaking style, mannerisms, address form). Template voice system prompt from personality config in generate_tokens_streaming(). Bypass 40-char sentence coalescing for first sentence in each turn. Write initial FRIDAY personality config."

- [ ] **Step 5: Create ticket -- Flutter Profiles screen**

Title: "Flutter Profiles screen -- unified voice + personality management"
Project: Frontend, Priority: Medium, Label: Feature
Description: "Replace Voice Profiles stub screen with unified profile management UI. Display TTS voice settings (reference audio, exaggeration, temperature) and personality config (openers, speaking style, mannerisms) side by side. Allow editing personality YAML from Flutter UI."

- [ ] **Step 6: Create ticket -- FRIDAY personality review**

Title: "FRIDAY personality review and finalization"
Project: Voice, Priority: Medium, Label: Improvement
Description: "Dedicated pass to review and tune FRIDAY personality profile. Test openers for natural feel, adjust mannerisms frequency, verify speaking style produces conversational output. Iterate on personality.yaml until voice interactions feel authentic."

- [ ] **Step 7: Create ticket -- Streaming TTS research**

Title: "Research: Streaming TTS model evaluation (alternatives + Chatterbox sub-chunk)"
Project: Voice, Priority: Medium, Label: Research
Description: "Direction A: Evaluate streaming TTS models (Kokoro, F5-TTS, MaskGCT, XTTS v2) against criteria (streaming output, voice cloning, RTF, VRAM, license, quality). Direction B: Investigate Chatterbox sub-chunk streaming (partial codec token decoding mid-generation). Deliverable: vault research note with recommendation."

- [ ] **Step 8: Update MIS-44 and MIS-50**

MIS-44 (Voice Pipeline Review): Add comment noting pipeline parallelism addressed the architecture rework. Review remaining scope.
MIS-50 (Voice Pipeline Refactoring Phase 1): Add comment noting pipeline parallelism, sentence detector, true token streaming addressed critical refactoring items. Review remaining scope.

---

## Task 1: Backend Binary Protocol Module

**Files:**
- Create: `backend/audio_protocol.py`
- Create: `tests/unit/test_audio_protocol.py`

This file is taken from the `feat/voice-streaming-pipeline` branch unchanged. It has no dependencies on the pipeline parallelism changes.

- [ ] **Step 1: Extract audio_protocol.py from the branch**

```bash
git show feat/voice-streaming-pipeline:backend/audio_protocol.py > backend/audio_protocol.py
```

- [ ] **Step 2: Extract tests from the branch**

```bash
git show feat/voice-streaming-pipeline:tests/unit/test_audio_protocol.py > tests/unit/test_audio_protocol.py
```

- [ ] **Step 3: Run tests to verify**

```bash
docker compose run --rm --no-deps mist-backend pytest tests/unit/test_audio_protocol.py -v
```

Expected: All tests PASS (binary frame construction, RMS normalization, fade-out, PCM16 conversion).

- [ ] **Step 4: Commit**

```bash
git add backend/audio_protocol.py tests/unit/test_audio_protocol.py
git commit -m "feat(voice): add binary audio frame protocol module

Extracted from feat/voice-streaming-pipeline branch. Implements MIST
binary frame format (16-byte header + PCM16 payload), RMS volume
normalization (-20 dBFS), interrupt fade-out generation, and float32
to PCM16 conversion."
```

---

## Task 2: Backend Binary Transport Integration

**Files:**
- Modify: `backend/voice_processor.py` (lines 154-236: `_tts_consumer()`, lines 238-346: `_process_conversation_turn()`)
- Modify: `backend/server.py` (lines 64-81: `broadcast_messages()`)

- [ ] **Step 1: Update server.py broadcast_messages() for mixed text/binary**

In `backend/server.py`, change the message queue type annotation and add type-checked dispatch.

Replace the `broadcast_messages()` function (lines 64-81) with:

```python
async def broadcast_messages():
    """Background task to broadcast messages to all connected clients."""
    while True:
        message = await message_queue.get()

        async with active_connections_lock:
            stale = []
            for websocket in active_connections:
                try:
                    if isinstance(message, bytes):
                        await websocket.send_bytes(message)
                    else:
                        await websocket.send_text(message)
                except Exception as e:
                    logger.error(f"Error sending to client: {e}")
                    stale.append(websocket)
            for ws in stale:
                active_connections.discard(ws)

        message_queue.task_done()
```

Also update the queue type annotation (find `message_queue` declaration near the top):

```python
message_queue: asyncio.Queue[str | bytes] = asyncio.Queue()
```

- [ ] **Step 2: Update all non-audio message puts to use JSON strings**

In `backend/voice_processor.py`, every `message_queue.put()` for non-audio messages must serialize to JSON string instead of passing a dict. Add `import json` at the top of voice_processor.py.

Find every `self.message_queue.put({...})` that is NOT an audio_chunk and wrap with `json.dumps()`:

For `llm_token` messages (in `_process_conversation_turn`, ~line 278):
```python
asyncio.run_coroutine_threadsafe(
    self.message_queue.put(
        json.dumps({"type": "llm_token", "token": token})
    ),
    self.loop,
)
```

For `llm_response` messages (~line 300):
```python
asyncio.run_coroutine_threadsafe(
    self.message_queue.put(
        json.dumps({"type": "llm_response", "text": full_response})
    ),
    self.loop,
)
```

For `error` messages (~line 335):
```python
asyncio.run_coroutine_threadsafe(
    self.message_queue.put(
        json.dumps({"type": "error", "message": f"Generation error: {e}"})
    ),
    self.loop,
)
```

- [ ] **Step 3: Replace audio_np.tolist() with binary frames in _tts_consumer()**

In `backend/voice_processor.py`, add imports at the top. **Verify the exact exported names** by reading `backend/audio_protocol.py` (extracted in Task 1). The names below are based on the branch; adjust if the file uses different names:

```python
from audio_protocol import (
    MSG_AUDIO_CHUNK,
    MSG_AUDIO_COMPLETE,
    MSG_INTERRUPT_FADE,
    build_audio_frame,
    float32_to_pcm16,
    generate_fade_out,
    normalize_rms,
)
```

Replace the `_tts_consumer()` method (lines 154-236) with:

```python
def _tts_consumer(self, sentence_queue: queue.Queue, tts_start_time: float) -> None:
    """Consume sentences from queue, generate TTS, send audio to client.

    Runs in a dedicated thread. Processes sentences as they arrive
    from the LLM producer, generating and sending audio for each.
    Audio is sent as binary PCM16 frames via the MIST binary protocol.

    Args:
        sentence_queue: Queue of sentences to synthesize. None = stop signal.
        tts_start_time: Timestamp when TTS phase started (for logging).
    """
    chunk_count = 0
    chunk_seq = 0
    first_chunk_time = None
    first_sentence_time = None

    # Minimum chars for quality TTS output. Short inputs (<40 chars)
    # cause Chatterbox to glitch on the first utterance because the
    # model lacks enough text context to match the reference voice.
    # Bypassed for the first sentence in a turn (first-utterance priming).
    min_tts_chars = 40
    first_sentence = True

    while True:
        try:
            sentence = sentence_queue.get(timeout=1.0)
        except queue.Empty:
            if self.interrupt_flag.is_set():
                break
            continue
        if sentence is None:
            break
        if self.interrupt_flag.is_set():
            break

        # Coalesce short sentences with the next to avoid TTS glitches.
        # Skip coalescing for the first sentence (first-utterance priming).
        if not first_sentence:
            while len(sentence) < min_tts_chars:
                try:
                    next_item = sentence_queue.get(timeout=2.0)
                except queue.Empty:
                    break
                if next_item is None:
                    break
                sentence = sentence + " " + next_item
        first_sentence = False

        if first_sentence_time is None:
            first_sentence_time = time.time()

        log_timestamp(f"TTS: Generating sentence ({len(sentence)} chars)")

        for audio_chunk in self.models.generate_tts_audio(sentence):
            if self.interrupt_flag.is_set():
                # Generate fade-out frame for clean audio cutoff
                if isinstance(audio_chunk, torch.Tensor):
                    audio_np = audio_chunk.cpu().numpy().astype(np.float32)
                else:
                    audio_np = audio_chunk.astype(np.float32)
                fade_audio = generate_fade_out(audio_np, sample_rate=24000)
                chunk_seq += 1
                fade_frame = build_audio_frame(
                    MSG_INTERRUPT_FADE,
                    0,
                    chunk_seq,
                    24000,
                    float32_to_pcm16(normalize_rms(fade_audio)),
                )
                asyncio.run_coroutine_threadsafe(
                    self.message_queue.put(fade_frame), self.loop
                )
                break

            chunk_count += 1
            chunk_seq += 1
            if first_chunk_time is None:
                elapsed_from_sentence = time.time() - first_sentence_time
                elapsed_from_turn = time.time() - tts_start_time
                log_timestamp(
                    f"TTS: First audio chunk "
                    f"({elapsed_from_sentence:.2f}s from first sentence, "
                    f"{elapsed_from_turn:.2f}s from turn start)"
                )
                first_chunk_time = elapsed_from_turn

            if isinstance(audio_chunk, torch.Tensor):
                audio_np = audio_chunk.cpu().numpy().astype(np.float32)
            else:
                audio_np = audio_chunk.astype(np.float32)

            # Binary frame: RMS normalize -> PCM16 -> MIST frame
            pcm16_bytes = float32_to_pcm16(normalize_rms(audio_np))
            frame = build_audio_frame(
                MSG_AUDIO_CHUNK, 0, chunk_seq, 24000, pcm16_bytes
            )
            asyncio.run_coroutine_threadsafe(
                self.message_queue.put(frame), self.loop
            )

    # Send completion frame
    chunk_seq += 1
    complete_frame = build_audio_frame(
        MSG_AUDIO_COMPLETE, 0, chunk_seq, 24000, b""
    )
    asyncio.run_coroutine_threadsafe(
        self.message_queue.put(complete_frame), self.loop
    )

    tts_total = time.time() - tts_start_time
    log_timestamp(f"TTS consumer done ({tts_total:.2f}s, {chunk_count} chunks)")
```

- [ ] **Step 4: Update audio_complete in _process_conversation_turn()**

In `_process_conversation_turn()`, remove the JSON audio_complete message at the end (~line 325). The `_tts_consumer` now sends the binary `MSG_AUDIO_COMPLETE` frame itself. Replace:

```python
# Send completion signal
asyncio.run_coroutine_threadsafe(
    self.message_queue.put({"type": "audio_complete"}),
    self.loop,
)
```

With:

```python
# audio_complete signal is sent by _tts_consumer via binary frame.
# Only send for text-only mode (no TTS).
if not self.config.tts_enabled:
    asyncio.run_coroutine_threadsafe(
        self.message_queue.put(
            json.dumps({"type": "audio_complete"})
        ),
        self.loop,
    )
```

- [ ] **Step 5: Run backend tests**

```bash
docker compose run --rm --no-deps mist-backend pytest tests/unit/ -v --tb=short
```

Expected: All existing tests pass. New audio_protocol tests pass.

- [ ] **Step 6: Commit**

```bash
git add backend/voice_processor.py backend/server.py
git commit -m "feat(voice): integrate binary audio transport into pipeline architecture

Replace audio_np.tolist() JSON serialization with MIST binary frames
(PCM16 + 16-byte header). Add RMS normalization (-20 dBFS), interrupt
fade-out generation, and sequence numbering. Server broadcast_messages()
now dispatches binary frames via send_bytes() and text via send_text()."
```

---

## Task 3: Frontend Binary Frame Parser

**Files:**
- Create: `mist_desktop/lib/models/binary_audio_frame.dart`
- Create: `mist_desktop/test/unit/models/binary_audio_frame_test.dart`

- [ ] **Step 1: Extract binary_audio_frame.dart from branch**

```bash
git show feat/voice-streaming-pipeline:mist_desktop/lib/models/binary_audio_frame.dart > mist_desktop/lib/models/binary_audio_frame.dart
```

- [ ] **Step 2: Extract tests from branch**

```bash
git show feat/voice-streaming-pipeline:mist_desktop/test/unit/models/binary_audio_frame_test.dart > mist_desktop/test/unit/models/binary_audio_frame_test.dart
```

If the test file path differs on the branch, find it:
```bash
git ls-tree -r --name-only feat/voice-streaming-pipeline | grep binary_audio_frame
```

- [ ] **Step 3: Run tests**

```bash
cd mist_desktop && flutter test test/unit/models/binary_audio_frame_test.dart
```

Expected: All frame parsing tests pass (valid frame, malformed magic, short header, zero payload).

- [ ] **Step 4: Commit**

```bash
git add mist_desktop/lib/models/binary_audio_frame.dart mist_desktop/test/unit/models/binary_audio_frame_test.dart
git commit -m "feat(frontend): add BinaryAudioFrame parser for WebSocket binary protocol"
```

---

## Task 4: Frontend WebSocket Binary/Text Frame Discrimination

**Files:**
- Modify: `mist_desktop/lib/services/websocket_service.dart`

- [ ] **Step 1: Add audioFrameStream and binary frame routing**

In `websocket_service.dart`, add a new stream controller for binary audio frames and update the message handler to discriminate between text and binary frames.

Add import at top:
```dart
import 'dart:typed_data';
import '../models/binary_audio_frame.dart';
```

Add a new StreamController after the existing `_messageController`:
```dart
final _audioFrameController = StreamController<BinaryAudioFrame>.broadcast();
Stream<BinaryAudioFrame> get audioFrameStream => _audioFrameController.stream;
```

In the WebSocket listen handler, update the data handling to check type:
```dart
_channel!.stream.listen(
  (data) {
    if (data is List<int>) {
      // Binary frame -- parse as audio
      try {
        final frame = BinaryAudioFrame.parse(Uint8List.fromList(data));
        _audioFrameController.add(frame);
      } catch (e) {
        _logger.warning('Malformed binary frame: $e');
      }
    } else if (data is String) {
      // Text frame -- parse as JSON message
      try {
        final message = WebSocketMessage.fromJson(data);
        _messageController.add(message);
      } catch (e) {
        _logger.warning('Failed to parse message: $e');
      }
    }
  },
  // ... existing onError, onDone handlers ...
);
```

Add cleanup in dispose:
```dart
void dispose() {
  // ... existing cleanup ...
  _audioFrameController.close();
}
```

- [ ] **Step 2: Run existing WebSocket tests**

```bash
cd mist_desktop && flutter test test/unit/services/websocket_service_test.dart
```

Expected: Existing tests still pass (they use text frames).

- [ ] **Step 3: Commit**

```bash
git add mist_desktop/lib/services/websocket_service.dart
git commit -m "feat(frontend): route binary/text WebSocket frames to separate streams"
```

---

## Task 5: Frontend Audio Playback Service Rewrite

**Files:**
- Rewrite: `mist_desktop/lib/services/audio_playback_service.dart`
- Rewrite: `mist_desktop/test/unit/services/audio_playback_service_test.dart`
- Modify: `mist_desktop/pubspec.yaml`

- [ ] **Step 1: Update pubspec.yaml -- swap audioplayers for flutter_soloud**

In `mist_desktop/pubspec.yaml`, replace:
```yaml
audioplayers: ^6.1.0              # Audio playback for TTS
```
With:
```yaml
flutter_soloud: ^3.1.5            # Low-latency PCM audio streaming
```

Run:
```bash
cd mist_desktop && flutter pub get
```

- [ ] **Step 2: Extract audio_playback_service.dart from branch**

```bash
git show feat/voice-streaming-pipeline:mist_desktop/lib/services/audio_playback_service.dart > mist_desktop/lib/services/audio_playback_service.dart
```

- [ ] **Step 3: Modify for first-chunk immediate playback**

In the extracted `audio_playback_service.dart`, find the jitter buffer threshold logic (the `Buffering` state handler). Modify it so the first chunk triggers immediate transition to `Playing` state without waiting for a buffer threshold.

Find the constant for jitter buffer threshold (should be around 7200 bytes / 150ms). Change the buffering logic to:

```dart
/// Write a PCM16 audio chunk into the playback pipeline.
void writeChunk(Uint8List pcm16Data, int sampleRate) {
  // ... existing validation ...

  switch (_state) {
    case PlaybackState.idle:
      // First chunk of a new response -- start immediately
      _openStream(sampleRate);
      _writeToStream(pcm16Data);
      _state = PlaybackState.playing;
      _logger.info('First chunk received, playing immediately');
      break;
    case PlaybackState.buffering:
      // Should not reach here with immediate-start, but handle gracefully
      _writeToStream(pcm16Data);
      _state = PlaybackState.playing;
      break;
    case PlaybackState.playing:
      _writeToStream(pcm16Data);
      break;
    case PlaybackState.draining:
      // New response while draining previous -- stop old, start new
      stopWithFade();
      _openStream(sampleRate);
      _writeToStream(pcm16Data);
      _state = PlaybackState.playing;
      break;
  }

  _chunkCount++;
}
```

Remove the `Buffering` state entirely from the enum if it's no longer used, or keep it as a no-op pass-through.

- [ ] **Step 4: Add underrun tracking**

Ensure the playback service has underrun diagnostic tracking. The branch version should already have this. Verify the service tracks:
- `_underrunCount` -- incremented when buffer runs empty during playback
- `_chunkCount` -- total chunks received per response
- Logging when underruns exceed 3 per response

- [ ] **Step 5: Extract and update tests**

```bash
git show feat/voice-streaming-pipeline:mist_desktop/test/unit/services/audio_playback_service_test.dart > mist_desktop/test/unit/services/audio_playback_service_test.dart
```

Update test expectations to reflect first-chunk immediate playback (no buffering state).

- [ ] **Step 6: Run tests**

```bash
cd mist_desktop && flutter test test/unit/services/audio_playback_service_test.dart
```

- [ ] **Step 7: Commit**

```bash
git add mist_desktop/pubspec.yaml mist_desktop/pubspec.lock mist_desktop/lib/services/audio_playback_service.dart mist_desktop/test/unit/services/audio_playback_service_test.dart
git commit -m "feat(frontend): rewrite AudioPlaybackService for PCM streaming via flutter_soloud

Replace audioplayers WAV-queue model with continuous PCM16 streaming.
First chunk triggers immediate playback (no buffer threshold). Jitter
buffer handles subsequent chunks. Underrun tracking for diagnostics.
Removes 6s pre-buffer latency penalty."
```

---

## Task 6: Frontend Voice Provider Binary Audio Routing

**Files:**
- Modify: `mist_desktop/lib/providers/voice_provider.dart`

- [ ] **Step 1: Apply provider changes from branch**

Check what the branch changed in the voice provider:
```bash
git diff main...feat/voice-streaming-pipeline -- mist_desktop/lib/providers/voice_provider.dart
```

Apply the key changes:
- Subscribe to `audioFrameStream` from WebSocket service
- Route `MSG_AUDIO_CHUNK` (0x01) to `audioService.writeChunk(frame.payload, frame.sampleRate)`
- Route `MSG_AUDIO_COMPLETE` (0x02) to `audioService.drain()`
- Route `MSG_INTERRUPT_FADE` (0x03) to `audioService.fadeAndClose(frame.payload)`
- Remove old JSON `audio_chunk` / `audio_complete` handling from the text message handler

- [ ] **Step 2: Run flutter analyze**

```bash
cd mist_desktop && flutter analyze
```

Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add mist_desktop/lib/providers/voice_provider.dart
git commit -m "feat(voice): wire VoiceNotifier to binary audio frame stream

Route binary audio frames from WebSocket to AudioPlaybackService.
Handle audio_chunk, audio_complete, and interrupt_fade frame types.
Remove old JSON audio message handling."
```

---

## Task 7: Personality Config Structure + FRIDAY Config

**Files:**
- Create: `voice_profiles/friday/personality.yaml`
- Create: `voice_profiles/jarvis/personality.yaml`
- Create: `voice_profiles/cortana/personality.yaml`
- Create: `tests/unit/test_personality_config.py`

- [ ] **Step 1: Write the test for personality config loading**

Create `tests/unit/test_personality_config.py`:

```python
"""Tests for personality config loading and system prompt templating."""

import os
from pathlib import Path

import yaml


def _load_personality(profile_name: str) -> dict:
    """Load personality config for a voice profile."""
    config_path = (
        Path(__file__).parent.parent.parent
        / "voice_profiles"
        / profile_name
        / "personality.yaml"
    )
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _build_voice_system_prompt(personality: dict) -> str:
    """Build voice system prompt from personality config."""
    if not personality:
        return (
            "You are M.I.S.T, a helpful voice assistant. "
            "Keep responses concise and conversational."
        )

    name = personality.get("name", "M.I.S.T")
    style = personality.get("speaking_style", "").strip()
    openers = personality.get("characteristic_openers", [])
    mannerisms = personality.get("mannerisms", [])

    parts = [f"You are {name}, a personal AI assistant."]

    if style:
        parts.append(f"\nSpeaking style: {style}")

    if openers:
        opener_list = "\n".join(f"- {o}" for o in openers)
        parts.append(
            "\nWhen responding, begin with a brief acknowledgment or "
            "opening phrase before your main answer. Examples of "
            f"characteristic openers you use:\n{opener_list}"
        )

    if mannerisms:
        mannerism_list = "\n".join(f"- {m}" for m in mannerisms)
        parts.append(f"\nAdditional guidelines:\n{mannerism_list}")

    return "\n".join(parts)


class TestPersonalityConfig:
    """Test personality config loading and prompt generation."""

    def test_friday_personality_exists(self):
        config = _load_personality("friday")
        assert config, "friday/personality.yaml should exist and be non-empty"

    def test_friday_has_required_fields(self):
        config = _load_personality("friday")
        assert "name" in config
        assert "characteristic_openers" in config
        assert "speaking_style" in config
        assert len(config["characteristic_openers"]) >= 3

    def test_friday_openers_are_short(self):
        """Openers must be short enough for fast first TTS."""
        config = _load_personality("friday")
        for opener in config["characteristic_openers"]:
            assert len(opener) <= 40, (
                f"Opener too long for first-utterance priming: "
                f"'{opener}' ({len(opener)} chars, max 40)"
            )

    def test_friday_prompt_generation(self):
        config = _load_personality("friday")
        prompt = _build_voice_system_prompt(config)
        assert "FRIDAY" in prompt
        assert "opening phrase" in prompt
        assert "Boss" in prompt

    def test_empty_personality_fallback(self):
        prompt = _build_voice_system_prompt({})
        assert "M.I.S.T" in prompt
        assert "concise" in prompt

    def test_jarvis_stub_loads(self):
        config = _load_personality("jarvis")
        assert isinstance(config, dict)

    def test_cortana_stub_loads(self):
        config = _load_personality("cortana")
        assert isinstance(config, dict)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose run --rm --no-deps mist-backend pytest tests/unit/test_personality_config.py -v
```

Expected: FAIL (personality.yaml files don't exist yet).

- [ ] **Step 3: Create FRIDAY personality config**

Create `voice_profiles/friday/personality.yaml`:

```yaml
name: FRIDAY
address_form: Boss
characteristic_openers:
  - "Sure thing, Boss."
  - "Got it."
  - "Right away."
  - "On it, Boss."
  - "Here's what I found."
  - "One moment."
  - "Working on it."
  - "Right."
speaking_style: |
  Concise, professional, slightly warm. Military-assistant cadence.
  Favor short declarative sentences. No filler or hedging.
  Address the user as "Boss" occasionally but not every response.
  Lead with action or acknowledgment, then substance.
  Keep voice responses to 1-3 sentences for simple questions.
  For detailed requests, provide thorough but structured answers.
mannerisms:
  - 'Uses "Boss" as form of address (2-3x per conversation, not every turn)'
  - Dry humor when appropriate
  - Never says "I think" -- states things directly
  - Avoids unnecessary qualifiers and hedging
  - Prioritizes correctness and accuracy
```

- [ ] **Step 4: Create Jarvis personality stub**

Create `voice_profiles/jarvis/personality.yaml`:

```yaml
name: JARVIS
address_form: Sir
characteristic_openers:
  - "Right away, Sir."
  - "Of course."
  - "Certainly, Sir."
  - "At once."
speaking_style: |
  Formal British butler cadence. Dry humor. Precise and measured.
  Address the user as "Sir" occasionally.
mannerisms:
  - 'Uses "Sir" as form of address'
  - Understated, formal tone
  - Dry humor, subtle and understated
```

- [ ] **Step 5: Create Cortana personality stub**

Create `voice_profiles/cortana/personality.yaml`:

```yaml
name: Cortana
address_form: null
characteristic_openers:
  - "On it."
  - "I've got this."
  - "Here we go."
  - "Let me check."
speaking_style: |
  Direct, confident, slightly playful. No formal address.
  Balances warmth with efficiency.
mannerisms:
  - No formal address form
  - Confident and slightly playful tone
  - Direct without being curt
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
docker compose run --rm --no-deps mist-backend pytest tests/unit/test_personality_config.py -v
```

Expected: All 7 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add voice_profiles/ tests/unit/test_personality_config.py
git commit -m "feat(voice): add personality config system with FRIDAY profile

YAML-based personality configs per voice profile: openers, speaking
style, mannerisms, address form. FRIDAY personality written with
short characteristic openers for first-utterance priming. Jarvis
and Cortana stubs included."
```

---

## Task 8: System Prompt Templating from Personality Config

**Files:**
- Modify: `backend/chat/knowledge_integration.py` (lines 104-231: `generate_tokens_streaming()`)

- [ ] **Step 1: Add personality loading to KnowledgeIntegration**

In `backend/chat/knowledge_integration.py`, add imports at the top:

```python
from pathlib import Path

import yaml
```

Add a personality loading method and cache to the class. After `__init__()`:

```python
def _load_personality(self, profile_name: str) -> dict:
    """Load personality config for a voice profile.

    Args:
        profile_name: Name of the voice profile (e.g., 'friday').

    Returns:
        Personality config dict, or empty dict if not found.
    """
    config_path = (
        Path(__file__).parent.parent.parent
        / "voice_profiles"
        / profile_name
        / "personality.yaml"
    )
    if not config_path.exists():
        logger.debug("No personality config at %s", config_path)
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}

def _build_voice_system_prompt(self, personality: dict) -> str:
    """Build voice system prompt from personality config.

    Args:
        personality: Personality config dict from YAML.

    Returns:
        Formatted system prompt string.
    """
    if not personality:
        return (
            "You are M.I.S.T, a helpful voice assistant and friend to "
            "your creator, Raj Gadhia.\n\n"
            "Response Guidelines:\n"
            "- For simple questions or greetings: 1-3 sentences\n"
            "- For detailed requests: provide complete, thorough responses\n"
            "- Use a warm, friendly tone suitable for spoken conversation\n"
            "- Prioritize correctness, accuracy, and thoroughness\n"
            "- Don't artificially truncate content the user explicitly "
            "requested"
        )

    name = personality.get("name", "M.I.S.T")
    style = personality.get("speaking_style", "").strip()
    openers = personality.get("characteristic_openers", [])
    mannerisms = personality.get("mannerisms", [])

    parts = [
        f"You are {name}, a personal AI assistant and friend to "
        "your creator, Raj Gadhia."
    ]

    if style:
        parts.append(f"\nSpeaking style: {style}")

    if openers:
        opener_list = "\n".join(f"- {o}" for o in openers)
        parts.append(
            "\nWhen responding, begin with a brief acknowledgment or "
            "opening phrase before your main answer. Examples of "
            f"characteristic openers you use:\n{opener_list}"
        )

    if mannerisms:
        mannerism_list = "\n".join(f"- {m}" for m in mannerisms)
        parts.append(f"\nAdditional guidelines:\n{mannerism_list}")

    parts.append(
        "\nResponse Guidelines:\n"
        "- For simple questions or greetings: 1-3 sentences\n"
        "- For detailed requests: provide complete, thorough responses\n"
        "- Prioritize correctness, accuracy, and thoroughness\n"
        "- Don't artificially truncate content the user explicitly "
        "requested"
    )

    return "\n".join(parts)
```

- [ ] **Step 2: Update generate_tokens_streaming() to use personality**

In `generate_tokens_streaming()`, replace the hardcoded `system_prompt` (lines ~168-178) with:

```python
        # Step 2: Build voice-optimized messages with personality
        voice_profile = getattr(self, "_voice_profile", "friday")
        personality = self._load_personality(voice_profile)
        system_prompt = self._build_voice_system_prompt(personality)
```

Add a method to set the active voice profile:

```python
def set_voice_profile(self, profile_name: str) -> None:
    """Set the active voice profile for personality templating."""
    self._voice_profile = profile_name
    logger.info("Voice profile set to: %s", profile_name)
```

- [ ] **Step 3: Wire voice profile to KnowledgeIntegration**

In `backend/voice_processor.py`, find where KnowledgeIntegration is initialized (in `__init__()` or where the voice profile config is loaded). After the knowledge integration object is created, call `set_voice_profile()` with the configured profile name:

```python
# After self.knowledge_integration is created:
if self.knowledge_integration and self.knowledge_integration.enabled:
    voice_profile = self.config.voice_profile or "friday"
    self.knowledge_integration.set_voice_profile(voice_profile)
```

Verify that `self.config.voice_profile` or equivalent exists by reading `backend/config.py`. The config key is `VOICE_PROFILE` from `.env` (currently defaults to `friday`).

- [ ] **Step 4: Run tests**

```bash
docker compose run --rm --no-deps mist-backend pytest tests/unit/test_personality_config.py tests/unit/ -v --tb=short
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add backend/chat/knowledge_integration.py backend/voice_processor.py
git commit -m "feat(voice): template system prompt from personality config

Load personality YAML for active voice profile, build voice system
prompt with characteristic openers, speaking style, and mannerisms.
Falls back to generic M.I.S.T prompt if no personality config found."
```

---

## Task 9: E2E Validation

**No files changed -- manual testing.**

- [ ] **Step 1: Start the stack**

```bash
python scripts/start_dev.py
```

Wait for health checks to pass.

- [ ] **Step 2: Start Flutter**

```bash
cd mist_desktop && flutter run -d windows
```

- [ ] **Step 3: Validate binary audio transport**

Send a text message via the chat interface. Verify:
- Audio plays back (confirms binary frames are being sent and parsed)
- No JSON errors in backend logs related to audio
- Backend logs show "TTS: First audio chunk" timing

- [ ] **Step 4: Validate gapless playback**

Send a message that produces a multi-sentence response (e.g., "Tell me about yourself"). Verify:
- Audio plays continuously without gaps between sentences
- Volume is consistent across chunks (RMS normalization working)
- No stuttering or skipping

- [ ] **Step 5: Validate interrupt**

While audio is playing, start speaking. Verify:
- Audio fades out cleanly (no pop or hard cut)
- STT picks up your speech
- New response begins normally

- [ ] **Step 6: Validate first-utterance priming**

Check backend logs for TTS timing. The first TTS chunk should be short (opener like "Sure thing, Boss." or "Got it."). Verify:
- First sentence is under 40 chars
- TTS generation for first chunk is faster than before (~0.8-1.0s vs ~2.5s)
- TTFA improved from ~4-5s to ~2.5-3.5s

- [ ] **Step 7: Check for underruns**

In Flutter logs, look for underrun messages. Acceptable: 0-1 underruns per response. If more than 3, the jitter buffer needs tuning.

---

## Task 10: Streaming TTS Research

**Files:**
- Create: vault research note (`D:\Users\rajga\knowledge-vault\04-Research\2026-03-XX-streaming-tts-evaluation.md`)

This is a research task -- no implementation code.

- [ ] **Step 1: Research Direction A -- Streaming TTS alternatives**

For each candidate model, evaluate against criteria:

| Model | Streaming? | Voice Cloning | RTF | VRAM | License | Quality |
|-------|-----------|---------------|-----|------|---------|---------|

Candidates:
- **Kokoro** (82M params): Check if output is streaming or batch
- **F5-TTS**: Flow matching architecture -- does it support chunked generation?
- **MaskGCT**: Masked generative codec transformer -- streaming capability?
- **XTTS v2** (Coqui): Known streaming support -- evaluate quality + VRAM
- **Any new models** since MIS-75 evaluation (2026-03-25)

For each: read the paper/repo, check API for streaming output, note VRAM requirements, check license.

- [ ] **Step 2: Research Direction B -- Chatterbox sub-chunk streaming**

Investigate Chatterbox internals:
- Read the Chatterbox source code (installed via pip: `chatterbox-tts`)
- Identify where codec tokens are generated autoregressively
- Determine if partial codec token sequences can be decoded to audio
- What is the minimum token count for coherent audio output?
- Does the T3 transformer expose hooks for mid-generation extraction?

```bash
pip show chatterbox-tts | grep Location
# Then read the source files for the generation loop
```

- [ ] **Step 3: Document findings**

Create a vault research note with:
- Evaluation table for each candidate model
- Chatterbox sub-chunk feasibility assessment
- Recommendation: swap model, modify Chatterbox, or stay current
- If swap recommended: migration path and voice profile compatibility concerns

Propose the research note to the user for vault storage.

---

## Task 11: Linear Ticket Updates + Cleanup

- [ ] **Step 1: Update all Linear tickets with implementation status**

After E2E validation passes:
- MIS-76: Mark In Progress or Done depending on E2E results
- New tickets created in Task 0: Update with commit hashes and status
- Close any tickets that are fully addressed

- [ ] **Step 2: Update CODEBASE.md**

Update to reflect:
- Binary WebSocket transport active
- Adaptive pre-buffering (first-chunk immediate, no 6s delay)
- Personality profile system active (FRIDAY default)
- Expected TTFA: ~2.5-3.5s
- feat/voice-streaming-pipeline branch merged/integrated

- [ ] **Step 3: Final commit**

```bash
git add CODEBASE.md
git commit -m "docs: update CODEBASE.md after voice pipeline final optimizations"
```
