# Multimodal Components

Speech-to-text, text-to-speech, and voice interface implementations for M.I.S.T AI.

## Overview

This directory contains the core multimodal components for voice conversation:

```
multimodal/
├── stt.py                          # Speech-to-Text (Whisper)
├── tts.py                          # Text-to-Speech (Sesame CSM)
├── voice_interface_streaming.py    # Streaming voice interface
└── voice_interface_interrupt.py    # Interruptible voice interface
```

## Components

### stt.py - Speech-to-Text

**Model:** OpenAI Whisper
**Supported Sizes:** tiny, base, small, medium, large
**Default:** base (74M parameters)

#### WhisperSTT Class

```python
from src.multimodal.stt import WhisperSTT

stt = WhisperSTT(model_size="base")
text = stt.listen(duration=5)  # Record and transcribe
# Or
text = stt.transcribe_audio(audio_data, sample_rate=16000)
```

**Methods:**
- `listen(duration)` - Record from microphone and transcribe
- `transcribe_audio(audio_data, sample_rate)` - Transcribe audio array
- `transcribe_file(audio_path)` - Transcribe audio file

**Features:**
- Auto-detects language
- Resamples audio to 16kHz (Whisper's native rate)
- Handles float32 numpy arrays
- Thread-safe (no shared state)

**Performance:**
| Model | Size | VRAM | Latency (5s audio) |
|-------|------|------|--------------------|
| tiny  | 39M  | 150MB | ~100ms |
| base  | 74M  | 300MB | ~300ms |
| small | 244M | 800MB | ~500ms |
| medium| 769M | 2GB | ~1s |
| large | 1.5B | 4GB | ~2s |

---

### tts.py - Text-to-Speech

**Model:** Sesame CSM-1B (Fine-tuned)
**Voice:** Elise dataset (consistent female voice)
**Sample Rate:** 24kHz

#### SesameTTS Class

```python
from src.multimodal.tts import SesameTTS

tts = SesameTTS(device="cuda", use_context=True)
audio = tts.speak("Hello, this is M.I.S.T!", play=False)
```

**Methods:**
- `speak(text, output_path, play, temperature, topk, streaming)` - Generate speech

**Parameters:**
- `text` (str): Text to synthesize
- `output_path` (str, optional): Save audio to file
- `play` (bool): Play audio immediately via sounddevice
- `temperature` (float): Sampling temperature (0.55 default)
- `topk` (int): Top-k sampling (20 default)
- `streaming` (bool): Enable streaming generation

**Features:**
- **Context preservation** - Maintains voice consistency across utterances
- **Reference audio loading** - Initializes with 3 Elise dataset clips
- **Streaming generation** - Chunk-by-chunk audio delivery
- **Fine-tuned model** - Epoch 20 checkpoint (best validation loss: 6.110)

**Context Management:**
```python
# Context is automatically managed:
tts.speak("First sentence")   # Uses references only
tts.speak("Second sentence")  # Uses references + previous audio
# Context: [ref1, ref2, ref3, prev_audio]
```

**Voice Consistency Tips:**
- Lower temperature (0.5-0.6) = more consistent
- Lower top-k (15-25) = more focused
- Enable context (`use_context=True`) = better continuity
- Keep context size moderate (3 refs + 2 recent = 5 total)

**Performance:**
- Real-Time Factor (RTF): ~2-3x (generates 2-3s per second of audio)
- Warmup time: ~5s (first generation)
- Subsequent generations: ~3-5s for 10-20 words

---

### voice_interface_streaming.py

**Status:** ✅ Production Ready
**Use Case:** Standard voice conversation with streaming audio

#### StreamingVoiceInterface Class

```python
from src.multimodal.voice_interface_streaming import StreamingVoiceInterface

vi = StreamingVoiceInterface()
vi.start_conversation()
```

**Features:**
- VAD-based speech detection (Silero VAD)
- Streaming TTS audio (chunk-by-chunk delivery)
- Real-time microphone input
- Seamless audio playback (OutputStream pattern)

**Flow:**
```
Microphone Input
    ↓
VAD Detection
    ↓
[Speech Detected]
    ↓
STT Transcription
    ↓
LLM Generation
    ↓
TTS Synthesis (streaming)
    ↓
Audio Playback (chunks)
```

**Configuration:**
```python
vi = StreamingVoiceInterface()
vi.vad.set_threshold(0.5)  # Adjust sensitivity
vi.config.min_speech_duration_ms = 250
vi.config.min_silence_duration_ms = 500
```

---

### voice_interface_interrupt.py

**Status:** ✅ Production Ready
**Use Case:** Interruptible conversation with <100ms latency

#### InterruptibleVoiceInterface Class

```python
from src.multimodal.voice_interface_interrupt import InterruptibleVoiceInterface

vi = InterruptibleVoiceInterface(vad_threshold=0.5)
vi.start_conversation()
```

**Features:**
- Everything from StreamingVoiceInterface
- **Interrupt support:** Stop AI mid-sentence when user speaks
- **Ultra-low latency:** <100ms VAD response
- **Clean state management:** Proper cleanup on interrupt

**Interrupt Flow:**
```
AI Speaking
    ↓
[User Starts Speaking]
    ↓
VAD Detects Speech (50ms)
    ↓
Stop TTS Generation
    ↓
Clear Audio Buffer
    ↓
Listen to User
```

**Use Cases:**
- Natural conversations (back-and-forth)
- Quick Q&A sessions
- Interactive tutoring
- Real-time collaboration

---

## Model Selection Guide

### When to Use Each STT Model

**Tiny (39M):**
- Real-time requirements (<200ms latency)
- Low VRAM environments
- English-only conversations
- ❌ Poor accuracy on accents/noise

**Base (74M) - Recommended:**
- Balanced accuracy and speed
- Multi-language support
- Good noise tolerance
- ✅ Best default choice

**Small (244M):**
- Higher accuracy needed
- Technical/medical terminology
- Multiple speakers
- Can spare 500ms latency

**Medium/Large (769M+):**
- Professional transcription quality
- Mission-critical accuracy
- Unlimited VRAM available
- Latency not a concern

### When to Use Each Voice Interface

**StreamingVoiceInterface:**
- Standard conversations
- Presentations or storytelling
- When user waits for complete responses
- Simpler implementation

**InterruptibleVoiceInterface:**
- Natural back-and-forth dialogue
- Q&A sessions
- Tutoring or coaching
- User impatience tolerance low

---

## Integration Examples

### Standalone Usage

```python
# Simple voice loop
from src.multimodal.voice_interface_streaming import StreamingVoiceInterface

vi = StreamingVoiceInterface()
print("Speak now! (Ctrl+C to exit)")
vi.start_conversation()
```

### WebSocket Integration

```python
# Backend integration (see backend/voice_processor.py)
from src.multimodal.stt import WhisperSTT
from src.multimodal.tts import SesameTTS

# Initialize models
stt = WhisperSTT("base")
tts = SesameTTS(device="cuda")

# In WebSocket message handler:
async def handle_audio_chunk(audio_data):
    # VAD determines speech boundaries
    if vad.is_speech_end():
        text = stt.transcribe_audio(audio_buffer)
        response = llm.generate(text)

        # Stream TTS audio
        for audio_chunk in tts.generate_stream(response):
            await websocket.send_audio(audio_chunk)
```

### Custom Model Integration

```python
# Replace STT with custom model
class CustomSTT:
    def transcribe_audio(self, audio_data, sample_rate):
        # Your STT implementation
        return transcribed_text

# Replace TTS with custom model
class CustomTTS:
    def speak(self, text, **kwargs):
        # Your TTS implementation
        return audio_tensor

# Use in voice interface
vi.stt = CustomSTT()
vi.tts = CustomTTS()
```

---

## Advanced Configuration

### Fine-Tuning Voice Characteristics

```python
tts = SesameTTS()

# More robotic/precise
tts.speak(text, temperature=0.3, topk=10)

# More natural/varied
tts.speak(text, temperature=0.8, topk=40)

# Elise default (recommended)
tts.speak(text, temperature=0.55, topk=20)
```

### Context Window Management

```python
# Aggressive context trimming (less memory, slightly less consistent)
tts.context = tts.context[:3] + tts.context[-1:]  # 3 refs + 1 recent

# Generous context (more memory, more consistent)
tts.context = tts.context[:3] + tts.context[-3:]  # 3 refs + 3 recent
```

### VAD Sensitivity Tuning

```python
vi = InterruptibleVoiceInterface()

# More sensitive (faster response, more false positives)
vi.vad.set_threshold(0.3)

# Less sensitive (slower response, fewer false positives)
vi.vad.set_threshold(0.7)

# Default (balanced)
vi.vad.set_threshold(0.5)
```

---

## Performance Optimization

### Reduce Latency

1. **Use smaller STT model:** `WhisperSTT("tiny")`
2. **Lower VAD thresholds:** `vad_threshold=0.4`
3. **Shorter silence duration:** `min_silence_duration_ms=300`
4. **GPU acceleration:** Ensure CUDA available
5. **Preload models:** Initialize before user interaction

### Reduce Memory Usage

1. **Disable TTS context:** `SesameTTS(use_context=False)`
2. **Use smaller Whisper:** `WhisperSTT("tiny")`
3. **Trim context aggressively:** Keep only 3-4 segments
4. **Close other GPU applications**

### Improve Voice Quality

1. **Use fine-tuned model:** Already using epoch-20 checkpoint ✅
2. **Enable context:** `use_context=True` ✅
3. **Lower temperature:** `temperature=0.5-0.6`
4. **Consistent topk:** `topk=15-25`
5. **Quality audio input:** Use good microphone, quiet environment

---

## Troubleshooting

### STT Issues

**Problem:** "Failed to load model"
- **Solution:** Run `pip install openai-whisper` and ensure internet connection for first download

**Problem:** Poor transcription accuracy
- **Solutions:**
  1. Increase model size: `WhisperSTT("small")`
  2. Improve audio quality (better mic, less background noise)
  3. Speak more clearly and slowly
  4. Check `sample_rate=16000` is correct

**Problem:** High latency
- **Solutions:**
  1. Use smaller model: `WhisperSTT("tiny")`
  2. Ensure GPU acceleration active
  3. Reduce audio buffer size

### TTS Issues

**Problem:** Audio quality degradation on long responses
- **Solution:** Already fixed via sentence-boundary chunking at 150 tokens ✅

**Problem:** Voice inconsistency between utterances
- **Solutions:**
  1. Enable context: `use_context=True`
  2. Lower temperature: `temperature=0.5`
  3. Keep context size moderate (5 segments)

**Problem:** CUDA out of memory
- **Solutions:**
  1. Reduce context size to 3 segments
  2. Use `device="cpu"` (slower but no VRAM limit)
  3. Close other GPU applications
  4. Restart Python process to clear cache

**Problem:** "Index out of bounds" error after ~2000 frames
- **Solution:** Already fixed via sliding window in generator.py ✅

### Voice Interface Issues

**Problem:** VAD not detecting speech
- **Solutions:**
  1. Lower threshold: `vad_threshold=0.3`
  2. Check microphone input level
  3. Verify `sounddevice` can access microphone

**Problem:** VAD triggering on silence
- **Solutions:**
  1. Raise threshold: `vad_threshold=0.6`
  2. Increase `min_speech_duration_ms=500`
  3. Reduce background noise

**Problem:** Interrupt not working
- **Solutions:**
  1. Use `InterruptibleVoiceInterface` (not Streaming)
  2. Lower VAD threshold for faster detection
  3. Ensure TTS uses streaming generation

---

## Testing

### Unit Tests
```bash
pytest tests/test_stt.py
pytest tests/test_tts.py
pytest tests/test_voice_interface.py
```

### Manual Testing
```bash
# Test STT
python -c "from src.multimodal.stt import WhisperSTT; stt = WhisperSTT(); print(stt.listen(5))"

# Test TTS
python -c "from src.multimodal.tts import SesameTTS; tts = SesameTTS(); tts.speak('Hello world', play=True)"

# Test Voice Interface
python test_streaming_simple.py
```

---

## Future Enhancements

### Planned Features
- [ ] Multi-language voice cloning
- [ ] Emotion-aware TTS
- [ ] Real-time voice conversion
- [ ] Speaker diarization (multi-speaker support)
- [ ] Noise cancellation preprocessing
- [ ] Voice activity visualization

### Research Directions
- [ ] Zero-shot voice cloning (single reference clip)
- [ ] Cross-lingual voice transfer
- [ ] Prosody control (speed, pitch, emphasis)
- [ ] Integration with vision models (multi-modal grounding)

---

## Related Documentation

- [Backend README](../../backend/README.md) - WebSocket server integration
- [Fine-Tuning Guide](../../docs/guides/voice_finetuning.md) - How to fine-tune TTS
- [Architecture Decisions](../../docs/decisions/) - Design rationale
- [CSM Model Documentation](../../dependencies/csm/README.md) - TTS model details
