# Backend - Voice Conversation Server

WebSocket-based voice conversation backend for M.I.S.T AI.

## Architecture

```
WebSocket Client
    ->
FastAPI Server (server.py)
    ->
Voice Processor (voice_processor.py)
    ->
┌─────────────────────────────────────┐
│ VAD -> STT -> LLM -> TTS -> Audio Out  │
└─────────────────────────────────────┘
    ->
Model Manager (voice_models/model_manager.py)
```

## Components

### server.py
FastAPI WebSocket server managing client connections and message routing.

**Endpoints:**
- `ws://localhost:8001/ws` - Main WebSocket endpoint

**Message Protocol:** See [Message Types](#message-types) below

### voice_processor.py
Orchestrates the voice conversation pipeline using Silero VAD callbacks.

**Flow:**
1. Audio chunks arrive via WebSocket
2. VAD detects speech/silence
3. On speech end: STT transcribes -> LLM generates -> TTS synthesizes
4. Audio streams back via WebSocket

**Features:**
- Real-time VAD with <100ms latency
- Streaming TTS audio (chunk-by-chunk delivery)
- Conversation state management

### voice_models/model_manager.py
Manages model lifecycle and TTS worker thread.

**Key Features:**
- Lazy model loading (STT on first use, TTS on first use)
- Dedicated TTS worker thread to prevent blocking
- Context management for voice consistency
- Sentence-boundary chunking for long responses (>150 tokens)
- Token-aware generation limits

**TTS Chunking Strategy:**
- Short responses (<150 tokens): Single generation
- Long responses (>150 tokens): Sentence-boundary chunking at ~120 tokens/chunk
- Context preserved: 3 reference clips + 2 recent utterances
- Automatic cleanup after chunked generation

## Message Types

### Client -> Server

#### 1. Audio Chunk
```json
{
  "type": "audio_chunk",
  "data": "<base64_encoded_audio>",
  "sample_rate": 16000
}
```
Raw audio from microphone (16kHz mono, float32)

#### 2. Stop Generation
```json
{
  "type": "stop"
}
```
Interrupt current TTS generation

### Server -> Client

#### 1. Transcription
```json
{
  "type": "transcription",
  "text": "User's spoken text..."
}
```
STT result after speech detection

#### 2. Response Text
```json
{
  "type": "response",
  "text": "AI response text..."
}
```
LLM-generated response (before audio)

#### 3. Audio Chunk
```json
{
  "type": "audio_chunk",
  "data": "<base64_encoded_audio>"
}
```
TTS audio chunk (24kHz mono, float32)

Sent continuously as audio is generated (streaming).

#### 4. Audio End
```json
{
  "type": "audio_end"
}
```
Signals completion of TTS generation

#### 5. Status Update
```json
{
  "type": "status",
  "message": "Thinking...",
  "stage": "llm" | "tts" | "ready"
}
```
Progress updates during processing

#### 6. Error
```json
{
  "type": "error",
  "message": "Error description"
}
```
Error during processing

## Configuration

Configuration is managed via `config.py` using Pydantic:

```python
class VoiceConfig(BaseModel):
    # STT
    whisper_model: str = "base"

    # VAD
    vad_threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 500

    # LLM
    llm_model: str = "qwen2.5:latest"

    # TTS
    tts_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_voice_context: bool = True
    tts_temperature: float = 0.55
    tts_topk: int = 20
```

## Model Details

### STT: OpenAI Whisper
- **Model:** `base` (74M params)
- **Language:** Auto-detect
- **Latency:** ~200-500ms per utterance

### LLM: Ollama
- **Default Model:** Qwen 2.5 (7B)
- **System Prompt:** Conversational AI assistant (M.I.S.T)
- **Token Limit:** 400 tokens max (150 tokens preferred for quality)

### TTS: Sesame CSM-1B
- **Model:** Fine-tuned on Elise dataset (epoch 20)
- **Sample Rate:** 24kHz
- **Voice:** Consistent female voice (Elise)
- **Latency:** ~2-3s RTF (Real-Time Factor)

## Running the Server

### Standard Mode
```bash
python backend/server.py
```

Server starts on `ws://localhost:8001/ws`

### With Custom Config
```python
from backend.config import VoiceConfig
from backend.voice_processor import VoiceProcessor

config = VoiceConfig(
    whisper_model="small",
    vad_threshold=0.6,
    llm_model="mistral:latest"
)

processor = VoiceProcessor(config, message_queue)
```

## Threading Model

```
Main Thread (FastAPI)
    ->
Voice Processor Thread
    ->
┌──────────────────────────┐
│ TTS Worker Thread        │
│ - Dedicated thread       │
│ - Queue-based requests   │
│ - Prevents UI blocking   │
└──────────────────────────┘
```

**Why Threading:**
- TTS generation can take 10-60 seconds for long responses
- Worker thread prevents blocking VAD/STT pipeline
- Queue ensures orderly generation
- Clean shutdown on disconnect

## Performance Characteristics

### Latency Breakdown
```
User stops speaking
  -> 100ms (VAD detection)
STT transcription
  -> 300ms (Whisper)
LLM generation
  -> 1-3s (Qwen streaming)
TTS generation
  -> 2-5s (CSM first chunk)
Audio playback starts
  -> Streaming continues...
Complete audio
```

**Total Time to First Audio:** ~3-6 seconds

### Memory Usage
- **Whisper base:** ~400MB VRAM
- **CSM-1B:** ~3GB VRAM
- **Total:** ~4GB VRAM minimum

### Long Response Handling
- Responses >150 tokens automatically chunked
- Each chunk: ~120 tokens = ~900 audio frames
- Chunking prevents quality degradation
- Context preserved between chunks

## Troubleshooting

### TTS Quality Degradation
**Symptom:** Audio gets quieter or stutters during long generations

**Cause:** Position overflow in CSM generator

**Fix:** Applied sliding window fix in `dependencies/csm/generator.py` line 233-237

### CUDA Out of Memory
**Symptom:** RuntimeError: CUDA out of memory

**Solutions:**
1. Reduce context: Set `use_voice_context=False` in config
2. Use smaller Whisper: `whisper_model="tiny"`
3. Close other GPU applications

### Connection Issues
**Symptom:** WebSocket disconnects frequently

**Solutions:**
1. Check firewall settings
2. Increase timeout in client
3. Monitor server logs for errors

## Development

### Adding New Models

**STT:**
```python
# In voice_models/model_manager.py
self.stt = YourSTTWrapper()
```

**LLM:**
```python
# Modify generate_llm_response() in model_manager.py
# Ensure streaming support
```

**TTS:**
```python
# In voice_models/model_manager.py
self.tts = YourTTSWrapper()
# Must implement generate_stream() or similar
```

### Testing

**Unit Test:**
```bash
pytest tests/test_voice_processor.py
```

**Integration Test:**
```bash
python cli_client/voice_client.py
# Speak into microphone, verify response
```

## Future Enhancements

- [ ] Multi-speaker support
- [ ] Voice activity visualization
- [ ] Conversation history persistence
- [ ] RAG integration for knowledge retrieval
- [ ] Emotion detection and voice adaptation
- [ ] Real-time translation

## Related Documentation

- [Model Manager Details](voice_models/README.md)
- [Frontend Integration](../frontend/README.md)
- [CLI Client Usage](../cli_client/README.md)
- [Architecture Decisions](../docs/decisions/)
