# MIST.AI - Repository Structure

Production-ready repository with Flutter desktop frontend + Python backend.

---

## Root Directory Files

### Configuration
- `.env` - Environment variables (Neo4j, API keys)
- `.python-version` - Python version specification
- `requirements.txt` - Python dependencies

### Documentation
- `README.md` - Main project documentation
- `E2E_TEST_GUIDE.md` - End-to-end testing instructions
- `INTEGRATION_STATUS.md` - Knowledge integration status
- `QUICKSTART_KNOWLEDGE_INTEGRATION.md` - Quick start guide
- `NEO4J_QUERIES.md` - Useful Neo4j queries reference
- `LICENSE` - License file
- `NOTICE` - Attribution notices

### Utility Scripts
- `regenerate_graph.py` - Rebuild knowledge graph from utterances
- `initialize_schema.py` - Initialize Neo4j schema (indexes, constraints)
- `wipe_neo4j.py` - Clean Neo4j database
- `export_graph.py` - Export graph structure for analysis

### Test Scripts
- `test_neo4j_connection.py` - Quick Neo4j connection test
- `test_conversation_handler.py` - Test conversation flow with knowledge integration

---

## Backend Structure

```
backend/
├── server.py                    # Main WebSocket server (port 8001)
├── config.py                    # Voice/system configuration
├── voice_processor.py           # Voice processing pipeline
├── knowledge_config.py          # Knowledge graph configuration
│
├── voice_models/
│   └── model_manager.py         # ML models + knowledge integration
│
├── chat/                        # Conversation handling
│   ├── __init__.py
│   ├── conversation_handler.py  # MCP-like autonomous tool use
│   └── knowledge_integration.py # Integration bridge for voice system
│
└── knowledge/                   # Knowledge graph system
    ├── config.py               # Extraction configuration
    ├── models.py               # Data models (Entity, Utterance, etc.)
    │
    ├── extraction/             # Entity extraction
    │   ├── entity_extractor.py
    │   └── property_enricher.py
    │
    ├── retrieval/              # Knowledge retrieval
    │   └── knowledge_retriever.py
    │
    ├── storage/                # Neo4j storage
    │   └── graph_store.py
    │
    └── embeddings/             # Vector embeddings
        └── embedding_generator.py
```

---

## Flutter Frontend Structure

```
mist_desktop/                    # Flutter desktop app
├── lib/
│   ├── main.dart               # App entry point
│   │
│   ├── config/
│   │   ├── app_config.dart     # App configuration (WebSocket URL, etc.)
│   │   └── theme_config.dart   # Material theme (dark mode)
│   │
│   ├── models/
│   │   ├── message_model.dart  # Chat message model
│   │   └── websocket_message.dart # WebSocket message types
│   │
│   ├── providers/              # Riverpod state providers
│   │   ├── chat_provider.dart  # Chat state & logic
│   │   ├── websocket_provider.dart # WebSocket connection state
│   │   ├── audio_provider.dart # Audio recording/playback state
│   │   └── speech_provider.dart # Speech processing state
│   │
│   ├── services/               # Business logic services
│   │   ├── websocket_service.dart # WebSocket client
│   │   ├── audio_recording_service.dart # Voice recording
│   │   ├── audio_playback_service.dart # TTS audio playback
│   │   └── speech_service.dart # Speech coordination
│   │
│   ├── screens/
│   │   └── chat_screen.dart    # Main chat UI
│   │
│   └── widgets/                # Reusable UI components
│       ├── chat_message_widget.dart
│       ├── connection_status_widget.dart
│       └── voice_input_button.dart
│
├── windows/                     # Windows platform code
├── pubspec.yaml                 # Flutter dependencies
└── README.md                    # Flutter app documentation
```

---

## Documentation Structure

```
docs/
├── guides/
│   ├── windows_dev_setup.md        # Development environment setup
│   └── TORCH_COMPILE_FIX.md        # PyTorch optimization fix
│
└── decisions/
    ├── adr_001_vision.md           # Project vision & philosophy
    └── adr_007_sesame_csm.md       # TTS selection rationale

Root documentation:
├── README.md                        # Main project readme
├── FLUTTER_MIGRATION_PLAN.md        # Comprehensive Flutter migration guide (1600 lines)
├── INTEGRATION_STATUS.md            # Knowledge graph integration status
├── QUICKSTART_KNOWLEDGE_INTEGRATION.md # Quick KG setup
├── E2E_TEST_GUIDE.md               # End-to-end testing guide
├── NEO4J_QUERIES.md                # Useful Neo4j query reference
└── REPOSITORY_STRUCTURE.md          # This file
```

---

## Key Features

### 1. Knowledge Graph System
- **Extraction**: Converts conversations to structured knowledge
- **Storage**: Neo4j with provenance tracking
- **Retrieval**: Vector search + graph traversal
- **Regeneration**: Rebuild from immutable utterances

### 2. MCP-Like Conversation
- **Autonomous tool use**: LLM decides when to query/extract
- **Two tools**: `query_knowledge_graph`, `extract_knowledge`
- **Session management**: Track conversation history
- **Graceful fallback**: Works without Neo4j

### 3. Voice System (Backend)
- **WebSocket server**: Real-time voice communication (port 8001)
- **STT**: Whisper-based speech recognition
- **TTS**: Sesame CSM-1B speech synthesis
- **VAD**: Voice activity detection (Silero)

### 4. Flutter Desktop Frontend
- **Cross-platform**: Windows, macOS, Linux support
- **Voice recording**: Real-time audio capture
- **Audio playback**: TTS audio from backend
- **WebSocket client**: Real-time backend communication
- **State management**: Riverpod 3.x for reactive UI

---

## Integration Points

### Flutter Frontend -> Backend
```
Flutter App (mist_desktop/)
  -> WebSocket (ws://localhost:8001)
Backend Server (server.py)
  -> Receives audio/text messages
Voice Processor -> STT -> LLM -> TTS
  -> Sends transcription/tokens/audio
Flutter App (displays messages, plays audio)
```

### Backend -> Knowledge Graph
```python
# In model_manager.py:
if self.knowledge and self.knowledge.is_enabled():
    # Use knowledge-augmented LLM
    for token in self.knowledge.generate_response_streaming(user_text):
        yield token
else:
    # Fallback to standard LLM
    ...
```

### Knowledge Graph -> Neo4j
```
ConversationHandler
  -> (autonomous tool use)
query_knowledge_graph -> KnowledgeRetriever -> GraphStore -> Neo4j
extract_knowledge -> EntityExtractor -> GraphStore -> Neo4j
```

---

## Environment Variables

Required in `.env`:
```bash
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Features
ENABLE_KNOWLEDGE_INTEGRATION=true

# Optional
HF_TOKEN=your_huggingface_token
MODEL=qwen2.5:7b-instruct
```

---

## Dependencies

Key packages (from requirements.txt):
- `neo4j` - Graph database driver
- `langchain-ollama` - LLM integration
- `sentence-transformers` - Embeddings
- `fastapi` - WebSocket server
- `torch` - ML models
- `sounddevice` - Audio I/O

---

## Usage

### Start System
```bash
# 1. Start Neo4j
neo4j start

# 2. Start backend
venv/Scripts/python.exe backend/server.py

# 3. Start Flutter frontend (separate terminal)
cd mist_desktop
flutter run -d windows
```

### Run Tests
```bash
# Test Neo4j connection
venv/Scripts/python.exe test_neo4j_connection.py

# Test conversation handler
venv/Scripts/python.exe test_conversation_handler.py --mode simple
```

### Maintenance
```bash
# Regenerate graph from utterances
venv/Scripts/python.exe regenerate_graph.py

# Initialize/reset schema
venv/Scripts/python.exe initialize_schema.py

# Wipe database (careful!)
venv/Scripts/python.exe wipe_neo4j.py
```

---

## Development Workflow

### Adding New Knowledge Features
1. Define data models in `backend/knowledge/models.py`
2. Add extraction logic in `backend/knowledge/extraction/`
3. Add retrieval logic in `backend/knowledge/retrieval/`
4. Update ConversationHandler tools if needed

### Adding New Voice Features
1. Modify `backend/voice_processor.py` for processing
2. Update `backend/voice_models/model_manager.py` for models
3. Add WebSocket message types in `backend/server.py`
4. Update Flutter message handlers in `mist_desktop/lib/providers/chat_provider.dart`

### Adding Flutter UI Features
1. Create new widgets in `mist_desktop/lib/widgets/`
2. Add state providers in `mist_desktop/lib/providers/`
3. Update screens in `mist_desktop/lib/screens/`
4. Add services if needed in `mist_desktop/lib/services/`

### Testing
1. Backend unit tests: Test individual components
2. Integration: Use `test_conversation_handler.py`
3. E2E: Follow `E2E_TEST_GUIDE.md`
4. Flutter tests: `cd mist_desktop && flutter test`

---

## Production Checklist

Before deploying:
- [ ] Update Neo4j password in `.env`
- [ ] Configure CORS in `server.py`
- [ ] Set proper log levels
- [ ] Test with production Neo4j instance
- [ ] Verify knowledge extraction quality
- [ ] Monitor performance metrics
- [ ] Set up Neo4j backups

---

## File Count Summary

**Root files:** ~20
- Config: 3 (.env, .gitignore, .python-version)
- Docs: 7 (README + guides)
- Scripts: 6 (test, init, regenerate, wipe, export)

**Backend files:** ~20 Python modules
- Voice: 3 (server, processor, config)
- Chat: 2 (handler, integration)
- Knowledge: 8 (extraction, retrieval, storage, embeddings)
- Config: 2 (config, knowledge_config)

**Flutter files:** ~20 Dart files
- Screens: 1 (chat_screen)
- Providers: 4 (chat, websocket, audio, speech)
- Services: 4 (websocket, audio recording/playback, speech)
- Widgets: 3 (message, status, voice button)
- Models: 2 (message, websocket_message)
- Config: 2 (app_config, theme_config)

**Documentation:** 7 comprehensive guides

**Tests:** 3 test scripts (neo4j, conversation, vector search)

---

## Migration History

### React -> Flutter Migration (Dec 2024)
**Removed:**
- React/TypeScript frontend (~127MB)
- Old frontend architecture docs

**Added:**
- Flutter desktop app (mist_desktop/)
- Comprehensive Flutter migration plan (1600 lines)
- Riverpod state management
- Audio recording/playback services
- WebSocket integration

**Reason:** Flutter provides:
- True cross-platform (Windows/macOS/Linux/iOS/Android)
- Better desktop performance
- Native audio handling
- Single codebase for mobile expansion

---

## Current Status

**Backend:**  Production-ready
- Voice pipeline complete
- Knowledge graph integrated
- Autonomous tool usage working
- WebSocket server stable

**Frontend:**  In Development
- Flutter UI scaffolding complete
- WebSocket connection working
- Voice recording implemented
- Audio playback needs TTS enabling (TTS_ENABLED=false in .env)
- Graph visualization planned

---

## Next Steps

1. **Enable TTS** in backend (.env: TTS_ENABLED=true)
2. **Test audio playback** in Flutter app
3. **Polish Flutter UI** - animations, error states
4. **Add graph visualization** in Flutter
5. **Production deployment** prep
