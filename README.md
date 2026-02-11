# MIST.AI

A self-aware hybrid cognitive architecture combining transparent local reasoning with strategic cloud delegation.

**Current Status:** Fully functional voice conversation system with integrated knowledge graph, autonomous tool usage, and auto-RAG. Frontend UI in development.

## What Is This?

MIST.AI is a research platform building a transparent, continuously learning AI system. This is **not a ChatGPT wrapper** - it's a cognitive architecture built from first principles with persistent memory, autonomous reasoning, and strategic intelligence.

### Core Philosophy

- **Transparency**: See every decision the AI makes - every entity extracted, every tool called
- **Privacy-first**: Runs entirely local, air-gapped capable
- **Continuous Growth**: Becomes more knowledgeable over time through accumulated personal knowledge
- **Research-focused**: A learning platform for exploring AI architecture

### Key Capabilities

-  **Real-time voice conversation** with natural interruption support
-  **Knowledge graph integration** - persistent memory that grows over time
-  **Autonomous tool usage** - LLM decides when to extract/query knowledge (MCP-like pattern)
-  **Auto-RAG** - automatic document injection for context-aware responses
-  **Personalized responses** - leverages accumulated user knowledge
-  **WebSocket architecture** - production-ready real-time communication
-  **Local-first** - complete air-gapped operation

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Flutter Desktop (mist_desktop/)                                 │
│  - Windows/macOS/Linux support                                   │
│  - Voice recording & playback                                    │
│  - Real-time conversation UI                                     │
│  - WebSocket communication                                       │
└────────────────────────┬─────────────────────────────────────────┘
                         │ WebSocket (Port 8001)
┌────────────────────────┴─────────────────────────────────────────┐
│  Backend Server (FastAPI)                                        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Voice Processor                                          │  │
│  │  ├─ VAD (Silero) - Speech detection                       │  │
│  │  ├─ STT (Whisper base) - Transcription                    │  │
│  │  ├─ LLM (Qwen 2.5 7B) - Response generation               │  │
│  │  └─ TTS (Sesame CSM-1B) - Audio synthesis                 │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Knowledge System (Neo4j)                                 │  │
│  │  ├─ Entity Extraction (LLMGraphTransformer)               │  │
│  │  ├─ Vector Search (Sentence Transformers)                 │  │
│  │  ├─ Graph Traversal (Multi-hop relationships)             │  │
│  │  ├─ Auto-RAG (Document injection)                         │  │
│  │  └─ Autonomous Tool Usage (extract/query)                 │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

## Technology Stack

### Backend
- **LLM**: Qwen 2.5 7B Instruct via Ollama (easily swappable)
- **STT**: OpenAI Whisper (base model, 1.4GB)
- **TTS**: Sesame CSM-1B (high-quality streaming synthesis)
- **VAD**: Silero VAD (voice activity detection)
- **Server**: FastAPI + Uvicorn + WebSockets
- **Database**: Neo4j 5.x (graph database)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2, 384-dim)
- **Graph Processing**: LangChain's LLMGraphTransformer
- **Audio**: sounddevice, numpy, scipy

### Frontend (Flutter Desktop - In Development)
- **Framework**: Flutter 3.24+ (Dart 3.10+)
- **Platform**: Windows Desktop (macOS/Linux ready)
- **State Management**: Riverpod 3.x
- **Audio**: record (recording) + audioplayers (playback)
- **WebSocket**: web_socket_channel
- **UI**: Material Design 3

### Platform Requirements
- **OS**: Windows 11, macOS, or Linux
- **Python**: 3.11+
- **Node.js**: 18+
- **Hardware**: NVIDIA GPU with 12GB+ VRAM (RTX 4070 Super tested), 32GB RAM
- **CUDA**: 12.4+ with cuDNN 9.5

## Quick Start

### 1. Prerequisites

Install the following:
- **Ollama** - Local LLM infrastructure ([ollama.ai](https://ollama.ai))
- **Neo4j** - Graph database ([neo4j.com](https://neo4j.com))
- **Python 3.11+** with CUDA support
- **Node.js 18+** (for frontend)

### 2. Backend Setup

```bash
# Clone repository
git clone https://github.com/gadhiar/mist.ai.git
cd mist.ai

# Install Ollama and pull model
ollama pull qwen2.5:7b-instruct

# Create Python virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install Python dependencies
pip install -r requirements.txt

# Configure .env file
# Edit .env with your Neo4j credentials and settings
```

### 3. Configure Environment

Create/edit `.env` file in the root directory:

```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# Knowledge Integration
ENABLE_KNOWLEDGE_INTEGRATION=true

# Voice Configuration
TTS_ENABLED=true  # Set to false to disable TTS (text-only mode, saves memory)

# Model Selection
MODEL=qwen2.5:7b-instruct
```

### 4. Initialize Knowledge Graph

```bash
# Start Neo4j
neo4j start

# Verify connection
python test_neo4j_connection.py

# Initialize schema, indexes, and constraints
python initialize_schema.py
```

### 5. Start Backend Server

```bash
# Start backend (from root directory)
python backend/server.py
```

Backend will start on `ws://localhost:8001`

### 6. Test with CLI Client

```bash
# In a new terminal
cd cli_client
python voice_client.py
```

**Features:**
- Speak naturally - VAD detects when you start/stop
- Interrupt anytime - AI stops within 50-100ms
- Gap-free audio playback
- Autonomous knowledge extraction and querying

**Try these conversation patterns:**
1. **Learning**: "I use Python and FastAPI for backend development" -> LLM extracts knowledge
2. **Querying**: "What technologies do I use?" -> LLM queries graph for personalized response
3. **Natural chat**: "How are you today?" -> Normal conversation without tools

### 7. Set Up Git Hooks (Recommended)

Install pre-commit hooks for automated code quality checks:

```bash
# Install pre-commit framework
pip install pre-commit

# Install hooks
pre-commit install

# Or use installation script
bash scripts/install-git-hooks.sh     # Linux/macOS
scripts\install-git-hooks.bat          # Windows
```

See [Quick Start Guide](QUICKSTART_GIT_HOOKS.md) or [Full Documentation](docs/GIT_WORKFLOWS.md).

### 8. Flutter Frontend Setup (In Development)

```bash
cd mist_desktop

# Get Flutter dependencies
flutter pub get

# Run on Windows
flutter run -d windows

# Or run on other platforms
flutter devices  # List available devices
flutter run -d <device-id>
```

See [FLUTTER_MIGRATION_PLAN.md](FLUTTER_MIGRATION_PLAN.md) for detailed setup instructions.

## Key Features

### Voice Conversation System
- **Real-time bidirectional dialogue** with WebSocket architecture
- **Natural interruption support** - <100ms response time
- **Gap-free audio playback** - optimized streaming for smooth experience
- **Automatic speech detection** via Voice Activity Detection (VAD)
- **Full pipeline**: Speech -> Transcription -> LLM -> Audio

### Knowledge Graph Integration
- **Autonomous entity extraction** from conversational input
- **Persistent memory** - Neo4j-backed knowledge storage
- **Intelligent retrieval** - hybrid vector search + graph traversal
- **Provenance tracking** - know which utterances created which entities
- **Relationship types**: USES, WORKS_ON, EXPERT_IN, LEARNING, STRUGGLES_WITH, etc.
- **Multi-hop graph expansion** for comprehensive context

### Autonomous Tool Usage (MCP-like Pattern)
- **LLM decides autonomously** when to use tools - no explicit intent classification needed
- **Two primary tools**:
  - `query_knowledge_graph` - Retrieves relevant context from stored knowledge
  - `extract_knowledge` - Stores new information from conversation
- **Smart context awareness** - LLM knows when to extract, query, or just respond
- **Streaming support** - token-by-token response generation visible in real-time
- **Graceful degradation** - works without Neo4j (falls back to standard LLM)

### Auto-RAG (Retrieval-Augmented Generation)
- **Automatic document injection** into LLM context
- **Vector similarity search** on source documents
- **Configurable retrieval** (threshold, limit via `.env`)
- **Future extensibility** - support for user-uploaded documentation

## Project Structure

```
mist.ai/
├── backend/                          # AI voice & knowledge systems
│   ├── server.py                     # WebSocket server (port 8001)
│   ├── voice_processor.py            # Voice pipeline orchestration
│   ├── config.py                     # Voice system configuration
│   ├── knowledge_config.py           # Knowledge graph configuration
│   ├── voice_models/
│   │   └── model_manager.py          # Model lifecycle + knowledge integration
│   ├── chat/
│   │   ├── conversation_handler.py   # MCP-like autonomous tool usage
│   │   └── knowledge_integration.py  # Voice -> Knowledge bridge
│   └── knowledge/                    # Knowledge graph system
│       ├── extraction/               # Entity extraction (LLMGraphTransformer)
│       ├── retrieval/                # Hybrid retrieval (vector + graph)
│       ├── storage/                  # Neo4j persistence layer
│       └── embeddings/               # Vector embeddings (Sentence Transformers)
│
├── mist_desktop/                     # Flutter Desktop UI
│   ├── lib/
│   │   ├── main.dart                # App entry point
│   │   ├── screens/                 # UI screens (chat, settings)
│   │   ├── providers/               # Riverpod state providers
│   │   ├── services/                # WebSocket, audio services
│   │   ├── models/                  # Data models
│   │   ├── widgets/                 # Reusable UI components
│   │   └── config/                  # App configuration
│   ├── pubspec.yaml                 # Flutter dependencies
│   └── windows/                     # Windows platform code
│
├── cli_client/                       # Python test client for voice
│   └── voice_client.py
│
├── dependencies/csm/                 # Sesame CSM TTS (modified fork)
├── docs/                             # Comprehensive documentation
│   ├── guides/                       # Setup & operation guides
│   ├── decisions/                    # Architecture Decision Records
│   └── frontend/                     # Frontend architecture spec
│
├── requirements.txt                  # Python dependencies
├── .env                              # Configuration (create from example)
├── README.md                         # This file
├── QUICKSTART_KNOWLEDGE_INTEGRATION.md # Quick KG setup guide
├── INTEGRATION_STATUS.md             # Knowledge graph integration status
├── E2E_TEST_GUIDE.md                # End-to-end testing guide
├── NEO4J_QUERIES.md                 # Useful Neo4j query reference
└── LICENSE                           # MIT License
```

### Key Scripts
- `initialize_schema.py` - Set up Neo4j schema, indexes, constraints
- `regenerate_graph.py` - Rebuild knowledge graph from stored utterances
- `test_neo4j_connection.py` - Verify Neo4j connectivity
- `test_conversation_handler.py` - Test conversation flow with tools
- `wipe_neo4j.py` - Clean database (use with caution!)
- `export_graph.py` - Export graph structure for analysis

## Performance Characteristics

### Voice Conversation Latencies
- **VAD detection**: <50ms (speech start/stop)
- **STT (Whisper base)**: ~500ms (transcription)
- **LLM response**: 2-5s (token generation, first token ~1s)
- **TTS (CSM-1B)**: 1-3s per sentence (streaming)
- **Total first-token latency**: ~3-8s
- **Interruption response**: <100ms

### Knowledge System Performance
- **Simple conversation** (no tools): ~1s
- **Query scenario** (vector search + graph traversal): ~1.7-2s
- **Learning scenario** (entity extraction): ~3.5-4.5s

### Resource Usage
- **GPU VRAM**: 8-10GB (Qwen 7B + TTS model)
- **System RAM**: 4-6GB active
- **CPU**: Moderate (audio processing threads)

## Testing & Verification

### End-to-End Testing

1. **Start Neo4j**: `neo4j start`
2. **Verify connection**: `python test_neo4j_connection.py`
3. **Start backend**: `python backend/server.py`
4. **Connect voice client**: `python cli_client/voice_client.py`
5. **Test three scenarios**:
   - **Learning**: "I use Python and FastAPI" -> LLM extracts knowledge
   - **Querying**: "What technologies do I use?" -> LLM queries graph
   - **Natural chat**: "How are you?" -> No tools used

### Verify in Neo4j Browser

```cypher
// View all entities
MATCH (e:__Entity__) RETURN e LIMIT 25

// View User's knowledge
MATCH (u:__Entity__ {id: "User"})-[r]->(e) RETURN u, r, e

// Check specific relationships
MATCH (u)-[r:USES]->(t) RETURN u, r, t
```

See [E2E_TEST_GUIDE.md](E2E_TEST_GUIDE.md) for comprehensive testing instructions.

## Documentation

### Core Documentation
- [QUICKSTART_KNOWLEDGE_INTEGRATION.md](QUICKSTART_KNOWLEDGE_INTEGRATION.md) - Quick KG setup
- [INTEGRATION_STATUS.md](INTEGRATION_STATUS.md) - Knowledge graph integration status
- [E2E_TEST_GUIDE.md](E2E_TEST_GUIDE.md) - End-to-end testing workflow
- [NEO4J_QUERIES.md](NEO4J_QUERIES.md) - Useful Neo4j query reference
- [REPOSITORY_STRUCTURE.md](REPOSITORY_STRUCTURE.md) - Detailed project structure

### Technical Guides
- [Flutter Migration Plan](FLUTTER_MIGRATION_PLAN.md) - Comprehensive Flutter desktop implementation guide
- [Windows Dev Setup](docs/guides/windows_dev_setup.md) - Environment setup
- [Torch Compile Fix](docs/guides/TORCH_COMPILE_FIX.md) - PyTorch optimization issues

### Architecture Decision Records
- [ADR 001: Vision](docs/decisions/adr_001_vision.md) - Project vision and philosophy
- [ADR 007: Sesame CSM](docs/decisions/adr_007_sesame_csm.md) - TTS selection rationale

## Project Status

### Current Branch: `feat/entity_extraction`

Latest developments focus on knowledge graph extraction, retrieval, and autonomous tool usage.

### Completed Features

- [x] WebSocket backend server
- [x] Voice processing pipeline (VAD -> STT -> LLM -> TTS)
- [x] Real-time audio streaming
- [x] Interruption support (<100ms latency)
- [x] Gap-free audio playback
- [x] CLI voice client
- [x] **Knowledge graph integration** (Neo4j)
- [x] **Entity extraction** (LLMGraphTransformer)
- [x] **Vector search** (Sentence Transformers)
- [x] **Hybrid retrieval** (vector + graph traversal)
- [x] **Autonomous tool usage** (MCP-like pattern)
- [x] **Auto-RAG** (document injection)
- [x] **Provenance tracking** (utterance -> entity mapping)
- [x] Flutter desktop app scaffolding (Windows/macOS/Linux ready)
- [x] Flutter WebSocket integration with backend
- [x] Flutter voice recording and message display

### In Progress

- [ ] Flutter UI polish and testing
- [ ] Audio playback integration (TTS from backend)
- [ ] Knowledge graph visualization (Flutter)
- [ ] Rich content support (markdown, images, links)

### Roadmap

**Phase 1 (Complete):** Voice conversation system with WebSocket
**Phase 2 (Complete):** Knowledge graph integration with autonomous tool use
**Phase 3 (In Progress):** Flutter desktop UI with real-time updates
**Phase 4 (Planned):** Rich content (markdown, images, links in responses)
**Phase 5 (Planned):** Vision integration (Qwen 2.5 Vision)
**Phase 6 (Planned):** Meta-reasoning layer for explainability
**Phase 7 (Planned):** Strategic cloud delegation (Claude, GPT-4)
**Phase 8 (Planned):** Mobile app (Flutter iOS/Android)

### Long-Term Vision

- **2025**: Working voice system with knowledge foundation  (achieved)
- **2027**: Competent specialist with deep domain knowledge
- **2030**: Genuine intelligent system with 70B+ local model + years of accumulated knowledge

## Why This Matters

This project explores building AI systems from first principles:

1. **Transparency**: See and understand every decision - every tool call, extraction, retrieval
2. **Privacy**: Runs entirely local, air-gapped capable
3. **Control**: You own the data, the logic, the knowledge
4. **Learning**: System designed to accumulate knowledge and grow over time
5. **Research**: Platform for exploring cognitive architectures, not just using existing APIs

**This is NOT a ChatGPT wrapper.** It's a cognitive architecture built from scratch with persistent memory, autonomous reasoning, and continuous learning capabilities.

## Configuration Options

### Voice System (`backend/config.py`)

```python
# TTS settings
tts_enabled: bool = True  # Set to False to disable TTS (text-only mode, saves memory)
tts_temperature: float = 0.8
tts_topk: int = 50
use_voice_context: bool = True

# VAD settings
vad_enabled: bool = True
vad_threshold: float = 0.5
```

### Knowledge Graph (`backend/knowledge/config.py`)

```python
# Auto-RAG configuration
auto_inject_docs: bool = True  # Enable/disable auto-injection
auto_inject_limit: int = 3  # Number of chunks to auto-inject
auto_inject_threshold: float = 0.4  # Similarity threshold (0.0-1.0)

# Extraction settings
extract_node_properties: bool = True
extract_relationship_properties: bool = True
min_extraction_confidence: float = 0.5
```

All settings can be configured via environment variables in `.env`.

## Troubleshooting

### Common Issues

**Neo4j connection fails:**
- Verify Neo4j is running: `neo4j status`
- Check credentials in `.env` match Neo4j database
- Test connection: `python test_neo4j_connection.py`

**CUDA out of memory:**
- Reduce model size: Switch to `qwen2.5:3b-instruct` in `.env`
- Disable TTS: Set `TTS_ENABLED=false` in `.env` to save VRAM

**Knowledge extraction not working:**
- Verify `ENABLE_KNOWLEDGE_INTEGRATION=true` in `.env`
- Check logs for tool call details: Look for `[TOOL_CALL]` prefix
- Test conversation handler: `python test_conversation_handler.py`

**Auto-RAG not finding documents:**
- Verify chunks exist: Run query in Neo4j Browser: `MATCH (c:DocumentChunk) RETURN count(c)`
- Check threshold: Lower `AUTO_INJECT_THRESHOLD` in `.env` (default 0.4)
- Verify vector index: `SHOW INDEXES` in Neo4j Browser

## Contributing

This is primarily a personal research project. Contributions, ideas, and discussions are welcome via GitHub issues.

**Areas of interest:**
- Knowledge graph optimization
- Frontend UI/UX improvements
- Alternative model integrations
- Performance optimizations
- Documentation improvements

## License

MIT License - See [LICENSE](LICENSE) for details

## Acknowledgments

- [Ollama](https://ollama.ai/) - Local LLM infrastructure
- [Sesame CSM](https://huggingface.co/sesame-ai/csm-1b) - High-quality conversational TTS
- [Neo4j](https://neo4j.com/) - Graph database platform
- [LangChain](https://langchain.com/) - LLM framework and graph transformers
- [Anthropic Claude](https://www.anthropic.com/) - Development assistance via Claude Code

---

**Current Milestone:** Knowledge graph integration complete. Flutter desktop app in development.

**Next Milestone:** Complete Flutter audio playback + knowledge graph visualization in Flutter
