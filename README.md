# MIST.AI

A transparent, locally-run cognitive architecture with persistent memory, autonomous reasoning, and real-time voice interaction.

**Status:** Fully functional voice conversation system with knowledge graph integration and autonomous tool usage. Desktop UI in development.

## What Is This?

MIST.AI is a cognitive architecture built from first principles. It combines a local LLM with a persistent knowledge graph, real-time voice I/O, and autonomous tool usage into a single system that runs entirely on consumer hardware.

- **Transparent** -- every decision the AI makes is visible: tool calls, entity extractions, graph retrievals
- **Local-first** -- runs air-gapped on consumer GPUs (tested on RTX 4070 Super, 12GB VRAM)
- **Continuously learning** -- accumulates personal knowledge over time via a Neo4j graph database
- **Research-focused** -- a platform for exploring cognitive architectures, not a wrapper around an API

## Architecture

```
+------------------------------------------------------------------+
|  Flutter Desktop (mist_desktop/)                                  |
|  - Windows/macOS/Linux support                                    |
|  - Voice recording and playback                                   |
|  - Real-time conversation UI                                      |
|  - WebSocket communication                                        |
+------------------------------+-----------------------------------+
                               | WebSocket (Port 8001)
+------------------------------+-----------------------------------+
|  Backend Server (FastAPI)                                         |
|  +------------------------------------------------------------+  |
|  |  Voice Pipeline                                             |  |
|  |  VAD (Silero) -> STT (Whisper) -> LLM (Qwen 2.5 7B) -> TTS (Sesame CSM-1B)  |
|  +------------------------------------------------------------+  |
|  +------------------------------------------------------------+  |
|  |  Knowledge System (Neo4j)                                   |  |
|  |  - Entity extraction (LLMGraphTransformer)                  |  |
|  |  - Vector search (Sentence Transformers, 384-dim)           |  |
|  |  - Multi-hop graph traversal                                |  |
|  |  - Auto-RAG document injection                              |  |
|  |  - Autonomous tool usage (extract / query)                  |  |
|  +------------------------------------------------------------+  |
+------------------------------------------------------------------+
```

## Technology Stack

**Backend (Python 3.11+)**

| Component   | Technology                          |
|-------------|-------------------------------------|
| LLM         | Qwen 2.5 7B Instruct via Ollama    |
| STT         | OpenAI Whisper (base, 1.4GB)        |
| TTS         | Sesame CSM-1B (streaming synthesis) |
| VAD         | Silero VAD                          |
| Server      | FastAPI + Uvicorn + WebSockets      |
| Database    | Neo4j 5.x                          |
| Embeddings  | Sentence Transformers (all-MiniLM-L6-v2) |
| Graph       | LangChain LLMGraphTransformer       |

**Frontend (Flutter 3.24+ / Dart 3.10+)**

| Component        | Technology               |
|------------------|--------------------------|
| State management | Riverpod 3.x             |
| Audio            | record + audioplayers     |
| Communication    | web_socket_channel        |
| UI               | Material Design 3         |

**Requirements:** Windows 11 / macOS / Linux, NVIDIA GPU with 12GB+ VRAM, CUDA 12.4+, 32GB RAM

## Quick Start

```bash
# Clone and set up
git clone https://github.com/gadhiar/mist.ai.git
cd mist.ai

# Pull the LLM
ollama pull qwen2.5:7b-instruct

# Python environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt

# Configure .env (Neo4j credentials, model selection, TTS toggle)
cp .env.example .env         # Edit with your settings

# Initialize Neo4j
neo4j start
python initialize_schema.py

# Start backend
python backend/server.py     # Listens on ws://localhost:8001

# Test with CLI client (new terminal)
python cli_client/voice_client.py
```

**Flutter frontend** (in development):
```bash
cd mist_desktop && flutter pub get && flutter run -d windows
```

## Key Capabilities

- Real-time voice conversation with natural interruption support (<100ms)
- Persistent knowledge graph -- entities, relationships, and provenance tracked in Neo4j
- Autonomous tool usage -- the LLM decides when to extract knowledge or query the graph
- Auto-RAG -- automatic document injection for context-aware responses
- Gap-free audio streaming with voice activity detection
- Full pipeline: Speech -> Transcription -> LLM -> Audio synthesis

## Project Status

**Completed**

- WebSocket backend server
- Voice pipeline (VAD -> STT -> LLM -> TTS)
- Real-time audio streaming with interruption support
- Knowledge graph integration (Neo4j)
- Entity extraction (LLMGraphTransformer)
- Hybrid retrieval (vector search + graph traversal)
- Autonomous tool usage (MCP-like pattern)
- Auto-RAG document injection
- Provenance tracking (utterance -> entity mapping)
- Flutter desktop app scaffolding with WebSocket integration
- CLI voice client

**In Progress**

- Flutter UI polish and audio playback integration
- Knowledge graph visualization in Flutter
- Rich content support (markdown, images, links)

**Planned**

- Vision integration (Qwen 2.5 Vision)
- Meta-reasoning layer for explainability
- Strategic cloud delegation
- Mobile app (Flutter iOS/Android)

## Project Structure

See [REPOSITORY_STRUCTURE.md](REPOSITORY_STRUCTURE.md) for the full tree. Key directories:

```
mist.ai/
  backend/              # Python -- FastAPI server, voice pipeline, knowledge system
  mist_desktop/         # Flutter -- cross-platform desktop UI
  cli_client/           # Python -- CLI voice test client
  dependencies/csm/     # Modified Sesame CSM TTS fork (Apache 2.0)
  docs/                 # Guides, setup, architecture decisions
```

## Documentation

- [Repository Structure](REPOSITORY_STRUCTURE.md)
- [Contributing](CONTRIBUTING.md)
- [Knowledge Integration Quickstart](docs/QUICKSTART_KNOWLEDGE_INTEGRATION.md)
- [Neo4j Query Reference](docs/NEO4J_QUERIES.md)
- [Windows Dev Setup](docs/guides/windows_dev_setup.md)

## License

MIT -- see [LICENSE](LICENSE).

## Acknowledgments

- [Ollama](https://ollama.ai/) -- local LLM infrastructure
- [Sesame CSM](https://huggingface.co/sesame-ai/csm-1b) -- conversational TTS
- [Neo4j](https://neo4j.com/) -- graph database
- [LangChain](https://langchain.com/) -- LLM framework and graph transformers
