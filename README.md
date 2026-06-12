# MIST.AI

A transparent, locally-run cognitive architecture with persistent memory, autonomous reasoning, and real-time voice interaction.

**Status:** Fully functional voice conversation system with knowledge graph integration and autonomous tool usage. Spatial frontend in active development (separate repo, see Frontend section below).

## What Is This?

MIST.AI is a cognitive architecture built from first principles. It combines a local LLM with a persistent knowledge graph, real-time voice I/O, and autonomous tool usage into a single system that runs entirely on consumer hardware.

- **Transparent** -- every decision the AI makes is visible: tool calls, entity extractions, graph retrievals
- **Local-first** -- runs air-gapped on consumer GPUs (tested on RTX 4070 Super, 12GB VRAM)
- **Continuously learning** -- accumulates personal knowledge over time via a Neo4j graph database
- **Research-focused** -- a platform for exploring cognitive architectures, not a wrapper around an API

## Architecture

```
+------------------------------------------------------------------+
|  Tauri Frontend (nested repo: ./mist-frontend/)    |
|  - Tauri 2.x + React 19 + react-three-fiber                      |
|  - Spatial composition (forms, ring, graph, cards)               |
|  - Voice + text input; streaming text + binary audio output      |
|  - WebSocket per ADR-016 (BE-mediated tool calls)                |
|                   + ADR-017 (message contract)                    |
+------------------------------+-----------------------------------+
                               | WebSocket (Port 8001)
+------------------------------+-----------------------------------+
|  Backend Server (FastAPI)                                         |
|  +------------------------------------------------------------+  |
|  |  Voice Pipeline                                             |  |
|  |  VAD (Silero) -> STT (Whisper) -> LLM (Gemma 4 E4B)         |  |
|  |                                -> TTS (Chatterbox Turbo)    |  |
|  +------------------------------------------------------------+  |
|  +------------------------------------------------------------+  |
|  |  Knowledge System (four-layer per ADR-010)                  |  |
|  |  - Event store (SQLite, raw turn evidence)                  |  |
|  |  - Vault (mist-memory/ markdown, canonical history)         |  |
|  |  - Graph (Neo4j, MIST's reasoning substrate)                |  |
|  |  - Sidecar index (sqlite-vec + FTS5, hybrid retrieval)      |  |
|  +------------------------------------------------------------+  |
+------------------------------------------------------------------+
```

## Technology Stack

**Backend (Python 3.11+)**

| Component   | Technology                                  |
|-------------|---------------------------------------------|
| LLM         | Gemma 4 E4B Q5_K_M via llama-server         |
| STT         | OpenAI Whisper                              |
| TTS         | Chatterbox Turbo (zero-shot voice cloning)  |
| VAD         | Silero VAD                                  |
| Server      | FastAPI + Uvicorn + WebSockets              |
| Database    | Neo4j 5.x (knowledge graph)                 |
| Vector idx  | sqlite-vec + FTS5 (vault sidecar)           |
| Embeddings  | Sentence Transformers (all-MiniLM-L6-v2)    |
| Container   | Docker Compose (CUDA 12.4 + Python 3.11)    |

**Frontend (separate git repo nested at `./mist-frontend/`)**

| Component        | Technology                              |
|------------------|-----------------------------------------|
| Shell            | Tauri 2.x (cross-platform desktop)      |
| Framework        | React 19 + TypeScript strict            |
| 3D rendering     | three.js + @react-three/fiber + drei    |
| Build            | Vite                                    |
| Communication    | Native WebSocket via Tauri shell        |

Connect: backend exposes `ws://localhost:8001/ws`; frontend connects with the protocol documented in ADR-016 (backend-mediated tool calls) and ADR-017 (WebSocket message contract).

**Requirements:** Windows 11 / macOS / Linux, NVIDIA GPU with 12GB+ VRAM, CUDA 12.4+, 32GB RAM

## Quick Start

**Backend (this repo):**

```bash
# Clone and configure
git clone https://github.com/gadhiar/mist.ai.git
cd mist.ai
cp .env.example .env         # Edit Neo4j credentials, model paths, TTS toggle

# Start the Docker stack (backend + Neo4j + llama-server)
docker compose up -d

# Or via the dev script
python scripts/start_dev.py

# Verify backend is up
docker compose logs -f mist-backend
```

**Frontend (separate repo):**

The frontend lives at `./mist-frontend/` (separate git repo nested in this one) (Tauri 2.x + React 19 + r3f). See that repo's own README for setup. In short:

```bash
cd mist-frontend
npm install
npm run dev    # Vite dev server on localhost:1420 + Tauri shell
```

The frontend connects to this backend automatically at `ws://localhost:8001/ws` once both are running.

## Key Capabilities

- Real-time voice conversation with natural interruption support (<100ms)
- Persistent knowledge graph -- entities, relationships, and provenance tracked in Neo4j
- Vault layer (ADR-010) -- canonical markdown corpus with user-editable history
- Autonomous tool usage -- the LLM decides when to query the graph
- Hybrid retrieval (graph + vector + RRF merge) with intent-driven routing
- Gap-free audio streaming with voice activity detection
- Full pipeline: Speech -> Transcription -> LLM -> Audio synthesis
- Spatial frontend with WebGL-driven 60fps composition (separate Tauri repo)

## Project Status

**Completed**

- WebSocket backend server with binary audio protocol
- Voice pipeline (VAD -> STT -> LLM -> TTS) with streaming parallelism (~4-5s TTFA)
- Knowledge graph integration (Neo4j) with ADR-009 provenance separation
- Entity extraction + curation pipeline (ontology v1.0.0+)
- Hybrid retrieval (vector + graph + RRF) with intent classifier
- Vault layer (ADR-010) with sidecar index + filewatcher
- MIST.md auto-load primitive (ADR-014)
- WebSocket FE/BE protocol contract (ADR-016 + ADR-017)
- CLI voice client
- Spatial Tauri frontend (separate repo, production-ready 2026-05-08)

**In Progress**

- FE/BE integration: Wave 1 shipped 2026-05-10 (handshake, heartbeat, state_cycle, turn-streaming, error discrimination, vad_status, log streaming, health_status); subsequent waves cover tool-call events, cards, graph_subgraph, visual polish
- Cognitive substrate (ADR-012 v2 in design)
- Conversation summary layer (ADR-018 candidate)

**Planned**

- Vision integration (multimodal Gemma)
- Hook taxonomy + dispatcher (ADR-019 candidate)
- Settings hierarchy and spec/config split (ADR-020, proposed)

## Project Structure

See [REPOSITORY_STRUCTURE.md](REPOSITORY_STRUCTURE.md) for the full tree. Key directories:

```
mist.ai/
  backend/              # Python -- FastAPI server, voice pipeline, knowledge system
  mist-memory/          # ADR-010 vault layer (sessions, identity, users, decisions)
  data/                 # Runtime artifacts (event store, sidecar SQLite, snapshots)
  dependencies/csm/     # Legacy Sesame CSM TTS fork (Apache 2.0); Chatterbox Turbo is current
  docs/                 # Guides, setup, architecture decisions
```

The frontend repo nested at `./mist-frontend/` has its own structure (Vite + Tauri + React).

## Documentation

- [Repository Structure](REPOSITORY_STRUCTURE.md)
- [Contributing](CONTRIBUTING.md)
- [Codebase Status](CODEBASE.md)
- [Testing Guide](TESTING.md)
- [Known Issues](KNOWN_ISSUES.md)
- ADRs live under `docs/decisions/` (project-scoped) and `knowledge-vault/Decisions/` (cross-project + integration ADRs)

## License

MIT -- see [LICENSE](LICENSE).

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) -- local LLM inference
- [Chatterbox](https://huggingface.co/ResembleAI/chatterbox) -- conversational TTS
- [Neo4j](https://neo4j.com/) -- graph database
- [LangChain](https://langchain.com/) -- ontology-constrained extraction
- [sqlite-vec](https://github.com/asg017/sqlite-vec) -- vector indexing
