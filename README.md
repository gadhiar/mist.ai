# Mist.AI

A self-aware hybrid cognitive architecture combining transparent local reasoning with strategic cloud delegation.

**Current Status:** Voice conversation system with real-time WebSocket architecture. Frontend UI in development.

## What Is This?

Mist.AI is a research platform for building true AI through a unique multi-layer architecture. The long-term vision includes knowledge graphs, transparent reasoning, and strategic delegation. **Currently implemented:** Real-time voice conversation system with natural interruption support.

### Long-Term Vision

- **Transparent reasoning** via knowledge graph - see every decision
- **Natural learning** through multimodal conversation (speech, vision, text)
- **Strategic delegation** to cloud models when needed
- **Continuous growth** that becomes more capable over time

This isn't a ChatGPT wrapper. It's a cognitive architecture being built from first principles.

## Current Implementation (v0.1)

### Working Features

- **Real-time voice conversation** - Natural dialogue with near-instant interruption
- **WebSocket architecture** - Backend (Python) ↔ Frontend (React, in progress)
- **Gap-free audio** - Smooth TTS playback with no stuttering
- **Voice Activity Detection** - Automatic speech detection and interruption handling
- **Local-first** - Runs entirely on your hardware (air-gapped capable)

### Architecture

```
┌──────────────────────────────────────────────────┐
│  Frontend (React + Redux)                        │
│  - Conversation display                          │
│  - Audio visualization                           │
│  - Connection status                             │
└────────────────┬─────────────────────────────────┘
                 │ WebSocket
┌────────────────┴─────────────────────────────────┐
│  Backend Server (Python, port 8001)              │
│  ┌──────────────────────────────────────────────┐│
│  │  Voice Processor                             ││
│  │  - VAD (Silero)                              ││
│  │  - STT (Whisper base)                        ││
│  │  - LLM (Qwen 2.5 32B via Ollama)             ││
│  │  - TTS (Sesame CSM-1B)                       ││
│  └──────────────────────────────────────────────┘│
└──────────────────────────────────────────────────┘
```

### Technology Stack

**Backend:**
- **LLM**: Qwen 2.5 32B Instruct via Ollama
- **STT**: OpenAI Whisper (base model)
- **TTS**: Sesame CSM-1B (high-quality conversational voice)
- **VAD**: Silero VAD
- **Server**: FastAPI + WebSockets
- **Audio**: sounddevice, numpy, scipy

**Frontend (in development):**
- **Framework**: React 18 + TypeScript
- **Build**: Vite
- **State**: Redux Toolkit
- **Styling**: Tailwind CSS
- **WebSocket**: Native WebSocket API

**Platform:**
- **OS**: Windows 11
- **Python**: 3.11+
- **Hardware**: RTX 4070 Super (12GB VRAM), 32GB RAM
- **CUDA**: 12.9 + cuDNN 9.5

## Quick Start

### Prerequisites

1. **Hardware**: NVIDIA GPU with 12GB+ VRAM, 32GB+ RAM
2. **CUDA**: 12.x installed
3. **Ollama**: Installed and running
4. **Python**: 3.11 or higher
5. **Node.js**: 18+ (for frontend)

### Backend Setup

```bash
# Clone repository
git clone https://github.com/yourusername/mist.ai.git
cd mist.ai

# Install Ollama and pull model
ollama pull qwen2.5:32b-instruct

# Create Python virtual environment
python -m venv venv
venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt

# Start backend server
cd backend
python server.py
```

Backend will start on `ws://localhost:8001`

### CLI Client (Voice Conversation)

```bash
# In a new terminal
cd cli_client
python voice_client.py
```

**Features:**
- Speak naturally - VAD detects when you start/stop
- Interrupt anytime - AI stops within 50-100ms
- Gap-free audio playback

### Frontend Setup (In Development)

```bash
cd frontend

# Install dependencies
npm install
npm install @reduxjs/toolkit react-redux framer-motion

# Start dev server
npm run dev
```

Frontend will start on `http://localhost:5173`

**Note:** Frontend UI is not yet functional - components are being built.

## Project Status

### Completed ✓

- [x] WebSocket backend server
- [x] Voice processing pipeline (VAD → STT → LLM → TTS)
- [x] Real-time audio streaming
- [x] Interruption support (<100ms latency)
- [x] Gap-free audio playback
- [x] CLI voice client
- [x] Frontend scaffolding (React + Redux + TypeScript)
- [x] Redux store architecture (conversation, audio, connection slices)
- [x] WebSocket middleware for state sync

### In Progress 🚧

- [ ] Frontend UI components
- [ ] Web Audio API integration for browser playback
- [ ] Connection status indicators
- [ ] Conversation history display

### Planned 📋

- [ ] Knowledge graph foundation
- [ ] Teaching interface
- [ ] Vision integration (Qwen 2.5 Vision)
- [ ] Meta-reasoning layer
- [ ] Strategic cloud delegation (Claude, GPT-4)
- [ ] Mobile app (React Native)

## Documentation

### Core Documentation
- [Frontend Architecture](docs/frontend/FRONTEND_ARCHITECTURE.md) - Complete frontend technical spec
- [Project Progress](PROJECT_PROGRESS.md) - Development timeline and status

### Technical Guides
- [Windows Dev Setup](docs/guides/windows_dev_setup.md) - Complete environment setup
- [Torch Compile Fix](docs/guides/TORCH_COMPILE_FIX.md) - PyTorch optimization issues

### Frontend Documentation
- [Frontend Setup](frontend/SETUP.md) - Frontend installation guide
- [Redux Implementation](frontend/REDUX_IMPLEMENTATION.md) - State management architecture
- [Redux Store Guide](frontend/src/stores/README.md) - Using the Redux store

### Architecture Decision Records
- [ADR 001: Vision](docs/decisions/adr_001_vision.md) - Project vision and philosophy
- [ADR 007: Sesame CSM](docs/decisions/adr_007_sesame_csm.md) - TTS selection rationale

## Development Notes

### Current Branch: `feat/web_socket`

This branch implements the WebSocket architecture and frontend foundation.

### Performance Characteristics

**Voice Conversation:**
- VAD detection: <50ms
- STT (Whisper base): ~500ms
- LLM (Qwen 32B): 2-5s
- TTS (CSM-1B): 1-3s per sentence (streaming)
- Total first-token latency: ~3-8s
- Interruption response: <100ms

**Resource Usage:**
- GPU VRAM: ~8-10GB (Qwen 32B + TTS)
- System RAM: ~4-6GB
- CPU: Moderate (audio processing)

### Known Issues

- [ ] Frontend components not yet implemented
- [ ] No conversation history persistence
- [ ] No error recovery UI
- [ ] Documentation needs updating for knowledge graph plans

## Why This Matters

This project explores building AI systems from first principles:

1. **Transparency**: See and understand every decision
2. **Privacy**: Runs entirely local, air-gapped capable
3. **Control**: You own the data and logic
4. **Learning**: System designed to grow over time

**Long-term Vision:**
- **2025**: Working voice system with knowledge foundation
- **2027**: Competent specialist with deep domain knowledge
- **2030**: Genuine intelligent system with 70B+ local model + years of accumulated knowledge

## Contributing

This is primarily a personal research project. Contributions, ideas, and discussions welcome via GitHub issues.

## License

MIT License - See LICENSE for details

## Acknowledgments

- [Ollama](https://ollama.ai/) - Local LLM infrastructure
- [Sesame CSM](https://huggingface.co/sesame-ai/csm-1b) - High-quality TTS
- [Anthropic Claude](https://www.anthropic.com/) - Development assistance via Claude Code

---

**Next Milestone:** Frontend UI components + conversation history persistence
