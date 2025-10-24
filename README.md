# Mist.ai

A self-aware hybrid cognitive architecture that combines transparent local reasoning with strategic cloud delegation.

## What Is This?

Mist.ai is a research platform for building true AI through a unique three-layer architecture:

- **Transparent reasoning** via knowledge graph - see every decision
- **Natural learning** through multimodal conversation (speech, vision, text)
- **Strategic delegation** to cloud models when needed
- **Continuous growth** that becomes more capable over time

This isn't a ChatGPT wrapper or LLM fine-tuning project. It's a cognitive architecture that learns, reasons transparently, and strategically augments itself.

## Core Architecture

```
┌─────────────────────────────────────┐
│      Meta-Reasoning Layer           │
│   (Decides HOW to answer)           │
└──────────┬──────────────────────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐  ┌──────────┐
│  Local  │  │  Cloud   │
│  Graph  │  │ Delegate │
│Reasoning│  │          │
└────┬────┘  └────┬─────┘
     │            │
     └─────┬──────┘
           ▼
┌──────────────────────────┐
│   Knowledge Graph        │
│   + Vector Store         │
└──────────────────────────┘
```

### Key Features

- **Hybrid Intelligence**: Local Qwen 2.5 32B for transparent reasoning, cloud APIs (Claude/GPT) for complex tasks
- **Knowledge Graph**: Structured memory with full provenance tracking
- **Multimodal Learning**: Teach through speech, vision, and text
- **Model Agnostic**: Swap base models as they improve - knowledge persists
- **Self-Aware**: Knows its limits and when to delegate
- **Privacy-First**: All data local by default, user controls cloud sharing, fully air-gapped capable

## Technology Stack

- **Base Model**: Qwen 2.5 32B Instruct (Ollama) - swappable
- **Knowledge**: NetworkX (graph) + ChromaDB (vectors)
- **Speech**: Whisper base (STT) + Sesame CSM-1B (TTS)
- **Vision**: Future - Qwen 2.5 Vision (planned)
- **Cloud**: Claude Sonnet 4.5, GPT-4 (strategic use)
- **Language**: Python 3.11+
- **Platform**: Windows-native
- **Hardware**: RTX 4070 Super (12GB VRAM), 32GB RAM (64GB recommended)

## Quick Start

**Voice System:** See [VOICE_SETUP.md](VOICE_SETUP.md) for voice conversation setup (includes PyTorch with CUDA).

**General Setup:** See [docs/windows_dev_setup.md](docs/windows_dev_setup.md) for complete development environment setup.

```bash
# Clone the repository
git clone https://github.com/yourusername/mist.ai.git
cd mist.ai

# Install Ollama and pull the model
ollama pull qwen2.5:32b-instruct

# Set up Python environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run the system
python -m src.main

# Or run voice conversation (optimized, gap-free audio)
python run_continuous_voice.py        # Turn-based conversation
python run_interruptible_voice.py     # Natural flow - interrupt AI anytime
```

### Voice Conversation Modes

**Turn-Based Mode** (`run_continuous_voice.py`):
- Simpler, lower resource usage
- Wait for AI to finish before speaking
- 8-9s response time, gap-free audio
- Good for Q&A interactions

**Interruptible Mode** (`run_interruptible_voice.py`):
- Natural conversation flow
- Interrupt AI anytime by speaking
- AI stops within 50-100ms
- Best for dialogue and discussion

See [VOICE_IMPLEMENTATION_FINAL.md](VOICE_IMPLEMENTATION_FINAL.md) and [INTERRUPTION_FEATURE.md](INTERRUPTION_FEATURE.md) for details.

## Development Roadmap

- **Phase 0** (Weeks 1-2): Knowledge graph foundation
- **Phase 1** (Weeks 3-4): Speech integration
- **Phase 2** (Weeks 5-6): Vision integration
- **Phase 3** (Weeks 7-8): Reasoning & meta-cognition
- **Phase 4** (Weeks 9-10): Strategic cloud delegation
- **Phase 5+**: Advanced capabilities & emergent behaviors

See [docs/phase_0_detailed.md](docs/phase_0_detailed.md) for detailed implementation plans.

## Why This Matters

**2025**: Working multimodal system that learns from conversation
**2027**: Competent specialist with deep domain knowledge
**2030**: With 70B+ local model + 5 years knowledge = genuine intelligent system

This is actual AI architecture that grows more capable over time while remaining completely transparent and under your control.

## Documentation

- [Project Handover](docs/PROJECT_HANDOVER.md) - Complete project context and status
- [System Architecture](docs/design/system_architecture.md) - Complete technical design
- [Scaling Architecture](docs/design/SCALING_ARCHITECTURE.md) - Performance and scaling strategy
- [ADR 001: Vision](docs/decisions/adr_001_vision.md) - Project vision and decisions
- [ADR 006: Multimodal](docs/decisions/adr_006_multimodal.md) - Multimodal learning approach
- [ADR 007: Sesame CSM](docs/decisions/adr_007_sesame_csm.md) - TTS selection
- [Windows Setup Guide](docs/guides/windows_dev_setup.md) - Complete environment setup
- [Multimodal Quickstart](docs/guides/multimodal_quickstart.md) - Speech/vision guide
- [Migration Guide](docs/guides/migration_guide.md) - Moving to Claude Code

## Cost Structure

- **Development**: ~$40-60 over 10 weeks (Claude Pro for development support)
- **Year 1**: ~$90-120 (higher during learning phase)
- **Long-term**: Decreasing as local capability grows
- **Hardware**: RTX 4070 Super or better (12GB+ VRAM), 32GB+ RAM

## Research Goals

1. **Understand AI** by building it from first principles
2. **Explore transparency** vs power trade-offs in reasoning
3. **Investigate meta-learning** and self-improvement
4. **Study knowledge accumulation** over extended periods
5. **Demonstrate that** local + strategic cloud = practical AI

## License

MIT License - See [LICENSE](LICENSE) for details

## Contributing

This is primarily a personal research project, but contributions, ideas, and discussions are welcome. Open an issue to start a conversation.

## Acknowledgments

Built with support from:
- [Ollama](https://ollama.ai/) - Local LLM infrastructure
- [Anthropic Claude](https://www.anthropic.com/) - Strategic reasoning augmentation
- Claude Code - Development assistance

---

**Status**: Foundation phase - actively under development
**Next Milestone**: Phase 0 complete (knowledge graph + teaching interface)

*This is actual AI architecture. This is the real thing.*
