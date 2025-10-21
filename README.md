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

- **Hybrid Intelligence**: Local Llama 3.2 for transparent reasoning, cloud APIs (Claude/GPT) for complex tasks
- **Knowledge Graph**: Structured memory with full provenance tracking
- **Multimodal Learning**: Teach through speech, vision, and text
- **Model Agnostic**: Swap base models as they improve - knowledge persists
- **Self-Aware**: Knows its limits and when to delegate
- **Privacy-First**: All data local by default, user controls cloud sharing

## Technology Stack

- **Base Model**: Llama 3.2 8B (Ollama) - swappable
- **Knowledge**: NetworkX (graph) + ChromaDB (vectors)
- **Speech**: Whisper (STT) + Piper (TTS)
- **Vision**: Llama 3.2 Vision
- **Cloud**: Claude Sonnet 4.5, GPT-4 (strategic use)
- **Language**: Python 3.11+
- **Platform**: Windows-native (WSL optional)

## Quick Start

See [docs/windows_dev_setup.md](docs/windows_dev_setup.md) for complete setup instructions.

```bash
# Clone the repository
git clone https://github.com/yourusername/mist.ai.git
cd mist.ai

# Install Ollama and pull the model
ollama pull llama3.2:8b

# Set up Python environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run the system
python -m src.main
```

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

- [System Architecture](docs/system_architecture.md) - Complete technical design
- [ADR 001](docs/adr_001.md) - Project vision and decisions
- [ADR 006](docs/adr_006_multimodal.md) - Multimodal learning approach
- [Setup Instructions](docs/setup_instructions.md) - Detailed environment setup
- [Migration Guide](docs/migration_guide.md) - Moving to Claude Code
- [Multimodal Quickstart](docs/multimodal_quickstart.md) - Speech/vision guide

## Cost Structure

- **Development**: ~$40-60 over 10 weeks (Claude Pro for development support)
- **Year 1**: ~$90-120 (higher during learning phase)
- **Long-term**: Decreasing as local capability grows
- **Hardware**: RTX 3070 or better (8GB+ VRAM)

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
