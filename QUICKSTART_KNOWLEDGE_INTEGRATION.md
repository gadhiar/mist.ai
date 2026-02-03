# Quick Start: Knowledge Integration

Get your knowledge-augmented conversation system running in 5 minutes.

---

## Prerequisites

✅ Neo4j running on `bolt://localhost:7687`
✅ Python environment with dependencies installed
✅ Existing MIST voice system working

---

## Step 1: Test the System (2 minutes)

```bash
# Test imports
venv/Scripts/python.exe -c "from backend.chat import ConversationHandler; print('✅ Ready')"

# Test simple conversation (requires Neo4j)
venv/Scripts/python.exe test_conversation_handler.py --mode simple
```

**Expected output:**
```
USER: I use Python and FastAPI for backend development.
MIST: [Autonomous response with tool use]
✅ Test passed!
```

---

## Step 2: Configure Neo4j (1 minute)

Create `.env` file in project root:

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
ENABLE_KNOWLEDGE_INTEGRATION=true
```

---

## Step 3: Integrate into Voice System (2 minutes)

Edit `backend/voice_models/model_manager.py`:

**Add imports at top:**
```python
from backend.chat.knowledge_integration import KnowledgeIntegration
from backend.knowledge_config import DEFAULT_KNOWLEDGE_CONFIG
```

**In `ModelManager.__init__()`, add:**
```python
# Add after self.llm_model initialization
self.knowledge = None
if DEFAULT_KNOWLEDGE_CONFIG.enable_knowledge_integration:
    try:
        self.knowledge = KnowledgeIntegration(
            neo4j_uri=DEFAULT_KNOWLEDGE_CONFIG.neo4j_uri,
            neo4j_user=DEFAULT_KNOWLEDGE_CONFIG.neo4j_user,
            neo4j_password=DEFAULT_KNOWLEDGE_CONFIG.neo4j_password,
            model_name=DEFAULT_KNOWLEDGE_CONFIG.knowledge_model
        )
        logger.info("✅ Knowledge integration enabled")
    except Exception as e:
        logger.warning(f"⚠️  Knowledge integration disabled: {e}")
```

**In `generate_llm_response()`, replace the ollama.chat call:**
```python
def generate_llm_response(self, user_text):
    """Generate LLM response with optional knowledge integration"""

    # Use knowledge-augmented if available
    if self.knowledge and self.knowledge.is_enabled():
        for token in self.knowledge.generate_response_streaming(user_text):
            yield token
    else:
        # Existing fallback code
        system_prompt = """You are M.I.S.T, a helpful voice assistant..."""
        response = ollama.chat(...)
        for chunk in response:
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]
```

---

## Step 4: Test Integration

```bash
# Start your voice server
venv/Scripts/python.exe backend/server.py

# Connect frontend and test:
# - Say: "I use Python and Neo4j"
# - Wait for response (should extract knowledge)
# - Say: "What technologies do I use?"
# - Wait for response (should query knowledge graph)
```

---

## Done! 🎉

Your voice assistant now has:
- ✅ Autonomous knowledge extraction
- ✅ Contextual knowledge retrieval
- ✅ Personalized responses based on accumulated knowledge

---

## Verify It's Working

**Check logs for:**
```
✅ Knowledge integration enabled
LLM made 1 tool calls
Executing tool: extract_knowledge with args: {...}
```

**Check Neo4j Browser:**
```cypher
// View extracted entities
MATCH (e:__Entity__) RETURN e LIMIT 25
```

---

## Troubleshooting

**"Knowledge integration disabled"**
→ Check Neo4j is running: `neo4j status`

**"No information found for query"**
→ Graph is empty, say something to learn first

**Slow responses**
→ Reduce `max_retrieval_facts=10` in `knowledge_config.py`

---

## Full Documentation

- [Integration Guide](docs/KNOWLEDGE_INTEGRATION_GUIDE.md) - Complete setup instructions
- [Task 4 Summary](TASK_4_COMPLETION_SUMMARY.md) - Architecture and design decisions
- [E2E Plan](docs/E2E_IMPLEMENTATION_PLAN.md) - Overall project roadmap

---

## What's Next?

1. **Test with diverse conversations** - Verify extraction and retrieval quality
2. **Monitor performance** - Check response times in production
3. **Tune configuration** - Adjust limits and thresholds as needed
4. **Build frontend features** - Visualize knowledge graph, show sources

Enjoy your knowledge-augmented voice assistant! 🚀
