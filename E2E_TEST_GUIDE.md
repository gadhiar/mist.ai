# End-to-End Testing Guide

## Quick E2E Test

Test the full system from Neo4j → Backend → Frontend → Voice

---

## Prerequisites

- ✅ Neo4j installed and running
- ✅ Python environment activated
- ✅ Frontend setup complete
- ✅ `.env` file configured

---

## Step 1: Start Neo4j (30 seconds)

```bash
# Start Neo4j
neo4j start

# Verify it's running
neo4j status
```

Expected output:
```
Neo4j is running at pid 12345
```

**Or open Neo4j Desktop and start your database**

---

## Step 2: Verify Neo4j Connection (15 seconds)

```bash
venv/Scripts/python.exe test_neo4j_connection.py
```

Expected output:
```
Testing Neo4j connection...
✅ Connection successful!
```

If this fails, check:
- Neo4j is actually running
- Password in `.env` matches Neo4j password
- Port 7687 is not blocked

---

## Step 3: Start Backend (1 minute)

```bash
venv/Scripts/python.exe backend/server.py
```

**Look for these log messages:**

```
Loading all models...
✅ Knowledge graph integration ENABLED
Server ready on ws://0.0.0.0:8765/ws
```

**If you see:**
```
⚠️  Knowledge integration disabled (Neo4j unavailable)
```
→ Go back to Step 1, Neo4j isn't accessible

---

## Step 4: Start Frontend

```bash
# Navigate to your frontend directory and start it
# (exact command depends on your frontend setup)
```

---

## Step 5: E2E Voice Test

### Test Scenario 1: Learning (Knowledge Extraction)

**Say:** "I use Python and FastAPI for backend development"

**Expected behavior:**
1. Speech recognized
2. LLM responds naturally
3. **Backend logs show:**
   ```
   Using knowledge-augmented LLM response
   LLM made 1 tool calls
   Executing tool: extract_knowledge
   ```

### Test Scenario 2: Querying (Knowledge Retrieval)

**Say:** "What backend technologies do I use?"

**Expected behavior:**
1. Speech recognized
2. LLM responds with personalized answer mentioning Python/FastAPI
3. **Backend logs show:**
   ```
   Using knowledge-augmented LLM response
   LLM made 1 tool calls
   Executing tool: query_knowledge_graph
   ```

### Test Scenario 3: Normal Conversation

**Say:** "How are you today?"

**Expected behavior:**
1. Speech recognized
2. Normal conversational response
3. **Backend logs show:**
   ```
   Using knowledge-augmented LLM response
   ```
   (No tool calls - just normal conversation)

---

## Step 6: Verify in Neo4j Browser

Open http://localhost:7474

Login with credentials from `.env`:
- Username: `neo4j`
- Password: `password`

Run this query:
```cypher
// View all extracted entities
MATCH (e:__Entity__)
RETURN e.id AS entity, labels(e) AS types
LIMIT 25
```

**Expected results:**
- User
- Python
- FastAPI
- (any other entities mentioned)

Run this query:
```cypher
// View User's relationships
MATCH (u:__Entity__ {id: "User"})-[r]->(e)
RETURN u.id AS user, type(r) AS relationship, e.id AS entity
```

**Expected results:**
```
User | USES | Python
User | USES | FastAPI
```

---

## Troubleshooting

### Backend shows "Knowledge integration disabled"

**Problem:** Neo4j connection failed

**Solutions:**
1. Check Neo4j is running: `neo4j status`
2. Verify credentials in `.env` match Neo4j
3. Test connection: `venv/Scripts/python.exe test_neo4j_connection.py`

### No entities in Neo4j after speaking

**Problem:** Extraction not triggering

**Check backend logs for:**
```
Executing tool: extract_knowledge
```

If missing:
- LLM may not be recognizing it as information to store
- Try more explicit: "Remember that I use Python"

### LLM not querying knowledge graph

**Problem:** Query tool not triggering

**Check backend logs for:**
```
Executing tool: query_knowledge_graph
```

If missing:
- LLM may not recognize it as a query about stored knowledge
- Try more explicit: "What do you know about my technologies?"

### Slow responses (>10 seconds)

**Normal latencies:**
- Simple chat: ~1s
- Query: ~2s
- Extraction: ~4s

**If slower:**
- Check Neo4j isn't overloaded
- Reduce `max_retrieval_facts` in `backend/knowledge_config.py`

---

## Alternative: Manual Test (No Frontend)

If you want to test without the frontend:

```bash
# Test conversation handler directly
venv/Scripts/python.exe test_conversation_handler.py --mode simple
```

This simulates the conversation flow without voice/frontend.

---

## Success Criteria

✅ Backend starts with "Knowledge integration ENABLED"
✅ Voice input → LLM response works
✅ Backend logs show tool calls (extract_knowledge, query_knowledge_graph)
✅ Neo4j Browser shows extracted entities and relationships
✅ LLM gives personalized responses based on stored knowledge

---

## Quick Verification Checklist

- [ ] Neo4j running
- [ ] Backend shows "Knowledge integration ENABLED"
- [ ] Frontend connects to backend WebSocket
- [ ] Can speak and get responses
- [ ] Entities appear in Neo4j Browser
- [ ] LLM uses stored knowledge in responses

---

## Next Steps After E2E Test

1. **Test diverse conversations** - Different topics, relationships, entities
2. **Monitor graph growth** - Watch entities accumulate in Neo4j
3. **Tune performance** - Adjust config if needed
4. **Build frontend features** - Visualize graph, show sources, etc.

---

## Summary

**Full E2E flow:**
```
You speak
  ↓
Frontend (microphone)
  ↓
WebSocket → Backend
  ↓
VoiceProcessor → ModelManager
  ↓
KnowledgeIntegration → ConversationHandler
  ↓
LLM with Tools (autonomous decision)
  ↓
query_knowledge_graph OR extract_knowledge
  ↓
Neo4j (read/write)
  ↓
Response with context
  ↓
TTS → Audio
  ↓
You hear response
```

**Just follow Steps 1-5 and you're testing E2E!**
