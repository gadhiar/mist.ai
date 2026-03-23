# Phase 2B: ConversationHandler Migration + Legacy Cleanup

**Date:** 2026-03-22
**Status:** Ready for Review
**Scope:** ConversationHandler DI refactor, automatic extraction, GraphRegenerator migration, legacy code removal
**Prerequisite:** Phase 2A complete (curation pipeline, DI refactoring, 139 tests passing)
**Not in scope:** Periodic curation / background jobs (Phase 4), internal knowledge derivation (Phase 3)

---

## 1. Problem Statement

The ConversationHandler still uses the legacy EntityExtractor for knowledge extraction, invoked as an LLM tool (extract_knowledge). This has several problems:

1. **Unreliable extraction:** The LLM decides when to extract, often skipping useful information or extracting at inappropriate times.
2. **Legacy code path:** EntityExtractor uses LLMGraphTransformer + PropertyEnricher (two-pass), which Phase 1B replaced with the single-call OntologyConstrainedExtractor pipeline.
3. **No DI:** ConversationHandler constructs EntityExtractor internally, violating the DI pattern established in Phase 2A.
4. **No curation:** Extracted knowledge bypasses the Phase 2A curation pipeline (no dedup, no conflict resolution, no provenance tracking).
5. **GraphRegenerator uses legacy path:** Still constructs EntityExtractor directly.

---

## 2. Goals

1. Replace EntityExtractor with ExtractionPipeline in ConversationHandler via constructor DI
2. Make extraction automatic (fire-and-forget on every user turn) instead of LLM-invoked
3. Remove extract_knowledge and extract_knowledge_from_document tools
4. Migrate GraphRegenerator to ExtractionPipeline (with include_curation=False)
5. Delete legacy EntityExtractor and PropertyEnricher
6. Update system prompt to reflect single tool (query_knowledge_graph only)
7. All changes tested with existing test suite passing

---

## 3. Non-Goals

- Changing KnowledgeRetriever or query_knowledge_graph tool (works fine as-is)
- Adding new tools to replace extracted ones
- Periodic curation or background jobs (Phase 4)
- Modifying the event store or its integration

---

## 4. Architecture

### 4.1 Automatic Extraction Flow

```
handle_message(user_message, session_id)
  -> session.add_message("user", user_message)
  -> _build_messages(session, max_history, doc_context)
  -> llm_with_tools.ainvoke(messages)           # Only query_knowledge_graph available
  -> [tool execution if LLM called query tool]
  -> session.add_message("assistant", response)
  -> _record_turn_event(...)                    # Synchronous, <5ms
  -> asyncio.create_task(                       # Fire-and-forget
       self._extract_knowledge_async(
         utterance=user_message,
         conversation_history=session.get_history(),
         event_id=event_id,                     # From event store turn
         session_id=session_id,
       )
     )
  -> return assistant_message                   # User sees response immediately
```

The extraction task runs in the background. Failures are logged but never propagate to the user. The LLM response is returned before extraction completes.

### 4.2 ConversationHandler Constructor

```python
class ConversationHandler:
    def __init__(
        self,
        config: KnowledgeConfig,
        graph_store: GraphStore,
        extraction_pipeline: ExtractionPipeline,
        model_name: str = "qwen2.5:7b",
    ) -> None:
```

ExtractionPipeline is a required constructor parameter. Factory wiring in `build_conversation_handler()` provides the real implementation. Tests pass fakes directly.

### 4.3 GraphRegenerator Constructor

```python
class GraphRegenerator:
    def __init__(
        self,
        config: KnowledgeConfig,
        extraction_pipeline: ExtractionPipeline,
        graph_store: GraphStore,
    ) -> None:
```

ExtractionPipeline and GraphStore are required constructor params (no hidden construction, consistent with CLAUDE.md DI rule). Factory wiring in `build_graph_regenerator()` provides real implementations with `include_curation=False`. The regenerator calls `pipeline.extract_from_utterance()` with synthetic IDs.

### 4.4 Module Dependency Changes

```
ConversationHandler
  OLD: EntityExtractor (hidden construction)
  NEW: ExtractionPipeline (injected via constructor)

GraphRegenerator
  OLD: EntityExtractor (hidden construction), build_graph_store (hidden)
  NEW: ExtractionPipeline (injected or factory-built with include_curation=False)
```

---

## 5. Interface Contracts

### 5.1 ConversationHandler._extract_knowledge_async

```python
async def _extract_knowledge_async(
    self,
    utterance: str,
    conversation_history: list[dict[str, str]],
    event_id: str,
    session_id: str,
) -> None:
    """Fire-and-forget background extraction.

    Called via asyncio.create_task after every user turn.
    Failures are logged but never propagated.

    Args:
        utterance: The user's message text.
        conversation_history: Recent conversation as role/content dicts.
            session.get_history() already returns this format.
        event_id: Event store turn ID for provenance.
        session_id: Conversation session ID.
    """
```

Implementation:
- Wraps `self._extraction_pipeline.extract_from_utterance()` in try/except
- Logs extraction results at DEBUG level (entity count, relationship count, curation stats)
- Logs errors at ERROR level but never raises
- Short messages (<3 words) are skipped (same heuristic as auto-RAG)

### 5.2 GraphRegenerator._extract_and_store (updated)

```python
async def _extract_and_store(self, utterance: Utterance) -> tuple[int, int]:
    """Re-extract entities from utterance via ExtractionPipeline.

    Uses synthetic event_id and session_id since regeneration
    operates on historical data without original provenance.

    Args:
        utterance: Utterance to process.

    Returns:
        Tuple of (entities_count, relationships_count).
    """
```

Implementation:
- Calls `self._pipeline.extract_from_utterance()` with:
  - `event_id=f"regen_{utterance.utterance_id}"`
  - `session_id=utterance.conversation_id or "regeneration"`
  - `conversation_history=[]` (no context during regeneration)
- Returns `(len(result.entities), len(result.relationships))` from ValidationResult
- With `include_curation=False`, the pipeline stops at Stage 6 (validation) and does NOT write to Neo4j. The regenerator must store results itself.

**Regeneration storage strategy:**

Add a new method `GraphStore.store_validated_entities(entities: list[dict], relationships: list[dict], utterance_id: str, ontology_version: str | None)` that accepts the dict-based format from `ValidationResult` instead of `GraphDocument` with `Node` objects. This method uses the same MERGE Cypher as the existing `store_extracted_entities` but reads from dict keys (`id`, `type`, `name`, `confidence`, etc.) instead of Node attributes.

The existing `store_extracted_entities(graph_document: GraphDocument, ...)` remains for backward compatibility until all callers are migrated. Both methods use the same underlying MERGE queries.

### 5.3 Updated System Prompt

The system prompt in `_build_messages()` is updated to:
- Remove documentation for extract_knowledge and extract_knowledge_from_document tools
- Keep query_knowledge_graph documentation unchanged
- Add brief note: "Knowledge from conversations is captured automatically -- you do not need to extract it manually."
- Remove references to "three tools" (now one tool)
- Simplify tool usage strategy section

### 5.4 _setup_tools (updated)

```python
def _setup_tools(self):
    """Setup LLM tools -- query_knowledge_graph only."""
    retriever = self.retriever

    @tool
    async def query_knowledge_graph(...) -> str:
        # ... unchanged ...

    self.tools = [query_knowledge_graph]
    self.llm_with_tools = self.llm.bind_tools(self.tools)
```

extract_knowledge and extract_knowledge_from_document tool definitions are deleted entirely.

---

## 6. Integration Points

### 6.1 handle_message Changes

In `handle_message()`, after `_record_turn_event()` returns (line 402), add:

```python
# Fire-and-forget background extraction
# _record_turn_event returns the event_id (or None if event store disabled)
event_id = self._record_turn_event(...)
if event_id and len(user_message.split()) >= 3:
    asyncio.create_task(
        self._extract_knowledge_async(
            utterance=user_message,
            conversation_history=session.get_history(max_history),
            event_id=event_id,
            session_id=session_id,
        )
    )
```

The `_record_turn_event` method is updated to return `str | None` instead of `None`. Returns the event_id on success, None when the event store is disabled or on failure. This avoids shared mutable state (`self._last_event_id`) which would break under concurrent sessions.

**Note:** When event store is disabled, extraction is silently skipped. This is intentional -- without an event store, there is no event_id for provenance, so extraction results would lack traceability. The raw utterance is still preserved in session history.

**Note:** The conversation history passed to extraction includes the assistant's response to the current turn (added at line 386-391 before extraction fires). This is intentional -- the assistant's response provides additional context for entity extraction.

### 6.2 Factory Updates

```python
def build_conversation_handler(
    config: KnowledgeConfig,
    model_name: str = "qwen2.5:7b",
) -> ConversationHandler:
    """Create a fully wired ConversationHandler."""
    gs = build_graph_store(config)
    pipeline = build_extraction_pipeline(config, graph_store=gs, include_curation=True)
    return ConversationHandler(
        config=config,
        graph_store=gs,
        extraction_pipeline=pipeline,
        model_name=model_name,
    )


def build_graph_regenerator(config: KnowledgeConfig) -> GraphRegenerator:
    """Create a fully wired GraphRegenerator (no curation)."""
    gs = build_graph_store(config)
    pipeline = build_extraction_pipeline(config, graph_store=gs, include_curation=False)
    return GraphRegenerator(
        config=config,
        extraction_pipeline=pipeline,
        graph_store=gs,
    )
```

### 6.3 Callers of ConversationHandler

Search for all places that construct `ConversationHandler(...)` and update them to use the factory or pass the pipeline:

- `backend/chat/knowledge_integration.py:39` -- KnowledgeIntegration constructs ConversationHandler (primary production caller, used by voice pipeline via model_manager.py)
- `scripts/` -- any CLI scripts that construct ConversationHandler directly
- Grep for `ConversationHandler(` to catch all

### 6.4 Callers of GraphRegenerator

- `scripts/regenerate_graph.py:245` -- constructs GraphRegenerator(config) directly
- Update to use `build_graph_regenerator(config)` factory

---

## 7. Files

### Modified Files

| File | Change |
|------|--------|
| `backend/chat/conversation_handler.py` | DI refactor, remove 2 tools, add automatic extraction, update system prompt |
| `backend/chat/knowledge_integration.py` | Update ConversationHandler construction (primary production caller) |
| `backend/knowledge/regeneration/graph_regenerator.py` | Replace EntityExtractor with ExtractionPipeline, DI refactor |
| `backend/knowledge/storage/graph_store.py` | Add store_validated_entities() for dict-based input |
| `backend/factories.py` | Add build_conversation_handler(), build_graph_regenerator() |
| `backend/knowledge/extraction/__init__.py` | Remove EntityExtractor, PropertyEnricher from exports |
| `backend/knowledge/scripts/seed_from_docs.py` | Update stale reference to removed extraction tool |
| `REPOSITORY_STRUCTURE.md` | Update extraction flow diagram |

### Deleted Files

| File | Reason |
|------|--------|
| `backend/knowledge/extraction/entity_extractor.py` | Replaced by ExtractionPipeline |
| `backend/knowledge/extraction/property_enricher.py` | Replaced by OntologyConstrainedExtractor |

### New Test Files

| File | Responsibility |
|------|---------------|
| `tests/unit/chat/test_conversation_handler.py` | ConversationHandler extraction integration |
| `tests/unit/knowledge/regeneration/test_graph_regenerator.py` | GraphRegenerator pipeline migration |

---

## 8. Testing Strategy

### ConversationHandler Tests

**test_extract_knowledge_async_fires_on_user_turn:** Verify that after handle_message returns, an extraction task was created. Use FakeExtractionPipeline that records calls.

**test_extraction_failure_does_not_crash_conversation:** Verify that when extraction raises, the conversation response is still returned and the error is logged.

**test_short_messages_skip_extraction:** Verify that messages with fewer than 3 words do not trigger extraction.

**test_tools_only_include_query:** Verify that self.tools contains only query_knowledge_graph (not extract_knowledge or extract_knowledge_from_document).

### GraphRegenerator Tests

**test_regenerator_uses_extraction_pipeline:** Verify that _extract_and_store calls pipeline.extract_from_utterance with correct synthetic IDs.

**test_regenerator_uses_no_curation:** Verify the pipeline is built with include_curation=False.

---

## 9. Implementation Order

1. Update ConversationHandler constructor (DI for ExtractionPipeline)
2. Add _extract_knowledge_async method
3. Update handle_message to fire extraction task
4. Remove extract_knowledge and extract_knowledge_from_document tools
5. Update _setup_tools (query_knowledge_graph only)
6. Update system prompt
7. Update _record_turn_event to expose event_id
8. Migrate GraphRegenerator to ExtractionPipeline
9. Add build_conversation_handler() factory
10. Update server.py and other callers
11. Delete entity_extractor.py and property_enricher.py
12. Update extraction __init__.py exports
13. Write tests
14. Full test suite verification

Items 1-7 are the ConversationHandler block (sequential).
Item 8 is independent (can parallelize with 1-7).
Items 9-12 are cleanup (sequential, after 1-8).
Items 13-14 are verification (after all above).

---

## 10. Architectural Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Automatic extraction via asyncio.create_task | More reliable than LLM-decided extraction; user sees no latency |
| 2 | Skip extraction for <3 word messages | Matches auto-RAG heuristic; "hi", "ok", "thanks" yield nothing useful |
| 3 | ExtractionPipeline via constructor DI | Consistent with Phase 2A pattern; testable with fakes |
| 4 | GraphRegenerator uses include_curation=False | No benefit to dedup/conflict resolution on a wiped graph |
| 5 | Synthetic event_id for regeneration | f"regen_{utterance_id}" preserves traceability without real event store |
| 6 | Delete legacy files immediately | No grace period needed -- all callers are migrated in this phase |
| 7 | _record_turn_event exposes event_id | Needed for provenance linking in automatic extraction |

---

## 11. Risk Assessment

**GPU contention:** Extraction's Ollama call runs concurrently with response generation. Since extraction fires AFTER the response is generated and returned, contention only affects the next turn's response time if extraction is still running. Qwen 2.5 7B extraction completes in ~1-2s; typical user think time is >5s. Low risk.

**Silent extraction failures:** Fire-and-forget means failures are only visible in logs. Acceptable because the event store preserves the raw utterance -- missed extractions can be recovered via regeneration.

**Breaking external callers:** Any code that constructs ConversationHandler directly (outside the factory) will break. The server.py update covers the known caller. A grep for `ConversationHandler(` should catch all others.
