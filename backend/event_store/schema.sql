-- MIST Event Store Schema
-- Layer 1: Immutable conversation event storage.
-- Applied idempotently via EventStore.initialize().

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- Conversation session lifecycle
CREATE TABLE IF NOT EXISTS conversation_sessions (
    session_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,                -- ISO-8601
    ended_at TEXT,                           -- ISO-8601, NULL if still active
    turn_count INTEGER DEFAULT 0,
    input_modality TEXT DEFAULT 'voice'      -- 'voice', 'text', 'api'
);

-- Individual conversation turns (immutable after creation)
CREATE TABLE IF NOT EXISTS conversation_turn_events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES conversation_sessions(session_id),
    turn_index INTEGER NOT NULL,
    timestamp TEXT NOT NULL,                 -- ISO-8601

    -- Raw conversation data (NEVER changes)
    user_utterance TEXT NOT NULL,
    system_response TEXT NOT NULL,

    -- Context window (what the LLM actually saw)
    context_window TEXT,                     -- JSON: [{"role": "user", "content": "..."}, ...]

    -- RAG context (what retrieval injected)
    retrieval_context TEXT,                  -- JSON: {"doc_chunks": [...], "graph_facts": [...]}

    -- Tool usage
    tool_calls TEXT,                         -- JSON: [{"name": "...", "args": {...}, "result_summary": "..."}]

    -- Audio reference
    audio_hash TEXT,                         -- SHA-256 of audio file in archive
    audio_format TEXT,                       -- "wav", "opus"
    audio_duration_ms INTEGER,
    audio_sample_rate INTEGER,

    -- Model metadata
    stt_model TEXT,
    tts_model TEXT,
    llm_model TEXT,
    llm_parameters TEXT,                     -- JSON: {"temperature": 0.7, "top_p": 1.0, ...}

    -- Versioning
    ontology_version TEXT NOT NULL DEFAULT '1.0.0',

    -- Immutability timestamp
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Re-extraction job tracking (Phase 5)
CREATE TABLE IF NOT EXISTS re_extraction_jobs (
    job_id TEXT PRIMARY KEY,
    target_ontology_version TEXT NOT NULL,
    source_ontology_version TEXT,
    status TEXT NOT NULL DEFAULT 'pending',   -- 'pending', 'running', 'paused', 'completed', 'failed'
    total_events INTEGER DEFAULT 0,
    processed INTEGER DEFAULT 0,
    failed INTEGER DEFAULT 0,
    last_event_id TEXT,
    started_at TEXT,
    updated_at TEXT,
    errors TEXT                               -- JSON array of error strings
);

-- Graph health history (Phase 4)
CREATE TABLE IF NOT EXISTS graph_health_events (
    event_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    health_score REAL NOT NULL,
    metrics TEXT NOT NULL,                    -- JSON: component scores
    entity_count INTEGER,
    relationship_count INTEGER,
    archived_count INTEGER,
    community_count INTEGER
);

-- Materialized graph registry (Phase 5)
CREATE TABLE IF NOT EXISTS materialized_graph_registry (
    graph_id TEXT PRIMARY KEY,
    ontology_version TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    is_active INTEGER DEFAULT 0,
    generation_stats TEXT,                    -- JSON
    source_event_count INTEGER,
    neo4j_database TEXT DEFAULT 'neo4j'
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_turns_session ON conversation_turn_events(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_timestamp ON conversation_turn_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_turns_ontology ON conversation_turn_events(ontology_version);
