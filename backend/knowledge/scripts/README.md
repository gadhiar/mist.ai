# Knowledge Graph Seeding Scripts

Scripts to initialize and populate the MIST knowledge graph.

## Workflow

### 1. Wipe Database (Start Fresh)

```bash
python backend/knowledge/scripts/wipe_database.py
```

**What it does:**
- Deletes ALL nodes and relationships from Neo4j
- Drops all constraints and indexes
- Gives you a completely clean slate

**Warning:** This is destructive and irreversible. You'll be prompted to confirm.

### 2. Seed from MIST Documentation

```bash
python backend/knowledge/scripts/seed_from_docs.py
```

**What it does:**
- Scans all markdown files in `docs/` directory
- Chunks documents into 100-200 word segments
- Creates **SourceDocument** nodes (one per file)
- Creates **DocumentChunk** nodes with embeddings for vector search
- Builds searchable RAG corpus
- **Note:** Entity extraction happens selectively via LLM tools, not during seeding

**Architecture:**
```
SourceDocument (file metadata, hash)
    -> FROM_SOURCE
DocumentChunk (text + embedding for RAG)
    -> EXTRACTED_FROM (created on-demand by LLM)
Entity (knowledge graph nodes)
```

**Expected output:**
- Source documents: ~15-20
- Document chunks: ~50-200 (with embeddings)

**Time:** ~30 seconds - 2 minutes (fast! No extraction during seeding)

**Why no extraction during seeding?**
- Keeps seeding fast and scalable (think: entire codebases, millions of docs)
- LLM searches documents via vector search (RAG)
- LLM optionally extracts important facts into KG when needed
- First query: slower (vector search), subsequent: faster (graph traversal)

## Usage Example

```bash
# Complete reset and seed
python backend/knowledge/scripts/wipe_database.py
python backend/knowledge/scripts/seed_from_docs.py
```

## Verification

After seeding, verify in Neo4j Browser:

```cypher
// Check source documents
MATCH (s:SourceDocument)
RETURN s.title, s.source_type, s.file_path, s.file_size
ORDER BY s.title

// Check document chunks
MATCH (c:DocumentChunk)
RETURN count(c) as chunk_count

// Check a few chunks with embeddings
MATCH (s:SourceDocument)-[:FROM_SOURCE]->(c:DocumentChunk)
RETURN s.title, c.position, c.text, size(c.embedding) as embedding_dim
LIMIT 5

// Verify embeddings are present
MATCH (c:DocumentChunk)
WHERE c.embedding IS NOT NULL
RETURN count(c) as chunks_with_embeddings

// Check the document architecture
MATCH (s:SourceDocument)-[:FROM_SOURCE]->(c:DocumentChunk)
RETURN s.title, count(c) as chunk_count
ORDER BY chunk_count DESC

// Check if any entities were extracted (should be 0 after fresh seeding)
MATCH (e:__Entity__)
RETURN count(e) as entity_count

// When entities ARE extracted via LLM tools, verify provenance
MATCH (c:DocumentChunk)<-[:EXTRACTED_FROM]-(e:__Entity__)
RETURN e.id, e.entity_type, c.chunk_id
LIMIT 10
```

## Configuration

Scripts use environment variables from `.env`:
- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `MODEL` (for entity extraction)

## Future Enhancements

Potential additional seeding sources:
- Curated conversational facts
- PersonaChat dataset
- Synthetic user preferences
- External knowledge datasets (ConceptNet, DBpedia)
