# Knowledge Graph Maintenance Scripts

Scripts for maintaining the MIST knowledge graph.

## Wipe Database (Start Fresh)

```bash
python backend/knowledge/scripts/wipe_database.py
```

**What it does:**
- Deletes ALL nodes and relationships from Neo4j
- Drops all constraints and indexes
- Gives you a completely clean slate

**Warning:** This is destructive and irreversible. You'll be prompted to confirm.

## Repopulating Documents

The former `GraphStore`-backed document seeder has been removed. Document
ingestion goes through `IngestionPipeline`
(`backend/knowledge/ingestion/pipeline.py`), built by
`build_ingestion_pipeline()` in `backend/factories.py`. It stores chunk text
and embeddings in the vector store and, when given a graph store, records
provenance in Neo4j as `ExternalSource` and `VectorChunk` nodes.

## Verification

After a wipe, verify in Neo4j Browser:

```cypher
// Should return 0 after a wipe
MATCH (n)
RETURN count(n) as node_count
```

## Configuration

Scripts use environment variables from `.env`:
- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
