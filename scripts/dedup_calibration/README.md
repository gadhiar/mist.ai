# Dedup threshold calibration

Measurement tool for `SIMILARITY_THRESHOLD` in
`backend/knowledge/curation/deduplication.py`. It reports where labelled
duplicate pairs and labelled distinct pairs fall under the embedding score and
what threshold each policy implies. It does not change the threshold.

## Scale

`SIMILARITY_THRESHOLD = 0.92` is compared against Neo4j's
`vector.similarity.cosine`, which returns `(1 + cos) / 2`. So 0.92 is a raw
cosine of 0.84. The report shows both scales; every threshold it prints is a
Neo4j score, with the raw-cosine equivalent beside it. A pair merges when
`score >= threshold`.

## Run it (live, inside the backend container)

The container holds the HuggingFace model cache. From the repo root on the
host, in Git Bash:

```bash
MSYS_NO_PATHCONV=1 docker compose exec -T mist-backend python -m scripts.dedup_calibration --output /app/data/dedup_calibration.json
```

The table prints to the terminal. The JSON lands in `./data/dedup_calibration.json`
on the host (`./data` is bind-mounted at `/app/data`). The dev override mounts
`./scripts` and `./backend` read-only, so no image rebuild is needed; without the
override, rebuild the backend image first.

The tool opens no Neo4j connection and makes no network call of its own. It
loads the embedding model eagerly and exits 2 with a message if it cannot.

Options: `--dataset PATH` (default `pairs.json` next to this file), `--output PATH`,
`--threshold X` (default: the live `SIMILARITY_THRESHOLD`).

Exit codes: 0 report produced; 1 dataset invalid or output unwritable; 2 embedding
model cannot load or run; 64 usage error.

## What it reports

Per slice (`extracted_vs_extracted`, `extracted_vs_seed`) and overall:

- distributions (min, p10, p25, median, p75, p90, max, mean) of raw cosine and Neo4j
  score, for duplicate pairs and for distinct pairs
- the zero-false-merge point: the highest score among distinct pairs, and the lowest
  observed duplicate score above it (the smallest cutoff with no false merge that still
  merges at least one duplicate; absent when no duplicate outscores every distinct pair)
- the best-F1 threshold (searched over every observed score; ties go to the higher one)
- precision, recall and F1 at the current threshold
- the five highest-scoring distinct pairs and five lowest-scoring duplicates

The JSON adds every scored pair, so a later goal can re-cut thresholds without
re-embedding.

## What text is embedded

- Probe side: the display name, as `EntityDeduplicator._find_existing` embeds it
  (`deduplication.py:149-154`).
- Stored side, `extracted_vs_extracted`: the display name, as the extraction write path
  stores it (`graph_writer.py:384-388`).
- Stored side, `extracted_vs_seed`: `embedding_text_for(display_name, description, id)`
  (`admin.py:319`, `:387`), which is the display name when the seed node has no
  description. Datasets may add `b_description` to a seed-slice pair to model one that has.

Two other builders write node embeddings and are not modelled here:
`embedding_maintenance._build_embedding_text` (`"name entity_type description"`, job
registered with `enabled=False` at `factories.py:995`) and `GraphStore._store_validated_node`
(`"id entity_type description"`, reached from `graph_regenerator.py:439`). If either has run
against the live graph, stored vectors there differ from what this tool measures.

## Dataset

`pairs.json` holds labelled pairs, each with a slice, an entity type from the ontology,
a label (`duplicate` or `distinct`) and two names. `hard: true` marks same-type near
misses with overlapping tokens. The names are illustrative fixtures shaped like MIST
entities; the host seed files are not in the repo. Only same-type pairs are included, so
the Abstraction-cluster widening in `dedup_type_filter` is not exercised. Add pairs
that mirror real graph entities before trusting a threshold for production; `load_dataset`
enforces count floors, slice and type validity, and rejects repeated pairs.

## Tests

`tests/unit/dedup_calibration/` (hermetic; a fake embedder stands in for the model).
