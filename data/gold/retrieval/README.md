# Extraction retrieval queries

## Purpose

This set measures memory end to end: what extraction stored from a conversation, as the
retriever finds it at turn time. The dev turns (the 60 probes of
`data/ingest/extraction-gold-2026-06-14.jsonl` and the 40 of
`data/gold/extraction/dev-lists-2026-11.jsonl`) are passed through extraction, the result is
built into an isolated eval graph, and each query here is put to the retriever. A query scores
when a fact that answers it comes back in the top five.

An extraction score says whether the right facts were produced; this one says whether they
are any use afterwards. A fact extracted under an id the retriever cannot reach, or ranked
below five unrelated facts, counts here as a miss even when the extraction score counted it
as a hit. The bar is capability first, then feel, and this is a capability measure.

The target is 0.90 hit at 5 over the non-negative queries.

## How it differs from the read path's recall fixture

The read path's recall fixture (from the mist-read-path goal) scores retrieval over facts
authored directly into a graph. It never passes an utterance through extraction, so it
isolates the retriever. This set keeps extraction in the loop: its graph holds only what
extraction stored for the dev turns, so a miss here can come from extraction (the fact was
never stored, or stored under another id or predicate) as well as from the retriever. Read
the two together: a fact the fixture finds and this set misses points at extraction.

## Query file: `queries-2026-11.jsonl`

One JSON object per line:

```json
{"id": "rq-direct-01", "query": "What do I use Rust for?", "kind": "direct",
 "relevant": [{"source": "rust", "predicate": "USED_FOR", "target": "backend-work", "grade": 2},
              {"source": "user", "predicate": "USES", "target": "rust", "grade": 1}]}
```

- `kind` is one of:
  - `direct` (15): the query names the fact's entities;
  - `paraphrase` (12): the query shares no content word with its facts' entity names;
  - `list_item` (10): the answering facts come from list-shaped dev probes;
  - `type_scoped` (8): the query asks for every entity of a kind ("which databases do I use");
  - `negative` (5): nothing in the dev turns answers it, and `relevant` is empty.
- `grade` 2 means the fact answers the query; grade 1 means it supports the answer. Every
  non-negative query has at least one grade-2 fact and at most five relevant facts, so recall
  at 5 can reach 1.
- `source` and `target` are canonical gold entity ids; `predicate` is the gold predicate.

Every relevant fact is a gold relationship of the 60-probe set or of `dev-lists` that no dev
probe ceases or retracts, and none is a relationship of the held-out set.
`tests/unit/eval_harness/test_retrieval_queries.py` checks all of that, along with the kind
counts and the paraphrase rule. The data is synthetic: a fictional user, fictional
organisations and first names only.

A fact that a dev probe asserts can still be absent from a built graph, for example when a
later turn's cessation retires it. Whatever builds the graph should list those facts, so a
miss on one is read as a build question, not a retrieval one.

## Run file

The scorer reads one JSON object per line, one line per query that was run:

```json
{"id": "rq-direct-01",
 "retrieved": [{"source": "rust", "predicate": "USED_FOR", "target": "backend-work"},
               {"source": "user", "predicate": "USES", "target": "rust",
                "source_type": "User", "target_type": "Technology"}]}
```

- `retrieved` is in rank order, best first, and may be empty.
- `source_type` and `target_type` are optional; other keys on a row or a fact are ignored.
- A query with no line was not run. It scores as a miss, and a negative query that was not
  run does not count as returning nothing.

Produced ids are matched with the extraction scorer's normalisation
(`scripts/eval_harness/score_extraction_run.py`): `canonical_id` for every endpoint, and the
value-first id for a Metric endpoint (a Metric by its stated type, or because the predicate
admits only a Metric there, as `HAS_METRIC`'s target does). Predicates compare upper-cased.

## Scoring

```bash
python -m scripts.eval_harness.extraction_retrieval_score \
    --queries data/gold/retrieval/queries-2026-11.jsonl --run <run.jsonl> \
    --json-out retrieval_summary.json --out retrieval.md
```

Per query, at k = 5:

- hit at 5: any grade-2 fact in the top five;
- recall at 5: the share of the query's relevant facts, of either grade, in the top five;
- reciprocal rank: 1 / the rank of the first grade-2 fact in the top five, else 0; its mean
  is MRR;
- nDCG at 5: gain 2^grade - 1, discounted by log2(rank + 1), over the ideal ordering.

A fact retrieved twice is credited once, at its first rank. Each metric is averaged per kind
and over every non-negative query. For the negative queries the one number is the share that
returned nothing. Exit 0, or 2 when an input is missing or malformed.

## Extending the set

Add queries in the same schema, draw facts only from the dev gold files, and keep the kind
counts in the test in step. Never open the held-out set to choose or check a fact: the test
compares against it programmatically and reports only query ids.
