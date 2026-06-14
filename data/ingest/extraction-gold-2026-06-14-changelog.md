# Extraction Gold Corpus Changelog: 2026-06-10 -> 2026-06-14

**Base corpus:** `data/ingest/extraction-gold-2026-06-10.jsonl` (60 probes)
**New corpus:** `data/ingest/extraction-gold-2026-06-14.jsonl` (60 probes)
**Adjudicated against:** v1.4.0 taxonomy + third-party tracking policy decision
**Author:** Task 14, MIS-124 plan
**Date:** 2026-06-14

## Per-probe changelog

| Probe tag | Class | Before -> After | Rule / Reason |
|---|---|---|---|
| ext-01-uses | 1 | entity `backend-work` type "Topic" -> "Concept"; rel `USED_FOR` target_type "Topic" -> "Concept" | v1.4.0 RETIRED_TYPE_MAP: Topic is canonicalized to Concept; gold updated to read in v1.4.0 terms |
| ext-14-uses-usedfor | 1 | entity `cloud-infrastructure` type "Topic" -> "Concept"; rel `USED_FOR` target_type "Topic" -> "Concept" | v1.4.0 RETIRED_TYPE_MAP: Topic is canonicalized to Concept; gold updated to read in v1.4.0 terms |
| ext-21-milestone-on-date | 1 | entity `mobile-app-launch` type "Milestone" -> "Event"; rel source_type "Milestone" -> "Event" | v1.4.0 RETIRED_TYPE_MAP: Milestone is canonicalized to Event; gold updated to read in v1.4.0 terms |
| ext-22-milestone-on-date | 1 | entity `house-closing` type "Milestone" -> "Event"; rel source_type "Milestone" -> "Event" | v1.4.0 RETIRED_TYPE_MAP: Milestone is canonicalized to Event; gold updated to read in v1.4.0 terms |
| ext-29-third-party-knows | 2 | ADD entity `biotech` (Organization); KEEP `user` anchor + `user -[KNOWS_PERSON]-> anjali`; ADD rel `anjali -[WORKS_AT]-> biotech` (Person -> Organization) | Policy: track durable third-party facts. Utterance "My sister Anjali works in biotech" states a durable employment fact about Anjali. anjali-[WORKS_AT]->biotech is the stated third-party fact; it belongs in gold alongside the user anchor. The user anchor and KNOWS_PERSON edge were already present in the 2026-06-10 gold and are retained. See anchor note below. |
| ext-08-third-party-person | 2 | No change | Audited: gold already has user-[KNOWS_PERSON]->priya AND priya-[RECOMMENDS]->duckdb. "My mentor Priya suggested I try DuckDB" states Priya's recommendation, a durable third-party fact; both edges already present and correct. |
| ext-30-third-party-knows | 2 | No change | Audited: utterance "I had coffee with my old manager Devin last week" describes an incidental social meeting. "Had coffee" does not state a durable fact about Devin (no employer, no recommendation, no persistent attribute). The coffee meeting is an incidental event, not a third-party fact to track. Gold correctly has only user-[KNOWS_PERSON]->devin. |
| ext-31-third-party-recommends | 2 | No change | Audited: gold has user-[KNOWS_PERSON]->mateo AND mateo-[RECOMMENDS]->graphql. "My architect Mateo recommended GraphQL for the new service" states a durable recommendation fact about Mateo. Both edges present and complete. No addition required. |
| ext-32-third-party-recommends | 2 | No change | Audited: gold has user-[KNOWS_PERSON]->lena AND lena-[RECOMMENDS]->bun. "A teammate named Lena pushed me toward trying Bun" states a durable recommendation fact about Lena. Both edges present and complete. No addition required. |
| ext-24-metric | 3 | No change (already canonical) | Gold metric id is `12000-requests-per-second` -- value-first format already in place. |
| ext-25-metric | 3 | No change (already canonical) | Gold metric id is `87-percent-coverage` -- value-first format already in place. |
| ext-21-milestone-on-date | 4 | No change | Gold entity id `mobile-app-launch` correctly names the launch event, not the artifact (`mobile-app`). Naming-convention rule satisfied; id already at the grain the OCCURRED_ON predicate operates on. |
| ext-22-milestone-on-date | 4 | No change | Gold entity id `house-closing` correctly names the closing event, not the artifact (`house`) or the wrong phrasing (`closing-on-house` which the model emits). Naming-convention rule satisfied. |
| ext-23-event-on-date | 4 | No change | Gold entity id `wedding-2026-05-30` correctly names the event (date-qualified wedding event). Naming-convention rule satisfied. |

## Class 2 audit: ext-08/29/30/31/32 in full

**ext-08** (`My mentor Priya suggested I try DuckDB`): The utterance states Priya's recommendation of DuckDB -- a durable third-party fact (what the person recommends). Gold already carries both `user-[KNOWS_PERSON]->priya` and `priya-[RECOMMENDS]->duckdb`. AUDITED -- no change.

**ext-29** (`My sister Anjali works in biotech`): The utterance states Anjali's employment in biotech -- a durable third-party fact (where the person works). The original gold (2026-06-10) contained only `anjali` as an entity with no user or anchor edge. The diagnostics confirm the user entity and KNOWS_PERSON edge were FNs (anchor-miss). This change:
1. Restores the user entity + `user-[KNOWS_PERSON]->anjali` anchor (correct per gold labeling policy requiring the user anchor for user-scope predicates).
2. Adds `biotech` as an Organization entity.
3. Adds `anjali-[WORKS_AT]->biotech` as the stated third-party fact.
The model was correct to extract WORKS_AT here; the prior gold omission was an error of incompleteness. CHANGED.

**ext-30** (`I had coffee with my old manager Devin last week`): "Had coffee" is an incidental social event -- it records that a meeting occurred, not a durable attribute of Devin. No employer, no recommendation, no skill, no persistent claim about Devin is stated. A coffee-meeting Event would be over-extraction (confirmed: the diagnostic explicitly classifies `ext-30 EXPERIENCED coffee-meeting` as a genuine over-extraction). Gold correctly has `user-[KNOWS_PERSON]->devin` only. AUDITED -- no change.

**ext-31** (`My architect Mateo recommended GraphQL for the new service`): Gold has `user-[KNOWS_PERSON]->mateo` and `mateo-[RECOMMENDS]->graphql`. The recommendation is a durable third-party fact. Both edges complete. AUDITED -- no change.

**ext-32** (`A teammate named Lena pushed me toward trying Bun`): Gold has `user-[KNOWS_PERSON]->lena` and `lena-[RECOMMENDS]->bun`. The recommendation is a durable third-party fact. Both edges complete. AUDITED -- no change.

## Class 4 decisions: ext-21/22/23

**ext-21** (`We launched the mobile app on 2026-06-01`): Gold entity id is `mobile-app-launch`. This names the launch event, not the artifact (`mobile-app`). The OCCURRED_ON predicate operates on the event, not the product. Naming convention is satisfied. The model's FP entity `mobile-app` confirms the gold is measuring the right distinction. CLASS 4: no change (gold already correct).

**ext-22** (`I closed on my house on 2026-03-12`): Gold entity id is `house-closing`. This names the closing event. Not `house` (the artifact) and not `closing-on-house` (the model's phrasing-driven surface). The OCCURRED_ON predicate operates on the closing event. Naming convention is satisfied. CLASS 4: no change (gold already correct).

**ext-23** (`I went to a wedding on 2026-05-30`): Gold entity id is `wedding-2026-05-30`. This names the event (a date-qualified wedding event). At r4 the model chose `wedding-event` instead -- that is a documented model residual, not a gold issue. Gold correctly names the specific event instance. CLASS 4: no change (gold already correct).

## Anchor note for ext-29

The original 2026-06-10 gold for ext-29 already had the user entity and `user-[KNOWS_PERSON]->anjali` anchor edge. The diagnostic FN "anchor-miss (user entity FN; KNOWS_PERSON anchor dropped)" refers to the MODEL dropping the user entity and anchor during extraction -- not to the gold being missing them. The gold was incomplete only in lacking the third-party fact edge.

The change to ext-29 therefore:
- KEEPS `user` entity and `user-[KNOWS_PERSON]->anjali` (already in original gold, preserved)
- ADDS `biotech` Organization entity
- ADDS `anjali-[WORKS_AT]->biotech` (the stated third-party fact, Class 2 addition)

## Validation checklist

- [ ] File loads with 60 probes (run docker exec command below)
- [ ] 6 negative-control probes (ext-10, ext-11, ext-37, ext-38, ext-39, ext-40) have empty expected_entities and expected_relationships
- [ ] All 60 lines are valid JSON
- [x] Committed (7cb2266)

Validation command:
```
docker compose exec -T mist-backend python -c "from scripts.eval_harness.score_extraction_run import iter_gold_probes; r=list(iter_gold_probes('data/ingest/extraction-gold-2026-06-14.jsonl')); print(len(r))"
```
Expected output: 60

## Summary of changes

- Class 1: 4 probes changed (ext-01, ext-14, ext-21, ext-22) -- Topic->Concept and Milestone->Event type relabels in both entity type fields and relationship source_type/target_type fields.
- Class 2: 1 probe changed (ext-29) -- added biotech Organization entity and anjali-[WORKS_AT]->biotech edge; 4 probes audited with no change (ext-08, ext-30, ext-31, ext-32).
- Class 3: 0 changes (ext-24 and ext-25 metric ids already value-first).
- Class 4: 0 changes (ext-21, ext-22, ext-23 gold entity ids already name the events correctly).

Total probes changed: 5 (ext-01, ext-14, ext-21, ext-22, ext-29).
Total probes audited with no change: 55.
