---
title: "ADR-009 Query Audit"
date: 2026-04-17
plan: "2026-04-17-adr-009-graph-provenance-separation"
reference_count: 105
---

# ADR-009 Query Audit

Classification of every `__Entity__` reference in backend/knowledge/ to guide the
graph-provenance-separation migration. See ADR-009 for decision rationale.

Note: grep found 105 references, not 121 as the plan estimated. The shortfall likely
reflects references that were already removed or never added in this worktree baseline.
All 105 actual references are classified below; the summary counts reflect reality.

## Summary

| Classification | Count | Downstream action |
|---|---|---|
| user-facing | 66 | Keep `:__Entity__` filter -- no change |
| audit-trail | 4 | Switch to `:__Provenance__` (Tasks 3-7 in plan) |
| cross-layer | 8 | Reference both labels (already cross-layer MATCH patterns) |
| schema | 6 | Add parallel `:__Provenance__` DDL (Task 2) |
| stats/dump | 21 | Extend to split label families (Tasks 10-12) |

Total: 105

## Classifications

| File | Line | Pattern | Classification | Action |
|---|---|---|---|---|
| backend/knowledge/admin.py | 90 | `FOR (e:__Entity__) REQUIRE e.id IS UNIQUE` | schema | Task 2: add parallel `FOR (p:__Provenance__) REQUIRE p.id IS UNIQUE` |
| backend/knowledge/admin.py | 95 | `FOR (e:__Entity__) ON (e.entity_type)` | schema | Task 2: add parallel `FOR (p:__Provenance__) ON (p.entity_type)` |
| backend/knowledge/admin.py | 101 | `FOR (e:__Entity__) ON e.embedding` | schema | Task 2: add parallel vector index for `:__Provenance__` if provenance nodes need vector search |
| backend/knowledge/admin.py | 194 | `MATCH (n:__Entity__) WHERE n.provenance = 'seed'` | user-facing | No change -- already correct; backfill targets seed entities only |
| backend/knowledge/admin.py | 211 | `MATCH (n:__Entity__ {id: $id}) SET n.embedding` | user-facing | No change -- already correct; embedding update on specific entity |
| backend/knowledge/admin.py | 237 | `MERGE (m:__Entity__:MistIdentity {id: $id})` | user-facing | No change -- already correct; MistIdentity is user-facing self-model |
| backend/knowledge/admin.py | 276 | `MERGE (n:__Entity__:{label} {{id: $id}})` | user-facing | No change -- already correct; seeds MistTrait/Capability/Preference entities |
| backend/knowledge/admin.py | 315 | `MERGE (n:__Entity__:{label} {{id: $id}})` | user-facing | No change -- already correct; seeds User and anchor entities |
| backend/knowledge/admin.py | 372 | `between two existing __Entity__ nodes` (docstring) | user-facing | No change -- already correct; docstring only, describes entity-to-entity relationships |
| backend/knowledge/admin.py | 381 | `MATCH (s:__Entity__ {{id: $source_id}})` | user-facing | No change -- already correct; seeds relationships between user-facing entities |
| backend/knowledge/admin.py | 404 | `excluding non-__Entity__ nodes` (docstring) | stats/dump | Task 10: docstring update -- comment will need to say "excluding non-user-facing nodes" |
| backend/knowledge/admin.py | 406 | `MATCH (n:__Entity__)` (count_nodes_by_type) | stats/dump | Task 10: add sibling `MATCH (n:__Provenance__)` count variant |
| backend/knowledge/admin.py | 414 | `only between __Entity__ nodes` (docstring) | stats/dump | Task 10: docstring update -- note relationship count covers user-facing layer only |
| backend/knowledge/admin.py | 416 | `MATCH (:__Entity__)-[r]->(:__Entity__)` (rel count) | stats/dump | Task 10: add sibling `MATCH (:__Provenance__)-[r]->(:__Provenance__)` variant |
| backend/knowledge/admin.py | 426 | `MATCH (n:__Entity__) WHERE n.confidence IS NOT NULL` | stats/dump | Task 10: add provenance variant; confidence dist is entity-only today |
| backend/knowledge/admin.py | 432 | `MATCH (:__Entity__)-[r]->(:__Entity__)` (confidence) | stats/dump | Task 10: add provenance variant |
| backend/knowledge/admin.py | 446 | `not an __Entity__` (docstring, find_orphan_relationships) | stats/dump | Task 12: update after `:__Provenance__` is live -- orphan check must allow provenance endpoints |
| backend/knowledge/admin.py | 453 | `WHERE NOT s:__Entity__ OR NOT t:__Entity__` | stats/dump | Task 12: update predicate to exclude cross-layer bridging edges from orphan count |
| backend/knowledge/admin.py | 462 | `seeded vs derived __Entity__ nodes` (docstring) | stats/dump | Task 10: docstring update -- provenance nodes do not carry provenance='seed' |
| backend/knowledge/admin.py | 464 | `MATCH (n:__Entity__)` (count_provenance) | stats/dump | Task 10: provenance nodes counted separately; add `:__Provenance__` variant |
| backend/knowledge/admin.py | 472 | `__Entity__ nodes whose provenance is NOT 'seed'` (docstring) | stats/dump | Task 10: docstring update -- function guards reset; provenance nodes always non-seed |
| backend/knowledge/admin.py | 478 | `MATCH (n:__Entity__) WHERE provenance <> 'seed'` | stats/dump | Task 12: safety guard should also count provenance nodes; extend query |
| backend/knowledge/admin.py | 492 | `full __Entity__ subgraph` (docstring, dump_graph_json) | stats/dump | Task 11: extend dump to include `:__Provenance__` subgraph as separate top-level key |
| backend/knowledge/admin.py | 494 | `MATCH (n:__Entity__)` (dump_graph_json nodes) | stats/dump | Task 11: also dump `MATCH (n:__Provenance__)` nodes into separate key |
| backend/knowledge/admin.py | 499 | `MATCH (s:__Entity__)-[r]->(t:__Entity__)` (dump rels) | stats/dump | Task 11: also dump cross-layer and provenance-only relationships |
| backend/knowledge/admin.py | 507 | `label != "__Entity__"` (label filter in dump) | stats/dump | Task 11: analogous filter needed for `"__Provenance__"` label stripping in provenance dump |
| backend/knowledge/admin.py | 525 | `full __Entity__ subgraph as Cypher script` (docstring) | stats/dump | Task 11: extend Cypher dump to include provenance subgraph |
| backend/knowledge/admin.py | 537 | `["__Entity__", *node["labels"]]` (dump_graph_cypher) | stats/dump | Task 11: conditionally emit `"__Provenance__"` prefix for provenance nodes |
| backend/knowledge/admin.py | 549 | `MATCH (s:__Entity__ {{id: {src}}})` (dump_graph_cypher) | stats/dump | Task 11: Cypher dump reconstructs user-facing rels only; extend for cross-layer |
| backend/knowledge/admin.py | 589 | `Wipe __Entity__ nodes` (docstring, reset_graph) | stats/dump | Task 12: docstring clarify reset wipes user-facing layer; add option for provenance wipe |
| backend/knowledge/admin.py | 600 | `MATCH (n:__Entity__) RETURN count(n)` (reset before) | stats/dump | Task 12: extend reset to optionally wipe `:__Provenance__` nodes |
| backend/knowledge/admin.py | 604 | `MATCH (:__Entity__)-[r]->(:__Entity__)` (reset rels) | stats/dump | Task 12: extend to count cross-layer rels before reset |
| backend/knowledge/admin.py | 606 | `MATCH (n:__Entity__) DETACH DELETE n` (reset write) | stats/dump | Task 12: provenance nodes survive entity reset by default; add separate provenance wipe path |
| backend/knowledge/admin.py | 619 | `MATCH (n:__Entity__) RETURN count(n)` (probe_neo4j) | stats/dump | Task 10: health probe count is user-facing only; note that in returned dict key |
| backend/knowledge/curation/centrality.py | 21 | `'__Entity__'` (GDS project node label) | user-facing | No change -- already correct; GDS projection operates on user-facing entity graph |
| backend/knowledge/curation/community.py | 21 | `'__Entity__'` (GDS project node label) | user-facing | No change -- already correct; Louvain community detection on user-facing entity graph |
| backend/knowledge/curation/confidence_decay.py | 24 | `MATCH (e:__Entity__)` (_FETCH_QUERY) | user-facing | No change -- already correct; decay applies to user-facing entities only |
| backend/knowledge/curation/confidence_decay.py | 33 | `MATCH (e:__Entity__ {id: $id})` (_UPDATE_CONFIDENCE_QUERY) | user-facing | No change -- already correct; confidence update targets specific entity |
| backend/knowledge/curation/confidence_decay.py | 38 | `MATCH (e:__Entity__ {id: $id})` (_ARCHIVE_QUERY) | user-facing | No change -- already correct; archive status set on specific entity |
| backend/knowledge/curation/conflict_resolver.py | 187 | `MATCH (s:__Entity__ {id: $source})-[r]->(t:__Entity__)` | user-facing | No change -- already correct; conflict check is between user-facing entity nodes |
| backend/knowledge/curation/conflict_resolver.py | 208 | `MATCH (s:__Entity__ {id: $source})-[r]->(t:__Entity__ ...)` | user-facing | No change -- already correct; contradiction check is between user-facing entity nodes |
| backend/knowledge/curation/conflict_resolver.py | 228 | `MATCH (s:__Entity__ {id: $source})-[r]->(t:__Entity__ ...)` | user-facing | No change -- already correct; progression check is between user-facing entity nodes |
| backend/knowledge/curation/deduplication.py | 102 | `MATCH (e:__Entity__) WHERE toLower(e.id) = $entity_id` | user-facing | No change -- already correct; exact-ID tier of 3-tier entity matching |
| backend/knowledge/curation/deduplication.py | 115 | `MATCH (e:__Entity__) WHERE $entity_id IN [a IN e.aliases...]` | user-facing | No change -- already correct; alias tier of 3-tier entity matching |
| backend/knowledge/curation/embedding_maintenance.py | 20 | `MATCH (e:__Entity__)` (_STALE_EMBEDDINGS_QUERY) | user-facing | No change -- already correct; stale-embedding scan targets active user-facing entities |
| backend/knowledge/curation/embedding_maintenance.py | 30 | `MATCH (e:__Entity__ {id: $id})` (_UPDATE_EMBEDDING_QUERY) | user-facing | No change -- already correct; embedding update targets specific entity |
| backend/knowledge/curation/graph_writer.py | 190 | `MERGE (ctx:__Entity__:ConversationContext ...)` | audit-trail | Task 3: rewrite to `:__Provenance__:ConversationContext`; remove `:__Entity__` label |
| backend/knowledge/curation/graph_writer.py | 216 | `MERGE (e:__Entity__ {id: $entity_id})` (_upsert_entity) | user-facing | No change -- already correct; creates/updates user-facing knowledge entity |
| backend/knowledge/curation/graph_writer.py | 261 | `MATCH (s:__Entity__ {{id: $source}})` (_upsert_relationship) | user-facing | No change -- already correct; both endpoints are user-facing entities |
| backend/knowledge/curation/graph_writer.py | 262 | `MATCH (t:__Entity__ {{id: $target}})` (_upsert_relationship) | user-facing | No change -- already correct; both endpoints are user-facing entities |
| backend/knowledge/curation/graph_writer.py | 288 | `MATCH (e:__Entity__ {id: $entity_id})` (_create_provenance_edge) | cross-layer | Task 5: entity side keeps `:__Entity__`; ctx side must become `:__Provenance__:ConversationContext` |
| backend/knowledge/curation/graph_writer.py | 341 | `MATCH (e:__Entity__ {id: $entity_id})` (_create_document_provenance SOURCED_FROM) | cross-layer | Task 6: entity side keeps `:__Entity__`; ExternalSource target becomes `:__Provenance__:ExternalSource` |
| backend/knowledge/curation/graph_writer.py | 360 | `MATCH (e:__Entity__ {{id: $entity_id}})` (_create_document_provenance chunk) | cross-layer | Task 6: entity side keeps `:__Entity__`; VectorChunk target becomes `:__Provenance__:VectorChunk` |
| backend/knowledge/curation/graph_writer.py | 380 | `MATCH (s:__Entity__)-[r:...]->(t:__Entity__ ...)` (_apply_supersession) | user-facing | No change -- already correct; supersession marks relationship between two user-facing entities |
| backend/knowledge/curation/graph_writer.py | 425 | `MERGE (le:__Entity__:LearningEvent {id: $learning_id})` | audit-trail | Task 4: rewrite to `:__Provenance__:LearningEvent`; remove `:__Entity__` label |
| backend/knowledge/curation/graph_writer.py | 432 | `MATCH (target:__Entity__ {id: $about_target})` (_create_learning_event) | cross-layer | Task 5: ABOUT edge source becomes `:__Provenance__:LearningEvent`; target keeps `:__Entity__` |
| backend/knowledge/curation/graph_writer.py | 461 | `MERGE (le:__Entity__:LearningEvent {id: $learning_id})` | audit-trail | Task 4: rewrite to `:__Provenance__:LearningEvent`; remove `:__Entity__` label |
| backend/knowledge/curation/graph_writer.py | 467 | `MATCH (target:__Entity__ {id: $entity_id})` (_create_new_fact_learning_event) | cross-layer | Task 5: ABOUT edge source becomes `:__Provenance__:LearningEvent`; target keeps `:__Entity__` |
| backend/knowledge/curation/health.py | 36 | `MATCH (e:__Entity__)` (_COUNT_QUERY) | user-facing | No change -- already correct; health scorer counts active user-facing entities |
| backend/knowledge/curation/health.py | 42 | `MATCH (e:__Entity__)` (_FRESHNESS_QUERY) | user-facing | No change -- already correct; freshness check on user-facing entities only |
| backend/knowledge/curation/health.py | 49 | `MATCH (e:__Entity__)` (_CONFIDENCE_QUERY) | user-facing | No change -- already correct; avg confidence over user-facing entities |
| backend/knowledge/curation/health.py | 55 | `MATCH (e:__Entity__)` (_CONNECTIVITY_QUERY) | user-facing | No change -- already correct; connectivity metric on user-facing entity graph |
| backend/knowledge/curation/health.py | 63 | `MATCH (e:__Entity__)` (_CONSISTENCY_QUERY) | user-facing | No change -- already correct; type distribution consistency on user-facing entities |
| backend/knowledge/curation/health.py | 69 | `MATCH (e:__Entity__)` (_SELF_MODEL_QUERY) | user-facing | No change -- already correct; internal domain entities are user-facing (MistIdentity graph) |
| backend/knowledge/curation/orphan_detector.py | 50 | `MATCH (e:__Entity__) WHERE e.status = 'active'` (count) | user-facing | No change -- already correct; orphan scan covers user-facing entities; provenance nodes are legitimately relationship-sparse |
| backend/knowledge/curation/orphan_detector.py | 66 | `MATCH (e:__Entity__) WHERE e.status = 'active'` (find orphans) | user-facing | No change -- already correct; after label split, query naturally excludes provenance nodes |
| backend/knowledge/curation/orphan_detector.py | 85 | `MATCH (e:__Entity__) WHERE e.id IN $ids` (archive) | user-facing | No change -- already correct; archives orphaned user-facing entities only |
| backend/knowledge/curation/skill_derivation.py | 142 | `MATCH (e:__Entity__ {id: $skill_id, entity_type: 'Skill'})` | user-facing | No change -- already correct; Skill is a user-facing entity type |
| backend/knowledge/curation/skill_derivation.py | 158 | `MERGE (e:__Entity__ {id: $skill_id})` (_create_skill) | user-facing | No change -- already correct; creates Skill entity in user-facing graph |
| backend/knowledge/curation/skill_derivation.py | 171 | `MERGE (u:__Entity__ {id: 'user'})` (_create_skill KNOWS edge) | user-facing | No change -- already correct; User entity is user-facing; KNOWS is a user-entity relationship |
| backend/knowledge/curation/skill_derivation.py | 185 | `MATCH (e:__Entity__ {id: $skill_id, entity_type: 'Skill'})` | user-facing | No change -- already correct; updates proficiency on Skill entity |
| backend/knowledge/curation/skill_derivation.py | 206 | `MATCH (e:__Entity__ {id: $cap_id, entity_type: 'MistCapability'})` | user-facing | No change -- already correct; reads MistCapability which is user-facing internal entity |
| backend/knowledge/curation/skill_derivation.py | 213 | `MATCH (e:__Entity__ {id: $cap_id, entity_type: 'MistCapability'})` | user-facing | No change -- already correct; updates proficiency on MistCapability |
| backend/knowledge/curation/skill_derivation.py | 220 | `MERGE (e:__Entity__ {id: $cap_id})` (_ensure_capability create) | user-facing | No change -- already correct; creates MistCapability in user-facing graph |
| backend/knowledge/curation/staleness.py | 60 | `MATCH (n:__Entity__)` (staleness scan) | user-facing | No change -- already correct; staleness tiers apply to active user-facing entities |
| backend/knowledge/extraction/internal_derivation.py | 169 | `MATCH (m:MistIdentity)-[r]->(e:__Entity__)` | user-facing | No change -- already correct; fetches internal entities linked from MistIdentity hub |
| backend/knowledge/extraction/internal_derivation.py | 241 | `MERGE (e:__Entity__ {{id: $entity_id}})` (_apply_operation) | user-facing | No change -- already correct; creates internal entity (MistTrait/Capability etc.) in user-facing graph |
| backend/knowledge/extraction/internal_derivation.py | 265 | `MATCH (e:__Entity__ {{id: $entity_id}})` (UPDATE op) | user-facing | No change -- already correct; UPDATE op sets properties on existing internal entity |
| backend/knowledge/extraction/internal_derivation.py | 276 | `MATCH (e:__Entity__ {id: $entity_id})` (DEPRECATE op) | user-facing | No change -- already correct; DEPRECATE op marks internal entity status |
| backend/knowledge/extraction/normalizer.py | 236 | `MATCH (e:__Entity__) WHERE toLower(e.id) = $canonical_id` | user-facing | No change -- already correct; exact-ID tier of normalizer entity lookup |
| backend/knowledge/extraction/normalizer.py | 250 | `MATCH (e:__Entity__) WHERE $canonical_id IN [a IN e.aliases...]` | user-facing | No change -- already correct; alias tier of normalizer entity lookup |
| backend/knowledge/regeneration/graph_regenerator.py | 321 | `Delete all __Entity__ nodes` (docstring) | stats/dump | Task 12: docstring must clarify that provenance nodes are NOT deleted by default |
| backend/knowledge/regeneration/graph_regenerator.py | 331 | `MATCH (e:__Entity__) DETACH DELETE e` (_delete_graph_entities) | stats/dump | Task 12: delete should remain entity-only; provenance nodes are preserved; align docstring |
| backend/knowledge/regeneration/graph_regenerator.py | 339 | `MATCH (e:__Entity__) RETURN count(e)` (verify deletion) | stats/dump | Task 12: post-delete verification counts entity-only, which is correct after label split |
| backend/knowledge/regeneration/graph_regenerator.py | 361 | `(u:Utterance)-[:HAS_ENTITY]->(e:__Entity__)` (_delete_conversation_entities) | cross-layer | Task 7: traversal from Utterance through HAS_ENTITY to `:__Entity__`; entity label is correct; Utterance is a legacy non-provenance node -- review if Utterance moves under `:__Provenance__` in later tasks |
| backend/knowledge/scripts/README.md | 93 | `MATCH (e:__Entity__)` (README count query) | stats/dump | Task 11: update README example queries to show both entity and provenance counts |
| backend/knowledge/scripts/README.md | 97 | `(c:DocumentChunk)<-[:EXTRACTED_FROM]-(e:__Entity__)` | cross-layer | Task 11: update README to show `(c:__Provenance__:DocumentChunk)` form after migration |
| backend/knowledge/storage/graph_store.py | 75 | `FOR (e:__Entity__) REQUIRE e.id IS UNIQUE` | schema | Task 2: add parallel `FOR (p:__Provenance__) REQUIRE p.id IS UNIQUE` |
| backend/knowledge/storage/graph_store.py | 93 | `FOR (e:__Entity__) ON (e.entity_type)` | schema | Task 2: add parallel `FOR (p:__Provenance__) ON (p.entity_type)` |
| backend/knowledge/storage/graph_store.py | 115 | `FOR (e:__Entity__) ON e.embedding` (vector index) | schema | Task 2: add parallel vector index for `:__Provenance__` nodes that carry embeddings if needed |
| backend/knowledge/storage/graph_store.py | 499 | `List of '__Entity__' node IDs to link` (docstring) | user-facing | No change -- already correct; docstring describes entity-side of provenance link API |
| backend/knowledge/storage/graph_store.py | 520 | `MATCH (e:__Entity__ {id: eid})` (SOURCED_FROM batch) | cross-layer | Task 6: entity side keeps `:__Entity__`; ExternalSource target becomes `:__Provenance__:ExternalSource` after Task 3 |
| backend/knowledge/storage/graph_store.py | 536 | `MATCH (e:__Entity__ {id: eid})` (REFERENCES batch) | cross-layer | Task 6: entity side keeps `:__Entity__`; VectorChunk target becomes `:__Provenance__:VectorChunk` after Task 3 |
| backend/knowledge/storage/graph_store.py | 638 | `MERGE (e:__Entity__ {id: $node_id})` (_store_validated_node) | user-facing | No change -- already correct; writes extracted user-facing entity from dict format |
| backend/knowledge/storage/graph_store.py | 684 | `MATCH (s:__Entity__ {{id: $source}})` (_store_validated_relationship) | user-facing | No change -- already correct; both endpoints are user-facing entities |
| backend/knowledge/storage/graph_store.py | 685 | `MATCH (t:__Entity__ {{id: $target}})` (_store_validated_relationship) | user-facing | No change -- already correct; both endpoints are user-facing entities |
| backend/knowledge/storage/graph_store.py | 782 | `MERGE (e:__Entity__ {{id: $node_id}})` (_store_node utterance branch) | user-facing | No change -- already correct; extracted entity from Utterance source |
| backend/knowledge/storage/graph_store.py | 792 | `MERGE (e:__Entity__ {{id: $node_id}})` (_store_node chunk branch) | user-facing | No change -- already correct; extracted entity from DocumentChunk source |
| backend/knowledge/storage/graph_store.py | 869 | `MATCH (source:__Entity__ {{id: $source_id}})` (_store_relationship) | user-facing | No change -- already correct; both endpoints are user-facing entities |
| backend/knowledge/storage/graph_store.py | 870 | `MATCH (target:__Entity__ {{id: $target_id}})` (_store_relationship) | user-facing | No change -- already correct; both endpoints are user-facing entities |
| backend/knowledge/storage/graph_store.py | 893 | `(u:Utterance)-[:HAS_ENTITY]->(e:__Entity__)` (get_entities_for_conversation) | cross-layer | Task 7: traversal starts at ConversationEvent through Utterance to `:__Entity__`; entity label stays; if Utterance becomes `:__Provenance__` later this becomes a provenance-to-entity traversal needing explicit label on Utterance side |
| backend/knowledge/storage/graph_store.py | 1073 | `(start:__Entity__ {{id: $entity_id}})-[*1..N]-(related:__Entity__)` | user-facing | No change -- already correct; graph-hop traversal is confined to user-facing entity subgraph |
| backend/knowledge/storage/graph_store.py | 1112 | `MATCH (user:__Entity__ {{id: $user_id}})-[r]-(entity:__Entity__)` | user-facing | No change -- already correct; user-entity relationship query within user-facing graph |
| backend/knowledge/storage/graph_store.py | 1160 | `MATCH (user:__Entity__ {{id: $user_id}})-[r]->(entity:__Entity__)` | user-facing | No change -- already correct; all-user-relationships query within user-facing graph |
| backend/knowledge/storage/graph_store.py | 1186 | `MERGE (m:__Entity__:MistIdentity {id: 'mist-identity'})` | user-facing | No change -- already correct; MistIdentity singleton is the root of the self-model in user-facing graph |
