---
type: mist-session
session_id: test-baseline-beta-2026-01-02
date: '2026-01-02'
turn_count: 1
participants:
- test-user
- mist
authored_by: pipeline
status: completed
append_sentinel_offset: 0
related_entities:
- test-user
- PostgreSQL
- Distributed Systems
ontology_version: 1.1.0
extraction_version: 2026-05-06-r1
model_hash: null
tags: []
---


## Turn 1

**User:** I'm using PostgreSQL with the FastAPI service. Should I use SQLAlchemy or a thinner driver?

**MIST:** For a backend service with multiple resources and joins across them, SQLAlchemy (Core or ORM) earns its weight. You get migrations via Alembic, a unified query language across read and write paths, and type-checkable models.

If the service is read-heavy with few joins and you want minimal abstraction, a thinner driver like asyncpg directly works fine -- you write SQL, you get rows. The cost is hand-rolling migrations and losing the type integration.

For a service evolving over months with a growing schema, SQLAlchemy. For a single-table or read-mostly cache layer, asyncpg.
