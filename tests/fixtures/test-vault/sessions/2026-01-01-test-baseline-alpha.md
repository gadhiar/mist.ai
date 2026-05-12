---
type: mist-session
session_id: test-baseline-alpha-2026-01-01
date: '2026-01-01'
turn_count: 1
participants:
- test-user
- mist
authored_by: pipeline
status: completed
append_sentinel_offset: 0
related_entities:
- test-user
- Python
- FastAPI
ontology_version: 1.1.0
extraction_version: 2026-05-06-r1
model_hash: null
tags: []
---


## Turn 1

**User:** I'm setting up a new FastAPI service in Python. What's the recommended structure for the project layout?

**MIST:** For a FastAPI service in Python, the conventional layout starts with a package directory holding your application module (typically `app/` or your project name), a tests directory, and a single `main.py` or `app.py` entry point.

Inside the application package, split by concern: a `routers/` directory for HTTP route handlers grouped by resource, a `models/` directory for Pydantic schemas, a `services/` directory for business logic separated from the HTTP layer, and a `db/` or `database/` module for persistence concerns.

This separation keeps the HTTP layer thin -- routers translate requests to service calls, services hold the actual logic, and persistence stays isolated from business rules. Tests at each layer become straightforward.
