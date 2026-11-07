---
type: mist-session
session_id: dev-fixture-evening
title: API routes for the Seedling Planner beta
date: '2026-01-02'
authored_by: mist
status: completed
related_entities:
- dev-user
- goal-ship-planner-beta
- skill-api-design
- tech-python
ontology_version: 1.4.0
extraction_version: 2026-06-14-r5
model_hash: null
tags:
- dev-fixture
---
# Session: API routes for the Seedling Planner beta

**Date:** 2026-01-02

### What Was Accomplished
The Dev User drafted the Python routes for the beta: list plantings, add a planting and mark a
planting harvested. MIST suggested naming each route after the record it changes.

### Decisions Made
- One route per action on a planting record, each named for the record.

### Next Actions
- Write a test for each route before the beta ships.

### Context for Next Session
The table from the earlier session is in place; the routes read and write it.
