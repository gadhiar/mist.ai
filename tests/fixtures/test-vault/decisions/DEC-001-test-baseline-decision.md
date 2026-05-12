---
type: mist-decision
id: DEC-001
date: '2026-01-01'
status: accepted
title: "Test Baseline: Use Python for backend services"
authored_by: pipeline
tags:
- python
- backend
---

# DEC-001: Use Python for backend services

## Status
Accepted -- 2026-01-01.

## Context
The test baseline project needs a backend language choice. Options
considered: Python, Go, Rust.

## Decision
Use Python with FastAPI for the backend service layer.

## Rationale
- Strong ecosystem for API services
- Type hints via Pydantic give compile-time-like safety at the HTTP boundary
- FastAPI's async support handles concurrent requests cleanly
- Team expertise concentrated in Python

## Consequences
- Async / await discipline required throughout the request path
- GIL constraints on CPU-bound work -- offload to background workers if it
  surfaces
- Dependency on Python 3.11+ for modern typing syntax
