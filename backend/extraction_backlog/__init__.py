"""Durable, ordered extraction backlog over the event log (T2a, T2b).

The backend keeps the log, the backlog, Gates 0/2/3, the extraction cache,
Stages 3-6, curation and every graph write. The extraction service
(`backend/extraction_service/`) runs Stages 1.5, 2 and 9 and returns the raw
Stage-2 record. `ExtractionDispatcher` connects the two: one consumer, one job
in flight, turns applied strictly in log order.

Modules:
    inference   -- `ExtractionInference` protocol, `RemoteExtractionInference`
    errors      -- typed inference failures
    store       -- `BacklogStore`, the backlog as a view over log + cache
    dispatcher  -- `ExtractionDispatcher`
    settings    -- `DispatcherSettings` (`MIST_EXTRACTION_*`)
    status      -- the one `ExtractionStatus` producer (WS, HTTP, /health)
    telemetry   -- the process-wide unrecorded-turn counter
    cutover     -- epoch cutover: candidate fill, check, promote (T2b)
    admin       -- `python -m backend.extraction_backlog.admin`

The package-level names below are resolved lazily (PEP 562). `telemetry` is
imported from `ModelManager.generate_llm_response`'s no-knowledge fallback,
which runs precisely when the knowledge subsystem failed to come up; an eager
`from .dispatcher import ...` here would pull the extraction pipeline into that
branch and could fail the turn for the same reason knowledge did.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "BacklogStore": ".store",
    "DispatcherSettings": ".settings",
    "ExtractionDispatcher": ".dispatcher",
    "ExtractionInference": ".inference",
    "ExtractionInferenceError": ".errors",
    "InferenceResponseInvalidError": ".errors",
    "InferenceServiceError": ".errors",
    "InferenceUnreachableError": ".errors",
    "RemoteExtractionInference": ".inference",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module, __name__), name)
