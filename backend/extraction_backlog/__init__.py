"""Durable, ordered extraction backlog over the event log (T2a).

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
    admin       -- `python -m backend.extraction_backlog.admin`
"""

from .dispatcher import ExtractionDispatcher
from .errors import (
    ExtractionInferenceError,
    InferenceResponseInvalidError,
    InferenceServiceError,
    InferenceUnreachableError,
)
from .inference import ExtractionInference, RemoteExtractionInference
from .settings import DispatcherSettings
from .store import BacklogStore

__all__ = [
    "BacklogStore",
    "DispatcherSettings",
    "ExtractionDispatcher",
    "ExtractionInference",
    "ExtractionInferenceError",
    "InferenceResponseInvalidError",
    "InferenceServiceError",
    "InferenceUnreachableError",
    "RemoteExtractionInference",
]
