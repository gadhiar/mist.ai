"""Backend <-> extraction-service wire contract.

This package is a LEAF: it imports only the standard library and
`pydantic`, nothing from `backend.*` and no other third-party package. The
extraction service (T1), the backend's backlog dispatcher (T2), the
streaming path (T3), and the context budget work (T4) all import these
names directly -- keep the shapes and names stable.
"""

from .models import (
    CONTRACT_MAJOR,
    CONTRACT_VERSION,
    ERROR_DEFAULT_RETRYABLE,
    ERROR_HTTP_STATUS,
    CutoverStatus,
    DerivationInput,
    DerivationOut,
    ErrorCode,
    ErrorDetail,
    ErrorEnvelope,
    ExpectStamps,
    ExtractionPayload,
    ExtractionStatus,
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    HistoryMessage,
    InfoResponse,
    LastJob,
    ResultStamps,
    ScopeLabel,
    ScopeOut,
    ServiceStatus,
    TimingsMs,
    error_envelope,
    is_compatible,
)

__all__ = [
    "CONTRACT_MAJOR",
    "CONTRACT_VERSION",
    "ERROR_DEFAULT_RETRYABLE",
    "ERROR_HTTP_STATUS",
    "CutoverStatus",
    "DerivationInput",
    "DerivationOut",
    "ErrorCode",
    "ErrorDetail",
    "ErrorEnvelope",
    "ExpectStamps",
    "ExtractRequest",
    "ExtractResponse",
    "ExtractionPayload",
    "ExtractionStatus",
    "HealthResponse",
    "HistoryMessage",
    "InfoResponse",
    "LastJob",
    "ResultStamps",
    "ScopeLabel",
    "ScopeOut",
    "ServiceStatus",
    "TimingsMs",
    "error_envelope",
    "is_compatible",
]
