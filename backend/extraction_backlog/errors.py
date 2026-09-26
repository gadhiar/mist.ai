"""Typed failures of an `ExtractionInference` call.

Every failure carries the two facts the dispatcher branches on: the contract
`ErrorCode` (None when the failure never reached the service's error taxonomy,
e.g. a refused connection) and whether it is `retryable`. The three subclasses
separate the three things that can go wrong, because they are treated
differently:

- `InferenceUnreachableError` -- the service could not be reached or is not
  serving yet. Not the job's fault: no attempt is counted.
- `InferenceServiceError` -- the service answered with a contract
  `ErrorEnvelope`. The dispatcher branches on `code`.
- `InferenceResponseInvalidError` -- the service answered, but the body is not
  a valid contract object. Job-attributable: counts toward dead-lettering.

All three subclass `backend.errors.ExtractionError` so existing
`except ExtractionError` handlers see them.
"""

from __future__ import annotations

from backend.errors import ExtractionError
from backend.extraction_contract.models import ErrorCode


class ExtractionInferenceError(ExtractionError):
    """Base class for a failed `ExtractionInference` call.

    Attributes:
        code: The contract error code, or None when the failure is outside the
            service's taxonomy.
        retryable: Whether retrying the same request unchanged may succeed.
        http_status: The HTTP status the service answered with, when it did.
    """

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | None,
        retryable: bool,
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.http_status = http_status


class InferenceUnreachableError(ExtractionInferenceError):
    """The service did not answer (transport failure or gateway error)."""

    def __init__(self, message: str, *, http_status: int | None = None) -> None:
        super().__init__(message, code=None, retryable=True, http_status=http_status)


class InferenceServiceError(ExtractionInferenceError):
    """The service answered with a contract `ErrorEnvelope`."""

    def __init__(
        self, message: str, *, code: ErrorCode, retryable: bool, http_status: int | None
    ) -> None:
        super().__init__(message, code=code, retryable=retryable, http_status=http_status)


class InferenceResponseInvalidError(ExtractionInferenceError):
    """The service answered with a body that is not a valid contract object."""

    def __init__(self, message: str, *, http_status: int | None = None) -> None:
        super().__init__(message, code=None, retryable=True, http_status=http_status)
