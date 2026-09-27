"""The one `ExtractionStatus` producer behind the WebSocket push, HTTP and `/health`.

Three surfaces read the same object so they cannot disagree:

- the `extraction_status` WebSocket message (ADR-017 additive), pushed by
  `backend.server.extraction_status_loop` on a timer and on every dispatcher
  state transition;
- `GET /extraction/status`;
- the `extraction` block of `/health` (`health_block`).

It never raises. With no dispatcher, or a dispatcher in `off` mode, the status
is state `disabled` with zero counts; `unrecorded_turns` is filled in either
way, from `telemetry`. A storage failure while the dispatcher computes its
snapshot is logged and reported as the dispatcher's current state with zero
counts, because a status endpoint that errors hides the state an operator is
looking for.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

from backend.errors import MistError
from backend.extraction_contract.models import ExtractionStatus, ServiceStatus

from . import telemetry
from .dispatcher import STATE_DISABLED, ExtractionDispatcher

logger = logging.getLogger(__name__)

_UNKNOWN_SERVICE = ServiceStatus(
    reachable=False,
    location_label=None,
    model_id=None,
    extraction_version=None,
    contract_version=None,
    last_health_ms=None,
)


def _empty(state: str) -> ExtractionStatus:
    return ExtractionStatus(
        state=state,  # type: ignore[arg-type]
        backlog_depth=0,
        apply_pending=0,
        dead_lettered=0,
        oldest_pending_age_ms=None,
        unrecorded_turns=telemetry.unrecorded_turns(),
        legacy_unextracted=0,
        service=_UNKNOWN_SERVICE,
        last_job=None,
        cutover=None,
    )


def extraction_status(dispatcher: ExtractionDispatcher | None) -> ExtractionStatus:
    """The current `ExtractionStatus`, for any dispatcher or none.

    Args:
        dispatcher: The server's dispatcher, or None when knowledge integration
            is off or the dispatcher failed to start.

    Returns:
        The status. `unrecorded_turns` is always the process-wide count.
    """
    if dispatcher is None or dispatcher.mode == "off":
        return _empty(STATE_DISABLED)
    try:
        snapshot = dispatcher.snapshot()
    except (MistError, sqlite3.Error, OSError) as exc:
        logger.warning("Extraction status snapshot failed; reporting zero counts: %s", exc)
        return _empty(dispatcher.state)
    # `snapshot()` reads `telemetry.unrecorded_turns()` itself.
    return snapshot


def health_block(status: ExtractionStatus) -> dict[str, Any]:
    """The `extraction` block `/health` carries: a subset of the full status."""
    return {
        "state": status.state,
        "backlog_depth": status.backlog_depth,
        "dead_lettered": status.dead_lettered,
        "unrecorded_turns": status.unrecorded_turns,
        "service": {"reachable": status.service.reachable},
    }
