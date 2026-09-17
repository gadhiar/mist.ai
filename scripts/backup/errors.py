"""Failure types for the disaster-recovery backup package.

Deliberately NOT `scripts.hydration.manifest.HydrationError`, because the two
tools have opposite trust models and a shared exception type would invite a
shared check. A hydration artifact is refused when its PRODUCER STAMPS drift
from the tree (`grep -n "def assert_fresh" scripts/hydration/manifest.py` ->
:242); a backup artifact is refused only when its LAYOUT cannot be parsed, which
is the same rule `load_artifact` applies to the graph leg
(`grep -n "The version check" backend/knowledge/graph_artifact.py` -> :411).

All three derive from `MistError` (`grep -n "class MistError" backend/errors.py`)
so `scripts/mist_admin.py` prints them as a refusal rather than a traceback:
its `main` catches `(MistError, EvalIsolationError)`
(`grep -n "except (MistError, EvalIsolationError)" scripts/mist_admin.py` ->
:3113).
"""

from __future__ import annotations

from backend.errors import MistError


class BackupError(MistError):
    """Raised when a backup cannot be produced, read, or trusted."""


class BackupDestinationError(BackupError):
    """Raised when a destination is unset, or resolves onto state it must outlive."""


class BackupManifestError(BackupError):
    """Raised when an artifact has no manifest, or a manifest layout this code cannot read."""
