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


class RestoreTargetError(BackupError):
    """Raised when a restore target is live state, or has not identified itself.

    Separate from `BackupDestinationError` because the two describe opposite
    directions of travel and their refusals are not interchangeable. A
    destination is where bytes are WRITTEN and must outlive the source; a target
    is what a restore OVERWRITES, and `<repo>/dev-state` is a legitimate target
    while being an illegitimate destination.
    """


class RestoreConfirmationError(BackupError):
    """Raised when the typed confirmation token is absent or is not the resolved target."""


class RestoreAbortedError(BackupError):
    """Raised when a pre-restore backup of the target failed, so nothing was overwritten.

    Its own type because it is the one refusal that happens AFTER the operator
    has passed every guard, and the operator's next action differs: the target
    and the artifact are both fine, and what failed is the safety net.
    """
