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


class RestorePreflightError(BackupError):
    """Raised when an artifact fails a check made BEFORE the target is touched.

    Its own type because it is what lets the CLI say "nothing was written"
    truthfully. Digest mismatches, an unreadable graph leg, a `format_version`
    this build does not read and a relationship endpoint that cannot be
    re-anchored are all discovered while the target is still intact, and they
    exit 2 (refused) rather than 1 (failed part way).

    It also wraps `GraphArtifactError`, which is a `RuntimeError` and not a
    `MistError` (`grep -n "class GraphArtifactError"
    backend/knowledge/graph_artifact.py` -> :95), so it would otherwise escape
    every `except MistError` arm in this package as a raw traceback.
    """


class RestoreTargetStateError(RestorePreflightError):
    """Raised when the TARGET, not the artifact, fails a check made before it is touched.

    A subclass of `RestorePreflightError` rather than a sibling, and that is the
    whole point of the choice: `scripts/backup/restore.py`'s `main` already
    catches `RestorePreflightError` in its exit-2 tuple
    (`grep -n "Exit 2 is a PROMISE" scripts/backup/restore.py`, and the tuple
    directly under it), so a new refusal reaches exit 2 without that tuple being
    edited and without any guard being widened. Every check that raises this reads the target and
    writes nothing, so the exit-2 promise -- "nothing in the target was
    overwritten" -- still holds when it is raised.

    Distinct from its parent because the operator's next action differs. A
    `RestorePreflightError` means reach for a different artifact; this means fix
    the target: free disk space, start the graph, or deal by hand with the
    `restore.in-progress.json` a previous run left behind.
    """


class RestoreAbortedError(BackupError):
    """Raised when a pre-restore backup of the target failed, so nothing was overwritten.

    Its own type because it is the one refusal that happens AFTER the operator
    has passed every guard, and the operator's next action differs: the target
    and the artifact are both fine, and what failed is the safety net.
    """
