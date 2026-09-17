"""Make a restore target identify itself, and make the operator type where it lands.

MIS-140 T2. Restore is the irreversible half of this package: it detach-deletes
the target graph (`grep -n "deleted = clear_graph" backend/knowledge/admin.py` ->
:1162) and replaces the target's store files. A dump aimed at the wrong place
wastes a disk; a restore aimed at the wrong place destroys the thing it was run
to save.

TWO GATES LIVE HERE, AND THEY FAIL IN DIFFERENT DIRECTIONS ON PURPOSE.

1. THE HANDSHAKE. `scripts/hydration/target.py` states the principle this module
   reuses, in its module docstring: "A denylist can only refuse what it
   enumerates. The handshake inverts that". There, the target is a running
   backend and it answers over HTTP with `hydration_isolation: true`
   (`grep -n "hydration_isolation" scripts/hydration/target.py` -> :20,118,126).
   Here the target is a DIRECTORY, which answers nothing, so the same inversion
   is a marker file the operator places by hand:

       touch <target-root>/MIST_RESTORE_TARGET

   The live state root will never carry that file, however it is spelled,
   symlinked, bind-mounted or `..`-smuggled, because nothing in this repository
   creates it: `grep -rn "MIST_RESTORE_TARGET" .` returns only
   `scripts/backup/target.py`, `scripts/backup/restore.py` and
   `scripts/backup/README.md` -- three files that NAME it, and no code anywhere
   that writes it. Placing it is a deliberate act on a directory the operator is
   looking at.

   The handshake COMBINES WITH `assert_isolated_root` rather than replacing it.
   The denylist arm runs first, so a target that IS live is refused for being
   live -- the accurate message -- rather than for being unmarked, which would
   invite the operator to fix it by creating the marker.

2. THE TYPED CONFIRMATION. The operator must type the RESOLVED absolute path
   back. The comparison is against the resolved path and never against the
   string they passed, so `--target-root ./dev-state --confirm-target
   ./dev-state` is REFUSED: retyping a relative path proves nothing about where
   it lands, and "where it lands" is the entire question. The refusal prints the
   resolved path, so the second attempt is a copy-paste of a string the operator
   has now read.

NO FLAG DISABLES EITHER GATE, and none should be added. A `--force` here is a
flag whose only user is someone in a hurry at 3am, which is the exact caller
these gates exist to stop.
"""

from __future__ import annotations

from pathlib import Path

from backend.knowledge.eval_isolation import IsolatedRootError, assert_isolated_root

from .errors import RestoreConfirmationError, RestoreTargetError

# The marker file an operator places in a restore target. Named in the refusal
# text, in `scripts/backup/README.md`, and nowhere else in this repository --
# see the module docstring for the grep that establishes there is no writer.
RESTORE_MARKER_FILENAME = "MIST_RESTORE_TARGET"

_HOW_TO_MARK = (
    f"If this really is the target, place the marker in it by hand -- "
    f"`touch <target-root>/{RESTORE_MARKER_FILENAME}` -- and re-run. Creating "
    "that file is the act of consent; nothing in MIST.AI creates it for you, "
    "which is why the live state root can never carry it."
)


def restore_marker_path(root: Path | str) -> Path:
    """Return where the handshake marker must sit for `root`.

    Exposed so a refusal, a test and the runbook all name the same path rather
    than each joining the filename themselves.
    """
    return Path(root) / RESTORE_MARKER_FILENAME


def assert_restore_target_root(root: Path | str, *, purpose: str = "restore") -> Path:
    """Resolve a restore target, refusing live state and any unmarked directory.

    Args:
        root: The operator's `--target-root`. Resolved before any check, so a
            symlink or a `..` segment cannot smuggle a path past the guards.
        purpose: Named in the refusal so the operator knows which tool refused.

    Returns:
        The resolved target root. Nothing is created, and the marker is not
        consumed -- a target stays marked across rehearsals.

    Raises:
        RestoreTargetError: When the target is, sits under, or contains live
            state; when it does not exist or is not a directory; or when it
            carries no handshake marker.
    """
    resolved = Path(root).expanduser().resolve()

    # Denylist arm FIRST. It is pure and needs no filesystem, and it produces
    # the accurate message for the worst case: a target that IS live is refused
    # for being live rather than for being unmarked.
    try:
        assert_isolated_root(resolved, purpose=purpose)
    except IsolatedRootError as exc:
        # Re-raised, not re-implemented. The shared guard's own text names the
        # path and the live directory it collided with; this adds only the
        # sentence that belongs to a restore.
        raise RestoreTargetError(
            f"{exc} (That wording is the shared live-state guard's, which this "
            "restore reuses unchanged.) A restore REPLACES what it is pointed at, "
            "so there is no --force for this."
        ) from exc

    if not resolved.is_dir():
        raise RestoreTargetError(
            f"refusing {purpose} target {resolved}: it is not an existing directory. "
            "The target is created and marked by the operator before a restore, not "
            f"by this tool -- a path this tool would create is a path nobody looked "
            f"at. {_HOW_TO_MARK}"
        )

    marker = restore_marker_path(resolved)
    if not marker.is_file():
        raise RestoreTargetError(
            f"refusing {purpose} target {resolved}: it does not carry the handshake "
            f"marker {marker}. This tool does not reason about whether a path looks "
            "live; it requires the target to identify itself, because a denylist can "
            f"only refuse the spellings someone thought of. {_HOW_TO_MARK}"
        )

    return resolved


def assert_target_confirmed(resolved_root: Path, token: str | None) -> None:
    """Refuse unless `token` is the resolved target path, typed back.

    The comparison is byte-for-byte against `str(resolved_root)` after stripping
    surrounding whitespace, and that is the whole design:

    - Whitespace is stripped because a token arrives through a shell and a
      copy-paste picks up a trailing space or newline. That is a typo, not a
      different target.
    - Case is NOT normalised and neither is separator style. On Windows the
      resolved path is what `Path.resolve()` printed in the refusal, so a
      copy-paste matches exactly; anything else is the operator typing from
      memory, which is what this gate exists to interrupt.
    - The token is NEVER resolved before comparison. Resolving it would accept
      `./dev-state` as confirmation of `./dev-state` -- retyping the string the
      operator already typed, which demonstrates nothing about where it lands.

    Args:
        resolved_root: The output of `assert_restore_target_root`. Passing an
            unresolved path here would defeat the gate, which is why this takes
            the guard's return value rather than the operator's argument.
        token: What the operator typed, or None when they passed nothing.

    Raises:
        RestoreConfirmationError: When the token is absent, empty, or not equal
            to the resolved path.
    """
    expected = str(resolved_root)
    if token is None or not token.strip():
        raise RestoreConfirmationError(
            "refusing to restore: no confirmation token was given. This restore "
            "overwrites the target's stores, its vault tree and its entire graph, "
            "and none of that comes back. Re-run with exactly:\n"
            f"    --confirm-target {expected}\n"
            "Type it because you have read it, not because it was in your history."
        )
    if token.strip() != expected:
        raise RestoreConfirmationError(
            f"refusing to restore: the confirmation token {token.strip()!r} is not "
            f"the resolved target path {expected!r}. The token is compared against "
            "the path this run RESOLVED, not against the path you passed, so a "
            "relative path cannot be confirmed by retyping it -- if these two "
            "strings surprise you, that surprise is the point. Check that the "
            "resolved path above is the machine and directory you meant, then pass "
            "it verbatim."
        )
