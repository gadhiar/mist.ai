"""The volume-safety guard `instance.py` applies to every command before it runs.

v1's rule for the instance CLI: no command runs `down -v`, `rm -v` or `volume rm`. The graph
lives in an external volume, and promotion keeps the previous one under its old name, so every
form that could delete a volume is refused here, at the one place commands execute
(`subprocess_runner` in `instance.py`). The restore drill needs `docker volume inspect` and
`docker volume create`, which stay allowed.

This module holds the guard and its token data and nothing else, so the source scans of
`instance.py` run with no exemption. It imports only the standard library, because
`instance.py` loads it by file path from wherever it runs, `<root>/bin/` included.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

# A token that is one of these, whatever surrounds it, removes volumes or containers in bulk.
FORBIDDEN_TOKENS = frozenset({"-v", "prune", "down"})
# `--volumes` and `--volumes=<value>` alike.
FORBIDDEN_PREFIX = "--volumes"
# A short-option cluster such as `-fv`, which carries `-v` inside it.
SHORT_OPTION_CLUSTER = re.compile(r"-[A-Za-z]{2,}")
# `volume` is allowed only as the subcommand group of these.
VOLUME_SUBCOMMANDS = frozenset({"inspect", "create"})


class DestructiveCommandError(Exception):
    """A command that could delete a volume was refused before it ran."""

    def __init__(self, argv: Sequence[str], reason: str = "") -> None:
        self.argv = list(argv)
        self.reason = reason
        detail = f" ({reason})" if reason else ""
        super().__init__(f"refused a volume-deleting command{detail}: {' '.join(self.argv)}")


def _refusal(token: str, following: str | None) -> str | None:
    """Why `token` makes a command destructive, or None when it does not."""
    if token in FORBIDDEN_TOKENS:
        return f"`{token}`"
    if token.startswith(FORBIDDEN_PREFIX):
        return f"`{token}`"
    if SHORT_OPTION_CLUSTER.fullmatch(token) and "v" in token:
        return f"`{token}` carries -v"
    if token == "volume" and following not in VOLUME_SUBCOMMANDS:
        return f"`volume {following}`" if following is not None else "`volume` with no subcommand"
    return None


def assert_volume_safe(argv: Sequence[str]) -> None:
    """Refuse `argv` when any token could delete a volume.

    Raises:
        DestructiveCommandError: Naming the argv, when a token is `-v`, starts with
            `--volumes`, is a short-option cluster holding `v`, is `prune` or `down`, or is
            `volume` not followed directly by `inspect` or `create`.
    """
    tokens = [str(token) for token in argv]
    for index, token in enumerate(tokens):
        following = tokens[index + 1] if index + 1 < len(tokens) else None
        reason = _refusal(token, following)
        if reason is not None:
            raise DestructiveCommandError(tokens, reason)
