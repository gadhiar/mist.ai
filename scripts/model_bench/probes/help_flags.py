"""Parser and checker for `llama-server --help` flag spellings.

Used by `bench_host.py`'s `plan --host-checks` (lead-only, on the host) to
confirm every arm's built argv only uses flags the pinned build's own
`--help` text actually lists -- the class of bug that blocked session S2,
where the pinned b11151 build had already dropped `--no-mmap` in favor of
`-lm, --load-mode`. No docker, no network: this module only parses text
handed to it by the caller.
"""

from __future__ import annotations

import re

# A flag token: 1-2 leading dashes, then a letter, then letters/digits/dashes.
# Deliberately excludes bare `--` and section-header rules like
# `----- common params -----` (whose dashes are not followed by a letter).
_FLAG_TOKEN = r"-{1,2}[A-Za-z][\w-]*"

# An option line: 1-8 leading spaces/tabs (indented, but not as far as a
# wrapped description continuation line, which llama.cpp's --help indents
# well past the flag column), then one or more comma-separated flag tokens,
# then either whitespace (an argument placeholder or the description) or
# end of line.
_OPTION_LINE_RE = re.compile(rf"^[ \t]{{1,8}}(?P<flags>{_FLAG_TOKEN}(?:,\s*{_FLAG_TOKEN})*)(?=[ \t]|$)")


def parse_help_flags(text: str) -> set[str]:
    """Extract every flag spelling (short and long form) from `--help` text.

    Matches option lines of the shape `  -x, --long-name [ARG]  description`,
    including options with only a long form (`  --no-webui`) or only a short
    one. Section headers (`----- common params -----`), the usage line, and
    indented description-continuation lines are not option lines and are
    ignored.
    """
    flags: set[str] = set()
    for line in text.splitlines():
        match = _OPTION_LINE_RE.match(line)
        if not match:
            continue
        for token in match.group("flags").split(","):
            flags.add(token.strip())
    return flags


def unknown_flags(argv: list[str], help_flags: set[str]) -> list[str]:
    """Flag-shaped tokens in `argv` that `help_flags` does not list.

    A token counts as flag-shaped if it starts with `-` (this argv space --
    llama-server's built argv -- never places a bare `-`-prefixed value next
    to a flag, so this is unambiguous here). Order-preserving, deduplicated.
    """
    seen: list[str] = []
    for tok in argv:
        if tok.startswith("-") and tok not in help_flags and tok not in seen:
            seen.append(tok)
    return seen
