"""Line-number citations in `backend/knowledge/curation/graph_writer.py` must hold.

`GraphWriter._upsert_entity`'s docstring justifies the extraction backlog's
crash-replay guard with `grep -n '<pattern>' <file>` -> <line> citations into
the dispatcher, the backlog store and the pipeline. Two of them went stale
(the dispatcher's `head = scan.head` and `apply_cached_turn` call moved), and
a stale citation reads exactly like a checked one. This test re-runs every
citation in the module that still names a line number, so a drift fails here
instead of in review. Citations
that name a function instead of a line (the repository convention since
0092cc3) are not checked; they cannot drift this way.

Lives with the extraction-backlog tests because every cited invariant is the
backlog's.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GRAPH_WRITER = REPO_ROOT / "backend" / "knowledge" / "curation" / "graph_writer.py"

_CITATION = re.compile(r"`grep -n '([^']+)' ([^`\s]+)`\s*\)?[;,]?\s*->\s*(\d+)")


def _citations() -> list[tuple[str, str, int]]:
    source = GRAPH_WRITER.read_text(encoding="utf-8")
    return [(m.group(1), m.group(2), int(m.group(3))) for m in _CITATION.finditer(source)]


def test_citations_are_found():
    # Guards the regex itself: an empty list would make the check vacuous.
    assert len(_citations()) >= 5


@pytest.mark.parametrize("pattern,path,line", _citations())
def test_cited_line_matches_pattern(pattern, path, line):
    lines = (REPO_ROOT / path).read_text(encoding="utf-8").splitlines()
    hits = [i + 1 for i, text in enumerate(lines) if re.search(pattern, text)]
    assert line in hits, f"grep -n '{pattern}' {path} -> {hits}, docstring cites {line}"
