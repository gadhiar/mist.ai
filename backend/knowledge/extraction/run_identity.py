"""The run identity: one hash naming the whole setup that produced an extraction.

`extraction_version` and `model_hash` key the epoch and the cache, but they
leave out most of what decides an extraction's output: the code that renders
the prompt and parses the reply, the request settings, the adapter and the
llama.cpp build. Two runs that differ in any of those can share both stamps
and still produce different graphs. The run identity covers all of them, so
a result, a gauntlet run or a report can say exactly which setup it came
from, and two runs can be compared knowing whether they measured the same
thing (MIS-188). It is recorded, not keyed: nothing matches on it.

The code side is hashed from source with comments, docstrings, blank lines
and trailing whitespace removed, so editing an explanation does not move it
and any other edit does. Removal works on positions: `ast` finds docstrings,
`tokenize` finds comments and the extent of every string literal, and a line
that lies inside a string literal is kept exactly as written, because there
its blank lines and trailing spaces are content. The token stream depends on
the Python version (3.12 tokenizes f-strings into parts), which is why
`tests/unit/knowledge/extraction/test_run_identity.py` pins the hash of a
fixed sample.

This module is a leaf: standard library only. The service assembles the
components (`backend.extraction_service.app.run_identity_components`), and
`backend.knowledge.version_stamps.CODE_REVISIONS` records each code identity
a deliberate change produced.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import io
import json
import textwrap
import tokenize
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import ModuleType

_RUN_IDENTITY_PREFIX = "ri1-"
_CODE_IDENTITY_PREFIX = "ci1-"

# Separators that cannot occur in Python source text a module would hold,
# so a name, a source and the next object can never run together.
_NAME_SEPARATOR = "\x1e"
_OBJECT_SEPARATOR = "\x1f"

# The tokenize types that open and close an f-string from Python 3.12 on;
# absent (None) on 3.11, where an f-string is one STRING token.
_FSTRING_START = getattr(tokenize, "FSTRING_START", None)
_FSTRING_END = getattr(tokenize, "FSTRING_END", None)


@dataclass(frozen=True, slots=True)
class RunIdentityComponents:
    """Everything that shapes an extraction result, as the service runs it.

    Attributes:
        prompt_sha256: The prompt text, output schemas and adapter name and
            version sent to the model (`ResultStamps.prompt_sha256`).
        code_sha256: `code_sha256` over the service's `RUN_IDENTITY_CODE`.
        ontology_version: The active ontology's version.
        extraction_version: The extraction version stamped on every result.
        adapter: The model-family adapter's name.
        model_hash: The bare model identity the service is configured with.
        llama_cpp_build: The llama.cpp build its llama-server runs.
        constrained_mode: The effective constrained-decoding mode, after a
            settings override is applied to the adapter default; None when
            neither names one.
        reasoning_effort: The configured reasoning-effort hint.
        reasoning_budget_tokens: The reasoning token budget; None is unbudgeted.
        temperature: The sampling temperature every stage uses.
        max_attempts: Stage 2 attempts per job, the repair retries included.
        scope_enabled: Whether Stage 1.5 runs.
        scope_max_tokens: Stage 1.5's output token limit.
        extract_max_tokens: Stage 2's output token limit.
        derive_max_tokens: Stage 9's output token limit.
        list_batch_max_units: Most list units per Stage 2 call. None is how a
            service from before the list pass reported it.
        list_context_max_units: Most earlier units sent as context; None as
            for `list_batch_max_units`.
    """

    prompt_sha256: str
    code_sha256: str
    ontology_version: str
    extraction_version: str
    adapter: str
    model_hash: str
    llama_cpp_build: str
    constrained_mode: str | None
    reasoning_effort: str
    reasoning_budget_tokens: int | None
    temperature: float
    max_attempts: int
    scope_enabled: bool
    scope_max_tokens: int
    extract_max_tokens: int
    derive_max_tokens: int
    list_batch_max_units: int | None
    list_context_max_units: int | None


# The components that are code rather than deployment configuration: what
# `code_identity` hashes and the drift guard pins. A settings value, the model
# or the llama.cpp build can change per host without a code change, so they
# stay out.
CODE_IDENTITY_FIELDS: tuple[str, ...] = (
    "prompt_sha256",
    "code_sha256",
    "ontology_version",
    "scope_max_tokens",
    "extract_max_tokens",
    "derive_max_tokens",
    "list_batch_max_units",
    "list_context_max_units",
)


def _canonical_sha256(values: dict[str, object]) -> str:
    """sha256 of `values` as canonical JSON: sorted keys, no whitespace."""
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_run_identity(components: RunIdentityComponents) -> str:
    """Return `"ri1-"` and the sha256 of every component, as canonical JSON."""
    return _RUN_IDENTITY_PREFIX + _canonical_sha256(dataclasses.asdict(components))


def code_identity(components: RunIdentityComponents) -> str:
    """Return `"ci1-"` and the same hash over the code-side components only.

    The prefix differs from the run identity's so the two can never be
    mistaken for each other in a record.
    """
    subset = {name: getattr(components, name) for name in CODE_IDENTITY_FIELDS}
    return _CODE_IDENTITY_PREFIX + _canonical_sha256(subset)


def code_sha256(objects: Sequence[ModuleType | Callable[..., object]]) -> str:
    """sha256 over each object's normalised source, in the order given.

    Each object contributes its qualified name and `normalized_source` of its
    source, so renaming, reordering or moving code between the objects moves
    the hash too.

    Raises:
        OSError: An object's source cannot be read.
        SyntaxError: An object's source does not parse on its own.
    """
    parts = [
        _qualified_name(obj) + _NAME_SEPARATOR + normalized_source(_source_of(obj))
        for obj in objects
    ]
    return hashlib.sha256(_OBJECT_SEPARATOR.join(parts).encode("utf-8")).hexdigest()


def _qualified_name(obj: ModuleType | Callable[..., object]) -> str:
    if isinstance(obj, ModuleType):
        return obj.__name__
    return f"{obj.__module__}.{obj.__qualname__}"


def _source_of(obj: ModuleType | Callable[..., object]) -> str:
    # A method's source is indented; dedent it so it parses on its own.
    return textwrap.dedent(inspect.getsource(obj))


def normalized_source(source: str) -> str:
    """`source` with comments, docstrings, blank lines and trailing whitespace removed.

    A line inside a string literal (other than a docstring) is kept as
    written, blank or not.

    Raises:
        SyntaxError: `source` does not parse.
    """
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    lines = source.split("\n")
    tree = ast.parse(source)
    docstrings = _docstring_spans(tree, lines)
    removals, protected_rows = _token_spans(source, docstrings)

    kept: list[str] = []
    for row, line in enumerate(lines, start=1):
        text = _without(line, removals.get(row, []))
        if row in protected_rows:
            kept.append(text)
            continue
        text = text.rstrip()
        if text:
            kept.append(text)
    return "\n".join(kept)


# A position is (row, column): row 1-based, column a character offset.
_Position = tuple[int, int]
_Span = tuple[_Position, _Position]


def _docstring_spans(tree: ast.AST, lines: list[str]) -> list[_Span]:
    """The span of every module, class and function docstring."""
    spans: list[_Span] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if not (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
            and first.end_lineno is not None
            and first.end_col_offset is not None
        ):
            continue
        spans.append(
            (
                (first.lineno, _char_column(lines[first.lineno - 1], first.col_offset)),
                (first.end_lineno, _char_column(lines[first.end_lineno - 1], first.end_col_offset)),
            )
        )
    return spans


def _char_column(line: str, byte_offset: int) -> int:
    """`ast` columns count UTF-8 bytes; `tokenize` columns count characters."""
    return len(line.encode("utf-8")[:byte_offset].decode("utf-8"))


def _token_spans(
    source: str, docstrings: list[_Span]
) -> tuple[dict[int, list[tuple[int, int]]], set[int]]:
    """Per-row column ranges to remove, and the rows that lie inside a string literal.

    Removed: every comment token and every docstring. Protected: each row a
    string literal continues past, from its first row to the row before its
    last, so trailing whitespace and blank lines inside it are kept.
    """
    removals: dict[int, list[tuple[int, int]]] = {}
    protected: set[int] = set()

    def remove(start: _Position, end: _Position) -> None:
        (start_row, start_col), (end_row, end_col) = start, end
        for row in range(start_row, end_row + 1):
            first = start_col if row == start_row else 0
            last = end_col if row == end_row else -1
            removals.setdefault(row, []).append((first, last))

    def in_docstring(position: _Position) -> bool:
        return any(start <= position < end for start, end in docstrings)

    for start, end in docstrings:
        remove(start, end)

    fstring_starts: list[_Position] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            remove(token.start, token.end)
        elif token.type == tokenize.STRING:
            if not in_docstring(token.start):
                protected.update(range(token.start[0], token.end[0]))
        elif _FSTRING_START is not None and token.type == _FSTRING_START:
            fstring_starts.append(token.start)
        elif _FSTRING_END is not None and token.type == _FSTRING_END and fstring_starts:
            protected.update(range(fstring_starts.pop()[0], token.end[0]))
    return removals, protected


def _without(line: str, ranges: list[tuple[int, int]]) -> str:
    """`line` with each `(start, end)` column range cut out; an end of -1 is the line's end."""
    if not ranges:
        return line
    cut = [False] * len(line)
    for start, end in ranges:
        for column in range(start, len(line) if end == -1 else end):
            cut[column] = True
    return "".join(char for char, removed in zip(line, cut, strict=True) if not removed)
