"""Cell P: the active ontology pack gives what the tree before it gave.

Moving the ontology into a pack must change no stored id, no relationship, and nothing
the model is sent. This cell checks that against stored payloads, with no model call:

    python -B -m scripts.eval_harness.ceiling.pack_equivalence --base-root <dir>
        --raw c3r2=<path> --raw sixty=<path>

`<dir>` holds the tree of the commit to compare against (`git archive <commit>`
unpacked). Each `--raw` names a set and the raw file of an extraction-service run over
it: one JSON object per line, whose `response.payload` holds the `entities` and
`relationships` the model wrote. A row with no payload is counted and left out.

The six checks, printed on the first line:

- `ids identical N/N`: the rows whose stored entities are equal, each entity whole (its
  id, type, name and properties) and in order, between the base's `normalize_payload`
  and this tree's `run_rails` with the registry's active pack.
- `relations identical N/N`: the rows whose stored relationships are equal likewise.
- `related_to_rate equal`: per set, the share of stored relationships that are the
  pack's escape edge.
- `prompt equal`: the system prompt template and the user template.
- `pairs equal`: the pair table, each predicate with its allowed sources and targets,
  in order.
- `schema equal`: the output schema, as JSON text, so key order counts.

Then, for every rule the rails run and for each retired type and the fallback type of
the manifest, how many rows of each set it fires on: the rows whose stored payload
changes when the pack is run without it. One that no set fires is named as resting on
its pinned case alone, after checking that a pinned case does fire it.

The output holds counts, set names, row numbers, rule ids and type names only, never a
payload or an utterance. The exit status is 0 when all six checks hold, 1 when any
differs, and 2 when the cell could not run.

The base half runs in an interpreter of its own, started on this file with the base
root first on its path, so the two trees' modules never share a process. Neither half
writes a file: both run with bytecode off, and the cell opens its inputs for reading
only. It runs under a read-only mount with a tmpfs at `/tmp`. On a mount that can be
written, start it with `-B` as above: the interpreter imports this module's packages
before the module can turn bytecode off, and would write theirs.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import subprocess  # nosec B404 - starts this file in a second interpreter, with a fixed argv
import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path

_BASE_HALF_FLAG = "--base-half"
# The name a base tree from before T46 gives its assembled prompt. A later tree, this
# one included, holds no such constant: its prompt is its active pack's.
BASE_PROMPT_NAME = "EXTRACTION_SYSTEM_PROMPT"
_BASE_TIMEOUT_SECONDS = 900

# What the base tree must hold for the base half to run.
_BASE_MODULES = (
    "backend/knowledge/extraction/id_rails.py",
    "backend/knowledge/extraction/prompts.py",
    "backend/knowledge/extraction/validator.py",
    "backend/extraction_service/schemas.py",
)

# A type no ontology has: the fallback type of a pack run without its fallback.
_NO_TYPE = "\x00"

Payload = dict[str, list]


class PackEquivalenceError(Exception):
    """The cell could not run as asked; `exit_code` is the process exit status."""

    exit_code = 2


class UsageError(PackEquivalenceError):
    """An argument that cannot be used."""


class InputError(PackEquivalenceError):
    """A raw file that cannot be read as rows, or that holds no payload."""


class BaseHalfError(PackEquivalenceError):
    """The base tree's half did not give its results."""


class HeadHalfError(PackEquivalenceError):
    """This tree's half did not give its results: the registry's pack is refused."""


# ---------------------------------------------------------------------------
# The raw rows
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Row:
    """One raw row's payload and what else of the row the rails read.

    Attributes:
        number: The row's line number in its file, from 1.
        entities: The payload's entities, as the model wrote them.
        relationships: The payload's relationships.
        scope: The row's scope label, or None.
        batch_record: The row's batch record, or None.
        turn_plan: The row's plan, or None.
    """

    number: int
    entities: list
    relationships: list
    scope: str | None
    batch_record: list | None
    turn_plan: Mapping | None


@dataclasses.dataclass(frozen=True, slots=True)
class RawSet:
    """A named set: its rows that hold a payload, and how many rows the file has."""

    name: str
    rows: tuple[Row, ...]
    total: int


def _row(number: int, record: object) -> Row | None:
    """The row of one raw record, or None for a record with no payload."""
    response = record.get("response") if isinstance(record, Mapping) else None
    payload = response.get("payload") if isinstance(response, Mapping) else None
    if not isinstance(payload, Mapping):
        return None
    entities, relationships = payload.get("entities"), payload.get("relationships")
    if not isinstance(entities, list) or not isinstance(relationships, list):
        return None
    scope = response.get("scope")
    label = scope.get("label") if isinstance(scope, Mapping) else None
    batch_record = response.get("batch_record")
    turn_plan = response.get("turn_plan")
    return Row(
        number=number,
        entities=entities,
        relationships=relationships,
        scope=label if isinstance(label, str) else None,
        batch_record=batch_record if isinstance(batch_record, list) else None,
        turn_plan=turn_plan if isinstance(turn_plan, Mapping) else None,
    )


def read_raw(name: str, path: Path) -> RawSet:
    """Read one set's raw file.

    Raises:
        InputError: The file cannot be read, a line is not JSON, or no row holds a
            payload. The message names the file and the line, never their content.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise InputError(f"set {name}: {path} cannot be read ({type(exc).__name__})") from exc
    rows: list[Row] = []
    total = 0
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        total += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputError(f"set {name}: line {number} of {path} is not JSON") from exc
        row = _row(number, record)
        if row is not None:
            rows.append(row)
    if not rows:
        raise InputError(f"set {name}: no row of {path} holds a payload, so nothing is compared")
    return RawSet(name=name, rows=tuple(rows), total=total)


# ---------------------------------------------------------------------------
# The base half: the base tree's rails, prompt, pair table and schema
# ---------------------------------------------------------------------------


def _pairs(table: Mapping) -> list[list]:
    """A pair table as ordered rows: predicate, allowed sources, allowed targets."""
    return [
        [name, *(None if end is None else sorted(end) for end in ends)]
        for name, ends in table.items()
    ]


def _outside(root: Path) -> list[str]:
    """The loaded `backend` modules whose file is not under `root`."""
    names: list[str] = []
    for name, module in sorted(sys.modules.items()):
        if name != "backend" and not name.startswith("backend."):
            continue
        file = getattr(module, "__file__", None)
        if file is not None and root not in Path(file).resolve().parents:
            names.append(name)
    return names


def _base_system_prompt(base_prompts) -> str:
    """The base tree's system prompt: its constant where it has one, else its pack's.

    Args:
        base_prompts: The base tree's `backend.knowledge.extraction.prompts`.
    """
    constant = getattr(base_prompts, BASE_PROMPT_NAME, None)
    if constant is not None:
        return constant
    from backend.knowledge.ontologies import registry

    return registry.active_pack().system_prompt


def _run_base_half(base_root: Path) -> int:
    """Run the base tree over the payloads on stdin and write its results to stdout.

    This function runs in the base half's own interpreter. The base root goes first on
    the path and every other entry that holds a `backend` directory is dropped, so the
    tree this file belongs to cannot be imported here.
    """
    root = base_root.resolve()
    kept = [entry for entry in sys.path if entry and not (Path(entry) / "backend").is_dir()]
    sys.path[:] = [str(root), *kept]

    from backend.extraction_service.schemas import EXTRACTION_OUTPUT_SCHEMA
    from backend.knowledge.extraction import prompts as base_prompts
    from backend.knowledge.extraction.id_rails import normalize_payload
    from backend.knowledge.extraction.validator import RELATIONSHIP_CONSTRAINTS

    system_prompt = _base_system_prompt(base_prompts)
    outside = _outside(root)
    if outside:
        sys.stderr.write(f"modules loaded from outside the base root: {', '.join(outside)}\n")
        return 3

    sets = json.load(sys.stdin)
    stored: dict[str, list[Payload]] = {}
    for name, payloads in sets.items():
        stored[name] = []
        for payload in payloads:
            entities, relationships = normalize_payload(
                payload["entities"], payload["relationships"]
            )
            stored[name].append({"entities": entities, "relationships": relationships})
    json.dump(
        {
            "sets": stored,
            "system_prompt": system_prompt,
            "user_template": base_prompts.EXTRACTION_USER_TEMPLATE,
            "pairs": _pairs(RELATIONSHIP_CONSTRAINTS),
            "schema": json.dumps(EXTRACTION_OUTPUT_SCHEMA),
        },
        sys.stdout,
    )
    return 0


def run_base(base_root: Path, sets: Sequence[RawSet]) -> dict:
    """The base tree's results for `sets`, from an interpreter of its own.

    Raises:
        BaseHalfError: The interpreter failed or gave no results. Its error output is
            not repeated, since a traceback may quote a payload: the message carries
            the exit status and the name of the error only.
    """
    payloads = {
        raw.name: [
            {"entities": row.entities, "relationships": row.relationships} for row in raw.rows
        ]
        for raw in sets
    }
    # -P keeps this file's directory off the child's path; -B keeps bytecode off disk.
    argv = [sys.executable, "-B", "-P", str(Path(__file__).resolve()), _BASE_HALF_FLAG]
    try:
        done = subprocess.run(  # nosec B603 - a fixed argv: this interpreter on this file
            [*argv, str(base_root)],
            input=json.dumps(payloads),
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(base_root),
            timeout=_BASE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise BaseHalfError(
            f"the base half gave nothing in {_BASE_TIMEOUT_SECONDS} seconds"
        ) from exc
    if done.returncode != 0:
        last = done.stderr.strip().splitlines()[-1:] or [""]
        error = last[0].split(":", 1)[0].strip() or "no error output"
        raise BaseHalfError(
            f"the base half failed with status {done.returncode} ({error}); its error "
            "output is not shown, since it may quote a payload"
        )
    try:
        return json.loads(done.stdout)
    except json.JSONDecodeError as exc:
        raise BaseHalfError("the base half's results are not JSON") from exc


# ---------------------------------------------------------------------------
# The head half: this tree's rails with the registry's active pack
# ---------------------------------------------------------------------------


def _as_json(value: object) -> object:
    """`value` as the base half's results arrive: through JSON."""
    return json.loads(json.dumps(value))


def _without_cases(pack, **changes):
    """`pack` with `changes` and no pinned case: a changed pack need not hold the cases."""
    rules = changes.pop("rules", pack.rules)
    return dataclasses.replace(
        pack,
        rules=tuple(dataclasses.replace(rule, pinned_case=None) for rule in rules),
        **changes,
    )


def ablations(pack) -> list[tuple[str, object]]:
    """Each thing the rails apply, named, with the pack run without it.

    A rule the rails run is left out of the rules. A retired type is left out of the
    manifest. The fallback type is replaced by a type no ontology has, so an entity it
    would have typed is typed differently.
    """
    from backend.knowledge.ontologies.pack import POST_LLM_PHASES

    manifest = pack.manifest
    named: list[tuple[str, object]] = []
    for rule in pack.rules:
        if rule.phase in POST_LLM_PHASES:
            kept = tuple(other for other in pack.rules if other.id != rule.id)
            named.append((f"rule {rule.id}", _without_cases(pack, rules=kept)))
    for retired in manifest.retired_types:
        others = tuple(other for other in manifest.retired_types if other.name != retired.name)
        changed = dataclasses.replace(manifest, retired_types=others)
        named.append(
            (f"manifest retired type {retired.name}", _without_cases(pack, manifest=changed))
        )
    changed = dataclasses.replace(manifest, fallback_type=_NO_TYPE)
    named.append(
        (f"manifest fallback type {manifest.fallback_type}", _without_cases(pack, manifest=changed))
    )
    return named


def _stored(row: Row, pack) -> Payload:
    """What `pack`'s rails store for one row; the row itself is not changed."""
    from backend.knowledge.extraction import rails

    entities, relationships = rails.run_rails(
        rails.rail_row(
            copy.deepcopy(row.entities),
            copy.deepcopy(row.relationships),
            scope=row.scope,
            batch_record=copy.deepcopy(row.batch_record),
            turn_plan=copy.deepcopy(row.turn_plan),
        ),
        pack,
    )
    return _as_json({"entities": entities, "relationships": relationships})


def _case_rows(pack) -> list[Row]:
    """The pack's pinned cases as rows."""
    rows: list[Row] = []
    for number, rule in enumerate(pack.rules, start=1):
        case = rule.pinned_case
        if case is not None:
            rows.append(
                Row(
                    number=number,
                    entities=list(case.entities),
                    relationships=list(case.relationships),
                    scope=case.scope,
                    batch_record=None if case.batch_record is None else list(case.batch_record),
                    turn_plan=case.turn_plan,
                )
            )
    return rows


@dataclasses.dataclass(frozen=True, slots=True)
class HeadResults:
    """This tree's half.

    Attributes:
        stored: Per set, what the active pack stores for each row.
        fired: Per thing the rails apply, per set, how many rows it fires on.
        pinned: The things the rails apply that some pinned case of the pack fires.
        pre_llm: The ids of the rules that run before the model.
        escape_edge: The pack's escape edge, or None.
        system_prompt: The pack's system prompt template.
        user_template: The user template.
        pairs: The pack's pair table, as ordered rows.
        schema: The pack's output schema, as JSON text.
    """

    stored: Mapping[str, list[Payload]]
    fired: Mapping[str, Mapping[str, int]]
    pinned: frozenset[str]
    pre_llm: tuple[str, ...]
    escape_edge: str | None
    system_prompt: str
    user_template: str
    pairs: list
    schema: str


def run_head(sets: Sequence[RawSet]) -> HeadResults:
    """This tree's results for `sets`, with the registry's active pack."""
    from backend.knowledge.extraction import prompts
    from backend.knowledge.ontologies import registry
    from backend.knowledge.ontologies.pack import PRE_LLM

    pack = registry.active_pack()
    whole = _without_cases(pack)
    without = ablations(pack)
    stored = {raw.name: [_stored(row, pack) for row in raw.rows] for raw in sets}
    fired: dict[str, dict[str, int]] = {}
    for name, ablated in without:
        fired[name] = {
            raw.name: sum(
                1
                for row, kept in zip(raw.rows, stored[raw.name], strict=True)
                if _stored(row, ablated) != kept
            )
            for raw in sets
        }
    cases = _case_rows(pack)
    pinned = frozenset(
        name
        for name, ablated in without
        if any(_stored(row, ablated) != _stored(row, whole) for row in cases)
    )
    return HeadResults(
        stored=stored,
        fired=fired,
        pinned=pinned,
        pre_llm=tuple(rule.id for rule in pack.rules if rule.phase == PRE_LLM),
        escape_edge=pack.manifest.escape_edge,
        system_prompt=pack.system_prompt,
        user_template=prompts.EXTRACTION_USER_TEMPLATE,
        pairs=_pairs(pack.pair_table),
        schema=json.dumps(pack.extraction_schema),
    )


# ---------------------------------------------------------------------------
# The comparison and the report
# ---------------------------------------------------------------------------


def _escape_rate(payloads: Sequence[Mapping], escape_edge: str | None) -> Fraction | None:
    """The share of a set's stored relationships that are the escape edge.

    Exact, so two rates are equal only when they are the same share. None for a set
    that stores no relationship.
    """
    edges = [edge for payload in payloads for edge in payload["relationships"]]
    if not edges:
        return None
    return Fraction(sum(1 for edge in edges if edge.get("type") == escape_edge), len(edges))


def compare(sets: Sequence[RawSet], base: Mapping, head: HeadResults) -> tuple[bool, list[str]]:
    """The report's lines, and True when all six checks hold."""
    rows = sum(len(raw.rows) for raw in sets)
    differing: dict[str, dict[str, list[int]]] = {}
    rates_equal = True
    for raw in sets:
        theirs, mine = base["sets"][raw.name], head.stored[raw.name]
        differing[raw.name] = {
            part: [
                row.number
                for row, old, new in zip(raw.rows, theirs, mine, strict=True)
                if old[part] != new[part]
            ]
            for part in ("entities", "relationships")
        }
        if _escape_rate(theirs, head.escape_edge) != _escape_rate(mine, head.escape_edge):
            rates_equal = False
    ids_same = rows - sum(len(parts["entities"]) for parts in differing.values())
    relations_same = rows - sum(len(parts["relationships"]) for parts in differing.values())
    prompt_equal = (base["system_prompt"], base["user_template"]) == (
        head.system_prompt,
        head.user_template,
    )
    pairs_equal = base["pairs"] == head.pairs
    schema_equal = base["schema"] == head.schema
    holds = (
        ids_same == rows
        and relations_same == rows
        and rates_equal
        and prompt_equal
        and pairs_equal
        and schema_equal
    )
    lines = [
        f"cell P: ids identical {ids_same}/{rows}; "
        f"relations identical {relations_same}/{rows}; "
        f"related_to_rate {'equal' if rates_equal else 'differs'}; "
        f"prompt {'equal' if prompt_equal else 'differs'}; "
        f"pairs {'equal' if pairs_equal else 'differ'}; "
        f"schema {'equal' if schema_equal else 'differs'}"
    ]
    for raw in sets:
        lines.append(f"set {raw.name}: {raw.total} rows, {len(raw.rows)} with a payload")
        for part, what in (("entities", "ids"), ("relationships", "relations")):
            numbers = differing[raw.name][part]
            if numbers:
                listed = ", ".join(str(number) for number in numbers)
                lines.append(f"set {raw.name}: rows whose {what} differ: {listed}")
    for name, counts in head.fired.items():
        line = f"{name}: " + ", ".join(f"{raw.name} {counts[raw.name]}" for raw in sets)
        if not any(counts.values()):
            line += (
                "; no set fires it, so it rests on its pinned case alone"
                if name in head.pinned
                else "; no set fires it, and no pinned case does either"
            )
        lines.append(line)
    for rule_id in head.pre_llm:
        lines.append(f"rule {rule_id}: runs before the model, so no payload fires it")
    return holds, lines


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def _parse_raw(values: Sequence[str]) -> list[tuple[str, Path]]:
    named: list[tuple[str, Path]] = []
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not name or not path:
            raise UsageError(f"--raw takes NAME=PATH, not {value!r}")
        if name in {other for other, _ in named}:
            raise UsageError(f"--raw names the set {name} twice")
        named.append((name, Path(path)))
    return named


def _check_base_root(base_root: Path) -> None:
    missing = [name for name in _BASE_MODULES if not (base_root / name).is_file()]
    if missing:
        raise UsageError(f"--base-root {base_root} holds no {', '.join(missing)}")


def run(argv: Sequence[str]) -> int:
    """Run the cell and print its report; 0 when all six checks hold, 1 otherwise."""
    parser = argparse.ArgumentParser(
        prog="pack_equivalence",
        description="Cell P: the active ontology pack gives what the base tree gave.",
    )
    parser.add_argument("--base-root", required=True, type=Path, help="the base commit's tree")
    parser.add_argument(
        "--raw",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="a set's name and its extraction-service raw file; repeat for each set",
    )
    args = parser.parse_args(argv)
    # Resolved once, against this process's working directory. The base half runs with
    # the base root as its own, so a relative root it resolved there would name a
    # directory inside the root.
    base_root = args.base_root.resolve()
    _check_base_root(base_root)
    sets = [read_raw(name, path) for name, path in _parse_raw(args.raw)]
    base = run_base(base_root, sets)

    from backend.knowledge.ontologies.pack import PackLoadError

    try:
        head = run_head(sets)
    except PackLoadError as exc:
        raise HeadHalfError(f"the active pack is refused: {exc}") from exc
    holds, lines = compare(sets, base, head)
    print("\n".join(lines))
    return 0 if holds else 1


def main(argv: Sequence[str] | None = None) -> int:
    """The command line of the cell, and of its base half."""
    # Set before either half imports a tree, so no import writes bytecode into it.
    sys.dont_write_bytecode = True
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == [_BASE_HALF_FLAG] and len(arguments) == 2:
        return _run_base_half(Path(arguments[1]))
    try:
        return run(arguments)
    except PackEquivalenceError as exc:
        print(f"cell P could not run: {exc}", file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
