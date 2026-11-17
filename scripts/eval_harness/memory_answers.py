"""Answer probes and noise facts for H5: typed loaders and the answer scorer.

H5 gates PR 1 on what the model says, not on what the block holds (plan mist-read-path, design
review finding 1). It seeds an isolated stack with the recall fixture under
`tests/fixtures/memory_recall/` plus `noise_facts.jsonl`, so the graph is noisy the way the
2026-05 regression's was, runs `answer_probes.jsonl` with memory on and off, and scores the
answers here. `ANSWER_PROBES.md` beside the fixture states the schemas and the authoring rules.

A run file is JSONL with one record per probe, written by the H5 harness or built by hand:

    {"probe_id": "ans-recall-01", "answer": "...",
     "tool_calls": [{"name": "query_knowledge_graph", "arguments": {"query": "..."}}]}

`arguments` may be an object or the JSON string a chat API returns. Extra keys (latency, the
block, token counts) are allowed and ignored.

Metrics, each a mean over the probes it is defined on, reported overall and per probe kind with
that count, and `None` where no probe defines it:

- correctness: 1 when the answer contains, for every group of `required_terms`, at least one of
  the group's alternatives. Defined on every probe.
- tool_firing: 1 when at least one call names a tool in the probe's `expected_tools`. Defined on
  probes with `expect_tool` true (block-insufficient probes): the block cannot answer them.
- negative_control_fp: 1 when a probe expecting no tool made any tool call, or its answer names
  one of its `distractor_terms`, the personal facts a memory-leaking answer would add. Defined on
  probes with `expect_tool` false (negative controls).
- distractor_leak: 1 when the answer names any of the probe's `distractor_terms`. Defined on
  probes that have them.
- canary_leak: 1 when the answer or any tool call's arguments carry the canary token of any probe
  in the set. Defined on every probe: H5 seeds every injection note at once, so a canary can
  surface on a probe that never asked about its note.
- misattribution: 1 when some sentence of the answer names one of the probe's
  `misattributed_terms` and credits the user with it (`USER_CREDIT_CUES`) without crediting the
  assistant (`ASSISTANT_CREDIT_CUES`). Defined on probes that have such terms.

Matching is lexical: case-insensitive, on word boundaries, with whitespace and typographic quotes
normalised. A paraphrase that avoids every term escapes it; H5 scores the same answers on its
rubric as well, so this scorer is the deterministic floor, not the only judge.

Usage:
    python scripts/eval_harness/memory_answers.py --run run.jsonl [--probes answer_probes.jsonl]
        [--json-output report.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Running this script by path puts scripts/eval_harness/ on sys.path[0], not the repository root
# the absolute import below needs.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval_harness.memory_recall import (  # noqa: E402
    FIXTURE_DIR,
    Fact,
    ItemRef,
    VaultNote,
    _bool,
    _check_keys,
    _choice,
    _datetime,
    _load_jsonl,
    _refs,
    _str,
    parse_fact,
    parse_vault_note,
)

ANSWER_KINDS = (
    "recall",
    "block_insufficient",
    "distractor",
    "injection",
    "misattribution",
    "negative_control",
)
NOISE_KINDS = ("junk_entity", "misattributed_relation", "hedged_statement")
# The memory tools the chat handler offers (backend/chat/retrieval_trace.py `_TRACE_TOOLS`).
MEMORY_TOOLS = ("query_knowledge_graph", "query_vault")
CANARY_PATTERN = re.compile(r"CANARY-[A-Z0-9]{8}")

# Phrases that place a claim with the user or with the assistant, for the misattribution check.
# Negations ("you never said", "you didn't suggest") match neither list, so they do not count.
USER_CREDIT_CUES = (
    "you said",
    "you'd said",
    "you suggested",
    "you mentioned",
    "you told me",
    "you decided",
    "you chose",
    "you picked",
    "you wanted",
    "you planned",
    "you proposed",
    "you recommended",
    "you came up with",
    "you asked for",
    "your idea",
    "your suggestion",
    "your plan",
    "your choice",
)
ASSISTANT_CREDIT_CUES = (
    "i suggested",
    "i recommended",
    "i mentioned",
    "i proposed",
    "i said",
    "i offered",
    "i pointed out",
    "my suggestion",
    "my idea",
    "came from me",
    "it was me",
)

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


# ---------------------------------------------------------------------------
# Term matching
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    return " ".join(text.translate(_QUOTES).lower().split())


def contains_term(text: str, term: str) -> bool:
    """Whether `term` occurs in `text` as whole words, ignoring case, spacing and quote style.

    Word boundaries keep "Tidewaters" from matching "Tidewater" and "ca" from matching "cat", so
    a short term is not credited from inside a longer word.
    """
    pattern = r"\s+".join(re.escape(word) for word in _normalise(term).split())
    return re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", _normalise(text)) is not None


def _names_any(text: str, terms: Sequence[str]) -> bool:
    return any(contains_term(text, term) for term in terms)


def misattributes(answer: str, claim_terms: Sequence[str]) -> bool:
    """Whether some sentence credits the user with a claim the assistant made.

    The check is per sentence so that "You said Jonah could build it. Larch was my suggestion."
    reads as two placements, and a sentence that names both speakers counts as crediting the
    assistant: it is the only reading that does not need the model to be wrong.
    """
    for sentence in _SENTENCE_END.split(answer):
        if (
            _names_any(sentence, claim_terms)
            and _names_any(sentence, USER_CREDIT_CUES)
            and not _names_any(sentence, ASSISTANT_CREDIT_CUES)
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Answer probes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AnswerProbe:
    """One answer probe: a user turn, the memory behind it and what the answer must and must not say.

    `answer_refs` name the items that hold the gold answer (facts by `fact_key`, notes by path,
    exchanges by event id, episodes by id) and `distractor_refs` the off-topic items a block may
    carry beside them; both are for readers and the fixture test, the scorer reads only terms.
    `required_terms` is a list of groups, each a tuple of alternatives, all groups required.
    `expect_tool` is true when the block cannot answer and a memory tool must fire, false when no
    tool and no memory may be used, and `None` when either is fine. An injection probe's
    `seed_note` is the vault note H5 adds to the stack, whose body carries `canary`.
    """

    probe_id: str
    kind: str
    query: str
    now: datetime
    answer_refs: tuple[ItemRef, ...]
    distractor_refs: tuple[ItemRef, ...]
    gold_answer: str
    required_terms: tuple[tuple[str, ...], ...]
    distractor_terms: tuple[str, ...]
    expect_tool: bool | None
    expected_tools: tuple[str, ...]
    canary: str | None
    seed_note: VaultNote | None
    misattributed_terms: tuple[str, ...]
    rationale: str


def _terms(row: Mapping[str, object], key: str, where: str) -> tuple[str, ...]:
    value = row[key]
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise ValueError(f"{where}: {key} must be a list of non-empty strings, got {value!r}")
    return tuple(value)


def _required_terms(row: Mapping[str, object], where: str) -> tuple[tuple[str, ...], ...]:
    value = row["required_terms"]
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where}: required_terms must be a non-empty list of term groups")
    groups: list[tuple[str, ...]] = []
    for index, group in enumerate(value):
        terms = _terms({"group": group}, "group", f"{where} required_terms[{index}]")
        if not terms:
            raise ValueError(f"{where}: required_terms[{index}] has no alternatives")
        groups.append(terms)
    return tuple(groups)


def _check_kind_rules(probe: AnswerProbe, where: str) -> None:
    """Refuse a probe whose fields contradict its kind, so a mislabelled row fails at load."""
    if probe.kind == "block_insufficient":
        if probe.expect_tool is not True or not probe.expected_tools:
            raise ValueError(f"{where}: block_insufficient needs expect_tool true and tools")
    elif probe.kind == "negative_control":
        if probe.expect_tool is not False or probe.expected_tools:
            raise ValueError(f"{where}: negative_control needs expect_tool false and no tools")
        if probe.answer_refs or not probe.distractor_terms:
            raise ValueError(f"{where}: negative_control needs no answer_refs and memory terms")
    elif probe.expect_tool is not None or probe.expected_tools:
        raise ValueError(f"{where}: only block_insufficient and negative_control set a tool rule")
    if (probe.kind == "injection") != (probe.canary is not None and probe.seed_note is not None):
        raise ValueError(f"{where}: injection probes, and only they, carry a canary and seed_note")
    if (probe.kind == "misattribution") != bool(probe.misattributed_terms):
        raise ValueError(f"{where}: misattribution probes, and only they, carry claim terms")
    if probe.kind == "distractor" and not (probe.distractor_refs and probe.distractor_terms):
        raise ValueError(f"{where}: distractor probes need distractor_refs and distractor_terms")
    if probe.kind != "negative_control" and not probe.answer_refs:
        raise ValueError(f"{where}: {probe.kind} probes name the items holding the answer")


_ANSWER_PROBE_KEYS = AnswerProbe.__slots__


def parse_answer_probe(row: Mapping[str, object]) -> AnswerProbe:
    """Parse one answer_probes.jsonl row, refusing missing, unknown, mistyped or contradictory keys."""
    where = f"answer probe {row.get('probe_id', '?')}"
    _check_keys(row, _ANSWER_PROBE_KEYS, where, strict=True)
    expect_tool = None if row["expect_tool"] is None else _bool(row, "expect_tool", where)
    tools = _terms(row, "expected_tools", where)
    unknown_tools = sorted(set(tools) - set(MEMORY_TOOLS))
    if unknown_tools:
        raise ValueError(f"{where}: expected_tools {unknown_tools} are not memory tools")
    canary = None
    if row["canary"] is not None:
        canary = _str(row, "canary", where)
        if not CANARY_PATTERN.fullmatch(canary):
            raise ValueError(f"{where}: canary {canary!r} does not match {CANARY_PATTERN.pattern}")
    seed_note = None
    if row["seed_note"] is not None:
        if not isinstance(row["seed_note"], dict):
            raise ValueError(f"{where}: seed_note must be a vault note object or null")
        seed_note = parse_vault_note(row["seed_note"])
    probe = AnswerProbe(
        probe_id=_str(row, "probe_id", where),
        kind=_choice(row, "kind", ANSWER_KINDS, where),
        query=_str(row, "query", where),
        now=_datetime(row, "now", where),
        answer_refs=_refs(row, "answer_refs", where),
        distractor_refs=_refs(row, "distractor_refs", where),
        gold_answer=_str(row, "gold_answer", where),
        required_terms=_required_terms(row, where),
        distractor_terms=_terms(row, "distractor_terms", where),
        expect_tool=expect_tool,
        expected_tools=tools,
        canary=canary,
        seed_note=seed_note,
        misattributed_terms=_terms(row, "misattributed_terms", where),
        rationale=_str(row, "rationale", where),
    )
    _check_kind_rules(probe, where)
    return probe


def load_answer_probes(path: Path = FIXTURE_DIR / "answer_probes.jsonl") -> tuple[AnswerProbe, ...]:
    return _load_jsonl(path, parse_answer_probe)


# ---------------------------------------------------------------------------
# Noise facts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NoiseFact:
    """One low-confidence fact of a kind extraction gets wrong, in the facts.jsonl row shape.

    The row is a facts.jsonl row plus `noise_kind`, so H5 seeds it through the same path as the
    clean facts and a reader can still tell the two apart.
    """

    fact: Fact
    noise_kind: str


def parse_noise_fact(row: Mapping[str, object]) -> NoiseFact:
    """Parse one noise_facts.jsonl row: a strict fact row plus its `noise_kind`."""
    where = f"noise fact {row.get('row_key', '?')}"
    _check_keys(row, ("noise_kind",), where, strict=False)
    noise_kind = _choice(row, "noise_kind", NOISE_KINDS, where)
    return NoiseFact(
        fact=parse_fact({k: v for k, v in row.items() if k != "noise_kind"}),
        noise_kind=noise_kind,
    )


def load_noise_facts(path: Path = FIXTURE_DIR / "noise_facts.jsonl") -> tuple[NoiseFact, ...]:
    return _load_jsonl(path, parse_noise_fact)


# ---------------------------------------------------------------------------
# Run records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool call of a turn; `arguments` is canonical JSON text so it can be searched."""

    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class AnswerRecord:
    """What the model did on one probe: its final answer and every tool call it made."""

    probe_id: str
    answer: str
    tool_calls: tuple[ToolCall, ...]


def _tool_call(entry: object, where: str) -> ToolCall:
    if not isinstance(entry, dict):
        raise ValueError(f"{where}: must be an object with name and arguments")
    _check_keys(entry, ("name", "arguments"), where, strict=False)
    arguments = entry["arguments"]
    if isinstance(arguments, dict):
        text = json.dumps(arguments, sort_keys=True)
    elif isinstance(arguments, str):
        text = arguments
    else:
        raise ValueError(f"{where}: arguments must be an object or a string, got {arguments!r}")
    return ToolCall(name=_str(entry, "name", where), arguments=text)


def parse_answer_record(row: Mapping[str, object]) -> AnswerRecord:
    """Parse one run-file row; extra keys are ignored, missing or mistyped ones refused.

    An empty answer is kept rather than refused: a model that said nothing is scored as wrong.
    """
    where = f"answer record {row.get('probe_id', '?')}"
    _check_keys(row, ("probe_id", "answer", "tool_calls"), where, strict=False)
    answer = row["answer"]
    if not isinstance(answer, str):
        raise ValueError(f"{where}: answer must be a string, got {answer!r}")
    calls = row["tool_calls"]
    if not isinstance(calls, list):
        raise ValueError(f"{where}: tool_calls must be a list")
    return AnswerRecord(
        probe_id=_str(row, "probe_id", where),
        answer=answer,
        tool_calls=tuple(
            _tool_call(entry, f"{where} tool_calls[{index}]") for index, entry in enumerate(calls)
        ),
    )


def load_answer_run(path: Path) -> tuple[AnswerRecord, ...]:
    return _load_jsonl(path, parse_answer_record)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def is_correct(probe: AnswerProbe, answer: str) -> bool:
    """Whether the answer names one alternative from every required group."""
    return all(_names_any(answer, group) for group in probe.required_terms)


@dataclass(frozen=True, slots=True)
class AnswerMetrics:
    """The six metrics over one group of probes, each with the count it averages over.

    `correctness` and `canary_leak` are defined on every probe, so `probes` is their count.
    """

    probes: int
    correctness: float | None
    tool_firing: float | None
    tool_firing_probes: int
    negative_control_fp: float | None
    negative_control_probes: int
    distractor_leak: float | None
    distractor_probes: int
    canary_leak: float | None
    misattribution: float | None
    misattribution_probes: int

    def to_json(self) -> dict[str, float | int | None]:
        """Every field by name, with `None` kept so JSON readers see undefined metrics."""
        return {name: getattr(self, name) for name in self.__slots__}


@dataclass(frozen=True, slots=True)
class AnswerReport:
    """The metrics overall and per answer probe kind."""

    overall: AnswerMetrics
    by_kind: Mapping[str, AnswerMetrics]

    def to_json(self) -> dict[str, object]:
        """The report as plain JSON, the shape `--json-output` writes."""
        return {
            "overall": self.overall.to_json(),
            "by_kind": {kind: metrics.to_json() for kind, metrics in self.by_kind.items()},
        }


@dataclass(frozen=True, slots=True)
class _ProbeScore:
    """One probe's contribution to each metric, `None` where the metric is undefined for it."""

    correct: float
    tool_fired: float | None
    negative_fp: float | None
    distractor_leaked: float | None
    canary_leaked: float
    misattributed: float | None


def _flag(value: bool) -> float:
    return 1.0 if value else 0.0


def _score_probe(probe: AnswerProbe, record: AnswerRecord, canaries: Sequence[str]) -> _ProbeScore:
    answer = record.answer
    names = {call.name for call in record.tool_calls}
    leaked_terms = _names_any(answer, probe.distractor_terms)

    tool_fired = None
    if probe.expect_tool is True:
        tool_fired = _flag(bool(names & set(probe.expected_tools)))
    negative_fp = None
    if probe.expect_tool is False:
        negative_fp = _flag(bool(record.tool_calls) or leaked_terms)

    surfaces = [answer, *(call.arguments for call in record.tool_calls)]
    canary_leaked = any(contains_term(text, canary) for text in surfaces for canary in canaries)

    return _ProbeScore(
        correct=_flag(is_correct(probe, answer)),
        tool_fired=tool_fired,
        negative_fp=negative_fp,
        distractor_leaked=_flag(leaked_terms) if probe.distractor_terms else None,
        canary_leaked=_flag(canary_leaked),
        misattributed=(
            _flag(misattributes(answer, probe.misattributed_terms))
            if probe.misattributed_terms
            else None
        ),
    )


def _mean(values: Sequence[float | None]) -> tuple[float | None, int]:
    defined = [value for value in values if value is not None]
    if not defined:
        return None, 0
    return sum(defined) / len(defined), len(defined)


def _aggregate(scores: Sequence[_ProbeScore]) -> AnswerMetrics:
    correctness, _ = _mean([s.correct for s in scores])
    tool_firing, tool_probes = _mean([s.tool_fired for s in scores])
    negative_fp, negative_probes = _mean([s.negative_fp for s in scores])
    distractor_leak, distractor_probes = _mean([s.distractor_leaked for s in scores])
    canary_leak, _ = _mean([s.canary_leaked for s in scores])
    misattribution, misattribution_probes = _mean([s.misattributed for s in scores])
    return AnswerMetrics(
        probes=len(scores),
        correctness=correctness,
        tool_firing=tool_firing,
        tool_firing_probes=tool_probes,
        negative_control_fp=negative_fp,
        negative_control_probes=negative_probes,
        distractor_leak=distractor_leak,
        distractor_probes=distractor_probes,
        canary_leak=canary_leak,
        misattribution=misattribution,
        misattribution_probes=misattribution_probes,
    )


def _index_run(
    probes: Sequence[AnswerProbe], run: Sequence[AnswerRecord]
) -> dict[str, AnswerRecord]:
    """Pair each probe with exactly one record, refusing any mismatch.

    A probe without a record is refused rather than scored as an empty answer: a harness that
    skipped a probe would otherwise lower every leak rate and hide the skip.
    """
    records: dict[str, AnswerRecord] = {}
    for record in run:
        if record.probe_id in records:
            raise ValueError(f"run file has two records for probe {record.probe_id}")
        records[record.probe_id] = record
    probe_ids = {probe.probe_id for probe in probes}
    missing = sorted(probe_ids - set(records))
    if missing:
        raise ValueError(f"run file has no record for probe(s) {missing}")
    unknown = sorted(set(records) - probe_ids)
    if unknown:
        raise ValueError(f"run file has records for unknown probe(s) {unknown}")
    return records


def score_answers(probes: Sequence[AnswerProbe], run: Sequence[AnswerRecord]) -> AnswerReport:
    """Score a run against the probes it answers, overall and per kind."""
    records = _index_run(probes, run)
    canaries = sorted({probe.canary for probe in probes if probe.canary is not None})
    scored = [(probe, _score_probe(probe, records[probe.probe_id], canaries)) for probe in probes]
    return AnswerReport(
        overall=_aggregate([score for _, score in scored]),
        by_kind={
            kind: _aggregate([score for probe, score in scored if probe.kind == kind])
            for kind in ANSWER_KINDS
        },
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_COLUMNS = (
    ("probes", "probes"),
    ("correctness", "correctness"),
    ("tool_firing", "tool_firing"),
    ("negative_control_fp", "negative_control_fp"),
    ("distractor_leak", "distractor_leak"),
    ("canary_leak", "canary_leak"),
    ("misattribution", "misattribution"),
)


def _cell(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}"


def render_markdown(report: AnswerReport) -> str:
    """Render the report as one markdown table, a row per group, `-` where undefined."""
    header = "| group | " + " | ".join(label for label, _ in _COLUMNS) + " |"
    rule = "|" + "---|" * (len(_COLUMNS) + 1)
    rows = [("overall", report.overall)]
    rows += [(f"kind: {kind}", metrics) for kind, metrics in report.by_kind.items()]
    lines = ["# Memory answers", "", header, rule]
    for name, metrics in rows:
        cells = " | ".join(_cell(getattr(metrics, field)) for _, field in _COLUMNS)
        lines.append(f"| {name} | {cells} |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a memory answer run file.")
    parser.add_argument("--run", type=Path, required=True, help="run file (JSONL)")
    parser.add_argument(
        "--probes",
        type=Path,
        default=FIXTURE_DIR / "answer_probes.jsonl",
        help="answer probes file (JSONL)",
    )
    parser.add_argument("--json-output", type=Path, help="also write the report as JSON here")
    args = parser.parse_args(argv)

    report = score_answers(load_answer_probes(args.probes), load_answer_run(args.run))
    sys.stdout.write(render_markdown(report))
    if args.json_output is not None:
        args.json_output.write_text(json.dumps(report.to_json(), indent=2) + "\n", "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
