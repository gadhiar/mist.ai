"""Answer rounds, end to end, through the unchanged extraction service.

Each round the tool under test starts the relay over a store, starts the unchanged
service with `uvicorn backend.extraction_service.app:build_app_from_env --factory`, runs the
unchanged bench probe once per probe, and offers each probe's first pending request. The
test plays the answerer: it reads only what the tool shows (stdout, `pending-1.md`, the
prompt and schema files), writes `answers-in-1.jsonl` and calls `--add`.

The probes are synthetic: one prose turn, and one list turn of six items, which the
engine extracts in two batches (at most four units each). Ports are chosen by the system,
so the test never collides with a running relay or service. The rounds run once per
module, and each test reads the record they leave.

The replay tests then rerun every stored answer through the service twice, with the bench
probe once over both probes, and compare the two replays; an altered answer must make the
comparison fail and name the probe, without printing a value.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess  # nosec B404 - runs the tools under test with fixed argument lists
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from backend.extraction_service.schemas import (  # noqa: E402
    EXTRACTION_OUTPUT_SCHEMA,
    SCOPE_OUTPUT_SCHEMA,
)
from backend.knowledge.extraction.scope_classifier import (  # noqa: E402
    SCOPE_CLASSIFIER_SYSTEM_PROMPT,
)
from backend.knowledge.ontologies.registry import active_pack  # noqa: E402
from scripts.eval_harness.ceiling import replay as rp  # noqa: E402
from scripts.eval_harness.ceiling import round as rd  # noqa: E402
from scripts.eval_harness.ceiling.store import AnswerStore, OfferedRequest  # noqa: E402

H3_MARKERS = (
    "expected_relationships",
    "expected_entities",
    "stage2_calls",
    "timings_ms",
    "run_identity",
    "Relationship recall",
)
HIDDEN_RUN_DIR = "eval-ceiling-a"

PROSE_TAG = "syn-01-prose"
LIST_TAG = "syn-02-list"
PROBES = [
    {"tag": PROSE_TAG, "utterance": "My sister Priya moved to Lisbon last spring."},
    {
        "tag": LIST_TAG,
        "utterance": (
            "Things I need to buy for the party:\n"
            "- eggs\n- milk\n- flour\n- sugar\n- butter\n- lemons"
        ),
    },
]

_LINE_RE = re.compile(
    r"^round (?P<round>\d+) answerer (?P<k>\d): probes (?P<probes>\d+), "
    r"complete (?P<complete>\d+), offered (?P<offered>\d+) \(scope (?P<scope>\d+), "
    r"extract (?P<extract>\d+), repair (?P<repair>\d+), derive (?P<derive>\d+)\), "
    r"superseded (?P<superseded>\d+), stored (?P<stored>\d+)$"
)
_ITEM_RE = re.compile(
    r"^- key: `(?P<key>[0-9a-f]{64})`\n- tag: (?P<tag>\S+)\n- kind: (?P<kind>\w+)\n"
    r"- system prompt: (?P<prompt>\S+)\n- schema: (?P<schema>\S+)$",
    re.MULTILINE,
)


@dataclass
class Layout:
    """One answerer's files, all under `tmp_path`, as the round tool's flags name them."""

    root: Path
    probes: Path
    store_dir: Path
    local_dir: Path

    @property
    def store(self) -> Path:
        return self.store_dir / "answers-1.jsonl"

    @property
    def state(self) -> Path:
        return self.local_dir / "ceiling" / "state-1.json"

    @property
    def pending(self) -> Path:
        return self.local_dir / "ceiling" / "pending-1.md"

    @property
    def answers_in(self) -> Path:
        return self.local_dir / "ceiling" / "answers-in-1.jsonl"

    def args(self) -> list[str]:
        return [
            "--probes",
            str(self.probes),
            "--store-dir",
            str(self.store_dir),
            "--local-dir",
            str(self.local_dir),
            "--relay-port",
            "0",
            "--service-port",
            "0",
        ]


@dataclass
class Shown:
    """Everything the round tool showed the answerer, kept for the marker check."""

    texts: list[str] = field(default_factory=list)

    def add(self, text: str) -> None:
        self.texts.append(text)


@dataclass
class Campaign:
    """The record the module's rounds leave for each test to read."""

    layout: Layout
    rounds: list[subprocess.CompletedProcess[str]]
    lines: list[dict[str, int]]
    items: list[list[dict[str, str]]]
    adds: list[subprocess.CompletedProcess[str]]
    refused_add: subprocess.CompletedProcess[str]
    shown: Shown


def _run_tool(module: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603 - fixed argument list, no shell
        [sys.executable, "-m", module, *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def _round(layout: Layout, *extra: str) -> subprocess.CompletedProcess[str]:
    return _run_tool(
        "scripts.eval_harness.ceiling.round", "--answerer", "1", *layout.args(), *extra
    )


def _parse_line(stdout: str) -> dict[str, int]:
    lines = stdout.splitlines()
    assert len(lines) == 1, f"stdout is not exactly one line: {lines}"
    match = _LINE_RE.match(lines[0])
    assert match, f"stdout is not the counts-only line: {lines[0]}"
    return {k: int(v) for k, v in match.groupdict().items()}


def _items(pending: str) -> list[dict[str, str]]:
    return [m.groupdict() for m in _ITEM_RE.finditer(pending)]


def _answer(item: dict[str, str]) -> dict:
    """The test answerer's reply: a fixed scope, or one entity named for the request."""
    if item["kind"] == "scope":
        return {"scope": "user-scope", "confidence": 0.9}
    return {
        "entities": [{"id": f"thing-{item['key'][:8]}", "type": "Thing", "name": item["tag"]}],
        "relationships": [],
    }


def _write_answers(layout: Layout, answers: list[dict]) -> None:
    layout.answers_in.write_text(
        "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in answers), encoding="utf-8"
    )


def _show_round_files(layout: Layout, shown: Shown, items: list[dict[str, str]]) -> None:
    shown.add(layout.pending.read_text(encoding="utf-8"))
    for item in items:
        for name in ("prompt", "schema"):
            shown.add((layout.pending.parent / item[name]).read_text(encoding="utf-8"))


def make_layout(root: Path) -> Layout:
    root.mkdir(parents=True, exist_ok=True)
    probes = root / "probes.jsonl"
    probes.write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in PROBES), encoding="utf-8"
    )
    layout = Layout(root=root, probes=probes, store_dir=root / "data", local_dir=root / "local")
    layout.store_dir.mkdir()
    return layout


def run_campaign(layout: Layout, answer: Callable[[dict[str, str]], dict]) -> Campaign:
    """Run rounds until nothing is pending, answering every offer, refusals first."""
    shown = Shown()
    rounds, lines, all_items, adds = [], [], [], []
    refused_add = None
    for _ in range(8):
        result = _round(layout)
        shown.add(result.stdout)
        shown.add(result.stderr)
        rounds.append(result)
        assert result.returncode in (0, 3), result.stderr
        line = _parse_line(result.stdout)
        lines.append(line)
        items = _items(layout.pending.read_text(encoding="utf-8"))
        all_items.append(items)
        _show_round_files(layout, shown, items)
        if result.returncode == 0:
            break
        assert len(items) == line["offered"]
        if refused_add is None and any(i["kind"] == "extract" for i in items):
            target = next(i for i in items if i["kind"] == "extract")
            _write_answers(
                layout,
                [
                    {"key": target["key"], "answer": {"entities": "not a list"}},
                    {
                        "key": target["key"],
                        "answer": {"entities": [], "relationships": [], "note": "<|channel|>x"},
                    },
                ],
            )
            refused_add = _round(layout, "--add")
            shown.add(refused_add.stdout)
            shown.add(refused_add.stderr)
        _write_answers(layout, [{"key": i["key"], "answer": answer(i)} for i in items])
        added = _round(layout, "--add")
        shown.add(added.stdout)
        shown.add(added.stderr)
        adds.append(added)
    assert refused_add is not None
    return Campaign(
        layout=layout,
        rounds=rounds,
        lines=lines,
        items=all_items,
        adds=adds,
        refused_add=refused_add,
        shown=shown,
    )


@pytest.fixture(scope="module")
def campaign(tmp_path_factory: pytest.TempPathFactory) -> Campaign:
    layout = make_layout(tmp_path_factory.mktemp("answerer"))
    return run_campaign(layout, _answer)


# --- the env table and the prompts, which need no service --------------------------------


def test_the_service_env_is_the_plan_table() -> None:
    assert rd.service_env("http://127.0.0.1:18080", base={}) == {
        "EXTRACTION_LLM_BASE_URL": "http://127.0.0.1:18080",
        "EXTRACTION_ADAPTER": "gptoss",
        "EXTRACTION_REASONING_EFFORT": "low",
        "EXTRACTION_LLM_TIMEOUT_SECONDS": "300",
        "EXTRACTION_MAX_ATTEMPTS": "2",
        "EXTRACTION_CONSTRAINED_MODE": "",
        "EXTRACTION_SCOPE_ENABLED": "true",
        "EXTRACTION_TEMPERATURE": "0.0",
        "EXTRACTION_MODEL_FILE": "answerer-relay",
        "EXTRACTION_MODEL_HASH": "answerer-relay",
        "LLAMA_CPP_BUILD": "relay",
        "EXTRACTION_LOCATION_LABEL": "container-relay",
    }


def test_the_service_env_drops_every_inherited_extraction_setting() -> None:
    env = rd.service_env(
        "http://127.0.0.1:1",
        base={
            "PATH": "/usr/bin",
            "EXTRACTION_REASONING_BUDGET_TOKENS": "64",
            "EXTRACTION_REASONING_EFFORT": "high",
            "LLAMA_CPP_BUILD": "b1",
        },
    )
    assert env["PATH"] == "/usr/bin"
    assert "EXTRACTION_REASONING_BUDGET_TOKENS" not in env
    assert env["EXTRACTION_REASONING_EFFORT"] == "low"
    assert env["LLAMA_CPP_BUILD"] == "relay"


def test_the_default_command_line_is_the_plan_one() -> None:
    args = rd.parse_args(["--answerer", "2"])
    assert (args.relay_port, args.service_port) == (18080, 18090)
    assert args.probes == _REPO_ROOT / "data" / "ceiling" / "probes.jsonl"
    assert args.store_dir == _REPO_ROOT / "data" / "ceiling"
    assert args.local_dir == _REPO_ROOT / ".local"
    assert rd.service_command(18090) == [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.extraction_service.app:build_app_from_env",
        "--factory",
        "--host",
        "127.0.0.1",
        "--port",
        "18090",
    ]


def test_answerer_k_takes_rows_10k_minus_9_to_10k() -> None:
    rows = [rd.Probe(tag=f"t{n}", utterance="u", line="{}") for n in range(1, 41)]
    assert [p.tag for p in rd.answerer_probes(rows, 3)] == [f"t{n}" for n in range(21, 31)]
    with pytest.raises(rd.ProbeFileError):
        rd.answerer_probes(rows[:10], 2)


@pytest.mark.parametrize("prompt", [active_pack().system_prompt, SCOPE_CLASSIFIER_SYSTEM_PROMPT])
def test_no_h3_marker_is_in_the_system_or_scope_prompt(prompt: str) -> None:
    assert [m for m in H3_MARKERS if m in prompt] == []


# --- the rounds --------------------------------------------------------------------------


def test_round_1_offers_each_scope_request_and_supersedes_the_degraded_extract(
    campaign: Campaign,
) -> None:
    assert campaign.rounds[0].returncode == 3
    assert campaign.lines[0] == {
        "round": 1,
        "k": 1,
        "probes": 2,
        "complete": 0,
        "offered": 2,
        "scope": 2,
        "extract": 0,
        "repair": 0,
        "derive": 0,
        "superseded": 2,
        "stored": 0,
    }
    assert [(i["tag"], i["kind"]) for i in campaign.items[0]] == [
        (PROSE_TAG, "scope"),
        (LIST_TAG, "scope"),
    ]


def test_rounds_continue_until_nothing_is_pending(campaign: Campaign) -> None:
    assert [r.returncode for r in campaign.rounds] == [3, 3, 3, 0]
    by_round = [(ln["complete"], ln["scope"], ln["extract"], ln["stored"]) for ln in campaign.lines]
    # Round 2: the prose extract and the list's first batch; round 3: the list's second
    # batch, once the first is answered; round 4: both probes complete.
    assert by_round == [(0, 2, 0, 0), (0, 0, 2, 2), (1, 0, 1, 4), (2, 0, 0, 5)]
    assert [(i["tag"], i["kind"]) for i in campaign.items[2]] == [(LIST_TAG, "extract")]
    assert campaign.items[3] == []
    assert "nothing is pending" in campaign.layout.pending.read_text(encoding="utf-8")


def test_no_round_offers_a_repair_or_derive_request(campaign: Campaign) -> None:
    assert all(ln["repair"] == 0 and ln["derive"] == 0 for ln in campaign.lines)
    assert all(i["kind"] in ("scope", "extract") for items in campaign.items for i in items)


def test_add_refuses_a_schema_invalid_answer_and_one_holding_the_channel_marker(
    campaign: Campaign,
) -> None:
    result = campaign.refused_add
    assert result.returncode == 3
    lines = result.stdout.splitlines()
    assert lines[0] == "added 0, refused 2, stored 2"
    key12 = campaign.items[1][0]["key"][:12]
    assert lines[1].startswith(f"refused {key12} $")
    assert "is not of type 'array'" in lines[1] or "is a required property" in lines[1]
    assert lines[2].startswith(f"refused {key12}") and "<|channel|>" in lines[2]
    assert len(lines) == 3


def test_add_appends_valid_answers_and_prints_its_counts(campaign: Campaign) -> None:
    assert [a.returncode for a in campaign.adds] == [0, 0, 0]
    assert [a.stdout for a in campaign.adds] == [
        "added 2, refused 0, stored 2\n",
        "added 2, refused 0, stored 4\n",
        "added 1, refused 0, stored 5\n",
    ]
    store = AnswerStore.load(campaign.layout.store, state_path=campaign.layout.state)
    assert [(e.round, e.kind) for e in store.entries] == [
        (1, "scope"),
        (1, "scope"),
        (2, "extract"),
        (2, "extract"),
        (3, "extract"),
    ]


def test_each_offer_names_a_prompt_and_a_schema_file_holding_the_request_s_own(
    campaign: Campaign,
) -> None:
    ceiling = campaign.layout.local_dir / "ceiling"
    scope_item, extract_item = campaign.items[0][0], campaign.items[1][0]
    assert json.loads((ceiling / scope_item["schema"]).read_text()) == SCOPE_OUTPUT_SCHEMA
    assert json.loads((ceiling / extract_item["schema"]).read_text()) == EXTRACTION_OUTPUT_SCHEMA
    assert (ceiling / scope_item["prompt"]).read_text(
        encoding="utf-8"
    ) == SCOPE_CLASSIFIER_SYSTEM_PROMPT
    extract_prompt = (ceiling / extract_item["prompt"]).read_text(encoding="utf-8")
    assert extract_prompt.startswith(active_pack().system_prompt[:200])
    assert len(list((ceiling / "prompts").iterdir())) == 2
    assert len(list((ceiling / "schemas").iterdir())) == 2


def test_the_pending_file_shows_every_non_system_message(campaign: Campaign) -> None:
    pending = campaign.shown.texts
    scope_pending = next(t for t in pending if PROSE_TAG in t and "- kind: scope" in t)
    assert 'Utterance: "My sister Priya moved to Lisbon last spring."' in scope_pending
    assert "#### user" in scope_pending
    assert "#### system" not in scope_pending


def test_stdout_is_one_counts_only_line_each_round(campaign: Campaign) -> None:
    for result in campaign.rounds:
        _parse_line(result.stdout)
        assert "{" not in result.stdout


def test_nothing_shown_holds_an_h3_marker_or_names_the_hidden_run_directory(
    campaign: Campaign,
) -> None:
    shown = "\n".join(campaign.shown.texts)
    assert shown.count("- key: `") == 5, "the record holds every offer's pending file"
    assert [m for m in H3_MARKERS if m in shown] == []
    assert HIDDEN_RUN_DIR not in shown


def test_the_hidden_run_directory_holds_each_round_s_bench_output(campaign: Campaign) -> None:
    run_dir = campaign.layout.local_dir / HIDDEN_RUN_DIR / "1"
    assert sorted(p.name for p in run_dir.iterdir()) == [f"round-{r}" for r in range(1, 5)]
    for tag in (PROSE_TAG, LIST_TAG):
        assert (run_dir / "round-4" / tag / "extraction_service_raw.jsonl").is_file()


def _copy_layout(campaign: Campaign, root: Path) -> Layout:
    shutil.copytree(campaign.layout.root, root)
    layout = Layout(
        root=root, probes=root / "probes.jsonl", store_dir=root / "data", local_dir=root / "local"
    )
    return layout


def test_a_truncated_store_stops_the_next_round(campaign: Campaign, tmp_path: Path) -> None:
    layout = _copy_layout(campaign, tmp_path / "copy")
    lines = layout.store.read_bytes().splitlines(keepends=True)
    layout.store.write_bytes(b"".join(lines[:-1]))
    result = _round(layout)
    assert result.returncode == 1
    assert result.stdout == ""
    assert "4 lines, the state records 5" in result.stderr
    assert HIDDEN_RUN_DIR not in result.stderr


def test_a_rewritten_last_line_stops_the_next_round(campaign: Campaign, tmp_path: Path) -> None:
    layout = _copy_layout(campaign, tmp_path / "copy")
    lines = layout.store.read_bytes().splitlines(keepends=True)
    assert b'"name":"syn-02-list"}' in lines[-1]
    lines[-1] = lines[-1].replace(b'"name":"syn-02-list"}', b'"name":"syn-02-lisT"}')
    layout.store.write_bytes(b"".join(lines))
    result = _round(layout)
    assert result.returncode == 1
    assert "changed since the state" in result.stderr
    assert _round(layout, "--add").returncode == 1


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_check_exits_0_at_the_end_and_writes_nothing_under_the_layout(
    campaign: Campaign,
) -> None:
    before = _tree(campaign.layout.root)
    result = _round(campaign.layout, "--check")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "check answerer 1: probes 2, complete 2, pending 0, stored 5\n"
    assert _tree(campaign.layout.root) == before
    assert [m for m in H3_MARKERS if m in result.stdout + result.stderr] == []
    assert HIDDEN_RUN_DIR not in result.stdout + result.stderr


def test_check_exits_1_while_requests_are_pending(tmp_path: Path) -> None:
    layout = make_layout(tmp_path / "fresh")
    result = _round(layout, "--check")
    assert result.returncode == 1
    assert result.stdout == "check answerer 1: probes 2, complete 0, pending 4, stored 0\n"
    assert not (layout.local_dir).exists()


def test_add_without_an_input_file_is_an_error(tmp_path: Path) -> None:
    layout = make_layout(tmp_path / "fresh")
    result = _round(layout, "--add")
    assert result.returncode == 1
    assert result.stdout == ""


def test_add_refuses_a_key_not_offered_and_a_line_that_is_not_an_answer(
    campaign: Campaign, tmp_path: Path
) -> None:
    layout = _copy_layout(campaign, tmp_path / "copy")
    stored_key = campaign.items[0][0]["key"]
    layout.answers_in.write_text(
        json.dumps({"key": stored_key, "answer": {"scope": "user-scope"}})
        + "\n"
        + json.dumps({"key": "0" * 64, "answer": {"scope": "user-scope"}})
        + "\n[1, 2]\n",
        encoding="utf-8",
    )
    result = _round(layout, "--add")
    assert result.returncode == 3
    lines = result.stdout.splitlines()
    assert lines[0] == "added 0, refused 3, stored 5"
    assert lines[1] == f"refused {stored_key[:12]} already has an answer"
    assert lines[2] == "refused 000000000000 was not offered in round 4"
    assert lines[3].startswith("refused line 3:")


# --- the replay --------------------------------------------------------------------------

ALTERED = "ALTERED-VALUE"


def _replay(*args: str) -> subprocess.CompletedProcess[str]:
    return _run_tool("scripts.eval_harness.ceiling.replay", *args)


def _replay_out(layout: Layout, out: Path, store_dir: Path | None = None):
    return _replay(
        "--out",
        str(out),
        "--probes",
        str(layout.probes),
        "--store-dir",
        str(store_dir or layout.store_dir),
        "--relay-port",
        "0",
        "--service-port",
        "0",
    )


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@dataclass
class Replays:
    first: Path
    second: Path
    results: tuple[subprocess.CompletedProcess[str], subprocess.CompletedProcess[str]]


@pytest.fixture(scope="module")
def replays(campaign: Campaign, tmp_path_factory: pytest.TempPathFactory) -> Replays:
    root = tmp_path_factory.mktemp("replay")
    first, second = root / "run-a-1", root / "run-a-2"
    results = (_replay_out(campaign.layout, first), _replay_out(campaign.layout, second))
    return Replays(first=first, second=second, results=results)


def test_two_replays_of_the_stored_answers_leave_nothing_pending(replays: Replays) -> None:
    for result in replays.results:
        assert result.returncode == 0, result.stderr
        assert result.stdout == "replay: probes 2, complete 2, requests 5, pending 0\n"
    for out in (replays.first, replays.second):
        assert [r["id"] for r in _jsonl(out / "extraction_service_raw.jsonl")] == [
            PROSE_TAG,
            LIST_TAG,
        ]


def test_the_replay_log_attributes_every_request_to_its_probe(
    campaign: Campaign, replays: Replays
) -> None:
    requests = _jsonl(replays.first / "requests.jsonl")
    assert [list(r) for r in requests] == [["seq", "tag", "key", "kind", "answered"]] * 5
    assert [(r["tag"], r["kind"]) for r in requests] == [
        (PROSE_TAG, "scope"),
        (PROSE_TAG, "extract"),
        (LIST_TAG, "scope"),
        (LIST_TAG, "extract"),
        (LIST_TAG, "extract"),
    ]
    assert all(r["answered"] for r in requests)
    stored = AnswerStore.load(campaign.layout.store).entries
    assert sorted(r["key"] for r in requests) == sorted(e.key for e in stored)


def test_two_replays_of_the_same_answers_compare_equal(replays: Replays) -> None:
    result = _replay("--compare", str(replays.first), str(replays.second))
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == "compare: equal, requests 5, rows 2\n"


def _altered_store(campaign: Campaign, store_dir: Path) -> None:
    """Rebuild the store through its own API with the list's last answer altered."""
    schemas = {"scope": SCOPE_OUTPUT_SCHEMA, "extract": EXTRACTION_OUTPUT_SCHEMA}
    entries = AnswerStore.load(campaign.layout.store).entries
    store_dir.mkdir()
    store = AnswerStore.load(store_dir / "answers-1.jsonl")
    store.offer(
        1, [OfferedRequest(key=e.key, kind=e.kind, schema=schemas[e.kind]) for e in entries]
    )
    for entry in entries:
        answer = entry.answer
        if entry.seq == len(entries):
            extra = {"id": "altered-entity", "type": "Thing", "name": ALTERED}
            answer = {**answer, "entities": [*answer["entities"], extra]}
        store.add(entry.key, answer)


def test_an_altered_answer_makes_compare_exit_1_and_name_the_tag(
    campaign: Campaign, replays: Replays, tmp_path: Path
) -> None:
    _altered_store(campaign, tmp_path / "altered-store")
    out = tmp_path / "run-altered"
    assert _replay_out(campaign.layout, out, tmp_path / "altered-store").returncode == 0
    result = _replay("--compare", str(replays.first), str(out))
    assert result.returncode == 1
    # The batch record holds each call's payload, so the altered answer moves it too.
    assert result.stdout == (
        f"compare: rows differ at tag {LIST_TAG}: response.batch_record, response.payload.entities\n"
    )
    assert ALTERED not in result.stdout + result.stderr


def test_a_differing_request_key_makes_compare_exit_1_without_the_key(
    replays: Replays, tmp_path: Path
) -> None:
    copy = tmp_path / "copy"
    shutil.copytree(replays.second, copy)
    requests = _jsonl(copy / "requests.jsonl")
    original = requests[1]["key"]
    requests[1]["key"] = "0" * 64
    (copy / "requests.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in requests), encoding="utf-8"
    )
    result = _replay("--compare", str(replays.first), str(copy))
    assert result.returncode == 1
    assert result.stdout == f"compare: requests differ at request 2, tag {PROSE_TAG}: key\n"
    assert original[:12] not in result.stdout and "0" * 12 not in result.stdout


def test_compare_exits_1_on_a_missing_replay(replays: Replays, tmp_path: Path) -> None:
    result = _replay("--compare", str(replays.first), str(tmp_path / "absent"))
    assert result.returncode == 1
    assert result.stdout == ""
    assert "requests.jsonl" in result.stderr


def test_replay_refuses_an_out_dir_that_holds_a_replay(
    campaign: Campaign, replays: Replays
) -> None:
    before = (replays.first / "requests.jsonl").read_bytes()
    result = _replay_out(campaign.layout, replays.first)
    assert result.returncode == 1
    assert "already" in result.stderr
    assert (replays.first / "requests.jsonl").read_bytes() == before


def test_replay_exits_1_while_answers_are_missing(tmp_path: Path) -> None:
    layout = make_layout(tmp_path / "fresh")
    result = _replay_out(layout, tmp_path / "out")
    assert result.returncode == 1
    assert result.stdout == "replay: probes 2, complete 0, requests 4, pending 4\n"


def test_replay_refuses_a_key_held_by_two_stores(campaign: Campaign, tmp_path: Path) -> None:
    store_dir = tmp_path / "stores"
    store_dir.mkdir()
    shutil.copy(campaign.layout.store, store_dir / "answers-1.jsonl")
    shutil.copy(campaign.layout.store, store_dir / "answers-2.jsonl")
    result = _replay_out(campaign.layout, tmp_path / "out", store_dir)
    assert result.returncode == 1
    assert "SharedKeyError" in result.stderr


def test_replay_defaults_to_the_four_committed_stores_and_the_probes_file() -> None:
    args = rp.parse_args(["--out", "x"])
    assert args.probes == _REPO_ROOT / "data" / "ceiling" / "probes.jsonl"
    assert args.store_dir == _REPO_ROOT / "data" / "ceiling"
    assert (args.relay_port, args.service_port) == (18080, 18090)
    assert tuple(f"answers-{k}.jsonl" for k in (1, 2, 3, 4)) == rp.STORE_NAMES
