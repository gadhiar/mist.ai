"""T14: the label schema, its projection onto the speaking states, and carrying labels forward.

Every test builds its own records and roots; no test reads private data or audio.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.voice_data import labels, states
from scripts.voice_data import root as root_mod
from scripts.voice_data.commands.extract import TurnRecord
from scripts.voice_data.errors import InputError
from scripts.voice_data.labels import LabelRecord, PrefillRecord
from scripts.voice_data.records import Prosody, SegmentRecord

# ---------------------------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------------------------


def segment(
    sid: str = "seg-000001",
    *,
    source: str = "src-0001",
    start: float = 100.0,
    end: float = 110.0,
    channel: str = "FC",
    separator: str = "melband",
    diar_cluster: str | None = "spk-01",
    overlap_ms: int = 0,
) -> SegmentRecord:
    return SegmentRecord(
        segment_id=sid, source_id=source, start_s=start, end_s=end, channel=channel,
        channel_reason=None, separator=separator, audio=f"seed/runs/r9/audio/{sid}.flac",
        diar_cluster=diar_cluster, cluster="primary", flag_group=None,
        sim={"wavlm-sv": 0.9}, overlap_ms=overlap_ms, snr_db=30.0, c50_db=40.0,
        transcripts={}, asr_agreement_cer=0.0,
        prosody=Prosody(f0_mean_hz=210.0, f0_std_hz=4.0, rate_sps=4.0, energy_db=-20.0),
        arousal=0.4, state_proposed="neutral", state=None, set="short", kept=True, reasons=(),
    )  # fmt: skip


def turn(start: float, end: float, *, role: str = "other", source: str = "src-0001") -> TurnRecord:
    return TurnRecord(
        source_id=source, turn=0, start_s=start, end_s=end, diar_cluster="spk-02", role=role,
        embedded=True, touches_overlap=False, sim_anchor=None, sim_target=None, gaps=None,
        passed=None, reasons=(),
    )  # fmt: skip


def label(
    sid: str = "seg-000001",
    *,
    run: str = "r1",
    source: str = "src-0001",
    start: float = 100.0,
    end: float = 110.0,
    channel: str = "FC",
    separator: str = "melband",
    voices: str = "her",
    energy: int = 3,
    mood: int = 3,
    tags: tuple[str, ...] = (),
    keep: tuple[float, float] | None = None,
    has_note: bool = False,
    state: str | None = None,
    removed: str | None = None,
    carried_from: tuple[str, ...] = (),
    duration: float | None = None,
) -> LabelRecord:
    return LabelRecord(
        run=run, segment_id=sid, source_id=source, start_s=start, end_s=end, channel=channel,
        separator=separator, duration_s=round(end - start, 2) if duration is None else duration,
        voices=voices, energy=energy, mood=mood, tags=tags,
        keep_from_s=None if keep is None else keep[0], keep_to_s=None if keep is None else keep[1],
        has_note=has_note, state=state, removed=removed, carried_from=carried_from,
    )  # fmt: skip


def prefill(
    sid: str, *, energy: int = 3, mood: int = 3, voices: str = "her", duration: float = 4.0
) -> PrefillRecord:
    return PrefillRecord(
        segment_id=sid, source_id="src-0001", start_s=10.0, end_s=10.0 + duration,
        duration_s=duration, arousal=0.5, dominance=0.5, valence=0.5, energy=energy, mood=mood,
        voices=voices, keep_from_s=None, keep_to_s=None, other_s=0.0, overlap_ms=0,
    )  # fmt: skip


@pytest.fixture()
def root(tmp_path: Path) -> root_mod.Root:
    return root_mod.init_root(tmp_path / "voice-root", "private")


def write(root: root_mod.Root, run: str, records: list[LabelRecord], replace_: bool = False):
    return labels.write_labels(root, run, records, {}, {"clips": len(records)}, replace=replace_)


# ---------------------------------------------------------------------------------------------
# (1) Vocabularies, paths and constants
# ---------------------------------------------------------------------------------------------


def test_vocabularies_match_interfaces() -> None:
    assert labels.VOICES == ("her", "her_overlap", "her_nearby", "not_her")
    assert labels.MODES == ("whispered", "laughing", "sighing", "shouted")
    assert labels.ATTITUDES == (
        "warm", "playful", "teasing", "sarcastic", "tender", "confident", "curious",
        "surprised", "worried", "sad", "annoyed", "tired",
    )  # fmt: skip
    assert labels.TAGS == labels.MODES + labels.ATTITUDES
    assert len(labels.TAGS) == len(set(labels.TAGS)) == 16
    assert labels.LOW_TAGS == ("tired", "sad", "annoyed", "worried")
    assert set(labels.LOW_TAGS) <= set(labels.ATTITUDES)
    assert labels.MAX_TAGS == 3
    assert labels.NOTE_MAX == 200
    assert labels.REMOVED == ("not-her", "flag-group", "overlap", "nearby")


def test_paths_and_constants_match_interfaces() -> None:
    assert labels.labels_rel("seed-r2") == "seed/labels/seed-r2"
    assert labels.prefill_rel("seed-r2") == "seed/prefill/seed-r2"
    assert labels.LABELS_NAME == "labels.jsonl"
    assert labels.NOTES_NAME == "notes.private.jsonl"
    assert labels.REPORT_NAME == "report.json"
    assert labels.LEGACY_LABELS_REL == "seed/labels.csv"
    assert labels.PREFILL_NAME == "prefill.jsonl"
    assert labels.PREFILL_PLAN_NAME == "prefill-plan.json"
    assert labels.CARRY_SHARE == 0.9
    assert labels.CARRY_TOL_S == 0.01
    assert labels.OTHER_MIN_S == 0.1
    assert Path(labels.__file__).with_name("delivery.json") == labels.DELIVERY_PATH


@pytest.mark.parametrize("run", ["", "..", "a/b", "a\\b", ".hidden", "../x"])
def test_run_paths_refuse_anything_but_one_folder_name(run: str) -> None:
    with pytest.raises(InputError):
        labels.labels_rel(run)
    with pytest.raises(InputError):
        labels.prefill_rel(run)


# ---------------------------------------------------------------------------------------------
# (2) delivery.json and load_delivery
# ---------------------------------------------------------------------------------------------


def delivery_data() -> dict:
    return json.loads(labels.DELIVERY_PATH.read_text(encoding="utf-8"))


def load_from(tmp_path: Path, data: object) -> labels.Delivery:
    path = tmp_path / "delivery.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return labels.load_delivery(path)


def test_delivery_states_are_pinned_to_the_speaking_states() -> None:
    delivery = labels.load_delivery()
    assert sorted(r.state for r in delivery.regions) == sorted(states.SPEAKING_STATES)
    assert len(delivery.regions) == len(states.SPEAKING_STATES)


def test_delivery_keeps_design_part_3_order() -> None:
    names = [r.state for r in labels.load_delivery().regions]
    assert names == ["sleepy", "excited", "calm", "awake", "neutral"]


def test_each_example_is_tagless_and_projects_to_its_own_state() -> None:
    delivery = labels.load_delivery()
    for entry in delivery_data()["states"]:
        assert set(entry["example"]) == {"energy", "mood"}
    for region in delivery.regions:
        energy, mood = region.example
        assert labels.project(delivery, energy, mood, ()) == region.state


def test_delivery_records_the_prefill_formulas_as_estimates() -> None:
    data = delivery_data()
    assert data["schema"] == 1
    assert set(data) == {"schema", "prefill", "states"}
    prefill_ = data["prefill"]
    assert "1 + floor(4 x arousal + 0.5)" in prefill_["energy"]
    assert "1 + floor(4 x valence + 0.5)" in prefill_["mood"]
    assert "estimate" in prefill_["note"].lower()
    assert labels.load_delivery().prefill == prefill_


def _mutate(path: tuple, value: object, *, delete: bool = False):
    def apply(data: dict) -> dict:
        target = data
        for key in path[:-1]:
            target = target[key]
        if delete:
            del target[path[-1]]
        else:
            target[path[-1]] = value
        return data

    return apply


def _state(name: str):
    def find(data: dict) -> int:
        return next(i for i, e in enumerate(data["states"]) if e["state"] == name)

    return find


def _swap_states(data: dict) -> dict:
    data["states"][0]["state"] = "angry"
    return data


def _duplicate_state(data: dict) -> dict:
    data["states"][4] = dict(data["states"][0])
    return data


def _drop_state(data: dict) -> dict:
    data["states"].pop()
    return data


def _clause(state: str, index: int, key: str, value: object):
    def apply(data: dict) -> dict:
        data["states"][_state(state)(data)]["any"][index][key] = value
        return data

    return apply


def _example(state: str, value: object):
    def apply(data: dict) -> dict:
        data["states"][_state(state)(data)]["example"] = value
        return data

    return apply


def _entry_key(state: str, key: str, value: object):
    def apply(data: dict) -> dict:
        data["states"][_state(state)(data)][key] = value
        return data

    return apply


MALFORMED = {
    "schema 2": _mutate(("schema",), 2),
    "unknown top key": _mutate(("extra",), 1),
    "no prefill": _mutate(("prefill",), None, delete=True),
    "prefill not text": _mutate(("prefill", "energy"), 4),
    "states not a list": _mutate(("states",), {}),
    "unknown state": _swap_states,
    "duplicate state": _duplicate_state,
    "missing state": _drop_state,
    "unknown entry key": _entry_key("calm", "extra", 1),
    "empty any": _entry_key("calm", "any", []),
    "tag outside TAGS in tags_any": _clause("sleepy", 1, "tags_any", ["sleepy"]),
    "tag outside TAGS in tags_none": _clause("calm", 0, "tags_none", ["grumpy"]),
    "empty tags_any": _clause("sleepy", 1, "tags_any", []),
    "duplicate tag": _clause("calm", 0, "tags_none", ["sad", "sad"]),
    "energy bound 0": _clause("sleepy", 0, "energy", [0, 1]),
    "energy bound 6": _clause("excited", 0, "energy", [5, 6]),
    "mood bound 0": _clause("calm", 0, "mood", [0, 5]),
    "mood bound 6": _clause("calm", 0, "mood", [3, 6]),
    "low above high": _clause("calm", 0, "mood", [5, 3]),
    "bound not a pair": _clause("calm", 0, "mood", [3]),
    "bound not an integer": _clause("calm", 0, "mood", [3, 4.5]),
    "bound a bool": _clause("calm", 0, "mood", [True, 5]),
    "unknown clause key": _clause("calm", 0, "tags_all", ["warm"]),
    "example projects elsewhere": _example("calm", {"energy": 3, "mood": 3}),
    "example outside every region": _example("neutral", {"energy": 3, "mood": 1}),
    "example with tags": _example("neutral", {"energy": 3, "mood": 3, "tags": []}),
    "example energy 6": _example("excited", {"energy": 6, "mood": 5}),
}


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_load_delivery_refuses_each_malformed_case(tmp_path: Path, case: str) -> None:
    data = MALFORMED[case](delivery_data())
    with pytest.raises(InputError):
        load_from(tmp_path, data)


def test_load_delivery_refuses_a_missing_or_unparsable_file(tmp_path: Path) -> None:
    with pytest.raises(InputError):
        labels.load_delivery(tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(InputError):
        labels.load_delivery(bad)
    with pytest.raises(InputError):
        load_from(tmp_path, [])


def test_a_clause_with_no_keys_holds_every_point(tmp_path: Path) -> None:
    data = delivery_data()
    # Put an unconstrained neutral clause last: every clip outside the other four is neutral.
    data["states"][_state("neutral")(data)]["any"] = [{}]
    delivery = load_from(tmp_path, data)
    assert labels.project(delivery, 3, 1, ("annoyed",)) == "neutral"
    assert labels.project(delivery, 4, 4, ("shouted",)) == "neutral"
    assert labels.project(delivery, 1, 1, ()) == "sleepy"


# ---------------------------------------------------------------------------------------------
# (3) project
# ---------------------------------------------------------------------------------------------

EDGES = [
    # sleepy: energy 1 with anything; energy 2 with tired or whispered
    (1, 1, (), "sleepy"),
    (1, 5, (), "sleepy"),
    (1, 3, ("laughing", "warm"), "sleepy"),
    (2, 3, ("tired",), "sleepy"),
    (2, 1, ("whispered",), "sleepy"),
    (2, 5, ("whispered", "tired"), "sleepy"),
    (3, 3, ("tired",), None),
    (3, 3, ("whispered",), None),
    # excited: energy 5 and mood 4-5, no low tag; energy 4, mood 5, laughing, no low tag
    (5, 4, (), "excited"),
    (5, 5, (), "excited"),
    (5, 3, (), None),
    (5, 4, ("sad",), None),
    (5, 5, ("tired",), None),
    (4, 5, ("laughing",), "excited"),
    (4, 5, ("laughing", "warm"), "excited"),
    (4, 4, ("laughing",), None),
    (4, 5, ("laughing", "worried"), None),
    (3, 5, ("laughing",), None),
    # calm: energy 2, mood 3-5, no mode, no low tag
    (2, 3, (), "calm"),
    (2, 5, (), "calm"),
    (2, 4, ("warm", "tender"), "calm"),
    (2, 2, (), None),
    (2, 4, ("sighing",), None),
    (2, 4, ("laughing",), None),
    (2, 4, ("worried",), None),
    (2, 4, ("annoyed",), None),
    # awake: energy 4, mood 3-5, no mode, no low tag
    (4, 3, (), "awake"),
    (4, 5, (), "awake"),
    (4, 4, ("playful", "curious"), "awake"),
    (4, 2, (), None),
    (4, 4, ("whispered",), None),
    (4, 4, ("sad",), None),
    (5, 3, ("confident",), None),
    # neutral: energy 3, mood 3-4, no mode, no low tag
    (3, 3, (), "neutral"),
    (3, 4, (), "neutral"),
    (3, 3, ("confident",), "neutral"),
    (3, 2, (), None),
    (3, 5, (), None),
    (3, 3, ("sighing",), None),
    (3, 4, ("tired",), None),
]


@pytest.mark.parametrize(("energy", "mood", "tags", "state"), EDGES)
def test_project_region_edges(energy: int, mood: int, tags: tuple, state: str | None) -> None:
    assert labels.project(labels.load_delivery(), energy, mood, tags) == state


@pytest.mark.parametrize(
    ("energy", "mood", "tags"),
    [(3, 2, ("annoyed",)), (3, 5, ("warm",))] + [(4, m, ("shouted",)) for m in range(1, 6)],
)
def test_design_part_3_none_examples(energy: int, mood: int, tags: tuple) -> None:
    assert labels.project(labels.load_delivery(), energy, mood, tags) is None


@pytest.mark.parametrize(
    ("energy", "mood", "tags", "state"),
    [
        (1, 3, ("whispered",), "sleepy"),
        (1, 3, ("sad",), "sleepy"),
        (1, 1, ("whispered", "sad"), "sleepy"),
        (5, 4, ("shouted",), "excited"),
    ],
)
def test_the_two_deliberate_admissions(energy: int, mood: int, tags: tuple, state: str) -> None:
    assert labels.project(labels.load_delivery(), energy, mood, tags) == state


def test_project_takes_tags_in_any_order() -> None:
    delivery = labels.load_delivery()
    assert labels.project(delivery, 4, 5, ("warm", "laughing")) == "excited"


def all_points():
    for size in range(labels.MAX_TAGS + 1):
        for tags in itertools.combinations(labels.TAGS, size):
            for energy in range(1, 6):
                for mood in range(1, 6):
                    yield energy, mood, tags


def test_no_two_delivery_regions_share_a_point() -> None:
    delivery = labels.load_delivery()
    shared = []
    for energy, mood, tags in all_points():
        holding = [r.state for r in delivery.regions if r.holds(energy, mood, tags)]
        if len(holding) > 1:
            shared.append((energy, mood, tags, holding))
    assert shared == []


def overlapping(first: str) -> dict:
    """Five regions where calm (energy 2-3) and neutral (energy 3-4, mood 3) share (3, 3)."""
    regions = {
        "calm": {"state": "calm", "example": {"energy": 2, "mood": 3},
                 "any": [{"energy": [2, 3]}]},
        "neutral": {"state": "neutral", "example": {"energy": 4, "mood": 3},
                    "any": [{"energy": [3, 4], "mood": [3, 3]}]},
        "sleepy": {"state": "sleepy", "example": {"energy": 1, "mood": 3},
                   "any": [{"energy": [1, 1]}]},
        "excited": {"state": "excited", "example": {"energy": 5, "mood": 5},
                    "any": [{"energy": [5, 5]}]},
        "awake": {"state": "awake", "example": {"energy": 4, "mood": 5},
                  "any": [{"energy": [4, 4], "mood": [4, 5]}]},
    }  # fmt: skip
    second = "neutral" if first == "calm" else "calm"
    order = [first, second, "sleepy", "excited", "awake"]
    return {
        "schema": 1,
        "prefill": delivery_data()["prefill"],
        "states": [regions[s] for s in order],
    }


def test_first_match_in_file_order_decides_an_overlap(tmp_path: Path) -> None:
    calm_first = load_from(tmp_path, overlapping("calm"))
    assert labels.project(calm_first, 3, 3, ()) == "calm"
    neutral_first = load_from(tmp_path, overlapping("neutral"))
    assert labels.project(neutral_first, 3, 3, ()) == "neutral"
    # A point only one region holds is unaffected by the order.
    assert labels.project(calm_first, 4, 3, ()) == "neutral"
    assert labels.project(neutral_first, 2, 1, ()) == "calm"


@pytest.mark.parametrize(
    ("energy", "mood", "tags"),
    [
        (0, 3, ()),
        (6, 3, ()),
        (3, 0, ()),
        (3, 6, ()),
        (3.0, 3, ()),
        (True, 3, ()),
        (3, 3, ("grumpy",)),
        (3, 3, ("warm", "warm")),
        (3, 3, ("warm", "playful", "teasing", "tender")),
    ],
)
def test_project_refuses_values_outside_the_schema(energy, mood, tags) -> None:
    with pytest.raises(InputError):
        labels.project(labels.load_delivery(), energy, mood, tags)


# ---------------------------------------------------------------------------------------------
# (4) prefill_voices
# ---------------------------------------------------------------------------------------------

MIN_S = 3.0


def test_turn_cut_segment_beside_another_clusters_turn_stays_her() -> None:
    seg = segment(start=100.0, end=110.0, diar_cluster="spk-01")
    turns = [turn(95.0, 100.0), turn(110.0, 112.0), turn(100.0, 110.0, role="primary")]
    assert labels.prefill_voices(seg, turns, MIN_S) == ("her", None, None, 0.0)


def test_turn_cut_segment_is_her_and_still_reports_other_seconds() -> None:
    seg = segment(start=100.0, end=110.0, diar_cluster="spk-01", overlap_ms=300)
    assert labels.prefill_voices(seg, [turn(108.0, 112.0)], MIN_S) == ("her", None, None, 2.0)


def test_off_centre_segment_with_overlap_is_her_overlap() -> None:
    seg = segment(diar_cluster=None, overlap_ms=120)
    assert labels.prefill_voices(seg, [turn(100.0, 101.5)], MIN_S) == (
        "her_overlap", None, None, 1.5,
    )  # fmt: skip


@pytest.mark.parametrize("seconds", [0.05, 0.1])
def test_off_centre_segment_with_little_other_speech_is_her(seconds: float) -> None:
    seg = segment(diar_cluster=None)
    turns = [turn(104.0, 104.0 + seconds)]
    assert labels.prefill_voices(seg, turns, MIN_S) == ("her", None, None, seconds)


def test_other_seconds_at_the_limit_sum_over_turns() -> None:
    seg = segment(diar_cluster=None)
    turns = [turn(101.0, 101.05), turn(105.0, 105.05)]
    assert labels.prefill_voices(seg, turns, MIN_S) == ("her", None, None, 0.1)


def test_off_centre_segment_with_more_other_speech_is_her_nearby_with_a_range() -> None:
    seg = segment(start=100.0, end=110.0, diar_cluster=None)
    turns = [turn(98.0, 100.5), turn(107.0, 107.2)]
    # Uncovered: 100.5-107.0 (6.5 s) and 107.2-110.0 (2.8 s); the longer one, clip-relative.
    assert labels.prefill_voices(seg, turns, MIN_S) == ("her_nearby", 0.5, 7.0, 0.7)


def test_her_nearby_range_is_rounded_to_hundredths() -> None:
    seg = segment(start=100.0, end=110.0, diar_cluster=None)
    turns = [turn(100.0, 102.333), turn(108.0071, 110.0)]
    assert labels.prefill_voices(seg, turns, MIN_S) == ("her_nearby", 2.33, 8.01, 4.326)


def test_her_nearby_range_is_null_when_the_longest_stretch_is_under_min_s() -> None:
    seg = segment(start=100.0, end=106.0, diar_cluster=None)
    turns = [turn(101.5, 102.5), turn(104.5, 106.0)]
    assert labels.prefill_voices(seg, turns, MIN_S) == ("her_nearby", None, None, 2.5)


def test_her_nearby_range_of_exactly_min_s_is_kept() -> None:
    seg = segment(start=100.0, end=104.0, diar_cluster=None)
    assert labels.prefill_voices(seg, [turn(103.0, 104.0)], MIN_S) == (
        "her_nearby", 0.0, 3.0, 1.0,
    )  # fmt: skip


def test_overlapping_other_turns_count_once() -> None:
    seg = segment(start=100.0, end=110.0, diar_cluster=None)
    turns = [turn(100.0, 101.0), turn(100.5, 101.5)]
    assert labels.prefill_voices(seg, turns, MIN_S) == ("her_nearby", 1.5, 10.0, 1.5)


def test_turns_of_another_source_or_role_are_ignored() -> None:
    seg = segment(diar_cluster=None)
    turns = [
        turn(100.0, 110.0, source="src-0002"),
        turn(100.0, 110.0, role="primary"),
        turn(100.0, 110.0, role="flagged"),
    ]
    assert labels.prefill_voices(seg, turns, MIN_S) == ("her", None, None, 0.0)


# ---------------------------------------------------------------------------------------------
# (5) scale_from
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "scale"),
    [
        (0.0, 1),
        (0.125, 2),
        (0.5, 3),
        (0.625, 4),
        # 1 + floor(4 x 0.875 + 0.5) = 1 + floor(4.0) = 5 (Decision 3's formula).
        (0.875, 5),
        (1.0, 5),
        (0.1249, 1),
        (0.3749, 2),
        (0.8749, 4),
        (-0.4, 1),
        (1.7, 5),
        (-math.inf, 1),
        (math.inf, 5),
    ],
)
def test_scale_from(value: float, scale: int) -> None:
    assert labels.scale_from(value) == scale


def test_scale_from_rounds_halves_up_where_round_would_not() -> None:
    assert round(4 * 0.125) == 0 and labels.scale_from(0.125) == 2
    assert round(4 * 0.625) == 2 and labels.scale_from(0.625) == 4


def test_scale_from_refuses_nan() -> None:
    with pytest.raises(InputError):
        labels.scale_from(math.nan)


# ---------------------------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------------------------


def test_label_record_round_trips() -> None:
    record = label(
        voices="her_nearby", tags=("laughing", "warm"), keep=(0.5, 7.25), has_note=True,
        state="calm", carried_from=("r0/seg-000002", "r1/seg-000001"),
    )  # fmt: skip
    data = json.loads(json.dumps(record.to_json()))
    assert LabelRecord.from_json(data) == record


def test_prefill_record_round_trips() -> None:
    record = replace(
        prefill("seg-000004", voices="her_nearby", duration=8.0), keep_from_s=1.0, keep_to_s=6.5
    )
    assert PrefillRecord.from_json(json.loads(json.dumps(record.to_json()))) == record


BAD_LABELS = {
    "unknown voices": {"voices": "them"},
    "energy 0": {"energy": 0},
    "mood 6": {"mood": 6},
    "energy float": {"energy": 3.0},
    "tag outside TAGS": {"tags": ("grumpy",)},
    "tags out of vocabulary order": {"tags": ("warm", "laughing")},
    "duplicate tags": {"tags": ("warm", "warm")},
    "four tags": {"tags": ("whispered", "warm", "playful", "teasing")},
    "range on her": {"keep_from_s": 1.0, "keep_to_s": 5.0},
    "half a range": {"voices": "her_nearby", "keep_from_s": 1.0},
    "range outside clip": {"voices": "her_nearby", "keep_from_s": 1.0, "keep_to_s": 10.5},
    "range reversed": {"voices": "her_nearby", "keep_from_s": 5.0, "keep_to_s": 1.0},
    "unknown state": {"state": "angry"},
    "unknown removed": {"removed": "gone"},
    "state on a removed clip": {"removed": "overlap", "state": "neutral"},
    "carried_from unsorted": {"carried_from": ("r2/seg-000001", "r1/seg-000001")},
    "carried_from not run/segment": {"carried_from": ("seg-000001",)},
    "has_note not bool": {"has_note": 1},
    "span reversed": {"start_s": 120.0},
    "empty run": {"run": ""},
}


@pytest.mark.parametrize("case", sorted(BAD_LABELS))
def test_label_record_refuses_each_malformed_value(case: str) -> None:
    with pytest.raises(InputError):
        replace(label(), **BAD_LABELS[case])


BAD_PREFILLS = {
    "arousal above 1": {"arousal": 1.2},
    "valence below 0": {"valence": -0.1},
    "energy 6": {"energy": 6},
    "voices not_her": {"voices": "not_her"},
    "range on her": {"keep_from_s": 0.0, "keep_to_s": 3.0},
    "negative other_s": {"other_s": -1.0},
    "overlap_ms float": {"overlap_ms": 1.5},
}


@pytest.mark.parametrize("case", sorted(BAD_PREFILLS))
def test_prefill_record_refuses_each_malformed_value(case: str) -> None:
    with pytest.raises(InputError):
        replace(prefill("seg-000001"), **BAD_PREFILLS[case])


# ---------------------------------------------------------------------------------------------
# (6) read_labels and write_labels
# ---------------------------------------------------------------------------------------------


def test_read_labels_returns_none_for_a_run_without_labels(root: root_mod.Root) -> None:
    assert labels.read_labels(root, "r1") is None


def test_write_then_read_labels_per_run(root: root_mod.Root) -> None:
    one = [label("seg-000001", run="r1"), label("seg-000002", run="r1", energy=2, mood=4)]
    two = [label("seg-000001", run="r2", energy=5, mood=5)]
    write(root, "r1", one)
    write(root, "r2", two)
    assert labels.read_labels(root, "r1") == tuple(one)
    assert labels.read_labels(root, "r2") == tuple(two)


def test_write_labels_writes_the_three_files(root: root_mod.Root) -> None:
    records = [label("seg-000001", has_note=True), label("seg-000002")]
    tables = labels.tables(records, None)
    labels.write_labels(root, "r1", records, {"seg-000001": "a short note"}, tables)
    folder = root.path / "seed/labels/r1"
    assert sorted(p.name for p in folder.iterdir()) == [
        "labels.jsonl", "notes.private.jsonl", "report.json",
    ]  # fmt: skip
    notes = [json.loads(line) for line in (folder / "notes.private.jsonl").read_text().splitlines()]
    assert notes == [{"segment_id": "seg-000001", "note": "a short note"}]
    report = json.loads((folder / "report.json").read_text())
    assert report == {"schema": 1, "run": "r1", "pack": None, "tables": tables}
    assert "a short note" not in (folder / "labels.jsonl").read_text()
    assert "a short note" not in (folder / "report.json").read_text()


def test_write_labels_records_the_pack(root: root_mod.Root) -> None:
    labels.write_labels(root, "r1", [label()], {}, {}, pack="c04240d3")
    report = json.loads((root.path / "seed/labels/r1/report.json").read_text())
    assert report["pack"] == "c04240d3"


def test_write_labels_writes_each_file_through_the_root(
    root: root_mod.Root, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []
    real = root_mod.Root.write_bytes

    def spy(self, target, data):
        seen.append(Path(target).as_posix())
        return real(self, target, data)

    monkeypatch.setattr(root_mod.Root, "write_bytes", spy)
    write(root, "r1", [label()])
    assert sorted(seen) == [
        "seed/labels/r1/labels.jsonl", "seed/labels/r1/notes.private.jsonl",
        "seed/labels/r1/report.json",
    ]  # fmt: skip


def test_write_labels_refuses_an_existing_folder_without_replace(root: root_mod.Root) -> None:
    write(root, "r1", [label(energy=2)])
    with pytest.raises(InputError, match="replace"):
        write(root, "r1", [label(energy=4)])
    assert labels.read_labels(root, "r1") == (label(energy=2),)
    write(root, "r1", [label(energy=4)], replace_=True)
    assert labels.read_labels(root, "r1") == (label(energy=4),)


def test_write_labels_refuses_an_empty_existing_folder_without_replace(
    root: root_mod.Root,
) -> None:
    (root.path / "seed/labels/r1").mkdir(parents=True)
    with pytest.raises(InputError):
        write(root, "r1", [label()])


@pytest.mark.parametrize(
    ("records", "notes"),
    [
        ([label(run="r2")], {}),
        ([label("seg-000001"), label("seg-000001")], {}),
        ([label(has_note=True)], {}),
        ([label()], {"seg-000001": "note"}),
        ([label(has_note=True)], {"seg-000001": "x" * 201}),
        ([label(has_note=True)], {"seg-000001": "two\nlines"}),
        ([label(has_note=True)], {"seg-000001": "   "}),
    ],
)
def test_write_labels_refuses_inconsistent_records_or_notes(
    root: root_mod.Root, records: list, notes: dict
) -> None:
    with pytest.raises(InputError):
        labels.write_labels(root, "r1", records, notes, {})
    assert not (root.path / "seed/labels/r1").exists()


def test_a_note_of_note_max_characters_is_accepted(root: root_mod.Root) -> None:
    labels.write_labels(root, "r1", [label(has_note=True)], {"seg-000001": "x" * 200}, {})


def _write_raw(root: root_mod.Root, run: str, rows: list[dict]) -> None:
    folder = root.path / "seed/labels" / run
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_read_labels_refuses_a_record_of_another_run(root: root_mod.Root) -> None:
    _write_raw(root, "r1", [label(run="r2").to_json()])
    with pytest.raises(InputError, match="r2"):
        labels.read_labels(root, "r1")


@pytest.mark.parametrize(
    "row",
    [
        {**label().to_json(), "energy": 7},
        {**label().to_json(), "extra": 1},
        {k: v for k, v in label().to_json().items() if k != "mood"},
        {**label().to_json(), "tags": ["warm", "laughing"]},
    ],
)
def test_read_labels_refuses_a_malformed_record(root: root_mod.Root, row: dict) -> None:
    _write_raw(root, "r1", [row])
    with pytest.raises(InputError):
        labels.read_labels(root, "r1")


def test_read_labels_refuses_a_duplicate_segment(root: root_mod.Root) -> None:
    _write_raw(root, "r1", [label().to_json(), label().to_json()])
    with pytest.raises(InputError):
        labels.read_labels(root, "r1")


def test_every_read_refuses_while_the_legacy_csv_exists(root: root_mod.Root) -> None:
    write(root, "r1", [label()])
    (root.path / "seed/labels.csv").write_text("segment_id,state\n")
    with pytest.raises(InputError, match="pack labels"):
        labels.read_labels(root, "r1")
    with pytest.raises(InputError, match="pack labels"):
        labels.read_labels(root, "r-absent")
    with pytest.raises(InputError, match="pack labels"):
        labels.earlier_labels(root, "r2")
    with pytest.raises(InputError, match="pack labels"):
        labels.carried_stale(root, [label(run="r2", carried_from=("r1/seg-000001",))])


# ---------------------------------------------------------------------------------------------
# (7) states.py's docstring
# ---------------------------------------------------------------------------------------------


def test_states_docstring_names_the_labels_and_the_projection() -> None:
    doc = states.__doc__ or ""
    assert doc.startswith("The speaking states the voice data is labelled and generated in.")
    assert "labels.py" in doc and "delivery.json" in doc
    assert "mist-realtime-voice" in doc
    assert states.NEUTRAL_STATE == "neutral"
    assert states.SPEAKING_STATES == ("neutral", "calm", "awake", "sleepy", "excited")


# ---------------------------------------------------------------------------------------------
# (8) tables and table_lines
# ---------------------------------------------------------------------------------------------


def hand_built() -> list[LabelRecord]:
    def lab(n: int, seconds: float, **kw) -> LabelRecord:
        return label(f"seg-{n:06d}", run="r2", start=10.0 * n, end=10.0 * n + seconds, **kw)

    return [
        lab(1, 4.0, state="neutral", has_note=True),
        lab(2, 5.0, energy=2, mood=4, tags=("warm",), state="calm"),
        lab(3, 6.0, voices="her_nearby", keep=(1.0, 4.5), energy=4, mood=4,
            tags=("playful", "curious"), state="awake"),
        lab(4, 3.5, voices="her_overlap", removed="overlap"),
        lab(5, 4.5, voices="not_her", mood=2, removed="not-her"),
        lab(6, 3.0, mood=2, removed="flag-group"),
        lab(7, 4.0, voices="her_nearby", energy=2, removed="nearby"),
        lab(8, 5.0, energy=4, tags=("shouted",), carried_from=("r1/seg-000003",)),
        lab(9, 3.2, energy=1, state="sleepy",
            carried_from=("r0/seg-000001", "r1/seg-000009")),
    ]  # fmt: skip


def hand_prefill() -> list[PrefillRecord]:
    return [
        prefill("seg-000001"),
        prefill("seg-000002", energy=3, mood=4),
        prefill("seg-000003", energy=4, mood=2, voices="her_nearby"),
        prefill("seg-000004"),
        prefill("seg-000005"),
        prefill("seg-000006", mood=2),
        prefill("seg-000007", energy=2, voices="her_nearby"),
        # A carried clip's pre-fill is not part of the agreement.
        prefill("seg-000008", energy=1, mood=1, voices="her_overlap"),
    ]


def test_tables_who_is_audible() -> None:
    t = labels.tables(hand_built(), None)
    assert t["clips"] == {"count": 9, "seconds": 38.2}
    assert t["voices"] == {
        "her": {"count": 5, "seconds": 20.2},
        "her_overlap": {"count": 1, "seconds": 3.5},
        "her_nearby": {"count": 2, "seconds": 10.0},
        "not_her": {"count": 1, "seconds": 4.5},
    }
    assert t["removed"] == {
        "not-her": {"count": 1, "seconds": 4.5},
        "flag-group": {"count": 1, "seconds": 3.0},
        "overlap": {"count": 1, "seconds": 3.5},
        "nearby": {"count": 1, "seconds": 4.0},
    }
    assert t["trimmed_s"] == 2.5


def test_tables_energy_by_mood_grid_with_marginals() -> None:
    grid = labels.tables(hand_built(), None)["energy_mood"]
    counts = [[0] * 5 for _ in range(5)]
    seconds = [[0.0] * 5 for _ in range(5)]
    for energy, mood, s in [(3, 3, 4.0), (2, 4, 5.0), (4, 4, 3.5), (4, 3, 5.0), (1, 3, 3.2)]:
        counts[energy - 1][mood - 1] += 1
        seconds[energy - 1][mood - 1] += s
    assert grid["count"] == counts
    assert grid["seconds"] == seconds
    assert grid["energy"] == {"count": [1, 1, 1, 2, 0], "seconds": [3.2, 5.0, 4.0, 8.5, 0.0]}
    assert grid["mood"] == {"count": [0, 0, 3, 2, 0], "seconds": [0.0, 0.0, 12.2, 8.5, 0.0]}


def test_tables_tags_and_untagged() -> None:
    t = labels.tables(hand_built(), None)
    assert set(t["tags"]) == set(labels.TAGS)
    nonzero = {tag: v for tag, v in t["tags"].items() if v["count"]}
    assert nonzero == {
        "warm": {"count": 1, "seconds": 5.0},
        "playful": {"count": 1, "seconds": 3.5},
        "curious": {"count": 1, "seconds": 3.5},
        "shouted": {"count": 1, "seconds": 5.0},
    }
    assert t["untagged"] == 2


def test_tables_states_with_thin_ones_flagged() -> None:
    t = labels.tables(hand_built(), None, state_seconds_min=4.0)
    assert t["state_seconds_min"] == 4.0
    assert t["states"] == {
        "neutral": {"count": 1, "seconds": 4.0, "thin": False},
        "calm": {"count": 1, "seconds": 5.0, "thin": False},
        "awake": {"count": 1, "seconds": 3.5, "thin": True},
        "sleepy": {"count": 1, "seconds": 3.2, "thin": True},
        "excited": {"count": 0, "seconds": 0.0, "thin": True},
    }
    assert t["state_none"] == {"count": 1, "seconds": 5.0}


def test_tables_thin_threshold_defaults_to_the_gate_default() -> None:
    t = labels.tables(hand_built(), None)
    assert t["state_seconds_min"] == 30
    assert all(entry["thin"] for entry in t["states"].values())


def test_tables_carried_and_notes() -> None:
    t = labels.tables(hand_built(), None)
    assert t["carried"] == {"clips": 2, "runs": {"r0": 1, "r1": 2}}
    assert t["notes"] == 1


def test_tables_without_prefill_have_no_agreement() -> None:
    assert "agreement" not in labels.tables(hand_built(), None)


def test_tables_prefill_agreement_over_rated_clips() -> None:
    agreement = labels.tables(hand_built(), hand_prefill())["agreement"]
    assert agreement["clips"] == 7
    assert agreement["energy"] == {"mean_abs_change": 0.143, "share_changed": 0.143}
    assert agreement["mood"] == {"mean_abs_change": 0.429, "share_changed": 0.286}
    voices = agreement["voices"]
    assert list(voices) == list(labels.VOICES)
    assert all(list(row) == list(labels.VOICES) for row in voices.values())
    assert voices["her"] == {"her": 3, "her_overlap": 1, "her_nearby": 0, "not_her": 1}
    assert voices["her_nearby"] == {"her": 0, "her_overlap": 0, "her_nearby": 2, "not_her": 0}
    assert sum(voices["her_overlap"].values()) == sum(voices["not_her"].values()) == 0


def test_tables_of_no_labels() -> None:
    t = labels.tables([], [])
    assert t["clips"] == {"count": 0, "seconds": 0.0}
    assert t["agreement"]["clips"] == 0
    assert t["agreement"]["energy"] == {"mean_abs_change": 0.0, "share_changed": 0.0}


def test_table_lines_print_counts_seconds_and_ids_only() -> None:
    records = hand_built()
    lines = labels.table_lines(labels.tables(records, hand_prefill(), state_seconds_min=4.0))
    text = "\n".join(lines)
    assert all(isinstance(line, str) and "\n" not in line for line in lines)
    assert "labels: 9 clips, 38.20 s" in lines
    assert (
        "voices: her 5 (20.20 s); her_overlap 1 (3.50 s); her_nearby 2 (10.00 s); "
        "not_her 1 (4.50 s)" in lines
    )
    assert (
        "removed: not-her 1 (4.50 s); flag-group 1 (3.00 s); overlap 1 (3.50 s); "
        "nearby 1 (4.00 s); trimmed away 2.50 s" in lines
    )
    assert (
        "states: neutral 1 (4.00 s); calm 1 (5.00 s); awake 1 (3.50 s); sleepy 1 (3.20 s); "
        "excited 0 (0.00 s); none 1 (5.00 s)" in lines
    )
    assert "thin states (under 4 s): awake, sleepy, excited" in lines
    assert "carried: 2 clips; from r0 1, r1 2" in lines
    assert "notes: 1 clip" in lines
    assert "pre-fill agreement over 7 rated clips" in text
    assert (
        "tags: shouted 1 (5.00 s); warm 1 (5.00 s); playful 1 (3.50 s); "
        "curious 1 (3.50 s); none 2" in lines
    )


def test_table_lines_without_agreement() -> None:
    lines = labels.table_lines(labels.tables(hand_built(), None))
    assert not any("pre-fill" in line for line in lines)


# ---------------------------------------------------------------------------------------------
# (9) earlier_labels, carry and carried_stale
# ---------------------------------------------------------------------------------------------


def carry_one(seg: SegmentRecord, *earlier: LabelRecord) -> LabelRecord | None:
    return labels.carry([seg], list(earlier), run="r9").get(seg.segment_id)


def test_carry_a_match_at_90_percent_of_the_earlier_length() -> None:
    earlier = label("seg-000004", tags=("warm",), energy=2, mood=4, has_note=True, state="calm")
    seg = segment("seg-000017", start=100.5, end=109.5)
    carried = carry_one(seg, earlier)
    assert carried == LabelRecord(
        run="r9", segment_id="seg-000017", source_id="src-0001", start_s=100.5, end_s=109.5,
        channel="FC", separator="melband", duration_s=9.0, voices="her", energy=2, mood=4,
        tags=("warm",), keep_from_s=None, keep_to_s=None, has_note=False, state=None,
        removed=None, carried_from=("r1/seg-000004",),
    )  # fmt: skip


def test_carry_refuses_a_span_just_under_90_percent() -> None:
    assert carry_one(segment(start=100.5, end=109.49), label()) is None


def test_carry_accepts_a_span_past_the_earlier_one_within_the_tolerance() -> None:
    assert carry_one(segment(start=99.995, end=110.0), label()) is not None
    assert carry_one(segment(start=100.0, end=110.01), label()) is not None


def test_carry_refuses_a_span_reaching_002_s_past_the_earlier_one() -> None:
    assert carry_one(segment(start=100.0, end=110.02), label()) is None
    assert carry_one(segment(start=99.98, end=110.0), label()) is None


@pytest.mark.parametrize(
    "change",
    [{"source": "src-0002"}, {"channel": "FL"}, {"separator": "tiger"}],
    ids=["another source", "another channel", "another separator"],
)
def test_carry_needs_the_same_source_channel_and_separator(change: dict) -> None:
    assert carry_one(segment(**change), label()) is None
    assert carry_one(segment(), label()) is not None


def test_carry_skips_an_earlier_flag_group_removal() -> None:
    assert carry_one(segment(), label(removed="flag-group")) is None


def test_carry_ignores_a_flag_group_removal_beside_a_rated_match() -> None:
    dropped = label("seg-000002", run="r0", energy=5, removed="flag-group")
    rated = label("seg-000003", run="r1", energy=2, mood=4)
    carried = carry_one(segment(), dropped, rated)
    assert carried is not None
    assert carried.energy == 2 and carried.carried_from == ("r1/seg-000003",)


def test_carry_an_earlier_not_her_as_not_her() -> None:
    carried = carry_one(segment(), label(voices="not_her", removed="not-her", mood=2))
    assert carried is not None
    assert carried.voices == "not_her" and carried.removed is None and carried.state is None


def test_carry_two_agreeing_labels_lists_both_sorted() -> None:
    first = label("seg-000008", run="r2", tags=("teasing",))
    second = label("seg-000003", run="r1", tags=("teasing",), start=99.5, end=110.2)
    carried = carry_one(segment(start=100.0, end=109.8), first, second)
    assert carried is not None
    assert carried.carried_from == ("r1/seg-000003", "r2/seg-000008")


@pytest.mark.parametrize(
    "change",
    [{"energy": 4}, {"mood": 2}, {"tags": ("warm",)}, {"voices": "her_overlap"}],
)
def test_carry_refuses_two_disagreeing_labels(change: dict) -> None:
    first = label("seg-000008", run="r2")
    second = replace(label("seg-000003", run="r1"), **change)
    assert carry_one(segment(), first, second) is None


def test_carry_her_nearby_range_inside_the_new_span_is_re_expressed() -> None:
    earlier = label(voices="her_nearby", keep=(1.0, 6.0))  # source 101.0-106.0
    carried = carry_one(segment(start=100.5, end=109.6), earlier)
    assert carried is not None
    assert (carried.keep_from_s, carried.keep_to_s) == (0.5, 5.5)
    assert carried.duration_s == 9.1


def test_carry_her_nearby_range_outside_the_new_span_is_asked_again() -> None:
    earlier = label(voices="her_nearby", keep=(0.2, 5.0))  # source 100.2-105.0
    assert carry_one(segment(start=100.5, end=110.0), earlier) is None


@pytest.mark.parametrize(
    ("keep", "span"),
    [((0.0, 5.0), (10.01, 20.0)), ((5.0, 10.0), (10.0, 19.99))],
    ids=["start-clamped", "end-clamped"],
)
def test_carry_her_nearby_range_clamped_into_the_span_is_asked_again(
    keep: tuple[float, float], span: tuple[float, float]
) -> None:
    # The range reaches 0.01 s past the new span, inside `CARRY_TOL_S`, but clamping it into the
    # span moves it 0.01 s in source time: past `carried_stale`'s tolerance, so it is not carried.
    earlier = label(start=10.0, end=20.0, voices="her_nearby", keep=keep)
    assert carry_one(segment(start=span[0], end=span[1]), earlier) is None


_OFFSETS = (-0.01, -0.006, -0.005, -0.004, -0.003, 0.0, 0.003, 0.004, 0.005, 0.006, 0.01)


@pytest.mark.parametrize(
    ("keep", "inside"),
    [((0.5, 9.5), True), ((2.0, 7.0), True), ((0.0, 5.0), False), ((5.0, 10.0), False),
     ((0.0, 10.0), False)],
)  # fmt: skip
def test_every_her_nearby_label_carry_returns_agrees_with_its_sources(
    root: root_mod.Root, keep: tuple[float, float], inside: bool
) -> None:
    write(root, "r1", [label(run="r1", start=10.0, end=20.0, voices="her_nearby", keep=keep)])
    segs = [
        segment(f"seg-{index:06d}", start=10.0 + a, end=20.0 + b)
        for index, (a, b) in enumerate(itertools.product(_OFFSETS, _OFFSETS), start=100)
    ]
    carried = labels.carry(segs, labels.earlier_labels(root, "r2"), run="r2")
    if inside:
        # A range well inside every span still lies inside it after rounding: each one carries.
        assert len(carried) == len(segs)
    assert carried
    assert labels.carried_stale(root, tuple(carried.values())) == ()


def test_carry_her_nearby_without_a_range_is_asked_again() -> None:
    earlier = label(voices="her_nearby", removed="nearby")
    assert carry_one(segment(), earlier) is None


def test_carry_her_nearby_labels_must_agree_on_the_range_in_source_time() -> None:
    first = label("seg-000001", run="r1", voices="her_nearby", keep=(1.0, 6.0))
    # Same source range (101.0-106.0) from a span starting 0.5 s later: agrees.
    second = label("seg-000005", run="r2", start=100.5, end=110.0, voices="her_nearby",
                   keep=(0.5, 5.5))  # fmt: skip
    carried = carry_one(segment(start=100.5, end=109.8), first, second)
    assert carried is not None and (carried.keep_from_s, carried.keep_to_s) == (0.5, 5.5)
    moved = replace(second, keep_to_s=6.0)
    assert carry_one(segment(start=100.5, end=109.8), first, moved) is None


def test_carry_never_carries_a_carried_label_or_one_of_its_own_run() -> None:
    assert carry_one(segment(), label(carried_from=("r0/seg-000001",))) is None
    assert carry_one(segment(), label(run="r9")) is None


def test_carry_handles_several_segments() -> None:
    segs = [segment("seg-000001"), segment("seg-000002", source="src-0003")]
    carried = labels.carry(segs, [label()], run="r9")
    assert list(carried) == ["seg-000001"]


def test_earlier_labels_reads_other_runs_rated_labels_only(root: root_mod.Root) -> None:
    rated = label("seg-000001", run="r1")
    carried = label("seg-000002", run="r1", carried_from=("r0/seg-000001",))
    write(root, "r1", [rated, carried])
    write(root, "r2", [label("seg-000001", run="r2", energy=5)])
    write(root, "r0", [label("seg-000001", run="r0", energy=1)])
    (root.path / "seed/labels/r-empty").mkdir()
    found = labels.earlier_labels(root, "r2")
    assert found == (label("seg-000001", run="r0", energy=1), rated)


def test_a_label_carried_into_another_run_is_not_carried_back(root: root_mod.Root) -> None:
    write(root, "r1", [label("seg-000001", run="r1", energy=2)])
    seg = segment("seg-000040")
    carried_r2 = labels.carry([seg], labels.earlier_labels(root, "r2"), run="r2")
    write(root, "r2", [replace(carried_r2["seg-000040"], state="calm")])
    # Re-asking r1: its own folder is skipped and r2 holds only a carried label.
    assert labels.earlier_labels(root, "r1") == ()
    assert labels.carry([segment("seg-000001")], labels.earlier_labels(root, "r1"), run="r1") == {}


def test_earlier_labels_with_no_label_folder(root: root_mod.Root) -> None:
    assert labels.earlier_labels(root, "r1") == ()


def stale_setup(root: root_mod.Root) -> tuple[LabelRecord, ...]:
    write(root, "r1", [
        label("seg-000001", run="r1", voices="her_nearby", keep=(1.0, 6.0)),
        label("seg-000002", run="r1", start=200.0, end=210.0, energy=4),
    ])  # fmt: skip
    write(root, "r0", [label("seg-000007", run="r0", start=200.0, end=210.0, energy=4)])
    segs = [
        segment("seg-000011", start=100.5, end=109.6),
        segment("seg-000012", start=200.0, end=209.5),
    ]
    carried = labels.carry(segs, labels.earlier_labels(root, "r2"), run="r2")
    assert carried["seg-000012"].carried_from == ("r0/seg-000007", "r1/seg-000002")
    return (*carried.values(), label("seg-000013", run="r2", start=300.0, end=305.0))


def test_carried_stale_is_empty_while_every_source_is_current(root: root_mod.Root) -> None:
    assert labels.carried_stale(root, stale_setup(root)) == ()


def test_carried_stale_names_a_changed_source(root: root_mod.Root) -> None:
    current = stale_setup(root)
    write(root, "r0", [label("seg-000007", run="r0", start=200.0, end=210.0, energy=5)],
          replace_=True)  # fmt: skip
    assert labels.carried_stale(root, current) == ("seg-000012",)


def test_carried_stale_names_a_changed_range(root: root_mod.Root) -> None:
    current = stale_setup(root)
    write(root, "r1", [
        label("seg-000001", run="r1", voices="her_nearby", keep=(1.0, 5.0)),
        label("seg-000002", run="r1", start=200.0, end=210.0, energy=4),
    ], replace_=True)  # fmt: skip
    assert labels.carried_stale(root, current) == ("seg-000011",)


def test_carried_stale_names_a_source_that_is_gone(root: root_mod.Root) -> None:
    current = stale_setup(root)
    write(root, "r1", [label("seg-000002", run="r1", start=200.0, end=210.0, energy=4)],
          replace_=True)  # fmt: skip
    assert labels.carried_stale(root, current) == ("seg-000011",)


def test_carried_stale_names_a_source_whose_run_is_gone(root: root_mod.Root) -> None:
    current = stale_setup(root)
    for path in (root.path / "seed/labels/r0").iterdir():
        path.unlink()
    (root.path / "seed/labels/r0").rmdir()
    assert labels.carried_stale(root, current) == ("seg-000012",)


def test_carried_stale_names_a_source_now_removed_with_its_flag_group(
    root: root_mod.Root,
) -> None:
    current = stale_setup(root)
    write(root, "r0", [label("seg-000007", run="r0", start=200.0, end=210.0, energy=4,
                             removed="flag-group")], replace_=True)  # fmt: skip
    assert labels.carried_stale(root, current) == ("seg-000012",)


def test_carried_stale_names_a_source_now_carried(root: root_mod.Root) -> None:
    current = stale_setup(root)
    write(root, "r1", [
        label("seg-000001", run="r1", voices="her_nearby", keep=(1.0, 6.0),
              carried_from=("r0/seg-000003",)),
        label("seg-000002", run="r1", start=200.0, end=210.0, energy=4),
    ], replace_=True)  # fmt: skip
    assert labels.carried_stale(root, current) == ("seg-000011",)
