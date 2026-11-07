"""The speech bench report: `report.py` (mist-model-host plan v2, task T6).

Every test builds a results tree from `fixtures/speech/report/rows.jsonl`, whose rows carry the
shapes the merged suites write, and renders it with no GPU, no network, no Docker and no engine
package. `REPORT.md` is compared byte for byte with the golden `expected_REPORT.md`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.model_bench import speech_bench  # noqa: E402
from scripts.model_bench.speech import listening, report  # noqa: E402

RUN = "mh1"


def _speech(root: Path) -> Path:
    return root / RUN / "speech"


def _tree(tmp_path: Path, *, skip=None) -> Path:
    root = tmp_path / "results"
    report.write_fixture_tree(report.FIXTURE_ROWS, _speech(root), skip=skip)
    return root


def _render(root: Path, *extra: str) -> int:
    return speech_bench.main(["report", "--results-root", str(root), "--run", RUN, *extra])


def _numbers(root: Path) -> dict:
    return json.loads((_speech(root) / report.NUMBERS_FILE).read_text(encoding="utf-8"))


def _without_games(rel: str) -> bool:
    return rel.startswith("game/")


def _run(numbers: dict, number: int) -> dict:
    return next(r for r in numbers["runs"] if r["run"] == number)


def _all_figures(doc: object):
    if isinstance(doc, dict):
        if "label" in doc and "stat" in doc:
            yield doc
        for value in doc.values():
            yield from _all_figures(value)
    elif isinstance(doc, list):
        for value in doc:
            yield from _all_figures(value)


# ---------------------------------------------------------------------------
# The golden file
# ---------------------------------------------------------------------------


def test_report_from_the_fixture_rows_matches_the_golden_report(tmp_path, capsys):
    root = _tree(tmp_path)

    assert _render(root) == 0

    text = (_speech(root) / report.REPORT_FILE).read_text(encoding="utf-8")
    assert text == report.GOLDEN_REPORT.read_text(encoding="utf-8")
    numbers = _numbers(root)
    assert numbers["schema"] == 1 and numbers["run"] == RUN
    assert "wrote" in capsys.readouterr().out


def test_numbers_json_holds_one_object_per_run(tmp_path):
    root = _tree(tmp_path)
    assert _render(root) == 0

    numbers = _numbers(root)

    assert [r["run"] for r in numbers["runs"]] == [1, 2, 3, 4, 5, 7, 8, 9]
    for run in numbers["runs"]:
        assert run["status"] in ("measured", report.NOT_YET_RUN)
        assert {"run", "title", "status", "tables"} <= set(run)


def test_the_selftest_renders_the_fixture_against_the_golden_file():
    report.selftest()


# ---------------------------------------------------------------------------
# Unblinding
# ---------------------------------------------------------------------------


def test_report_refuses_to_unblind_before_ratings_csv_exists(tmp_path, capsys):
    root = _tree(tmp_path, skip=lambda rel: rel == "listening/ratings.csv")

    assert _render(root) == 1

    out = capsys.readouterr().out
    assert "[FAIL]" in out and "sealed" in out
    assert not (_speech(root) / report.NUMBERS_FILE).exists()
    assert not (_speech(root) / report.REPORT_FILE).exists()


def test_the_refusal_is_listenings_own_sealed_key_error(tmp_path):
    root = _tree(tmp_path, skip=lambda rel: rel == "listening/ratings.csv")

    with pytest.raises(listening.KeySealedError):
        report.build_report(_speech(root), run=RUN, git_sha="x")


def test_without_a_listening_pack_the_listening_section_is_not_yet_run(tmp_path):
    root = _tree(tmp_path, skip=lambda rel: rel.startswith("listening"))

    numbers, text = report.build_report(_speech(root), run=RUN, git_sha="x")

    assert numbers["listening"]["status"] == report.NOT_YET_RUN
    assert "## Blind listening\n\nNot yet run" in text


def test_ratings_are_unblinded_by_engine_setting_and_voice(tmp_path):
    root = _tree(tmp_path)

    numbers, _ = report.build_report(_speech(root), run=RUN, git_sha="x")

    figures = numbers["listening"]["tables"][0]["figures"]
    natural = {
        (f["keys"]["engine"], f["keys"]["setting"], f["keys"]["voice"]): f
        for f in figures
        if f["label"].startswith("naturalness")
    }
    assert ("qwen3", "default", "aiden") in natural
    assert ("original", "expressive", "builtin") in natural
    assert natural[("qwen3", "default", "aiden")]["n"] == 2
    assert natural[("qwen3", "default", "aiden")]["value"] == 4.5


# ---------------------------------------------------------------------------
# Every number with its sample count and its mark
# ---------------------------------------------------------------------------


def test_every_figure_carries_its_sample_count_and_a_mark(tmp_path):
    root = _tree(tmp_path)
    assert _render(root) == 0

    figures = list(_all_figures(_numbers(root)))

    assert len(figures) > 100
    for fig in figures:
        assert isinstance(fig["n"], int), fig
        assert isinstance(fig["mark"], str) and fig["mark"], fig
        if fig["stat"] == "summary":
            assert {"median", "p90"} <= set(fig), fig


def test_first_audio_is_marked_against_250_and_500_ms(tmp_path):
    root = _tree(tmp_path)
    numbers, _ = report.build_report(_speech(root), run=RUN, git_sha="x")

    first = [
        f
        for t in _run(numbers, 1)["tables"]
        for f in t["figures"]
        if f["label"] == "first audio, lines"
    ]

    assert first
    for fig in first:
        assert fig["target"] == {
            "name": "speech first audio",
            "target": 250,
            "floor": 500,
            "unit": "ms",
        }
        assert fig["mark"] == report.mark(fig["median"], report.TTS_FIRST_AUDIO)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(250, "target met"), (250.5, "floor met"), (500, "floor met"), (501, "over floor")],
)
def test_marks_against_target_and_floor(value, expected):
    assert report.mark(value, report.TTS_FIRST_AUDIO) == expected


def test_end_of_turn_inference_is_marked_against_15_ms():
    assert report.mark(14.9, report.TURN_INFERENCE) == "under 15 ms"
    assert report.mark(15, report.TURN_INFERENCE) == "15 ms or over"
    assert report.mark(None, report.TURN_INFERENCE) == report.NO_DATA


@pytest.mark.parametrize(
    ("headroom", "fits", "expected"),
    [
        (2048, True, "2 GiB gate met"),
        (2047, True, "512 MiB margin met"),
        (511, True, "under 512 MiB margin"),
        (None, False, "does not fit"),
    ],
)
def test_memory_headroom_is_marked_against_the_margin_and_the_gate(headroom, fits, expected):
    assert report.memory_mark(headroom, fits=fits) == expected


def test_a_percentile_carries_its_sample_count(tmp_path):
    root = _tree(tmp_path)
    numbers, _ = report.build_report(_speech(root), run=RUN, git_sha="x")

    stt = _run(numbers, 3)["tables"][0]["figures"]
    whisper = next(
        f for f in stt if f["keys"]["engine"] == "whisper-base" and f["label"] == "final text"
    )

    assert whisper["n"] == 4
    assert whisper["median"] == 262.5 and whisper["p90"] == 290.0


# ---------------------------------------------------------------------------
# The section 8 budget
# ---------------------------------------------------------------------------


def test_budget_puts_each_measured_stage_in_and_labels_the_rest_derived(tmp_path):
    root = _tree(tmp_path)
    numbers, text = report.build_report(_speech(root), run=RUN, git_sha="x")

    stages = {s["stage"]: s for s in numbers["budget"]["stages"]}

    assert sorted(stages) == [1, 2, 3, 4, 5, 6, 7]
    for number, run in ((1, "run 4"), (2, "run 3"), (4, "run 2"), (6, "run 1")):
        assert stages[number]["source"] == f"measured, {run}"
        assert stages[number]["options"]
        assert all(not f["derived"] for f in stages[number]["options"])
    for number in (3, 5, 7):
        assert stages[number]["source"] == "derived"
        (fig,) = stages[number]["options"]
        assert fig["derived"] and fig["inputs"]
    total = numbers["budget"]["total"]
    assert total["median"]["derived"] and total["slow"]["derived"]
    assert "the reassessment, section 8" in text
    assert "## Budget: the reassessment, section 8" in text


def test_the_first_speakable_words_are_derived_from_the_measured_decode_rate(tmp_path):
    root = _tree(tmp_path)
    numbers, _ = report.build_report(_speech(root), run=RUN, git_sha="x")

    (fig,) = next(s for s in numbers["budget"]["stages"] if s["stage"] == 5)["options"]
    c0 = next(
        f
        for f in _run(numbers, 2)["tables"][0]["figures"]
        if f["keys"]["arm"] == "c0" and f["label"] == "decode rate"
    )

    assert fig["value"] == pytest.approx(round(10 / c0["median"] * 1000.0, 3))
    assert any("decode rate" in i for i in fig["inputs"])


def test_the_budget_total_adds_the_fastest_option_of_each_stage(tmp_path):
    root = _tree(tmp_path)
    numbers, _ = report.build_report(_speech(root), run=RUN, git_sha="x")

    chosen = []
    for stage in numbers["budget"]["stages"]:
        best = min(stage["options"], key=lambda f: f.get("median", f.get("value")) or float("inf"))
        chosen.append(best.get("median", best.get("value")))

    assert numbers["budget"]["total"]["median"]["value"] == pytest.approx(sum(chosen))


# ---------------------------------------------------------------------------
# The levels section
# ---------------------------------------------------------------------------


def test_the_levels_section_answers_each_design_question(tmp_path):
    root = _tree(tmp_path)
    numbers, text = report.build_report(_speech(root), run=RUN, git_sha="x")

    questions = [level["question"] for level in numbers["levels"]]

    assert questions == list(report.LEVEL_QUESTIONS)
    assert len(questions) == 11
    for level in numbers["levels"]:
        assert level["tables"] or level["notes"], level["question"]
        assert f"### {level['question']}" in text


def test_the_placement_table_sets_each_measure_side_by_side(tmp_path):
    root = _tree(tmp_path)
    numbers, _ = report.build_report(_speech(root), run=RUN, git_sha="x")

    grid = numbers["levels"][0]["tables"][0]
    rows = {(r["keys"]["model"], r["keys"]["placement"]): r["cells"] for r in grid["rows"]}

    assert grid["columns"] == list(report.PLACEMENT_COLUMNS)
    cells = rows[("gemma4-e2b", "gpu-full")]
    turn = cells["chained turn with endpointing"]
    assert turn["derived"] and turn["target"]["target"] == 2000 and turn["target"]["floor"] == 3000
    assert cells["tool-call validity"]["stat"] == "share"
    assert cells["hand-off validity"]["stat"] == "share"
    assert cells["memory"]["unit"] == "MiB"
    assert cells["load"]["unit"] == "s" and cells["unload"]["unit"] == "s"
    assert cells["game cost, run 8"]["unit"] == "fps"
    assert rows[("gemma4-e2b", "cpu")]["game cost, run 9"] == report.NOT_YET_RUN


def test_the_chained_turn_adds_endpointing_as_a_derived_figure(tmp_path):
    root = _tree(tmp_path)
    numbers, _ = report.build_report(_speech(root), run=RUN, git_sha="x")

    figs = [
        f
        for t in _run(numbers, 7)["tables"]
        for f in t["figures"]
        if f.get("keys", {}).get("placement") == "gpu-full"
        and f["keys"].get("model") == "gemma4-e2b"
        and f["keys"].get("turns") == "command and short"
    ]
    measured = next(f for f in figs if f["label"].startswith("chained turn, measured"))
    derived = next(f for f in figs if f["label"] == "chained turn with endpointing")
    inference = next(
        f for f in _run(numbers, 4)["tables"][0]["figures"] if f["label"] == "inference"
    )

    assert derived["derived"] and not measured["derived"]
    assert derived["median"] == pytest.approx(measured["median"] + 200 + inference["median"])
    assert derived["n"] == measured["n"]


def test_runs_8_and_9_are_not_yet_run_until_their_rows_exist(tmp_path):
    root = _tree(tmp_path, skip=_without_games)
    numbers, text = report.build_report(_speech(root), run=RUN, git_sha="x")

    assert _run(numbers, 8)["status"] == report.NOT_YET_RUN
    assert _run(numbers, 9)["status"] == report.NOT_YET_RUN
    assert "## Run 8: low level while a light game runs\n\nNot yet run" in text
    assert "## Run 9: Minecraft with its server on this PC\n\nNot yet run" in text
    grid = numbers["levels"][0]["tables"][0]
    assert {r["cells"]["game cost, run 8"] for r in grid["rows"]} == {report.NOT_YET_RUN}


def test_a_game_session_fills_run_8_and_leaves_run_9_not_yet_run(tmp_path):
    root = _tree(tmp_path)
    numbers, _ = report.build_report(_speech(root), run=RUN, git_sha="x")

    assert _run(numbers, 8)["status"] == "measured"
    assert _run(numbers, 9)["status"] == report.NOT_YET_RUN


def test_report_renders_again_with_replace_when_run_8_arrives(tmp_path):
    root = _tree(tmp_path, skip=_without_games)
    assert _render(root) == 0
    first = (_speech(root) / report.REPORT_FILE).read_text(encoding="utf-8")
    report.write_fixture_tree(
        report.FIXTURE_ROWS, _speech(root), skip=lambda rel: not _without_games(rel)
    )

    assert _render(root) == 1
    assert (_speech(root) / report.REPORT_FILE).read_text(encoding="utf-8") == first

    assert _render(root, "--replace") == 0
    text = (_speech(root) / report.REPORT_FILE).read_text(encoding="utf-8")
    assert text == report.GOLDEN_REPORT.read_text(encoding="utf-8")
    assert _run(_numbers(root), 8)["status"] == "measured"
    assert not list(_speech(root).glob("*.tmp"))


def test_report_refuses_a_results_root_inside_the_repository(capsys):
    assert speech_bench.main(["report", "--results-root", str(_REPO_ROOT), "--run", RUN]) == 1
    assert "[FAIL]" in capsys.readouterr().out


def test_report_refuses_a_run_with_no_speech_results(tmp_path, capsys):
    assert _render(tmp_path / "empty") == 1
    assert "no speech results" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The larger-card section
# ---------------------------------------------------------------------------


def test_the_larger_card_section_labels_every_figure_derived_and_names_its_inputs(tmp_path):
    root = _tree(tmp_path)
    numbers, text = report.build_report(_speech(root), run=RUN, git_sha="x")

    larger = numbers["larger_card"]
    figures = list(_all_figures(larger))

    assert figures
    for fig in figures:
        assert fig["derived"] is True, fig
        assert fig["inputs"], fig
    sizes = [i for f in figures for i in f["inputs"] if "file size" in i]
    assert any("gemma-4-26B-A4B" in i for i in sizes)
    assert any("gemma-4-12B" in i for i in sizes)
    assert "## What a 16 to 24 GB card would add (derived)" in text
    assert "only NVIDIA GPUs" in text


def test_without_the_larger_model_sizes_the_fit_says_what_is_missing(tmp_path):
    root = _tree(tmp_path, skip=lambda rel: rel == f"host/{report.LARGER_MODELS_FILE}")

    numbers, text = report.build_report(_speech(root), run=RUN, git_sha="x")

    assert numbers["larger_card"]["larger_models"] is None
    assert "--larger-models" in text


def test_the_larger_model_sizes_file_can_be_named_on_the_command_line(tmp_path):
    root = _tree(tmp_path, skip=lambda rel: rel == f"host/{report.LARGER_MODELS_FILE}")
    sizes = tmp_path / "sizes.json"
    sizes.write_text(
        json.dumps(
            {
                "schema": 1,
                "source": "Hugging Face API",
                "models": [{"id": "gemma-4-12B Q4_K_M", "size_bytes": 7_000_000_000}],
            }
        ),
        encoding="utf-8",
    )

    assert _render(root, "--larger-models", str(sizes)) == 0

    larger = _numbers(root)["larger_card"]
    assert [m["id"] for m in larger["larger_models"]["models"]] == ["gemma-4-12B Q4_K_M"]


def test_a_malformed_larger_model_sizes_file_is_refused_before_any_write(tmp_path, capsys):
    root = _tree(tmp_path)
    bad = tmp_path / "sizes.json"
    bad.write_text(json.dumps({"schema": 1, "models": [{"id": "x"}]}), encoding="utf-8")

    assert _render(root, "--larger-models", str(bad)) == 1

    assert "size_bytes" in capsys.readouterr().out
    assert not (_speech(root) / report.REPORT_FILE).exists()
