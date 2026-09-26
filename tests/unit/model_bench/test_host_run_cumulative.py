"""cmd_run end to end: cumulative meta.json / vram.csv across several `run` calls
(finding 1), and the two refusal shapes -- an already-existing suite output, and a
config mismatch against an arm dir's existing meta.json -- both of which must write
nothing.

Docker, /props, and the probes are stubbed; nvidia-smi's subprocess is replaced with a
fake Popen yielding one CSV row per sampler start (so vram.csv's real append-vs-truncate
behavior, from GpuSampler itself, is exercised, not reimplemented here). `collect_git_state`
is stubbed too, so no real `git` subprocess call happens under the same globally-patched
`subprocess.Popen` this test needs for GpuSampler. No docker, no network, no GPU.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.model_bench import analyse as analyse_mod  # noqa: E402
from scripts.model_bench import bench_host  # noqa: E402
from scripts.model_bench.probes import correctness as correctness_probe  # noqa: E402
from scripts.model_bench.probes import ttft as ttft_probe  # noqa: E402

HARNESS_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures" / "analyse" / "run" / "a1" / "harness" / "harness" / "bench-c0.jsonl"
)

FAKE_ARMS_DOC = {
    "schema": 1,
    "common_args": [],
    "sampling": {"gemma": []},
    "thinking_args": {"off": [], "on": ["--reasoning-budget", "{budget}"]},
    "arms": {
        "test-arm": {
            "gguf": "fake/model.gguf",
            "image": "fake:image",
            "family": "gemma",
            "thinking": None,
            "suites": ["ttft", "correctness", "harness", "layout"],
            "harness": {"candidate": "bench-c0", "tests": "tuning", "iterations": 1},
        }
    },
}

EXPECTED_ARGS = ["-m", "/models/fake/model.gguf"]
FAKE_PROPS = {"model_path": "/models/fake/model.gguf", "default_generation_settings": {"n_ctx": 8192}}
FAKE_VRAM_LINE = "3200, 12288, 45.23, 1800, 9500, 60, 0x0000000000000000, 4, 16\n"


class FakePopen:
    """Replaces subprocess.Popen for GpuSampler's nvidia-smi process: one CSV row, then EOF."""

    def __init__(self, argv, **kwargs):
        self.stdout = iter([FAKE_VRAM_LINE])

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


class FakeCompletedProcess:
    def __init__(self, returncode=0):
        self.returncode = returncode


def _fake_subprocess_run(argv, cwd=None, shell=False, **kwargs):
    if "scripts.eval_harness.run" in argv:
        results_dir = Path(argv[argv.index("--results-dir") + 1])
        candidate = argv[argv.index("--models") + 1]
        out_path = results_dir / "harness" / f"{candidate}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(HARNESS_FIXTURE, out_path)
        return FakeCompletedProcess(0)
    if "run_host.py" in argv:
        out_idx = argv.index("--out")
        result_dir = Path(cwd) / argv[out_idx + 1]
        result_dir.mkdir(parents=True, exist_ok=True)
        graded_rows = [
            {
                "id": "L001", "layout_id": "L001", "task": "t", "correct": True, "parse_ok": True,
                "status": "ok", "finish_reason": "stop", "wall_ms": 500.0, "phase": "accuracy",
                "model": "fake", "usage": {"completion_tokens": 40, "prompt_tokens": 10, "total_tokens": 50},
            },
            {
                "id": "L002", "layout_id": "L002", "task": "t", "correct": False, "parse_ok": True,
                "status": "ok", "finish_reason": "stop", "wall_ms": 600.0, "phase": "accuracy",
                "model": "fake", "usage": {"completion_tokens": 42, "prompt_tokens": 10, "total_tokens": 52},
            },
        ]
        (result_dir / "graded.jsonl").write_text(
            "\n".join(json.dumps(r) for r in graded_rows) + "\n", encoding="utf-8"
        )
        (result_dir / "calls.jsonl").write_text("", encoding="utf-8")
        (result_dir / "manifest.json").write_text("{}", encoding="utf-8")
        return FakeCompletedProcess(0)
    if "analyse.py" in argv:
        return FakeCompletedProcess(0)
    raise AssertionError(f"unexpected subprocess.run call in this test: {argv}")


def _fake_docker_inspect(names):
    assert names == [bench_host.BENCH_LLM_CONTAINER]
    return {
        bench_host.BENCH_LLM_CONTAINER: {
            "Args": EXPECTED_ARGS,
            "Image": "sha256:" + "11" * 32,
            "Config": {"Image": "fake:image"},
        }
    }


def _fake_wait_for_llama_props(base_url, timeout=30.0):
    return dict(FAKE_PROPS)


def _fake_check_decision_rules_clean(path, repo_root):
    return "fixturesha256"


def _fake_collect_git_state(repo_root, layout_dir):
    return {"mist_ai": "fixture", "mist_ai_dirty": False, "command_center": None}


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    monkeypatch.setattr(bench_host, "load_arms_doc", lambda path=bench_host.ARMS_JSON_PATH: FAKE_ARMS_DOC)
    monkeypatch.setattr(bench_host, "docker_inspect", _fake_docker_inspect)
    monkeypatch.setattr(bench_host, "wait_for_llama_props", _fake_wait_for_llama_props)
    monkeypatch.setattr(bench_host, "check_decision_rules_clean", _fake_check_decision_rules_clean)
    monkeypatch.setattr(bench_host, "collect_git_state", _fake_collect_git_state)
    monkeypatch.setattr(bench_host.subprocess, "run", _fake_subprocess_run)
    monkeypatch.setattr(bench_host.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(
        ttft_probe,
        "run_ttft_probe",
        lambda base_url, **kw: [
            {
                "ctx_target": 2048, "prompt_tokens": 2048, "rep": 0, "warmup": False,
                "ttft_ms": 100.0, "error": None, "predicted_per_second": 20.0,
            }
        ],
    )
    monkeypatch.setattr(
        correctness_probe,
        "run_correctness_probe",
        lambda base_url, **kw: [
            {"prompt_id": f"p{i:02d}", "tokens": [i, i + 1], "tokens_sha256": "x", "error": None}
            for i in range(1, 21)
        ],
    )

    results_root = tmp_path / "results"
    layout_dir = tmp_path / "layout-perception"
    layout_dir.mkdir()
    return {"results_root": results_root, "layout_dir": layout_dir}


def _run(stubbed, *, suites=None, rep=None, layout_pass=None, tuning_label=None, arm="test-arm", run="run1"):
    argv = [
        "run", arm, "--run", run,
        "--results-root", str(stubbed["results_root"]),
        "--layout-dir", str(stubbed["layout_dir"]),
    ]
    if suites is not None:
        argv += ["--suites", *suites]
    if rep is not None:
        argv += ["--rep", str(rep)]
    if layout_pass is not None:
        argv += ["--layout-pass", layout_pass]
    if tuning_label is not None:
        argv += ["--tuning-label", tuning_label]
    return bench_host.main(argv)


def _a_dir(stubbed, arm="test-arm", run="run1"):
    return stubbed["results_root"] / run / arm


def test_cumulative_sequence_across_four_calls(stubbed, capsys):
    a_dir = _a_dir(stubbed)

    # Call 1: c0 runs ttft, correctness, harness and screen layout.
    rc = _run(stubbed, suites=["ttft", "correctness", "harness", "layout"], layout_pass="screen")
    assert rc == 0, capsys.readouterr()
    meta1 = json.loads((a_dir / "meta.json").read_text())
    assert meta1["suites_completed"] == ["ttft", "correctness", "harness", "layout"]
    assert set(meta1["layout"]) == {"screen"}
    assert meta1["harness"]["candidate"] == "bench-c0"
    assert len(meta1["calls"]) == 1
    vram1 = (a_dir / "vram.csv").read_bytes()
    assert vram1.count(b"\n") == 2  # header + 1 sample row

    # Call 2: --rep 2 correctness.
    rc = _run(stubbed, suites=["correctness"], rep=2)
    assert rc == 0
    meta2 = json.loads((a_dir / "meta.json").read_text())
    assert meta2["suites_completed"] == ["ttft", "correctness", "harness", "layout"]
    assert len(meta2["calls"]) == 2
    assert (a_dir / "correctness.r1.jsonl").exists()
    assert (a_dir / "correctness.r2.jsonl").exists()

    # vram.csv appends -- holds call 1's row AND call 2's row, header written once.
    vram2 = (a_dir / "vram.csv").read_bytes()
    assert vram2.startswith(vram1)
    assert vram2.count(b"\n") == 3  # header + 2 sample rows
    assert vram2.decode().count("t_unix") == 1  # header written exactly once

    # Call 3: a refused ttft re-run (ttft.jsonl already exists) -- writes nothing.
    meta_before_refusal = (a_dir / "meta.json").read_bytes()
    vram_before_refusal = (a_dir / "vram.csv").read_bytes()
    rc = _run(stubbed, suites=["ttft"])
    assert rc == 1
    assert (a_dir / "meta.json").read_bytes() == meta_before_refusal
    assert (a_dir / "vram.csv").read_bytes() == vram_before_refusal

    # Call 4: a finalist layout pass.
    rc = _run(stubbed, suites=["layout"], layout_pass="finalist")
    assert rc == 0
    meta4 = json.loads((a_dir / "meta.json").read_text())
    assert set(meta4["layout"]) == {"screen", "finalist"}
    # 3 successful calls total (the refused call added no entry).
    assert len(meta4["calls"]) == 3
    vram4 = (a_dir / "vram.csv").read_bytes()
    assert vram4.count(b"\n") == 4  # header + 3 sample rows (the refusal added none)

    # analyse.py, fed this exact meta.json / on-disk shape, computes harness_score and
    # the layout metrics (both passes) as non-missing.
    results_dir = stubbed["results_root"] / "run1"
    inputs = analyse_mod.load_arm_inputs(results_dir, "test-arm")
    assert inputs.meta is not None
    stats_cfg = {
        "wilson_z": analyse_mod.WILSON_Z_DEFAULT,
        "B": analyse_mod.BOOTSTRAP_B_DEFAULT,
        "seed": analyse_mod.BOOTSTRAP_SEED_DEFAULT,
        "confidence": analyse_mod.BOOTSTRAP_CONFIDENCE_DEFAULT,
    }
    harness_scores, _ = analyse_mod.compute_harness_scores_for_arm(inputs, stats_cfg)
    assert "schema_conformance" in harness_scores
    assert harness_scores["schema_conformance"].usable(), harness_scores["schema_conformance"]

    for layout_pass in ("screen", "finalist"):
        acc = analyse_mod.compute_layout_acc(inputs.layout_rows.get(layout_pass), n_expected=999, stats_cfg=stats_cfg)
        assert acc.missing is False, f"{layout_pass}: {acc}"


def test_config_mismatch_between_calls_is_refused_and_writes_nothing(stubbed):
    a_dir = _a_dir(stubbed)
    rc = _run(stubbed, suites=["ttft"])
    assert rc == 0
    meta_before = (a_dir / "meta.json").read_bytes()
    vram_before = (a_dir / "vram.csv").read_bytes()

    rc = _run(stubbed, suites=["correctness"], tuning_label="different-tuning-label")
    assert rc == 1
    assert (a_dir / "meta.json").read_bytes() == meta_before
    assert (a_dir / "vram.csv").read_bytes() == vram_before
    assert not (a_dir / "correctness.r1.jsonl").exists()
