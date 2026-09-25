"""Probe parsers against the hand-built fixtures in fixtures/host/.

No live server, no network: sse_stream.txt, nvidia_smi_sample.csv,
voice_probe_output.json and correctness_response.json are hand-built from
documented formats, not recordings -- see README.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.model_bench.probes import nvidia_smi as nvidia_smi_probe  # noqa: E402
from scripts.model_bench.probes import correctness as correctness_probe  # noqa: E402
from scripts.model_bench.probes import ttft as ttft_probe  # noqa: E402

FIXTURES = _REPO_ROOT / "tests" / "unit" / "model_bench" / "fixtures" / "host"


# --- SSE stream -------------------------------------------------------------


def test_iter_sse_events_parses_data_lines_only():
    lines = (FIXTURES / "sse_stream.txt").read_text(encoding="utf-8").splitlines()
    events = list(ttft_probe.iter_sse_events(lines))
    # 4 "data:" events; the leading comment block is skipped.
    assert len(events) == 4
    assert events[0]["content"] == ""
    assert events[-1]["stop"] is True


def test_extract_ttft_row_reads_final_timings():
    lines = (FIXTURES / "sse_stream.txt").read_text(encoding="utf-8").splitlines()
    events = list(ttft_probe.iter_sse_events(lines))
    row = ttft_probe.extract_ttft_row(events, first_content_t=1.5, start_t=1.0)
    assert row["ttft_ms"] == pytest.approx(500.0)
    assert row["predicted_ms"] == pytest.approx(150.0)
    assert row["prompt_ms"] == pytest.approx(12.5)
    assert row["total_ms"] == pytest.approx(162.5)
    assert row["predicted_n"] == 3


def test_extract_ttft_row_no_first_content_gives_none_ttft():
    lines = (FIXTURES / "sse_stream.txt").read_text(encoding="utf-8").splitlines()
    events = list(ttft_probe.iter_sse_events(lines))
    row = ttft_probe.extract_ttft_row(events, first_content_t=None, start_t=1.0)
    assert row["ttft_ms"] is None


def test_extract_ttft_row_without_stop_event_raises():
    with pytest.raises(ttft_probe.TtftProbeError):
        ttft_probe.extract_ttft_row([{"content": "x", "stop": False}], first_content_t=1.0, start_t=0.5)


def test_cap_target_respects_ctx_minus_predict_minus_margin():
    assert ttft_probe.cap_target(2048, n_ctx=32768, n_predict=256) == 2048
    # 32000 requested, n_ctx 8192: cap = 8192 - 256 - 16 = 7920.
    assert ttft_probe.cap_target(32000, n_ctx=8192, n_predict=256) == 7920


# --- ttft filler cycling (finding 4: the filler must reach 32K tokens) --


def test_cycle_to_length_reaches_every_ttft_target_exactly():
    # A "fake tokenizer" that only ever produces 3 ids per call -- if the old
    # tokenize-once-and-slice approach were used, the filler text would need to
    # already be long enough; cycling reaches any target regardless.
    base_ids = [7, 8, 9]
    for target in ttft_probe.TTFT_TARGETS:
        cycled = ttft_probe.cycle_to_length(base_ids, target)
        assert len(cycled) == target
    assert ttft_probe.cycle_to_length([1, 2, 3], 7) == [1, 2, 3, 1, 2, 3, 1]


def test_cycle_to_length_empty_base_refuses_rather_than_shortening():
    with pytest.raises(ttft_probe.TtftProbeError):
        ttft_probe.cycle_to_length([], 2048)


def test_run_ttft_probe_reaches_all_targets_exactly_with_a_sparse_fake_tokenizer(monkeypatch):
    """End to end: a fake /tokenize returning only 3 ids per call must still let
    run_ttft_probe build exact-length prompts at 2048, 8192, and 32000 -- and must
    call the fake tokenizer exactly once, not once per target."""
    call_count = 0

    def fake_tokenize(base_url, text, *, timeout=30.0):
        nonlocal call_count
        call_count += 1
        return [7, 8, 9]

    monkeypatch.setattr(ttft_probe, "tokenize", fake_tokenize)

    rows = ttft_probe.run_ttft_probe(
        "http://127.0.0.1:1", n_ctx=32768, warmup_reps=0, measured_reps=1
    )
    by_target = {r["ctx_target"]: r["prompt_tokens"] for r in rows}
    assert by_target[2048] == 2048
    assert by_target[8192] == 8192
    assert by_target[32000] == 32000
    assert call_count == 1


# --- plan v2 (2026-09-25): 65000/130000 targets, skip-only, timeout ------


def test_resolve_targets_to_run_skips_new_targets_outright_at_ctx_32768():
    # ceiling = 32768 - 256 - 16 = 32496; both 65000 and 130000 exceed it and must be
    # skipped entirely (no row, capped or otherwise) -- not clamped to the ceiling.
    resolved = ttft_probe.resolve_targets_to_run(n_ctx=32768)
    assert resolved == [(2048, 2048), (8192, 8192), (32000, 32000)]


def test_resolve_targets_to_run_includes_65000_at_ctx_65536():
    resolved = dict(ttft_probe.resolve_targets_to_run(n_ctx=65536))
    assert resolved[65000] == 65000  # raw target, not capped -- it fits under the ceiling
    assert 130000 not in resolved  # 131072's own arm reaches this; 65536 does not


def test_resolve_targets_to_run_includes_130000_at_ctx_131072():
    resolved = dict(ttft_probe.resolve_targets_to_run(n_ctx=131072))
    assert resolved[65000] == 65000
    assert resolved[130000] == 130000
    # The original targets are still present and still capped the old way.
    assert resolved[2048] == 2048
    assert resolved[32000] == 32000


def test_resolve_targets_to_run_original_targets_stay_capped_never_skipped():
    # Legacy behavior, pinned: an original target is never dropped, only shortened.
    resolved = dict(ttft_probe.resolve_targets_to_run(n_ctx=8192))
    assert resolved[32000] == 7920  # cap_target(32000, n_ctx=8192, n_predict=256)
    assert 65000 not in resolved
    assert 130000 not in resolved


def test_run_ttft_probe_request_sequence_is_byte_identical_at_ctx_32768(monkeypatch):
    """Recorded request list from a fake server: adding the 65000/130000 targets must
    not change a single byte of the request sequence an existing 32768-ctx arm makes --
    same targets, same order, same prompt lengths, same count of requests.
    """
    monkeypatch.setattr(ttft_probe, "tokenize", lambda base_url, text, *, timeout=30.0: [1, 2, 3])

    recorded: list[tuple[int, int]] = []  # (ctx_target, len(prompt_tokens)) per request

    def fake_run_one_request(base_url, prompt_tokens, *, n_predict=ttft_probe.N_PREDICT, timeout=None):
        recorded.append((len(prompt_tokens),))
        return {
            "ttft_ms": 1.0, "total_ms": 1.0, "prompt_ms": 1.0, "prompt_per_second": 1.0,
            "predicted_n": 1, "predicted_ms": 1.0, "predicted_per_second": 1.0,
        }

    monkeypatch.setattr(ttft_probe, "run_one_request", fake_run_one_request)

    rows = ttft_probe.run_ttft_probe("http://127.0.0.1:1", n_ctx=32768, warmup_reps=1, measured_reps=5)

    # Exactly the legacy sequence: 3 targets x (1 warmup + 5 measured) = 18 requests,
    # in order, at exactly the capped lengths -- 65000/130000 contribute nothing.
    expected_targets = [t for t in (2048, 8192, 32000) for _ in range(6)]
    assert [r["ctx_target"] for r in rows] == expected_targets
    assert [r["prompt_tokens"] for r in rows] == [
        ttft_probe.cap_target(t, n_ctx=32768) for t in expected_targets
    ]
    assert recorded == [(ttft_probe.cap_target(t, n_ctx=32768),) for t in expected_targets]


def test_cycle_to_length_reaches_130000_exactly_with_a_fake_tokenizer():
    # A sparse fake tokenizer (3 ids) must still let cycle_to_length reach the new,
    # largest target exactly.
    base_ids = [11, 22, 33]
    cycled = ttft_probe.cycle_to_length(base_ids, 130000)
    assert len(cycled) == 130000
    assert cycled[:6] == [11, 22, 33, 11, 22, 33]


def test_http_timeout_covers_130k_prefill_plus_256_decode():
    # At least 600s per the brief; TTFT_HTTP_TIMEOUT_S is the default run_one_request
    # (and therefore run_ttft_probe) uses for every request.
    assert ttft_probe.TTFT_HTTP_TIMEOUT_S >= 600.0
    import inspect

    assert inspect.signature(ttft_probe.run_one_request).parameters["timeout"].default == (
        ttft_probe.TTFT_HTTP_TIMEOUT_S
    )


# --- nvidia-smi CSV -----------------------------------------------------


def test_parse_csv_line_handles_na_and_typed_fields():
    lines = (FIXTURES / "nvidia_smi_sample.csv").read_text(encoding="utf-8").splitlines()
    rows = [nvidia_smi_probe.parse_csv_line(line) for line in lines if line.strip()]
    assert len(rows) == 3
    assert rows[0]["memory_used_mib"] == 3200
    assert rows[0]["power_w"] == pytest.approx(45.23)
    assert rows[0]["throttle_reasons"] == "0x0000000000000000"
    assert rows[1]["throttle_reasons"] is None  # [N/A] -> None, not a crash


def test_parse_csv_line_wrong_field_count_raises():
    with pytest.raises(nvidia_smi_probe.NvidiaSmiParseError):
        nvidia_smi_probe.parse_csv_line("1,2,3")


def test_summarize_rows_computes_median_max_total_samples():
    rows = [
        {"memory_used_mib": 100, "memory_total_mib": 12288},
        {"memory_used_mib": 200, "memory_total_mib": 12288},
        {"memory_used_mib": 300, "memory_total_mib": 12288},
    ]
    summary = nvidia_smi_probe.summarize_rows(rows)
    assert summary["median_mib"] == 200
    assert summary["max_mib"] == 300
    assert summary["total_mib"] == 12288
    assert summary["samples"] == 3


def test_summarize_rows_empty_list():
    summary = nvidia_smi_probe.summarize_rows([])
    assert summary == {"median_mib": None, "max_mib": None, "total_mib": None, "samples": 0}


def test_build_query_args_is_argv_list_no_shell_string():
    argv = nvidia_smi_probe.build_query_args()
    assert isinstance(argv, list)
    assert all(isinstance(tok, str) for tok in argv)
    assert argv[0] == "nvidia-smi"
    assert "-lms" in argv


# --- voice probe output --------------------------------------------------


def test_voice_probe_output_fixture_has_expected_shape():
    data = json.loads((FIXTURES / "voice_probe_output.json").read_text(encoding="utf-8"))
    assert set(data) == {
        "max_memory_reserved_mib",
        "max_memory_allocated_mib",
        "stt_ok",
        "tts_ok",
        "error",
    }
    assert data["stt_ok"] is True
    assert data["tts_ok"] is True
    assert data["error"] is None


# --- correctness response ------------------------------------------------


def test_parse_correctness_response_fixture():
    response = json.loads((FIXTURES / "correctness_response.json").read_text(encoding="utf-8"))
    row = correctness_probe.parse_correctness_response("p01", response)
    assert row["prompt_id"] == "p01"
    assert row["tokens"] == [1, 22145, 8]
    assert row["n_predict"] == 3
    assert row["error"] is None
    assert len(row["tokens_sha256"]) == 64


def test_parse_correctness_response_missing_tokens_raises():
    with pytest.raises(correctness_probe.CorrectnessProbeError):
        correctness_probe.parse_correctness_response("p02", {"content": "no tokens field here"})


def test_correctness_prompts_are_20_and_unique():
    ids = [p["id"] for p in correctness_probe.PROMPTS]
    assert len(ids) == 20
    assert len(set(ids)) == 20


def test_build_request_payload_is_greedy_deterministic():
    payload = correctness_probe.build_request_payload("hello")
    assert payload["temperature"] == 0
    assert payload["top_k"] == 1
    assert payload["seed"] == 3407
    assert payload["return_tokens"] is True
    assert payload["stream"] is False
