"""T2 harness arms: bench candidates, the json_object schema arm, and metrics capture.

Three things this file proves, matching the mist-model-bench T2 acceptance criteria:

1. `tests/schema_conformance_json_object.yaml` carries the identical cases,
   system_prompt, test_type, temperature_mode and max_tokens as
   `tests/schema_conformance.yaml`; only use_grammar, response_format and
   description differ.
2. `ChatMetrics.timings` and `ChatMetrics.reasoning_content` (client.py) arrive
   verbatim in the JSONL `metrics` object through the real
   `run.run_candidate` path, driven against a local fake llama-server over
   HTTP. A response that omits them records `None`, not an error.
3. The five bench candidates in models.yaml load with the tier, sampling and
   gguf the T2 brief specifies, and the default primary-tier selection is
   unchanged.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.eval_harness import run  # noqa: E402

# ---------------------------------------------------------------------------
# 1. schema_conformance vs schema_conformance_json_object case equality
# ---------------------------------------------------------------------------


def test_json_object_arm_matches_schema_conformance_except_three_keys():
    strict, json_object = run.load_test_files(
        ["schema_conformance", "schema_conformance_json_object"], run.DEFAULT_TESTS_DIR
    )

    assert json_object.cases == strict.cases
    assert json_object.system_prompt == strict.system_prompt
    assert json_object.test_type == strict.test_type == "schema_conformance"
    assert json_object.temperature_mode == strict.temperature_mode
    assert json_object.max_tokens == strict.max_tokens

    # The only three keys that are allowed to differ.
    assert strict.use_grammar is True
    assert json_object.use_grammar is False
    assert strict.response_format is None
    assert json_object.response_format == {"type": "json_object"}
    assert json_object.description != strict.description


# ---------------------------------------------------------------------------
# 2. timings / reasoning_content capture through the real run_candidate path
# ---------------------------------------------------------------------------


def _completion_payload(
    *, content: str, finish_reason: str, with_timings: bool, with_reasoning: bool
) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if with_reasoning:
        message["reasoning_content"] = "because the prompt says so"
    payload: dict = {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": 1,
        "model": "fake-model",
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish_reason},
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
    }
    if with_timings:
        payload["timings"] = {
            "prompt_n": 12,
            "prompt_ms": 40.0,
            "predicted_n": 4,
            "predicted_per_second": 33.3,
        }
    return payload


def _make_fake_server(responses: list[dict]) -> HTTPServer:
    """A local, no-network http.server standing in for llama-server.

    GET /health always answers 200 (client.wait_for_ready polls this before
    the harness sends any chat completion). POST /v1/chat/completions
    returns each entry of `responses` in order, one per call -- letting a
    test send a first request that carries timings/reasoning_content and a
    second that omits them.
    """

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # noqa: A002 - stdlib signature
            pass  # silence per-request stderr logging during tests

        def do_GET(self):
            if self.path == "/health":
                body = b"{}"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)  # drain the request body, unused by the fake
            index = self.server.call_index  # type: ignore[attr-defined]
            self.server.call_index += 1  # type: ignore[attr-defined]
            body = json.dumps(self.server.responses[index]).encode(  # type: ignore[attr-defined]
                "utf-8"
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    server.responses = responses  # type: ignore[attr-defined]
    server.call_index = 0  # type: ignore[attr-defined]
    return server


def test_metrics_capture_timings_and_reasoning_content_through_run_candidate(tmp_path):
    responses = [
        _completion_payload(
            content="{}", finish_reason="stop", with_timings=True, with_reasoning=True
        ),
        _completion_payload(
            content="{}", finish_reason="stop", with_timings=False, with_reasoning=False
        ),
    ]
    server = _make_fake_server(responses)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]

        defaults = run.HarnessDefaults(
            base_url=f"http://127.0.0.1:{port}",
            models_dir=".",
            llama_server_binary="llama-server",
            shared_server_args=(),
            ctx_size=2048,
            spawn_health_timeout_seconds=5.0,
            request_timeout_seconds=10.0,
            api_key="not-needed",
        )
        candidate = run.Candidate(
            id="fake-bench-candidate",
            display_name="Fake bench candidate",
            tier="bench",
            family="fake",
            vendor="fake",
            gguf="fake.gguf",
            quant="",
            size_gb=0.0,
            total_params_b=0.0,
            active_params_b=0.0,
            architecture="dense",
            context_size=2048,
            served_model_name="fake-model",
            temperature={"extraction": 0.0, "conversation": 0.7},
            top_p={"extraction": 0.9, "conversation": 0.9},
            chat_template=None,
            chat_template_file=None,
            tool_parser=None,
            gbnf_supported=False,
            stop_sequences=(),
            extra_server_args=(),
            shared_server_args_override=None,
            notes="",
        )
        test_file = run.TestFile(
            name="fake_test",
            test_type="schema_conformance",
            description="fake",
            temperature_mode="extraction",
            max_tokens=64,
            use_grammar=False,
            response_format=None,
            system_prompt=None,
            cases=(
                run.TestCase(
                    id="case_with_extras",
                    prompt="hello",
                    system_prompt=None,
                    expected={},
                    tools=(),
                    context=None,
                    metadata={},
                ),
                run.TestCase(
                    id="case_without_extras",
                    prompt="world",
                    system_prompt=None,
                    expected={},
                    tools=(),
                    context=None,
                    metadata={},
                ),
            ),
        )

        jsonl_path = run.run_candidate(
            candidate,
            defaults,
            [test_file],
            iterations=1,
            seed=None,
            external=True,
            llama_server_bin="unused",
            models_dir=tmp_path,
            log_dir=tmp_path / "logs",
            results_dir=tmp_path / "results",
            grammar_text=None,
        )

        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        records = [json.loads(line) for line in lines]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)

    with_extras, without_extras = records

    assert with_extras["case_id"] == "case_with_extras"
    assert with_extras["finish_reason"] == "stop"
    assert with_extras["metrics"]["timings"] == {
        "prompt_n": 12,
        "prompt_ms": 40.0,
        "predicted_n": 4,
        "predicted_per_second": 33.3,
    }
    assert with_extras["metrics"]["reasoning_content"] == "because the prompt says so"

    assert without_extras["case_id"] == "case_without_extras"
    assert without_extras["finish_reason"] == "stop"
    assert without_extras["metrics"]["timings"] is None
    assert without_extras["metrics"]["reasoning_content"] is None


# ---------------------------------------------------------------------------
# 3. Bench candidates load with the specified tier, sampling and gguf
# ---------------------------------------------------------------------------


def test_bench_candidates_load_with_specified_fields_and_primary_tier_unchanged():
    defaults, candidates = run.load_models_config(run.DEFAULT_CONFIG_PATH)
    by_id = {c.id: c for c in candidates}

    bench_ids = ["bench-c0", "bench-c0-prod", "bench-c2", "bench-c3", "bench-c4"]
    for candidate_id in bench_ids:
        assert candidate_id in by_id, f"{candidate_id} missing from models.yaml"
        assert by_id[candidate_id].tier == "bench"

    expected_gguf = {
        "bench-c0": "unsloth/gemma-4-E4B-it-Q5_K_M.gguf",
        "bench-c0-prod": "unsloth/gemma-4-E4B-it-Q5_K_M.gguf",
        "bench-c2": "google/gemma-4-12b-it-qat-q4_0.gguf",
        "bench-c3": "unsloth/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf",
        "bench-c4": "unsloth/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
    }
    for candidate_id, gguf in expected_gguf.items():
        assert by_id[candidate_id].gguf == gguf

    # Gemma family sampling: temperature 1.0 / top_p 0.95, both modes.
    for candidate_id in ("bench-c0", "bench-c2", "bench-c3"):
        candidate = by_id[candidate_id]
        assert candidate.temperature == {"extraction": 1.0, "conversation": 1.0}
        assert candidate.top_p == {"extraction": 0.95, "conversation": 0.95}

    # bench-c0-prod: MIST's live request settings, not the harness defaults.
    prod = by_id["bench-c0-prod"]
    assert prod.temperature == {"extraction": 0.0, "conversation": 0.7}
    assert prod.top_p == {"extraction": 0.9, "conversation": 0.9}

    # Qwen3.6-without-thinking sampling: temperature 0.7 / top_p 0.8, both modes.
    qwen = by_id["bench-c4"]
    assert qwen.temperature == {"extraction": 0.7, "conversation": 0.7}
    assert qwen.top_p == {"extraction": 0.8, "conversation": 0.8}

    # Primary-tier selection is unchanged: bench tier is invisible to the
    # default (--models empty) selection.
    primary_selection = run.resolve_candidate_selection("", candidates)
    assert {c.id for c in primary_selection} == {
        "gemma-4-26b-a4b-iq4xs",
        "qwen-3.5-9b-q8",
        "gemma-3-12b-q5km",
        "qwen-2.5-14b-q5km",
    }
    assert all(c.tier == "primary" for c in primary_selection)


def test_t5_bench_candidates_load_with_specified_fields_and_primary_tier_unchanged():
    # T5 (plan v3): bench-c5, bench-c6, bench-c3-q3, bench-c3-iq4.
    defaults, candidates = run.load_models_config(run.DEFAULT_CONFIG_PATH)
    by_id = {c.id: c for c in candidates}

    bench_ids = ["bench-c5", "bench-c6", "bench-c3-q3", "bench-c3-iq4"]
    for candidate_id in bench_ids:
        assert candidate_id in by_id, f"{candidate_id} missing from models.yaml"
        assert by_id[candidate_id].tier == "bench"

    expected_gguf = {
        "bench-c5": "unsloth/Qwen3.5-9B-Q8_0.gguf",
        "bench-c6": "unsloth/gemma-4-E4B-it-Q8_0.gguf",
        "bench-c3-q3": "unsloth/gemma-4-26B-A4B-it-UD-Q3_K_XL.gguf",
        "bench-c3-iq4": "unsloth/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf",
    }
    for candidate_id, gguf in expected_gguf.items():
        assert by_id[candidate_id].gguf == gguf

    # bench-c5: qwen non-thinking sampling, same profile as bench-c4.
    c5 = by_id["bench-c5"]
    assert c5.temperature == {"extraction": 0.7, "conversation": 0.7}
    assert c5.top_p == {"extraction": 0.8, "conversation": 0.8}

    # bench-c6, bench-c3-q3, bench-c3-iq4: gemma family sampling, same
    # profile as bench-c0/bench-c3.
    for candidate_id in ("bench-c6", "bench-c3-q3", "bench-c3-iq4"):
        candidate = by_id[candidate_id]
        assert candidate.temperature == {"extraction": 1.0, "conversation": 1.0}
        assert candidate.top_p == {"extraction": 0.95, "conversation": 0.95}

    # qwen-3.5-9b-q8 (the existing primary candidate bench-c5 shares a gguf
    # with) is untouched, and primary-tier selection is still unchanged.
    primary_selection = run.resolve_candidate_selection("", candidates)
    assert {c.id for c in primary_selection} == {
        "gemma-4-26b-a4b-iq4xs",
        "qwen-3.5-9b-q8",
        "gemma-3-12b-q5km",
        "qwen-2.5-14b-q5km",
    }
    assert all(c.tier == "primary" for c in primary_selection)


# ---------------------------------------------------------------------------
# T7 (plan v3 scan): bench-c7 (Granite 4.2 8B), bench-c8 (Spark-X2.5-4B),
# bench-c9 (gpt-oss-20b)
# ---------------------------------------------------------------------------


def test_t7_bench_candidates_load_with_specified_fields_and_primary_tier_unchanged():
    defaults, candidates = run.load_models_config(run.DEFAULT_CONFIG_PATH)
    by_id = {c.id: c for c in candidates}

    bench_ids = ["bench-c7", "bench-c8", "bench-c9"]
    for candidate_id in bench_ids:
        assert candidate_id in by_id, f"{candidate_id} missing from models.yaml"
        assert by_id[candidate_id].tier == "bench"

    expected_gguf = {
        "bench-c7": "ibm-granite/granite-4.2-8b-Q6_K.gguf",
        "bench-c8": "XHToken/Spark-X2.5-4B-Q8_0.gguf",
        "bench-c9": "ggml-org/gpt-oss-20b-MXFP4.gguf",
    }
    for candidate_id, gguf in expected_gguf.items():
        assert by_id[candidate_id].gguf == gguf

    # bench-c7/bench-c8: no citable model-card sampling (no network access from
    # this worker's container) -- the gemma-style defaults arms.json's "granite"
    # and "spark" sampling families mirror (temperature 1.0/top_p 0.95, both
    # modes), per the T7 brief's fallback instruction.
    for candidate_id in ("bench-c7", "bench-c8"):
        candidate = by_id[candidate_id]
        assert candidate.temperature == {"extraction": 1.0, "conversation": 1.0}
        assert candidate.top_p == {"extraction": 0.95, "conversation": 0.95}

    # bench-c9: OpenAI's gpt-oss guidance (temperature 1.0, top_p 1.0), not a
    # citable model-card value fetched from a live source (no network access).
    c9 = by_id["bench-c9"]
    assert c9.temperature == {"extraction": 1.0, "conversation": 1.0}
    assert c9.top_p == {"extraction": 1.0, "conversation": 1.0}
    assert c9.architecture == "moe"

    primary_selection = run.resolve_candidate_selection("", candidates)
    assert {c.id for c in primary_selection} == {
        "gemma-4-26b-a4b-iq4xs",
        "qwen-3.5-9b-q8",
        "gemma-3-12b-q5km",
        "qwen-2.5-14b-q5km",
    }
    assert all(c.tier == "primary" for c in primary_selection)


# ---------------------------------------------------------------------------
# T5+T7 additive-only guarantee for models.yaml, extended to the base commit
# T7 branched from (044699b, which already contains T5's own bench-c5/c6/
# c3-q3/c3-iq4 candidates). tests/unit/eval_harness/test_t5_models_yaml_
# additive_only.py already covers this against T5's own original base
# (61f4822); that file is outside this task's write zone
# (tests/unit/eval_harness/test_bench_arms.py is the only eval_harness test
# file T7 may edit), so this extends the same guarantee here instead of
# there.
# ---------------------------------------------------------------------------

_T7_BASE_COMMIT = "044699b6b9d4f3f1759d1f67df65012eff56699d"


def _git_show_at_t7_base(rel_path: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "show", f"{_T7_BASE_COMMIT}:{rel_path}"],
        capture_output=True,
        text=True,
        shell=False,
    )
    if proc.returncode != 0:
        pytest.skip(
            f"cannot read {rel_path} at {_T7_BASE_COMMIT} via `git show` "
            f"(exit {proc.returncode}): {proc.stderr.strip()}"
        )
    return proc.stdout


def test_every_044699b_candidate_id_still_present_and_unchanged(tmp_path):
    text = _git_show_at_t7_base("scripts/eval_harness/models.yaml")
    base_path = tmp_path / "models_base.yaml"
    base_path.write_text(text, encoding="utf-8")
    base_defaults, base_candidates = run.load_models_config(base_path)
    current_defaults, current_candidates = run.load_models_config(run.DEFAULT_CONFIG_PATH)

    assert current_defaults == base_defaults

    current_by_id = {c.id: c for c in current_candidates}
    base_ids = {c.id for c in base_candidates}
    current_ids = {c.id for c in current_candidates}
    missing = base_ids - current_ids
    assert not missing, f"T7 removed candidate id(s): {sorted(missing)}"

    for base_candidate in base_candidates:
        assert current_by_id[base_candidate.id] == base_candidate, (
            f"candidate {base_candidate.id!r} parsed differently under the current "
            f"models.yaml -- T7 must be additive-only"
        )
