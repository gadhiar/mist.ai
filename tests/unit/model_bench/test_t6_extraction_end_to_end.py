"""T6 reviewer findings 1a/1b: end-to-end `probes/extraction.py` runs against
a FAST STUB llama-server (a real local HTTP server on loopback, not a fake
object) -- the same shape of check the reviewer ran manually in the
container ("End-to-end in the container against a fast stub: exit 0, 30 of
60 matched"). No docker, no external network: the stub binds 127.0.0.1 on an
ephemeral port within this test process.

- More than 30 probes (the pre-fix `rate_limit_max_per_minute` default,
  backend/knowledge/config.py:154) must all be matched: the rate-limit
  override (probes/extraction.py's `run_probe`) must not throttle a run
  larger than the production default.
- A probe that never reaches the LLM (rate limit or otherwise) must fail the
  run closed: `main()` returns non-zero, and extraction_summary.json is
  still written, with `complete: false` and the unmatched tag list.
"""

from __future__ import annotations

import http.server
import json
import sys
import threading
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.model_bench.probes import extraction as extraction_probe  # noqa: E402

# Every env var probes/extraction.py's run_probe() sets on os.environ
# directly (not via monkeypatch-scoped fixtures internally) -- cleared
# before each test in this module so one test's run cannot leak state
# (e.g. a stale RATE_LIMIT_MAX_PER_MINUTE or MIST_VAULT_ROOT) into the next.
_PROBE_ENV_VARS = [
    "EVENT_STORE_DB_PATH",
    "MIST_SIDECAR_DB_PATH",
    "MIST_VAULT_ROOT",
    "LLM_SERVER_URL",
    "LLM_TEMPERATURE",
    "MIST_FIXED_CLOCK",
    "MIST_DEBUG_JSONL",
    "MIST_DEBUG_LLM_JSONL",
    "MIST_SESSION_ORIGIN",
    "RATE_LIMIT_MAX_PER_MINUTE",
    "SIGNIFICANCE_THRESHOLD",
    "DEDUP_SIMILARITY_THRESHOLD",
    "DEDUP_CACHE_SIZE",
    "DEDUP_CACHE_TTL_SECONDS",
    "PYTHONHASHSEED",
]


@pytest.fixture(autouse=True)
def _clear_probe_env(monkeypatch):
    for var in _PROBE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class _StubLLMHandler(http.server.BaseHTTPRequestHandler):
    """Minimal OpenAI-chat-completions-shaped stub: always returns an empty
    extraction (`{"entities": [], "relationships": []}`), instantly."""

    def do_POST(self):  # noqa: N802 (http.server's naming convention)
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = {
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "created": 0,
            "model": "stub",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": '{"entities": [], "relationships": []}',
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        payload = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802
        # health_check() hits GET /health.
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):  # noqa: A002
        pass  # silence -- keep test output clean


@pytest.fixture
def stub_llm_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _StubLLMHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _write_gold_corpus(path: Path, n: int) -> Path:
    lines = []
    for i in range(n):
        lines.append(
            json.dumps(
                {
                    "utterance": f"probe utterance number {i:03d} about distinct topic {i:03d}",
                    "tag": f"synthetic-{i:03d}",
                    "expected_entities": [],
                    "expected_relationships": [],
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_more_than_30_probes_all_matched_against_a_fast_stub(tmp_path, stub_llm_server):
    gold_path = _write_gold_corpus(tmp_path / "gold.jsonl", 35)
    out_dir = tmp_path / "out"

    rc = extraction_probe.main(
        ["--base-url", stub_llm_server, "--gold", str(gold_path), "--out", str(out_dir)]
    )

    assert rc == 0
    summary = json.loads((out_dir / "extraction_summary.json").read_text())
    assert summary["total_probes"] == 35
    assert summary["matched_probes"] == 35
    assert summary["complete"] is True
    assert summary["unmatched_probe_ids"] == []
    # The override must exceed the production default (30) -- that IS the
    # bug this test guards against regressing.
    assert summary["env"]["rate_limit_max_per_minute"] > 30
    rows = [
        json.loads(line)
        for line in (out_dir / "extraction.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 35
    assert all(r["matched"] for r in rows)


def test_fail_closed_when_a_probe_never_reaches_the_llm(tmp_path, stub_llm_server, monkeypatch):
    """Forces a partial match by dropping one input before it ever reaches
    the pipeline (standing in for "rate-limited" / "gated" / any cause the
    scorer sees as an unmatched probe) -- main() must exit non-zero, and the
    summary must still be written, with complete: false and the tag."""
    import scripts.mist_admin as mist_admin

    real_replay = mist_admin.run_extraction_only_replay

    async def dropping_replay(handler, inputs, session_id, *a, **kw):
        return await real_replay(handler, inputs[:-1], session_id, *a, **kw)

    monkeypatch.setattr(mist_admin, "run_extraction_only_replay", dropping_replay)

    gold_path = _write_gold_corpus(tmp_path / "gold.jsonl", 5)
    out_dir = tmp_path / "out"

    rc = extraction_probe.main(
        ["--base-url", stub_llm_server, "--gold", str(gold_path), "--out", str(out_dir)]
    )

    assert rc == 1
    summary_path = out_dir / "extraction_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["complete"] is False
    assert summary["matched_probes"] == 4
    assert summary["total_probes"] == 5
    assert summary["unmatched_probe_ids"] == ["synthetic-004"]
    # extraction.jsonl is still written too (fail CLOSED, not fail silent).
    assert (out_dir / "extraction.jsonl").exists()
