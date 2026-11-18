"""The night co-residence fit probe against fake HTTP functions.

`night_fit.main` takes its HTTP layer as a `ServerIO` of plain callables, so every
test here injects fakes: no network, no GPU, no llama-server. An autouse fixture
also makes `urllib.request.urlopen` raise, so a code path that bypassed the
injected functions would fail loudly instead of reaching a real socket.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.model_bench.probes import night_fit  # noqa: E402
from scripts.model_bench.probes import ttft as ttft_probe  # noqa: E402

CHAT_URL = "http://chat.invalid:8080"
EXTRACTION_URL = "http://extraction.invalid:8080"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("night_fit tests must not open a real HTTP connection")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)


def _sse(event: dict) -> str:
    return "data: " + json.dumps(event) + "\n"


class FakeServers:
    """Both llama-servers behind the `ServerIO` callables.

    Chat requests go through `run_request` (ttft.run_one_request's shape). Extraction
    requests go through `open_stream` as SSE lines. The first `trials` extraction
    streams are the chat-idle phase; later ones are contended.

    Time is a fake clock (`clock`) that only the fakes advance, so the contended
    timeline is exact. A contended stream emits its first decode token at +1 s, waits
    for the chat request to start, emits `window_tokens` tokens spaced at
    1/`overlap_tps` s while the chat request is held open, then lets the chat request
    return and waits until the probe has read the clock for the chat's end before it
    emits its last token and its stop event 2 s later. With `ends_before_chat_returns`
    the stream instead stops inside the chat window.
    """

    def __init__(
        self,
        *,
        trials: int = 3,
        chat_n_ctx: int = 8192,
        extraction_n_ctx: int = 32768,
        chat_ttft_idle_ms: float = 300.0,
        chat_ttft_contended_ms: tuple[float, ...] = (800.0, 900.0, 1000.0),
        idle_tps: tuple[float, ...] = (27.0, 26.0, 28.0),
        contended_tps: tuple[float, ...] = (18.0, 19.0, 17.0),
        overlap_tps: tuple[float, ...] = (4.0, 5.0, 2.0),
        window_tokens: int = 8,
        first_token_delay_s: float = 0.05,
        contended_emits_tokens: bool = True,
        ends_before_chat_returns: bool = False,
        props_error: Exception | None = None,
    ) -> None:
        self.trials = trials
        self.n_ctx = {CHAT_URL: chat_n_ctx, EXTRACTION_URL: extraction_n_ctx}
        self.chat_ttft_idle_ms = chat_ttft_idle_ms
        self.chat_ttft_contended_ms = list(chat_ttft_contended_ms)
        self.idle_tps = list(idle_tps)
        self.contended_tps = list(contended_tps)
        self.overlap_tps = list(overlap_tps)
        self.window_tokens = window_tokens
        self.first_token_delay_s = first_token_delay_s
        self.contended_emits_tokens = contended_emits_tokens
        self.ends_before_chat_returns = ends_before_chat_returns
        self.props_error = props_error

        self.lock = threading.Lock()
        self.log: list[str] = []
        self.tokenized: list[tuple[str, str]] = []
        self.chat_prompt_lens: list[int] = []
        self.stream_payloads: list[dict] = []
        self.stream_calls = 0
        self.chat_calls = 0
        self.decoding = False
        self.now = 100.0
        self.awaiting_chat_end_read = False
        self.chat_started = threading.Event()
        self.window_done = threading.Event()
        self.chat_end_read = threading.Event()

    def _note(self, entry: str) -> None:
        with self.lock:
            self.log.append(entry)

    def _advance(self, seconds: float) -> None:
        with self.lock:
            self.now += seconds

    def _wait(self, event: threading.Event, what: str) -> None:
        if not event.wait(timeout=5.0):
            self._note(f"gave_up_waiting_for:{what}")

    def clock(self) -> float:
        with self.lock:
            if (
                self.awaiting_chat_end_read
                and threading.current_thread() is threading.main_thread()
            ):
                # The probe's first main-thread read after a contended chat returns is
                # its chat-end stamp; only then may the stream go on.
                self.awaiting_chat_end_read = False
                self.chat_end_read.set()
            return self.now

    # --- ServerIO callables ---------------------------------------------

    def get_props(self, base_url: str) -> dict:
        if self.props_error is not None:
            raise self.props_error
        return {"default_generation_settings": {"n_ctx": self.n_ctx[base_url]}}

    def tokenize(self, base_url: str, text: str) -> list[int]:
        self.tokenized.append((base_url, text))
        return [5, 6, 7, 8] if base_url == CHAT_URL else [9, 10, 11]

    def run_request(self, base_url: str, prompt_tokens: list[int], *, n_predict: int) -> dict:
        assert base_url == CHAT_URL, "only chat requests go through run_request"
        with self.lock:
            k = self.chat_calls
            self.chat_calls += 1
            contended = self.decoding
        self.chat_prompt_lens.append(len(prompt_tokens))
        self._note(f"chat_start:{k}")
        if contended:
            self.chat_started.set()
            self._wait(self.window_done, f"window:{k}")
            ttft_ms = self.chat_ttft_contended_ms.pop(0)
            with self.lock:
                self.awaiting_chat_end_read = True
        else:
            self._advance(1.0)
            ttft_ms = self.chat_ttft_idle_ms
        self._note(f"chat_end:{k}")
        return {
            "ttft_ms": ttft_ms,
            "total_ms": None,
            "prompt_ms": None,
            "prompt_per_second": None,
            "predicted_n": n_predict,
            "predicted_ms": None,
            "predicted_per_second": None,
        }

    def open_stream(self, base_url: str, payload: dict):
        assert base_url == EXTRACTION_URL, "only extraction requests are streamed"
        with self.lock:
            k = self.stream_calls
            self.stream_calls += 1
        self.stream_payloads.append(payload)
        contended = k >= self.trials
        return self._stream(k, contended, payload["n_predict"])

    def _stop_event(self, n_predict: int, tps: float) -> str:
        # predicted_ms deliberately disagrees with predicted_per_second, so a probe
        # that derived the rate from anything but timings.predicted_per_second fails.
        return _sse(
            {
                "content": "",
                "stop": True,
                "tokens_predicted": n_predict,
                "timings": {
                    "prompt_n": 16384,
                    "prompt_ms": 9000.0,
                    "prompt_per_second": 1820.4,
                    "predicted_n": n_predict,
                    "predicted_ms": 1.0,
                    "predicted_per_second": tps,
                },
            }
        )

    def _stream(self, k: int, contended: bool, n_predict: int):
        self._note(f"ext_open:{k}")
        if contended:
            self.chat_started.clear()
            self.window_done.clear()
            self.chat_end_read.clear()
        # Prompt processing: llama-server emits nothing with content until decode.
        yield _sse({"content": "", "stop": False})
        time.sleep(self.first_token_delay_s)
        if contended and not self.contended_emits_tokens:
            yield _sse({"content": "", "stop": True, "tokens_predicted": 0, "timings": {}})
            self._note(f"ext_end:{k}")
            return
        self._advance(1.0)
        if contended:
            with self.lock:
                self.decoding = True
        self._note(f"ext_first_token:{k}")
        yield _sse({"content": "The", "stop": False})
        if not contended:
            self._advance(1.0)
            yield _sse({"content": " tide", "stop": False})
            self._advance(1.0)
            yield self._stop_event(n_predict, self.idle_tps[k])
            self._note(f"ext_end:{k}")
            return

        trial = k - self.trials
        step = 1.0 / self.overlap_tps[trial]
        self._wait(self.chat_started, f"chat_start:{k}")
        for _ in range(self.window_tokens):
            self._advance(step)
            yield _sse({"content": " x", "stop": False})
        if self.ends_before_chat_returns:
            yield self._stop_event(n_predict, self.contended_tps[trial])
            self._note(f"ext_end:{k}")
            self.window_done.set()
            return
        # Runs once the probe has stamped the last window token, so it is in the window.
        self.window_done.set()
        self._wait(self.chat_end_read, f"chat_end_read:{k}")
        with self.lock:
            self.decoding = False
        self._advance(1.0)
        yield _sse({"content": " tide", "stop": False})
        self._advance(1.0)
        yield self._stop_event(n_predict, self.contended_tps[trial])
        self._note(f"ext_end:{k}")

    def io(self) -> night_fit.ServerIO:
        return night_fit.ServerIO(
            get_props=self.get_props,
            tokenize=self.tokenize,
            run_request=self.run_request,
            open_stream=self.open_stream,
            clock=self.clock,
        )


def _run(tmp_path, fake: FakeServers, *extra: str) -> tuple[int, dict | None]:
    out = tmp_path / "night-fit.json"
    argv = [
        "--chat-url",
        CHAT_URL,
        "--extraction-url",
        EXTRACTION_URL,
        "--out",
        str(out),
        *extra,
    ]
    code = night_fit.main(argv, io=fake.io())
    data = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    return code, data


# --- CLI ------------------------------------------------------------------


def test_cli_defaults_match_the_plan():
    args = night_fit.parse_args(
        ["--chat-url", CHAT_URL, "--extraction-url", EXTRACTION_URL, "--out", "x.json"]
    )
    assert args.trials == 3
    assert args.chat_prompt_tokens == 2048
    assert args.extraction_prompt_tokens == 16384
    assert args.extraction_n_predict == 2048
    assert args.ttft_bar == pytest.approx(1.5)
    assert args.decode_bar == pytest.approx(23.0)


@pytest.mark.parametrize("missing", ["--chat-url", "--extraction-url", "--out"])
def test_cli_requires_both_urls_and_out(missing):
    argv = {"--chat-url": CHAT_URL, "--extraction-url": EXTRACTION_URL, "--out": "x.json"}
    del argv[missing]
    flat = [tok for pair in argv.items() for tok in pair]
    with pytest.raises(SystemExit) as exc:
        night_fit.parse_args(flat)
    assert exc.value.code == 2


def test_cli_refuses_zero_trials():
    with pytest.raises(SystemExit) as exc:
        night_fit.parse_args(
            [
                "--chat-url",
                CHAT_URL,
                "--extraction-url",
                EXTRACTION_URL,
                "--out",
                "x.json",
                "--trials",
                "0",
            ]
        )
    assert exc.value.code == 2


# --- output ---------------------------------------------------------------


def test_output_json_carries_every_contract_field(tmp_path):
    fake = FakeServers(chat_n_ctx=8192, extraction_n_ctx=32768)
    code, data = _run(tmp_path, fake)
    assert code == 0
    assert data["chat_n_ctx"] == 8192
    assert data["extraction_n_ctx"] == 32768
    assert data["chat_ttft_idle_s"] == pytest.approx([0.3, 0.3, 0.3])
    assert data["chat_ttft_contended_s"] == pytest.approx([0.8, 0.9, 1.0])
    assert data["extraction_decode_tps_chat_idle"] == pytest.approx(27.0)
    assert data["extraction_decode_tps_chat_idle_trials"] == pytest.approx([27.0, 26.0, 28.0])
    assert data["extraction_decode_tps_contended"] == pytest.approx(18.0)
    assert data["extraction_decode_tps_contended_trials"] == pytest.approx([18.0, 19.0, 17.0])
    assert data["passed"] is True
    assert data["error"] is None


def test_trials_sets_the_number_of_measurements(tmp_path):
    fake = FakeServers(
        trials=2,
        chat_ttft_contended_ms=(700.0, 600.0),
        idle_tps=(30.0, 24.0),
        contended_tps=(20.0, 21.0),
    )
    code, data = _run(tmp_path, fake, "--trials", "2")
    assert code == 0
    assert len(data["chat_ttft_idle_s"]) == 2
    assert data["chat_ttft_contended_s"] == pytest.approx([0.7, 0.6])
    assert data["extraction_decode_tps_chat_idle"] == pytest.approx(27.0)  # median of 30, 24
    assert fake.stream_calls == 4
    assert fake.chat_calls == 4


def test_decode_rates_come_from_llama_server_timings(tmp_path):
    # The fake's timings.predicted_ms (1.0 ms for 2048 tokens) implies ~2,000,000 tok/s;
    # only timings.predicted_per_second gives the recorded rates.
    fake = FakeServers(idle_tps=(24.5, 24.5, 24.5), contended_tps=(11.25, 11.25, 11.25))
    _, data = _run(tmp_path, fake)
    assert data["extraction_decode_tps_chat_idle"] == pytest.approx(24.5)
    assert data["extraction_decode_tps_contended"] == pytest.approx(11.25)


def test_prompts_are_synthetic_filler_at_the_requested_lengths(tmp_path):
    fake = FakeServers()
    _run(
        tmp_path,
        fake,
        "--chat-prompt-tokens",
        "1000",
        "--extraction-prompt-tokens",
        "5000",
        "--extraction-n-predict",
        "300",
    )
    # Each server tokenizes the ttft filler once, with its own tokenizer.
    assert sorted(url for url, _ in fake.tokenized) == [CHAT_URL, EXTRACTION_URL]
    assert all(text == ttft_probe.FILLER_TEXT for _, text in fake.tokenized)
    assert fake.chat_prompt_lens == [1000] * 6
    assert [len(p["prompt"]) for p in fake.stream_payloads] == [5000] * 6
    assert {p["n_predict"] for p in fake.stream_payloads} == {300}
    assert all(p["stream"] is True for p in fake.stream_payloads)
    assert fake.stream_payloads[0]["prompt"][:4] == [9, 10, 11, 9]


# --- reuse of ttft.py ------------------------------------------------------


def test_reuses_the_ttft_helpers_by_import_not_by_copy(tmp_path, monkeypatch):
    calls: dict[str, int] = {"build_prompt_tokens": 0, "iter_sse_events": 0, "extract_ttft_row": 0}
    for name in calls:
        original = getattr(ttft_probe, name)

        def spy(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(ttft_probe, name, spy)

    code, _ = _run(tmp_path, FakeServers())
    assert code == 0
    assert all(count > 0 for count in calls.values()), calls

    source = Path(night_fit.__file__).read_text(encoding="utf-8")
    for name in ("build_prompt_tokens", "run_one_request", "iter_sse_events", "extract_ttft_row"):
        assert f"def {name}(" not in source

    io = night_fit.default_io()
    assert io.run_request is ttft_probe.run_one_request
    assert io.tokenize is ttft_probe.tokenize


# --- contended ordering ---------------------------------------------------


def test_contended_chat_starts_only_after_the_first_extraction_decode_token(tmp_path):
    # The extraction stream spends 0.2 s in "prompt processing" before its first
    # decode token; a chat request sent when the stream opens would start inside it.
    fake = FakeServers(first_token_delay_s=0.2)
    code, _ = _run(tmp_path, fake)
    assert code == 0
    log = fake.log
    assert not any(e.startswith("gave_up") for e in log), log
    for trial in range(3):
        k_ext = 3 + trial  # streams 0-2 are the chat-idle phase
        k_chat = 3 + trial  # chat requests 0-2 are the idle TTFT phase
        first = log.index(f"ext_first_token:{k_ext}")
        start = log.index(f"chat_start:{k_chat}")
        end = log.index(f"chat_end:{k_chat}")
        ext_end = log.index(f"ext_end:{k_ext}")
        assert log.index(f"ext_open:{k_ext}") < first < start < end < ext_end, log


def test_idle_phases_run_one_server_at_a_time(tmp_path):
    fake = FakeServers()
    _run(tmp_path, fake)
    log = fake.log
    # Idle chat TTFTs: no extraction stream open yet.
    assert log.index("chat_end:2") < log.index("ext_open:0")
    # Chat-idle extraction decodes: each ends before the next phase opens anything.
    for k in range(3):
        assert log.index(f"ext_end:{k}") < log.index(f"ext_open:{k + 1}")
    assert log.index("ext_end:2") < log.index("chat_start:3")


def test_contended_timeline_is_recorded_on_one_clock(tmp_path):
    # Trial 0: first decode token at +1 s; the chat request starts then and holds for 8
    # tokens at 4 tok/s (2 s); the stream stops 2 s after the chat returns.
    fake = FakeServers(overlap_tps=(4.0, 5.0, 2.0), window_tokens=8)
    code, data = _run(tmp_path, fake)
    assert code == 0
    assert not any(e.startswith("gave_up") for e in fake.log), fake.log
    timeline = data["contended_timeline"]
    assert len(timeline) == 3
    assert timeline[0] == pytest.approx(
        {
            "extraction_first_token_s": 1.0,
            "chat_start_s": 1.0,
            "chat_end_s": 3.0,
            "extraction_end_s": 5.0,
            "overlap_tokens": 8,
            "overlap_decode_tps": 4.0,
        }
    )
    assert timeline[2]["chat_end_s"] == pytest.approx(5.0)  # 8 tokens at 2 tok/s


def test_overlap_decode_rate_counts_stream_tokens_inside_the_chat_window(tmp_path, capsys):
    fake = FakeServers(overlap_tps=(4.0, 5.0, 2.0), contended_tps=(18.0, 19.0, 17.0))
    code, data = _run(tmp_path, fake)
    assert code == 0
    assert data["extraction_decode_tps_overlap_trials"] == pytest.approx([4.0, 5.0, 2.0])
    assert data["extraction_decode_tps_overlap"] == pytest.approx(4.0)
    # The timings-based contended rate is still recorded as the plan gives it.
    assert data["extraction_decode_tps_contended"] == pytest.approx(18.0)
    # Recorded only: a slow overlap rate does not add a bar or fail one.
    assert len(_bar_lines(capsys.readouterr().out)) == 2


def test_overlap_rate_is_recorded_not_gated(tmp_path):
    code, data = _run(tmp_path, FakeServers(overlap_tps=(0.5, 0.5, 0.5)))
    assert code == 0
    assert data["extraction_decode_tps_overlap"] == pytest.approx(0.5)


def test_extraction_ending_before_the_chat_returns_is_an_error(tmp_path, capsys):
    fake = FakeServers(ends_before_chat_returns=True)
    code, data = _run(tmp_path, fake)
    assert code == 2
    err = capsys.readouterr().err
    assert "--extraction-n-predict 2048 is too short for the overlap" in err
    assert data["chat_ttft_contended_s"] == []
    assert "too short" in data["error"]
    # The refused trial's timeline is kept, and shows why.
    (timeline,) = data["contended_timeline"]
    assert timeline["extraction_end_s"] <= timeline["chat_end_s"]
    assert fake.stream_calls == 4  # three chat-idle streams, then the refused trial


def test_each_phase_is_announced_on_stderr_as_it_starts(tmp_path, capsys, monkeypatch):
    fake = FakeServers()
    announce = night_fit._phase

    def noting_phase(text):
        fake._note(f"phase:{text}")
        announce(text)

    monkeypatch.setattr(night_fit, "_phase", noting_phase)
    code, _ = _run(tmp_path, fake)
    assert code == 0
    phases = [
        line
        for line in capsys.readouterr().err.splitlines()
        if line.startswith("night_fit: phase: ")
    ]
    assert phases == [
        "night_fit: phase: idle chat TTFT, extraction idle, 3 trials",
        "night_fit: phase: chat-idle extraction decode, 3 trials",
        "night_fit: phase: contended trial 1/3",
        "night_fit: phase: contended trial 2/3",
        "night_fit: phase: contended trial 3/3",
        "night_fit: phase: done",
    ]
    log = fake.log
    assert log.index("phase:idle chat TTFT, extraction idle, 3 trials") < log.index("chat_start:0")
    assert log.index("chat_end:2") < log.index("phase:chat-idle extraction decode, 3 trials")
    assert log.index("phase:chat-idle extraction decode, 3 trials") < log.index("ext_open:0")
    for trial in range(3):
        k = 3 + trial
        announced = log.index(f"phase:contended trial {trial + 1}/3")
        assert log.index(f"ext_end:{k - 1}") < announced < log.index(f"ext_open:{k}")


def test_extraction_stream_sends_the_body_ttft_run_one_request_sends(monkeypatch):
    """Both requests go through one fake `urlopen`; the bodies must agree except where
    the probe chooses to differ on purpose. It differs nowhere today.
    """
    deliberate_differences: frozenset[str] = frozenset()
    sent: list[tuple[str, str, dict, dict]] = []
    stop_line = (
        _sse(
            {
                "content": "",
                "stop": True,
                "tokens_predicted": 1,
                "timings": {"predicted_ms": 10.0, "prompt_ms": 5.0, "predicted_per_second": 9.0},
            }
        )
    ).encode("utf-8")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __iter__(self):
            yield _sse({"content": "a", "stop": False}).encode("utf-8")
            yield stop_line

    def fake_urlopen(req, timeout=None):
        sent.append(
            (
                req.full_url,
                req.get_method(),
                {k.lower(): v for k, v in req.header_items()},
                json.loads(req.data.decode("utf-8")),
            )
        )
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    prompt = [3, 1, 4, 1, 5]
    ttft_probe.run_one_request(EXTRACTION_URL, prompt, n_predict=77)
    stream = night_fit.stream_extraction(
        night_fit.default_io(), EXTRACTION_URL, prompt, n_predict=77
    )
    assert stream.row["predicted_per_second"] == pytest.approx(9.0)

    (ttft_url, ttft_method, ttft_headers, ttft_body), (url, method, headers, body) = sent
    assert url == ttft_url == EXTRACTION_URL + "/completion"
    assert method == ttft_method == "POST"
    assert headers == ttft_headers
    differing = {k for k in body.keys() | ttft_body.keys() if body.get(k) != ttft_body.get(k)}
    assert differing == deliberate_differences


def test_contended_trial_without_a_decode_token_is_an_error(tmp_path, capsys):
    fake = FakeServers(contended_emits_tokens=False)
    code, data = _run(tmp_path, fake)
    assert code == 2
    assert "no decode token" in capsys.readouterr().err
    assert fake.chat_calls == 3  # the idle TTFTs only; no contended chat was sent
    assert data["error"]


# --- bars and exit codes ----------------------------------------------------


def _bar_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith("bar ")]


def test_every_bar_passes_exits_0_with_one_line_per_bar(tmp_path, capsys):
    code, data = _run(tmp_path, FakeServers())
    assert code == 0
    lines = _bar_lines(capsys.readouterr().out)
    assert len(lines) == 2
    assert lines[0].startswith("bar chat_ttft_contended_s: PASS")
    assert lines[1].startswith("bar extraction_decode_tps_chat_idle: PASS")
    assert [b["passed"] for b in data["bars"]] == [True, True]


def test_one_contended_ttft_over_the_bar_exits_1(tmp_path, capsys):
    fake = FakeServers(chat_ttft_contended_ms=(800.0, 1600.0, 900.0))
    code, data = _run(tmp_path, fake)
    assert code == 1
    lines = _bar_lines(capsys.readouterr().out)
    assert len(lines) == 2
    assert lines[0].startswith("bar chat_ttft_contended_s: FAIL")
    assert "1.600" in lines[0]
    assert lines[1].startswith("bar extraction_decode_tps_chat_idle: PASS")
    assert data["passed"] is False


def test_ttft_bar_is_inclusive_and_settable(tmp_path):
    fake = FakeServers(chat_ttft_contended_ms=(1000.0, 1000.0, 1000.0))
    code, _ = _run(tmp_path, fake, "--ttft-bar", "1.0")
    assert code == 0


def test_idle_ttft_is_recorded_not_gated(tmp_path):
    fake = FakeServers(chat_ttft_idle_ms=5000.0)
    code, data = _run(tmp_path, fake)
    assert code == 0
    assert data["chat_ttft_idle_s"] == pytest.approx([5.0, 5.0, 5.0])


def test_chat_idle_decode_under_the_bar_exits_1(tmp_path, capsys):
    fake = FakeServers(idle_tps=(22.0, 22.9, 21.0))
    code, data = _run(tmp_path, fake)
    assert code == 1
    lines = _bar_lines(capsys.readouterr().out)
    assert len(lines) == 2
    assert lines[0].startswith("bar chat_ttft_contended_s: PASS")
    assert lines[1].startswith("bar extraction_decode_tps_chat_idle: FAIL")
    assert data["passed"] is False


def test_decode_bar_is_settable(tmp_path):
    code, _ = _run(tmp_path, FakeServers(idle_tps=(22.0, 22.0, 22.0)), "--decode-bar", "21")
    assert code == 0


def test_contended_decode_is_recorded_not_gated(tmp_path):
    fake = FakeServers(contended_tps=(3.0, 2.0, 1.0))
    code, data = _run(tmp_path, fake)
    assert code == 0
    assert data["extraction_decode_tps_contended"] == pytest.approx(2.0)


def test_unreachable_server_exits_2(tmp_path, capsys):
    fake = FakeServers(props_error=urllib.error.URLError("connection refused"))
    code, data = _run(tmp_path, fake)
    assert code == 2
    captured = capsys.readouterr()
    assert "night_fit: error:" in captured.err
    assert _bar_lines(captured.out) == []
    assert "connection refused" in data["error"]


def test_prompt_that_does_not_fit_the_extraction_context_exits_2(tmp_path, capsys):
    fake = FakeServers(extraction_n_ctx=16384)  # 16384 prompt + 2048 decode cannot fit
    code, data = _run(tmp_path, fake)
    assert code == 2
    assert "n_ctx" in capsys.readouterr().err
    assert data["extraction_n_ctx"] == 16384
    assert fake.stream_calls == 0
    assert fake.chat_calls == 0


def test_props_without_n_ctx_exits_2(tmp_path):
    fake = FakeServers()
    fake.get_props = lambda base_url: {"default_generation_settings": {}}
    code, _ = _run(tmp_path, fake)
    assert code == 2
