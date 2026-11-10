"""Run 7's chained low-level turn: recording, Parakeet, the low-level model, Pocket TTS.

Each of Raj's fourteen chain recordings goes through the speech-to-text service, then the served
low-level model with the tool set and the low-level prompt. A tool call gets the fixed result
`done` back and a second request for the spoken reply; a `hand_off` call is answered with
"One moment." at once, as MIST would answer while the hand-off runs; a reply without a call is
spoken as it is. The first clause of the reply then goes to the Pocket TTS service, and the turn
ends at its first audio chunk.

Each turn is labelled by its tool prompt's kind: `command`, `short` (a no-tool prompt) or
`hand_off`. The row keeps every stage time; `turn_ms` is the critical path (speech-to-text, then
the model to its first speakable clause, then speech to first audio) and `turn_wall_ms` the wall
time to first audio, which also holds the rest of each stream the client reads before it speaks.

`--busy-arm c0` runs the same turns on `gpu-full` while T1's load generator streams from
`mist-bench-llm` throughout, and records E4B's decode rate over each turn.

The two services run in containers (`probes/stt_service.py`, `probes/tts_service.py`) on 4 CPU
threads each, published on 127.0.0.1 only; this module only starts, polls and removes them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .. import bench_host
from . import core, docker_cmds, lowtier_suite, stt_suite, toolcall, tts_suite, wer
from .probes import load_gen, tts_engines, tts_probe

SUITE = "chain"
STT_ENGINE = "parakeet-cpu"
TTS_ENGINE = "pocket"
STT_PORT = 8091
TTS_PORT = 8092
STT_URL = f"http://{docker_cmds.LOCALHOST}:{STT_PORT}"
TTS_URL = f"http://{docker_cmds.LOCALHOST}:{TTS_PORT}"
STT_CONTAINER = "mist-bench-chain-stt"
TTS_CONTAINER = "mist-bench-chain-tts"
STT_SERVICE_MODULE = "scripts.model_bench.speech.probes.stt_service"
TTS_SERVICE_MODULE = "scripts.model_bench.speech.probes.tts_service"
# Docker forwards a published port to the container's own interface, so the service listens on
# every interface inside its container; `-p 127.0.0.1:...` keeps it off the host's network.
CONTAINER_BIND = "0.0.0.0"  # noqa: S104 - inside the container only; published to loopback

DONE_RESULT = "done"
HAND_OFF_ACK = "One moment."
CHAIN_UTTERANCES: tuple[str, ...] = (
    "U01", "U03", "U04", "U06", "U08", "U09", "U11", "U12", "U16", "U18",
    "U19", "U20",
    "U07", "U15",
)  # fmt: skip
TURN_KINDS = {"command": "command", "no_tool": "short", "hand_off": "hand_off"}
DEFAULT_REPS = 2
BUSY_REPS = 1
BUSY_ARM = lowtier_suite.E4B_ARM
BUSY_PLACEMENT = "gpu-full"
BUSY_PLACEMENT_DIR = "gpu-full-e4b-writing"
DEFAULT_BASELINE_S = 60.0
SERVICE_TIMEOUT_S = 600.0
HEALTH_POLL_S = 1.0
LOAD_GEN_BASE_URL = tts_suite.LOAD_GEN_BASE_URL

Clock = Callable[[], float]
Opener = Callable[..., object]


# ---------------------------------------------------------------------------
# Inputs: the recordings, their tool prompts and labels
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChainItem:
    """One chain recording, the tool prompt with its words, and the turn's label."""

    uid: str
    script: str
    path: Path
    prompt: toolcall.ToolPrompt
    kind: str


@dataclass(frozen=True, slots=True)
class ChainInputs:
    """The recordings in run order, and what every turn sends the model."""

    items: dict[str, ChainItem]
    system_prompt: str
    tools: list[dict]


def _script_texts(path: Path = core.UTTERANCES_PATH) -> dict[str, str]:
    doc = core.load_json(path)
    return {str(u["id"]): str(u["text"]) for u in doc["utterances"]}


def _prompts_by_text(tools: Sequence[Mapping[str, object]]) -> dict[str, toolcall.ToolPrompt]:
    return {p.text: p for p in toolcall.load_prompts(tools=tools)}


def load_chain_inputs(material_dir: Path, utt_ids: Sequence[str] = CHAIN_UTTERANCES) -> ChainInputs:
    """Verify the material and pair each recording with the tool prompt of the same words."""
    manifest = core.verify_material(material_dir / "manifest.json")
    by_id = {item.id: item for item in manifest.items}
    missing = [uid for uid in utt_ids if uid not in by_id]
    if missing:
        raise core.ManifestError(f"{manifest.path} has no recording for {missing}")
    tools = toolcall.load_tools()
    prompts = _prompts_by_text(tools)
    scripts = _script_texts()
    items = {}
    for uid in utt_ids:
        script = scripts.get(uid)
        prompt = prompts.get(script or "")
        if prompt is None:
            raise core.InputError(f"{uid}: no tool prompt has its words {script!r}")
        path = material_dir / by_id[uid].file
        items[uid] = ChainItem(uid, script, path, prompt, TURN_KINDS[prompt.kind])
    return ChainInputs(items, lowtier_suite.load_lowtier_prompt(), tools)


# ---------------------------------------------------------------------------
# Clients for the three services
# ---------------------------------------------------------------------------


def _url_of(request: object) -> str:
    return request.full_url if isinstance(request, urllib.request.Request) else str(request)


class _LineRecorder:
    """Passes a stream through, keeping each `data:` line in arrival order."""

    def __init__(self, inner: object, lines: list[str]):
        self._inner = inner
        self._entered: object = inner
        self._lines = lines

    def __enter__(self) -> _LineRecorder:
        self._entered = self._inner.__enter__()
        return self

    def __exit__(self, *exc: object) -> object:
        return self._inner.__exit__(*exc)

    @property
    def status(self) -> object:
        return getattr(self._entered, "status", None)

    def __iter__(self) -> Iterator[bytes | str]:
        for raw in self._entered:
            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if line.startswith("data:"):
                self._lines.append(line.rstrip("\r\n"))
            yield raw


def _line_pieces(line: str) -> tuple[int, bool]:
    """Characters of reply text in one `data:` line, and whether it carries a tool-call piece."""
    payload = line[len("data:") :].strip()
    if payload == "[DONE]":
        return 0, False
    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        return 0, False
    if not isinstance(event, dict):
        return 0, False
    chars, tool = 0, False
    for choice in event.get("choices") or []:
        delta = choice.get("delta") or {}
        chars += len(delta.get("content") or "")
        tool = tool or bool(delta.get("tool_calls"))
    return chars, tool


@dataclass(frozen=True, slots=True)
class LlmReply:
    """One streamed reply, with when each piece of text and the last tool-call piece arrived."""

    result: core.ChatStreamResult
    text_marks: tuple[tuple[float, int], ...]
    t_last_tool_call: float | None

    def t_text(self, chars: int) -> float | None:
        """When the reply's text first reached `chars` characters."""
        for t, total in self.text_marks:
            if total >= chars:
                return t
        return None


class LlmClient:
    """The served low-level model: its alias, and streamed chat with per-line arrival times."""

    def __init__(
        self,
        base_url: str,
        *,
        opener: Opener | None = None,
        clock: Clock = time.perf_counter,
        timeout: float = 300.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._opener = opener or urllib.request.urlopen
        self._clock = clock
        self._timeout = timeout

    def served_alias(self) -> str | None:
        """The one alias the server lists, or None while nothing answers."""
        client = lowtier_suite.ServerClient(
            self.base_url, opener=self._opener, clock=self._clock, timeout=self._timeout
        )
        return client.served_alias()

    def chat(self, body: Mapping[str, object]) -> LlmReply:
        """Stream one reply; `core.stream_chat` reads the clock once, then once per line."""
        readings: list[float] = []
        lines: list[str] = []

        def clock() -> float:
            value = self._clock()
            readings.append(value)
            return value

        def opener(request: urllib.request.Request, timeout: float) -> _LineRecorder:
            return _LineRecorder(self._opener(request, timeout=timeout), lines)

        result = core.stream_chat(
            self.base_url, body, clock=clock, timeout=self._timeout, opener=opener
        )
        marks: list[tuple[float, int]] = []
        total = 0
        last_tool: float | None = None
        for t, line in zip(readings[1:], lines, strict=False):
            chars, tool = _line_pieces(line)
            if chars:
                total += chars
                marks.append((t, total))
            if tool:
                last_tool = t
        return LlmReply(result, tuple(marks), last_tool)


def _post(url: str, data: bytes, content_type: str) -> urllib.request.Request:
    return urllib.request.Request(
        url, data=data, headers={"Content-Type": content_type}, method="POST"
    )


@dataclass(frozen=True, slots=True)
class SttReply:
    """The text, the service's own transcription time, and the client's send and reply times."""

    text: str
    server_ms: float | None
    t_request: float
    t_end: float


class SttClient:
    """`POST /transcribe` on the Parakeet service: a WAV body, a JSON reply."""

    def __init__(
        self,
        base_url: str = STT_URL,
        *,
        opener: Opener | None = None,
        clock: Clock = time.perf_counter,
        timeout: float = 120.0,
    ):
        self.url = base_url.rstrip("/") + "/transcribe"
        self._opener = opener or urllib.request.urlopen
        self._clock = clock
        self._timeout = timeout

    def transcribe(self, wav: bytes) -> SttReply:
        """The service's text for `wav`, timed from sending to the whole reply."""
        request = _post(self.url, wav, "audio/wav")
        t_request = self._clock()
        try:
            with self._opener(request, timeout=self._timeout) as response:
                status = getattr(response, "status", None)
                if status != 200:
                    raise core.StreamError(f"POST {self.url} returned HTTP {status}")
                body = response.read()
            t_end = self._clock()
            doc = json.loads(body.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise core.StreamError(f"POST {self.url} returned HTTP {exc.code}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise core.StreamError(f"POST {self.url} failed: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise core.StreamError(f"POST {self.url} returned malformed JSON: {exc}") from exc
        if not isinstance(doc, dict) or not isinstance(doc.get("text"), str):
            raise core.StreamError(f"POST {self.url} returned no text")
        server_ms = doc.get("server_ms")
        return SttReply(
            doc["text"], float(server_ms) if server_ms is not None else None, t_request, t_end
        )


@dataclass(frozen=True, slots=True)
class TtsReply:
    """When the text was sent, when its first and last audio arrived, and how much there was."""

    t_request: float
    t_first_audio: float
    t_end: float
    n_bytes: int
    sample_rate: int


class TtsClient:
    """`POST /speak` on the Pocket TTS service: text in, chunked 16-bit PCM out."""

    def __init__(
        self,
        base_url: str = TTS_URL,
        *,
        opener: Opener | None = None,
        clock: Clock = time.perf_counter,
        timeout: float = 120.0,
    ):
        self.url = base_url.rstrip("/") + "/speak"
        self._opener = opener or urllib.request.urlopen
        self._clock = clock
        self._timeout = timeout

    def speak(self, text: str) -> TtsReply:
        """Speak `text`, noting when the first audio chunk arrived and when the last did."""
        data = json.dumps({"text": text}).encode("utf-8")
        request = _post(self.url, data, "application/json")
        t_request = self._clock()
        t_first: float | None = None
        n_bytes = 0
        try:
            with self._opener(request, timeout=self._timeout) as response:
                status = getattr(response, "status", None)
                if status != 200:
                    raise core.StreamError(f"POST {self.url} returned HTTP {status}")
                rate = int(response.headers.get("X-Sample-Rate") or 0)
                read = getattr(response, "read1", None) or response.read
                while chunk := read(65536):
                    if t_first is None:
                        t_first = self._clock()
                    n_bytes += len(chunk)
            t_end = self._clock()
        except urllib.error.HTTPError as exc:
            raise core.StreamError(f"POST {self.url} returned HTTP {exc.code}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise core.StreamError(f"POST {self.url} failed: {exc}") from exc
        except ValueError as exc:
            raise core.StreamError(f"POST {self.url} sent a bad X-Sample-Rate: {exc}") from exc
        if t_first is None or rate <= 0:
            raise core.StreamError(f"POST {self.url} returned no audio")
        return TtsReply(t_request, t_first, t_end, n_bytes, rate)


# ---------------------------------------------------------------------------
# One turn
# ---------------------------------------------------------------------------

ROW_FIELDS = (
    "utt_id", "rep", "kind", "prompt_id", "alias", "transcript", "stt_ms", "stt_server_ms",
    "llm_first_token_ms", "llm_server_prompt_ms", "tool_call_ms", "done_first_token_ms",
    "first_clause_ms", "spoken_text", "tts_first_audio_ms", "tts_total_ms", "tts_audio_s",
    "turn_ms", "turn_wall_ms", "transcript_wer", "tool_valid", "tool_valid_first",
    "tool_n_calls", "tool_problems", "raw_tool_calls", "reply", "t_turn_start", "t_turn_end",
    "error",
)  # fmt: skip


def _ms(later: float | None, earlier: float | None) -> float | None:
    if later is None or earlier is None:
        return None
    return (later - earlier) * 1000.0


def clause_chars(reply: str, spoken: str) -> int:
    """Characters of the reply the client must hold before it knows the clause is whole.

    A clause boundary counts only once the space after it arrives, so a clause shorter than the
    reply needs one character more; a reply with no boundary is spoken whole, at its end.
    """
    return len(spoken) + 1 if len(spoken) < len(reply) else len(reply)


def first_request_body(alias: str, inputs: ChainInputs, transcript: str, seed: int) -> dict:
    """Run 7's tool request (as `lowtier`'s), with the transcript as the user's words."""
    return {
        "model": alias,
        "messages": [
            {"role": "system", "content": inputs.system_prompt},
            {"role": "user", "content": transcript},
        ],
        "tools": inputs.tools,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "max_tokens": lowtier_suite.TOOL_REPLY_TOKENS,
        "seed": seed,
        "cache_prompt": True,
    }


def follow_up_body(first: Mapping[str, object], result: core.ChatStreamResult) -> dict:
    """The first request plus the model's calls, each answered with the fixed result `done`."""
    calls = []
    results = []
    for index, call in enumerate(result.tool_calls):
        call_id = call.get("id") or f"call_{index}"
        function = call.get("function") or {}
        name = str(function.get("name", ""))
        calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": str(function.get("arguments", ""))},
            }
        )
        results.append(
            {"role": "tool", "tool_call_id": call_id, "name": name, "content": DONE_RESULT}
        )
    assistant = {"role": "assistant", "content": result.content, "tool_calls": calls}
    return {**first, "messages": [*first["messages"], assistant, *results]}


def run_turn(
    item: ChainItem,
    *,
    rep: int,
    alias: str,
    inputs: ChainInputs,
    llm: LlmClient,
    stt: SttClient,
    tts: TtsClient,
    clock: Clock = time.perf_counter,
) -> dict[str, object]:
    """One chained turn; a stage that fails leaves its error on the row instead of raising."""
    row: dict[str, object] = dict.fromkeys(ROW_FIELDS)
    row.update(utt_id=item.uid, rep=rep, kind=item.kind, prompt_id=item.prompt.id, alias=alias)
    wav = item.path.read_bytes()
    t0 = clock()
    row["t_turn_start"] = t0
    try:
        heard = stt.transcribe(wav)
        stt_ms = _ms(heard.t_end, heard.t_request)
        row.update(stt_ms=stt_ms, stt_server_ms=heard.server_ms, transcript=heard.text)
        row["transcript_wer"] = wer.word_error_rate(item.script, heard.text).wer

        body = first_request_body(alias, inputs, heard.text, rep)
        first = llm.chat(body)
        res = first.result
        row["llm_first_token_ms"] = _ms(res.t_first_delta, res.t_request)
        row["llm_server_prompt_ms"] = (res.timings or {}).get("prompt_ms")
        calls = list(res.tool_calls)
        score = toolcall.score_calls(item.prompt, calls, inputs.tools)
        row.update(
            tool_valid=score.valid_all,
            tool_valid_first=score.valid_first,
            tool_n_calls=score.n_calls,
            tool_problems=list(score.problems),
            raw_tool_calls=calls,
        )
        if calls:
            row["tool_call_ms"] = _ms(first.t_last_tool_call, res.t_request)

        names = {str((c.get("function") or {}).get("name", "")) for c in calls}
        if toolcall.HAND_OFF_TOOL in names:
            spoken = HAND_OFF_ACK
            first_clause_ms = row["tool_call_ms"]
        elif calls:
            second = llm.chat(follow_up_body(body, res))
            done = second.result
            row["done_first_token_ms"] = _ms(done.t_first_content, done.t_request)
            reply = done.content
            spoken = tts_probe.first_segment(reply)
            t_clause = second.t_text(clause_chars(reply, spoken))
            clause = _ms(t_clause, done.t_request)
            first_clause_ms = None if clause is None else _ms(res.t_end, res.t_request) + clause
            row["reply"] = reply
        else:
            reply = res.content
            spoken = tts_probe.first_segment(reply)
            first_clause_ms = _ms(first.t_text(clause_chars(reply, spoken)), res.t_request)
            row["reply"] = reply
        if not spoken.strip() or first_clause_ms is None:
            raise core.StreamError("the model returned no text to speak")
        row.update(first_clause_ms=first_clause_ms, spoken_text=spoken)

        voice = tts.speak(spoken)
        tts_first = _ms(voice.t_first_audio, voice.t_request)
        row.update(
            tts_first_audio_ms=tts_first,
            tts_total_ms=_ms(voice.t_end, voice.t_request),
            tts_audio_s=voice.n_bytes / 2 / voice.sample_rate,
            turn_ms=stt_ms + first_clause_ms + tts_first,
            turn_wall_ms=_ms(voice.t_first_audio, t0),
        )
        row["t_turn_end"] = voice.t_end
    except core.SpeechBenchError as exc:
        row["error"] = str(exc)
        row["t_turn_end"] = clock()
    return row


# ---------------------------------------------------------------------------
# The two services' containers
# ---------------------------------------------------------------------------


def _arms(arms: Mapping[str, object] | None) -> Mapping[str, object]:
    return arms if arms is not None else core.load_json(core.SPEECH_ARMS_PATH)


def _service_head(name: str, port: int, cfg: Mapping[str, object], what: str) -> list[str]:
    argv = ["docker", "run", "-d", "--name", name, *docker_cmds.publish_local(port, port)]
    argv += stt_suite.resource_args(cfg, what)
    return argv + docker_cmds.env_args({"PYTHONDONTWRITEBYTECODE": "1"})


def stt_service_argv(
    *, repo_dir: str, speech_models_dir: str, arms: Mapping[str, object] | None = None
) -> list[str]:
    """`docker run -d` of the Parakeet service: the asr image, 4 CPU threads, no GPU."""
    cfg = _arms(arms)["stt"][STT_ENGINE]
    threads = stt_suite.cpu_threads(cfg, STT_ENGINE)
    argv = _service_head(STT_CONTAINER, STT_PORT, cfg, STT_ENGINE)
    argv += docker_cmds.ro_mount(repo_dir, docker_cmds.REPO_MOUNT)
    argv += docker_cmds.ro_mount(speech_models_dir, stt_suite.CONTAINER_WEIGHTS)
    argv += ["-w", docker_cmds.REPO_MOUNT, str(cfg["image"]), "python", "-m", STT_SERVICE_MODULE]
    argv += ["--engine", STT_ENGINE, "--weights", stt_suite.CONTAINER_WEIGHTS]
    return argv + ["--threads", str(threads), "--host", CONTAINER_BIND, "--port", str(STT_PORT)]


def tts_service_argv(
    *, repo_dir: str, speech_models_dir: str, arms: Mapping[str, object] | None = None
) -> list[str]:
    """`docker run -d` of the Pocket TTS service: its image, 4 CPU threads, the stock voice."""
    cfg = _arms(arms)["tts"][TTS_ENGINE]
    argv = _service_head(TTS_CONTAINER, TTS_PORT, cfg, TTS_ENGINE)
    argv += docker_cmds.ro_mount(repo_dir, docker_cmds.REPO_MOUNT)
    argv += docker_cmds.ro_mount(speech_models_dir, tts_engines.CONTAINER_MODELS_DIR)
    argv += ["-w", docker_cmds.REPO_MOUNT, str(cfg["image"]), "python", "-m", TTS_SERVICE_MODULE]
    argv += ["--models-dir", tts_engines.CONTAINER_MODELS_DIR]
    return argv + ["--host", CONTAINER_BIND, "--port", str(TTS_PORT)]


class ServiceDocker(Protocol):
    """The docker operations the services need; `core.HostDocker` is the real one."""

    def container_state(self, name: str) -> str:
        """`running`, `exited`, `absent` or another state."""
        ...

    def run_detached(self, argv: list[str]) -> str:
        """Run a detached container."""
        ...

    def logs(self, name: str) -> tuple[str, str]:
        """The container's stdout and stderr."""
        ...

    def stop(self, name: str) -> None:
        """Stop the container."""
        ...

    def rm(self, name: str) -> None:
        """Remove the container."""
        ...


@dataclass(frozen=True, slots=True)
class ServiceSpec:
    """One service container: its name, argv, health route and log file name."""

    container: str
    argv: tuple[str, ...]
    health_url: str
    log_name: str


@dataclass
class ChainServices:
    """Starts both service containers, waits for each to answer, and removes them after."""

    specs: tuple[ServiceSpec, ...]
    docker: ServiceDocker
    opener: Opener
    clock: Clock
    sleep: Callable[[float], None]
    timeout_s: float = SERVICE_TIMEOUT_S
    started: list[ServiceSpec] = field(default_factory=list)

    @classmethod
    def default(
        cls,
        *,
        repo_dir: str,
        speech_models_dir: str,
        docker: ServiceDocker | None = None,
        opener: Opener | None = None,
        clock: Clock = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        timeout_s: float = SERVICE_TIMEOUT_S,
    ) -> ChainServices:
        """The Parakeet service, then the Pocket TTS service."""
        specs = (
            ServiceSpec(
                STT_CONTAINER,
                tuple(stt_service_argv(repo_dir=repo_dir, speech_models_dir=speech_models_dir)),
                f"{STT_URL}/health",
                "stt_service.log",
            ),
            ServiceSpec(
                TTS_CONTAINER,
                tuple(tts_service_argv(repo_dir=repo_dir, speech_models_dir=speech_models_dir)),
                f"{TTS_URL}/health",
                "tts_service.log",
            ),
        )
        return cls(
            specs,
            docker or core.HostDocker(),
            opener or urllib.request.urlopen,
            clock,
            sleep,
            timeout_s,
        )

    def start(self) -> None:
        """Refuse a leftover container by name, then start each service and wait for it."""
        for spec in self.specs:
            state = self.docker.container_state(spec.container)
            if state != "absent":
                raise core.SpeechBenchError(
                    f"container {spec.container} already exists ({state}); remove it first"
                )
        for spec in self.specs:
            self.docker.run_detached(list(spec.argv))
            self.started.append(spec)
            self._wait_healthy(spec)

    def _answers(self, url: str) -> bool:
        try:
            with self.opener(urllib.request.Request(url), timeout=5.0) as response:
                return getattr(response, "status", None) == 200
        except (urllib.error.URLError, OSError):
            return False

    def _wait_healthy(self, spec: ServiceSpec) -> None:
        deadline = self.clock() + self.timeout_s
        while not self._answers(spec.health_url):
            state = self.docker.container_state(spec.container)
            if state != "running":
                raise core.ServeFailedError(
                    f"{spec.container} is {state} before {spec.health_url} answered"
                )
            if self.clock() > deadline:
                raise core.ServeFailedError(
                    f"{spec.health_url} did not answer within {self.timeout_s:.0f} s"
                )
            self.sleep(HEALTH_POLL_S)

    def stop(self, log_dir: Path) -> None:
        """Stop each started service, save its logs to `log_dir`, then remove it."""
        while self.started:
            spec = self.started.pop(0)
            try:
                self.docker.stop(spec.container)
                out, err = self.docker.logs(spec.container)
                text = f"=== stdout ===\n{out}\n=== stderr ===\n{err}\n"
                (Path(log_dir) / spec.log_name).write_text(text, encoding="utf-8")
                self.docker.rm(spec.container)
            except (bench_host.BenchHostError, OSError) as exc:
                print(f"[WARN] {spec.container}: {exc}; remove it by hand", file=sys.stderr)


class Services(Protocol):
    """What `run_chain` needs of the services; `ChainServices` is the real one."""

    def start(self) -> None:
        """Start the services."""
        ...

    def stop(self, log_dir: Path) -> None:
        """Stop the services, keeping their logs."""
        ...


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChainRequest:
    """One chain run: a served alias, the material, and optionally E4B writing beside it."""

    results_root: Path
    run: str
    model: str
    placement: str
    material_dir: Path
    speech_models_dir: Path
    reps: int | None = None
    busy_arm: str | None = None
    baseline_s: float = DEFAULT_BASELINE_S
    remote_url: str | None = None
    utt_ids: tuple[str, ...] = CHAIN_UTTERANCES


class LoadGen(Protocol):
    """The part of `load_gen.LoadGenerator` a busy run uses."""

    def run_for(self, seconds: float) -> list[load_gen.Completion]:
        """Run completions in this thread for `seconds`."""
        ...

    def start(self) -> None:
        """Start the loop in the background."""
        ...

    def stop(self) -> list[load_gen.Completion]:
        """Stop and return the completions."""
        ...


def _check_request(request: ChainRequest, container_state: Callable[[str], str]) -> str:
    """Refuse a request that cannot run, before anything is written; return its alias."""
    core.speech_root(request.results_root, request.run)
    arms = core.load_small_arms()
    if request.model not in arms["models"]:
        raise core.PlacementError(
            f"unknown small model {request.model!r}; known: {list(arms['models'])}"
        )
    alias = core.small_alias(request.model, request.placement)
    if request.remote_url is not None:
        lowtier_suite.validate_remote_aliases([alias])
    elif request.placement not in core.LOCAL_PLACEMENTS:
        raise core.PlacementError(
            f"placement {request.placement!r} is not local ({list(core.LOCAL_PLACEMENTS)}); "
            f"a host-1070 placement needs --remote-url"
        )
    if request.busy_arm is not None:
        if request.busy_arm != BUSY_ARM:
            raise core.SpeechBenchError(f"--busy-arm takes only {BUSY_ARM}")
        if request.placement != BUSY_PLACEMENT or request.remote_url is not None:
            raise core.SpeechBenchError(
                f"--busy-arm {BUSY_ARM} runs on {BUSY_PLACEMENT} only, with E4B on the same card"
            )
        state = container_state(bench_host.BENCH_LLM_CONTAINER)
        if state != "running":
            raise core.SpeechBenchError(
                f"--busy-arm {BUSY_ARM} needs {bench_host.BENCH_LLM_CONTAINER} running "
                f"(it is {state}); serve E4B inside a bench window first"
            )
    if request.reps is not None and request.reps < 1:
        raise core.SpeechBenchError(f"--reps must be at least 1, not {request.reps}")
    return alias


def _busy_fields(row: Mapping[str, object], completions: Sequence[load_gen.Completion]) -> dict:
    start, end = row.get("t_turn_start"), row.get("t_turn_end")
    if start is None or end is None:
        return {"busy_completions": 0, "busy_decode_tps": None}
    overlapping = [c for c in completions if c.t_request < end and c.t_end > start]
    return {
        "busy_completions": len(overlapping),
        "busy_decode_tps": load_gen.decode_tps(overlapping),
    }


def run_chain(
    request: ChainRequest,
    *,
    llm: LlmClient,
    stt: SttClient,
    tts: TtsClient,
    services: Services,
    container_state: Callable[[str], str],
    load_gen_factory: Callable[[], LoadGen] | None = None,
    clock: Clock = time.perf_counter,
    git_sha: str | None = None,
) -> Path:
    """Run every recording `reps` times through the chain; return the claimed directory."""
    alias = _check_request(request, container_state)
    busy = request.busy_arm is not None
    if busy and load_gen_factory is None:
        raise core.SpeechBenchError("--busy-arm needs a load generator")
    inputs = load_chain_inputs(request.material_dir, request.utt_ids)
    served = llm.served_alias()
    if served != alias:
        raise lowtier_suite.AliasError(
            f"{llm.base_url} serves {served!r}, not {alias!r}; serve {alias} first"
        )
    reps = request.reps or (BUSY_REPS if busy else DEFAULT_REPS)
    placement_dir = BUSY_PLACEMENT_DIR if busy else request.placement
    claim = core.claim_output(
        request.results_root,
        request.run,
        SUITE,
        f"{request.model}/{placement_dir}",
        git_sha=git_sha or core.current_git_sha(),
    )
    voice = core.load_json(core.SPEECH_ARMS_PATH)["tts"][TTS_ENGINE]["timing_voice"]
    meta = {
        "alias": alias,
        "model": request.model,
        "placement": request.placement,
        "busy_arm": request.busy_arm,
        "reps": reps,
        "utterances": list(request.utt_ids),
        "llm_url": llm.base_url,
        "stt_engine": STT_ENGINE,
        "tts_engine": TTS_ENGINE,
        "voice": voice,
        "done_result": DONE_RESULT,
        "hand_off_reply": HAND_OFF_ACK,
        "turn_ms": "stt_ms + first_clause_ms + tts_first_audio_ms",
    }
    core.write_json_exclusive(claim.path / "chain.json", meta)

    def each_turn() -> Iterator[dict[str, object]]:
        # Recording by recording, its repetitions together, as run 3 orders its material.
        for item in inputs.items.values():
            for rep in range(1, reps + 1):
                yield run_turn(
                    item, rep=rep, alias=alias, inputs=inputs, llm=llm, stt=stt, tts=tts,
                    clock=clock,
                )  # fmt: skip

    turns = claim.jsonl("turns.jsonl")
    try:
        services.start()
        if busy:
            _run_busy(claim, turns, each_turn, load_gen_factory, request.baseline_s)
        else:
            for row in each_turn():
                turns.write(row)
        return claim.path
    finally:
        turns.close()
        services.stop(claim.path)


def _run_busy(
    claim: core.OutputClaim,
    turns: core.JsonlWriter,
    each_turn: Callable[[], Iterator[dict[str, object]]],
    factory: Callable[[], LoadGen],
    baseline_s: float,
) -> None:
    """E4B decoding alone first, then throughout the turns; each turn gets the overlap's rate."""
    alone = factory().run_for(baseline_s)
    generator = factory()
    generator.start()
    rows: list[dict[str, object]] = []
    stop_error: core.SpeechBenchError | None = None
    try:
        for row in each_turn():
            rows.append(row)
    finally:
        try:
            with_chain = generator.stop()
        except core.SpeechBenchError as exc:
            stop_error, with_chain = exc, []
        for row in rows:
            turns.write({**row, **_busy_fields(row, with_chain)})
        with claim.jsonl("decode.jsonl") as decode:
            for completion in alone:
                decode.write({"phase": "alone", **completion.as_row()})
            for completion in with_chain:
                decode.write({"phase": "with_chain", **completion.as_row()})
    if stop_error is not None:
        raise stop_error


# ---------------------------------------------------------------------------
# CLI and the suite contract
# ---------------------------------------------------------------------------


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("chain", help="run 7's chained low-level turn")
    core.add_results_args(parser)
    parser.add_argument("--model", required=True, help="small model id, as small_arms.json")
    parser.add_argument("--placement", required=True, help="the placement small-serve serves")
    parser.add_argument(
        "--material-dir", type=Path, required=True, help="material directory with manifest.json"
    )
    env_models = os.environ.get(tts_suite.SPEECH_MODELS_DIR_ENV)
    parser.add_argument(
        "--speech-models-dir",
        type=Path,
        default=Path(env_models) if env_models else None,
        help="host speech models directory, <org>/<repo>/ layout "
        f"(env {tts_suite.SPEECH_MODELS_DIR_ENV})",
    )
    parser.add_argument(
        "--busy-arm", choices=[BUSY_ARM], help="run while E4B writes on mist-bench-llm"
    )
    parser.add_argument("--reps", type=int, help=f"repetitions (default {DEFAULT_REPS}, busy 1)")
    parser.add_argument("--utt", action="append", default=[], help="limit to these recordings")
    parser.add_argument("--remote-url", help="host-1070's server, for a gtx1070-shared placement")
    parser.add_argument(
        "--baseline-s", type=float, default=DEFAULT_BASELINE_S, help="E4B alone before a busy run"
    )
    parser.add_argument("--service-timeout", type=float, default=SERVICE_TIMEOUT_S)
    parser.set_defaults(func=cmd_chain)


def cmd_chain(args: argparse.Namespace) -> int:
    if args.speech_models_dir is None:
        raise core.SpeechBenchError(
            f"--speech-models-dir (or {tts_suite.SPEECH_MODELS_DIR_ENV}) names the speech models"
        )
    unknown = [u for u in args.utt if u not in CHAIN_UTTERANCES]
    if unknown:
        raise core.SpeechBenchError(f"--utt {unknown} are not chain recordings")
    clock = time.perf_counter
    base_url = args.remote_url or core.SMALL_BASE_URL
    request = ChainRequest(
        results_root=args.results_root,
        run=args.run,
        model=args.model,
        placement=args.placement,
        material_dir=args.material_dir,
        speech_models_dir=args.speech_models_dir,
        reps=args.reps,
        busy_arm=args.busy_arm,
        baseline_s=args.baseline_s,
        remote_url=args.remote_url,
        utt_ids=tuple(u for u in CHAIN_UTTERANCES if u in args.utt) or CHAIN_UTTERANCES,
    )
    services = ChainServices.default(
        repo_dir=core.REPO_ROOT.resolve().as_posix(),
        speech_models_dir=args.speech_models_dir.resolve().as_posix(),
        timeout_s=args.service_timeout,
    )
    out = run_chain(
        request,
        llm=LlmClient(base_url, clock=clock),
        stt=SttClient(STT_URL, clock=clock),
        tts=TtsClient(TTS_URL, clock=clock),
        services=services,
        container_state=core.HostDocker().container_state,
        load_gen_factory=lambda: load_gen.LoadGenerator(LOAD_GEN_BASE_URL, clock=clock),
        clock=clock,
    )
    rows = [json.loads(line) for line in (out / "turns.jsonl").read_text("utf-8").splitlines()]
    failed = [f"{r['utt_id']} rep {r['rep']}" for r in rows if r.get("error")]
    print(f"chain: {len(rows)} turns at {out}" + (f"; failed: {failed}" if failed else ""))
    return 1 if failed else 0


def container_argv(ctx: core.PlanContext) -> list[tuple[str, list[str]]]:
    return [
        (
            "chain stt-service",
            stt_service_argv(repo_dir=ctx.repo_dir, speech_models_dir=ctx.speech_models_dir),
        ),
        (
            "chain tts-service",
            tts_service_argv(repo_dir=ctx.repo_dir, speech_models_dir=ctx.speech_models_dir),
        ),
    ]


def validate_inputs(repo_root: Path) -> list[str]:
    """Every chain recording is in the script, with a tool prompt of its words."""
    problems: list[str] = []
    try:
        scripts = _script_texts(repo_root / core.SPEECH_DIR_REL / "utterances.json")
        prompts = _prompts_by_text(toolcall.load_tools())
        arms = core.load_json(repo_root / core.SPEECH_DIR_REL / "speech_arms.json")
    except core.SpeechBenchError as exc:
        return [f"chain: {exc}"]
    kinds = []
    for uid in CHAIN_UTTERANCES:
        if uid not in scripts:
            problems.append(f"chain: {uid} is not in utterances.json")
            continue
        prompt = prompts.get(scripts[uid])
        if prompt is None:
            problems.append(f"chain: {uid} has no tool prompt of its words")
            continue
        kinds.append(TURN_KINDS[prompt.kind])
    if not problems and sorted(set(kinds)) != ["command", "hand_off", "short"]:
        problems.append(f"chain: the recordings cover {sorted(set(kinds))}, not all three kinds")
    pocket = arms.get("tts", {}).get(TTS_ENGINE, {})
    if pocket.get("timing_voice") not in pocket.get("voices", []):
        problems.append("speech_arms.json: pocket's timing voice is not one of its stock voices")
    parakeet = arms.get("stt", {}).get(STT_ENGINE, {})
    if parakeet.get("device") != "cpu" or int(parakeet.get("threads", 0)) != 4:
        problems.append("speech_arms.json: parakeet-cpu must run on 4 CPU threads")
    return problems


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def selftest() -> None:
    for build, port in ((stt_service_argv, STT_PORT), (tts_service_argv, TTS_PORT)):
        argv = build(repo_dir="/r", speech_models_dir="/m")
        _check(argv[argv.index("-p") + 1] == f"127.0.0.1:{port}:{port}", "service not on loopback")
        _check("--gpus" not in argv, "a chain service asks for the GPU")
        _check(argv[argv.index("--cpus") + 1] == "4", "a chain service is not on 4 CPUs")
    reply = "Your ten minute timer is set and running now, and it rings at ten past nine."
    spoken = tts_probe.first_segment(reply)
    _check(spoken == "Your ten minute timer is set and running now,", "first clause wrong")
    _check(clause_chars(reply, spoken) == len(spoken) + 1, "clause needs the space after it")
    _check(clause_chars("Yes.", "Yes.") == 4, "a whole reply ends at its end")
    _check(validate_inputs(core.REPO_ROOT) == [], "chain inputs do not validate")
