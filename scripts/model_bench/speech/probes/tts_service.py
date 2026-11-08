"""Run 7's speech service: T2's Pocket TTS adapter behind `POST /speak`, streaming PCM.

The chained turn (`chain.py`) posts the first clause of each reply here and times the first audio
chunk. The body is a JSON object holding `text` and nothing else: the service speaks only with
Pocket's stock timing voice from `speech_arms.json`, loaded by name at start-up, and takes no voice
of any kind from a request. The reply is 16-bit little-endian mono PCM at the engine's rate (the
`X-Sample-Rate` header), sent with chunked transfer encoding, one chunk per engine chunk, so the
first chunk leaves as soon as the engine yields it.

It binds 127.0.0.1 by default; in its container `chain.py` passes `--host 0.0.0.0` behind a port
published on the host's 127.0.0.1 only (see `stt_service.py`).

numpy is imported only to convert a chunk, and the engine package only inside the adapter's
`load`, so importing this module loads neither.
"""

from __future__ import annotations

import argparse
import importlib
import json
import signal
import sys
from collections.abc import Callable, Iterator, Mapping
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Protocol

from .. import core
from . import tts_engines

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8092
ENGINE_ID = "pocket"
MAX_BODY_BYTES = 64 * 1024
WARMUP_TEXT = "The kettle has boiled, and the tea is ready in the pot."


class SpeakError(ValueError):
    """The posted body is not a JSON object holding only a non-empty `text`."""


class Engine(Protocol):
    """The part of `engines.SpeechEngine` the service calls."""

    sample_rate: int

    def stream(self, text: str) -> Iterator[object]:
        """Yield the audio of `text` chunk by chunk."""
        ...


def parse_speak_body(data: bytes) -> str:
    """The text to speak, refusing any other key, so no request can name a voice."""
    try:
        doc = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpeakError(f"body is not JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise SpeakError("body is not a JSON object")
    extra = sorted(set(doc) - {"text"})
    if extra:
        raise SpeakError(f"body may hold only text; refusing {extra}")
    text = doc.get("text")
    if not isinstance(text, str) or not text.strip():
        raise SpeakError("body has no text to speak")
    return text


def pcm16(chunk: object) -> bytes:
    """One float chunk in [-1, 1] as 16-bit little-endian PCM."""
    import numpy

    audio = numpy.clip(numpy.asarray(chunk, dtype=numpy.float32).reshape(-1), -1.0, 1.0)
    return (audio * 32767.0).astype("<i2").tobytes()


def make_server(
    engine: Engine, *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> HTTPServer:
    """An HTTP/1.1 server answering `GET /health` and `POST /speak`; one request at a time."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _reply(self, status: int, doc: dict) -> None:
            body = json.dumps(doc).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - http.server's handler name
            if self.path == "/health":
                self._reply(200, {"status": "ok", "sample_rate": engine.sample_rate})
            else:
                self._reply(404, {"error": f"no route {self.path}"})

        def do_POST(self) -> None:  # noqa: N802 - http.server's handler name
            length = int(self.headers.get("Content-Length") or 0)
            data = self.rfile.read(length) if 0 < length <= MAX_BODY_BYTES else b""
            if self.path != "/speak":
                self._reply(404, {"error": f"no route {self.path}"})
                return
            try:
                text = parse_speak_body(data)
            except SpeakError as exc:
                self._reply(400, {"error": str(exc)})
                return
            self.send_response(200)
            self.send_header("Content-Type", "audio/L16")
            self.send_header("X-Sample-Rate", str(engine.sample_rate))
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for chunk in engine.stream(text):
                pcm = pcm16(chunk)
                if pcm:
                    self.wfile.write(f"{len(pcm):x}\r\n".encode("ascii") + pcm + b"\r\n")
                    self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            sys.stderr.write("tts_service: " + (format % args) + "\n")

    return HTTPServer((host, port), Handler)


def load_engine(
    models_dir: Path,
    *,
    make: Callable[..., object] = tts_engines.make_engine,
    arms: Mapping[str, object] | None = None,
    importer: Callable[[str], object] = importlib.import_module,
) -> object:
    """Pocket TTS with its stock timing voice, the voice run 1 timed."""
    doc = arms if arms is not None else core.load_json(core.SPEECH_ARMS_PATH)
    voice = str(doc["tts"][ENGINE_ID]["timing_voice"])
    engine = make(ENGINE_ID, models_dir=models_dir, arms=doc, importer=importer)
    engine.load(voice)
    return engine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-dir", type=Path, default=Path(tts_engines.CONTAINER_MODELS_DIR))
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser


def _interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = load_engine(args.models_dir)
    # One untimed sentence first, so the chain's first turn does not time the engine's warm-up.
    for _ in engine.stream(WARMUP_TEXT):
        pass
    server = make_server(engine, host=args.host, port=args.port)
    signal.signal(signal.SIGTERM, _interrupt)
    print(f"tts_service: {ENGINE_ID} on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
