"""Run 7's speech-to-text service: T3's Parakeet adapter behind `POST /transcribe`.

The chained turn (`chain.py`) posts each recording here as a WAV body and reads back the text and
the server's own transcription time. The service wraps `stt_engines.make_transcriber`, so the
chain times the adapter run 3 timed, with the same 4 CPU threads.

It binds 127.0.0.1 by default. In its container `chain.py` passes `--host 0.0.0.0`, because
Docker forwards a published port to the container's own interface, not to its loopback; the port
is published on the host's 127.0.0.1 only, so nothing off the machine reaches it.

numpy is imported only when a WAV is decoded, and the engine package only inside
`make_transcriber`, so importing this module loads neither.
"""

from __future__ import annotations

import argparse
import io
import json
import signal
import sys
import time
import wave
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Protocol

from . import stt_engines

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8091
DEFAULT_ENGINE = "parakeet-cpu"
DEFAULT_THREADS = 4
DEFAULT_WEIGHTS = "/weights"
SAMPLE_RATE = 16000
MAX_BODY_BYTES = 64 * 2**20


class WavError(ValueError):
    """The posted body is not a 16-bit mono 16 kHz WAV."""


class Transcriber(Protocol):
    """The part of `engines.Transcriber` the service calls."""

    def transcribe(self, audio: object, sample_rate: int) -> str:
        """Return the text of `audio`."""
        ...


def decode_wav(data: bytes) -> tuple[object, float]:
    """The WAV's samples as float32 in [-1, 1], and its length in seconds.

    The recordings `prepare` writes are 16-bit mono at 16 kHz; anything else is refused rather
    than resampled, so the service never times a conversion the adapter would not see in MIST.
    """
    try:
        with wave.open(io.BytesIO(data), "rb") as clip:
            channels, width, rate = clip.getnchannels(), clip.getsampwidth(), clip.getframerate()
            frames = clip.readframes(clip.getnframes())
    except (wave.Error, EOFError) as exc:
        raise WavError(f"not a WAV file: {exc}") from exc
    if (channels, width, rate) != (1, 2, SAMPLE_RATE):
        raise WavError(
            f"expected 16-bit mono {SAMPLE_RATE} Hz, got {channels} channel(s), "
            f"{8 * width}-bit, {rate} Hz"
        )
    import numpy

    audio = numpy.frombuffer(frames, dtype="<i2").astype(numpy.float32) / 32768.0
    return audio, len(audio) / SAMPLE_RATE


def make_server(
    transcriber: Transcriber,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    clock: Callable[[], float] = time.perf_counter,
) -> HTTPServer:
    """An HTTP server answering `GET /health` and `POST /transcribe`; one request at a time."""

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, doc: dict) -> None:
            body = json.dumps(doc).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - http.server's handler name
            if self.path == "/health":
                self._reply(200, {"status": "ok"})
            else:
                self._reply(404, {"error": f"no route {self.path}"})

        def do_POST(self) -> None:  # noqa: N802 - http.server's handler name
            if self.path != "/transcribe":
                self._reply(404, {"error": f"no route {self.path}"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_BODY_BYTES:
                self._reply(400, {"error": f"body of {length} bytes refused"})
                return
            data = self.rfile.read(length)
            try:
                audio, audio_s = decode_wav(data)
            except WavError as exc:
                self._reply(400, {"error": str(exc)})
                return
            t0 = clock()
            text = transcriber.transcribe(audio, SAMPLE_RATE)
            server_ms = (clock() - t0) * 1000.0
            reply = {"text": text, "server_ms": round(server_ms, 3), "audio_s": round(audio_s, 3)}
            self._reply(200, reply)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            sys.stderr.write("stt_service: " + (format % args) + "\n")

    return HTTPServer((host, port), Handler)


def load_transcriber(engine_id: str, weights_dir: Path, threads: int) -> Transcriber:
    """T3's adapter, loaded once at start-up from the read-only weights mount."""
    return stt_engines.make_transcriber(engine_id, weights_dir=weights_dir, threads=threads)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", default=DEFAULT_ENGINE)
    parser.add_argument("--weights", type=Path, default=Path(DEFAULT_WEIGHTS))
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser


def _interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    transcriber = load_transcriber(args.engine, args.weights, args.threads)
    import numpy

    # One untimed second of silence first, so the chain's first turn does not time the warm-up.
    transcriber.transcribe(numpy.zeros(SAMPLE_RATE, dtype=numpy.float32), SAMPLE_RATE)
    server = make_server(transcriber, host=args.host, port=args.port)
    signal.signal(signal.SIGTERM, _interrupt)
    print(f"stt_service: {args.engine} on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
