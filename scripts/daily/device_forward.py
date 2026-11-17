"""The LAN relay in front of the device listener: `python -m scripts.daily.device_forward`.

    python -m scripts.daily.device_forward --listen 0.0.0.0:8443 --to mist-backend:8443 \
        [--max-connections 8] [--idle-timeout-s 120]

It runs only inside `mist-device-forward` (`deploy/daily/compose.devices.yml`), the one
container that publishes a port on the PC's LAN address, so `mist-backend` publishes nothing new
and a LAN address missing at start-up never stops the desktop path. It copies bytes both ways
and leaves TLS alone: TLS ends in the backend, so this process holds no key and sees only
ciphertext.

Bounds: at most `--max-connections` relayed at once (a further one is accepted and closed at
once), and a connection with no bytes either way for `--idle-timeout-s` is closed. Each
connection logs one line when it ends, with its peer, outcome, byte counts and duration, never
its bytes.

Standard library only, so the release image runs it with nothing added.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import re
import signal
import sys
import time
from collections.abc import Callable

LOGGER_NAME = "device_forward"
logger = logging.getLogger(LOGGER_NAME)

DEFAULT_MAX_CONNECTIONS = 8
DEFAULT_IDLE_TIMEOUT_S = 120.0
DEFAULT_POLL_S = 1.0
CHUNK_BYTES = 65536
PORT = re.compile(r"[0-9]{1,5}", re.ASCII)


def _peer(writer: asyncio.StreamWriter) -> str:
    peer = writer.get_extra_info("peername")
    if isinstance(peer, tuple) and len(peer) >= 2:
        return f"{peer[0]}:{peer[1]}"
    return "unknown"


async def _close(writer: asyncio.StreamWriter | None) -> None:
    if writer is None:
        return
    writer.close()
    with contextlib.suppress(OSError):
        await writer.wait_closed()


class Forwarder:
    """Relays each accepted connection to one upstream address.

    `clock` is read to time idleness and `poll_s` is how often idleness is checked, so a test
    moves the clock instead of waiting out the timeout.
    """

    def __init__(
        self,
        upstream: tuple[str, int],
        *,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S,
        clock: Callable[[], float] = time.monotonic,
        poll_s: float = DEFAULT_POLL_S,
    ) -> None:
        self._upstream = upstream
        self._max_connections = max_connections
        self._idle_timeout_s = idle_timeout_s
        self._clock = clock
        self._poll_s = poll_s
        self._active = 0

    @property
    def active(self) -> int:
        """How many connections are being relayed now."""
        return self._active

    async def start(self, host: str, port: int) -> asyncio.AbstractServer:
        """Listen on `host:port`; port 0 picks a free one."""
        return await asyncio.start_server(self.handle, host, port)

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Relay one client connection until either side ends it or it goes idle."""
        peer = _peer(writer)
        if self._active >= self._max_connections:
            logger.info("device_forward peer=%s outcome=refused active=%d", peer, self._active)
            await _close(writer)
            return
        self._active += 1
        started = self._clock()
        counts = {"in": 0, "out": 0}
        outcome = "closed"
        upstream_writer: asyncio.StreamWriter | None = None
        try:
            try:
                upstream_reader, upstream_writer = await asyncio.open_connection(*self._upstream)
            except OSError:
                outcome = "upstream_unreachable"
                return
            last_activity = [self._clock()]
            pipes = {
                asyncio.create_task(
                    self._pipe(reader, upstream_writer, counts, "in", last_activity)
                ),
                asyncio.create_task(
                    self._pipe(upstream_reader, writer, counts, "out", last_activity)
                ),
            }
            idle = asyncio.create_task(self._idle(last_activity))
            pending: set[asyncio.Task] = {*pipes, idle}
            while pipes:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                if idle in done:
                    outcome = "idle"
                    break
                pipes -= done
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        finally:
            await _close(writer)
            await _close(upstream_writer)
            self._active -= 1
            logger.info(
                "device_forward peer=%s outcome=%s bytes_in=%d bytes_out=%d seconds=%.1f",
                peer,
                outcome,
                counts["in"],
                counts["out"],
                self._clock() - started,
            )

    async def _pipe(
        self,
        source: asyncio.StreamReader,
        sink: asyncio.StreamWriter,
        counts: dict[str, int],
        direction: str,
        last_activity: list[float],
    ) -> None:
        """Copy until `source` ends, then pass the end on as a half-close."""
        try:
            while data := await source.read(CHUNK_BYTES):
                last_activity[0] = self._clock()
                counts[direction] += len(data)
                sink.write(data)
                await sink.drain()
            if sink.can_write_eof():
                sink.write_eof()
        except OSError:
            # A reset on either side ends this direction; the other ends with its own EOF.
            return

    async def _idle(self, last_activity: list[float]) -> None:
        """Return once no byte has moved either way for the idle timeout."""
        while True:
            await asyncio.sleep(self._poll_s)
            if self._clock() - last_activity[0] >= self._idle_timeout_s:
                return


def _address(raw: str) -> tuple[str, int]:
    host, separator, port = raw.rpartition(":")
    if not separator or not host or PORT.fullmatch(port) is None or not 1 <= int(port) <= 65535:
        raise argparse.ArgumentTypeError(f"{raw!r} is not <host>:<port>")
    return host, int(port)


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError(f"{raw!r} is not a positive whole number")
    return value


def _positive_seconds(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError:
        value = 0.0
    if not value > 0:
        raise argparse.ArgumentTypeError(f"{raw!r} is not a positive number of seconds")
    return value


def build_parser() -> argparse.ArgumentParser:
    """The command-line parser."""
    parser = argparse.ArgumentParser(
        prog="device_forward", description="Relay the device listener's TCP port, bytes only."
    )
    parser.add_argument("--listen", required=True, type=_address, help="<host>:<port> to accept on")
    parser.add_argument("--to", required=True, type=_address, help="<host>:<port> to relay to")
    parser.add_argument("--max-connections", type=_positive_int, default=DEFAULT_MAX_CONNECTIONS)
    parser.add_argument("--idle-timeout-s", type=_positive_seconds, default=DEFAULT_IDLE_TIMEOUT_S)
    return parser


async def _serve(args: argparse.Namespace) -> None:
    forwarder = Forwarder(
        args.to, max_connections=args.max_connections, idle_timeout_s=args.idle_timeout_s
    )
    server = await forwarder.start(*args.listen)
    logger.info("device_forward listening on %s:%d, relaying to %s:%d", *args.listen, *args.to)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    # As PID 1 in its container the process gets no default SIGTERM action, so `docker stop`
    # would wait out its grace period without this handler.
    for signum in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(signum, stop.set)
    async with server:
        await stop.wait()


def main(argv: list[str] | None = None) -> int:
    """Run the relay until SIGTERM or SIGINT."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr
    )
    asyncio.run(_serve(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
