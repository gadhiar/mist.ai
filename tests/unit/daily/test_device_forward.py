"""Tests for `scripts/daily/device_forward.py`, the LAN relay in front of the device listener.

Each test runs the forwarder and a stand-in upstream on loopback sockets. The idle timeout reads
an injected clock that moves only when a test moves it, so no test waits two minutes.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from scripts.daily import device_forward

PAYLOAD = b"PAYLOAD-MARKER-" + bytes(range(256)) * 64
WAIT_S = 5.0


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Upstream:
    """A loopback server that answers each chunk with `reply:` and the chunk."""

    def __init__(self) -> None:
        self.received = bytearray()
        self.connections = 0
        self.closed = 0
        self.server: asyncio.AbstractServer | None = None

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        try:
            while data := await reader.read(65536):
                self.received += data
                writer.write(b"reply:" + data)
                await writer.drain()
        except ConnectionError:
            pass
        finally:
            self.closed += 1
            writer.close()

    async def start(self) -> int:
        self.server = await asyncio.start_server(self.handle, "127.0.0.1", 0)
        return self.server.sockets[0].getsockname()[1]


async def start_pair(
    **options,
) -> tuple[Upstream, device_forward.Forwarder, int, asyncio.AbstractServer]:
    upstream = Upstream()
    upstream_port = await upstream.start()
    forwarder = device_forward.Forwarder(("127.0.0.1", upstream_port), poll_s=0.01, **options)
    server = await forwarder.start("127.0.0.1", 0)
    return upstream, forwarder, server.sockets[0].getsockname()[1], server


async def stop(*servers: asyncio.AbstractServer | None) -> None:
    for server in servers:
        if server is not None:
            server.close()
            await server.wait_closed()


async def connect(port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    return await asyncio.open_connection("127.0.0.1", port)


async def exchange(reader, writer, data: bytes) -> bytes:
    writer.write(data)
    await writer.drain()
    expected = len(b"reply:") + len(data)
    received = bytearray()
    while len(received) < expected:
        chunk = await asyncio.wait_for(reader.read(65536), WAIT_S)
        assert chunk, "the relay closed before the reply arrived"
        received += chunk
    return bytes(received)


async def at_eof(reader: asyncio.StreamReader) -> bool:
    try:
        return await asyncio.wait_for(reader.read(1), WAIT_S) == b""
    except ConnectionError:
        return True


async def until(condition, timeout: float = WAIT_S) -> None:
    async def poll():
        while not condition():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(poll(), timeout)


def test_relays_bytes_both_ways_unchanged():
    async def scenario():
        upstream, _forwarder, port, server = await start_pair()
        reader, writer = await connect(port)

        reply = await exchange(reader, writer, PAYLOAD)

        assert bytes(upstream.received) == PAYLOAD
        writer.close()
        await stop(server, upstream.server)
        return reply

    # The upstream answers in one or more chunks, each prefixed; the bytes arrive unchanged.
    reply = asyncio.run(scenario())
    assert reply.startswith(b"reply:PAYLOAD-MARKER-")
    assert reply.replace(b"reply:", b"") == PAYLOAD


def test_refuses_a_ninth_connection_and_accepts_again_once_one_closes():
    async def scenario():
        upstream, forwarder, port, server = await start_pair(max_connections=8)
        clients = [await connect(port) for _ in range(8)]
        for reader, writer in clients:
            assert await exchange(reader, writer, b"hi") == b"reply:hi"

        ninth_reader, _ninth_writer = await connect(port)
        assert await at_eof(ninth_reader)
        assert upstream.connections == 8

        clients[0][1].close()
        await until(lambda: forwarder.active == 7)
        reader, writer = await connect(port)
        assert await exchange(reader, writer, b"again") == b"reply:again"
        assert upstream.connections == 9
        await stop(server, upstream.server)

    asyncio.run(scenario())


def test_closes_a_connection_idle_for_the_timeout_on_the_injected_clock():
    clock = FakeClock()

    async def scenario():
        upstream, forwarder, port, server = await start_pair(idle_timeout_s=120.0, clock=clock)
        reader, writer = await connect(port)
        assert await exchange(reader, writer, b"one") == b"reply:one"

        clock.now += 119.0
        await asyncio.sleep(0.05)
        # Traffic resets the idle time.
        assert await exchange(reader, writer, b"two") == b"reply:two"
        clock.now += 119.0
        await asyncio.sleep(0.05)
        assert forwarder.active == 1

        clock.now += 2.0
        assert await at_eof(reader)
        await until(lambda: upstream.closed == 1 and forwarder.active == 0)
        await stop(server, upstream.server)

    asyncio.run(scenario())


def test_logs_one_line_per_connection_with_no_payload(caplog):
    caplog.set_level(logging.INFO, logger=device_forward.LOGGER_NAME)

    async def scenario() -> int:
        upstream, forwarder, port, server = await start_pair(max_connections=1)
        reader, writer = await connect(port)
        reply = await exchange(reader, writer, PAYLOAD)
        refused_reader, _ = await connect(port)
        assert await at_eof(refused_reader)
        writer.close()
        await until(lambda: forwarder.active == 0)
        await stop(server, upstream.server)
        return len(reply)

    reply_length = asyncio.run(scenario())

    records = [r for r in caplog.records if r.name == device_forward.LOGGER_NAME]
    messages = [r.getMessage() for r in records]
    assert len(messages) == 2, messages
    refused = next(m for m in messages if "outcome=refused" in m)
    relayed = next(m for m in messages if "outcome=closed" in m)
    assert "active=1" in refused
    assert f"bytes_in={len(PAYLOAD)} " in relayed
    assert f"bytes_out={reply_length} " in relayed
    for message in messages:
        assert "PAYLOAD" not in message and "reply:" not in message
        assert "peer=127.0.0.1:" in message


def test_logs_and_closes_when_the_upstream_refuses(caplog):
    caplog.set_level(logging.INFO, logger=device_forward.LOGGER_NAME)

    async def scenario():
        upstream = Upstream()
        port = await upstream.start()
        await stop(upstream.server)  # the port now refuses connections
        forwarder = device_forward.Forwarder(("127.0.0.1", port), poll_s=0.01)
        server = await forwarder.start("127.0.0.1", 0)
        reader, _writer = await connect(server.sockets[0].getsockname()[1])
        assert await at_eof(reader)
        await until(lambda: forwarder.active == 0)
        await stop(server)

    asyncio.run(scenario())

    messages = [r.getMessage() for r in caplog.records if r.name == device_forward.LOGGER_NAME]
    assert len(messages) == 1
    assert "outcome=upstream_unreachable" in messages[0]


# --- The command line ------------------------------------------------------------------------


def test_parses_the_compose_command_with_its_defaults():
    args = device_forward.build_parser().parse_args(
        ["--listen", "0.0.0.0:8443", "--to", "mist-backend:8443"]
    )

    assert args.listen == ("0.0.0.0", 8443)
    assert args.to == ("mist-backend", 8443)
    assert args.max_connections == 8
    assert args.idle_timeout_s == 120.0


@pytest.mark.parametrize("value", ["8443", "host:", "host:0", "host:65536", "host:port", ":8443"])
def test_refuses_an_address_that_is_not_host_and_port(value):
    with pytest.raises(SystemExit):
        device_forward.build_parser().parse_args(["--listen", value, "--to", "mist-backend:8443"])
