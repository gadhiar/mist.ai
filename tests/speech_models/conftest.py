"""The host tier's shared fixtures: the voice settings, the generated fixtures, the records.

Nothing here imports an engine library, so the tier collects in an image without them, and each
test skips naming what is missing. The records are measurements the tier reports without judging
them (decode and first-chunk times, VRAM, the noise finals); they print as JSON lines in a
`speech_models records` section at the end of the run, so H1 can keep them with its results.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest

from backend.realtime.budget import VoiceSettings, load_voice_settings
from scripts.voice_fixtures.generate_fixtures import (
    GENERATED_DIR,
    MANIFEST_NAME,
    FixtureEntry,
    FixtureError,
    Manifest,
    read_manifest,
    read_wav,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATED = REPO_ROOT / GENERATED_DIR

RECORDS = pytest.StashKey[list[dict[str, object]]]()

Recorder = Callable[..., None]


@pytest.fixture(scope="session")
def voice_settings() -> VoiceSettings:
    """The voice settings the backend would read from this environment.

    A malformed variable fails rather than skips: it is a broken host, not a missing engine.
    """
    try:
        return load_voice_settings(os.environ)
    except ValueError as exc:
        pytest.fail(f"the voice settings do not load: {exc}")


@pytest.fixture(scope="session")
def manifest() -> Callable[[], Manifest]:
    """Read the generated fixtures' manifest, skipping naming its path when it was never made.

    A callable, so a test calls it after its own engine checks, and a host without the engine
    skips naming the engine rather than the manifest.
    """
    cache: list[Manifest] = []

    def load() -> Manifest:
        if cache:
            return cache[0]
        path = GENERATED / MANIFEST_NAME
        if not path.is_file():
            pytest.skip(
                f"no generated fixtures: {path.relative_to(REPO_ROOT).as_posix()} is missing; "
                "run scripts/voice_fixtures/generate_fixtures.py --voice-root <dir> "
                "--voice-id <id>"
            )
        try:
            cache.append(read_manifest(GENERATED))
        except FixtureError as exc:
            pytest.fail(f"the generated manifest does not read: {exc}")
        return cache[0]

    return load


@pytest.fixture(scope="session")
def fixture_audio() -> Callable[[FixtureEntry], np.ndarray]:
    """Read one fixture's audio, float32 at 16 kHz."""

    def read(entry: FixtureEntry) -> np.ndarray:
        return read_wav(GENERATED / entry.file)

    return read


@pytest.fixture
def speech_record(request: pytest.FixtureRequest) -> Iterator[Recorder]:
    """Record one measurement under a name; the test's id is added to it."""
    records = request.config.stash.setdefault(RECORDS, [])

    def record(name: str, **fields: object) -> None:
        records.append({"test": request.node.nodeid, "record": name, **fields})

    yield record


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter, config: pytest.Config):
    records = config.stash.get(RECORDS, [])
    if not records:
        return
    terminalreporter.section("speech_models records")
    for record in records:
        terminalreporter.write_line(json.dumps(record, sort_keys=True, default=str))
