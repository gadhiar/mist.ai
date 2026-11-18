"""Tests for the guided wake recorder, `scripts.wake.record`.

The microphone, the speakers, the keyboard and the clock are fakes passed in as a `record.Host`,
so no audio package is touched and no sound is made. Every capture the fake returns is distinct,
because `manifest.py` refuses two entries holding audio with one sha256. The keyboard is a
script of keys: `enter` and `timeout` answer a wait for Enter, and `""`, `r`, `p` and `q` answer
the keep, redo, play and stop question after each clip.
"""

import json
import subprocess
import sys
import wave
from collections.abc import Iterator, Sequence
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from scripts.wake import manifest as wake_manifest
from scripts.wake import record

REPO = Path(__file__).resolve().parents[3]
RATE = 16000


# -- Fakes -------------------------------------------------------------------------------------


class FakeCapture:
    def __init__(self, pcm: bytes) -> None:
        self._pcm = pcm

    def stop(self) -> bytes:
        return self._pcm


class FakeStream:
    def __init__(self, audio: "FakeAudio", rate: int) -> None:
        self._audio = audio
        self._rate = rate
        self.closed = False

    def read(self, frames: int) -> bytes:
        audio = self._audio
        if audio.interrupt_after_s is not None and audio.streamed >= audio.interrupt_after_s * (
            self._rate
        ):
            raise KeyboardInterrupt
        audio.streamed += frames
        audio.stream_reads += 1
        # Each read holds one distinct constant, loud enough to pass the silence check.
        level = 0 if audio.silent_stream else 1000 + audio.stream_reads % 20000
        return level.to_bytes(2, "little") * frames

    def close(self) -> None:
        self.closed = True


class FakeAudio:
    """A microphone at `rate` Hz that may or may not open at 16 kHz."""

    def __init__(
        self,
        *,
        rate: int = RATE,
        opens_16k: bool = True,
        clip_ms: int = 1000,
        interrupt_after_s: int | None = None,
        has_input: bool = True,
        silent_captures: int = 0,
        silent_stream: bool = False,
    ) -> None:
        self.rate = rate
        self.opens_16k = opens_16k
        self.clip_ms = clip_ms
        self.interrupt_after_s = interrupt_after_s
        self.has_input = has_input
        self.silent_captures = silent_captures
        self.silent_stream = silent_stream
        self.captures: list[bytes] = []
        self.capture_rates: list[int] = []
        self.played: list[tuple[bytes, int]] = []
        self.on_capture: list = []
        self.streamed = 0
        self.stream_reads = 0
        self.streams: list[FakeStream] = []

    def input_devices(self) -> list[record.InputDevice]:
        if not self.has_input:
            return []
        return [
            record.InputDevice(
                index=3,
                name="Headset Microphone",
                host_api="Windows WASAPI",
                default_rate=self.rate,
            )
        ]

    def capture_rate(self, device: int | None) -> int:
        if not self.has_input:
            raise record.RefusedError("no input device")
        return RATE if self.opens_16k else self.rate

    def start_capture(self, device: int | None, rate: int) -> FakeCapture:
        for hook in self.on_capture:
            hook()
        frames = rate * self.clip_ms // 1000
        step = len(self.captures) + 1
        if step <= self.silent_captures:
            # A muted headset: low noise, peak 100 (about -50 dBFS).
            pcm = ((np.arange(frames) * step) % 101).astype("<i2").tobytes()
        else:
            pcm = ((np.arange(frames) * step) % 20000).astype("<i2").tobytes()
        self.captures.append(pcm)
        self.capture_rates.append(rate)
        return FakeCapture(pcm)

    def open_stream(self, device: int | None, rate: int) -> FakeStream:
        if not self.has_input:
            raise record.RefusedError("no input device")
        stream = FakeStream(self, rate)
        self.streams.append(stream)
        return stream

    def play(self, pcm: bytes, rate: int) -> None:
        self.played.append((pcm, rate))


class FakeKeyboard:
    def __init__(self, keys: Sequence[str]) -> None:
        self._keys: Iterator[str] = iter(keys)
        self.timeouts: list[float | None] = []

    def _next(self) -> str:
        try:
            return next(self._keys)
        except StopIteration:
            raise AssertionError("the recorder asked for more keys than the script holds") from None

    def wait_enter(self, timeout: float | None) -> bool:
        self.timeouts.append(timeout)
        key = self._next()
        assert key in ("enter", "timeout"), f"expected a wait for Enter, the script has {key!r}"
        return key == "enter"

    def choose(self, prompt: str) -> str:
        key = self._next()
        assert key not in ("enter", "timeout"), f"expected a choice, the script has {key!r}"
        return key


class FakeClock:
    def __init__(self) -> None:
        self._now = datetime(2026, 11, 17, 21, 5)

    def __call__(self) -> datetime:
        current = self._now
        self._now += timedelta(minutes=1)
        return current


def keep(count: int) -> list[str]:
    """Record and keep `count` clips: Enter to start, Enter to stop, Enter to keep."""
    return ["enter", "enter", ""] * count


def host(audio: FakeAudio | None = None, keys: Sequence[str] = ()) -> record.Host:
    return record.Host(audio=audio or FakeAudio(), keyboard=FakeKeyboard(keys), now=FakeClock())


def draft(root: Path) -> dict:
    return json.loads((root / "manifest.draft.json").read_text(encoding="utf-8"))


def wav_format(path: Path) -> tuple[int, int, int, int]:
    with wave.open(str(path), "rb") as handle:
        return (
            handle.getframerate(),
            handle.getnchannels(),
            handle.getsampwidth(),
            handle.getnframes(),
        )


def run(argv: Sequence[str], machine: record.Host) -> int:
    return record.main(list(argv), host=machine)


def record_blocks(root: Path, count: int, audio: FakeAudio | None = None) -> FakeAudio:
    """Record `count` full blocks through one fake microphone, so no two captures match."""
    audio = audio or FakeAudio()
    for _ in range(count):
        assert run(["session", "--root", str(root)], host(audio, keep(BLOCK))) == 0
    return audio


BLOCK = 24
"""A block's prompts: 18 wake prompts and 6 sound-alike sentences."""
NAMES = {"mist", "hey mist"}
"""The phrase keys `eval_wake.py` measures hit lag on: a model name said alone."""


# -- session -----------------------------------------------------------------------------------


def test_session_writes_24_16k_clips_and_entries_the_manifest_accepts(tmp_path, capsys):
    root = tmp_path / "wake"

    assert run(["session", "--root", str(root)], host(keys=keep(BLOCK))) == 0

    data = draft(root)
    assert data["root"] == str(root.resolve())
    entries = data["entries"]
    assert len(entries) == BLOCK
    assert [item["label"] for item in entries].count("positive") == 18
    assert [item["label"] for item in entries].count("homophone") == 6
    assert [item["label"] for item in entries[:8]] == ["positive"] * 3 + ["homophone"] + [
        "positive"
    ] * 3 + ["homophone"]
    alone = [item for item in entries if wake_manifest.phrase_key(item["phrase"]) in NAMES]
    assert len(alone) == 6
    assert all(item["label"] == "positive" for item in alone)
    assert {item["session"] for item in entries} == {"20261117-2105-b1"}
    for position, item in enumerate(entries, 1):
        assert set(item) == {"file", "label", "phrase", "condition", "duration_ms", "session"}
        assert item["file"] == f"recordings/20261117-2105-b1/{item['label']}-{position:03d}.wav"
        assert item["condition"] == "headset"
        assert item["duration_ms"] == 1000
        assert wav_format(root / item["file"]) == (RATE, 1, 2, RATE)

    # The manifest's own check accepts every entry once each is given a split.
    checked = tmp_path / "checked.json"
    checked.write_text(
        json.dumps({"root": data["root"], "entries": [{**e, "split": "train"} for e in entries]}),
        encoding="utf-8",
    )
    assert len(wake_manifest.load_manifest(checked).split("train")) == BLOCK

    out = capsys.readouterr().out
    first = record.block_prompts(1)[0]
    assert f'Block 1 of 5, clip 1 of 24. Say: "{first.phrase}". Style: {first.style}' in out


def test_block_prompts_add_the_name_alone_and_rotate_phrasings_and_styles():
    lead_in = [style for style in record.STYLES if "4 s" in style]
    assert len(lead_in) == 1
    for block in range(1, 6):
        prompts = record.block_prompts(block)
        wake = [item for item in prompts if item.label == "positive"]
        assert len(prompts) == BLOCK
        assert len(wake) == 18
        alone = [item for item in wake if item.phrase in record.NAME_ALONE]
        assert sorted(item.phrase for item in alone) == ["hey mist"] * 3 + ["mist"] * 3
        assert {wake_manifest.phrase_key(phrase) for phrase in record.NAME_ALONE} == NAMES
        # Other talk before the name would put speech after it: never on a name said alone.
        assert not any(item.style in lead_in for item in alone)
        sentences = [item for item in wake if item not in alone]
        for family in record.WAKE_PHRASES:
            assert sum(item.phrase in family for item in sentences) == 4
        assert {item.style for item in sentences} == set(record.STYLES)
        assert len({item.style for item in alone}) > 1


def test_a_quiet_clip_is_not_offered_and_is_recorded_again(tmp_path, capsys):
    root = tmp_path / "wake"
    audio = FakeAudio(silent_captures=1)
    # The quiet capture asks no keep question: Enter, Enter, then the retake.
    keys = ["enter", "enter", *keep(1), "enter", "enter", "q"]

    assert run(["session", "--root", str(root)], host(audio, keys)) == 0

    assert "too quiet" in capsys.readouterr().out.lower()
    entries = draft(root)["entries"]
    assert len(entries) == 1
    with wave.open(str(root / entries[0]["file"]), "rb") as handle:
        assert handle.readframes(handle.getnframes()) == audio.captures[1]


def test_an_entry_another_process_wrote_survives_the_next_keep(tmp_path):
    root = tmp_path / "wake"
    audio = FakeAudio()
    foreign = {
        "file": "recordings/20261117-2104-p1/media-001.wav",
        "label": "media",
        "phrase": "",
        "condition": "headset",
        "duration_ms": 60000,
        "session": "20261117-2104-p1",
    }

    def passive_in_another_terminal() -> None:
        if len(audio.captures) == 1:
            data = draft(root)
            data["entries"].append(foreign)
            (root / "manifest.draft.json").write_text(json.dumps(data), encoding="utf-8")

    audio.on_capture.append(passive_in_another_terminal)

    assert (
        run(["session", "--root", str(root)], host(audio, [*keep(2), "enter", "enter", "q"])) == 0
    )

    entries = draft(root)["entries"]
    assert len(entries) == 3
    assert foreign in entries
    assert [item["label"] for item in entries if item != foreign] == ["positive", "positive"]
    assert not (root / record.LOCK_NAME).exists()


def test_a_lock_left_by_a_killed_process_is_cleared(tmp_path, monkeypatch):
    root = tmp_path / "wake"
    root.mkdir()
    (root / record.LOCK_NAME).write_text("", encoding="utf-8")
    monkeypatch.setattr(record, "LOCK_STALE_SECONDS", 0.05)

    assert run(["session", "--root", str(root)], host(keys=[*keep(1), "enter", "enter", "q"])) == 0

    assert len(draft(root)["entries"]) == 1
    assert not (root / record.LOCK_NAME).exists()


def test_redo_replaces_the_clip_in_hand_and_the_draft_is_rewritten_after_each_keep(tmp_path):
    root = tmp_path / "wake"
    audio = FakeAudio()
    seen: list[int] = []
    audio.on_capture.append(
        lambda: seen.append(
            len(draft(root)["entries"]) if (root / "manifest.draft.json").exists() else 0
        )
    )
    keys = ["enter", "enter", "r", *keep(3), "enter", "enter", "q"]

    assert run(["session", "--root", str(root)], host(audio, keys)) == 0

    entries = draft(root)["entries"]
    assert len(entries) == 3
    # The first capture was redone, so the first kept clip is the second capture.
    with wave.open(str(root / entries[0]["file"]), "rb") as handle:
        assert handle.readframes(handle.getnframes()) == audio.captures[1]
    # The draft held 0, 0, 1, 2 and 3 entries as captures 1 to 5 began.
    assert seen == [0, 0, 1, 2, 3]
    # `q` discarded the clip in hand: three files, not four.
    assert len(list((root / "recordings").rglob("*.wav"))) == 3


def test_play_back_plays_the_16k_clip_then_asks_again(tmp_path):
    root = tmp_path / "wake"
    audio = FakeAudio()
    keys = ["enter", "enter", "p", "", "enter", "enter", "q"]

    assert run(["session", "--root", str(root)], host(audio, keys)) == 0

    assert audio.played == [(audio.captures[0], RATE)]
    assert len(draft(root)["entries"]) == 1


def test_a_48k_device_is_resampled_to_16k_with_the_right_duration(tmp_path):
    root = tmp_path / "wake"
    audio = FakeAudio(rate=48000, opens_16k=False, clip_ms=1500)

    assert (
        run(["session", "--root", str(root)], host(audio, [*keep(1), "enter", "enter", "q"])) == 0
    )

    assert audio.capture_rates[0] == 48000
    item = draft(root)["entries"][0]
    rate, channels, width, frames = wav_format(root / item["file"])
    assert (rate, channels, width) == (RATE, 1, 2)
    assert frames == 24000
    assert abs(item["duration_ms"] - frames * 1000 / RATE) <= wake_manifest.DURATION_TOLERANCE_MS
    assert item["duration_ms"] == 1500


def test_a_clip_is_cut_at_eight_seconds(tmp_path):
    root = tmp_path / "wake"
    audio = FakeAudio(clip_ms=9500)
    keyboard_keys = ["enter", "timeout", "", "enter", "enter", "q"]
    machine = host(audio, keyboard_keys)

    assert run(["session", "--root", str(root)], machine) == 0

    assert machine.keyboard.timeouts[:2] == [None, record.MAX_CLIP_SECONDS]
    item = draft(root)["entries"][0]
    assert wav_format(root / item["file"])[3] == record.MAX_CLIP_SECONDS * RATE
    assert item["duration_ms"] == record.MAX_CLIP_SECONDS * 1000


def test_sound_alike_sentences_never_repeat_and_a_sixth_block_is_refused(tmp_path, capsys):
    root = tmp_path / "wake"
    record_blocks(root, 5)

    entries = draft(root)["entries"]
    sessions = sorted({item["session"] for item in entries})
    assert [session[-3:] for session in sessions] == ["-b1", "-b2", "-b3", "-b4", "-b5"]
    sentences = [item["phrase"] for item in entries if item["label"] == "homophone"]
    assert len(sentences) == 30
    assert len({wake_manifest.phrase_key(text) for text in sentences}) == 30
    assert sum("missed" in text.split() for text in sentences) == 15
    assert sum("mist" in text.split() for text in sentences) == 15
    capsys.readouterr()

    assert run(["session", "--root", str(root)], host(keys=keep(1))) == 2

    assert "all 5 blocks are recorded" in capsys.readouterr().err
    assert len(draft(root)["entries"]) == 5 * BLOCK


def test_a_block_stopped_before_its_first_keep_is_offered_again(tmp_path):
    root = tmp_path / "wake"
    assert run(["session", "--root", str(root)], host(keys=["enter", "enter", "q"])) == 0
    assert not (root / "manifest.draft.json").exists()

    assert run(["session", "--root", str(root)], host(keys=[*keep(1), "enter", "enter", "q"])) == 0

    assert draft(root)["entries"][0]["session"].endswith("-b1")


# -- passive -----------------------------------------------------------------------------------


def test_passive_writes_ten_minute_chunks_as_separate_media_sessions(tmp_path):
    root = tmp_path / "wake"
    audio = FakeAudio()

    assert run(["passive", "--root", str(root), "--minutes", "21"], host(audio)) == 0

    entries = draft(root)["entries"]
    assert record.CHUNK_MINUTES == 10
    assert [item["duration_ms"] for item in entries] == [600_000, 600_000, 60_000]
    assert [item["session"] for item in entries] == [
        "20261117-2105-p1",
        "20261117-2106-p2",
        "20261117-2107-p3",
    ]
    for item in entries:
        assert item["label"] == "media"
        assert item["condition"] == "headset"
        assert item["phrase"] == record.MEDIA_PHRASE
        assert item["file"] == f"recordings/{item['session']}/media-001.wav"
        rate, channels, width, frames = wav_format(root / item["file"])
        assert (rate, channels, width) == (RATE, 1, 2)
        assert frames == item["duration_ms"] * RATE // 1000
    assert audio.streams[0].closed


def test_passive_keeps_the_open_chunk_on_interrupt_and_numbers_on(tmp_path):
    root = tmp_path / "wake"
    audio = FakeAudio(interrupt_after_s=12 * 60)

    assert run(["passive", "--root", str(root), "--minutes", "60"], host(audio)) == 0

    entries = draft(root)["entries"]
    assert [item["duration_ms"] for item in entries] == [600_000, 120_000]
    assert audio.streams[0].closed

    assert run(["passive", "--root", str(root), "--minutes", "1"], host()) == 0

    assert draft(root)["entries"][-1]["session"].endswith("-p3")


def test_a_silent_passive_chunk_is_not_written(tmp_path, capsys):
    root = tmp_path / "wake"

    assert (
        run(["passive", "--root", str(root), "--minutes", "1"], host(FakeAudio(silent_stream=True)))
        == 0
    )

    assert "heard nothing" in capsys.readouterr().out
    assert not (root / "manifest.draft.json").exists()
    assert not list(root.rglob("*.wav"))


# -- status and finish -------------------------------------------------------------------------


def test_status_counts_blocks_labels_and_passive_minutes(tmp_path, capsys):
    root = tmp_path / "wake"
    record_blocks(root, 2)
    assert run(["passive", "--root", str(root), "--minutes", "2"], host()) == 0
    capsys.readouterr()

    assert run(["status", "--root", str(root)], host()) == 0

    out = capsys.readouterr().out
    assert "blocks: 2 of 5" in out
    assert "clips: positive 36, homophone 12, media 1" in out
    assert "passive: 2.0 of 60 minutes (optional)" in out
    assert "next: block 3" in out


def test_status_of_an_empty_root_names_block_one(tmp_path, capsys):
    assert run(["status", "--root", str(tmp_path / "none")], host()) == 0

    out = capsys.readouterr().out
    assert "blocks: 0 of 5" in out
    assert "next: block 1" in out


def test_finish_writes_a_manifest_with_positives_in_every_split(tmp_path, capsys):
    root = tmp_path / "wake"
    audio = record_blocks(root, 5)
    for _ in range(2):
        assert run(["passive", "--root", str(root), "--minutes", "1"], host(audio)) == 0
    capsys.readouterr()

    assert run(["finish", "--root", str(root), "--seed", "7"], host()) == 0

    loaded = wake_manifest.load_manifest(root / "manifest.json")
    assert len(loaded.entries) == 5 * BLOCK + 2
    for split in wake_manifest.SPLITS:
        assert any(item.label == "positive" for item in loaded.split(split))
    assert any(
        item.label == "positive" and wake_manifest.phrase_key(item.phrase) in NAMES
        for item in loaded.split("tune")
    )
    out = capsys.readouterr().out
    assert "seed 7" in out
    assert "train: positive" in out


def test_finish_writes_nothing_when_a_split_has_no_positives(tmp_path, capsys):
    root = tmp_path / "wake"
    record_blocks(root, 1)
    before = sorted(path.relative_to(root) for path in root.rglob("*"))
    capsys.readouterr()

    assert run(["finish", "--root", str(root), "--seed", "1"], host()) == 1

    assert not (root / "manifest.json").exists()
    assert sorted(path.relative_to(root) for path in root.rglob("*")) == before
    output = capsys.readouterr()
    assert "another seed" in output.out + output.err


def test_finish_refuses_an_existing_manifest(tmp_path, capsys):
    root = tmp_path / "wake"
    record_blocks(root, 1)
    (root / "manifest.json").write_text("{}", encoding="utf-8")

    assert run(["finish", "--root", str(root), "--seed", "1"], host()) == 2

    assert "already exists" in capsys.readouterr().err
    assert (root / "manifest.json").read_text(encoding="utf-8") == "{}"


def test_finish_without_a_draft_is_refused(tmp_path):
    assert run(["finish", "--root", str(tmp_path), "--seed", "1"], host()) == 2


@pytest.mark.parametrize("command", ["session", "passive", "status", "finish"])
def test_a_missing_root_exits_2(command):
    with pytest.raises(SystemExit) as info:
        record.main([command], host=host())
    assert info.value.code == 2


# -- devices and refusals ----------------------------------------------------------------------


def test_devices_lists_index_host_api_and_default_rate(capsys):
    assert run(["devices"], host(FakeAudio(rate=48000))) == 0

    assert "3: Headset Microphone [Windows WASAPI] (48000 Hz)" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [["devices"], ["session", "--root", "ROOT"], ["passive", "--root", "ROOT", "--minutes", "1"]],
)
def test_no_input_device_exits_2(tmp_path, argv):
    argv = [str(tmp_path) if item == "ROOT" else item for item in argv]

    assert run(argv, host(FakeAudio(has_input=False))) == 2


def test_the_guide_names_a_terminal_that_passes_key_presses():
    assert "PowerShell" in record.__doc__
    assert "Git Bash" in record.__doc__


def test_importing_the_recorder_imports_no_audio_package():
    code = (
        "import sys, scripts.wake.record; "
        "print(sorted(m for m in ('sounddevice', 'numpy', 'scipy') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
