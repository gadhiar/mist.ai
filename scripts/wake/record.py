"""A guided recorder for MIST's wake recordings: one sitting on the headset.

The wake models learn and are measured on the speaker's own voice, in the manifest
`scripts/wake/manifest.py` checks. This recorder says what to say and how, records it from the
headset microphone, and writes each kept clip as 16 kHz mono PCM16 WAV with its manifest entry,
so no file is named and no entry is typed by hand.

Host setup. The recorder needs the microphone, so it runs on the Windows host, never in a
container. Make a virtual environment outside the repository with `sounddevice` (which ships
PortAudio in its wheel), `numpy` and `scipy`, and run it from the repository root:

    python -m venv <venv>
    <venv>/Scripts/pip install sounddevice numpy scipy
    <venv>/Scripts/python -m scripts.wake.record devices

`devices` lists the input devices with their index and default rate; pass the headset's index as
`--device N`, or leave it out to use the system default input.

The sitting. `--root` is a directory outside the repository; there is no default, so recordings
never land in this public repository. Run five blocks, with a short break between them:

    python -m scripts.wake.record session --root <dir> --device N

A block is 12 wake prompts and 6 sound-alike sentences, interleaved. Each prompt names the
phrase and the style (normal, quiet, fast, tired, or a few words of other talk first). Enter
starts a clip and Enter stops it, at 8 s at most; then Enter keeps it, `r` records it again, `p`
plays it back and `q` stops the block, keeping the clips already kept. A block stopped before its
first keep is offered again; a block stopped later counts as done, and its remaining prompts are
skipped. After each block a line suggests a change of position for the next.

Optionally, record an hour of ordinary headset use (calls, videos, talk not meant for MIST) as
`media`, the recordings a wake model must not accept:

    python -m scripts.wake.record passive --root <dir> --minutes 60 --device N

It writes one file per 10 minutes, each its own session; Ctrl+C closes and keeps the open file.
`status --root <dir>` shows the blocks done, the clips per label and the passive minutes. Last:

    python -m scripts.wake.record finish --root <dir> --seed N

which runs `manifest assign` on the draft into `<dir>/manifest.json` and then `manifest check`,
printing the seed, each split's shares and each split's counts per label. A draft
`manifest.py` refuses exits 1, with its message naming the entries at fault.

Files under `--root`: `recordings/<session>/<label>-<nnn>.wav` and `manifest.draft.json`, the
entries without `split`, rewritten after every kept clip, so a closed terminal loses at most the
clip in hand (in `passive`, the open file). Block sessions are `<YYYYMMDD>-<HHMM>-b<n>`, passive
ones `<YYYYMMDD>-<HHMM>-p<n>`.

The recordings are personal audio. No agent listens to, opens, lists or copies them; agents read
only what `status` and `finish` print.

Exit codes: 0 done; 1 an audio or file error, with the message; 2 a refused invocation (no
`--root`, an existing `manifest.json` on `finish`, no input device, a sixth block).
"""

import argparse
import json
import math
import os
import re
import sys
import time
import wave
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from . import manifest as wake_manifest

SAMPLE_RATE = wake_manifest.SAMPLE_RATE
SAMPLE_WIDTH = wake_manifest.SAMPLE_WIDTH
BLOCKS = 5
MAX_CLIP_SECONDS = 8
MIN_CLIP_FRAMES = SAMPLE_RATE // 10
"""A clip shorter than 0.1 s is a missed key press, not a recording; it is taken again."""
CHUNK_MINUTES = 10
PASSIVE_GOAL_MINUTES = 60
CONDITION = "headset"
MEDIA_PHRASE = ""
"""`media` holds no prompted sentence; `manifest.py` accepts an empty phrase."""
DRAFT_NAME = "manifest.draft.json"
MANIFEST_NAME = "manifest.json"
RECORDINGS = "recordings"

WAKE_PHRASES: tuple[tuple[str, str], ...] = (
    ("mist, what's up", "mist, how's it going"),
    ("hey mist, what's the time", "hey mist, turn the lights off"),
    ("so, mist, what do you think", "are you there mist"),
)
"""Three placements of the name (first, after "hey", mid-sentence); each is said 4 times a block."""

STYLES: tuple[str, ...] = (
    "normal",
    "quiet",
    "fast",
    "tired",
    "other talk first (a few words, up to 4 s, then the phrase)",
)

MISSED: tuple[str, ...] = (
    "I missed the bus this morning",
    "she missed the last train home",
    "we missed the start of the film",
    "he missed two calls while he was out",
    "I think I missed a step there",
    "you missed a spot on the window",
    "they missed the deadline by a day",
    "I missed lunch again today",
    "the kick missed the goal by a mile",
    "sorry, I missed what you said",
    "we missed the turn off the highway",
    "she missed her flight to Denver",
    "I missed the meeting on Monday",
    "he never missed a single game",
    "I missed you at the party",
)
FOG: tuple[str, ...] = (
    "the mist rolled in over the hills",
    "a cold mist hung over the lake",
    "we drove slowly through the mist",
    "the morning mist burned off by noon",
    "there's a fine mist on the windscreen",
    "the valley was full of mist at dawn",
    "spray a light mist on the plants",
    "the mist made the street lights glow",
    "a thin mist covered the field",
    "the boat vanished into the mist",
    "the mist was thick along the river",
    "sea mist came in off the bay",
    "the forest looked grey in the mist",
    "the mist lifted after the rain",
    "the hills disappeared behind the mist",
)
"""Each sentence is used in one block only: `manifest.py` refuses one sentence in two splits."""

BLOCK_SETUPS: tuple[str, ...] = (
    "Sit as you usually do at the desk.",
    "Sit back in the chair, further from the desk.",
    "Turn your head to one side as you speak.",
    "Move the microphone a little further from your mouth.",
    "Lean forward and speak as if half paying attention.",
)

_SESSION_NUMBER = re.compile(r"-([bp])(\d+)$")


class RecorderError(Exception):
    """An audio or file error; the command exits 1 with the message."""


class RefusedError(RecorderError):
    """An invocation the recorder will not carry out; the command exits 2 with the message."""


# -- Prompts --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Prompt:
    """One clip to record: its label, the exact text to say, and how to say it."""

    label: str
    phrase: str
    style: str


def block_prompts(block: int) -> tuple[Prompt, ...]:
    """Block `block`'s 18 prompts: a sound-alike sentence after every two wake prompts.

    The wake prompts cycle the three placements of the name, each placement's two phrases
    alternating, and the styles rotate by block so a phrase is not always said the same way.
    Block n takes the n-th three "missed" and three fog sentences, so none repeats.
    """
    if not 1 <= block <= BLOCKS:
        raise ValueError(f"block must be 1 to {BLOCKS}, got {block}")
    first = 3 * (block - 1)
    sentences = [
        text for pair in zip(MISSED[first : first + 3], FOG[first : first + 3]) for text in pair
    ]
    wake = [
        Prompt(
            "positive",
            WAKE_PHRASES[index % 3][(index // 3) % 2],
            STYLES[(index + block - 1) % len(STYLES)],
        )
        for index in range(12)
    ]
    prompts: list[Prompt] = []
    for position in range(18):
        if position % 3 == 2:
            prompts.append(Prompt("homophone", sentences[position // 3], "normal"))
        else:
            prompts.append(wake[position - position // 3])
    return tuple(prompts)


# -- The host: microphone, speakers, keyboard and clock -------------------------------------------


@dataclass(frozen=True, slots=True)
class InputDevice:
    """An input device as `devices` lists it."""

    index: int
    name: str
    default_rate: int


class Capture(Protocol):
    """A clip being recorded."""

    def stop(self) -> bytes:
        """Stop recording and return the mono PCM16 samples at the capture rate."""
        ...


class Stream(Protocol):
    """A continuous recording read in blocks."""

    def read(self, frames: int) -> bytes:
        """Block until `frames` mono PCM16 samples are recorded, and return them."""
        ...

    def close(self) -> None:
        """Stop recording and release the device."""
        ...


class AudioBackend(Protocol):
    """The microphone and speakers."""

    def input_devices(self) -> list[InputDevice]:
        """Every device with an input channel."""
        ...

    def capture_rate(self, device: int | None) -> int:
        """16000 when the device opens at 16 kHz, else its default rate.

        Raises:
            RefusedError: There is no such input device, or no default one.
        """
        ...

    def start_capture(self, device: int | None, rate: int) -> Capture:
        """Start recording one clip, mono PCM16 at `rate`."""
        ...

    def open_stream(self, device: int | None, rate: int) -> Stream:
        """Start a continuous recording, mono PCM16 at `rate`."""
        ...

    def play(self, pcm: bytes, rate: int) -> None:
        """Play mono PCM16 at `rate` on the default output, returning when it has played."""
        ...


class Keyboard(Protocol):
    """The keys the guided session reads."""

    def wait_enter(self, timeout: float | None) -> bool:
        """Wait for Enter, at most `timeout` seconds; whether Enter was pressed."""
        ...

    def choose(self, prompt: str) -> str:
        """Show `prompt` and return the line typed."""
        ...


@dataclass(frozen=True, slots=True)
class Host:
    """What the commands use outside the process; tests pass fakes for all three."""

    audio: AudioBackend
    keyboard: Keyboard
    now: Callable[[], datetime]


class SoundDeviceAudio:
    """The host microphone and speakers through `sounddevice` (PortAudio)."""

    def __init__(self) -> None:
        # Imported here, so the module imports and its tests run where it is not installed.
        import sounddevice

        self._sd = sounddevice

    def input_devices(self) -> list[InputDevice]:
        """Every device with an input channel, by PortAudio index."""
        try:
            found = self._sd.query_devices()
        except self._sd.PortAudioError as exc:
            raise RecorderError(f"could not list the audio devices: {exc}") from exc
        return [
            InputDevice(index, str(info["name"]), int(info["default_samplerate"]))
            for index, info in enumerate(found)
            if info["max_input_channels"] > 0
        ]

    def capture_rate(self, device: int | None) -> int:
        """16000 when the device opens at 16 kHz, else its default rate, resampled later."""
        try:
            info = self._sd.query_devices(device, "input")
        except (ValueError, self._sd.PortAudioError) as exc:
            raise RefusedError(f"no input device: {exc}") from exc
        try:
            self._sd.check_input_settings(
                device=device, channels=1, dtype="int16", samplerate=SAMPLE_RATE
            )
        except (ValueError, self._sd.PortAudioError):
            return int(info["default_samplerate"])
        return SAMPLE_RATE

    def start_capture(self, device: int | None, rate: int) -> "_SoundDeviceCapture":
        """Start a callback stream that collects every block until `stop`."""
        blocks: list[bytes] = []

        def collect(data: bytes, frames: int, when: object, status: object) -> None:
            blocks.append(bytes(data))

        try:
            stream = self._sd.RawInputStream(
                samplerate=rate, device=device, channels=1, dtype="int16", callback=collect
            )
            stream.start()
        except self._sd.PortAudioError as exc:
            raise RecorderError(f"could not open the microphone: {exc}") from exc
        return _SoundDeviceCapture(stream, blocks, self._sd.PortAudioError)

    def open_stream(self, device: int | None, rate: int) -> "_SoundDeviceStream":
        """Start a blocking stream that `read` drains."""
        try:
            stream = self._sd.RawInputStream(
                samplerate=rate, device=device, channels=1, dtype="int16"
            )
            stream.start()
        except self._sd.PortAudioError as exc:
            raise RecorderError(f"could not open the microphone: {exc}") from exc
        return _SoundDeviceStream(stream, self._sd.PortAudioError)

    def play(self, pcm: bytes, rate: int) -> None:
        """Play at `rate`, or resampled to the output's default rate if it refuses `rate`."""
        try:
            try:
                self._sd.check_output_settings(channels=1, dtype="int16", samplerate=rate)
            except (ValueError, self._sd.PortAudioError):
                target = int(self._sd.query_devices(kind="output")["default_samplerate"])
                pcm, rate = resample(pcm, rate, target), target
            with self._sd.RawOutputStream(samplerate=rate, channels=1, dtype="int16") as out:
                out.write(pcm)
        except (ValueError, self._sd.PortAudioError) as exc:
            raise RecorderError(f"playback failed: {exc}") from exc


class _RawStream(Protocol):
    """The part of `sounddevice.RawInputStream` the recorder uses."""

    def read(self, frames: int) -> tuple[bytes, bool]: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class _SoundDeviceCapture:
    def __init__(self, stream: _RawStream, blocks: list[bytes], error: type[Exception]) -> None:
        self._stream = stream
        self._blocks = blocks
        self._error = error

    def stop(self) -> bytes:
        try:
            self._stream.stop()
            self._stream.close()
        except self._error as exc:
            raise RecorderError(f"the microphone failed: {exc}") from exc
        return b"".join(self._blocks)


class _SoundDeviceStream:
    def __init__(self, stream: _RawStream, error: type[Exception]) -> None:
        self._stream = stream
        self._error = error

    def read(self, frames: int) -> bytes:
        try:
            data, _overflowed = self._stream.read(frames)
        except self._error as exc:
            raise RecorderError(f"the microphone failed: {exc}") from exc
        return bytes(data)

    def close(self) -> None:
        try:
            self._stream.stop()
            self._stream.close()
        except self._error as exc:
            raise RecorderError(f"the microphone failed: {exc}") from exc


class TerminalKeyboard:
    """Enter and typed answers from the console."""

    def wait_enter(self, timeout: float | None) -> bool:
        """Wait for Enter; with a timeout, poll the console so the wait can end on its own."""
        if timeout is None:
            input()
            return True
        deadline = time.monotonic() + timeout
        if sys.platform == "win32":
            import msvcrt

            while time.monotonic() < deadline:
                while msvcrt.kbhit():
                    if msvcrt.getwch() in ("\r", "\n"):
                        return True
                time.sleep(0.02)
            return False
        import select

        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            sys.stdin.readline()
            return True
        return False

    def choose(self, prompt: str) -> str:
        """Read one line."""
        return input(prompt)


# -- Audio and files ------------------------------------------------------------------------------


def resample(pcm: bytes, rate: int, target: int) -> bytes:
    """Mono PCM16 at `rate` as mono PCM16 at `target`, by polyphase filtering."""
    if rate == target:
        return pcm
    import numpy as np
    from scipy.signal import resample_poly

    divisor = math.gcd(rate, target)
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    out = resample_poly(samples, target // divisor, rate // divisor)
    return np.clip(np.rint(out), -32768, 32767).astype("<i2").tobytes()


def duration_ms(pcm: bytes) -> int:
    """The length of 16 kHz PCM16, as `manifest.py` compares it."""
    return round(len(pcm) // SAMPLE_WIDTH * 1000 / SAMPLE_RATE)


def write_wav(path: Path, pcm: bytes) -> None:
    """Write 16 kHz mono PCM16, never over an existing file, since that would lose a recording."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle, wave.open(handle, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(SAMPLE_WIDTH)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(pcm)
    except (OSError, wave.Error) as exc:
        raise RecorderError(f"could not write {path.name}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class DraftEntry:
    """One kept recording, in `manifest.py`'s fields less `split`, which `finish` assigns."""

    file: str
    label: str
    phrase: str
    condition: str
    duration_ms: int
    session: str


def load_draft(root: Path) -> list[DraftEntry]:
    """The draft's entries, or none when there is no draft yet."""
    path = root / DRAFT_NAME
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [DraftEntry(**item) for item in raw["entries"]]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RecorderError(f"{DRAFT_NAME} could not be read: {exc}") from exc


def write_draft(root: Path, entries: Sequence[DraftEntry]) -> None:
    """Replace the draft in one step, so a crash leaves the old draft or the new one."""
    payload = {"root": str(root), "entries": [asdict(item) for item in entries]}
    path = root / DRAFT_NAME
    staged = path.with_name(DRAFT_NAME + ".tmp")
    try:
        staged.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(staged, path)
    except OSError as exc:
        raise RecorderError(f"could not write {DRAFT_NAME}: {exc}") from exc


def _numbers(entries: Sequence[DraftEntry], kind: str) -> set[int]:
    """The block (`b`) or passive (`p`) numbers the draft's sessions hold."""
    found = set()
    for item in entries:
        match = _SESSION_NUMBER.search(item.session)
        if match and match.group(1) == kind:
            found.add(int(match.group(2)))
    return found


def _keep(root: Path, entries: list[DraftEntry], entry: DraftEntry, pcm: bytes) -> None:
    """Write the clip, then the draft that lists it, so the draft never names a missing file."""
    write_wav(root / entry.file, pcm)
    entries.append(entry)
    write_draft(root, entries)


# -- Commands -------------------------------------------------------------------------------------


def run_devices(host: Host) -> int:
    """List the input devices with their index and default rate."""
    devices = host.audio.input_devices()
    if not devices:
        raise RefusedError("no input device")
    for device in devices:
        print(f"{device.index}: {device.name} ({device.default_rate} Hz)")
    return 0


def _take(host: Host, prompt: Prompt, header: str, device: int | None, rate: int) -> bytes | None:
    """Record one prompt until it is kept (its 16 kHz audio) or the block is stopped (None)."""
    print(f'{header} Say: "{prompt.phrase}". Style: {prompt.style}')
    while True:
        print("Press Enter to record.")
        host.keyboard.wait_enter(None)
        capture = host.audio.start_capture(device, rate)
        print(f"Recording. Press Enter to stop ({MAX_CLIP_SECONDS} s at most).")
        host.keyboard.wait_enter(MAX_CLIP_SECONDS)
        raw = capture.stop()[: MAX_CLIP_SECONDS * rate * SAMPLE_WIDTH]
        pcm = resample(raw, rate, SAMPLE_RATE)
        if len(pcm) // SAMPLE_WIDTH < MIN_CLIP_FRAMES:
            print("Nothing was recorded; take it again.")
            continue
        while True:
            choice = host.keyboard.choose("Enter keeps it, r redo, p play back, q stop the block: ")
            choice = choice.strip().lower()
            if choice == "":
                return pcm
            if choice == "q":
                return None
            if choice == "r":
                break
            if choice == "p":
                try:
                    host.audio.play(pcm, SAMPLE_RATE)
                except RecorderError as exc:
                    print(exc)
            else:
                print("Type Enter, r, p or q.")


def run_session(host: Host, root: Path, device: int | None, condition: str) -> int:
    """Record the next block; the clips kept stay kept whether or not it is finished."""
    entries = load_draft(root)
    block = max(_numbers(entries, "b"), default=0) + 1
    if block > BLOCKS:
        raise RefusedError(f"all {BLOCKS} blocks are recorded; run `passive` or `finish`")
    rate = host.audio.capture_rate(device)
    session = f"{host.now():%Y%m%d-%H%M}-b{block}"
    prompts = block_prompts(block)
    print(f"Block {block} of {BLOCKS}. {BLOCK_SETUPS[block - 1]}")
    if rate != SAMPLE_RATE:
        print(f"The device records at {rate} Hz; each clip is resampled to {SAMPLE_RATE} Hz.")
    kept = 0
    for position, prompt in enumerate(prompts, 1):
        header = f"Block {block} of {BLOCKS}, clip {position} of {len(prompts)}."
        pcm = _take(host, prompt, header, device, rate)
        if pcm is None:
            break
        entry = DraftEntry(
            file=f"{RECORDINGS}/{session}/{prompt.label}-{position:03d}.wav",
            label=prompt.label,
            phrase=prompt.phrase,
            condition=condition,
            duration_ms=duration_ms(pcm),
            session=session,
        )
        root.mkdir(parents=True, exist_ok=True)
        _keep(root, entries, entry, pcm)
        kept += 1
    print(f"Block {block}: kept {kept} of {len(prompts)} clips.")
    if kept and block < BLOCKS:
        print(f"Take a short break. Next, block {block + 1}: {BLOCK_SETUPS[block]}")
    elif kept:
        print("All five blocks are recorded. Optionally run `passive`, then `finish`.")
    return 0


def run_passive(host: Host, root: Path, minutes: int, device: int | None) -> int:
    """Record the microphone in `CHUNK_MINUTES` files, each its own `media` session."""
    if minutes < 1:
        raise RefusedError("--minutes must be at least 1")
    entries = load_draft(root)
    number = max(_numbers(entries, "p"), default=0)
    rate = host.audio.capture_rate(device)
    remaining = minutes * 60 * rate
    chunk_frames = CHUNK_MINUTES * 60 * rate
    root.mkdir(parents=True, exist_ok=True)

    def keep_chunk(session: str, blocks: list[bytes]) -> None:
        pcm = resample(b"".join(blocks), rate, SAMPLE_RATE)
        entry = DraftEntry(
            file=f"{RECORDINGS}/{session}/media-001.wav",
            label="media",
            phrase=MEDIA_PHRASE,
            condition=CONDITION,
            duration_ms=duration_ms(pcm),
            session=session,
        )
        _keep(root, entries, entry, pcm)
        print(f"Kept {session}: {entry.duration_ms / 60000:.1f} minutes.")

    stream = host.audio.open_stream(device, rate)
    print(f"Recording {minutes} minutes in {CHUNK_MINUTES}-minute files. Ctrl+C stops early.")
    session = ""
    pending: list[bytes] = []
    try:
        while remaining > 0:
            number += 1
            session = f"{host.now():%Y%m%d-%H%M}-p{number}"
            target = min(chunk_frames, remaining)
            frames = 0
            while frames < target:
                data = stream.read(min(rate, target - frames))
                pending.append(data)
                frames += len(data) // SAMPLE_WIDTH
            blocks, pending = pending, []
            remaining -= frames
            keep_chunk(session, blocks)
    except KeyboardInterrupt:
        if pending:
            keep_chunk(session, pending)
        print("Stopped; the open file is kept.")
    finally:
        stream.close()
    return 0


def run_status(root: Path) -> int:
    """Print the blocks done, the clips per label, the passive minutes and what comes next."""
    entries = load_draft(root)
    blocks = _numbers(entries, "b")
    counts = ", ".join(
        f"{label} {sum(item.label == label for item in entries)}" for label in wake_manifest.LABELS
    )
    media_minutes = sum(item.duration_ms for item in entries if item.label == "media") / 60000
    following = max(blocks, default=0) + 1
    print(f"blocks: {len(blocks)} of {BLOCKS}")
    print(f"clips: {counts}")
    print(f"passive: {media_minutes:.1f} of {PASSIVE_GOAL_MINUTES} minutes (optional)")
    print(f"next: block {following}" if following <= BLOCKS else "next: finish")
    return 0


def run_finish(root: Path, seed: int) -> int:
    """Assign the draft's splits into `manifest.json`, then check it, through `manifest.py`."""
    draft = root / DRAFT_NAME
    out = root / MANIFEST_NAME
    if not draft.is_file():
        raise RefusedError(f"no {DRAFT_NAME} under --root; record a block first")
    if out.exists():
        raise RefusedError(f"{MANIFEST_NAME} already exists under --root; finish never replaces it")
    assign = ["assign", "--manifest", str(draft), "--seed", str(seed), "--out", str(out)]
    if wake_manifest.main(assign) != 0:
        return 1
    return 0 if wake_manifest.main(["check", "--manifest", str(out)]) == 0 else 1


# -- Command line ---------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.wake.record",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("devices", help="list the input devices")
    session = commands.add_parser("session", help="record the next guided block")
    session.add_argument("--root", required=True, type=Path)
    session.add_argument("--device", type=int)
    session.add_argument("--condition", default=CONDITION)
    passive = commands.add_parser("passive", help="record ordinary headset use as media")
    passive.add_argument("--root", required=True, type=Path)
    passive.add_argument("--minutes", required=True, type=int)
    passive.add_argument("--device", type=int)
    status = commands.add_parser("status", help="show what is recorded and what comes next")
    status.add_argument("--root", required=True, type=Path)
    finish = commands.add_parser("finish", help="assign splits into manifest.json and check it")
    finish.add_argument("--root", required=True, type=Path)
    finish.add_argument("--seed", required=True, type=int)
    return parser


def _real_host() -> Host:
    try:
        audio = SoundDeviceAudio()
    except ImportError as exc:
        raise RecorderError(
            f"`sounddevice` is not installed in this Python ({exc}); see the host setup in --help"
        ) from exc
    return Host(audio=audio, keyboard=TerminalKeyboard(), now=datetime.now)


def main(argv: Sequence[str] | None = None, host: Host | None = None) -> int:
    """Run one command; `host` replaces the microphone, keyboard and clock in tests."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "status":
            return run_status(args.root.resolve())
        if args.command == "finish":
            return run_finish(args.root.resolve(), args.seed)
        host = host or _real_host()
        if args.command == "devices":
            return run_devices(host)
        if args.command == "passive":
            return run_passive(host, args.root.resolve(), args.minutes, args.device)
        try:
            return run_session(host, args.root.resolve(), args.device, args.condition)
        except KeyboardInterrupt:
            print("\nStopped; the clips already kept are in the draft.")
            return 0
    except RefusedError as exc:
        print(f"record: {exc}", file=sys.stderr)
        return 2
    except (RecorderError, OSError) as exc:
        print(f"record: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
