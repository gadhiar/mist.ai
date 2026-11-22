"""Tests for the verifier's clean-window store.

Every input is generated here: low noise and tone bursts as WAV and FLAC files, a train-only
recording manifest, room responses at 16 kHz and 8 kHz, and a dataset index over them. The
reader is the trainer's own `WindowReader`, wrapped to count its calls. No test reads a
recording, needs a network or the GPU.
"""

import collections
import json
import os
import pickle
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import soundfile

from scripts.wake.manifest import load_manifest
from scripts.wake.verifier import dataset, store, train
from tests.unit.scripts.test_verifier_train import (
    _entries,
    _write_manifest,
    burst,
    noise,
    write_dataset,
)

RATE = 16000
FRAME = 320
WINDOW = 116 * FRAME
PROVENANCE = "c" * 64
FIXTURE_NAMES = (
    "pos-1",
    "pos-2",
    "pos-3",
    "rec/",
    "ES2002a",
    "19-198-0001",
    "mix-public",
    "mix-private",
    "small",
    "slow",
    "long",
    ".wav",
    ".flac",
)


@dataclass
class CountingReader:
    """The trainer's reader, counting the windows it reads, public and private apart."""

    inner: train.WindowReader
    public: int = 0
    private: int = 0

    def samples(self, window: dataset.Window) -> np.ndarray:
        if any(ref.source == "manifest" for ref in window.refs()):
            self.private += 1
        else:
            self.public += 1
        return self.inner.samples(window)

    def rir(self, ref: dataset.FileRef) -> np.ndarray:
        return self.inner.rir(ref)


@dataclass
class World:
    root: Path
    dataset: Path
    windows: list[dataset.Window]
    responses: list[dataset.FileRef]
    reader: train.WindowReader
    private: list[int] = field(default_factory=list)

    @property
    def stores(self) -> Path:
        return self.root / "stores"

    def key(self) -> str:
        return store.index_key(self.dataset)

    def counting(self) -> CountingReader:
        return CountingReader(self.reader)

    def open(self, reader: object | None = None, **options: object) -> store.WindowStore:
        settings: dict[str, object] = {
            "provenance_sha256": PROVENANCE,
            "workers": 1,
            "progress": lambda line: None,
        }
        settings.update(options)
        return store.open_store(
            self.stores,
            self.key(),
            self.windows,
            self.responses,
            reader or self.reader,
            **settings,
        )


def _wav_ref(source: str, root: Path, relative: str, samples: np.ndarray) -> dataset.FileRef:
    path = root / relative
    dataset.write_wav(path, samples)
    return dataset.FileRef(source, dataset.sha256_file(path), relative, None, len(samples))


def _rir_ref(root: Path, relative: str, samples: np.ndarray, rate: int) -> dataset.FileRef:
    path = root / relative
    dataset.write_wav(path, samples, rate)
    audio, read_rate = dataset.read_rir(path)
    length = len(dataset.to_16k(audio, read_rate))
    return dataset.FileRef("rirs", dataset.sha256_file(path), relative, None, length)


def _decay(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    envelope = np.exp(-np.arange(count) / (count / 6))
    return (rng.normal(0, 8000, count) * envelope).astype("<i2")


def _window(ref: dataset.FileRef, start: int, end: int, kind: str, group: str, inputs=()):
    label = "positive" if kind == "real_positive" else "negative"
    return dataset.Window(ref, start, end, label, kind, group, (0.5,), tuple(inputs))


@pytest.fixture
def world(tmp_path: Path) -> World:
    recordings = tmp_path / "recordings"
    entries = _entries(recordings, ["train"] * 6)
    manifest = load_manifest(_write_manifest(tmp_path / "train_manifest.json", recordings, entries))
    roots = {name: str(tmp_path / name) for name in ("ami", "librispeech", "mixed", "rirs")}
    rirs = tmp_path / "rirs"
    responses = [
        _rir_ref(rirs, "small.wav", _decay(2000, 1), RATE),
        _rir_ref(rirs, "slow.wav", _decay(1500, 2), 8000),
        _rir_ref(rirs, "long.wav", _decay(3 * RATE, 3), RATE),
    ]
    ami = _wav_ref(
        "ami",
        tmp_path / "ami",
        "ES2002a.Mix-Headset.wav",
        np.concatenate([noise(3.0, 99), burst(0.4), noise(2.6, 98)]),
    )
    flac_relative = "LibriSpeech/train-clean-100/19/198/19-198-0001.flac"
    flac = tmp_path / "librispeech" / flac_relative
    flac.parent.mkdir(parents=True)
    soundfile.write(str(flac), np.concatenate([noise(2.5, 70), burst(0.5)]), RATE, "PCM_16")
    libri = dataset.FileRef("librispeech", dataset.sha256_file(flac), flac_relative, None, 3 * RATE)
    mixed = tmp_path / "mixed"
    public_mix = _wav_ref("mixed", mixed, "mix-public.wav", noise(2.5, 60))
    private_mix = _wav_ref("mixed", mixed, "mix-private.wav", noise(2.5, 61))
    recording_refs = [
        dataset.FileRef(
            "manifest",
            item.sha256,
            None,
            item.index,
            dataset.wav_info(item.path).frames,
        )
        for item in manifest.entries
    ]
    windows = [
        _window(ami, 0, WINDOW, "mined_hit", "ami:ES2002a"),
        _window(recording_refs[0], 0, WINDOW, "real_positive", "manifest:s1"),
        _window(ami, 20 * FRAME, 60 * FRAME, "mined_hit", "ami:ES2002a"),
        _window(libri, 0, 60 * FRAME, "random", "librispeech:19"),
        _window(recording_refs[1], 10 * FRAME, 10 * FRAME + WINDOW, "real_positive", "manifest:s2"),
        _window(ami, 150 * FRAME, 250 * FRAME, "mined_hit", "ami:ES2002a"),
        _window(libri, 10 * FRAME, 10 * FRAME + WINDOW, "random", "librispeech:19"),
        _window(public_mix, 0, 80 * FRAME, "synthetic", "piper:1", (ami, responses[0])),
        _window(
            private_mix,
            0,
            WINDOW,
            "synthetic",
            "manifest:s3",
            (recording_refs[2], responses[2]),
        ),
    ]
    data = tmp_path / "dataset"
    write_dataset(data, windows, "d" * 64, roots)
    private = [1, 4, 8]
    reader = train.WindowReader(manifest, roots)
    return World(tmp_path, data, windows, responses, reader, private)


def _public(world: World) -> list[int]:
    return [index for index in range(len(world.windows)) if index not in world.private]


def _length(window: dataset.Window) -> int:
    return window.end - window.start


def _store_dir(world: World) -> Path:
    return world.stores / world.key()


# -- Reading back -----------------------------------------------------------------------------


class TestReadBack:
    def test_every_public_window_reads_back_as_the_reader_gives_it(self, world: World) -> None:
        opened = world.open()
        assert len(opened) == len(world.windows)
        for index in _public(world):
            window = world.windows[index]
            expected = world.reader.samples(window)
            length = _length(window)
            assert opened.length(index) == length
            for start, count in ((0, length), (0, 1), (length - 1, 1), (320, 640), (777, 5000)):
                got = opened.read(index, start, count)
                assert got.dtype == np.dtype("<i2")
                np.testing.assert_array_equal(got, expected[start : start + count])

    def test_every_response_is_the_readers_cut_to_the_longest_window(self, world: World) -> None:
        opened = world.open()
        assert store.MAX_RESPONSE == 37120
        for index, ref in enumerate(world.responses):
            got = opened.response(index)
            expected = world.reader.rir(ref)[: store.MAX_RESPONSE]
            assert got.dtype == np.dtype("<f4")
            np.testing.assert_array_equal(got, expected)
        assert len(opened.response(2)) == store.MAX_RESPONSE
        assert len(opened.response(1)) == 3000

    def test_a_read_outside_the_window_is_refused(self, world: World) -> None:
        opened = world.open()
        length = _length(world.windows[0])
        for window, start, count in ((0, -1, 2), (0, length - 1, 2), (len(world.windows), 0, 1)):
            with pytest.raises(store.StoreError):
                opened.read(window, start, count)
        with pytest.raises(store.StoreError):
            opened.response(len(world.responses))


class TestPrivacy:
    def test_private_windows_come_from_the_reader_and_never_reach_disk(self, world: World) -> None:
        opened = world.open()
        directory = _store_dir(world)
        public_bytes = sum(2 * _length(world.windows[index]) for index in _public(world))
        stored = (directory / "windows.pcm").read_bytes()
        assert len(stored) == public_bytes
        everything = b"".join(path.read_bytes() for path in directory.iterdir())
        for index in world.private:
            window = world.windows[index]
            expected = world.reader.samples(window)
            np.testing.assert_array_equal(opened.read(index, 0, _length(window)), expected)
            np.testing.assert_array_equal(opened.read(index, 100, 50), expected[100:150])
            assert expected.tobytes() not in everything
            assert expected[: RATE // 2].tobytes() not in everything


# -- Keying and invalidation ------------------------------------------------------------------


class TestKeying:
    def test_the_store_is_named_by_the_index_sha256(self, world: World) -> None:
        digest = dataset.sha256_bytes((world.dataset / dataset.INDEX_FILE).read_bytes())
        assert store.index_key(world.dataset) == digest
        world.open()
        assert sorted(path.name for path in world.stores.iterdir()) == [".lock", digest]
        record = json.loads((world.stores / digest / "store.json").read_text(encoding="utf-8"))
        assert record["key"] == digest
        assert record["provenance_sha256"] == PROVENANCE
        assert record["windows"] == len(world.windows)
        assert (
            record["sizes"]["windows.pcm"] == (world.stores / digest / "windows.pcm").stat().st_size
        )

    def test_a_changed_index_byte_builds_a_new_store_and_removes_the_old(
        self, world: World
    ) -> None:
        world.open()
        old = world.key()
        index = world.dataset / dataset.INDEX_FILE
        data = index.read_bytes()
        position = data.index(b"piper:1") + len("piper:")
        index.write_bytes(data[:position] + b"2" + data[position + 1 :])
        reader = world.counting()
        world.open(reader)
        assert world.key() != old
        assert reader.public == len(_public(world))
        assert sorted(path.name for path in world.stores.iterdir()) == [".lock", world.key()]

    def test_a_matching_store_is_reused_without_a_public_read(self, world: World) -> None:
        world.open()
        reader = world.counting()
        opened = world.open(reader)
        assert reader.public == 0
        assert reader.private == len(world.private)
        index = _public(world)[0]
        np.testing.assert_array_equal(
            opened.read(index, 0, 10), world.reader.samples(world.windows[index])[:10]
        )


def _rewrite(field_name: str, value: object):
    def change(directory: Path, world: World) -> None:
        path = directory / "store.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record[field_name] = value
        path.write_text(json.dumps(record), encoding="utf-8")

    return change


def _truncate(directory: Path, world: World) -> None:
    path = directory / "windows.pcm"
    with path.open("r+b") as handle:
        handle.truncate(path.stat().st_size - 2)


def _remove_record(directory: Path, world: World) -> None:
    (directory / "store.json").unlink()


def _leave_partial(directory: Path, world: World) -> None:
    partial = directory.with_name(f"{directory.name}.partial-99999")
    directory.rename(partial)


STALE = {
    "version": _rewrite("version", 0),
    "provenance": _rewrite("provenance_sha256", "e" * 64),
    "window_lengths": _rewrite("windows_digest", "f" * 64),
    "responses": _rewrite("responses_digest", "f" * 64),
    "truncated": _truncate,
    "no_record": _remove_record,
    "partial": _leave_partial,
}


class TestInvalidation:
    @pytest.mark.parametrize("stale", sorted(STALE))
    def test_a_stale_store_is_rebuilt(self, world: World, stale: str) -> None:
        world.open()
        directory = _store_dir(world)
        STALE[stale](directory, world)
        reader = world.counting()
        opened = world.open(reader)
        assert reader.public == len(_public(world))
        assert sorted(path.name for path in world.stores.iterdir()) == [".lock", world.key()]
        again = world.counting()
        world.open(again)
        assert again.public == 0
        index = _public(world)[-1]
        window = world.windows[index]
        np.testing.assert_array_equal(
            opened.read(index, 0, _length(window)), world.reader.samples(window)
        )

    def test_another_provenance_passed_in_rebuilds(self, world: World) -> None:
        world.open()
        reader = world.counting()
        world.open(reader, provenance_sha256="9" * 64)
        assert reader.public == len(_public(world))

    def test_other_stores_and_partials_are_removed_by_a_build(self, world: World) -> None:
        world.stores.mkdir(parents=True)
        for name in ("a" * 64, f"{'b' * 64}.partial-12", "notes"):
            (world.stores / name).mkdir()
        world.open()
        names = sorted(path.name for path in world.stores.iterdir())
        assert names == [".lock", world.key(), "notes"]

    def test_a_reusing_open_also_removes_other_stores_and_partials(self, world: World) -> None:
        world.open()
        for name in ("a" * 64, f"{'b' * 64}.partial-12", f"{world.key()}.partial-7", "notes"):
            (world.stores / name).mkdir()
        reader = world.counting()
        world.open(reader)
        assert reader.public == 0
        names = sorted(path.name for path in world.stores.iterdir())
        assert names == [".lock", world.key(), "notes"]

    def test_a_rebuild_counts_the_space_the_stale_stores_free(
        self, world: World, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        world.open()
        directory = _store_dir(world)
        shutil.copytree(directory, world.stores / ("a" * 64))
        _rewrite("provenance_sha256", "e" * 64)(directory, world)
        needed = sum(2 * _length(world.windows[index]) for index in _public(world))
        needed += 4 * sum(min(ref.samples, store.MAX_RESPONSE) for ref in world.responses)
        usage = collections.namedtuple("usage", "total used free")

        def disk_usage(path: object) -> object:
            held = sum(item.stat().st_size for item in world.stores.rglob("*") if item.is_file())
            return usage(1 << 40, held, needed + store.FREE_MARGIN - held)

        monkeypatch.setattr(shutil, "disk_usage", disk_usage)
        reader = world.counting()
        world.open(reader)
        assert reader.public == len(_public(world))
        assert sorted(path.name for path in world.stores.iterdir()) == [".lock", world.key()]


# -- Building ---------------------------------------------------------------------------------


class TestBuild:
    def test_two_workers_write_the_bytes_one_does(self, world: World, tmp_path: Path) -> None:
        one = world.open()
        two = store.open_store(
            tmp_path / "two",
            world.key(),
            world.windows,
            world.responses,
            world.reader,
            provenance_sha256=PROVENANCE,
            workers=2,
            progress=lambda line: None,
        )
        for name in ("windows.pcm", "responses.f32", "store.json"):
            assert (_store_dir(world) / name).read_bytes() == (
                tmp_path / "two" / world.key() / name
            ).read_bytes()
        assert len(one) == len(two)

    def test_progress_names_no_file_and_carries_a_rate_and_an_eta(
        self, world: World, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(store, "TASK_WINDOWS", 1)
        ticks = iter(range(0, 10_000, 61))
        lines: list[str] = []
        world.open(progress=lines.append, clock=lambda: float(next(ticks)), interval=60.0)
        total = len(_public(world))
        assert len(lines) >= 3
        assert f"{total}/{total} windows" in lines[-1]
        for line in lines:
            assert "windows/s" in line
            assert "ETA" in line
            assert " GB" in line
            assert str(world.root) not in line
            assert str(world.stores) not in line
            for name in FIXTURE_NAMES:
                assert name not in line

    def test_progress_waits_for_its_interval(self, world: World, monkeypatch) -> None:
        monkeypatch.setattr(store, "TASK_WINDOWS", 1)
        lines: list[str] = []
        world.open(progress=lines.append, clock=lambda: 0.0, interval=60.0)
        assert len(lines) == 1

    def test_too_little_free_space_is_refused(self, world: World, monkeypatch) -> None:
        usage = collections.namedtuple("usage", "total used free")
        monkeypatch.setattr(
            shutil, "disk_usage", lambda path: usage(1 << 30, (1 << 30) - 4096, 4096)
        )
        public_bytes = sum(2 * _length(world.windows[index]) for index in _public(world))
        response_bytes = 4 * sum(min(ref.samples, store.MAX_RESPONSE) for ref in world.responses)
        with pytest.raises(store.StoreError) as raised:
            world.open()
        message = str(raised.value)
        assert str(public_bytes + response_bytes) in message
        assert "4096" in message
        assert [path.name for path in world.stores.iterdir()] == [".lock"]

    def test_a_reader_error_propagates_and_leaves_no_partial(self, world: World) -> None:
        (world.root / "ami" / "ES2002a.Mix-Headset.wav").unlink()
        with pytest.raises(dataset.DatasetError):
            world.open()
        assert [path.name for path in world.stores.iterdir()] == [".lock"]

    def test_a_response_of_another_length_than_its_index_says_is_refused(
        self, world: World
    ) -> None:
        responses = list(world.responses)
        ref = responses[0]
        responses[0] = dataset.FileRef(ref.source, ref.sha256, ref.path, None, ref.samples + 1)
        with pytest.raises(store.StoreError):
            store.open_store(
                world.stores,
                world.key(),
                world.windows,
                responses,
                world.reader,
                provenance_sha256=PROVENANCE,
                workers=1,
                progress=lambda line: None,
            )
        assert [path.name for path in world.stores.iterdir()] == [".lock"]


# -- Pickling and processes -------------------------------------------------------------------


class TestProcesses:
    def test_a_store_pickles_without_a_descriptor(self, world: World) -> None:
        opened = world.open()
        index = _public(world)[0]
        first = opened.read(index, 0, 100)
        state = opened.__getstate__()
        assert state["_descriptors"] is None
        data = pickle.dumps(opened)
        private_bytes = sum(2 * _length(world.windows[number]) for number in world.private)
        assert len(data) < private_bytes + 64 * 1024
        clone = pickle.loads(data)
        np.testing.assert_array_equal(clone.read(index, 0, 100), first)
        np.testing.assert_array_equal(
            clone.read(world.private[0], 0, 100),
            world.reader.samples(world.windows[world.private[0]])[:100],
        )

    def test_a_forked_process_opens_its_own_descriptor(
        self, world: World, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        opened = world.open()
        index = _public(world)[0]
        opened.read(index, 0, 10)
        parent = opened._descriptors
        assert parent is not None
        monkeypatch.setattr(os, "getpid", lambda: -1)
        np.testing.assert_array_equal(
            opened.read(index, 0, 10), world.reader.samples(world.windows[index])[:10]
        )
        assert opened._descriptors != parent
