"""The verifier's clean windows, decoded once into one flat file the trainer reads by offset.

Reading a window through `WindowReader` opens, parses and decodes its source file, and the
trainer did that for every example of every epoch, over hundreds of thousands of windows, so
the GPU waited on the decoder. The store decodes each window once, when its dataset index is
new, into `windows.pcm`: int16 little-endian, the windows' samples in index order, one after
another. An epoch then reads an example with one `pread` at an offset known in advance. The
room responses the augmentation convolves with are kept beside them in `responses.f32`, each
cut to `MAX_RESPONSE`, the longest window, since no later tap reaches a kept sample.

A store lives in `<root>/<key>`, `key` being the sha256 of the index's `index.jsonl`, and
`store.json`, written last, records what it was built from. A store whose record or file sizes
differ from what the caller asks for is rebuilt, and every open, reusing or rebuilding, removes
every other store under the root that no run holds open: a store for the full index is tens of
GB.

Locks: `<root>/.lock` serialises opens, so a sweep, a build and the taking of a store's lock
never overlap. Each open `WindowStore` holds a shared `flock` on its store's directory until
`close()`. A sweep removes another store only when it takes that lock exclusively without
waiting, and a rebuild needs it exclusively too, so no open removes or rewrites a store another
run is reading; it skips the store, or for its own key refuses. The lock is on the directory
itself, not on a lock file beside it, so it guards the very directory the holder reads from, and
the root gains no per-store file that could outlive its store and need a sweep of its own.

Privacy: private audio never reaches the disk. A window that reads or mixes any file of source
`manifest`, the train-only manifest's recordings, takes no bytes in `windows.pcm`; each open
reads those windows through the reader into memory, as the trainer reads every recording
through the manifest and never copies one.

Writes use `os.pwrite` into a file sized up front, not a writable shared mmap, because Docker
Desktop's bind mounts of Windows drives do not promise that writes through a shared mapping
reach the file. Decoding runs in processes started by spawn, since the trainer's own process
already runs torch's thread pools, and a child forked from a threaded process can inherit a lock
another thread held and wait on it forever.
"""

import fcntl
import hashlib
import json
import multiprocessing
import os
import re
import shutil
import sys
import time
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .dataset import (
    AFTER_FRAMES,
    BEFORE_FRAMES,
    FRAME_SAMPLES,
    INDEX_FILE,
    FileRef,
    Window,
    sha256_file,
)

STORE_VERSION = 1
MAX_RESPONSE = (BEFORE_FRAMES + AFTER_FRAMES + 1) * FRAME_SAMPLES
"""The samples kept of a room response: the longest window, so no later tap reaches a sample."""
WINDOWS_FILE = "windows.pcm"
RESPONSES_FILE = "responses.f32"
RECORD_FILE = "store.json"
LOCK_FILE = ".lock"
PRIVATE_SOURCE = "manifest"
"""The source whose audio is private: the train-only manifest's recordings."""
FREE_MARGIN = 1 << 30
"""Free space a build leaves on the disk beyond the store itself, 1 GiB."""
TASK_WINDOWS = 256
"""Windows one decode task holds: enough that a task outweighs its dispatch."""
TASK_RESPONSES = 64
IN_FLIGHT = 4
"""Tasks queued per decoding process, so the queue holds work but not the whole index."""

_PCM = np.dtype("<i2")
_F32 = np.dtype("<f4")
_KEY = re.compile(r"[0-9a-f]{64}")
_PARTIAL = re.compile(r"[0-9a-f]{64}\.partial-\d+")


class StoreError(ValueError):
    """A store that cannot be built or read as asked."""


class Reader(Protocol):
    """Where the store reads audio from; `train.WindowReader` is one."""

    def samples(self, window: Window) -> np.ndarray:
        """The window's int16 samples, `end - start` of them."""
        ...

    def rir(self, ref: FileRef) -> np.ndarray:
        """A room response at 16 kHz as float32, peak-normalised."""
        ...


def index_key(dataset: Path) -> str:
    """The key a dataset's store is kept under: the sha256 of its `index.jsonl` bytes."""
    return sha256_file(dataset / INDEX_FILE)


def is_private(window: Window) -> bool:
    """True when the window reads or mixes a recording, whose audio is never written."""
    return any(ref.source == PRIVATE_SOURCE for ref in window.refs())


# -- Layout and record ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Layout:
    """Where each window and response sits: byte offsets, -1 for a window kept in memory."""

    offsets: np.ndarray
    lengths: np.ndarray
    private: np.ndarray
    response_offsets: np.ndarray
    response_lengths: np.ndarray

    @property
    def windows_bytes(self) -> int:
        """The size of `windows.pcm`: the public windows' samples alone."""
        return int(self.lengths[~self.private].sum()) * _PCM.itemsize

    @property
    def responses_bytes(self) -> int:
        """The size of `responses.f32`."""
        return int(self.response_lengths.sum()) * _F32.itemsize


def _layout(windows: Sequence[Window], responses: Sequence[FileRef]) -> Layout:
    lengths = np.array([window.end - window.start for window in windows], dtype=np.int64)
    private = np.array([is_private(window) for window in windows], dtype=bool)
    public_lengths = np.where(private, 0, lengths)
    starts = (np.cumsum(public_lengths) - public_lengths) * _PCM.itemsize
    offsets = np.where(private, -1, starts).astype(np.int64)
    response_lengths = np.array(
        [min(ref.samples, MAX_RESPONSE) for ref in responses], dtype=np.int64
    )
    response_offsets = (np.cumsum(response_lengths) - response_lengths) * _F32.itemsize
    return Layout(offsets, lengths, private, response_offsets, response_lengths)


def _record(
    key: str, provenance_sha256: str, layout: Layout, responses: Sequence[FileRef]
) -> dict[str, object]:
    """What `store.json` holds for this request; a store whose record differs is rebuilt."""
    windows = hashlib.sha256()
    windows.update(layout.lengths.astype("<i8").tobytes())
    windows.update(layout.private.astype(np.uint8).tobytes())
    listed = [[ref.sha256, int(length)] for ref, length in zip(responses, layout.response_lengths)]
    return {
        "version": STORE_VERSION,
        "key": key,
        "provenance_sha256": provenance_sha256,
        "windows": len(layout.lengths),
        "windows_digest": windows.hexdigest(),
        "responses": len(listed),
        "responses_digest": hashlib.sha256(json.dumps(listed).encode("utf-8")).hexdigest(),
        "sizes": {WINDOWS_FILE: layout.windows_bytes, RESPONSES_FILE: layout.responses_bytes},
    }


def _matches(directory: Path, record: Mapping[str, object]) -> bool:
    """True when `directory` holds the store `record` describes, files at their sizes."""
    try:
        found = json.loads((directory / RECORD_FILE).read_text(encoding="utf-8"))
        sizes = {name: (directory / name).stat().st_size for name in (WINDOWS_FILE, RESPONSES_FILE)}
    except (OSError, ValueError):
        return False
    return found == record and sizes == record["sizes"]


# -- The store --------------------------------------------------------------------------------


class WindowStore:
    """The windows and room responses of one index, read by offset.

    The files are opened lazily in each process and reopened when the process id changes, so a
    DataLoader worker forked from the trainer reads through its own descriptor. The pickled
    state holds the layout and the private windows, never a descriptor.

    The store opened by `open_store` holds the shared lock on its directory from then until
    `close()`, or until it is garbage collected, so no other open removes it meanwhile. A
    worker's copy, forked or unpickled, holds no lock of its own; the opening process's lock
    covers the worker's reads while that process keeps the store open.
    """

    def __init__(
        self,
        directory: Path,
        layout: Layout,
        private: Mapping[int, np.ndarray],
        lock: int | None = None,
    ) -> None:
        self._directory = directory
        self._layout = layout
        self._private = dict(private)
        self._descriptors: tuple[int, int, int] | None = None
        self._holder = None if lock is None else weakref.finalize(self, _release, os.getpid(), lock)

    def __len__(self) -> int:
        return len(self._layout.lengths)

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state["_descriptors"] = None
        state["_holder"] = None
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)

    @property
    def directory(self) -> Path:
        """The store's directory."""
        return self._directory

    @property
    def responses(self) -> int:
        """How many room responses the store holds."""
        return len(self._layout.response_lengths)

    def length(self, window: int) -> int:
        """The window's length in samples."""
        self._check_window(window)
        return int(self._layout.lengths[window])

    def is_private(self, window: int) -> bool:
        """True for a window held in memory, never on disk."""
        self._check_window(window)
        return bool(self._layout.private[window])

    def read(self, window: int, start: int, count: int) -> np.ndarray:
        """Samples `[start, start + count)` of the window, relative to its start, as int16.

        A public window is read by one `os.preadv` straight into the returned array.

        Raises:
            StoreError: The window or the range is outside the store, or the file is short.
        """
        length = self.length(window)
        if start < 0 or count < 0 or start + count > length:
            raise StoreError(
                f"samples {start} to {start + count} are outside a {length}-sample window"
            )
        if self._layout.private[window]:
            return self._private[window][start : start + count].copy()
        out = np.empty(count, dtype=_PCM)
        offset = int(self._layout.offsets[window]) + start * _PCM.itemsize
        self._pread(self._fds()[1], out, offset, WINDOWS_FILE)
        return out

    def response(self, index: int) -> np.ndarray:
        """The room response `index`, in the order the store was opened with, as float32.

        Raises:
            StoreError: The index is outside the store, or the file is short.
        """
        if not 0 <= index < self.responses:
            raise StoreError(f"response {index} is outside the store's {self.responses}")
        out = np.empty(int(self._layout.response_lengths[index]), dtype=_F32)
        offset = int(self._layout.response_offsets[index])
        self._pread(self._fds()[2], out, offset, RESPONSES_FILE)
        return out

    def close(self) -> None:
        """Close this process's descriptors and release the store's lock.

        A later read opens the files again, but the store is no longer held: the next open of
        another key may remove it.
        """
        if self._descriptors is not None and self._descriptors[0] == os.getpid():
            os.close(self._descriptors[1])
            os.close(self._descriptors[2])
        self._descriptors = None
        if self._holder is not None:
            self._holder()
            self._holder = None

    def _check_window(self, window: int) -> None:
        if not 0 <= window < len(self):
            raise StoreError(f"window {window} is outside the store's {len(self)}")

    def _fds(self) -> tuple[int, int, int]:
        pid = os.getpid()
        if self._descriptors is None or self._descriptors[0] != pid:
            self._descriptors = (
                pid,
                os.open(self._directory / WINDOWS_FILE, os.O_RDONLY),
                os.open(self._directory / RESPONSES_FILE, os.O_RDONLY),
            )
        return self._descriptors

    @staticmethod
    def _pread(fd: int, out: np.ndarray, offset: int, name: str) -> None:
        if out.nbytes and os.preadv(fd, [out], offset) != out.nbytes:
            raise StoreError(f"{name} is shorter than its store.json says")


# -- Building ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _WindowJob:
    window: Window
    offset: int


@dataclass(frozen=True, slots=True)
class _ResponseJob:
    ref: FileRef
    offset: int
    length: int


_Task = tuple[_WindowJob | _ResponseJob, ...]


def _tasks(windows: Sequence[Window], responses: Sequence[FileRef], layout: Layout) -> list[_Task]:
    """The decode tasks: responses, then the public windows grouped by file, in file order."""
    tasks: list[_Task] = []
    jobs = [
        _ResponseJob(ref, int(offset), int(length))
        for ref, offset, length in zip(responses, layout.response_offsets, layout.response_lengths)
    ]
    tasks += [tuple(jobs[i : i + TASK_RESPONSES]) for i in range(0, len(jobs), TASK_RESPONSES)]
    public = [index for index in range(len(windows)) if not layout.private[index]]
    public.sort(key=lambda index: _file_order(windows[index]))
    window_jobs = [_WindowJob(windows[index], int(layout.offsets[index])) for index in public]
    tasks += [
        tuple(window_jobs[i : i + TASK_WINDOWS]) for i in range(0, len(window_jobs), TASK_WINDOWS)
    ]
    return tasks


def _file_order(window: Window) -> tuple[str, str, str, int, int]:
    ref = window.file
    entry = -1 if ref.entry is None else ref.entry
    return (ref.source, ref.sha256, ref.path or "", entry, window.start)


def _pwrite(fd: int, data: bytes, offset: int) -> None:
    view = memoryview(data)
    while view:
        written = os.pwrite(fd, view, offset)
        view, offset = view[written:], offset + written


def _decode(reader: Reader, windows_fd: int, responses_fd: int, task: _Task) -> int:
    """Decode a task's jobs and write each at its offset; the count of windows written."""
    done = 0
    for job in task:
        if isinstance(job, _ResponseJob):
            response = np.asarray(reader.rir(job.ref), dtype=_F32)[:MAX_RESPONSE]
            if len(response) != job.length:
                raise StoreError(
                    f"a room response decodes to {len(response)} samples kept, not the "
                    f"{job.length} its index gives"
                )
            _pwrite(responses_fd, response.tobytes(), job.offset)
            continue
        samples = reader.samples(job.window)
        expected = job.window.end - job.window.start
        if samples.dtype != _PCM or len(samples) != expected:
            raise StoreError(
                f"a {job.window.file.source} window decodes to {len(samples)} {samples.dtype} "
                f"samples, not {expected} int16"
            )
        _pwrite(windows_fd, samples.tobytes(), job.offset)
        done += 1
    return done


@dataclass(frozen=True, slots=True)
class _Worker:
    reader: Reader
    windows_fd: int
    responses_fd: int
    limits: object


_worker: _Worker | None = None


def _start_worker(reader: Reader, windows_path: str, responses_path: str) -> None:
    """A decoding process's setup: one intra-op thread, and its own write descriptors."""
    global _worker  # noqa: PLW0603 - one per process, set before its first task
    from threadpoolctl import threadpool_limits  # noqa: PLC0415 - decoding processes only

    limits = threadpool_limits(1)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(1)
    _worker = _Worker(
        reader,
        os.open(windows_path, os.O_WRONLY),
        os.open(responses_path, os.O_WRONLY),
        limits,
    )


def _run_task(task: _Task) -> int:
    if _worker is None:
        raise StoreError("a decode task ran in a process the pool did not start")
    return _decode(_worker.reader, _worker.windows_fd, _worker.responses_fd, task)


def _pooled(
    tasks: Sequence[_Task],
    reader: Reader,
    paths: tuple[Path, Path],
    workers: int,
    done: Callable[[int], None],
) -> None:
    """Run the tasks in `workers` spawned processes, passing each one's window count to `done`.

    At most `IN_FLIGHT` tasks per process are queued. On an error the queued ones are cancelled
    and the pool shuts down, its running tasks finished, before the error leaves, so no process
    still writes when the caller removes the partial build.
    """
    queue = iter(tasks)
    pending: set[Future[int]] = set()
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_start_worker,
        initargs=(reader, str(paths[0]), str(paths[1])),
    ) as pool:
        try:
            for task in queue:
                pending.add(pool.submit(_run_task, task))
                if len(pending) >= workers * IN_FLIGHT:
                    break
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    done(future.result())
                    task = next(queue, None)
                    if task is not None:
                        pending.add(pool.submit(_run_task, task))
        finally:
            for future in pending:
                future.cancel()


def _in_process(
    tasks: Sequence[_Task], reader: Reader, paths: tuple[Path, Path], done: Callable[[int], None]
) -> None:
    windows_fd = os.open(paths[0], os.O_WRONLY)
    try:
        responses_fd = os.open(paths[1], os.O_WRONLY)
        try:
            for task in tasks:
                done(_decode(reader, windows_fd, responses_fd, task))
        finally:
            os.close(responses_fd)
    finally:
        os.close(windows_fd)


class _Progress:
    """Progress lines: counts, rate, ETA and size, never a file name or a path."""

    def __init__(
        self,
        total: int,
        size: int,
        progress: Callable[[str], None],
        interval: float,
        clock: Callable[[], float],
    ) -> None:
        self.total = total
        self.size = size
        self.progress = progress
        self.interval = interval
        self.clock = clock
        self.done = 0
        self.started = clock()
        self.last = self.started

    def add(self, count: int) -> None:
        self.done += count
        now = self.clock()
        if now - self.last >= self.interval:
            self.last = now
            self._emit(now)

    def finish(self) -> None:
        self._emit(self.clock())

    def _emit(self, now: float) -> None:
        elapsed = now - self.started
        rate = self.done / elapsed if elapsed > 0 else 0.0
        left = self.total - self.done
        eta = _duration(left / rate) if rate > 0 else ("0:00:00" if left == 0 else "unknown")
        self.progress(
            f"store: {self.done}/{self.total} windows, {rate:.1f} windows/s, ETA {eta}, "
            f"{self.size / 1e9:.2f} GB"
        )


def _duration(seconds: float) -> str:
    whole = int(round(seconds))
    return f"{whole // 3600}:{whole // 60 % 60:02d}:{whole % 60:02d}"


def _build(
    root: Path,
    key: str,
    windows: Sequence[Window],
    responses: Sequence[FileRef],
    reader: Reader,
    layout: Layout,
    record: Mapping[str, object],
    *,
    workers: int,
    progress: _Progress,
) -> None:
    """Decode into `<key>.partial-<pid>`, write `store.json` last, and rename it into place.

    The stale store under `key` is removed before the free space is read, so the check counts
    the space it held: a rebuild of a full index would otherwise need room for two stores.
    """
    final = root / key
    shutil.rmtree(final, ignore_errors=True)
    needed = layout.windows_bytes + layout.responses_bytes
    free = shutil.disk_usage(root).free
    if free < needed + FREE_MARGIN:
        raise StoreError(
            f"the store needs {needed} bytes and {FREE_MARGIN} more to spare, but {free} bytes "
            "are free"
        )
    partial = root / f"{key}.partial-{os.getpid()}"
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir()
    built = False
    try:
        paths = (partial / WINDOWS_FILE, partial / RESPONSES_FILE)
        for path, size in zip(paths, (layout.windows_bytes, layout.responses_bytes)):
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                os.ftruncate(fd, size)
            finally:
                os.close(fd)
        tasks = _tasks(windows, responses, layout)
        if workers > 1 and len(tasks) > 1:
            _pooled(tasks, reader, paths, min(workers, len(tasks)), progress.add)
        else:
            _in_process(tasks, reader, paths, progress.add)
        for path in paths:
            _fsync(path)
        staged = partial / f"{RECORD_FILE}.tmp"
        staged.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        _fsync(staged)
        os.replace(staged, partial / RECORD_FILE)
        os.rename(partial, final)
        built = True
    finally:
        if not built:
            shutil.rmtree(partial, ignore_errors=True)
    progress.finish()


def _sweep(root: Path, key: str) -> None:
    """Remove every other store no run holds, and every partial build, `key`'s own included.

    Run under the root lock on every open, reused or rebuilt, so a build that follows counts
    the space the removed stores held. A store is a directory named by a key that holds
    `store.json`; another 64-hex directory is not one of ours and is left alone. A partial build
    is always dead here: builds run under the root lock this sweep holds.
    """
    for path in sorted(root.iterdir()):
        if _PARTIAL.fullmatch(path.name) and path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.name != key and _KEY.fullmatch(path.name) and (path / RECORD_FILE).is_file():
            _remove_unheld(root, path.name)


def _remove_unheld(root: Path, name: str) -> None:
    """Remove the store `name`, unless an open `WindowStore` holds its lock.

    The directory is renamed to a partial build's name before it is removed, so a removal cut
    short leaves what the next sweep removes, not a directory that has lost its `store.json`.
    """
    directory = root / name
    fd = _open_directory(directory)
    try:
        if not _try_exclusive(fd):
            return
        doomed = root / f"{name}.partial-{os.getpid()}"
        shutil.rmtree(doomed, ignore_errors=True)
        os.rename(directory, doomed)
        shutil.rmtree(doomed, ignore_errors=True)
    finally:
        os.close(fd)


def _fsync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def _locked(root: Path) -> Iterator[None]:
    """An exclusive lock on the root, so two runs never build or remove a store at once."""
    with (root / LOCK_FILE).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


# A store's own lock is `flock` on its directory, not `fcntl.lockf` or an `F_SETLK` record lock.
# A `flock` lock belongs to the open file description: a DataLoader worker forked from the
# trainer inherits the descriptor and so shares the trainer's lock, and the worker's exit, which
# closes its copy, leaves the lock held. A record lock belongs to the process instead: a forked
# child does not inherit it, the holder loses it when it closes any descriptor of the file, the
# one a sweep in the same process opens to test it included, and it never conflicts with another
# lock of the same process, so a second open in one run would not see the first. An exclusive
# record lock also needs a descriptor open for writing, which a directory cannot have; `flock`
# needs neither. Workers started by spawn hold no descriptor; they reopen the store's files by
# path, under the opener's lock.
#
# Every lock on a store's directory is taken under the root lock, so between a check that no one
# holds a store and its removal or rebuild, no other process can take the store's lock.


def _open_directory(directory: Path) -> int:
    return os.open(directory, os.O_RDONLY | os.O_DIRECTORY)


def _try_exclusive(fd: int) -> bool:
    """True when the exclusive lock was taken: no other open description holds any lock."""
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _claim(directory: Path) -> None:
    """Check that no open `WindowStore` holds `directory`, the stale store a rebuild removes.

    Raises:
        StoreError: Another holder has the store open.
    """
    if not directory.is_dir():
        return
    fd = _open_directory(directory)
    try:
        if not _try_exclusive(fd):
            raise StoreError(
                "the store for this index is stale and in use by another open store, so it is "
                "not rebuilt; close that run first"
            )
    finally:
        os.close(fd)


def _hold(directory: Path) -> int:
    """A descriptor of `directory` holding its shared lock, taken under the root lock.

    No exclusive lock is held while the root lock is, so this never waits.
    """
    fd = _open_directory(directory)
    held = False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        held = True
    finally:
        if not held:
            os.close(fd)
    return fd


def _release(owner: int, fd: int) -> None:
    """Release a store's lock: unlocked by the process that took it, closed by any.

    Only the owner unlocks: an unlock through a forked worker's copy of the descriptor would
    release the lock the owner still holds, since the two share one open file description.
    """
    try:
        if os.getpid() == owner:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def open_store(
    root: Path,
    key: str,
    windows: Sequence[Window],
    responses: Sequence[FileRef],
    reader: Reader,
    *,
    provenance_sha256: str,
    workers: int,
    progress: Callable[[str], None] = print,
    interval: float = 60.0,
    clock: Callable[[], float] = time.monotonic,
) -> WindowStore:
    """Open the store of `windows` and `responses` under `root`, building it when it is stale.

    A store is reused only when its `store.json` matches the request in every field and its
    files have their recorded sizes; anything else is rebuilt. Every open first removes every
    partial build under `root` and every other store no open `WindowStore` holds, and a rebuild
    removes the stale store before it checks the free space. A stale store another holder has
    open is not rebuilt: the open is refused. All of it runs under the root's lock, and the
    returned store holds the shared lock on its directory until `close()`. Private windows are
    read through `reader` on every open and never written.

    Args:
        root: The directory stores are kept in.
        key: `index_key` of the dataset `windows` came from.
        windows: The index's windows, in index order; `read` takes their positions.
        responses: The room responses, in the order `response` takes their positions; each
            `samples` is its length at 16 kHz, as `reader.rir` returns it.
        reader: Reads windows and room responses; it is pickled to each decoding process.
        provenance_sha256: The sha256 of the dataset's `provenance.json`.
        workers: Decoding processes; 1 or fewer decodes in this process.
        progress: Takes a progress line at most every `interval` seconds and one at the end.
        interval: Seconds of `clock` between progress lines.
        clock: The clock `interval` is measured on.

    Raises:
        StoreError: The key is not a sha256, the store is stale while another holder has it
            open, the disk has too little free space, or a file decodes to another length than
            its index gives.
        DatasetError: The reader could not read a file.
    """
    if not _KEY.fullmatch(key):
        raise StoreError("a store key is the index's sha256, 64 lower-case hex digits")
    layout = _layout(windows, responses)
    record = _record(key, provenance_sha256, layout, responses)
    root.mkdir(parents=True, exist_ok=True)
    directory = root / key
    with _locked(root):
        _sweep(root, key)
        if not _matches(directory, record):
            _claim(directory)
            size = layout.windows_bytes + layout.responses_bytes
            total = int((~layout.private).sum())
            tracker = _Progress(total, size, progress, interval, clock)
            _build(
                root,
                key,
                windows,
                responses,
                reader,
                layout,
                record,
                workers=workers,
                progress=tracker,
            )
        lock = _hold(directory)
    opened = None
    try:
        private = {
            index: np.asarray(reader.samples(windows[index]), dtype=_PCM)
            for index in np.flatnonzero(layout.private).tolist()
        }
        opened = WindowStore(directory, layout, private, lock)
    finally:
        if opened is None:
            os.close(lock)
    return opened
