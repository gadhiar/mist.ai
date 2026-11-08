"""The gaming watcher's pieces: the process lister, the game rules, and `decide`.

`instance.py gaming-watch` loads this file by path from beside itself, inside that command
only, so every other command runs from a `bin/` without it. Like `instance.py`, it runs on host
Python with the standard library alone.

Only a game's own process counts, never its launcher, client, anti-cheat, overlay,
redistributable or a background app (item 17, Watcher). Two rules decide it, in order:

1. A shared runtime (`java.exe`, `javaw.exe`) counts when the folder it runs for, its
   `--gameDir` or else its working directory, lies under a `MIST_GAME_RUNTIME_DIRS` entry, its
   arguments run no installer, and the folder is not under a `MIST_GAME_EXCLUDE` directory.
   Where the runtime itself is installed plays no part.
2. Any program counts when its image lies under a `MIST_GAME_DIRS` library and neither the
   shipped exclusion table nor a `MIST_GAME_EXCLUDE` entry matches it.

`classify`, `find_game` and `decide` are pure: the loop in `instance.py` lists processes,
reads the gaming-mode marker and runs the commands, and this module only says what the
evidence calls for. Paths are compared through `ntpath`, not `os.path`, so the rules are
Windows-correct in the Linux test runner too.
"""

from __future__ import annotations

import fnmatch
import ntpath
import os
import struct
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

GAME_DIRS_KEY = "MIST_GAME_DIRS"
RUNTIME_DIRS_KEY = "MIST_GAME_RUNTIME_DIRS"
EXCLUDE_KEY = "MIST_GAME_EXCLUDE"

# The programs whose location says nothing about what they run, so only these are opened for
# their command line and working directory.
SHARED_RUNTIMES = ("java.exe", "javaw.exe")

# File names that are never a game's own process, matched with `fnmatch.fnmatchcase` against
# the image's file name, lower-cased. The table is generic: no drive, user folder or title.
SHIPPED_EXCLUDED_NAMES = (
    # Launchers.
    "*launcher*.exe",
    "gamelaunchhelper.exe",
    "unrealversionselector.exe",
    # Clients.
    "leagueclient*.exe",
    # Crash reporting.
    "*crashhandler*.exe",
    "*crashreport*.exe",
    "*crashpad*.exe",
    "createdump.exe",
    # Web helpers.
    "*webhelper*.exe",
    "*cefsubprocess*.exe",
    "qtwebengineprocess.exe",
    # Overlays.
    "*overlay*.exe",
    # Installers, updaters and redistributables.
    "*installer*.exe",
    "*setup*.exe",
    "*redist*.exe",
    "dotnetfx*.exe",
    "*updater*.exe",
    "unins*.exe",
    "uninstall*.exe",
    # Anti-cheat.
    "easyanticheat*.exe",
    "start_protected_game.exe",
    "*_eac.exe",
    "beservice*.exe",
)

# Folders that never hold a game's own process: each is one or more path components, which
# must appear consecutively among the image's folders, compared lower-cased.
SHIPPED_EXCLUDED_FOLDERS = (
    "riot client",
    "riot vanguard",
    "epic games\\launcher",
    "minecraft launcher",
    "steamvr",
    "steamworks shared",
    "steam controller configs",
    "_commonredist",
    "redist",
    "directxredist",
    "easyanticheat",
    "battleye",
)

# Enough to open a process for its image name and command line without the rights a protected
# or elevated process withholds; such a process is skipped, not an error.
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_VM_READ = 0x0010
# The longest path `QueryFullProcessImageNameW` can return, in characters.
MAX_IMAGE_PATH = 32768
INITIAL_PID_SLOTS = 1024
# `NtQueryInformationProcess` classes and the status that asks for a larger buffer.
PROCESS_BASIC_INFORMATION_CLASS = 0
PROCESS_COMMAND_LINE_INFORMATION_CLASS = 60
STATUS_INFO_LENGTH_MISMATCH = -1073741820  # 0xC0000004 as a signed NTSTATUS.
# The x64 layouts the working-directory read rests on: `PEB.ProcessParameters` and
# `RTL_USER_PROCESS_PARAMETERS.CurrentDirectory.DosPath`. Microsoft does not document them;
# the scratch rehearsal (H3 step 14) checks them against a real Java process.
PEB_PROCESS_PARAMETERS_OFFSET = 0x20
PARAMETERS_CURRENT_DIRECTORY_OFFSET = 0x38
# A `UNICODE_STRING` length is a USHORT in bytes, so no read is ever longer than this.
MAX_UNICODE_STRING_BYTES = 0xFFFE

Action = Literal["on", "off"]


class GameWatchError(Exception):
    """The running processes could not be listed."""


@dataclass(frozen=True, slots=True)
class WatchState:
    """What the watcher remembers between ticks.

    `last_seen_game_at` is the monotonic time of the last tick that saw a game, or None.
    `marker_was_present` says whether the last tick saw a gaming-mode marker, so a marker that
    vanished in between is known to be someone else's `off`. `suppressed` holds the watcher back
    after such an `off` until a tick sees no game.
    """

    last_seen_game_at: float | None
    marker_was_present: bool
    suppressed: bool


INITIAL_STATE = WatchState(last_seen_game_at=None, marker_was_present=False, suppressed=False)


@dataclass(frozen=True, slots=True)
class ProcessRecord:
    """One running process, as far as the lister read it.

    Only a shared runtime's record carries `cwd` and `args` (the arguments after the program
    itself); None means not read, or unreadable.
    """

    image: str
    cwd: str | None = None
    args: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class GameRules:
    """The three folder variables of `daily.env`, parsed.

    `exclude_dirs` and `exclude_names` are aligned with `MIST_GAME_EXCLUDE`'s entries: position
    `k` holds entry `k` in the tuple of its kind and an empty string in the other, so a verdict
    can name the entry as `MIST_GAME_EXCLUDE[k]`. Names are lower-cased patterns.
    """

    libraries: tuple[str, ...]
    runtime_dirs: tuple[str, ...]
    exclude_dirs: tuple[str, ...]
    exclude_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Verdict:
    """Whether one process counts as a game, and why.

    A counting verdict names the variable and the index of the entry it counted under, and for
    a shared runtime the folder it counted for. `reason` is `library`, `runtime`, `outside`,
    `shipped name <pattern>`, `shipped folder <folder>`, `MIST_GAME_EXCLUDE[<k>]`,
    `no runtime folder`, `installer` or `folder unreadable`.
    """

    counts: bool
    variable: str | None
    index: int | None
    folder: str | None
    reason: str


def _normalized(path: str) -> str:
    return ntpath.normcase(ntpath.normpath(path))


def _is_under(path: str, directory: str) -> bool:
    r"""Whether `path` is `directory` or lies inside it, at a directory boundary.

    Case-insensitive: `X:\Games2\x.exe` is not under `X:\Games`, and a drive root such as
    `X:\` holds everything on its drive.
    """
    base = _normalized(directory)
    candidate = _normalized(path)
    prefix = base if base.endswith("\\") else base + "\\"
    return candidate == base or candidate.startswith(prefix)


def lowest_index(path: str, directories: Sequence[str]) -> int | None:
    """The index of the first of `directories` that `path` lies under, or None.

    When entries nest, the lower index wins, so a log line names one entry for one folder.
    Empty entries, the placeholders in `GameRules.exclude_dirs`, never match.
    """
    for index, directory in enumerate(directories):
        if directory and _is_under(path, directory):
            return index
    return None


def is_shared_runtime(image: str) -> bool:
    return ntpath.basename(image).lower() in SHARED_RUNTIMES


def _runtime_folder(record: ProcessRecord, runtime_dirs: Sequence[str]) -> tuple[str, int] | None:
    """The folder a shared runtime runs for and the runtime entry it lies under, or None.

    The folder is the value of `--gameDir` when that lies under a runtime entry, or else the
    working directory. A relative `--gameDir` is joined to the working directory, and ignored
    when that is unknown.
    """
    candidates: list[str] = []
    game_dir = _game_dir(record.args or ())
    if game_dir is not None:
        if ntpath.isabs(game_dir):
            candidates.append(game_dir)
        elif record.cwd is not None:
            candidates.append(ntpath.join(record.cwd, game_dir))
    if record.cwd is not None:
        candidates.append(record.cwd)
    for folder in candidates:
        index = lowest_index(folder, runtime_dirs)
        if index is not None:
            return folder, index
    return None


def _game_dir(args: Sequence[str]) -> str | None:
    """The first `--gameDir` value: the argument after one equal to it, ignoring case, or the
    text after `--gameDir=`.
    """
    for position, arg in enumerate(args):
        lowered = arg.lower()
        if lowered == "--gamedir":
            if position + 1 < len(args):
                return args[position + 1]
            return None
        if lowered.startswith("--gamedir="):
            return arg[len("--gameDir=") :]
    return None


def _runs_installer(args: Sequence[str]) -> bool:
    """Whether the arguments run an installer.

    That is an argument that, lower-cased with every leading `-` removed, starts with
    `install` (`--installServer`), or a `-jar` operand named like `*installer*.jar`.
    """
    for position, arg in enumerate(args):
        if arg.lower().lstrip("-").startswith("install"):
            return True
        if arg.lower() == "-jar" and position + 1 < len(args):
            operand = ntpath.basename(args[position + 1].replace("/", "\\")).lower()
            if fnmatch.fnmatchcase(operand, "*installer*.jar"):
                return True
    return False


def _rule_one(record: ProcessRecord, rules: GameRules) -> Verdict:
    """Rule 1 for a shared runtime: the counting verdict, or why it does not count."""
    if record.args is None and record.cwd is None:
        return _not_counted("folder unreadable")
    found = _runtime_folder(record, rules.runtime_dirs)
    if found is None:
        return _not_counted("no runtime folder")
    folder, index = found
    excluded = lowest_index(folder, rules.exclude_dirs)
    if excluded is not None:
        return _not_counted(f"{EXCLUDE_KEY}[{excluded}]")
    if _runs_installer(record.args or ()):
        return _not_counted("installer")
    return Verdict(True, RUNTIME_DIRS_KEY, index, folder, "runtime")


def _not_counted(reason: str) -> Verdict:
    return Verdict(False, None, None, None, reason)


def _contains_run(parts: Sequence[str], run: Sequence[str]) -> bool:
    width = len(run)
    return any(list(parts[i : i + width]) == list(run) for i in range(len(parts) - width + 1))


def _exclusion(image: str, rules: GameRules) -> str | None:
    r"""The first exclusion that matches a library program, as its reason, or None.

    Shipped folders in table order come first, then shipped names, then `MIST_GAME_EXCLUDE`
    in its order. A shipped folder is matched against the image's whole path, because one such
    as `epic games\launcher` spans a library's own name.
    """
    normalized = _normalized(image)
    folders = ntpath.dirname(normalized).split("\\")
    name = ntpath.basename(normalized)
    for folder in SHIPPED_EXCLUDED_FOLDERS:
        if _contains_run(folders, folder.split("\\")):
            return f"shipped folder {folder}"
    for pattern in SHIPPED_EXCLUDED_NAMES:
        if fnmatch.fnmatchcase(name, pattern):
            return f"shipped name {pattern}"
    for index in range(max(len(rules.exclude_dirs), len(rules.exclude_names))):
        directory = rules.exclude_dirs[index] if index < len(rules.exclude_dirs) else ""
        pattern = rules.exclude_names[index] if index < len(rules.exclude_names) else ""
        if directory and _is_under(image, directory):
            return f"{EXCLUDE_KEY}[{index}]"
        if pattern and fnmatch.fnmatchcase(name, pattern):
            return f"{EXCLUDE_KEY}[{index}]"
    return None


def classify(record: ProcessRecord, rules: GameRules) -> Verdict:
    """Whether `record` is a game's own process under `rules`, and why.

    A shared runtime that rule 1 does not count goes on to rule 2, so a game in a library that
    runs on its own bundled Java counts like any other game there. When its image lies under no
    library, the verdict keeps rule 1's reason. For any other program, `outside` comes before
    any exclusion, so only programs under a library are ever reported as excluded.
    """
    first: Verdict | None = None
    if is_shared_runtime(record.image):
        first = _rule_one(record, rules)
        if first.counts:
            return first
    library = lowest_index(record.image, rules.libraries)
    if library is None:
        return first if first is not None else _not_counted("outside")
    reason = _exclusion(record.image, rules)
    if reason is not None:
        return _not_counted(reason)
    return Verdict(True, GAME_DIRS_KEY, library, None, "library")


def find_game(records: Iterable[ProcessRecord], rules: GameRules) -> Verdict | None:
    """The counting verdict with the lowest (variable, index), `MIST_GAME_DIRS` first.

    A stable choice keeps the log from naming a new entry each tick while two games run. None
    when nothing counts.
    """
    best: Verdict | None = None
    for record in records:
        verdict = classify(record, rules)
        if verdict.counts and (best is None or _rank(verdict) < _rank(best)):
            best = verdict
    return best


def _rank(verdict: Verdict) -> tuple[int, int]:
    return (0 if verdict.variable == GAME_DIRS_KEY else 1, verdict.index or 0)


def decide(
    state: WatchState,
    game: bool,
    marker: str | None,
    now: float,
    resume_delay_s: float,
) -> tuple[Action | None, WatchState]:
    """The command one tick calls for, and the state the next tick starts from.

    `game` is whether `find_game` saw a game this tick. `marker` is the gaming-mode marker's
    `by` (`"manual"` or `"watcher"`), None when there is no marker, and any other string for a
    marker that does not parse. The backend yields to an unparsable marker, so the watcher
    treats it as nobody's to end, as it does a manual one.

    The watcher only ever ends its own gaming mode, and never starts one again while a game
    that someone else ended gaming mode for is still running.
    """
    last_seen = now if game else state.last_seen_game_at
    suppressed = state.suppressed
    action: Action | None = None
    if marker is None:
        if state.marker_was_present:
            suppressed = game
        if game and not suppressed:
            action = "on"
    elif (
        marker == "watcher"
        and not game
        and (last_seen is None or now - last_seen >= resume_delay_s)
    ):
        action = "off"
    if not game:
        suppressed = False
    # The watcher's own `off` removes the marker, so the next tick must not read the missing
    # marker as someone else's `off`.
    present = marker is not None and action != "off"
    return action, replace(
        state, last_seen_game_at=last_seen, marker_was_present=present, suppressed=suppressed
    )


def parse_game_dirs(value: str) -> list[str]:
    """Split the text after one of the folder variables' `=` into entries.

    Surrounding whitespace and one pair of matching outer quotes are stripped, the rest is
    split on `;`, and each entry is stripped, with empty ones dropped. Spaces inside a
    directory stay and backslashes are literal, so the value needs no quotes.
    """
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return [entry.strip() for entry in value.split(";") if entry.strip()]


def _last_values(text: str, keys: Sequence[str]) -> dict[str, str]:
    """The raw value of each key's last uncommented line in `daily.env`."""
    found: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        if key.strip() in keys:
            found[key.strip()] = value
    return found


def is_directory_entry(entry: str) -> bool:
    r"""A `MIST_GAME_EXCLUDE` entry with `\`, `/` or `:` is a directory; any other is a name."""
    return any(char in entry for char in "\\/:")


def rules_from_env(text: str) -> GameRules:
    """The game rules `daily.env`'s last uncommented line of each folder variable gives."""
    raw = _last_values(text, (GAME_DIRS_KEY, RUNTIME_DIRS_KEY, EXCLUDE_KEY))
    exclude = parse_game_dirs(raw.get(EXCLUDE_KEY, ""))
    return GameRules(
        libraries=tuple(parse_game_dirs(raw.get(GAME_DIRS_KEY, ""))),
        runtime_dirs=tuple(parse_game_dirs(raw.get(RUNTIME_DIRS_KEY, ""))),
        exclude_dirs=tuple(entry if is_directory_entry(entry) else "" for entry in exclude),
        exclude_names=tuple(
            "" if is_directory_entry(entry) else entry.lower() for entry in exclude
        ),
    )


def is_64_bit() -> bool:
    """Whether this Python is 64-bit; the working-directory read assumes the x64 layouts."""
    return struct.calcsize("P") == 8


def list_processes() -> list[ProcessRecord]:
    """Every running process this user can open, with a shared runtime's folder evidence.

    A process that cannot be opened, such as one running elevated or as another user, is
    skipped, so a game started as administrator may go unseen. Only a shared runtime is asked
    for its command line and working directory; every failed call leaves that field None.

    Raises:
        GameWatchError: Off Windows, or when `EnumProcesses` itself fails.
    """
    if os.name != "nt":
        raise GameWatchError(f"listing processes needs Windows; this host is {sys.platform}")
    return _WindowsReader().list()


class _WindowsReader:
    """The Win32 and native calls behind `list_processes`, bound once per listing."""

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self.ctypes = ctypes
        self.wintypes = wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        ntdll = ctypes.WinDLL("ntdll")
        shell32 = ctypes.WinDLL("shell32", use_last_error=True)

        class UnicodeString(ctypes.Structure):
            _fields_ = [
                ("Length", wintypes.USHORT),
                ("MaximumLength", wintypes.USHORT),
                ("Buffer", ctypes.c_void_p),
            ]

        class ProcessBasicInformation(ctypes.Structure):
            _fields_ = [
                ("ExitStatus", ctypes.c_long),
                ("PebBaseAddress", ctypes.c_void_p),
                ("AffinityMask", ctypes.c_size_t),
                ("BasePriority", ctypes.c_long),
                ("UniqueProcessId", ctypes.c_size_t),
                ("InheritedFromUniqueProcessId", ctypes.c_size_t),
            ]

        self.UnicodeString = UnicodeString
        self.ProcessBasicInformation = ProcessBasicInformation

        self.enum_processes = psapi.EnumProcesses
        self.enum_processes.argtypes = [
            ctypes.POINTER(wintypes.DWORD),
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.enum_processes.restype = wintypes.BOOL
        self.open_process = kernel32.OpenProcess
        self.open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.open_process.restype = wintypes.HANDLE
        self.query_image = kernel32.QueryFullProcessImageNameW
        self.query_image.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.query_image.restype = wintypes.BOOL
        self.close_handle = kernel32.CloseHandle
        self.close_handle.argtypes = [wintypes.HANDLE]
        self.close_handle.restype = wintypes.BOOL
        self.is_wow64 = kernel32.IsWow64Process
        self.is_wow64.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
        self.is_wow64.restype = wintypes.BOOL
        self.read_memory = kernel32.ReadProcessMemory
        self.read_memory.argtypes = [
            wintypes.HANDLE,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.read_memory.restype = wintypes.BOOL
        self.local_free = kernel32.LocalFree
        self.local_free.argtypes = [ctypes.c_void_p]
        self.local_free.restype = ctypes.c_void_p
        self.query_information = ntdll.NtQueryInformationProcess
        self.query_information.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.ULONG,
            ctypes.POINTER(wintypes.ULONG),
        ]
        self.query_information.restype = ctypes.c_long
        self.to_argv = shell32.CommandLineToArgvW
        self.to_argv.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
        self.to_argv.restype = ctypes.POINTER(wintypes.LPWSTR)

    def list(self) -> list[ProcessRecord]:
        records: list[ProcessRecord] = []
        buffer = self.ctypes.create_unicode_buffer(MAX_IMAGE_PATH)
        for pid in self._pids():
            if pid == 0:
                continue
            image = self._image(pid, buffer)
            if image is None:
                continue
            if is_shared_runtime(image):
                records.append(ProcessRecord(image, self._cwd(pid), self._args(pid)))
            else:
                records.append(ProcessRecord(image))
        return records

    def _pids(self) -> list[int]:
        ctypes, wintypes = self.ctypes, self.wintypes
        slots = INITIAL_PID_SLOTS
        while True:
            pids = (wintypes.DWORD * slots)()
            needed = wintypes.DWORD()
            if not self.enum_processes(pids, ctypes.sizeof(pids), ctypes.byref(needed)):
                raise GameWatchError(f"EnumProcesses failed with error {ctypes.get_last_error()}")
            # A full buffer may have cut the list short, so grow it and ask again.
            if needed.value < ctypes.sizeof(pids):
                break
            slots *= 2
        count = needed.value // ctypes.sizeof(wintypes.DWORD)
        return list(pids[:count])

    def _image(self, pid: int, buffer) -> str | None:
        handle = self.open_process(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return None
        try:
            size = self.wintypes.DWORD(MAX_IMAGE_PATH)
            if self.query_image(handle, 0, buffer, self.ctypes.byref(size)):
                return buffer.value
            return None
        finally:
            self.close_handle(handle)

    def _args(self, pid: int) -> tuple[str, ...] | None:
        """The arguments after the program, from `ProcessCommandLineInformation` (8.1+)."""
        ctypes, wintypes = self.ctypes, self.wintypes
        handle = self.open_process(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return None
        try:
            needed = wintypes.ULONG(0)
            status = self.query_information(
                handle, PROCESS_COMMAND_LINE_INFORMATION_CLASS, None, 0, ctypes.byref(needed)
            )
            if status != STATUS_INFO_LENGTH_MISMATCH or needed.value < ctypes.sizeof(
                self.UnicodeString
            ):
                return None
            raw = ctypes.create_string_buffer(needed.value)
            status = self.query_information(
                handle,
                PROCESS_COMMAND_LINE_INFORMATION_CLASS,
                raw,
                needed.value,
                ctypes.byref(needed),
            )
            if status < 0:
                return None
            header = self.UnicodeString.from_buffer(raw)
            if not header.Buffer or header.Length % 2:
                return None
            start = ctypes.addressof(raw)
            if not start <= header.Buffer <= start + len(raw) - header.Length:
                return None
            command_line = ctypes.wstring_at(header.Buffer, header.Length // 2)
        finally:
            self.close_handle(handle)
        return self._split(command_line)

    def _split(self, command_line: str) -> tuple[str, ...] | None:
        ctypes = self.ctypes
        if not command_line:
            return None
        count = ctypes.c_int(0)
        argv = self.to_argv(command_line, ctypes.byref(count))
        if not argv:
            return None
        try:
            return tuple(argv[index] for index in range(1, count.value))
        finally:
            self.local_free(ctypes.cast(argv, ctypes.c_void_p))

    def _cwd(self, pid: int) -> str | None:
        """The working directory, read from the target's x64 PEB, or None.

        A 32-bit target, or a 32-bit Python, gives None: the offsets are the x64 layouts.
        """
        ctypes, wintypes = self.ctypes, self.wintypes
        if not is_64_bit():
            return None
        handle = self.open_process(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, False, pid)
        if not handle:
            return None
        try:
            wow64 = wintypes.BOOL(False)
            if not self.is_wow64(handle, ctypes.byref(wow64)) or wow64.value:
                return None
            basic = self.ProcessBasicInformation()
            status = self.query_information(
                handle,
                PROCESS_BASIC_INFORMATION_CLASS,
                ctypes.byref(basic),
                ctypes.sizeof(basic),
                None,
            )
            if status < 0 or not basic.PebBaseAddress:
                return None
            parameters = ctypes.c_void_p()
            if not self._read(
                handle, basic.PebBaseAddress + PEB_PROCESS_PARAMETERS_OFFSET, parameters
            ):
                return None
            if not parameters.value:
                return None
            dos_path = self.UnicodeString()
            if not self._read(
                handle, parameters.value + PARAMETERS_CURRENT_DIRECTORY_OFFSET, dos_path
            ):
                return None
            length = dos_path.Length
            if (
                not dos_path.Buffer
                or length == 0
                or length % 2
                or length > dos_path.MaximumLength
                or length > MAX_UNICODE_STRING_BYTES
            ):
                return None
            text = ctypes.create_unicode_buffer(length // 2 + 1)
            if not self._read_bytes(handle, dos_path.Buffer, text, length):
                return None
            return text.value or None
        finally:
            self.close_handle(handle)

    def _read(self, handle, address: int, target) -> bool:
        return self._read_bytes(handle, address, target, self.ctypes.sizeof(target))

    def _read_bytes(self, handle, address: int, target, size: int) -> bool:
        ctypes = self.ctypes
        done = ctypes.c_size_t(0)
        ok = self.read_memory(handle, address, ctypes.byref(target), size, ctypes.byref(done))
        return bool(ok) and done.value == size
