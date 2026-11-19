# SPDX-License-Identifier: MIT
# Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.
"""Build the room endpoint's portable core and its host tests with g++.

The core (`firmware/room-endpoint/components/mist_core/`) is plain C++17 that also builds for the
device, where ESP-IDF's default C++ configuration disables exceptions and RTTI, so the host build
disables both too and adds AddressSanitizer and UndefinedBehaviorSanitizer. Every artifact goes to
a pytest temporary directory, never into the repository.

Without `g++` the tests skip with a reason, except when `CI` is set: a CI runner that lacks the
compiler would otherwise pass while testing nothing, so there the tests fail instead.
"""

import os
import shutil
import signal
import subprocess
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_DIR = REPO_ROOT / "firmware" / "room-endpoint" / "components" / "mist_core"
CPP_DIR = Path(__file__).resolve().parent / "cpp"
VECTORS = REPO_ROOT / "tests" / "fixtures" / "realtime_protocol_v2" / "vectors.json"

CXX_FLAGS = (
    "-std=c++17",
    "-Wall",
    "-Wextra",
    "-Werror",
    "-fsanitize=address,undefined",
    "-fno-sanitize-recover=all",
    "-fno-exceptions",
    "-fno-rtti",
    "-pthread",
    "-g",
    "-O1",
)

# One binary per suite, each linked with the whole core.
SUITES = ("codec", "ring", "uplink", "tag_queue", "alarm_rules", "ui_state")
CONTRACT = "contract"

# In the worker container the GCC 11.4 sanitizer runtime sometimes faults during its start-up,
# before `main`, and its own SEGV handler then faults again in an endless
# `AddressSanitizer:DEADLYSIGNAL` loop that fills the captured output until the run is killed for
# memory. The suspected cause, unverified (the container cannot read `vm.mmap_rnd_bits`), is a
# kernel with more mmap randomness than this runtime's memory layout allows. `handle_segv=0`
# removes the handler, so such a process ends at once with SIGSEGV instead of looping.
SANITIZER_ENV = {
    "ASAN_OPTIONS": "abort_on_error=0:halt_on_error=1:detect_leaks=1:handle_segv=0",
    "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1:handle_segv=0",
}

# The first line every test binary prints, from the first statement of its `main`.
START_MARKER = "MIST_TEST_START"

# A run that died of SIGSEGV before printing the start marker died in the runtime's start-up and
# is started again, up to this many runs in all. A crash after the marker, in a test case, fails
# at once and is never retried.
STARTUP_ATTEMPTS = 5

MISSING_COMPILER = "g++ is not on PATH, so the firmware core's host tests cannot be built"


def require_gxx(
    which: Callable[[str], str | None] = shutil.which,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Return the path of `g++`, or skip the calling test; fail instead when `CI` is set.

    Args:
        which: Finds an executable on PATH; tests pass a fake.
        environ: The environment to read `CI` from; the process environment when None.
    """
    compiler = which("g++")
    if compiler is not None:
        return compiler
    env = os.environ if environ is None else environ
    if env.get("CI"):
        pytest.fail(f"{MISSING_COMPILER}; CI is set, so a missing compiler is a failure")
    pytest.skip(MISSING_COMPILER)


@dataclass(frozen=True, slots=True)
class CoreBuild:
    """The built test binaries, keyed by suite name, and the directory they were built in."""

    directory: Path
    binaries: dict[str, Path]

    def run(self, name: str, *args: str) -> subprocess.CompletedProcess[str]:
        """Run one built binary with the sanitizers set to stop at the first error."""
        env = {**os.environ, **SANITIZER_ENV, "MIST_PROTOCOL_VECTORS": str(VECTORS)}
        return run_binary([str(self.binaries[name]), *args], env)


Launcher = Callable[..., subprocess.CompletedProcess[str]]


def faulted_at_startup(result: subprocess.CompletedProcess[str]) -> bool:
    """Whether a run died of SIGSEGV before `main` printed the start marker."""
    return result.returncode == -signal.SIGSEGV and START_MARKER not in result.stdout


def run_binary(
    command: list[str], env: Mapping[str, str], launch: Launcher = subprocess.run
) -> subprocess.CompletedProcess[str]:
    """Run `command`, starting it again when it faults before `main`, at most `STARTUP_ATTEMPTS`.

    Args:
        command: The binary and its arguments.
        env: The environment it runs with.
        launch: Runs one process; tests pass a fake.
    """
    result = launch(
        command, capture_output=True, text=True, timeout=120, env=dict(env), check=False
    )
    for _ in range(STARTUP_ATTEMPTS - 1):
        if not faulted_at_startup(result):
            break
        result = launch(
            command, capture_output=True, text=True, timeout=120, env=dict(env), check=False
        )
    return result


def _compile(compiler: str, args: list[str]) -> None:
    result = subprocess.run(
        [compiler, *args], capture_output=True, text=True, timeout=300, check=False
    )
    if result.returncode != 0:
        raise AssertionError(f"g++ {' '.join(args)}\n{result.stdout}{result.stderr}")


def build_core(compiler: str, out_dir: Path) -> CoreBuild:
    """Compile the core once, then link each suite and the contract driver against it."""
    include = [f"-I{CORE_DIR}", f"-I{CPP_DIR}"]
    sources = sorted(CORE_DIR.glob("*.cpp"))
    objects = [out_dir / f"core_{source.stem}.o" for source in sources]
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 2) as pool:
        list(
            pool.map(
                lambda pair: _compile(
                    compiler, [*CXX_FLAGS, *include, "-c", str(pair[0]), "-o", str(pair[1])]
                ),
                zip(sources, objects),
            )
        )
        names = [*SUITES, CONTRACT]
        binaries = {name: out_dir / f"{name}_bin" for name in names}

        def link(name: str) -> None:
            source = CPP_DIR / ("contract_main.cpp" if name == CONTRACT else f"test_{name}.cpp")
            _compile(
                compiler,
                [*CXX_FLAGS, *include, str(source), *map(str, objects), "-o", str(binaries[name])],
            )

        list(pool.map(link, names))
    return CoreBuild(directory=out_dir, binaries=binaries)


@pytest.fixture(scope="session")
def core_build(tmp_path_factory: pytest.TempPathFactory) -> CoreBuild:
    """The core and its host tests, built once per session into a temporary directory."""
    compiler = require_gxx()
    return build_core(compiler, tmp_path_factory.mktemp("mist_core_build"))
