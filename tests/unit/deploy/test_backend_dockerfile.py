"""Layer order and release identity of `docker/backend/Dockerfile`.

Hermetic: parses the Dockerfile's instructions only; no docker build. The image is tens of GB, so
a layer that rewrites the venv after every code change costs a full copy of it per release. The
venv is therefore handed to `appuser` while it is empty, every `pip install` runs as `appuser`,
the code is copied already owned by `appuser`, and only the runtime directories are chowned, by
name and without `-R`. The release tag and commit come last, after `CMD`, so a new tag
invalidates only metadata layers.

The interpreter is a python-build-standalone CPython archive pinned by release tag and sha256, not
Ubuntu's apt python3.11, which is the release candidate 3.11.0rc1. The archive is verified before
it is unpacked, the build checks the interpreter it unpacked, and `/opt/python/bin` goes first on
PATH before the venv is created, so that the venv's `home` is the pinned interpreter.

Each test fails on the edit its name describes, for example a `chown -R` reintroduced after the
code copies or a runtime directory dropped from the final `RUN`.
"""

import re
import shlex
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE_PATH = REPO_ROOT / "docker" / "backend" / "Dockerfile"

APP_USER = "appuser"
OWNER = "appuser:appgroup"
VENV = "/opt/venv"
CODE_SOURCES = ("requirements.txt", "backend/", "src/", "dependencies/", "scripts/")
# Every mount point and unmounted path the backend writes, so each stays writable by appuser:
# /app/dev-state is a named volume Docker initialises from the image directory's ownership.
RUNTIME_DIRECTORIES = (
    "/app",
    "/app/data",
    "/app/data/voice_profiles",
    "/app/data/vector_store",
    "/app/data/audio",
    "/app/logs",
    "/app/mist-memory",
    "/app/ops",
    "/app/dev-state",
)
RELEASE_LINES = [
    ("ARG", "MIST_RELEASE_TAG=unreleased"),
    ("ARG", "MIST_RELEASE_COMMIT=unknown"),
    ("ENV", "MIST_RELEASE_TAG=${MIST_RELEASE_TAG}"),
    ("ENV", "MIST_RELEASE_COMMIT=${MIST_RELEASE_COMMIT}"),
    (
        "LABEL",
        "org.opencontainers.image.version=${MIST_RELEASE_TAG} "
        "org.opencontainers.image.revision=${MIST_RELEASE_COMMIT}",
    ),
]
RECURSIVE_CHOWN = re.compile(r"\bchown\s+-R\b")

# The pinned interpreter archive. The three ARG defaults are the values recorded when the archive
# was downloaded and checked; ARCHIVE_URL is the asset URL it was downloaded from.
PYTHON_VERSION = "3.11.18"
PYTHON_PREFIX = "/opt/python"
INTERPRETER = f"{PYTHON_PREFIX}/bin/python3.11"
ARCHIVE = "/tmp/python.tar.gz"
ARCHIVE_URL_PREFIX = "https://github.com/astral-sh/python-build-standalone/releases/download/"
ARCHIVE_URL_TEMPLATE = (
    ARCHIVE_URL_PREFIX + "${PYTHON_BUILD_STANDALONE_RELEASE}/cpython-${PYTHON_VERSION}"
    "%2B${PYTHON_BUILD_STANDALONE_RELEASE}-x86_64-unknown-linux-gnu-install_only.tar.gz"
)
ARCHIVE_URL = (
    ARCHIVE_URL_PREFIX + "20261026/cpython-3.11.18%2B20261026-x86_64-unknown-linux-gnu-"
    "install_only.tar.gz"
)
APT_INTERPRETER_PACKAGES = (
    "python3",
    "python3.11",
    "python3.11-venv",
    "python3.11-dev",
    "python3-pip",
)
INTERPRETER_PATH = 'PATH="/opt/python/bin:$PATH"'


def _instructions() -> list[tuple[str, str]]:
    """(INSTRUCTION, arguments) per instruction, continuations joined.

    Comment lines are dropped even inside a continuation, as Docker drops them.
    """
    instructions, pending = [], ""
    for raw in DOCKERFILE_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        keyword, _sep, arguments = (pending + line).partition(" ")
        pending = ""
        instructions.append((keyword.upper(), " ".join(arguments.split())))
    return instructions


def _with_users() -> list[tuple[str, str, str]]:
    """(user, INSTRUCTION, arguments): the user each instruction runs under."""
    user, annotated = "root", []
    for keyword, arguments in _instructions():
        if keyword == "USER":
            user = arguments
        annotated.append((user, keyword, arguments))
    return annotated


def _copy_source(arguments: str) -> str:
    tokens = [token for token in arguments.split() if not token.startswith("--")]
    return tokens[0]


def _code_copy_indexes() -> list[int]:
    return [
        index
        for index, (keyword, arguments) in enumerate(_instructions())
        if keyword == "COPY" and _copy_source(arguments) in CODE_SOURCES
    ]


def _commands(arguments: str) -> list[list[str]]:
    """The `&&`-separated commands of a shell-form RUN, each split into tokens."""
    return [shlex.split(part) for part in arguments.split("&&")]


def _runtime_directory_run() -> tuple[str, str]:
    """(user, arguments) of the one RUN after the code copies that runs mkdir and chown."""
    last_copy = _code_copy_indexes()[-1]
    matches = [
        (user, arguments)
        for index, (user, keyword, arguments) in enumerate(_with_users())
        if index > last_copy
        and keyword == "RUN"
        and "mkdir -p" in arguments
        and "chown" in arguments
    ]
    assert len(matches) == 1, f"expected one mkdir/chown RUN after the code copies: {matches!r}"
    return matches[0]


def _command_arguments(arguments: str, program: str) -> list[str]:
    (command,) = (tokens for tokens in _commands(arguments) if tokens[0] == program)
    return command[1:]


def _arg_defaults() -> dict[str, str]:
    """ARG name to its default, for each ARG that declares one."""
    defaults = {}
    for keyword, arguments in _instructions():
        name, sep, default = arguments.partition("=")
        if keyword == "ARG" and sep:
            defaults[name] = default
    return defaults


def _interpreter_run() -> tuple[int, str, str]:
    """(index, user, arguments) of the one RUN that downloads the interpreter archive."""
    matches = [
        (index, user, arguments)
        for index, (user, keyword, arguments) in enumerate(_with_users())
        if keyword == "RUN" and ARCHIVE_URL_PREFIX in arguments
    ]
    assert len(matches) == 1, f"expected one RUN that downloads the interpreter: {matches!r}"
    return matches[0]


def _first_index(commands: list[list[str]], matches) -> int:
    (index, *_rest) = (i for i, tokens in enumerate(commands) if matches(tokens))
    return index


class TestInterpreter:
    def test_apt_installs_no_interpreter(self):
        apt_runs = [
            arguments
            for keyword, arguments in _instructions()
            if keyword == "RUN" and "apt-get install" in arguments
        ]
        installed = [
            token for arguments in apt_runs for tokens in _commands(arguments) for token in tokens
        ]

        assert apt_runs, "the Dockerfile installs nothing with apt-get"
        assert [package for package in APT_INTERPRETER_PACKAGES if package in installed] == []
        assert [
            (keyword, arguments)
            for keyword, arguments in _instructions()
            if "update-alternatives" in arguments or "/usr/bin/python3.11" in arguments
        ] == []

    def test_interpreter_is_a_pinned_final_release(self):
        defaults = _arg_defaults()

        assert defaults.get("PYTHON_VERSION") == PYTHON_VERSION
        assert re.fullmatch(r"\d{8}", defaults.get("PYTHON_BUILD_STANDALONE_RELEASE", ""))
        assert re.fullmatch(r"[0-9a-f]{64}", defaults.get("PYTHON_ARCHIVE_SHA256", ""))

    def test_archive_is_verified_before_it_is_unpacked(self):
        _index, _user, arguments = _interpreter_run()
        commands = _commands(arguments)

        download = _first_index(
            commands,
            lambda tokens: tokens[0] == "curl"
            and "--fail" in tokens
            and ARCHIVE_URL_TEMPLATE in tokens
            and tokens[tokens.index("-o") + 1] == ARCHIVE,
        )
        check = _first_index(
            commands,
            lambda tokens: "sha256sum" in tokens
            and "--check" in tokens[tokens.index("sha256sum") :]
            and ["${PYTHON_ARCHIVE_SHA256}", ARCHIVE] in [token.split() for token in tokens],
        )
        unpack = _first_index(
            commands,
            lambda tokens: tokens[0] == "tar"
            and "--no-same-owner" in tokens
            and ARCHIVE in tokens
            and tokens[tokens.index("-C") + 1] == PYTHON_PREFIX,
        )
        remove = _first_index(commands, lambda tokens: tokens == ["rm", ARCHIVE])

        assert download < check < unpack < remove, commands
        defaults = _arg_defaults()
        rendered = ARCHIVE_URL_TEMPLATE
        for name in ("PYTHON_VERSION", "PYTHON_BUILD_STANDALONE_RELEASE"):
            rendered = rendered.replace("${" + name + "}", defaults[name])
        assert rendered == ARCHIVE_URL

    def test_build_checks_the_interpreter(self):
        _index, _user, arguments = _interpreter_run()
        (script,) = (
            tokens[2] for tokens in _commands(arguments) if tokens[:2] == [INTERPRETER, "-c"]
        )

        for name in (
            "${PYTHON_VERSION}",
            "releaselevel",
            "get_default_verify_paths",
            "enable_load_extension",
            "fts5",
        ):
            assert name in script, name
        compile(script.replace("${PYTHON_VERSION}", PYTHON_VERSION), "<build check>", "exec")

    def test_interpreter_precedes_the_venv(self):
        index, user, _arguments = _interpreter_run()
        instructions = _instructions()
        (path,) = (
            i
            for i, (keyword, arguments) in enumerate(instructions)
            if keyword == "ENV" and arguments == INTERPRETER_PATH
        )
        (venv,) = (
            i
            for i, (keyword, arguments) in enumerate(instructions)
            if keyword == "RUN" and f"python -m venv {VENV}" in arguments
        )

        assert user == "root"
        assert index < path < venv


class TestVenvOwnership:
    def test_venv_is_chowned_in_the_run_that_creates_it(self):
        (creating,) = (
            arguments
            for keyword, arguments in _instructions()
            if keyword == "RUN" and f"python -m venv {VENV}" in arguments
        )
        assert f"chown -R {OWNER} {VENV}" in creating

    def test_that_run_holds_the_only_recursive_venv_chown(self):
        recursive = [
            arguments
            for keyword, arguments in _instructions()
            if keyword == "RUN" and RECURSIVE_CHOWN.search(arguments) and VENV in arguments
        ]
        assert len(recursive) == 1 and "python -m venv" in recursive[0], recursive

    def test_every_pip_install_runs_as_appuser(self):
        installs = [
            (user, arguments)
            for user, keyword, arguments in _with_users()
            if keyword == "RUN" and "pip install" in arguments
        ]

        assert installs, "the Dockerfile installs nothing with pip"
        assert [arguments for user, arguments in installs if user != APP_USER] == []


class TestCodeOwnership:
    @pytest.mark.parametrize("source", CODE_SOURCES)
    def test_copy_carries_the_appuser_owner(self, source):
        copies = [
            arguments
            for keyword, arguments in _instructions()
            if keyword == "COPY" and _copy_source(arguments) == source
        ]

        assert copies, f"{source} is never copied"
        assert all(f"--chown={OWNER}" in arguments.split() for arguments in copies), copies

    def test_no_recursive_chown_follows_the_first_code_copy(self):
        first_copy = _code_copy_indexes()[0]
        later = [
            arguments
            for keyword, arguments in _instructions()[first_copy:]
            if keyword == "RUN" and RECURSIVE_CHOWN.search(arguments)
        ]
        assert later == []


class TestRuntimeDirectories:
    def test_created_exactly(self):
        _user, arguments = _runtime_directory_run()
        tokens = _command_arguments(arguments, "mkdir")

        assert tokens[0] == "-p"
        assert sorted(tokens[1:]) == sorted(RUNTIME_DIRECTORIES)

    def test_chowned_exactly_and_without_recursion(self):
        _user, arguments = _runtime_directory_run()
        tokens = _command_arguments(arguments, "chown")

        assert not [token for token in tokens if token.startswith("-")], tokens
        assert tokens[0] == OWNER
        assert sorted(tokens[1:]) == sorted(RUNTIME_DIRECTORIES)

    @pytest.mark.parametrize("directory", RUNTIME_DIRECTORIES)
    def test_directory_is_created_and_chowned(self, directory):
        _user, arguments = _runtime_directory_run()

        assert directory in _command_arguments(arguments, "mkdir")
        assert directory in _command_arguments(arguments, "chown")

    def test_run_executes_as_root(self):
        user, _arguments = _runtime_directory_run()
        assert user == "root", "only root can hand the runtime directories to appuser"


class TestFinalImage:
    def test_ends_as_appuser_with_its_home(self):
        annotated = _with_users()
        last_user = max(i for i, (_u, keyword, _a) in enumerate(annotated) if keyword == "USER")

        assert annotated[-1][0] == APP_USER
        assert ("ENV", "HOME=/home/appuser") in [(k, a) for _u, k, a in annotated[last_user:]]

    def test_release_lines_follow_cmd(self):
        instructions = _instructions()
        (cmd_index,) = (i for i, (keyword, _a) in enumerate(instructions) if keyword == "CMD")

        assert instructions[cmd_index + 1 :] == RELEASE_LINES
