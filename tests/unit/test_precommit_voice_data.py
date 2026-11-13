"""Guard: voice data never enters this public repository.

Four `repo: local` pre-commit hooks keep it out: three `language: fail` hooks
refuse voice media, voice assets and voice-data records by path, and
`no-voice-private-terms` runs `scripts/check_voice_private_terms.py`, which
refuses any file whose path or text holds a term from the private root's
`guard/terms.txt`. The same script's `--range` mode checks unpushed history
before a push of the integration branch.

The hooks' text must equal the plan's Interfaces section exactly, because the
realtime-voice goal's test (`tests/unit/test_precommit_no_audio.py`) checks
the same two shared hook ids against that text.

Every planted term below is a made-up canary; no test reads a real list.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / ".pre-commit-config.yaml"
GITIGNORE_PATH = REPO_ROOT / ".gitignore"
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_voice_private_terms.py"

INTERFACES = {
    "no-audio-files": {
        "name": "No audio or video (voice data lives in the private root)",
        "entry": (
            "audio is not committed to this repository (it is public); video is not either; "
            "keep voice data under MIST_VOICE_PRIVATE_ROOT"
        ),
        "language": "fail",
        "files": (
            r"(?i)\.(wav|flac|mp3|ogg|opus|m4a|aac|ac3|eac3|dts|aif|aiff|wma|mka|mkv|mp4|m4v"
            r"|mov|avi|webm|wmv|flv|m2ts|vob)$"
        ),
    },
    "no-voice-assets": {
        "name": "No voice identities, voice-data manifests, arrays or model weights",
        "entry": (
            "voice identities and model weights are not committed to this repository "
            "(it is public); keep them under MIST_VOICE_PRIVATE_ROOT"
        ),
        "language": "fail",
        "files": (
            r"(?i)((^|/)(voice|student|target)\.json$|\.(npy|npz|pt|pth|bin|ckpt|onnx"
            r"|safetensors|nemo|gguf|pkl|h5|hdf5|parquet|arrow)$)"
        ),
    },
    "no-voice-data-records": {
        "name": "No private voice-data records, subtitles, alignments or archives",
        "entry": (
            "voice-data records are not committed to this public repository; keep them under "
            "MIST_VOICE_PRIVATE_ROOT"
        ),
        "language": "fail",
        "files": r"(?i)(\.private\.|\.(srt|vtt|ass|rttm|textgrid|lab|tar|tgz|zip|7z)$)",
    },
    "no-voice-private-terms": {
        "name": "No private voice-data terms",
        "entry": "python scripts/check_voice_private_terms.py",
        "language": "system",
    },
}

FAIL_HOOKS = ("no-audio-files", "no-voice-assets", "no-voice-data-records")

SUFFIXES = {
    "no-audio-files": (
        "wav flac mp3 ogg opus m4a aac ac3 eac3 dts aif aiff wma mka mkv mp4 m4v mov avi "
        "webm wmv flv m2ts vob"
    ).split(),
    "no-voice-assets": (
        "npy npz pt pth bin ckpt onnx safetensors nemo gguf pkl h5 hdf5 parquet arrow"
    ).split(),
    "no-voice-data-records": "srt vtt ass rttm textgrid lab tar tgz zip 7z".split(),
}

JSON_NAMES = ("voice.json", "student.json", "target.json")

NO_HOOK_MATCHES = (
    "x.py",
    "voice_json.py",
    "voice.jsonl",
    "README.md",
    "scripts/model_bench/speech/speech_arms.json",
    "onnx_asr.py",
    "wave.py",
    "voice.py",
)

# Made-up canaries: the private list's terms in these tests.
TERM = "zqxcanary"
UMLAUT_TERM = "zqxümlaut"
TERMS_TXT = f"# zqxcomment\nab\nZqxCanary\n{UMLAUT_TERM}\n"
TERM_NO = 3
UMLAUT_NO = 4


# --- the hook definitions -------------------------------------------------


def _all_hooks() -> list[dict]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return [hook for repo in config["repos"] for hook in repo["hooks"]]


def _hook(hook_id: str) -> dict:
    (hook,) = (hook for hook in _all_hooks() if hook["id"] == hook_id)
    return hook


def _matches(hook_id: str, path: str) -> bool:
    return re.search(_hook(hook_id)["files"], path) is not None


def test_the_four_hooks_live_in_the_local_repo_block() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    (local,) = (repo for repo in config["repos"] if repo["repo"] == "local")
    ids = [hook["id"] for hook in local["hooks"]]
    assert ids[ids.index("check-ai-slop") + 1 :] == list(INTERFACES)


@pytest.mark.parametrize("hook_id", list(INTERFACES))
def test_exactly_one_hook_per_id(hook_id: str) -> None:
    assert [hook["id"] for hook in _all_hooks()].count(hook_id) == 1


@pytest.mark.parametrize("hook_id", FAIL_HOOKS)
def test_fail_hooks_have_no_exclude_and_a_compiling_regex(hook_id: str) -> None:
    hook = _hook(hook_id)
    assert hook["language"] == "fail"
    assert "exclude" not in hook
    assert "types" not in hook
    re.compile(hook["files"])


@pytest.mark.parametrize("hook_id", list(INTERFACES))
def test_hook_text_equals_interfaces(hook_id: str) -> None:
    hook = dict(_hook(hook_id))
    del hook["id"]
    assert hook == INTERFACES[hook_id]


def test_entries_carry_the_words_the_realtime_test_checks() -> None:
    assert "audio" in _hook("no-audio-files")["entry"]
    assert "voice identities" in _hook("no-voice-assets")["entry"]
    assert "model weights" in _hook("no-voice-assets")["entry"]


def test_terms_hook_is_system_with_no_type_filter() -> None:
    hook = _hook("no-voice-private-terms")
    assert hook["language"] == "system"
    assert "types" not in hook
    assert "types_or" not in hook
    assert "exclude" not in hook


@pytest.mark.parametrize(
    ("hook_id", "path"),
    [
        (hook_id, f"{stem}.{suffix}")
        for hook_id in FAIL_HOOKS
        for suffix in SUFFIXES[hook_id]
        for stem in ("x", "deep/nested/dir/x")
    ]
    + [(hook_id, f"x.{suffix.upper()}") for hook_id in FAIL_HOOKS for suffix in SUFFIXES[hook_id]]
    + [
        (hook_id, f"a/b/X.{suffix.upper()}")
        for hook_id in FAIL_HOOKS
        for suffix in SUFFIXES[hook_id]
    ],
)
def test_every_suffix_matches_lower_upper_and_nested(hook_id: str, path: str) -> None:
    assert _matches(hook_id, path)


@pytest.mark.parametrize(
    "path",
    [f"{prefix}{name}" for name in JSON_NAMES for prefix in ("", "a/", "a/b/c/")]
    + [f"a/{name.upper()}" for name in JSON_NAMES],
)
def test_voice_json_names_match_at_any_depth(path: str) -> None:
    assert _matches("no-voice-assets", path)


@pytest.mark.parametrize("path", ["x.private.json", "a/b/notes.private.txt", "X.PRIVATE.md"])
def test_private_infix_matches(path: str) -> None:
    assert _matches("no-voice-data-records", path)


@pytest.mark.parametrize(
    ("hook_id", "path"),
    [
        ("no-audio-files", "x.WAV"),
        ("no-audio-files", "x.webm"),
        ("no-audio-files", "data/voice_profiles/sample/a.wav"),
        ("no-voice-assets", "voices/mist-v1/voice.json"),
        ("no-voice-assets", "x.SAFETENSORS"),
        ("no-voice-assets", "a/b.nemo"),
    ],
)
def test_realtime_goal_must_match_cases(hook_id: str, path: str) -> None:
    assert _matches(hook_id, path)


@pytest.mark.parametrize(
    ("hook_id", "path"),
    [
        ("no-voice-assets", "a.wav"),
        ("no-voice-assets", "myvoice.json"),
        ("no-audio-files", "wav.py"),
        ("no-audio-files", "x.wav.md"),
        ("no-audio-files", "docs/audio.md"),
    ],
)
def test_realtime_goal_must_not_match_cases(hook_id: str, path: str) -> None:
    assert not _matches(hook_id, path)


@pytest.mark.parametrize("path", NO_HOOK_MATCHES)
def test_ordinary_files_match_no_fail_hook(path: str) -> None:
    assert not any(_matches(hook_id, path) for hook_id in FAIL_HOOKS)


def test_every_pattern_is_in_gitignore() -> None:
    lines = {line.strip() for line in GITIGNORE_PATH.read_text(encoding="utf-8").splitlines()}
    assert "# Voice data never enters this public repository" in lines
    wanted = {f"*.{suffix}" for hook_id in FAIL_HOOKS for suffix in SUFFIXES[hook_id]}
    wanted |= {*JSON_NAMES, "*.private.*", "models/*.gguf"}
    assert wanted - lines == set()


# --- the terms script -----------------------------------------------------


def _git_env(home: Path) -> dict[str, str]:
    # Inherited GIT_DIR and friends would point every tmp repository at the host checkout.
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.pop("MIST_VOICE_PRIVATE_ROOT", None)
    env.update(
        HOME=str(home),
        GIT_CONFIG_NOSYSTEM="1",
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@example.invalid",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@example.invalid",
    )
    return env


def _git(cwd: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-c", "init.defaultBranch=main", "-c", "commit.gpgsign=false", *args],
        cwd=cwd,
        env=_git_env(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(result.stderr)
    return result.stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--allow-empty", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _private_root(tmp_path: Path, with_list: bool = True) -> Path:
    root = tmp_path / "private"
    (root / "guard").mkdir(parents=True, exist_ok=True)
    if with_list:
        (root / "guard" / "terms.txt").write_text(TERMS_TXT, encoding="utf-8")
    return root


def _run(args: list[str], cwd: Path, root: Path | None) -> subprocess.CompletedProcess[str]:
    env = _git_env(cwd)
    if root is not None:
        env["MIST_VOICE_PRIVATE_ROOT"] = str(root)
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _output(result: subprocess.CompletedProcess[str]) -> str:
    out = result.stdout + result.stderr
    assert TERM not in out.casefold()
    assert UMLAUT_TERM not in out.casefold()
    return out


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-q")
    (path / "base.txt").write_text("base\n", encoding="utf-8")
    _commit(path, "base")
    return path


def test_no_list_skips_in_file_mode(tmp_path: Path, repo: Path) -> None:
    (repo / "notes.txt").write_text(f"{TERM}\n", encoding="utf-8")
    result = _run(["notes.txt"], repo, _private_root(tmp_path, with_list=False))
    assert result.returncode == 0
    assert result.stdout.strip() == "voice private terms: no list, skipped"


def test_hit_prints_path_and_line_not_the_term(tmp_path: Path, repo: Path) -> None:
    (repo / "notes.txt").write_text(f"clean\nsay {TERM.upper()} here\n", encoding="utf-8")
    result = _run(["notes.txt"], repo, _private_root(tmp_path))
    assert result.returncode == 1
    assert _output(result).strip() == f"notes.txt:2: matches private term #{TERM_NO}"


def test_miss_passes_quietly(tmp_path: Path, repo: Path) -> None:
    (repo / "notes.txt").write_text("nothing private\n", encoding="utf-8")
    result = _run(["notes.txt"], repo, _private_root(tmp_path))
    assert (result.returncode, result.stdout) == (0, "")


def test_short_terms_and_comment_lines_are_not_terms(tmp_path: Path, repo: Path) -> None:
    (repo / "notes.txt").write_text("ab\nabc ab\n# zqxcomment\n", encoding="utf-8")
    result = _run(["notes.txt"], repo, _private_root(tmp_path))
    assert (result.returncode, result.stdout) == (0, "")


def test_undecodable_content_is_skipped(tmp_path: Path, repo: Path) -> None:
    (repo / "blob.dat").write_bytes(b"\xff\xfe\x00" + TERM.encode() + b"\n")
    result = _run(["blob.dat"], repo, _private_root(tmp_path))
    assert (result.returncode, result.stdout) == (0, "")


def test_variable_unset_finds_the_root_beside_the_main_checkout(tmp_path: Path) -> None:
    main = tmp_path / "main"
    main.mkdir()
    _git(main, "init", "-q")
    _commit(main, "base")
    worktree = tmp_path / "elsewhere" / "wt"
    _git(main, "worktree", "add", "-q", str(worktree))
    git_dir = _git(worktree, "rev-parse", "--absolute-git-dir")
    common = _git(worktree, "rev-parse", "--path-format=absolute", "--git-common-dir")
    assert Path(git_dir) != Path(common)
    (worktree / "notes.txt").write_text(f"{TERM}\n", encoding="utf-8")

    assert _run(["notes.txt"], worktree, None).returncode == 0
    guard = tmp_path / "mist-voice" / "guard"
    guard.mkdir(parents=True)
    (guard / "terms.txt").write_text(TERMS_TXT, encoding="utf-8")
    result = _run(["notes.txt"], worktree, None)
    assert result.returncode == 1
    assert _output(result).strip() == f"notes.txt:1: matches private term #{TERM_NO}"


PATH_CASES = {
    "file name": (f"{TERM}.txt", TERM_NO),
    "folder name": (f"{TERM.upper()}/notes.txt", TERM_NO),
    "binary file name": (f"bin/{TERM}.dat", TERM_NO),
    "non-ascii file name": (f"Notes-{UMLAUT_TERM.upper()}.txt", UMLAUT_NO),
}


def _write_case(repo: Path, rel: str) -> None:
    target = repo / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if rel.endswith(".dat"):
        target.write_bytes(b"\x00\xff" + TERM.encode())
    else:
        target.write_text(f"clean\n{TERM}\n", encoding="utf-8")


@pytest.mark.parametrize("case", list(PATH_CASES))
def test_file_mode_withholds_a_matching_path(tmp_path: Path, repo: Path, case: str) -> None:
    rel, number = PATH_CASES[case]
    _write_case(repo, rel)
    (repo / "plain.txt").write_text("fine\n", encoding="utf-8")
    result = _run(["plain.txt", rel], repo, _private_root(tmp_path))
    assert result.returncode == 1
    lines = _output(result).splitlines()
    assert lines[0] == f"file #2 of the batch: path matches private term #{number}"
    if not rel.endswith(".dat"):
        assert lines[1:] == [f"file #2:2: matches private term #{TERM_NO}"]
    else:
        assert lines[1:] == []


def test_file_mode_catches_a_renamed_files_new_name(tmp_path: Path, repo: Path) -> None:
    (repo / "clean.txt").write_text("fine\n", encoding="utf-8")
    _commit(repo, "clean")
    _git(repo, "mv", "clean.txt", f"{TERM}.txt")
    result = _run([f"{TERM}.txt"], repo, _private_root(tmp_path))
    assert result.returncode == 1
    assert _output(result).strip() == f"file #1 of the batch: path matches private term #{TERM_NO}"


def _range(repo: Path, base: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return _run(["--range", f"{base}..HEAD"], repo, _private_root(tmp_path))


def test_range_catches_a_message(tmp_path: Path, repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    sha = _commit(repo, f"tune the {TERM} voice")
    result = _range(repo, base, tmp_path)
    assert result.returncode == 1
    assert _output(result).strip() == f"{sha[:12]}: message matches private term #{TERM_NO}"


def test_range_catches_an_added_line(tmp_path: Path, repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "notes.txt").write_text(f"clean\n{TERM}\n", encoding="utf-8")
    sha = _commit(repo, "notes")
    result = _range(repo, base, tmp_path)
    assert result.returncode == 1
    assert _output(result).strip() == (
        f"{sha[:12]} notes.txt: added line matches private term #{TERM_NO}"
    )


def test_range_catches_a_term_added_then_removed(tmp_path: Path, repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "notes.txt").write_text(f"{TERM}\n", encoding="utf-8")
    added = _commit(repo, "add")
    (repo / "notes.txt").write_text("clean\n", encoding="utf-8")
    removed = _commit(repo, "remove")
    result = _range(repo, base, tmp_path)
    assert result.returncode == 1
    out = _output(result)
    assert f"{added[:12]} notes.txt: added line matches private term #{TERM_NO}" in out
    assert removed[:12] not in out

    only_removal = _range(repo, added, tmp_path)
    assert (only_removal.returncode, only_removal.stdout) == (0, "")


def test_range_catches_a_term_in_a_merge_resolution(tmp_path: Path, repo: Path) -> None:
    (repo / "f.txt").write_text("x\n", encoding="utf-8")
    base = _commit(repo, "f")
    _git(repo, "checkout", "-q", "-b", "side")
    (repo / "f.txt").write_text("side\n", encoding="utf-8")
    _commit(repo, "side")
    _git(repo, "checkout", "-q", "main")
    (repo / "f.txt").write_text("main\n", encoding="utf-8")
    _commit(repo, "main")
    _git(repo, "merge", "-q", "side", check=False)  # conflicts on f.txt
    (repo / "f.txt").write_text(f"{TERM}\n", encoding="utf-8")
    merge = _commit(repo, "merge side")
    assert len(_git(repo, "rev-list", "--parents", "-n1", "HEAD").split()) == 3
    result = _range(repo, base, tmp_path)
    assert result.returncode == 1
    assert _output(result).strip() == (
        f"{merge[:12]} f.txt: added line matches private term #{TERM_NO}"
    )


def test_range_with_no_list_exits_2(tmp_path: Path, repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    _commit(repo, "empty")
    result = _run(["--range", f"{base}..HEAD"], repo, _private_root(tmp_path, with_list=False))
    assert result.returncode == 2
    assert "voice private terms: no list" in result.stdout + result.stderr


@pytest.mark.parametrize("case", [*PATH_CASES, "renamed file"])
def test_range_withholds_a_matching_path(tmp_path: Path, repo: Path, case: str) -> None:
    if case == "renamed file":
        (repo / "clean.txt").write_text(f"clean\n{TERM}\n", encoding="utf-8")
        base = _commit(repo, "clean")
        _git(repo, "mv", "clean.txt", f"{TERM}.txt")
        rel, number = f"{TERM}.txt", TERM_NO
    else:
        base = _git(repo, "rev-parse", "HEAD")
        rel, number = PATH_CASES[case]
        _write_case(repo, rel)
    sha = _commit(repo, "files")
    result = _range(repo, base, tmp_path)
    assert result.returncode == 1
    lines = set(_output(result).splitlines())
    assert f"{sha[:12]}: a path matches private term #{number}" in lines
    if not rel.endswith(".dat"):
        assert f"{sha[:12]} path withheld: added line matches private term #{TERM_NO}" in lines
    assert all("path" in line for line in lines)


# --- the terms hook under pre-commit --------------------------------------


def test_precommit_refuses_a_staged_binary_named_with_a_term(tmp_path: Path, repo: Path) -> None:
    config = {"repos": [{"repo": "local", "hooks": [_hook("no-voice-private-terms")]}]}
    (repo / ".pre-commit-config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (repo / "scripts").mkdir()
    shutil.copy(SCRIPT_PATH, repo / "scripts" / SCRIPT_PATH.name)
    (repo / f"{TERM}.dat").write_bytes(b"\x00\xff\x00binary")
    _git(repo, "add", "-A")

    env = _git_env(repo)
    env["MIST_VOICE_PRIVATE_ROOT"] = str(_private_root(tmp_path))
    env["PRE_COMMIT_HOME"] = str(tmp_path / "pre-commit-home")
    result = subprocess.run(
        [sys.executable, "-m", "pre_commit", "run", "--color=never"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    out = _output(result)
    assert "No private voice-data terms" in out
    assert "Failed" in out
    assert re.search(rf"^file #\d+ of the batch: path matches private term #{TERM_NO}$", out, re.M)
