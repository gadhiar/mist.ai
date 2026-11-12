"""The `repo: local` pre-commit hooks that keep audio out of this public repository.

Hermetic: parses the committed `.pre-commit-config.yaml` only; no git, no pre-commit run. The
repository is public, so a voice clip committed once is published for good: the hook refuses it at
commit time, before it can reach a push.

pre-commit matches a hook's `files` with `re.search` against each repository-relative path and
compiles it without flags, so case-insensitivity has to be written into the pattern itself (`(?i)`);
`_matches` reproduces that call. An `exclude`, on the hook or at the top level, would carve a path
out of the guard, so neither may exist.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / ".pre-commit-config.yaml"


def _config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _local_hook(hook_id: str) -> dict:
    hooks = [
        hook
        for repo in _config()["repos"]
        if repo["repo"] == "local"
        for hook in repo["hooks"]
        if hook["id"] == hook_id
    ]
    assert len(hooks) == 1, f"expected one local hook {hook_id!r}, found {len(hooks)}"
    return hooks[0]


def _matches(hook_id: str, path: str) -> bool:
    return re.compile(_local_hook(hook_id)["files"]).search(path) is not None


def test_no_audio_files_hook_always_fails() -> None:
    hook = _local_hook("no-audio-files")
    assert hook["language"] == "fail"
    assert "audio" in hook["entry"].lower()


@pytest.mark.parametrize(
    "path",
    [
        "x.WAV",
        "x.webm",
        "data/voice_profiles/sample/a.wav",
        "a.mp3",
        "a.flac",
        "a.ogg",
        "a.opus",
        "a.m4a",
        "a.aac",
        "a.aif",
        "a.aiff",
        "a.wma",
        "A.Mp3",
    ],
)
def test_no_audio_files_matches_audio(path: str) -> None:
    assert _matches("no-audio-files", path)


@pytest.mark.parametrize("path", ["x.py", "wav.py", "x.wav.md", "docs/audio.md"])
def test_no_audio_files_ignores_other_files(path: str) -> None:
    assert not _matches("no-audio-files", path)


def test_no_audio_files_has_no_exclude() -> None:
    assert "exclude" not in _local_hook("no-audio-files")


def test_config_has_no_top_level_exclude() -> None:
    """A top-level `exclude` applies to every hook, so it would bypass the guard too."""
    assert "exclude" not in _config()
