"""Tests for the private root: its marker, layout, refusals and ids (Interfaces, T2)."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from scripts.voice_data import root as root_mod
from scripts.voice_data.errors import RootError

OLD_MTIME_NS = 1_600_000_000_000_000_000


def _plant(path: Path, data: bytes = b"untouched bytes") -> tuple[bytes, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    os.utime(path, ns=(OLD_MTIME_NS, OLD_MTIME_NS))
    return data, path.stat().st_mtime_ns


def _assert_unchanged(path: Path, before: tuple[bytes, int]) -> None:
    assert path.read_bytes() == before[0]
    assert path.stat().st_mtime_ns == before[1]


# --- resolving and opening -------------------------------------------------------------------


def test_unset_root_is_refused() -> None:
    with pytest.raises(RootError, match="MIST_VOICE_PRIVATE_ROOT"):
        root_mod.open_root(None, environ={})


def test_root_comes_from_the_environment(private_root: root_mod.Root) -> None:
    opened = root_mod.open_root(None, environ={"MIST_VOICE_PRIVATE_ROOT": str(private_root.path)})
    assert opened.path == private_root.path
    assert opened.kind == "private"


def test_argument_wins_over_the_environment(private_root: root_mod.Root, tmp_path: Path) -> None:
    opened = root_mod.open_root(
        str(private_root.path), environ={"MIST_VOICE_PRIVATE_ROOT": str(tmp_path / "elsewhere")}
    )
    assert opened.path == private_root.path


def test_missing_marker_is_refused(tmp_path: Path) -> None:
    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(RootError, match="marker"):
        root_mod.open_root(str(bare), environ={})


def test_unreadable_marker_is_refused(tmp_path: Path) -> None:
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / root_mod.MARKER).write_text("not json", encoding="utf-8")
    with pytest.raises(RootError, match="marker"):
        root_mod.open_root(str(bare), environ={})


# --- .git in the root or a parent ------------------------------------------------------------


def _make_git(where: Path, as_file: bool) -> None:
    git = where / ".git"
    if as_file:
        git.write_text("gitdir: /elsewhere\n", encoding="utf-8")
    else:
        git.mkdir()


GIT_CASES = [
    pytest.param("root", False, id="dir-in-root"),
    pytest.param("root", True, id="file-in-root"),
    pytest.param("parent", False, id="dir-in-parent"),
    pytest.param("parent", True, id="file-in-parent"),
]


@pytest.mark.parametrize(("where", "as_file"), GIT_CASES)
def test_git_refuses_open(tmp_path: Path, where: str, as_file: bool) -> None:
    root = root_mod.init_root(tmp_path / "parent" / "root", "private")
    _make_git(root.path if where == "root" else root.path.parent, as_file)
    with pytest.raises(RootError, match=r"\.git"):
        root_mod.open_root(str(root.path), environ={})


@pytest.mark.parametrize(("where", "as_file"), GIT_CASES)
def test_git_refuses_init(tmp_path: Path, where: str, as_file: bool) -> None:
    target = tmp_path / "parent" / "root"
    target.mkdir(parents=True)
    _make_git(target if where == "root" else target.parent, as_file)
    with pytest.raises(RootError, match=r"\.git"):
        root_mod.init_root(target, "private")
    assert not (target / root_mod.MARKER).exists()


@pytest.mark.parametrize(("where", "as_file"), GIT_CASES)
def test_git_refuses_a_writer(tmp_path: Path, where: str, as_file: bool) -> None:
    root = root_mod.init_root(tmp_path / "parent" / "root", "private")
    _make_git(root.path if where == "root" else root.path.parent, as_file)
    with pytest.raises(RootError, match=r"\.git"):
        root.write_bytes("seed/runs/r1/report.json", b"{}")
    assert not (root.path / "seed/runs/r1/report.json").exists()


# --- write paths -----------------------------------------------------------------------------


def test_write_inside_the_root(private_root: root_mod.Root) -> None:
    written = private_root.write_bytes("seed/runs/r1/report.json", b"{}")
    assert written == private_root.path / "seed/runs/r1/report.json"
    assert written.read_bytes() == b"{}"


def test_write_outside_the_root_is_refused(private_root: root_mod.Root, tmp_path: Path) -> None:
    with pytest.raises(RootError, match="outside"):
        private_root.write_bytes("../escaped.json", b"{}")
    with pytest.raises(RootError, match="outside"):
        private_root.write_bytes(tmp_path / "escaped-abs.json", b"{}")
    assert not (tmp_path / "escaped.json").exists()
    assert not (tmp_path / "escaped-abs.json").exists()


def test_write_through_a_symlink_out_of_the_root_is_refused(
    private_root: root_mod.Root, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (private_root.path / "corpus" / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RootError, match="outside"):
        private_root.write_bytes("corpus/link/takes.jsonl", b"")
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    "rel",
    ["seed/raw/new.bin", "seed/raw/sub/new.bin", "seed/runs/../raw/new.bin", "seed/raw"],
)
def test_write_under_seed_raw_is_refused(private_root: root_mod.Root, rel: str) -> None:
    with pytest.raises(RootError, match="seed/raw"):
        private_root.write_bytes(rel, b"x")
    assert list((private_root.path / "seed/raw").iterdir()) == []


def test_write_through_a_symlink_into_seed_raw_is_refused(private_root: root_mod.Root) -> None:
    (private_root.path / "corpus" / "raw-link").symlink_to(
        private_root.path / "seed" / "raw", target_is_directory=True
    )
    with pytest.raises(RootError, match="seed/raw"):
        private_root.write_bytes("corpus/raw-link/new.bin", b"x")


def test_write_leaves_no_temporary_file(private_root: root_mod.Root) -> None:
    private_root.write_bytes("text/v1/report.json", b"{}")
    assert sorted(p.name for p in (private_root.path / "text/v1").iterdir()) == ["report.json"]


def test_rel_is_root_relative_and_posix(private_root: root_mod.Root) -> None:
    assert private_root.rel(private_root.path / "seed" / "runs" / "r1") == "seed/runs/r1"


# --- init ------------------------------------------------------------------------------------


def test_init_writes_marker_and_layout(tmp_path: Path) -> None:
    root = root_mod.init_root(tmp_path / "fresh", "private")
    marker = json.loads((root.path / root_mod.MARKER).read_text(encoding="utf-8"))
    assert set(marker) == {"schema", "kind", "created"}
    assert marker["schema"] == 1
    assert marker["kind"] == "private"
    datetime.fromisoformat(marker["created"])
    for rel in root_mod.LAYOUT:
        assert (root.path / rel).is_dir(), rel


def test_init_refuses_non_empty_directory_without_marker(tmp_path: Path) -> None:
    target = tmp_path / "busy"
    before = _plant(target / "notes.txt")
    with pytest.raises(RootError, match="init"):
        root_mod.init_root(target, "private")
    assert sorted(p.name for p in target.iterdir()) == ["notes.txt"]
    _assert_unchanged(target / "notes.txt", before)


def test_init_accepts_v6_directories(tmp_path: Path) -> None:
    target = tmp_path / "v6"
    for name in ("voices", "design", "bench"):
        (target / name).mkdir(parents=True)
    before = _plant(target / "voices" / "v1" / "voice.json")
    root = root_mod.init_root(target, "private")
    assert (root.path / root_mod.MARKER).is_file()
    _assert_unchanged(target / "voices" / "v1" / "voice.json", before)


def test_init_rehearsal_accepts_only_public(tmp_path: Path) -> None:
    target = tmp_path / "rehearsal"
    before = _plant(target / "public" / "download.bin")
    root = root_mod.init_root(target, "rehearsal")
    assert root.kind == "rehearsal"
    _assert_unchanged(target / "public" / "download.bin", before)


def test_init_private_refuses_public(tmp_path: Path) -> None:
    target = tmp_path / "rehearsal"
    _plant(target / "public" / "download.bin")
    with pytest.raises(RootError, match="init"):
        root_mod.init_root(target, "private")
    assert not (target / root_mod.MARKER).exists()


def test_init_rehearsal_refuses_v6_directories(tmp_path: Path) -> None:
    target = tmp_path / "rehearsal"
    (target / "public").mkdir(parents=True)
    (target / "voices").mkdir()
    with pytest.raises(RootError, match="init"):
        root_mod.init_root(target, "rehearsal")


def test_init_private_accepts_seed_raw_and_leaves_it_untouched(tmp_path: Path) -> None:
    target = tmp_path / "seeded"
    (target / "voices").mkdir(parents=True)
    episode = target / "seed" / "raw" / "season" / "episode.bin"
    before = _plant(episode, b"\x00\x01episode bytes")
    clip = target / "seed" / "reference" / "solo.bin"
    clip_before = _plant(clip)
    root_mod.init_root(target, "private")
    assert (target / root_mod.MARKER).is_file()
    _assert_unchanged(episode, before)
    _assert_unchanged(clip, clip_before)


def test_init_refuses_other_entries_in_seed(tmp_path: Path) -> None:
    target = tmp_path / "seeded"
    _plant(target / "seed" / "raw" / "episode.bin")
    (target / "seed" / "tracks").mkdir()
    with pytest.raises(RootError, match="init"):
        root_mod.init_root(target, "private")


def test_init_accepts_guard_terms_only(tmp_path: Path) -> None:
    target = tmp_path / "guarded"
    terms = target / "guard" / "terms.txt"
    before = _plant(terms, b"# terms\nalpha\n")
    root_mod.init_root(target, "private")
    assert (target / root_mod.MARKER).is_file()
    _assert_unchanged(terms, before)


def test_init_refuses_second_file_in_guard(tmp_path: Path) -> None:
    target = tmp_path / "guarded"
    _plant(target / "guard" / "terms.txt")
    _plant(target / "guard" / "other.txt")
    with pytest.raises(RootError, match="init"):
        root_mod.init_root(target, "private")
    assert not (target / root_mod.MARKER).exists()


def test_init_refuses_guard_beside_unknown_entry(tmp_path: Path) -> None:
    target = tmp_path / "guarded"
    _plant(target / "guard" / "terms.txt")
    (target / "misc").mkdir()
    with pytest.raises(RootError, match="init"):
        root_mod.init_root(target, "private")
    assert not (target / root_mod.MARKER).exists()


def test_init_is_repeatable_on_its_own_root(tmp_path: Path) -> None:
    first = root_mod.init_root(tmp_path / "again", "private")
    marker = (first.path / root_mod.MARKER).read_bytes()
    second = root_mod.init_root(first.path, "private")
    assert second == first
    assert (first.path / root_mod.MARKER).read_bytes() == marker


def test_init_refuses_a_root_of_the_other_kind(tmp_path: Path) -> None:
    first = root_mod.init_root(tmp_path / "again", "private")
    with pytest.raises(RootError, match="kind"):
        root_mod.init_root(first.path, "rehearsal")


def test_init_refuses_an_unknown_kind(tmp_path: Path) -> None:
    with pytest.raises(RootError, match="kind"):
        root_mod.init_root(tmp_path / "x", "public")


# --- ids -------------------------------------------------------------------------------------


def test_ids_have_the_interface_formats() -> None:
    assert root_mod.source_id(1) == "src-0001"
    assert root_mod.segment_id(123) == "seg-000123"
    when = datetime(2026, 11, 12, 10, 15, 42, tzinfo=UTC)
    assert root_mod.job_id(when, 1) == "job-20261112T1015Z-1"


def test_take_id_is_a_stable_sha12_of_its_key() -> None:
    first = root_mod.take_id("line-1", "calm", "teacher-a", 7)
    assert first == root_mod.take_id("line-1", "calm", "teacher-a", 7)
    assert first.startswith("take-")
    assert len(first) == len("take-") + 12
    int(first[len("take-") :], 16)
    assert first != root_mod.take_id("line-1", "calm", "teacher-a", 8)
    assert first != root_mod.take_id("line-1", "sleepy", "teacher-a", 7)


# --- shared conftest helpers that later tasks rely on ----------------------------------------


def test_synth_voice_has_its_f0(synth_voice) -> None:
    rate = 16000
    voice = synth_voice(150.0, [700.0, (1200.0, 120.0), 2600.0], 2.0, rate)
    assert voice.dtype == np.float32
    assert len(voice) == 2 * rate
    assert 0.49 < float(np.max(np.abs(voice))) <= 0.5
    spectrum = np.abs(np.fft.rfft(voice))
    freqs = np.fft.rfftfreq(len(voice), 1.0 / rate)
    band = (freqs > 60) & (freqs < 200)
    assert abs(freqs[band][np.argmax(spectrum[band])] - 150.0) < 10.0


def test_mix_places_parts_and_adds_noise(synth_voice, mix) -> None:
    rate = 16000
    a = synth_voice(120.0, [700.0], 1.0, rate)
    b = synth_voice(220.0, [900.0], 1.0, rate)
    mixed = mix([(a, 0.0), (b, 0.5, -6.0)], rate)
    assert len(mixed) == int(1.5 * rate)
    assert np.allclose(mixed[: rate // 2], a[: rate // 2])
    clean = mix([(a, 0.0)], rate, seconds=2.0)
    assert len(clean) == 2 * rate
    assert not clean[rate:].any()
    noisy = mix([(a, 0.0)], rate, noise_snr_db=10.0, seed=1)
    noise = noisy.astype(np.float64) - a
    snr = 10 * np.log10(np.mean(a.astype(np.float64) ** 2) / np.mean(noise**2))
    assert abs(snr - 10.0) < 0.5
