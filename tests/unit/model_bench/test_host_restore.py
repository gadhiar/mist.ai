"""compute_restore_diff(): empty, non-empty, and the Id-changed (recreate) case."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.model_bench.bench_host import compute_restore_diff  # noqa: E402


def _inspect(id_="sha256:aaa", image="img:tag", cmd=None, env=None, host_config=None):
    return {
        "Id": id_,
        "Image": image,
        "Config": {"Cmd": cmd or [], "Env": env or ["A=1"]},
        "HostConfig": host_config or {"NetworkMode": "default"},
    }


def test_restore_diff_empty_when_identical():
    snap = {"mist-llm": _inspect(), "mist-backend": _inspect(id_="sha256:bbb")}
    current = {"mist-llm": _inspect(), "mist-backend": _inspect(id_="sha256:bbb")}
    diff = compute_restore_diff(snap, current)
    assert diff["empty"] is True
    assert diff["id_changed"] is False
    assert diff["containers"] == {}


def test_restore_diff_non_empty_on_env_change():
    snap = {"mist-llm": _inspect(env=["A=1"])}
    current = {"mist-llm": _inspect(env=["A=1", "B=2"])}
    diff = compute_restore_diff(snap, current)
    assert diff["empty"] is False
    assert diff["id_changed"] is False
    assert "Config.Env" in diff["containers"]["mist-llm"]["diffs"]
    assert diff["containers"]["mist-llm"]["id_changed"] is False


def test_restore_diff_id_changed_flags_recreate():
    snap = {"mist-llm": _inspect(id_="sha256:aaa")}
    current = {"mist-llm": _inspect(id_="sha256:zzz")}
    diff = compute_restore_diff(snap, current)
    assert diff["empty"] is False
    assert diff["id_changed"] is True
    assert diff["containers"]["mist-llm"]["id_changed"] is True
    assert diff["containers"]["mist-llm"]["diffs"]["Id"] == {"before": "sha256:aaa", "after": "sha256:zzz"}


def test_restore_diff_missing_container_counts_as_id_changed():
    snap = {"mist-llm": _inspect()}
    current: dict = {}
    diff = compute_restore_diff(snap, current)
    assert diff["empty"] is False
    assert diff["id_changed"] is True


def test_restore_diff_host_config_change_detected():
    snap = {"mist-llm": _inspect(host_config={"NetworkMode": "default"})}
    current = {"mist-llm": _inspect(host_config={"NetworkMode": "host"})}
    diff = compute_restore_diff(snap, current)
    assert diff["empty"] is False
    assert "HostConfig" in diff["containers"]["mist-llm"]["diffs"]
