"""T5 (plan v3) additive-only guarantee for scripts/eval_harness/models.yaml.

Every candidate id present at the integration base commit (61f4822, the commit this
task's worktree branched from) must parse to the EXACT SAME `Candidate` under the
current models.yaml -- T5 only adds new bench candidates (bench-c5, bench-c6,
bench-c3-q3, bench-c3-iq4); it never edits an existing one, including
qwen-3.5-9b-q8 (plan v3's c5 arm points at the same GGUF but must not touch that
existing primary-tier candidate).

Loads the base file via `git show <base commit>:scripts/eval_harness/models.yaml`
(subprocess git against this repo's own history), the same pattern
test_decision_rules_v1_unchanged.py uses in tests/unit/model_bench/.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.eval_harness import run  # noqa: E402

BASE_COMMIT = "61f48221b50321bbd18374dc88e3ab4313b3881a"


def _git_show(rel_path: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "show", f"{BASE_COMMIT}:{rel_path}"],
        capture_output=True,
        text=True,
        shell=False,
    )
    if proc.returncode != 0:
        pytest.skip(
            f"cannot read {rel_path} at {BASE_COMMIT} via `git show` "
            f"(exit {proc.returncode}): {proc.stderr.strip()}"
        )
    return proc.stdout


def _load_base_candidates(tmp_path: Path):
    text = _git_show("scripts/eval_harness/models.yaml")
    base_path = tmp_path / "models_base.yaml"
    base_path.write_text(text, encoding="utf-8")
    return run.load_models_config(base_path)


def test_every_base_candidate_id_still_present(tmp_path):
    _, base_candidates = _load_base_candidates(tmp_path)
    _, current_candidates = run.load_models_config(run.DEFAULT_CONFIG_PATH)
    base_ids = {c.id for c in base_candidates}
    current_ids = {c.id for c in current_candidates}
    missing = base_ids - current_ids
    assert not missing, f"T5 removed candidate id(s): {sorted(missing)}"


def test_every_base_candidate_parses_identically_under_current_models_yaml(tmp_path):
    base_defaults, base_candidates = _load_base_candidates(tmp_path)
    current_defaults, current_candidates = run.load_models_config(run.DEFAULT_CONFIG_PATH)
    assert current_defaults == base_defaults

    current_by_id = {c.id: c for c in current_candidates}
    for base_candidate in base_candidates:
        assert base_candidate.id in current_by_id
        assert current_by_id[base_candidate.id] == base_candidate, (
            f"candidate {base_candidate.id!r} parsed differently under the current "
            f"models.yaml -- T5 must be additive-only"
        )


def test_qwen_3_5_9b_q8_primary_candidate_is_untouched(tmp_path):
    _, base_candidates = _load_base_candidates(tmp_path)
    _, current_candidates = run.load_models_config(run.DEFAULT_CONFIG_PATH)
    base_qwen = next(c for c in base_candidates if c.id == "qwen-3.5-9b-q8")
    current_qwen = next(c for c in current_candidates if c.id == "qwen-3.5-9b-q8")
    assert current_qwen == base_qwen
