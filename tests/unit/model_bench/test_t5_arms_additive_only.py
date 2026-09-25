"""T5 (plan v3) additive-only guarantee for scripts/model_bench/arms.json.

Every arm id present at the integration base commit (61f4822, the commit this task's
worktree branched from) must resolve to the EXACT SAME config under the current
arms.json -- T5 only adds new arms; it never edits an existing one. A running driver
(host session S4b) compares `arm_config` and the decision_rules sha on every
`merge_run_meta` call, so an edit to an existing arm would make it refuse later calls.

Loads the base file via `git show <base commit>:scripts/model_bench/arms.json`
(subprocess git against this repo's own history -- read-only, no network needed, per
the worker container's git-metadata access), the same pattern
test_decision_rules_v1_unchanged.py already uses for decision_rules.json.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.model_bench.bench_host import (  # noqa: E402
    ARMS_JSON_PATH,
    load_arms_doc,
    resolve_all_arms,
)

# agent/mist-model-bench/integration's HEAD when this task's (t5-arms-v3) worktree
# was created -- the last commit before this task's additive-only changes.
BASE_COMMIT = "61f48221b50321bbd18374dc88e3ab4313b3881a"

DECISION_RULES_PATH = ARMS_JSON_PATH.parent / "decision_rules.json"
DECISION_RULES_EXPECTED_SHA256 = (
    "ad181060e24727e4ffedcf899e8999941561df3b4ea6f34482878c8cbbac8c0e"
)


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


def _load_base_arms_doc() -> dict:
    return json.loads(_git_show("scripts/model_bench/arms.json"))


def test_every_base_arm_id_still_present():
    base_doc = _load_base_arms_doc()
    current_doc = load_arms_doc()
    missing = set(base_doc["arms"]) - set(current_doc["arms"])
    assert not missing, f"T5 removed arm id(s): {sorted(missing)}"


def test_every_base_arm_resolves_identically_under_current_arms_json():
    base_doc = _load_base_arms_doc()
    current_doc = load_arms_doc()
    base_resolved = resolve_all_arms(base_doc)
    current_resolved = resolve_all_arms(current_doc)
    for arm_id, base_arm in base_resolved.items():
        assert arm_id in current_resolved, f"{arm_id} missing from current arms.json"
        assert current_resolved[arm_id] == base_arm, (
            f"arm {arm_id!r} resolved differently under the current arms.json -- "
            f"T5 must be additive-only"
        )


def test_common_args_sampling_and_thinking_args_are_byte_for_byte_unchanged():
    base_doc = _load_base_arms_doc()
    current_doc = load_arms_doc()
    assert current_doc["common_args"] == base_doc["common_args"]
    assert current_doc["sampling"] == base_doc["sampling"]
    assert current_doc["thinking_args"] == base_doc["thinking_args"]


def test_decision_rules_json_sha256_unchanged():
    digest = hashlib.sha256(DECISION_RULES_PATH.read_bytes()).hexdigest()
    assert digest == DECISION_RULES_EXPECTED_SHA256
