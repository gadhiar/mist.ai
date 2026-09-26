"""T5/T7 (plan v3) additive-only guarantee for scripts/model_bench/arms.json.

Every arm id present at the base commit must resolve to the EXACT SAME config under
the current arms.json -- T5 and T7 only add new arms; neither ever edits an existing
one. A running driver (host session S4b) compares `arm_config` and the decision_rules
sha on every `merge_run_meta` call, so an edit to an existing arm would make it refuse
later calls.

BASE_COMMIT was originally 61f48221b50321bbd18374dc88e3ab4313b3881a (T5's own base,
before T5 landed). T7 (the scan-picked c7/c8/c9 arms) branched from
044699b6b9d4f3f1759d1f67df65012eff56699d, which already contains every T5 arm
(c1-2048, c1-unbudgeted, c5, c5-think1024, c6, c3-q3, c3-iq4) committed -- moving
BASE_COMMIT to that commit extends this same additive-only guarantee to cover all of
T5's arms too, not just the pre-T5 set, exactly as the T7 brief's acceptance criterion
1 asks for ("every arm present at 044699b resolves identically"). Every base-commit
arm is still checked byte-for-byte; nothing about the comparison itself changed.

Pre-PR fixes (i64) add one narrow, named exception to that byte-for-byte guarantee:
`c0-ub1024` gains `"optional": true` (the plan allowed at most one such server-setting
arm, and running it was the lead's choice -- see scripts/model_bench/README.md). Every
other arm, including every other field of c0-ub1024 itself, is still required to be
byte-identical to its BASE_COMMIT resolution.

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

# agent/mist-model-bench/integration's HEAD when the T7 (scan-arms) worktree was
# created -- the last commit before T7's additive-only changes, and already the
# commit T5 itself landed on. See the module docstring for why this moved from
# T5's own original base (61f48221b50321bbd18374dc88e3ab4313b3881a).
BASE_COMMIT = "044699b6b9d4f3f1759d1f67df65012eff56699d"

DECISION_RULES_PATH = ARMS_JSON_PATH.parent / "decision_rules.json"
# Pre-PR fixes (i64) edited only prose fields (supersedes_note, exploratory_rules.note)
# and appended a second `supersedes` entry, moving decision_rules.json's own sha256 from
# the plan v2 value (ad181060...) to this one. The superseded value is pinned below too,
# so a future edit cannot drop it from `supersedes` without failing a test.
DECISION_RULES_EXPECTED_SHA256 = (
    "e6562bff1ccf2ab4374305597eeb6587a9eff8cbbeeff4987c3ef618bb268db2"
)
DECISION_RULES_SUPERSEDED_V2_SHA256 = (
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


# Pre-PR fixes (i64): c0-ub1024 gained `optional: true` (see module docstring). This is
# the ONLY arm, and the ONLY field, this test allows to differ from BASE_COMMIT.
_OPTIONAL_FLAG_EXEMPT_ARM = "c0-ub1024"


def test_every_base_arm_resolves_identically_under_current_arms_json():
    base_doc = _load_base_arms_doc()
    current_doc = load_arms_doc()
    base_resolved = resolve_all_arms(base_doc)
    current_resolved = resolve_all_arms(current_doc)
    for arm_id, base_arm in base_resolved.items():
        assert arm_id in current_resolved, f"{arm_id} missing from current arms.json"
        current_arm = current_resolved[arm_id]
        if arm_id == _OPTIONAL_FLAG_EXEMPT_ARM:
            assert base_arm["optional"] is False, (
                f"{arm_id!r} was expected to start with optional=False at BASE_COMMIT"
            )
            assert current_arm["optional"] is True, (
                f"{arm_id!r} was expected to resolve with optional=True after the "
                f"pre-PR edit"
            )
            base_without_optional = {k: v for k, v in base_arm.items() if k != "optional"}
            current_without_optional = {k: v for k, v in current_arm.items() if k != "optional"}
            assert current_without_optional == base_without_optional, (
                f"arm {arm_id!r} resolved differently under the current arms.json on "
                f"a field other than 'optional' -- only the optional flag may differ"
            )
            continue
        assert current_arm == base_arm, (
            f"arm {arm_id!r} resolved differently under the current arms.json -- "
            f"T5 must be additive-only"
        )


def test_common_args_sampling_and_thinking_args_are_byte_for_byte_unchanged():
    base_doc = _load_base_arms_doc()
    current_doc = load_arms_doc()
    assert current_doc["common_args"] == base_doc["common_args"]
    assert current_doc["thinking_args"] == base_doc["thinking_args"]
    # sampling: every key already present at the base commit stays byte-identical
    # (T7's brief explicitly permits ADDING a new family key -- e.g. "granite",
    # "spark", "gptoss" -- so this checks per-key equality on the base's own
    # keys, not whole-dict equality).
    for family, values in base_doc["sampling"].items():
        assert family in current_doc["sampling"], f"sampling family {family!r} was removed"
        assert current_doc["sampling"][family] == values, (
            f"sampling family {family!r} changed -- existing families must stay "
            f"byte-identical"
        )


def test_decision_rules_json_sha256_unchanged():
    digest = hashlib.sha256(DECISION_RULES_PATH.read_bytes()).hexdigest()
    assert digest == DECISION_RULES_EXPECTED_SHA256


def test_decision_rules_json_supersedes_lists_the_previous_sha():
    # The pre-PR edit (i64) moved decision_rules.json's own sha256 away from the plan v2
    # value; that value must still be listed in `supersedes` so recorded runs (mb1, mb2)
    # written under it keep reporting [INFO], not [WARN].
    doc = json.loads(DECISION_RULES_PATH.read_text(encoding="utf-8"))
    listed = {e["sha256"] for e in doc.get("supersedes", [])}
    assert DECISION_RULES_SUPERSEDED_V2_SHA256 in listed
