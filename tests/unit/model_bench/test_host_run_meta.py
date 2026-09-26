"""merge_run_meta(): the pure function behind meta.json's cumulative shape (finding 1b).

No docker, no filesystem -- these exercise merge_run_meta directly with hand-built
`existing` / `call` dicts. The end-to-end "several calls actually write a cumulative
meta.json" behavior (and the refusal-writes-nothing guarantee) is covered by
test_host_run_cumulative.py, which stubs docker/http/subprocess and drives cmd_run itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.model_bench.bench_host import (  # noqa: E402
    RUN_SUITE_ORDER,
    RunMetaConfigMismatchError,
    merge_run_meta,
)


def _base_call(**overrides) -> dict:
    call = {
        "schema": 1,
        "run": "run1",
        "arm": "c0",
        "arm_config": {"id": "c0", "gguf": "x.gguf"},
        "server_args": ["-m", "/models/x.gguf"],
        "image_ref": "fake:image",
        "image_id": "sha256:aaaa",
        "params": {},
        "tuning_label": None,
        "decision_rules_sha256": "deadbeef",
        "git": {"mist_ai": "abc123", "mist_ai_dirty": False, "command_center": None},
        "container_args": ["-m", "/models/x.gguf"],
        "props": {"model_path": "/models/x.gguf"},
        "started_utc": "2026-01-01T00:00:00+00:00",
        "finished_utc": "2026-01-01T00:01:00+00:00",
        "suites": ["ttft"],
        "suites_completed": ["ttft"],
        "rep": None,
        "layout_pass": None,
        "harness": None,
        "layout_result": None,
        "errors": [],
    }
    call.update(overrides)
    return call


def test_first_call_with_no_existing_meta_builds_the_full_document():
    merged = merge_run_meta(None, _base_call())
    assert merged["suites_completed"] == ["ttft"]
    assert merged["started_utc"] == "2026-01-01T00:00:00+00:00"
    assert merged["finished_utc"] == "2026-01-01T00:01:00+00:00"
    assert len(merged["calls"]) == 1
    assert merged["calls"][0]["suites"] == ["ttft"]
    assert merged["layout"] == {}
    assert merged["harness"] is None
    assert merged["errors"] == []


def test_suites_completed_is_the_union_kept_in_run_order():
    first = merge_run_meta(None, _base_call(suites=["ttft"], suites_completed=["ttft"]))
    second_call = _base_call(
        suites=["correctness"], suites_completed=["correctness"], rep=1,
        started_utc="2026-01-01T00:05:00+00:00", finished_utc="2026-01-01T00:06:00+00:00",
    )
    merged = merge_run_meta(first, second_call)
    # RUN_SUITE_ORDER = ttft, correctness, harness, layout -- union kept in that order
    # regardless of the order the two calls happened in.
    assert merged["suites_completed"] == [s for s in RUN_SUITE_ORDER if s in ("ttft", "correctness")]
    assert len(merged["calls"]) == 2


def test_errors_accumulate_across_calls():
    first = merge_run_meta(None, _base_call(errors=["harness exited 1"]))
    second = merge_run_meta(first, _base_call(suites=["correctness"], errors=["correctness: timeout"]))
    assert second["errors"] == ["harness exited 1", "correctness: timeout"]


def test_layout_becomes_a_dict_keyed_by_pass_not_a_single_last_pass_dict():
    first = merge_run_meta(
        None,
        _base_call(
            suites=["layout"], suites_completed=["layout"], layout_pass="screen",
            layout_result={"layouts_per_size": 2, "thinking": "off", "max_tokens": 256},
        ),
    )
    second = merge_run_meta(
        first,
        _base_call(
            suites=["layout"], suites_completed=["layout"], layout_pass="finalist",
            layout_result={"layouts_per_size": 6, "thinking": "off", "max_tokens": 256},
        ),
    )
    assert set(second["layout"]) == {"screen", "finalist"}
    assert second["layout"]["screen"]["layouts_per_size"] == 2
    assert second["layout"]["finalist"]["layouts_per_size"] == 6


def test_harness_is_set_by_whichever_call_ran_it_and_persists_after():
    harness_cfg = {"candidate": "bench-c0", "tests": ["schema_conformance"], "iterations": 1}
    first = merge_run_meta(
        None, _base_call(suites=["harness"], suites_completed=["harness"], harness=harness_cfg)
    )
    assert first["harness"] == harness_cfg
    # A later call that does NOT run harness must leave the existing value untouched.
    second = merge_run_meta(first, _base_call(suites=["correctness"], suites_completed=["correctness"]))
    assert second["harness"] == harness_cfg


def test_started_utc_is_the_first_calls_start_finished_utc_is_the_latest():
    first = merge_run_meta(
        None, _base_call(started_utc="2026-01-01T00:00:00+00:00", finished_utc="2026-01-01T00:01:00+00:00")
    )
    second = merge_run_meta(
        first,
        _base_call(
            suites=["correctness"], suites_completed=["correctness"],
            started_utc="2026-01-01T05:00:00+00:00", finished_utc="2026-01-01T05:02:00+00:00",
        ),
    )
    assert second["started_utc"] == "2026-01-01T00:00:00+00:00"
    assert second["finished_utc"] == "2026-01-01T05:02:00+00:00"


def test_calls_list_records_the_documented_per_call_fields():
    merged = merge_run_meta(None, _base_call(rep=2, layout_pass="finalist"))
    entry = merged["calls"][0]
    assert set(entry) == {
        "started_utc", "finished_utc", "suites", "rep", "layout_pass", "props",
        "container_args", "errors", "decision_rules_sha256",
    }
    assert entry["decision_rules_sha256"] == "deadbeef"
    assert entry["rep"] == 2
    assert entry["layout_pass"] == "finalist"


@pytest.mark.parametrize(
    "key,new_value",
    [
        ("arm_config", {"id": "c0", "gguf": "different.gguf"}),
        ("server_args", ["-m", "/models/different.gguf"]),
        ("image_ref", "fake:different-image"),
        ("image_id", "sha256:bbbb"),
        ("params", {"ncmoe": "20"}),
        ("tuning_label", "overclock-v2"),
        ("decision_rules_sha256", "cafef00d"),
    ],
)
def test_a_changed_identity_field_refuses_with_the_differing_key_named(key, new_value):
    existing = merge_run_meta(None, _base_call())
    with pytest.raises(RunMetaConfigMismatchError, match=key):
        merge_run_meta(existing, _base_call(**{key: new_value}))


def test_unlisted_decision_rules_sha_is_still_refused():
    """A differing decision_rules_sha256 is refused unless the OLD (stored) sha is named
    in the caller-supplied `superseded_rules_shas` set -- an arbitrary unlisted sha must
    still be refused exactly like every other identity-field mismatch. This must hold
    both with the default (empty) `superseded_rules_shas` and with a non-empty set that
    simply does not name the stored sha.
    """
    existing = merge_run_meta(None, _base_call(decision_rules_sha256="v1sha"))
    with pytest.raises(RunMetaConfigMismatchError, match="decision_rules_sha256"):
        merge_run_meta(existing, _base_call(decision_rules_sha256="v2sha"))
    with pytest.raises(RunMetaConfigMismatchError, match="decision_rules_sha256"):
        merge_run_meta(
            existing,
            _base_call(decision_rules_sha256="v2sha"),
            superseded_rules_shas=frozenset({"some-other-sha"}),
        )


def test_superseded_decision_rules_sha_merges_and_latest_wins():
    """S4 resumes c1-512 inside run mb1, whose meta.json was written under the v1 rules
    sha. A v2 decision_rules.json that lists the v1 sha in its own `supersedes` must let
    that call merge: the top-level decision_rules_sha256 becomes v2's (the latest call's)
    value, and each call entry keeps its own sha, so the full history survives.
    """
    existing = merge_run_meta(None, _base_call(decision_rules_sha256="v1sha"))
    merged = merge_run_meta(
        existing,
        _base_call(
            suites=["correctness"], suites_completed=["correctness"],
            decision_rules_sha256="v2sha",
        ),
        superseded_rules_shas=frozenset({"v1sha"}),
    )  # must not raise
    assert merged["decision_rules_sha256"] == "v2sha"
    assert [c["decision_rules_sha256"] for c in merged["calls"]] == ["v1sha", "v2sha"]


def test_non_identity_fields_are_allowed_to_differ_between_calls():
    existing = merge_run_meta(None, _base_call())
    # props, container_args, git, started_utc/finished_utc all legitimately vary call to
    # call (a re-served container, a later wall-clock time) and must not trigger a refusal.
    merge_run_meta(
        existing,
        _base_call(
            suites=["correctness"], suites_completed=["correctness"],
            props={"model_path": "/models/x.gguf", "extra": "field"},
            container_args=["-m", "/models/x.gguf", "--extra-flag"],
            git={"mist_ai": "def456", "mist_ai_dirty": True, "command_center": "abc"},
        ),
    )  # must not raise
