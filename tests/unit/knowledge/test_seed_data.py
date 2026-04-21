"""Tests that seed_data.yaml has expected canonical entries."""

from pathlib import Path

import yaml

# Resolve path relative to repo root regardless of pytest invocation directory.
_REPO_ROOT = Path(__file__).resolve().parents[3]
SEED_PATH = _REPO_ROOT / "scripts" / "seed_data.yaml"


def _load_seed() -> dict:
    with SEED_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestMistPreferenceNoAiSlop:
    """Cluster 3: pref-no-ai-slop preference exists and has HAS_PREFERENCE edge from mist-identity."""

    def test_preference_entry_exists(self):
        seed = _load_seed()
        prefs = {p["id"]: p for p in seed["preferences"]}
        assert "pref-no-ai-slop" in prefs

    def test_preference_enforcement_is_absolute(self):
        seed = _load_seed()
        prefs = {p["id"]: p for p in seed["preferences"]}
        assert prefs["pref-no-ai-slop"]["enforcement"] == "absolute"

    def test_preference_has_display_name_field(self):
        seed = _load_seed()
        prefs = {p["id"]: p for p in seed["preferences"]}
        assert prefs["pref-no-ai-slop"].get(
            "display_name"
        ), "pref-no-ai-slop must have a display_name"

    def test_preference_context_lists_slop_categories(self):
        seed = _load_seed()
        prefs = {p["id"]: p for p in seed["preferences"]}
        context = prefs["pref-no-ai-slop"]["context"].lower()
        # Context should reference the main slop categories
        assert "superlative" in context, "Context should mention superlatives"
        assert "filler" in context or "phrases" in context, "Context should mention filler/phrases"
        assert (
            "hype" in context
            or "cutting-edge" in context
            or "enterprise-grade" in context
            or "seamless" in context
        ), "Context should mention hype vocabulary"

    def test_has_preference_edge_includes_pref_no_ai_slop(self):
        seed = _load_seed()
        has_pref = [
            r
            for r in seed["identity_relationships"]
            if r["source"] == "mist-identity" and r["type"] == "HAS_PREFERENCE"
        ]
        assert len(has_pref) == 1, f"Expected exactly one HAS_PREFERENCE block, got {len(has_pref)}"
        assert "pref-no-ai-slop" in has_pref[0]["targets"]

    def test_pref_no_emoji_still_present(self):
        """Regression guard: adding pref-no-ai-slop must not displace pref-no-emoji."""
        seed = _load_seed()
        prefs = {p["id"]: p for p in seed["preferences"]}
        assert "pref-no-emoji" in prefs

        has_pref = next(
            r
            for r in seed["identity_relationships"]
            if r["source"] == "mist-identity" and r["type"] == "HAS_PREFERENCE"
        )
        assert "pref-no-emoji" in has_pref["targets"]
