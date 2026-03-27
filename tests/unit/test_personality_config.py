"""Tests for personality config loading and system prompt templating."""

from pathlib import Path

import yaml


def _load_personality(profile_name: str) -> dict:
    """Load personality config for a voice profile."""
    config_path = (
        Path(__file__).parent.parent.parent / "voice_profiles" / profile_name / "personality.yaml"
    )
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _build_voice_system_prompt(personality: dict) -> str:
    """Build voice system prompt from personality config."""
    if not personality:
        return (
            "You are M.I.S.T, a helpful voice assistant. "
            "Keep responses concise and conversational."
        )

    name = personality.get("name", "M.I.S.T")
    style = personality.get("speaking_style", "").strip()
    openers = personality.get("characteristic_openers", [])
    mannerisms = personality.get("mannerisms", [])

    parts = [f"You are {name}, a personal AI assistant."]

    if style:
        parts.append(f"\nSpeaking style: {style}")

    if openers:
        opener_list = "\n".join(f"- {o}" for o in openers)
        parts.append(
            "\nWhen responding, begin with a brief acknowledgment or "
            "opening phrase before your main answer. Examples of "
            f"characteristic openers you use:\n{opener_list}"
        )

    if mannerisms:
        mannerism_list = "\n".join(f"- {m}" for m in mannerisms)
        parts.append(f"\nAdditional guidelines:\n{mannerism_list}")

    return "\n".join(parts)


class TestPersonalityConfig:
    """Test personality config loading and prompt generation."""

    def test_friday_personality_exists(self):
        config = _load_personality("friday")
        assert config, "friday/personality.yaml should exist and be non-empty"

    def test_friday_has_required_fields(self):
        config = _load_personality("friday")
        assert "name" in config
        assert "characteristic_openers" in config
        assert "speaking_style" in config
        assert len(config["characteristic_openers"]) >= 3

    def test_friday_openers_are_short(self):
        """Openers must be short enough for fast first TTS."""
        config = _load_personality("friday")
        for opener in config["characteristic_openers"]:
            assert len(opener) <= 40, (
                f"Opener too long for first-utterance priming: "
                f"'{opener}' ({len(opener)} chars, max 40)"
            )

    def test_friday_prompt_generation(self):
        config = _load_personality("friday")
        prompt = _build_voice_system_prompt(config)
        assert "FRIDAY" in prompt
        assert "opening phrase" in prompt
        assert "Boss" in prompt

    def test_empty_personality_fallback(self):
        prompt = _build_voice_system_prompt({})
        assert "M.I.S.T" in prompt
        assert "concise" in prompt

    def test_jarvis_stub_loads(self):
        config = _load_personality("jarvis")
        assert isinstance(config, dict)

    def test_cortana_stub_loads(self):
        config = _load_personality("cortana")
        assert isinstance(config, dict)
