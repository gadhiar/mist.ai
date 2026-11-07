"""Tests for personality config loading and system prompt templating."""

from pathlib import Path

import yaml

VOICE_PROFILES_DIR = Path(__file__).parent.parent.parent / "voice_profiles"


def _load_personality(profile_name: str) -> dict:
    """Load personality config for a voice profile."""
    config_path = VOICE_PROFILES_DIR / profile_name / "personality.yaml"
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
    address_form = personality.get("address_form")

    parts = [f"You are {name}, a personal AI assistant."]

    if address_form:
        parts.append(f"\nAddress the user as {address_form}.")

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

    def test_mist_personality_exists(self):
        config = _load_personality("mist")
        assert config, "mist/personality.yaml should exist and be non-empty"

    def test_mist_has_required_fields(self):
        config = _load_personality("mist")
        assert config.get("name") == "M.I.S.T"
        assert "characteristic_openers" in config
        assert "speaking_style" in config
        assert "mannerisms" in config
        assert len(config["characteristic_openers"]) >= 3

    def test_mist_has_no_address_form(self):
        config = _load_personality("mist")
        assert "address_form" in config
        assert config["address_form"] is None

    def test_mist_openers_are_short(self):
        """Openers must be short enough for fast first TTS."""
        config = _load_personality("mist")
        for opener in config["characteristic_openers"]:
            assert len(opener) <= 40, (
                f"Opener too long for first-utterance priming: "
                f"'{opener}' ({len(opener)} chars, max 40)"
            )

    def test_mist_prompt_generation(self):
        config = _load_personality("mist")
        prompt = _build_voice_system_prompt(config)
        assert "M.I.S.T" in prompt
        assert "opening phrase" in prompt
        assert "Address the user as" not in prompt

    def test_empty_personality_fallback(self):
        prompt = _build_voice_system_prompt({})
        assert "M.I.S.T" in prompt
        assert "concise" in prompt
