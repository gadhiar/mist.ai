"""The persona register: the address form and the prompt's register line from one YAML file.

Capabilities' templated replies and the conversation prompt both address the user; reading the
form from the persona's `personality.yaml` in one place keeps the two from drifting apart.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from backend.capabilities.replies import ReplyRegister, ok_summary, with_address
from backend.chat.persona_register import PersonaRegister
from backend.errors import CapabilityConfigError

PROFILE = "tester"
CASUAL_LINE = "Keep a casual but respectful register."
BOSS_LINE = (
    "Keep a casual but respectful register. "
    "Address the user as boss now and then, never in every reply."
)


def _profile(root: Path, text: str) -> Path:
    path = root / PROFILE / "personality.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")
    return path


class TestLoad:
    def test_the_files_address_form_is_returned(self, tmp_path: Path):
        _profile(tmp_path, "name: M.I.S.T\naddress_form: boss\n")

        assert PersonaRegister.load(tmp_path, PROFILE).address_form() == "boss"

    def test_a_null_address_form_is_none(self, tmp_path: Path):
        _profile(tmp_path, "name: M.I.S.T\naddress_form: null\n")

        assert PersonaRegister.load(tmp_path, PROFILE).address_form() is None

    def test_an_absent_address_form_is_none(self, tmp_path: Path):
        _profile(tmp_path, "name: M.I.S.T\n")

        assert PersonaRegister.load(tmp_path, PROFILE).address_form() is None

    def test_an_empty_file_has_no_address_form(self, tmp_path: Path):
        _profile(tmp_path, "")

        assert PersonaRegister.load(tmp_path, PROFILE).address_form() is None

    def test_a_blank_address_form_is_none(self, tmp_path: Path):
        _profile(tmp_path, "address_form: '  '\n")

        assert PersonaRegister.load(tmp_path, PROFILE).address_form() is None

    def test_a_missing_file_gives_no_form_and_one_warning_naming_the_path(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        path = tmp_path / PROFILE / "personality.yaml"

        with caplog.at_level(logging.DEBUG, logger="backend.chat.persona_register"):
            register = PersonaRegister.load(tmp_path, PROFILE)

        assert register.address_form() is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert str(path) in warnings[0].getMessage()

    def test_yaml_that_does_not_parse_raises_naming_the_file(self, tmp_path: Path):
        path = _profile(tmp_path, "address_form: [boss\n")

        with pytest.raises(CapabilityConfigError, match=str(path).replace("\\", "\\\\")):
            PersonaRegister.load(tmp_path, PROFILE)

    def test_a_file_that_is_not_a_mapping_raises_naming_the_file(self, tmp_path: Path):
        path = _profile(tmp_path, "- boss\n")

        with pytest.raises(CapabilityConfigError, match=str(path).replace("\\", "\\\\")):
            PersonaRegister.load(tmp_path, PROFILE)

    def test_an_address_form_that_is_not_text_raises_naming_the_file(self, tmp_path: Path):
        path = _profile(tmp_path, "address_form: 5\n")

        with pytest.raises(CapabilityConfigError, match=str(path).replace("\\", "\\\\")):
            PersonaRegister.load(tmp_path, PROFILE)


class TestReplyRegister:
    def test_capabilities_address_the_first_templated_reply_of_a_session_only(self, tmp_path: Path):
        _profile(tmp_path, "address_form: boss\n")
        register: ReplyRegister = PersonaRegister.load(tmp_path, PROFILE)
        templated = [ok_summary("timers.start", duration="five minutes"), ok_summary("music.pause")]

        replies = [
            with_address(reply, register, first_in_session=index == 0)
            for index, reply in enumerate(templated)
        ]

        assert replies == ["Timer set for five minutes, boss.", "Paused."]

    def test_no_form_leaves_the_first_reply_as_is(self, tmp_path: Path):
        _profile(tmp_path, "address_form: null\n")
        register: ReplyRegister = PersonaRegister.load(tmp_path, PROFILE)

        assert with_address("Paused.", register, first_in_session=True) == "Paused."


class TestPromptLine:
    def test_the_line_names_the_form(self, tmp_path: Path):
        _profile(tmp_path, "address_form: boss\n")

        assert PersonaRegister.load(tmp_path, PROFILE).prompt_line() == BOSS_LINE

    def test_without_a_form_the_line_is_the_casual_register_alone(self, tmp_path: Path):
        _profile(tmp_path, "address_form: null\n")

        assert PersonaRegister.load(tmp_path, PROFILE).prompt_line() == CASUAL_LINE

    def test_the_line_is_the_same_on_every_call(self, tmp_path: Path):
        _profile(tmp_path, "address_form: boss\n")
        register = PersonaRegister.load(tmp_path, PROFILE)

        assert register.prompt_line() == register.prompt_line()
