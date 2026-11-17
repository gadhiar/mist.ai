"""The persona's register: how MIST addresses the user, read from the persona's YAML.

Capabilities' templated replies and the conversation prompt both address the user. Both take the
form from one `PersonaRegister`, loaded from `<profiles_root>/<profile>/personality.yaml`, so the
spoken action replies and the model's own replies cannot name the user differently. Changing the
register means changing that YAML, not code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from backend.errors import CapabilityConfigError

logger = logging.getLogger(__name__)

PERSONALITY_FILE = "personality.yaml"

CASUAL_REGISTER = "Keep a casual but respectful register."
"""The register sentence every prompt carries, with or without an address form."""

_ADDRESS_SENTENCE = "Address the user as {form} now and then, never in every reply."


@dataclass(frozen=True, slots=True)
class PersonaRegister:
    """The address form of one persona; a capabilities `ReplyRegister`.

    Attributes:
        form: The address form ("boss"), or None when the persona uses none.
    """

    form: str | None

    @classmethod
    def load(cls, profiles_root: Path, profile: str) -> PersonaRegister:
        """Read `<profiles_root>/<profile>/personality.yaml`.

        A missing file is not fatal: the persona then has no address form, and one warning
        names the path, so a deployment without the file still answers, unaddressed.

        Raises:
            CapabilityConfigError: The file is not valid YAML, is not a mapping, or its
                `address_form` is neither text nor null. The message names the file.
        """
        path = profiles_root / profile / PERSONALITY_FILE
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning("No personality file at %s; the persona has no address form", path)
            return cls(form=None)
        try:
            config = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise CapabilityConfigError(f"{path} is not valid YAML: {exc}") from exc
        if config is None:
            return cls(form=None)
        if not isinstance(config, dict):
            raise CapabilityConfigError(f"{path} must be a mapping, not {type(config).__name__}")
        form = config.get("address_form")
        if form is None:
            return cls(form=None)
        if not isinstance(form, str):
            raise CapabilityConfigError(
                f"{path}: address_form must be text or null, not {type(form).__name__}"
            )
        # A blank form would put an empty name in the prompt line; treat it as none.
        return cls(form=form.strip() or None)

    def address_form(self) -> str | None:
        """Return the address form, or None when the persona uses none."""
        return self.form

    def prompt_line(self) -> str:
        """Return the register sentence for the prompt's stable prefix.

        It depends only on the loaded file, so it is the same on every turn and never moves
        the cached prefix.
        """
        if self.form is None:
            return CASUAL_REGISTER
        return f"{CASUAL_REGISTER} {_ADDRESS_SENTENCE.format(form=self.form)}"
