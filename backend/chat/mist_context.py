"""Structured holder for MIST's seeded identity, plus system-prompt renderer.

Populated by KnowledgeRetriever.retrieve_mist_context() from the graph's
MistIdentity node and outgoing HAS_TRAIT/HAS_CAPABILITY/HAS_PREFERENCE edges.
Consumed by ConversationHandler to prepend a persona block to the system
prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MistTrait:
    """A personality or operational axis trait seeded onto the MistIdentity node."""

    id: str
    display_name: str
    axis: str  # "Persona" | "Platform"
    description: str


@dataclass(frozen=True)
class MistCapability:
    """A functional capability seeded onto the MistIdentity node."""

    id: str
    display_name: str
    description: str


@dataclass(frozen=True)
class MistPreference:
    """A behavioral preference seeded onto the MistIdentity node."""

    id: str
    display_name: str
    enforcement: str  # "absolute" | "preferred" | "informational"
    context: str


@dataclass(frozen=True)
class MistContext:
    """Aggregated identity snapshot returned by KnowledgeRetriever.retrieve_mist_context()."""

    display_name: str
    pronouns: str
    self_concept: str
    traits: list[MistTrait] = field(default_factory=list)
    capabilities: list[MistCapability] = field(default_factory=list)
    preferences: list[MistPreference] = field(default_factory=list)
    # Forward-compat for ADR-010 (Cluster 8 vault-owned identity). Safe
    # defaults preserve today's behavior; populated by Cluster 8 when
    # MistContext is derived from mist-memory/identity/mist.md.
    vault_note_path: str | None = None
    authored_by: str = (
        "mist"  # "mist" | "mist-pending-review" | "user" | "user-edit" | "user-rejected"
    )

    def as_system_prompt_block(self) -> str:
        """Render the persona as a system-prompt block with hard-constraint framing."""
        lines: list[str] = []
        lines.append(f"You are {self.display_name}.")
        if self.pronouns:
            lines.append(f"Pronouns: {self.pronouns}.")
        if self.self_concept:
            lines.append(self.self_concept.strip())
        lines.append("")

        absolute = [p for p in self.preferences if p.enforcement == "absolute"]
        non_absolute = [p for p in self.preferences if p.enforcement != "absolute"]

        if absolute:
            lines.append(
                "HARD RULES (non-negotiable — violating these breaks the response contract):"
            )
            for p in absolute:
                lines.append(f"- {p.display_name}. {p.context.strip()}")
            lines.append("")

        if self.traits:
            lines.append("Traits:")
            for t in self.traits:
                lines.append(f"- {t.display_name} ({t.axis}): {t.description.strip()}")
            lines.append("")

        if self.capabilities:
            lines.append("Capabilities:")
            for c in self.capabilities:
                lines.append(f"- {c.display_name}: {c.description.strip()}")
            lines.append("")

        if non_absolute:
            lines.append("Preferences (soft guidance):")
            for p in non_absolute:
                lines.append(f"- {p.display_name}. {p.context.strip()}")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"
