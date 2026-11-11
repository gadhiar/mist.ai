"""The base fixture pack's manifest."""

from backend.knowledge.ontologies.pack import Manifest, RetiredType

MANIFEST = Manifest(
    version="0.1.0",
    fallback_type="Deckhand",
    user_id="user",
    user_type="Deckhand",
    retired_types=(RetiredType("Stowaway", "Deckhand", (("rank", "none"),)),),
    escape_edge="ALONGSIDE",
    prompt_token_budget=2000,
)
