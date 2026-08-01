"""Tests for backend.knowledge.embeddings.embedding_text.

The separator this module owns is load-bearing in a way no presence-based
check can see. Every vector currently stored in the live graph (32/32
nodes, verified read-only before I7 began) was computed from text joined
with `" — "` -- U+2014 EM DASH, one space either side. Change that
join and every stored vector silently becomes the embedding of text that
is no longer authored anywhere, while every dimension check, every
non-null check and every norm check keeps passing. So it is pinned here
by explicit codepoint rather than by a literal character copied out of
the source file: a mojibake round-trip through a non-UTF-8 tool would
otherwise rewrite the expectation and the assertion in lockstep, and the
test would keep passing against a separator nobody chose.

The last two tests pin the other half of I7's Task 1: the two production
backfills embed exactly what this one builder returns. They were separate
copies of the same three lines until now, which is the shape C1 came in
through -- one question ("what text represents this node?") answered
independently in two places, and the answers drifting apart with nothing
able to see it.
"""

from backend.knowledge.embeddings.embedding_text import embedding_text_for
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection

# Built by codepoint, deliberately not copied as a literal character.
EM_DASH_JOIN = " " + chr(0x2014) + " "


class TestEmbeddingTextFor:
    def test_joins_display_name_and_description_with_a_spaced_em_dash(self):
        """The separator is load-bearing: it is what every vector in the live
        graph was computed with, and no presence-based, dimension-based or
        norm-based check can see it change.
        """
        text = embedding_text_for("Slalom", "Consulting firm", "slalom")

        assert text == "Slalom" + EM_DASH_JOIN + "Consulting firm"

    def test_returns_display_name_alone_when_description_is_absent(self):
        text = embedding_text_for("Python", None, "python")

        assert text == "Python"

    def test_omits_an_empty_description(self):
        """`seed/user.md` authors no `description` on any of its 11 nodes, so
        the no-description path is the majority path against real source, not
        an edge case.
        """
        text = embedding_text_for("Python", "", "python")

        assert text == "Python"

    def test_falls_back_to_the_node_id_when_display_name_is_none(self):
        """Task 10's documented trade-off: `reseed()`'s delete+recreate cycle
        drops `display_name`, so the backfill that runs after it may see only
        an id. A lower-quality embedding text, but a real one.
        """
        text = embedding_text_for(None, None, "mist-identity")

        assert text == "mist-identity"

    def test_falls_back_to_the_node_id_when_display_name_is_empty(self):
        """The production sites used `row["display_name"] or row["id"]`, which
        falls back on empty string as well as on None. Pinned so a rewrite to
        `if display_name is None` does not silently start embedding "".
        """
        text = embedding_text_for("", None, "mist-identity")

        assert text == "mist-identity"

    def test_appends_the_description_to_the_fallback_id(self):
        text = embedding_text_for(None, "Consulting firm", "slalom")

        assert text == "slalom" + EM_DASH_JOIN + "Consulting firm"


class TestBackfillsShareTheBuilder:
    """Both production backfills must embed exactly what the shared builder
    returns -- asserted against the builder's own output, not against a
    hardcoded string, so changing the separator in one place and not the
    other fails here rather than shipping.
    """

    def test_backfill_embeddings_embeds_exactly_the_shared_builders_text(self):
        from backend.knowledge import admin

        connection = FakeNeo4jConnection(
            query_results=[
                {
                    "id": "slalom",
                    "display_name": "Slalom",
                    "description": "Consulting firm",
                    "labels": ["__Entity__", "Organization"],
                }
            ]
        )
        generator = FakeEmbeddingGenerator()

        admin._backfill_embeddings(connection, generator)

        assert generator.calls == [embedding_text_for("Slalom", "Consulting firm", "slalom")]

    def test_backfill_embeddings_for_seed_embeds_exactly_the_shared_builders_text(self):
        from backend.knowledge import admin

        connection = FakeNeo4jConnection(
            query_results=[
                {"id": "slalom", "display_name": "Slalom", "description": "Consulting firm"}
            ]
        )
        generator = FakeEmbeddingGenerator()

        admin._backfill_embeddings_for_seed(connection, generator, "profile-v1")

        assert generator.calls == [embedding_text_for("Slalom", "Consulting firm", "slalom")]
