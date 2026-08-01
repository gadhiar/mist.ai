"""The one place that decides what text a graph node's embedding is computed from.

Extracted by I7 Task 1 from `admin._backfill_embeddings` and
`admin._backfill_embeddings_for_seed`, which had carried byte-identical
private copies of the same three lines since R1.4 Task 10 added the
second one. The verification gate `seed.gates.check_embeddings` would
have been the third copy, and a third copy is what this extraction
exists to prevent: R1.4's C1 regression was one question ("how does this
node appear?") answered independently in two places, the two answers
drifting apart, and nothing in the codebase able to see the drift --
`check_negation_proximity` reported `passed=True` having examined 0 of
20 facts because its copy of the answer had gone stale. The review's
remedy there was to extract `_search_term_for` and have both callers
share it; this module applies that remedy before the divergence rather
than after.

The stakes here are higher than for `_search_term_for`, because a change
to the join is invisible to every other check that exists. Dimension,
non-null and L2-norm checks all pass unchanged when the separator moves:
the vectors are still 384-d, still non-null, still unit-norm -- they are
simply the embeddings of text nobody authored. `check_embeddings`'
cosine-against-recomputed condition is the only thing that can see it,
and it can only see it because it recomputes through this function.
"""

from __future__ import annotations

# U+2014 EM DASH with exactly one space on each side. This is the join
# every vector in the live graph was computed with (32/32 nodes, verified
# read-only before I7 began), so it is a data-compatibility constant, not
# a formatting preference: changing it does not "reformat" anything, it
# silently invalidates every stored vector while leaving them structurally
# valid. Written as a codepoint rather than a pasted character so a
# non-UTF-8 round-trip of this file cannot quietly substitute a hyphen or
# an en dash. Pinned by `tests/unit/knowledge/test_embedding_text.py`.
_SEPARATOR = " " + chr(0x2014) + " "


def embedding_text_for(display_name: str | None, description: str | None, node_id: str) -> str:
    """Build the text to embed for one seeded graph node.

    `display_name` falls back to `node_id` when absent OR empty -- the
    fallback is reached in practice, not defensively: `reseed()`'s
    wipe-then-recreate cycle drops `display_name` and `description`
    (only `seed_version`/`created_at`/`updated_at` survive a Neo4j node
    deletion), so the backfill that runs immediately after a reseed can
    legitimately see a node with nothing but an id. R1.4 Task 10
    documented that as an accepted quality trade-off: a lower-quality
    embedding text, but a real one rather than an empty string, which
    `EmbeddingGenerator.generate_embedding` would turn into a zero
    vector -- structurally valid, semantically meaningless, and matching
    nothing at query time.

    `description` is appended only when truthy, so an authored-but-empty
    description does not produce a trailing separator.

    Args:
        display_name: The node's human-facing name, or None/empty.
        description: The node's descriptive prose, or None/empty. Only
            `seed/mist.md`'s 21 self-model nodes author one; none of
            `seed/user.md`'s 11 nodes do.
        node_id: The node's kebab id, used as the fallback subject when
            `display_name` is absent or empty.

    Returns:
        `"<display_name or node_id>"`, optionally followed by the
        separator and the description.
    """
    parts = [display_name or node_id]
    if description:
        parts.append(description)
    return _SEPARATOR.join(parts)
