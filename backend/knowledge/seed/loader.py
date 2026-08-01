"""Read and validate the versioned seed source from the vault."""

import difflib
import logging
from pathlib import Path

from pydantic import ValidationError

from backend.errors import SeedSourceError
from backend.knowledge.ontologies import ALL_NODE_TYPE_NAMES
from backend.knowledge.storage.partitions import ENTITY_LABEL
from backend.vault.models import parse_frontmatter

from .models import SeedDocument, SeedFact, SeedNode

logger = logging.getLogger(__name__)

_SEED_TYPE = "mist-seed"


def load_seed_documents(seed_dir: Path) -> list[SeedDocument]:
    """Load every `mist-seed` markdown document under `seed_dir`.

    Args:
        seed_dir: Directory holding the seed source (`mist-memory/seed/`).

    Returns:
        Documents sorted by filename, so application order is deterministic.

    Raises:
        SeedSourceError: The directory is missing or empty, a document is
            malformed (including an unrecognized `partition` value -- see
            `SeedDocument.partition`), the documents disagree on
            `seed_version` (one global version is the contract, spec O10;
            disagreement is a bug rather than something to reconcile
            silently), a node's `type` is not a recognized ontology node
            type, a node `id` is defined more than once (within or across
            documents), a fact's `subject`/`object` has no matching node
            definition anywhere in the seed source (R1.4 Task 11 -- the
            exact shape of the Task 10 live defect: a fact referencing an
            undefined node used to write silently instead of failing to
            load), or a node is defined but referenced by no fact at all
            (R1.4 whole-branch review, I5 -- the applier writes nodes driven
            by fact references, not `doc.nodes` membership, so an
            unreferenced node would be silently written nowhere).
    """
    if not seed_dir.is_dir():
        raise SeedSourceError(f"Seed directory does not exist: {seed_dir}")

    docs: list[SeedDocument] = []
    for path in sorted(seed_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(text)

        # `parse_frontmatter` swallows yaml.YAMLError and returns `{}`, which
        # is indistinguishable from "this file never had frontmatter" -- see
        # backend/vault/models.py:151-155. It is also indistinguishable from
        # well-formed YAML that legitimately resolves to no keys (e.g. a
        # frontmatter block containing only comments), so this message does
        # not claim a syntax error -- it may have parsed fine and simply had
        # nothing in it. For a dedicated seed directory every `.md` file is
        # expected to carry frontmatter, so a file that opens with the `---`
        # delimiter but comes back with an empty dict did not silently opt
        # out of being a seed doc. Fail loudly here rather than let it fall
        # through to the type check below and vanish as a silently-skipped
        # "non-seed" file.
        if text.startswith("---") and not fm:
            raise SeedSourceError(f"{path}: frontmatter is present but resolved to no keys")

        if fm.get("type") != _SEED_TYPE:
            logger.debug("Skipping non-seed document %s (type=%r)", path, fm.get("type"))
            continue

        version = fm.get("seed_version")
        if not version:
            raise SeedSourceError(f"{path}: missing `seed_version`")

        try:
            nodes = [SeedNode(**n) for n in fm.get("nodes", [])]
        except (ValidationError, TypeError) as exc:
            raise SeedSourceError(f"{path}: invalid `nodes` entry: {exc}") from exc

        try:
            facts = [SeedFact(**f) for f in fm.get("facts", [])]
        except (ValidationError, TypeError) as exc:
            raise SeedSourceError(f"{path}: invalid `facts` entry: {exc}") from exc

        if not facts:
            # Legitimate case (prose-only seed content, e.g. an identity
            # narrative with no typed assertions) -- Gate 3's containment
            # check passes vacuously with nothing to contain. Logged rather
            # than silent so a doc that *should* have facts and lost them to
            # a typo in the `facts:` key itself (which `extra="forbid"` on
            # SeedFact cannot catch -- that typo never reaches SeedFact) is
            # still visible at load time.
            logger.info("Seed document %s has no facts (prose-only)", path)

        partition = fm.get("partition", ENTITY_LABEL)

        try:
            doc = SeedDocument(
                seed_version=str(version),
                nodes=nodes,
                facts=facts,
                body=body,
                source_path=path,
                partition=partition,
            )
        except ValidationError as exc:
            raise SeedSourceError(f"{path}: invalid `partition` {partition!r}: {exc}") from exc
        docs.append(doc)

    if not docs:
        raise SeedSourceError(f"Found no seed documents in {seed_dir}")

    versions = {d.seed_version for d in docs}
    if len(versions) > 1:
        raise SeedSourceError(
            f"Seed documents disagree on seed_version: {sorted(versions)}. "
            "One global version is the contract."
        )

    # R1.4 Task 11 (ADDENDUM): whole-corpus node/fact validation, run only
    # after every document has parsed successfully and the version check has
    # passed -- these three checks operate across ALL documents together
    # (a fact in one file may reference a node defined in another), so
    # per-document validation cannot do them.
    _validate_node_types(docs)
    _validate_unique_node_ids(docs)
    _validate_referential_integrity(docs)
    _validate_no_unreferenced_node_definitions(docs)

    return docs


def _validate_node_types(documents: list[SeedDocument]) -> None:
    """Reject any node whose `type` is not a known ontology node type.

    Mirrors `applier.py`'s `_validate_predicates` in shape (same
    closest-match-suggestion pattern) but lives here, at load time, rather
    than only at the Cypher-interpolation boundary -- unlike `predicate`,
    which validates only in `applier.py` (Task 4 found that validating it
    here would break Task 1's own fixture, which deliberately uses a
    predicate outside the ontology for loader-only testing). `type` has no
    such conflict, and Task 12's applier-side check is intentionally
    redundant with this one, for the same reason `_validate_predicates`
    guards the Cypher interpolation site directly rather than trusting an
    upstream check alone.

    Args:
        documents: Parsed seed documents to validate.

    Raises:
        SeedSourceError: A node's `type` is not in `ALL_NODE_TYPE_NAMES`,
            naming the type, the node id, the source file, and the closest
            allowed type if there is an obvious near-match.
    """
    allowed = set(ALL_NODE_TYPE_NAMES)
    for doc in documents:
        for node in doc.nodes:
            if node.type in allowed:
                continue
            suggestion = difflib.get_close_matches(node.type, ALL_NODE_TYPE_NAMES, n=1)
            hint = f" Closest allowed type: {suggestion[0]!r}." if suggestion else ""
            raise SeedSourceError(
                f"{doc.source_path}: node {node.id!r} has unknown type {node.type!r}, "
                f"not a recognized ontology node type.{hint}"
            )


def _validate_unique_node_ids(documents: list[SeedDocument]) -> None:
    """Reject a node `id` defined more than once, within or across documents.

    Two documents (or two entries in one document) both claiming the same
    node id is an authoring conflict -- the same id cannot mean two
    different node definitions, and silently letting the last one win would
    make the seed source's meaning depend on file iteration order.

    Args:
        documents: Parsed seed documents to validate.

    Raises:
        SeedSourceError: A node id appears more than once, naming the id
            and both source files.
    """
    seen: dict[str, Path] = {}
    for doc in documents:
        for node in doc.nodes:
            first_seen = seen.get(node.id)
            if first_seen is not None:
                raise SeedSourceError(
                    f"{doc.source_path}: node id {node.id!r} is already defined in "
                    f"{first_seen} -- node ids must be unique across the whole seed source"
                )
            seen[node.id] = doc.source_path


def _validate_referential_integrity(documents: list[SeedDocument]) -> None:
    """Every fact's `subject` and `object` must have a matching `SeedNode`.

    This is the exact shape of the R1.4 Task 10 live defect: a fact
    referencing a node id the seed source never defines used to write
    silently -- the applier's `_MERGE_NODE` stamped `seed_version`/
    `created_at`/`updated_at` on whatever id it was handed and moved on,
    producing a graph node with no ontology label and no descriptive
    properties. Failing loudly here, before any graph write is attempted,
    is what would have caught it. Checked across the WHOLE corpus (a fact
    in `user.md` may reference a node defined in `mist.md`, or vice versa),
    not per-document.

    Args:
        documents: Parsed seed documents to validate.

    Raises:
        SeedSourceError: A fact's `subject` or `object` has no matching
            `SeedNode.id` anywhere in `documents`, naming the missing id,
            which role it played (subject/object), the offending fact, and
            the source file.
    """
    all_ids = {node.id for doc in documents for node in doc.nodes}
    for doc in documents:
        for fact in doc.facts:
            for role, node_id in (("subject", fact.subject), ("object", fact.object)):
                if node_id in all_ids:
                    continue
                raise SeedSourceError(
                    f"{doc.source_path}: fact {role} {node_id!r} "
                    f"({fact.subject} {fact.predicate} {fact.object}) has no matching "
                    "node definition -- every fact's subject and object must appear as "
                    "a node id somewhere in the seed source's `nodes:` blocks"
                )


def _validate_no_unreferenced_node_definitions(documents: list[SeedDocument]) -> None:
    """Every defined `SeedNode` must be referenced by at least one fact.

    R1.4 whole-branch review, I5: `apply_seed_documents` writes nodes driven
    by FACT references (`_assign_node_partitions`'s output, unchanged since
    Task 4) -- not by `doc.nodes` membership -- while `check_node_definitions`
    (Gate 4, Task 14) iterates `doc.nodes` directly. The two lists agree
    today (32 fact-referenced ids, 32 defined nodes, zero difference either
    way, verified live) but nothing enforces that BY CONSTRUCTION: the model
    fully permits defining a node with no fact naming it as subject or
    object. Such a node would be silently written nowhere -- the applier
    never visits an id `_assign_node_partitions` did not produce -- and the
    live graph would then genuinely fail `check_node_definitions` for that
    id, not because either the write path or the gate has a bug, but
    because the source authored something the write path was never going
    to act on. Rejecting it here at load time keeps `doc.nodes` and "what
    the applier writes" a single authority instead of two lists that
    happen, so far, to agree.

    The complementary direction (`_validate_referential_integrity`, a fact
    referencing a node with no definition) was Task 11's original headline
    check; this is the reverse gap, found by the whole-branch review rather
    than by symmetry with that check at the time.

    Checked across the whole corpus, like `_validate_referential_integrity`
    -- a node defined in one document may legitimately be referenced only
    by a fact in another.

    Args:
        documents: Parsed seed documents to validate.

    Raises:
        SeedSourceError: A `SeedNode.id` is defined but never appears as a
            fact's subject or object anywhere in `documents`, naming the id
            and its source file.
    """
    referenced_ids = {
        node_id
        for doc in documents
        for fact in doc.facts
        for node_id in (fact.subject, fact.object)
    }
    for doc in documents:
        for node in doc.nodes:
            if node.id in referenced_ids:
                continue
            raise SeedSourceError(
                f"{doc.source_path}: node {node.id!r} is defined but no fact "
                "references it as subject or object -- the applier writes nodes "
                "driven by fact references, so an unreferenced node definition "
                "would be silently written nowhere (R1.4 whole-branch review, I5)"
            )
