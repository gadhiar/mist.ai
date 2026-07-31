"""Read and validate the versioned seed source from the vault."""

import logging
from pathlib import Path

from pydantic import ValidationError

from backend.errors import SeedSourceError
from backend.vault.models import parse_frontmatter

from .models import SeedDocument, SeedFact

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
            malformed, or the documents disagree on `seed_version`. One global
            version is the contract (spec O10); disagreement is a bug rather
            than something to reconcile silently.
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

        docs.append(
            SeedDocument(seed_version=str(version), facts=facts, body=body, source_path=path)
        )

    if not docs:
        raise SeedSourceError(f"Found no seed documents in {seed_dir}")

    versions = {d.seed_version for d in docs}
    if len(versions) > 1:
        raise SeedSourceError(
            f"Seed documents disagree on seed_version: {sorted(versions)}. "
            "One global version is the contract."
        )

    return docs
