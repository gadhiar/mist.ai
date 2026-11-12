"""What a score was taken under: the scorer's code, the gold and the ontology pack.

When a score moves, the reading must say whether the model, the rails or the scorer
moved it (plan v14 T50, design h.5). So every summary and every `score_runs` line
carries four fields, fixed before the first probe runs:

- `scorer_sha`: sha256 over the contents of every Python file under
  `scripts/eval_harness/` and of `backend/knowledge/extraction/rails.py` and `id_rails.py`,
  the engine's rails the scorer runs. The files are read in the order of their paths
  relative to the repository root, as POSIX strings, and their bytes are hashed one
  after the other with nothing between them. A file that does not exist is left out,
  so an older checkout without the engine rails still has a sha.
- `gold_version`: the gold file's sha, the LABELLING guide's revision and the pack's
  version, as `gold-<12 hex>.labelling-<12 hex>.pack-<version>`. The guide's revision is
  the sha of its bytes: the `LABELLING.md` beside the gold file, or the repository's
  `data/gold/extraction/LABELLING.md` when there is none beside it.
- `pack_prompt_hash` and `pack_rails_hash`: the active pack's two hashes.

A summary without any of the four is not a reading, and `require_reading` refuses it.

This module imports nothing from `scripts` or `backend`, so `score_runs.py` can load it
by path while it holds `scripts.*` and `backend.*` to a pinned code root.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

SCORER_DIR = Path("scripts") / "eval_harness"
ENGINE_RAILS: tuple[Path, ...] = (
    Path("backend") / "knowledge" / "extraction" / "rails.py",
    Path("backend") / "knowledge" / "extraction" / "id_rails.py",
)
REPOSITORY_LABELLING = Path("data") / "gold" / "extraction" / "LABELLING.md"
READING_FIELDS: tuple[str, ...] = (
    "scorer_sha",
    "gold_version",
    "pack_prompt_hash",
    "pack_rails_hash",
)
_SHORT = 12


class NotAReadingError(ValueError):
    """A summary that does not name the scorer, the gold and the pack it was taken under."""


def scorer_files(root: Path) -> list[Path]:
    """The files `scorer_sha` hashes, in the order it hashes them.

    Args:
        root: The repository root (or a code root laid out as one).
    """
    files = [path for path in (root / SCORER_DIR).rglob("*.py") if path.is_file()]
    files += [root / rel for rel in ENGINE_RAILS if (root / rel).is_file()]
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def scorer_sha(root: Path) -> str:
    """sha256 over the contents of `scorer_files(root)`, in that order."""
    digest = hashlib.sha256()
    for path in scorer_files(root):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _file_sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def labelling_path(gold: Path, root: Path) -> Path:
    """The LABELLING guide the gold was labelled under: beside it, else the repository's."""
    beside = gold.parent / "LABELLING.md"
    return beside if beside.is_file() else root / REPOSITORY_LABELLING


def gold_version(gold: Path, root: Path, pack_version: str | None) -> str:
    """The gold file's sha, the guide's revision and the pack version, as one string.

    Args:
        gold: The gold file.
        root: The repository root, whose guide stands in when none is beside the gold.
        pack_version: The active pack's version; None when the code has no pack.
    """
    gold_sha = _file_sha(gold)
    guide_sha = _file_sha(labelling_path(gold, root))
    return (
        f"gold-{gold_sha[:_SHORT] if gold_sha else 'none'}"
        f".labelling-{guide_sha[:_SHORT] if guide_sha else 'none'}"
        f".pack-{pack_version or 'none'}"
    )


def reading_identity(
    *,
    root: Path,
    gold: Path,
    pack_version: str | None,
    pack_prompt_hash: str | None,
    pack_rails_hash: str | None,
) -> dict[str, str | None]:
    """The four fields a reading carries, in `READING_FIELDS` order."""
    return {
        "scorer_sha": scorer_sha(root),
        "gold_version": gold_version(gold, root, pack_version),
        "pack_prompt_hash": pack_prompt_hash,
        "pack_rails_hash": pack_rails_hash,
    }


def identity_text(identity: Mapping[str, object]) -> str:
    """The fields as `name=value` pairs, `none` for an absent value, for a printed line."""
    return " ".join(
        f"{name}={identity.get(name) if identity.get(name) else 'none'}" for name in READING_FIELDS
    )


def missing_fields(summary: Mapping[str, object]) -> list[str]:
    """The fields of `READING_FIELDS` the summary lacks or holds empty."""
    return [name for name in READING_FIELDS if not summary.get(name)]


def require_reading(summary: Mapping[str, object]) -> None:
    """Refuse a summary that does not carry all four fields.

    Raises:
        NotAReadingError: Naming each field that is absent or empty.
    """
    missing = missing_fields(summary)
    if missing:
        raise NotAReadingError(
            f"not a reading: the summary carries no {', '.join(missing)}; a score that does "
            "not name its scorer, gold and pack cannot be compared"
        )
