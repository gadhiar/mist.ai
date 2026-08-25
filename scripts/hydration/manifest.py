"""Producer identity for a hydration artifact -- what makes a stale one detectable.

A hydration artifact is a photograph of a graph that 87 LLM turns produced. It
stays correct only while the things that produced it are unchanged, and the
failure mode it must not have is the silent one: a developer restores an
artifact built under an older extraction prompt, gets a graph that looks
plausible, and debugs the fixture instead of the code.

So the artifact records its PRODUCER INPUTS, and restore recomputes them:

| Input                | Source                                   |
|----------------------|------------------------------------------|
| `ontology_version`   | `version_stamps.ONTOLOGY_VERSION`        |
| `extraction_version` | `version_stamps.EXTRACTION_VERSION`      |
| `model_hash`         | `version_stamps.compose_model_hash`      |
| corpus digest        | sha256 over the corpus file(s)           |

The first three used to be exactly the triple `extraction_cache.cache_key`
hashed. As of `extraction-cache-phase-1` spec D3, that equivalence no longer
holds: `cache_key` hashes only `event_id|extraction_version|model_hash`
(verified via `grep -n 'raw = "|".join' backend/knowledge/extraction_cache.py`),
so an `ontology_version`-only change no longer misses the extraction cache. It
still makes a hydration artifact STALE, because the check below compares all
three producer inputs DIRECTLY against the single authority -- unlike the
extraction cache, this comparison was never routed through `cache_key`, so D3
does not touch it. They are READ FROM THE SINGLE AUTHORITY, never restated
here. That is what makes staleness detection require no discipline: bumping
`EXTRACTION_VERSION` in `backend/knowledge/version_stamps.py` invalidates every
artifact on the next restore automatically. The person bumping it does not have
to know this file exists -- which is the only kind of invalidation that
survives contact with a real project.

The corpus digest is the fourth input because the first three cannot see it. A
re-cut turn schedule with the same prompt and the same ontology produces a
different graph and would otherwise pass every version check.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from backend.errors import MistError
from backend.knowledge.version_stamps import (
    EXTRACTION_VERSION,
    ONTOLOGY_VERSION,
    compose_model_hash,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# The corpus hydration replays. Default is R1.4.5's golden log; T2 may re-cut it
# into a multi-session corpus, and the digest below is what notices when it does.
DEFAULT_CORPUS_PATH = REPO_ROOT / "data" / "golden-log" / "golden-log.jsonl"

# Layout version of the artifact directory itself. Bump when the set of files or
# the shape of graph.json changes, so an old artifact is refused by structure
# rather than crashing a newer restore halfway through a destructive load.
ARTIFACT_SCHEMA_VERSION = 1

MANIFEST_FILENAME = "manifest.json"


class HydrationError(MistError):
    """Raised when a hydration artifact cannot be produced, read, or trusted."""


def _hash_file(path: Path) -> str:
    """sha256 of one file, read in chunks so a large corpus does not load whole."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_corpus(corpus_path: Path) -> tuple[str, int]:
    """Return `(sha256, file_count)` for a corpus file or corpus directory.

    A directory digest folds each file's own digest together with its POSIX
    relative path, so a rename is a change (it is -- turn ordering in this
    corpus is carried by filename) and platform path separators are not.

    Raises:
        HydrationError: When the corpus does not exist, or a directory corpus
            holds no files. An artifact that cannot name its corpus cannot be
            checked against one later, and recording "unknown" would defeat the
            whole mechanism.
    """
    if not corpus_path.exists():
        raise HydrationError(
            f"corpus {corpus_path} does not exist; refusing to record an artifact "
            "identity with no corpus digest, which would make every later "
            "staleness check vacuous"
        )
    if corpus_path.is_file():
        return _hash_file(corpus_path), 1

    files = sorted(p for p in corpus_path.rglob("*") if p.is_file())
    if not files:
        raise HydrationError(f"corpus directory {corpus_path} holds no files")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(corpus_path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_hash_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest(), len(files)


@dataclass(frozen=True, slots=True)
class SnapshotIdentity:
    """The four producer inputs an artifact is only valid against."""

    ontology_version: str
    extraction_version: str
    model_hash: str
    corpus_path: str
    corpus_sha256: str
    corpus_file_count: int

    @classmethod
    def current(cls, corpus_path: Path | None = None) -> SnapshotIdentity:
        """Read the identity of the CURRENT tree.

        The stamps come from `backend.knowledge.version_stamps`. There is
        deliberately no parameter and no literal for any of the three: a
        restated value could disagree with the single authority. A restated
        `extraction_version` or `model_hash` would also miss the extraction
        cache; `ontology_version` no longer carries that consequence (spec D3),
        but restating it would still reintroduce the drift collapsing the four
        authorities removed.
        """
        # Imported here rather than at module scope: `KnowledgeConfig.from_env`
        # reads the environment, and a manifest module that touches env at
        # import time would make test isolation depend on import order.
        from backend.knowledge.config import KnowledgeConfig

        path = corpus_path or DEFAULT_CORPUS_PATH
        corpus_sha256, corpus_file_count = digest_corpus(path)
        try:
            relative = path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            relative = path.resolve().as_posix()
        return cls(
            ontology_version=ONTOLOGY_VERSION,
            extraction_version=EXTRACTION_VERSION,
            model_hash=compose_model_hash(KnowledgeConfig.from_env()),
            corpus_path=relative,
            corpus_sha256=corpus_sha256,
            corpus_file_count=corpus_file_count,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the identity as a plain dict for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> SnapshotIdentity:
        """Rebuild an identity from a manifest dict, rejecting pre-schema artifacts."""
        missing = {f for f in cls.__slots__ if f not in raw}
        if missing:
            raise HydrationError(
                f"manifest identity is missing {sorted(missing)}; the artifact "
                "predates the current manifest schema and cannot be checked for "
                "staleness. Re-run hydration."
            )
        return cls(
            ontology_version=str(raw["ontology_version"]),
            extraction_version=str(raw["extraction_version"]),
            model_hash=str(raw["model_hash"]),
            corpus_path=str(raw["corpus_path"]),
            corpus_sha256=str(raw["corpus_sha256"]),
            corpus_file_count=int(raw["corpus_file_count"]),  # type: ignore[arg-type]
        )

    def drift_against(self, other: SnapshotIdentity) -> list[str]:
        """Return the field names on which `self` and `other` disagree.

        `corpus_path` is compared too. A corpus that moved is a corpus change
        as far as an artifact is concerned -- the digest would be computed over
        a different file and coincidental equality would be meaningless.
        """
        return [field for field in self.__slots__ if getattr(self, field) != getattr(other, field)]


@dataclass(frozen=True, slots=True)
class SnapshotManifest:
    """An artifact's identity plus what it should contain when restored."""

    artifact_schema_version: int
    created_at: str
    identity: SnapshotIdentity
    contents: dict[str, int]
    source: dict[str, str]

    def to_json(self) -> str:
        """Serialize the manifest deterministically (sorted keys, trailing newline)."""
        payload = {
            "artifact_schema_version": self.artifact_schema_version,
            "created_at": self.created_at,
            "identity": self.identity.to_dict(),
            "contents": self.contents,
            "source": self.source,
        }
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"

    def write(self, artifact_dir: Path) -> Path:
        """Write the manifest into `artifact_dir`. Returns the path written."""
        path = artifact_dir / MANIFEST_FILENAME
        path.write_text(self.to_json(), encoding="utf-8", newline="\n")
        return path

    @classmethod
    def read(cls, artifact_dir: Path) -> SnapshotManifest:
        """Load and validate the manifest in `artifact_dir`."""
        path = artifact_dir / MANIFEST_FILENAME
        if not path.exists():
            raise HydrationError(
                f"{path} not found; an artifact directory without a manifest cannot "
                "state what produced it, so restoring it could only ever be a guess"
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        schema = int(raw.get("artifact_schema_version", -1))
        if schema != ARTIFACT_SCHEMA_VERSION:
            raise HydrationError(
                f"{path} has artifact_schema_version {schema}, this tree expects "
                f"{ARTIFACT_SCHEMA_VERSION}. The artifact layout changed; re-run "
                "hydration rather than restoring a shape this code cannot read."
            )
        return cls(
            artifact_schema_version=schema,
            created_at=str(raw["created_at"]),
            identity=SnapshotIdentity.from_dict(raw["identity"]),
            contents={str(k): int(v) for k, v in raw.get("contents", {}).items()},
            source={str(k): str(v) for k, v in raw.get("source", {}).items()},
        )


def assert_fresh(manifest: SnapshotManifest, corpus_path: Path | None = None) -> None:
    """Refuse an artifact whose producer inputs no longer match this tree.

    Fail-closed and, by default, unskippable: a restore of a stale artifact is
    not a degraded restore, it is a fixture that lies. The CLI exposes
    `--allow-stale` for the one legitimate case (wanting SOME populated dev
    state, knowing it does not correspond to current code), and it prints the
    drift rather than hiding it.

    Raises:
        HydrationError: When any producer input differs, naming each one.
    """
    current = SnapshotIdentity.current(corpus_path)
    drift = manifest.identity.drift_against(current)
    if not drift:
        return
    lines = [
        f"  {field}: artifact={getattr(manifest.identity, field)!r} "
        f"current={getattr(current, field)!r}"
        for field in drift
    ]
    raise HydrationError(
        "hydration artifact is STALE -- it was produced by inputs this tree no "
        "longer has:\n"
        + "\n".join(lines)
        + "\nRe-run hydration to regenerate it, or pass --allow-stale to restore "
        "it anyway knowing the graph does not correspond to current code."
    )
