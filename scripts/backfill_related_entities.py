"""One-shot: backfill related_entities frontmatter on session notes from
graph DERIVED_FROM provenance edges.

Usage:
    python -m scripts.backfill_related_entities [--dry-run]

Connects to Neo4j via the project's Neo4jConnection + KnowledgeConfig.
Operates on the MIST vault sessions directory. Path format used for
VaultNote lookups is the container-side path (/app/mist-memory/sessions/*)
which is how the graph writes them; the script derives this from the
filename so it works even when run from the host.

The vault root on the host is D:/Users/rajga/mist.ai/mist-memory/.
The container sees the same directory as /app/mist-memory/ via bind mount.
"""

import argparse
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load .env from mist.ai root (host cwd or explicit path)
_ENV_PATH = Path("D:/Users/rajga/mist.ai/.env")
if _ENV_PATH.exists():
    load_dotenv(str(_ENV_PATH))
else:
    load_dotenv()

# mist-phase-3 repo root must be on sys.path for backend.* imports
_REPO_ROOT = Path("D:/Users/rajga/mist-phase-3")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.knowledge.config import KnowledgeConfig
from backend.knowledge.storage.neo4j_connection import Neo4jConnection

# Container-side vault root (how paths are stored in the graph).
# Must match MIST_VAULT_ROOT as seen from inside the container.
_CONTAINER_VAULT_ROOT = "/app/mist-memory"

# Host-side vault sessions directory
_HOST_SESSIONS_DIR = Path("D:/Users/rajga/mist.ai/mist-memory/sessions")


def collect_entities_for_path(connection: Neo4jConnection, container_path: str) -> list[str]:
    """Return sorted list of entity_id values derived from a VaultNote path.

    Args:
        connection: Open Neo4j connection.
        container_path: Container-side absolute path to the vault session note
            (e.g. /app/mist-memory/sessions/2026-05-07-plan-....md). This
            matches the `path` property on VaultNote provenance nodes.

    Returns:
        Sorted list of entity_id strings. Empty list when no edges exist.
    """
    cypher = (
        "MATCH (e:__Entity__)-[:DERIVED_FROM]->(p:__Provenance__:VaultNote {path: $path}) "
        "RETURN DISTINCT e.id AS entity_id "
        "ORDER BY entity_id"
    )
    rows = connection.execute_query(cypher, {"path": container_path})
    return [r["entity_id"] for r in rows if r.get("entity_id")]


def update_frontmatter_related(file_path: Path, entities: list[str]) -> bool:
    """Rewrite related_entities frontmatter field with wikilink list.

    Writes atomically via a sibling .tmp file + rename. Reads YAML
    frontmatter, checks whether the field already matches, and skips the
    write when no change is needed (idempotent).

    Args:
        file_path: Absolute path to the vault session note.
        entities: Sorted list of entity_id strings to embed as wikilinks.

    Returns:
        True when the file was modified, False when it was already
        up-to-date or the file had no parseable frontmatter block.
    """
    text = file_path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return False
    end_idx = text.find("\n---\n", 4)
    if end_idx == -1:
        return False
    fm_yaml = text[4:end_idx]
    body = text[end_idx + 5 :]
    fm = yaml.safe_load(fm_yaml) or {}
    new_related: list[str] = [f"[[{e}]]" for e in entities]
    if fm.get("related_entities") == new_related:
        return False
    fm["related_entities"] = new_related
    new_content = "---\n" + yaml.safe_dump(fm, sort_keys=False, allow_unicode=True) + "---\n" + body
    tmp = file_path.with_suffix(file_path.suffix + ".tmp")
    tmp.write_text(new_content, encoding="utf-8")
    tmp.replace(file_path)
    return True


def main(dry_run: bool) -> int:
    """Backfill related_entities on surviving session notes.

    Args:
        dry_run: When True, prints results without modifying files.

    Returns:
        0 on success, 1 on configuration / connectivity error.
    """
    if not _HOST_SESSIONS_DIR.exists():
        print(f"ERROR: sessions dir not found at {_HOST_SESSIONS_DIR}")
        return 1

    config = KnowledgeConfig.from_env()
    conn = Neo4jConnection(config.neo4j)
    try:
        conn.connect()
    except Exception as exc:
        print(f"ERROR: cannot connect to Neo4j at {config.neo4j.uri}: {exc}")
        return 1

    session_files = sorted(_HOST_SESSIONS_DIR.glob("*.md"))
    if not session_files:
        print("WARNING: no .md files found in sessions directory")
        conn.disconnect()
        return 0

    any_changed = False
    all_zero = True

    try:
        for file_path in session_files:
            container_path = f"{_CONTAINER_VAULT_ROOT}/sessions/{file_path.name}"
            entities = collect_entities_for_path(conn, container_path)
            if entities:
                all_zero = False
            if dry_run:
                sample = entities[:5]
                ellipsis = "..." if len(entities) > 5 else ""
                print(f"{file_path.name}: {len(entities)} entities -> {sample}{ellipsis}")
            else:
                changed = update_frontmatter_related(file_path, entities)
                if changed:
                    any_changed = True
                print(
                    f"{file_path.name}: {len(entities)} entities"
                    f" ({'changed' if changed else 'unchanged'})"
                )
    finally:
        conn.disconnect()

    if dry_run:
        if all_zero:
            print(
                "\nWARNING: all sessions returned zero entities from graph. "
                "DERIVED_FROM edges may not exist for these vault paths. "
                "Cliff accepted per 2026-05-10 decision; backfill cannot proceed."
            )
    else:
        if not any_changed:
            print("\nNo files modified (all already up-to-date or all zero-entity).")
        if all_zero:
            print(
                "\nWARNING: all sessions returned zero entities from graph. "
                "No vault changes written."
            )

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill related_entities frontmatter from graph DERIVED_FROM edges."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print per-session entity counts without modifying any files.",
    )
    args = parser.parse_args()
    sys.exit(main(args.dry_run))
