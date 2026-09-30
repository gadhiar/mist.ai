"""The `retrieve` subcommand's `--user-id` default is the stored user node id.

`cmd_retrieve` passes `args.user_id` to `KnowledgeRetriever.retrieve`, which
anchors on `GraphStore.get_user_relationships_to_entities`
(`MATCH (user:__Entity__ {id: $user_id})`). Neo4j matches ids
case-sensitively, so the former default "User" matched no node and the
command's user-scoped graph arm returned nothing (MIS-177 review).
"""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/ is not a package; insert it so mist_admin is importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import mist_admin  # noqa: E402  -- after sys.path insertion

from backend.knowledge.storage.partitions import USER_ENTITY_ID  # noqa: E402


def test_retrieve_user_id_defaults_to_the_stored_user_node_id() -> None:
    args = mist_admin.build_parser().parse_args(["retrieve", "what do I use"])

    assert args.user_id == USER_ENTITY_ID == "user"


def test_retrieve_user_id_is_still_overridable() -> None:
    args = mist_admin.build_parser().parse_args(["retrieve", "q", "--user-id", "someone"])

    assert args.user_id == "someone"
