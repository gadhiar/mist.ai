"""The boundary: no engine file and no scorer names a type or an edge of a pack.

The engine reads rules as data, and the scorers read the pack, so a type or edge name
belongs in pack files only. The search reads a pack's names and looks for each as a
quoted string literal in the engine files and in the eval harness's scripts. A later
ontology version joins the check by adding its pack to `_BOUNDARY_PACKS`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from backend.knowledge.ontologies import registry
from backend.knowledge.ontologies.pack import Pack, load_pack

BASE = load_pack("tests.unit.knowledge.ontologies.fixtures.base_pack")
PET = load_pack("tests.unit.knowledge.ontologies.fixtures.pet_pack")

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _engine_sources() -> dict[str, str]:
    """The engine files, by path."""
    backend = _REPO_ROOT / "backend"
    extraction = backend / "knowledge" / "extraction"
    paths = sorted((backend / "extraction_service").glob("*.py"))
    paths += [
        extraction / "structure_splitter.py",
        extraction / "normalizer.py",
        extraction / "rails.py",
        extraction / "pre_llm.py",
        extraction / "id_rails.py",
        extraction / "prompts.py",
    ]
    return {str(path.relative_to(_REPO_ROOT)): path.read_text(encoding="utf-8") for path in paths}


def _quoted_pack_names(pack: Pack, sources: dict[str, str]) -> list[tuple[str, str]]:
    """Every (file, name) where a type or edge of `pack` stands as a quoted string literal."""
    names = sorted(pack.type_names | set(pack.edges_by_name) | set(pack.retired))
    found = []
    for path, source in sources.items():
        for name in names:
            if re.search(rf"""["']{re.escape(name)}["']""", source):
                found.append((path, name))
    return found


def _scorer_sources() -> dict[str, str]:
    """The eval harness's scripts, by path, less the scorers of past probe sets."""
    paths = sorted(path for path in _SCORERS.glob("*.py") if path.name not in _LEGACY_SCORERS)
    return {
        path.relative_to(_REPO_ROOT).as_posix(): path.read_text(encoding="utf-8") for path in paths
    }


_SCORERS = _REPO_ROOT / "scripts" / "eval_harness"

# Scorers written for one past probe set each. A probe set's predicates are part of
# what such a scorer is, it scores no run of today's sets, and no pack changes it.
_LEGACY_SCORERS = ("score_v8_probe_run.py", "score_v9_predicate_run.py")


@dataclass(frozen=True)
class KnownLiteral:
    """One name a scorer still writes, and why it stands."""

    file: str
    literal: str
    reason: str


# The one exception among the scorers. Delete this entry, and the two lines of the
# scorer test that read it, when the alignment reads the type from the pack.
_KNOWN_LITERAL = KnownLiteral(
    file="scripts/eval_harness/entity_alignment.py",
    literal="Metric",
    reason=(
        "the alignment gives a gold entity of this type its own match rule, in "
        "`variant_kind` and in `_pair_kind`; reading it from the pack is an edit of "
        "those two lines, which T46 leaves as they are"
    ),
)

V1_4_0 = registry.get_pack("1.4.0")
_BOUNDARY_PACKS = {"base": BASE, "pet": PET, "v1.4.0": V1_4_0}


@pytest.mark.parametrize("name", sorted(_BOUNDARY_PACKS))
def test_no_engine_file_names_a_type_or_an_edge_of_the_pack(name):
    assert _quoted_pack_names(_BOUNDARY_PACKS[name], _engine_sources()) == []


@pytest.mark.parametrize("name", sorted(_BOUNDARY_PACKS))
def test_no_scorer_names_a_type_or_an_edge_of_the_pack(name):
    found = _quoted_pack_names(_BOUNDARY_PACKS[name], _scorer_sources())
    known = (_KNOWN_LITERAL.file, _KNOWN_LITERAL.literal)

    assert [hit for hit in found if hit != known] == []


def test_the_known_literal_still_stands_and_says_why():
    found = _quoted_pack_names(V1_4_0, _scorer_sources())

    assert found == [(_KNOWN_LITERAL.file, _KNOWN_LITERAL.literal)]
    assert _KNOWN_LITERAL.reason.strip()


def test_the_legacy_scorers_exist_and_each_names_the_pack():
    sources = {name: (_SCORERS / name).read_text(encoding="utf-8") for name in _LEGACY_SCORERS}

    assert sorted({path for path, _ in _quoted_pack_names(V1_4_0, sources)}) == sorted(
        _LEGACY_SCORERS
    )


def test_the_boundary_search_finds_a_quoted_name():
    planted = {"engine.py": "if entity['type'] == \"Pet\":\n    pass\n"}

    assert _quoted_pack_names(PET, planted) == [("engine.py", "Pet")]
    assert _quoted_pack_names(BASE, planted) == []


def test_the_boundary_covers_the_engine_files_the_plan_lists():
    covered = set(_engine_sources())

    assert {
        "backend/extraction_service/engine.py",
        "backend/extraction_service/turn_plan.py",
        "backend/knowledge/extraction/structure_splitter.py",
        "backend/knowledge/extraction/normalizer.py",
        "backend/knowledge/extraction/rails.py",
        "backend/knowledge/extraction/pre_llm.py",
        "backend/knowledge/extraction/id_rails.py",
        "backend/knowledge/extraction/prompts.py",
    } <= {path.replace("\\", "/") for path in covered}
