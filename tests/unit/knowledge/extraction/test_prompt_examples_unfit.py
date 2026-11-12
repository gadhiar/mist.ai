"""New prompt examples must stay off the gate sets.

A few-shot example that copies a dev-lists or 60-row utterance, or that teaches an id a
gold row expects, lets the prompt score a gate row by recall of its own text rather than
by the convention it teaches. Every example added after r6 (the 2026-06-14 prompt text the
r8 conventions are built on) is checked here: its utterance shares no five consecutive
lowercase words with any gate utterance, none of its capitalised words (a name, most
often) occurs as a whole word in any gate utterance, and none of its ids equals a gate id
or a gate alternative. r6's own 26 examples, the `user` and `mist-identity` anchors and ISO Date ids
are exempt, since they predate the gate sets or are fixed by convention rather than by
the example.

The assertion messages carry counts and example numbers only, so a failure prints no gate
utterance and no gate id.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from backend.knowledge.ontologies.registry import active_pack

_REPO_ROOT = Path(__file__).resolve().parents[4]
_GATE_FILES = (
    _REPO_ROOT / "data" / "gold" / "extraction" / "dev-lists-2026-11.jsonl",
    _REPO_ROOT / "data" / "ingest" / "extraction-gold-2026-06-14.jsonl",
)
_RUN_LENGTH = 5

# The utterances of r6's 26 examples, as the prompt held them.
R6_UTTERANCES: frozenset[str] = frozenset(
    {
        "I've been using Python for about 5 years",
        "I used to work with Java but stopped",
        "MIST uses LanceDB for vector search and Neo4j for the knowledge graph",
        "MIST is implemented with Python and llama.cpp",
        "I'm learning React at work. My team lead Sarah uses it for everything",
        "MIST is warm and playful by default",
        "My coworker says Rust is really fast",
        "Hey, how's it going?",
        "We shipped Cluster 8 Phase 6 on 2026-04-22",
        "Gemma 4 E4B gets 0.94 tool_selection on the eval harness",
        "I read ADR-010 yesterday and I like the vault-as-canon pattern",
        "I attended a conference on 2026-04-15",
        "I'm working through the Pragmatic Programmer",
        "The deploy happened after the code review",
        "The garbage collector reclaims unused memory fragments",
        "The retrieval pipeline consists of a query encoder and a reranker; "
        "user context feeds into the encoder",
        "Speculative decoding speeds up token generation, and the LRU cache cuts down on "
        "lookup latency",
        "Hybrid retrieval is our strategy for memory recall and applies to long-context "
        "conversations. The MemoryFragment data structure follows camelCase naming",
        "The LRU pattern is applicable to cache eviction and the cache mechanism comprises "
        "a doubly-linked list",
        "Speculative decoding is the mechanism behind faster token generation, and the "
        "retry-policy mechanism handles transient network failures",
        "I quit my job at Acme in March 2026, and I've given up on learning Haskell",
        "Scratch that -- I never actually used MongoDB",
        "My coworker Sarah recommended Postgres",
        "I work out every morning before standup",
        "My sister Anjali works at Google in biotech",
        "I find the notion of semantic closure really interesting",
    }
)

# The entity ids of r6's 26 examples.
R6_IDS: frozenset[str] = frozenset(
    {
        "python",
        "java",
        "lancedb",
        "neo4j",
        "llama-cpp",
        "react",
        "sarah",
        "warmth",
        "playfulness",
        "rust",
        "cluster-8-phase-6",
        "gemma-4-e4b",
        "tool-selection-0-94",
        "adr-010",
        "conference-2026-04-15",
        "pragmatic-programmer",
        "deploy",
        "code-review",
        "garbage-collector",
        "memory-fragment",
        "retrieval-pipeline",
        "query-encoder",
        "reranker",
        "user-context",
        "speculative-decoding",
        "token-generation",
        "lru-cache",
        "lookup-latency",
        "hybrid-retrieval",
        "memory-recall",
        "long-context-conversations",
        "memoryfragment",
        "camelcase",
        "lru",
        "cache-eviction",
        "cache",
        "doubly-linked-list",
        "retry-policy",
        "transient-failure",
        "acme",
        "haskell",
        "mongodb",
        "postgres",
        "working-out",
        "anjali",
        "google",
        "semantic-closure",
    }
)
_ANCHOR_IDS = frozenset({"user", "mist-identity"})
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# An example may show `Context:` lines before its subject scope (Example 30, since T52),
# as the user template places them. The test reads its utterance and ids, as before.
_EXAMPLE_RE = re.compile(
    r"^### Example (\d+):[^\n]*\n(?:Context:\n(?:[^\n]*\n)*?)?"
    r"Subject scope: [^\n]*\nUtterance: \"(.*)\"\nOutput:\n(.*)$",
    re.MULTILINE,
)


def _examples() -> dict[int, tuple[str, list[str]]]:
    """Example number to (utterance, entity ids), parsed from the production prompt."""
    examples: dict[int, tuple[str, list[str]]] = {}
    for match in _EXAMPLE_RE.finditer(active_pack().system_prompt):
        payload = json.loads(match.group(3).replace("{{", "{").replace("}}", "}"))
        ids = [str(entity["id"]) for entity in payload["entities"]]
        examples[int(match.group(1))] = (match.group(2), ids)
    return examples


_EXAMPLES = _examples()


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _runs(text: str) -> set[tuple[str, ...]]:
    words = _words(text)
    return {tuple(words[i : i + _RUN_LENGTH]) for i in range(len(words) - _RUN_LENGTH + 1)}


def _gate_rows() -> list[dict]:
    rows: list[dict] = []
    for path in _GATE_FILES:
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


@pytest.fixture(scope="module")
def gate_runs() -> list[set[tuple[str, ...]]]:
    return [_runs(str(row["utterance"])) for row in _gate_rows()]


@pytest.fixture(scope="module")
def gate_words() -> list[set[str]]:
    return [set(_words(str(row["utterance"]))) for row in _gate_rows()]


@pytest.fixture(scope="module")
def gate_ids() -> frozenset[str]:
    ids: set[str] = set()
    for row in _gate_rows():
        for entity in row.get("expected_entities", []):
            ids.add(str(entity["id"]))
            for alternative in entity.get("alternatives", None) or []:
                ids.add(str(alternative["id"] if isinstance(alternative, dict) else alternative))
    return frozenset(ids)


def test_every_example_parses_and_the_numbers_run_from_one() -> None:
    assert sorted(_EXAMPLES) == list(range(1, len(_EXAMPLES) + 1))
    assert len(_EXAMPLES) >= len(R6_UTTERANCES)


def test_the_gate_files_hold_rows(gate_runs: list, gate_ids: frozenset) -> None:
    assert len(gate_runs) > 0 and len(gate_ids) > 0, "a gate file read as empty"


@pytest.mark.parametrize("number", sorted(_EXAMPLES))
def test_a_new_example_utterance_shares_no_five_word_run_with_a_gate_utterance(
    number: int, gate_runs: list[set[tuple[str, ...]]]
) -> None:
    utterance, _ = _EXAMPLES[number]
    if utterance in R6_UTTERANCES:
        return
    runs = _runs(utterance)
    hits = sum(1 for gate in gate_runs if runs & gate)
    assert hits == 0, f"example {number} shares a five-word run with {hits} gate utterance(s)"


@pytest.mark.parametrize("number", sorted(_EXAMPLES))
def test_a_new_example_capitalised_word_occurs_in_no_gate_utterance(
    number: int, gate_words: list[set[str]]
) -> None:
    utterance, _ = _EXAMPLES[number]
    if utterance in R6_UTTERANCES:
        return
    words = re.findall(r"[A-Za-z0-9]+", utterance)
    capitalised = {w.lower() for w in words[1:] if w[0].isupper() and w != "I"}
    hits = sum(1 for gate in gate_words if capitalised & gate)
    assert hits == 0, f"example {number} has a capitalised word in {hits} gate utterance(s)"


@pytest.mark.parametrize("number", sorted(_EXAMPLES))
def test_a_new_example_id_equals_no_gate_id(number: int, gate_ids: frozenset[str]) -> None:
    _, ids = _EXAMPLES[number]
    new_ids = [
        entity_id
        for entity_id in ids
        if entity_id not in R6_IDS
        and entity_id not in _ANCHOR_IDS
        and not _ISO_DATE_RE.match(entity_id)
    ]
    hits = sum(1 for entity_id in new_ids if entity_id in gate_ids)
    assert hits == 0, f"example {number} has {hits} id(s) equal to a gate id or alternative"
