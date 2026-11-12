"""schema_conformance measures the production extraction prompt and output schema.

The three schema_conformance case files used to embed their own copy of an
extraction prompt (12 entity types including the retired `Topic`), and the
constrained arm forced a hand-written GBNF grammar of the same vintage. A score
from them said nothing about the prompt the extraction service sends. These
tests pin the replacement:

1. No case file carries prompt text; each names its source with a closed key.
2. Every case's messages equal what the service's Stage 2 renders
   (`render_extraction_messages`) for the same utterance with empty history
   and subject scope unknown.
3. The constrained arm sends the production output schema in the
   `json_schema` response format the provider builds for the engine's
   `schema` mode; a candidate with `gbnf_supported: false` is sent no
   constraint; the json_object arm keeps `{"type": "json_object"}`.
4. Importing the harness loads no backend module (the import is lazy).
5. The few-shot variant, the grammar file and the `--grammar` option are gone.
6. Case expectations name only extractable, non-retired ontology types.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402
import yaml  # noqa: E402

from backend.knowledge.extraction.ontology_extractor import (  # noqa: E402
    render_extraction_messages,
)
from backend.knowledge.extraction.preprocessor import PreProcessor  # noqa: E402
from backend.knowledge.ontologies.hierarchy import RETIRED_TYPE_MAP  # noqa: E402
from backend.knowledge.ontologies.registry import active_pack  # noqa: E402
from backend.knowledge.ontologies.v1_0_0 import (  # noqa: E402
    EXTRACTABLE_NODE_TYPES,
    EXTRACTABLE_RELATIONSHIP_TYPES,
)
from backend.llm.llama_server_provider import LlamaServerProvider  # noqa: E402
from backend.llm.models import LLMRequest  # noqa: E402
from scripts.eval_harness import run  # noqa: E402
from scripts.eval_harness.client import ChatMetrics, ChatResponse  # noqa: E402

SCHEMA_CONFORMANCE_FILES = (
    "schema_conformance",
    "schema_conformance_json_object",
    "schema_conformance_lenient",
)


def _raw_yaml(name: str) -> dict:
    path = run.DEFAULT_TESTS_DIR / f"{name}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _service_stage2_messages(utterance: str, reference_date: datetime) -> list[dict]:
    """What the extraction service sends at Stage 2 with no history and scope off.

    Mirrors `ExtractionEngine.run`: the PreProcessor packages the utterance,
    Stage 1.5 disabled writes `subject_scope = "unknown"`, then Stage 2 renders.
    """
    pre_processed = PreProcessor().pre_process(
        utterance=utterance,
        conversation_history=[],
        reference_date=reference_date,
        turn_index=0,
    )
    pre_processed.metadata["subject_scope"] = "unknown"
    return render_extraction_messages(pre_processed)


def _engine_schema_mode_response_format() -> dict:
    """The response_format the provider sends when the engine runs in `schema` mode."""
    provider = LlamaServerProvider(base_url="http://127.0.0.1:9", model="unused")
    request = LLMRequest(messages=[], response_schema=active_pack().extraction_schema)
    return provider._build_kwargs(request, stream=False)["response_format"]


# ---------------------------------------------------------------------------
# 1. No prompt text in the case files; a closed source key instead
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", SCHEMA_CONFORMANCE_FILES)
def test_case_file_names_the_production_source_and_carries_no_prompt_text(name):
    data = _raw_yaml(name)

    assert "system_prompt" not in data
    for case in data["cases"]:
        assert "system_prompt" not in case, case["id"]
        assert "context" not in case, case["id"]
    assert data["prompt_source"] == "production_extraction"
    assert "use_grammar" not in data


def test_an_unknown_prompt_source_is_refused():
    with pytest.raises(ValueError, match="prompt_source"):
        run._parse_test_file(
            "bad",
            {
                "prompt_source": "some_other_prompt",
                "reference_date": "2026-06-14",
                "cases": [{"id": "c", "prompt": "hello"}],
            },
        )


def test_a_prompt_source_beside_prompt_text_is_refused():
    with pytest.raises(ValueError, match="system_prompt"):
        run._parse_test_file(
            "bad",
            {
                "prompt_source": "production_extraction",
                "reference_date": "2026-06-14",
                "system_prompt": "stale copy",
                "cases": [{"id": "c", "prompt": "hello"}],
            },
        )


def test_constrained_without_a_prompt_source_is_refused():
    with pytest.raises(ValueError, match="constrained"):
        run._parse_test_file(
            "bad",
            {"constrained": True, "cases": [{"id": "c", "prompt": "hello"}]},
        )


# ---------------------------------------------------------------------------
# 2. Messages equal the service's Stage 2 rendering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", SCHEMA_CONFORMANCE_FILES)
def test_every_case_renders_exactly_the_service_stage2_messages(name):
    (test_file,) = run.load_test_files([name], run.DEFAULT_TESTS_DIR)
    reference_date = datetime.fromisoformat(_raw_yaml(name)["reference_date"])

    assert test_file.cases, f"{name} has no cases"
    for case in test_file.cases:
        assert run.build_messages(test_file, case) == _service_stage2_messages(
            case.prompt, reference_date
        ), case.id


# ---------------------------------------------------------------------------
# 3. What each arm sends
# ---------------------------------------------------------------------------


class _RecordingClient:
    """Stands in for HarnessClient; records each chat() call's keyword arguments."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def chat(self, **kwargs) -> ChatResponse:
        self.calls.append(kwargs)
        return ChatResponse(
            content='{"entities": [], "relationships": []}',
            tool_calls=(),
            finish_reason="stop",
            metrics=ChatMetrics(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
                total_time_ms=1.0,
                tokens_per_second=1.0,
            ),
        )


def _candidate(*, gbnf_supported: bool) -> run.Candidate:
    _, candidates = run.load_models_config(run.DEFAULT_CONFIG_PATH)
    base = next(c for c in candidates if c.gbnf_supported)
    return run.Candidate(**{**_as_dict(base), "gbnf_supported": gbnf_supported})


def _as_dict(candidate: run.Candidate) -> dict:
    return {f: getattr(candidate, f) for f in run.Candidate.__slots__}


def _sent(name: str, *, gbnf_supported: bool) -> dict:
    (test_file,) = run.load_test_files([name], run.DEFAULT_TESTS_DIR)
    client = _RecordingClient()
    result = run.run_case(
        client,
        _candidate(gbnf_supported=gbnf_supported),
        test_file,
        test_file.cases[0],
        iteration=1,
        seed=None,
    )
    assert result.error is None
    (call,) = client.calls
    assert call["messages"] == run.build_messages(test_file, test_file.cases[0])
    assert result.system_prompt == call["messages"][0]["content"]
    return call


@pytest.mark.parametrize("name", ["schema_conformance", "schema_conformance_lenient"])
def test_constrained_arm_sends_the_production_output_schema(name):
    call = _sent(name, gbnf_supported=True)

    assert call["response_format"] == _engine_schema_mode_response_format()
    assert call["response_format"]["json_schema"]["schema"] == active_pack().extraction_schema
    assert call.get("grammar") is None


@pytest.mark.parametrize("name", ["schema_conformance", "schema_conformance_lenient"])
def test_constrained_arm_sends_no_constraint_to_a_gbnf_unsupported_candidate(name):
    call = _sent(name, gbnf_supported=False)

    assert call["response_format"] is None
    assert call.get("grammar") is None


@pytest.mark.parametrize("gbnf_supported", [True, False])
def test_json_object_arm_sends_json_object_only(gbnf_supported):
    call = _sent("schema_conformance_json_object", gbnf_supported=gbnf_supported)

    assert call["response_format"] == {"type": "json_object"}
    assert call.get("grammar") is None


# ---------------------------------------------------------------------------
# 4. The harness imports no backend module at load
# ---------------------------------------------------------------------------


def test_importing_the_harness_loads_no_backend_module():
    code = (
        "import sys\n"
        "import scripts.eval_harness.run\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] == 'backend')\n"
        "print(loaded)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        shell=False,
        check=True,
    )
    assert proc.stdout.strip() == "[]"


# ---------------------------------------------------------------------------
# 5. The few-shot variant, grammar and grammar option are gone
# ---------------------------------------------------------------------------


def test_fewshot_variant_and_grammar_are_gone():
    assert not (run.DEFAULT_TESTS_DIR / "schema_conformance_fewshot.yaml").exists()
    assert not (run.HARNESS_ROOT / "grammars" / "mist_ontology.gbnf").exists()
    assert "schema_conformance_fewshot" not in run.DEFAULT_TEST_ORDER
    assert not hasattr(run, "load_grammar")
    with pytest.raises(SystemExit):
        run.parse_args(["--grammar", "mist_ontology"])


# ---------------------------------------------------------------------------
# 6. Expectations name live ontology types only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", SCHEMA_CONFORMANCE_FILES)
def test_case_expectations_name_only_extractable_non_retired_types(name):
    (test_file,) = run.load_test_files([name], run.DEFAULT_TESTS_DIR)
    for case in test_file.cases:
        entity_types = set(case.expected.get("entity_types", []))
        relationship_types = set(case.expected.get("relationship_types", []))
        assert not entity_types & set(RETIRED_TYPE_MAP), case.id
        assert entity_types <= set(EXTRACTABLE_NODE_TYPES), case.id
        assert relationship_types <= set(EXTRACTABLE_RELATIONSHIP_TYPES), case.id
