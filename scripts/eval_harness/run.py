"""Eval harness orchestrator -- CLI entry point.

Usage:
    python -m scripts.eval_harness.run \
        --models gemma-4-26b-a4b-iq4xs,qwen-3.5-9b-q8 \
        --tests schema_conformance,tool_selection,personality,rag_integration,coherence,speed \
        --iterations 1

    python -m scripts.eval_harness.run --external  # assume user pre-started llama-server
    python -m scripts.eval_harness.run --list-models  # print all candidate IDs
    python -m scripts.eval_harness.run --list-tests   # print all registered tests

High-level flow:
    1. Parse models.yaml into Candidate objects.
    2. Parse selected test YAMLs into TestCase objects.
    3. For each candidate: optionally spawn llama-server, wait for health,
       run every selected test case through the client, write per-candidate
       JSONL to results/.
    4. Invoke scorers.score_run() to produce RunScores, then
       report.generate_markdown() to write results/report-YYYY-MM-DD.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from scripts.eval_harness.client import (
    CandidateConfig,
    HarnessClient,
    HarnessError,
    HarnessRequestError,
    HarnessServerError,
    HarnessTimeoutError,
    ServerLauncher,
    ServerSpec,
)
from scripts.eval_harness.report import generate_markdown_report
from scripts.eval_harness.scorers import score_run

logger = logging.getLogger("eval_harness")

HARNESS_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = HARNESS_ROOT / "models.yaml"
DEFAULT_TESTS_DIR = HARNESS_ROOT / "tests"
DEFAULT_GRAMMARS_DIR = HARNESS_ROOT / "grammars"
DEFAULT_RESULTS_DIR = HARNESS_ROOT / "results"
DEFAULT_LOG_DIR = DEFAULT_RESULTS_DIR / "server-logs"

DEFAULT_TEST_ORDER: tuple[str, ...] = (
    "speed_minimal",
    "schema_conformance",
    "schema_conformance_lenient",
    "schema_conformance_fewshot",
    "tool_selection",
    "personality",
    "adversarial_persona",
    "rag_integration",
    "coherence",
    "cot_reasoning",
    "long_turn_coherence",
    "speed",
)


# ---------------------------------------------------------------------------
# Config types parsed from models.yaml and tests/*.yaml
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HarnessDefaults:
    """Global defaults applied to every candidate."""

    base_url: str
    models_dir: str
    llama_server_binary: str
    shared_server_args: tuple[str, ...]
    ctx_size: int
    spawn_health_timeout_seconds: float
    request_timeout_seconds: float
    api_key: str


@dataclass(frozen=True, slots=True)
class Candidate:
    """A single candidate model entry from models.yaml."""

    id: str
    display_name: str
    tier: str
    family: str
    vendor: str
    gguf: str
    quant: str
    size_gb: float
    total_params_b: float
    active_params_b: float
    architecture: str
    context_size: int
    served_model_name: str
    temperature: dict[str, float]
    top_p: dict[str, float]
    chat_template: str | None
    chat_template_file: str | None
    tool_parser: str | None
    gbnf_supported: bool
    stop_sequences: tuple[str, ...]
    extra_server_args: tuple[str, ...]
    shared_server_args_override: tuple[str, ...] | None
    notes: str


@dataclass(frozen=True, slots=True)
class TestCase:
    """One test case within a test file (one prompt/expectation pair)."""

    id: str
    prompt: str
    system_prompt: str | None
    expected: dict[str, Any]
    tools: tuple[dict[str, Any], ...]
    context: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TestFile:
    """A test YAML file. One test type per file."""

    name: str
    test_type: str
    description: str
    temperature_mode: str
    max_tokens: int
    use_grammar: bool
    response_format: dict[str, Any] | None
    system_prompt: str | None
    cases: tuple[TestCase, ...]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_models_config(path: Path) -> tuple[HarnessDefaults, list[Candidate]]:
    """Parse models.yaml into defaults + candidate list. Fails fast on bad schema."""
    if not path.exists():
        raise FileNotFoundError(f"models config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"models config is not a mapping: {path}")

    raw_defaults = data.get("defaults", {})
    defaults = HarnessDefaults(
        base_url=raw_defaults.get("base_url", "http://127.0.0.1:8080"),
        models_dir=os.environ.get(
            "LLAMA_MODELS_DIR",
            raw_defaults.get("models_dir", "."),
        ),
        llama_server_binary=os.environ.get(
            "LLAMA_SERVER_BIN",
            raw_defaults.get("llama_server_binary", "llama-server"),
        ),
        shared_server_args=tuple(raw_defaults.get("shared_server_args", [])),
        ctx_size=int(raw_defaults.get("ctx_size", 8192)),
        spawn_health_timeout_seconds=float(raw_defaults.get("spawn_health_timeout_seconds", 120)),
        request_timeout_seconds=float(raw_defaults.get("request_timeout_seconds", 180)),
        api_key=raw_defaults.get("api_key", "not-needed"),
    )

    raw_candidates = data.get("candidates") or []
    if not isinstance(raw_candidates, list):
        raise ValueError("'candidates' must be a list in models.yaml")

    candidates = [_parse_candidate(entry) for entry in raw_candidates]
    return defaults, candidates


def _parse_candidate(entry: dict[str, Any]) -> Candidate:
    required = ("id", "display_name", "tier", "gguf", "served_model_name")
    missing = [k for k in required if k not in entry]
    if missing:
        raise ValueError(f"candidate missing required fields {missing}: {entry}")
    return Candidate(
        id=entry["id"],
        display_name=entry["display_name"],
        tier=entry["tier"],
        family=entry.get("family", ""),
        vendor=entry.get("vendor", ""),
        gguf=entry["gguf"],
        quant=entry.get("quant", ""),
        size_gb=float(entry.get("size_gb", 0.0)),
        total_params_b=float(entry.get("total_params_b", 0.0)),
        active_params_b=float(entry.get("active_params_b", 0.0)),
        architecture=entry.get("architecture", "dense"),
        context_size=int(entry.get("context_size", 8192)),
        served_model_name=entry["served_model_name"],
        temperature=dict(entry.get("temperature", {"extraction": 0.0, "conversation": 0.7})),
        top_p=dict(entry.get("top_p", {"extraction": 0.95, "conversation": 0.95})),
        chat_template=entry.get("chat_template"),
        chat_template_file=entry.get("chat_template_file"),
        tool_parser=entry.get("tool_parser"),
        gbnf_supported=bool(entry.get("gbnf_supported", True)),
        stop_sequences=tuple(entry.get("stop_sequences", [])),
        extra_server_args=tuple(entry.get("extra_server_args", [])),
        shared_server_args_override=(
            tuple(entry["shared_server_args_override"])
            if "shared_server_args_override" in entry
            and entry["shared_server_args_override"] is not None
            else None
        ),
        notes=entry.get("notes", ""),
    )


def load_test_files(names: list[str], tests_dir: Path) -> list[TestFile]:
    """Parse each named test YAML into a TestFile."""
    out: list[TestFile] = []
    for name in names:
        path = tests_dir / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"test file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        out.append(_parse_test_file(name, data))
    return out


def _parse_test_file(name: str, data: dict[str, Any]) -> TestFile:
    cases_raw = data.get("cases") or []
    cases = tuple(
        TestCase(
            id=c["id"],
            prompt=c["prompt"],
            system_prompt=c.get("system_prompt"),
            expected=c.get("expected", {}),
            tools=tuple(c.get("tools", [])),
            context=c.get("context"),
            metadata=dict(c.get("metadata", {})),
        )
        for c in cases_raw
    )
    return TestFile(
        name=name,
        test_type=data.get("test_type", name),
        description=data.get("description", ""),
        temperature_mode=data.get("temperature_mode", "extraction"),
        max_tokens=int(data.get("max_tokens", 1024)),
        use_grammar=bool(data.get("use_grammar", False)),
        response_format=data.get("response_format"),
        system_prompt=data.get("system_prompt"),
        cases=cases,
    )


def load_grammar(grammar_name: str) -> str:
    """Read a GBNF grammar file from grammars/."""
    path = DEFAULT_GRAMMARS_DIR / f"{grammar_name}.gbnf"
    if not path.exists():
        raise FileNotFoundError(f"grammar file not found: {path}")
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Result records (persisted as JSONL)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CaseResult:
    """One test case executed against one candidate."""

    candidate_id: str
    test_name: str
    test_type: str
    case_id: str
    iteration: int
    prompt: str
    system_prompt: str | None
    expected: dict[str, Any]
    response_content: str
    response_tool_calls: list[dict[str, Any]]
    finish_reason: str
    metrics: dict[str, Any]
    error: str | None = None

    def to_jsonl(self) -> str:
        """Serialize this run record as a single JSONL line."""
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def build_messages(test_file: TestFile, case: TestCase) -> list[dict[str, Any]]:
    """Assemble OpenAI-format messages for one test case.

    System prompt priority: case.system_prompt > test_file.system_prompt.
    Context (if present) is folded into the single system message to avoid
    two consecutive system messages, which some bundled jinja templates
    reject: Qwen 3.5 requires system to be first-only, Gemma 3 requires
    strict user/assistant alternation.
    """
    messages: list[dict[str, Any]] = []
    system_prompt = case.system_prompt or test_file.system_prompt
    if system_prompt and case.context:
        combined = f"{system_prompt}\n\nContext:\n{case.context}"
        messages.append({"role": "system", "content": combined})
    elif system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    elif case.context:
        messages.append({"role": "system", "content": f"Context:\n{case.context}"})
    messages.append({"role": "user", "content": case.prompt})
    return messages


def run_case(
    client: HarnessClient,
    candidate: Candidate,
    test_file: TestFile,
    case: TestCase,
    *,
    iteration: int,
    grammar_text: str | None,
    seed: int | None,
) -> CaseResult:
    """Run one test case and return a CaseResult (success or error)."""
    messages = build_messages(test_file, case)
    temperature = candidate.temperature.get(test_file.temperature_mode, 0.0)
    top_p = candidate.top_p.get(test_file.temperature_mode, 0.95)

    base: dict[str, Any] = {
        "candidate_id": candidate.id,
        "test_name": test_file.name,
        "test_type": test_file.test_type,
        "case_id": case.id,
        "iteration": iteration,
        "prompt": case.prompt,
        "system_prompt": case.system_prompt or test_file.system_prompt,
        "expected": case.expected,
    }

    try:
        response = client.chat(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=test_file.max_tokens,
            stop=list(candidate.stop_sequences) if candidate.stop_sequences else None,
            tools=list(case.tools) if case.tools else None,
            tool_choice="auto" if case.tools else None,
            response_format=test_file.response_format,
            grammar=(grammar_text if test_file.use_grammar and candidate.gbnf_supported else None),
            seed=seed,
        )
    except HarnessRequestError as exc:
        logger.error("case %s failed on %s: %s", case.id, candidate.id, exc)
        return CaseResult(
            **base,
            response_content="",
            response_tool_calls=[],
            finish_reason="",
            metrics={},
            error=str(exc),
        )

    tool_calls_dict = [
        {"id": tc.id, "name": tc.name, "arguments_json": tc.arguments_json}
        for tc in response.tool_calls
    ]
    return CaseResult(
        **base,
        response_content=response.content,
        response_tool_calls=tool_calls_dict,
        finish_reason=response.finish_reason,
        metrics=dataclasses.asdict(response.metrics),
    )


def run_candidate(
    candidate: Candidate,
    defaults: HarnessDefaults,
    test_files: list[TestFile],
    *,
    iterations: int,
    seed: int | None,
    external: bool,
    llama_server_bin: str,
    models_dir: Path,
    log_dir: Path,
    results_dir: Path,
    grammar_text: str | None,
) -> Path:
    """Run all tests against one candidate. Returns the JSONL result path."""
    logger.info("=" * 72)
    logger.info("CANDIDATE: %s (%s)", candidate.display_name, candidate.id)
    logger.info("=" * 72)

    jsonl_path = results_dir / f"{candidate.id}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    client_config = CandidateConfig(
        candidate_id=candidate.id,
        base_url=defaults.base_url,
        served_model_name=candidate.served_model_name,
        api_key=defaults.api_key,
        request_timeout_seconds=defaults.request_timeout_seconds,
    )
    client = HarnessClient(client_config)

    launcher: ServerLauncher | None = None
    if not external:
        gguf_path = str(models_dir / candidate.gguf)
        shared_args = (
            candidate.shared_server_args_override
            if candidate.shared_server_args_override is not None
            else defaults.shared_server_args
        )
        spec = ServerSpec(
            candidate_id=candidate.id,
            binary_path=llama_server_bin,
            gguf_path=gguf_path,
            chat_template=candidate.chat_template,
            chat_template_file=candidate.chat_template_file,
            ctx_size=candidate.context_size,
            shared_args=shared_args,
            extra_args=candidate.extra_server_args,
        )
        launcher = ServerLauncher(spec, log_dir=log_dir)

    try:
        if launcher is not None:
            launcher.start()
        client.wait_for_ready(defaults.spawn_health_timeout_seconds)

        with jsonl_path.open("w", encoding="utf-8") as out:
            for test_file in test_files:
                logger.info(
                    "-> test: %s (%d cases x %d iterations)",
                    test_file.name,
                    len(test_file.cases),
                    iterations,
                )
                for iteration in range(1, iterations + 1):
                    for case in test_file.cases:
                        result = run_case(
                            client,
                            candidate,
                            test_file,
                            case,
                            iteration=iteration,
                            grammar_text=grammar_text,
                            seed=seed,
                        )
                        out.write(result.to_jsonl() + "\n")
                        out.flush()
                        os.fsync(out.fileno())
    except HarnessTimeoutError as exc:
        logger.error("candidate %s timed out during startup: %s", candidate.id, exc)
    except HarnessServerError as exc:
        logger.error("candidate %s server error: %s", candidate.id, exc)
    except HarnessError as exc:
        logger.error("candidate %s harness error: %s", candidate.id, exc)
    finally:
        if launcher is not None:
            launcher.stop()

    return jsonl_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MIST model backend A/B evaluation harness",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"models.yaml path (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--tests-dir",
        type=Path,
        default=DEFAULT_TESTS_DIR,
        help=f"tests directory (default: {DEFAULT_TESTS_DIR})",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"results output directory (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=f"llama-server stdout/stderr log directory (default: {DEFAULT_LOG_DIR})",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="comma-separated candidate IDs to run (default: all primary)",
    )
    parser.add_argument(
        "--tests",
        type=str,
        default=",".join(DEFAULT_TEST_ORDER),
        help="comma-separated test names to run",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="iterations per test case (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="sampling seed passed to llama-server (default: 42)",
    )
    parser.add_argument(
        "--external",
        action="store_true",
        help="do NOT spawn llama-server; assume one is already running at base_url",
    )
    parser.add_argument(
        "--llama-server-bin",
        type=str,
        default=None,
        help="path to llama-server binary (overrides LLAMA_SERVER_BIN and models.yaml)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="directory containing GGUF files (overrides LLAMA_MODELS_DIR and models.yaml)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="override the auto-generated UTC timestamp subdir with this name (orchestrator use: multi-invocation runs share one dir)",
    )
    parser.add_argument(
        "--grammar",
        type=str,
        default="mist_ontology",
        help="grammar file basename under grammars/ (default: mist_ontology)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="print candidate IDs and exit",
    )
    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="print test names and exit",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="skip markdown report generation",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="logging level (DEBUG|INFO|WARNING|ERROR)",
    )
    return parser.parse_args(argv)


def resolve_candidate_selection(
    args_models: str, all_candidates: list[Candidate]
) -> list[Candidate]:
    """Select candidates from the CLI string; default to primary tier."""
    if not args_models.strip():
        return [c for c in all_candidates if c.tier == "primary"]
    requested = [x.strip() for x in args_models.split(",") if x.strip()]
    by_id = {c.id: c for c in all_candidates}
    missing = [r for r in requested if r not in by_id]
    if missing:
        raise SystemExit(f"unknown candidate IDs: {missing}")
    return [by_id[r] for r in requested]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    defaults, all_candidates = load_models_config(args.config)

    if args.list_models:
        for c in all_candidates:
            print(f"{c.id:40s} {c.tier:10s} {c.display_name}")
        return 0

    if args.list_tests:
        for name in DEFAULT_TEST_ORDER:
            path = args.tests_dir / f"{name}.yaml"
            marker = "[OK]" if path.exists() else "[MISSING]"
            print(f"{name:25s} {marker} {path}")
        return 0

    candidates = resolve_candidate_selection(args.models, all_candidates)
    test_names = [t.strip() for t in args.tests.split(",") if t.strip()]
    test_files = load_test_files(test_names, args.tests_dir)

    try:
        grammar_text = load_grammar(args.grammar)
    except FileNotFoundError as exc:
        logger.warning("grammar load failed, continuing without GBNF: %s", exc)
        grammar_text = None

    models_dir_str = str(args.models_dir) if args.models_dir is not None else defaults.models_dir
    models_dir = Path(models_dir_str)
    llama_server_bin = args.llama_server_bin or defaults.llama_server_binary

    if args.run_name:
        run_timestamp = args.run_name
    else:
        run_timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_results_dir = args.results_dir / run_timestamp
    run_results_dir.mkdir(parents=True, exist_ok=True)
    run_log_dir = args.log_dir / run_timestamp

    logger.info("run timestamp: %s", run_timestamp)
    logger.info("results dir: %s", run_results_dir)
    logger.info("candidates: %s", [c.id for c in candidates])
    logger.info("tests: %s", [t.name for t in test_files])
    logger.info("iterations per case: %d", args.iterations)

    jsonl_paths: list[Path] = []
    for candidate in candidates:
        started = time.perf_counter()
        path = run_candidate(
            candidate,
            defaults,
            test_files,
            iterations=args.iterations,
            seed=args.seed,
            external=args.external,
            llama_server_bin=llama_server_bin,
            models_dir=models_dir,
            log_dir=run_log_dir,
            results_dir=run_results_dir,
            grammar_text=grammar_text,
        )
        jsonl_paths.append(path)
        elapsed = time.perf_counter() - started
        logger.info("candidate %s finished in %.1fs", candidate.id, elapsed)

    if args.skip_report:
        return 0

    run_scores = score_run(jsonl_paths, test_files)
    report_path = run_results_dir / "report.md"
    generate_markdown_report(
        report_path,
        run_scores,
        candidates=candidates,
        test_files=test_files,
        run_timestamp=run_timestamp,
    )
    logger.info("wrote report: %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
