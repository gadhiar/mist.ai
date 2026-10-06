# MIST.AI - Claude Code Guide

MIST.AI (or M.I.S.T; not "Mist.AI", "mist" or "MIST") is a cognitive architecture research
platform: a transparent, locally-run AI system with persistent memory. It is not a ChatGPT
replacement and not a simple productivity assistant.

This repository is PUBLIC: treat everything committed or pushed here as world-readable. Raj's
communication rules and delegation model live in the global `~/.claude/CLAUDE.md`; this file holds
what is specific to this repository, plus two rules that delegates working here need because they
never load the global file: the engineering philosophy below and the no-slop line under Code
Conventions.

---

## Engineering Philosophy

**Always design for the ideal solution.** Never let implementation complexity, cost, time, or difficulty influence the architectural recommendation. Lead with the production-ready, fully optimized architecture. We have agentic teams and the resources to build things right -- recommending a "simpler" approach when a better one exists wastes time building something that needs replacement.

---

## Where Things Are

- `CODEBASE.md` -- live state: branch, HEAD, phase, subsystem status, test count, next actions.
  Read it first (`/mist-status` does).
- `REPOSITORY_STRUCTURE.md` -- layout; update it when you add a directory or a major file.
- `CONTRIBUTING.md` -- code style, commit format, pre-commit hooks, the AI-slop checker.
- `TESTING.md` -- test conventions.
- `KNOWN_ISSUES.md` -- the backlog of known-dead and known-broken code.
- ADRs: repo-scoped in `docs/decisions/`; cross-project and integration ADRs (memory
  architecture, vault layer, FE/BE protocol) in the knowledge vault at
  `D:\Users\rajga\knowledge-vault\Decisions\`. Phase specs: `docs/superpowers/specs/`. `docs/`
  is gitignored apart from a few tracked files, so most of it exists only in Raj's checkout, not
  in a worktree.
- `mist-frontend/` -- the Tauri 2 + React 19 + react-three-fiber frontend: a separate git
  repository (remote `gadhiar/mist-frontend`) that this repository gitignores, with its own
  CLAUDE.md. The two meet at the WebSocket protocol (ADR-016, ADR-017), not at code style. The
  Flutter frontend (`mist_desktop/`) was decommissioned 2026-05-11; commit `e18c092` preserves it.
- `dependencies/csm/` -- legacy Sesame CSM TTS (Apache 2.0), kept for rollback only. Preserve its
  Apache 2.0 license headers in any file you modify there.
- Empty directories are runtime or placeholder paths. Do not remove them.
- `.env` is never committed. Secrets come from environment variables, documented in `.env.example`.

---

## Stack and Constraints

- Backend: Python 3.11+, FastAPI + Uvicorn WebSocket server on port 8001, in Docker Compose
  (services `mist-backend`, `mist-neo4j` with Neo4j 5, `mist-llm` with llama-server) on
  `nvidia/cuda:12.4.0-devel-ubuntu22.04` with PyTorch 2.6 + cu124. All backend work runs in the
  container.
- LLM inference: interactive: Gemma 4 E4B Q5_K_M via llama.cpp's llama-server; extraction: a
  separate llama-server host (`docker-compose.extraction.yml`), gpt-oss-20b since 2026-09-29.
  Read live model ids from each host's `/v1/info`.
- Voice: VAD -> Whisper STT -> LLM -> Chatterbox Turbo TTS (MIT license, zero-shot voice cloning;
  adapter `ChatterboxTTS` in `src/multimodal/tts.py`).
- Knowledge: Neo4j knowledge graph; vault layer (ADR-010, partly superseded by ADR-023):
  `mist-memory/` markdown corpus + sqlite-vec sidecar index + watchdog filewatcher. Embeddings:
  all-MiniLM-L6-v2 (384-dim).
- Hardware (queried 2026-03-23): NVIDIA GeForce RTX 4070 SUPER 12 GB VRAM, AMD Ryzen 7 7800X3D
  (16 threads), ~32 GB RAM, Windows 11 host.
- Subsystem status lives in `CODEBASE.md`, not here.

### Design Principles

- **Transparency.** Every decision the AI makes is visible: tool calls shown, entity extractions
  logged, knowledge-graph retrievals visualizable. No hidden behaviour.
- **Local-first.** Core functionality works without internet (llama-server, local Neo4j, offline
  knowledge system) and stays air-gapped capable; every integration must degrade offline
  (ADR-025). Cloud delegation only for strategic decisions.
- **Privacy.** The user controls all data: no telemetry without explicit consent, local storage
  only, export and delete supported.

---

## CODEBASE.md Maintenance Protocol

`CODEBASE.md` is the authoritative in-repo snapshot of current state and the first thing a fresh
session reads, so a stale one silently misleads every later session (on 2026-07-29 it still named
a long-merged feature branch and a HEAD five phases behind). Keep it current as a side effect of
routine work, not as a task deferred for later. Delegates never write it; they report the drift.

- **On any status, scan or context-loading pass** (`/mist-status`, session start, "where are we"):
  if the header (Last Updated, Branch, Status), Current Focus, test count or a subsystem bullet
  diverges from git or the vault workstream note, reconcile it in the same turn before reporting
  status.
- **On landing a milestone, merging to `main`, or changing the active branch or HEAD:** update the
  header block and Current Focus before the work counts as done.
- **On adding, removing or materially changing a subsystem:** update its bullet under Current
  Status.
- **Ground every claim in real state:** `git status` and `git log` in this repository, and the
  vault workstream note. Never copy a hash, count or version forward without verifying it; if a
  number cannot be verified, flag it rather than guessing.
- **Preserve history:** demote the prior header entry to a nested `PRIOR ENTRY --` rather than
  deleting it.
- Schema, convention and structure changes belong in this CLAUDE.md, not only in CODEBASE.md.
- Which documentation files are tracked and pushed: the global CLAUDE.md, Git Workflow.

---

## Plan Verification Protocol

Plans and specs assert things about the codebase ("X has no caller", "this would create a
cycle"), and people and agents execute those assertions without re-deriving them. On 2026-08-04,
R1.4.6 T0 shipped five falsified rationales on one branch, ten across two branches. In every case
the conclusion was right and the stated reason was invented; each fell to a few seconds of
`grep`, and two were already recorded in `KNOWN_ISSUES.md`. Every falsified claim was about code
its author had not opened; the claims about code that had been read survived two reviews.

- **Every causal claim about the codebase carries the command that establishes it.** Not "no
  cycle exists" but ``no cycle: `grep '^from\|^import' backend/chat/stream_events.py` ->
  dataclasses, typing only``. If you cannot produce the command, you have not checked.
- **Before mandating work on any attribute or method, grep `KNOWN_ISSUES.md` for it.** It is
  cheap to search and routinely not searched.
- **Run a claim-check pass before dispatching a plan:** one agent extracts every factual
  assertion about the codebase from the plan and verifies each against source, before any
  implementer builds on a false premise.
- **State unverified claims as unverified** ("do not reset this -- reason not verified") rather
  than supplying a plausible mechanism. A missing reason prompts a check; a wrong reason is a
  trap.
- **Every sentence is separately falsifiable; a sentence does not inherit its neighbour's
  evidence.** The error enters at compression: summarising several branches, files or facts into
  one clause gives the clause only one member's properties. Check each clause of a multi-part
  sentence separately, verify a sentence that describes several code paths against every path,
  and treat a docstring or comment block as a list of independent claims. Instances from
  2026-08-04: a docstring merged "`audio_queue` has no writer" with "`process_audio_chunk` ... has
  no caller" (two true facts, one false sentence); a comment said `aggregate_quality_score` "sums"
  when it returns `statistics.mean`; a message called `dump_run_scores_json` "the durable
  artifact" when it had zero callers, and an implementer copied that into a committed docstring.
- **Never waive the whole-branch review gate.** Scoped per-task review cannot catch this class,
  because the deadness lives outside the diff; only the whole-branch gate can.
- **Do not answer this failure mode with more downstream review layers.** That makes the late
  catch more expensive without moving it earlier.

---

## Code Conventions

Formatting (black and ruff, line length 100), PEP 585/604 type hints, Google-style docstrings and
the AI-slop checker are in `CONTRIBUTING.md`. In addition:

- Imports: relative within a package, absolute across packages. From `typing`, import only
  `TypeVar`, `Protocol`, `Literal` and `TypedDict`.
- Docstrings use single backticks for inline code, and explain why, not what.
- No superlatives, filler phrases or marketing tone in code, comments, docstrings, docs or commit
  messages (the list: `CONTRIBUTING.md`, Reviewing AI-Generated Code). The pre-commit hook runs
  `check_ai_slop.py --critical-only`, which checks emojis only; run
  `python scripts/check_ai_slop.py` without the flag to see the rest.
- **Dependency injection.** A class that depends on an external system (Neo4j, LLM backend,
  embeddings, event store) takes it as a required constructor parameter; no hidden construction
  in `__init__`. Real wiring lives in `backend/factories.py`; tests pass fakes to constructors.
- **Errors.** I/O error handling uses the specific `MistError` subclasses in `backend/errors.py`.
  Never catch bare `Exception` in new code.
- **Async boundaries.** Never call sync Neo4j from async code: use `GraphExecutor`
  (`backend/knowledge/storage/graph_executor.py`). `GraphStore` methods stay sync.
- **Resource lifetime.** Acquire, read or write, release; do CPU, I/O or inference work holding
  nothing; then acquire, write results, release. This applies to Neo4j transactions, LLM client
  calls and GPU tensor allocations. Never hold a Neo4j transaction open during LLM inference.
- **HTTP.** Every HTTP request (LLM backend, external services) calls
  `response.raise_for_status()` or checks the status code explicitly. Never consume an error
  response silently.
- **Data types.** `@dataclass(frozen=True)` for ontology and domain objects;
  `@dataclass(frozen=True, slots=True)` for new internal data structures; Pydantic `BaseModel`
  only for WebSocket message schemas or API validation. No raw dicts where a dataclass gives type
  safety.
- **External code.** Check license compatibility (MIT-compatible preferred), record it in
  LICENSE or NOTICE, preserve original copyright notices, and document your modifications.
- For multi-step implementation work in an interactive session, present the plan and get Raj's
  confirmation before executing.

---

## Commits

Conventional commits; format and types in `CONTRIBUTING.md`. The repository is public, so a
commit ends with the `Co-Authored-By` line only: no `Claude-Session:` trailer.

---

## Testing

Conventions: `TESTING.md`. Run tests inside the backend container. On Git
Bash for Windows, `MSYS_NO_PATHCONV=1` stops the shell rewriting container paths such as `/app`,
and `-T` skips TTY allocation for a non-interactive run:

```bash
MSYS_NO_PATHCONV=1 docker compose exec -T mist-backend python -m pytest tests/unit/
```

The unit tier never writes to the live graph. An autouse fixture in `tests/unit/conftest.py` sets
`MIST_EVAL_ISOLATION=1` and unsets `MIST_EVAL_NEO4J_HOSTS` for every unit test, so
`Neo4jConnection.connect()` refuses any endpoint outside the default eval allowlist, including the
live `bolt://mist-neo4j:7687` that `docker-compose.yml` sets as `NEO4J_URI` in the container.

That fixture is a live-write guard, not hermeticity: it still admits the eval endpoints, and it
does nothing about other I/O. A unit test that needs a graph or vector store injects a fake -- for
example `build_conversation_handler(graph_store=..., vector_store=..., llm_provider=...)` --
instead of letting a factory build a real one.

---

Last Updated: 2026-10-06 (claude-md-review: duplicates of CONTRIBUTING.md,
TESTING.md, the global CLAUDE.md and the harness removed; stale stack facts fixed)
