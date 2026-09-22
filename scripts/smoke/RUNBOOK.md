# Live-path smoke run: operator runbook

You are about to have ONE real conversation with MIST on a throwaway stack and
check that four artifacts appear. This document is the whole procedure. You do
not need to have read any plan, and nothing here assumes you can ask anyone a
question.

**Read the whole of Step 0 before typing anything.** Two of the steps are
ordered the way they are for reasons that are not obvious, and doing them in the
wrong order silently invalidates the result rather than failing loudly.

Every command is given literally. Where a command has an abort condition, it
says so, and "abort" means: stop, run the teardown in Step 8, and report what
you saw. Do not improvise a fix.

---

## Step 0 -- Before you start

### 0.1 What you need

- Docker Desktop running, with the live MIST stack in whatever state it is
  normally in. You will not stop it, and nothing here touches it.
- A Git Bash shell (this is a Windows machine), at the repository root
  `D:\Users\rajga\mist.ai`.
- The backend image. If `docker images | grep mistai-mist-backend` finds
  nothing, build it with `docker compose build mist-backend`.
- `websockets` in the Python you will run the driver with. Check now:

      python -c "import websockets; print(websockets.__version__)"

  If that fails, `python -m pip install websockets`. The driver prints the
  version it actually resolved into its transcript, so you do not have to trust
  this check later.

### 0.2 Two things about this machine

- **Use `python`, not `python3`.** `python3` opens the Windows Store prompt
  here and hangs.
- **Prefix any `docker compose exec` or `docker exec` that mentions a container
  path with `MSYS_NO_PATHCONV=1`.** Git Bash rewrites `/app` into a Windows
  path otherwise, and the command fails with a confusing "no such file".

### 0.3 Where the evidence goes

Pick a directory OUTSIDE the repository and keep every artifact there:

    export RUN="$HOME/mist-smoke-run"
    mkdir -p "$RUN"

This matters. Teardown ends with `rm -rf smoke-state`, and a transcript or a
baseline snapshot stored inside `smoke-state/` would be deleted along with the
thing it is evidence about.

### 0.4 THE GIT HAZARD: `smoke-state/` is not ignored

`dev-state/` is in `.gitignore` (`.gitignore:98`). `smoke-state/` is **not** --
`grep -n 'smoke-state' .gitignore` returns nothing. So from the moment you
create it until teardown, `git status` in this repository will show
`smoke-state/` as untracked, holding an event store, a vault and a vector store.

**While `smoke-state/` exists, never run `git add -A` or `git add .` in this
repository.** Stage files by name. Teardown removes the directory and the
hazard with it.

This is a known one-line follow-up (add `smoke-state/` to `.gitignore`) that was
outside the write zone of the task that produced this runbook. It is not a
containment defect: nothing in the smoke stack writes to a tracked path.

### 0.5 Environment for this shell

    export MIST_DEV_WS_HOSTS=localhost:8003,127.0.0.1:8003

The driver refuses to connect to anything the isolation guard does not
recognise. That guard's denylist (`backend/knowledge/eval_isolation.py:120-126`)
refuses the LIVE backend -- `localhost:8001`, `127.0.0.1:8001`,
`mist-backend:8001` -- and no environment variable widens it. Its allowlist arm
defaults to the DEV endpoints (`eval_isolation.py:79`, port 8002), so the smoke
port has to be named explicitly. The variable above is exactly that and nothing
more; it cannot admit a live endpoint.

Do NOT export `EVENT_STORE_DB_PATH`, `MIST_VAULT_ROOT` or `MIST_SIDECAR_DB_PATH`
in this shell. The base `docker-compose.yml` reads those same names from the
host shell (`docker-compose.yml:39,45,46`), so exporting one would retarget the
LIVE backend on its next recreate.

---

## Step 1 -- Baseline BEFORE anything exists

Run this before `docker compose up`, not after.

    mkdir -p smoke-state
    python -m scripts.smoke.baseline --phase pre --out "$RUN/baseline-pre.json"

The ordering is the point. If the compose file were mis-built and wrote
somewhere it should not, a baseline taken afterwards would record the damage as
the starting state. Taken first, the same damage shows up as a delta in Step 7.

**Abort if:** the command prints a `WARNING: these checks could not be read`
line naming `B1_live_event_store` or `B4_live_neo4j`. A check that could not run
cannot later prove isolation, and Step 7 will correctly report UNDECIDED rather
than clean. Fix the reason (usually: Docker is not running, or the live stack is
down) and re-take the baseline.

Also snapshot the smoke vault, so A3 has a real before-list rather than an
assumption:

    python -m scripts.smoke.assert_artifacts --snapshot-vault "$RUN/vault-before.json" \
        --smoke-state ./smoke-state

It will report 0 files. That is expected for a fresh `smoke-state`.

---

## Step 2 -- Phase 1: bring the stack up with the scheduler OFF

    docker compose -f docker-compose.yml -f docker-compose.override.yml \
      -f docker-compose.live-path-smoke.yml --profile smoke up -d \
      mist-neo4j-smoke mist-backend-smoke

`MIST_SMOKE_SCHEDULER` is unset, which resolves
`MIST_CURATION_SCHEDULER_ENABLED` to the literal `0`
(`docker-compose.live-path-smoke.yml:321`). The scheduler is off on purpose for
the conversation; Step 5 turns it on. Why that ordering is load-bearing is
explained there.

Every command in this runbook names its services explicitly. A bare
`docker compose up -d` or `docker compose down` reconciles the whole project,
**including the live services**.

Wait for the backend to become healthy. Its healthcheck has a 120-second start
period (`docker-compose.live-path-smoke.yml:370`), and Whisper still loads even
with TTS off (`backend/voice_models/model_manager.py:119-120`), so allow a
couple of minutes:

    docker ps --filter name=mist-backend-smoke --format "{{.Names}} {{.Status}}"

**Abort if:** after five minutes it is not `(healthy)`. Read
`docker logs mist-backend-smoke` and report what it says.

---

## Step 3 -- Prove the isolation, before writing anything

Three checks. All three must pass before you speak a single turn.

### 3.1 The mounts (this is the authoritative one)

    docker inspect mist-backend-smoke --format "{{json .Mounts}}"

Read the `Source` of every entry. The only writable host path must be the
repository's `smoke-state`. The code mounts (`backend`, `src`, `dependencies`,
`scripts`, `tests`, `voice_profiles`) must all show `"RW":false`.

**Abort if:** any `Source` ends in `\data`, `\mist-memory` or `\dev-state`.
Those are the live event store, the live vault, and the hydration fixture. If
one is mounted, the containment model's first layer is not in place and nothing
downstream of here means anything.

### 3.2 What the container can see

    MSYS_NO_PATHCONV=1 docker exec mist-backend-smoke ls -la /app/mist-memory /app/dev-state

Expect **"No such file or directory" for both.** Nothing in
`docker/backend/Dockerfile` creates them and the compose file does not mount
them.

    MSYS_NO_PATHCONV=1 docker exec mist-backend-smoke ls -la /app/data

This one **does** exist, and that is correct rather than alarming:
`docker/backend/Dockerfile:106` runs
`mkdir -p /app/data/voice_profiles /app/data/vector_store /app/data/audio`, so
the directory is baked into the image. It is a container-layer directory, not
the host's `./data`.

**Abort if:** `/app/data` contains an `event_store.db`, a `vault_sidecar.db`, or
anything under `vector_store/`. Empty directories are the expected state; live
content there would mean the host directory got mounted after all, and 3.1
should already have caught it.

    MSYS_NO_PATHCONV=1 docker exec mist-backend-smoke ls -la /app/smoke-state

Expect this to exist and be writable by the container's user.

### 3.3 The negative control

This one asserts the Neo4j guard actually refuses the live graph from inside
this container. It is a pure function and makes no network call
(`backend/knowledge/eval_isolation.py:366-398`), so it touches nothing.

    MSYS_NO_PATHCONV=1 docker exec mist-backend-smoke python -c "
from backend.knowledge.config import Neo4jConfig
from backend.knowledge.eval_isolation import EvalIsolationError, assert_neo4j_isolated
try:
    assert_neo4j_isolated(Neo4jConfig(uri='bolt://mist-neo4j:7687'))
except EvalIsolationError as exc:
    print('[OK] live URI refused:', exc)
else:
    print('[ABORT] the LIVE URI PASSED the guard')
    raise SystemExit(1)
assert_neo4j_isolated(Neo4jConfig(uri='bolt://mist-neo4j-smoke:7687'))
print('[OK] smoke URI accepted')
"

Expect `[OK] live URI refused:` followed by `[OK] smoke URI accepted`.

**Abort if:** you see `[ABORT]`, or if the smoke URI raises. The first means
`MIST_EVAL_ISOLATION` or `MIST_EVAL_NEO4J_HOSTS` did not reach the container
(`docker-compose.live-path-smoke.yml:206-207`); the second means the allowlist
does not name the smoke endpoint. Either way the backend could open a driver to
the wrong graph.

---

## Step 4 -- Schema, and the one thing you must NOT run

### 4.1 Initialise the smoke graph schema

    MSYS_NO_PATHCONV=1 docker exec mist-backend-smoke python scripts/initialize_schema.py

This creates constraints and indexes only. Every statement it issues is
`IF NOT EXISTS`, so re-running it is harmless.

### 4.2 DO NOT run `python scripts/mist_admin.py seed`

`initialize_schema.py` finishes by PRINTING "Next steps: 1. Run:
`python -m scripts.mist_admin seed`". **Ignore that line.** It is generic
advice for setting up a normal instance, not part of this procedure.

The reason is specific and worth knowing, because a future reader who does not
know it will add the step back as a convenience:

- `cmd_seed` calls `reseed(..., allow_live=True)`
  (`scripts/mist_admin.py:164-170`).
- `allow_live=True` makes `_assert_seed_target_permitted` return immediately
  without checking anything (`backend/knowledge/seed/applier.py:109-110`). It is
  the only caller in the repository that passes it, and it is ambient-config
  driven -- it refuses nothing.
- The experiment does not need seed DATA at all. The extraction path MERGEs its
  own anchor node: `graph_writer.py:210-216` creates the
  `ConversationContext` this run's assertions join through.

So seeding would add a guard-bypassing write for no benefit. Do not run it.

---

## Step 5 -- Have the conversation

    python -m scripts.smoke.drive_turns \
        --url ws://localhost:8003/ws \
        --transcript "$RUN/transcript.jsonl"

The driver speaks five turns and closes cleanly. Expect it to take several
minutes: each turn is a real LLM generation.

**Write down the `SMOKE_SESSION_ID=<uuid>` line it prints.** The server mints
that id itself (`backend/server.py:787`), once per connection; you cannot supply
one, and every assertion in Step 6 joins on it. It is also recorded in the
transcript, so it is not lost if you miss it on screen.

The driver also prints turn 5's answer. Turn 5 is a retrieval probe, not one of
the four artifacts -- whether the answer names Redpanda and Kafka is a finding
worth reporting either way.

Exit codes, which report whether the INSTRUMENT worked, not whether the pipeline
did:

| Code | Meaning | What to do |
|------|---------|------------|
| 0 | conversation ran to the end | continue |
| 2 | the isolation guard refused the URL | read its message; it names the variable to set |
| 3 | `websockets` is not installed | see Step 0.1 |
| 4 | the corpus file is broken | report it; do not edit `turns.json` |
| 5 | could not connect | the backend is not up; see Step 2 |
| 6 | a turn timed out or the socket dropped | continue to Step 6 anyway -- the transcript records how far it got, and A1 will correctly report INCONCLUSIVE rather than FAIL |

### 5.1 CAPTURE THE LOGS NOW, BEFORE STEP 6

This step is not optional and it cannot be done later.

    docker logs mist-backend-smoke > "$RUN/phase1-console.log" 2>&1
    MSYS_NO_PATHCONV=1 docker exec mist-backend-smoke cat /app/logs/mist-backend.log \
        > "$RUN/phase1-debug.log"

Two files because there are two streams at two levels:

- The console stream is INFO and above (`backend/server.py:72`). It carries the
  `Extraction skipped (` lines that assertion A2 needs.
- The file stream is DEBUG and above (`backend/server.py:75-80`). It carries the
  only two lines that can attribute the session note for A3, both `logger.debug`
  (`backend/chat/conversation_handler.py:1962` and `:1970`), which never reach
  the console.

`docker-compose.live-path-smoke.yml:324-330` deliberately bind-mounts no
`/app/logs`, so the file log lives in the container's writable layer. Step 6
replaces that container with `--force-recreate`, and both logs go with it. If
you skip this step, A2 and A3 degrade from decidable to INCONCLUSIVE for this
run and cannot be recovered.

---

## Step 6 -- Phase 2: recreate with the scheduler ON

    MIST_SMOKE_SCHEDULER=1 docker compose -f docker-compose.yml \
      -f docker-compose.override.yml -f docker-compose.live-path-smoke.yml \
      --profile smoke up -d --force-recreate mist-backend-smoke

The ordering -- conversation first, scheduler second -- is load-bearing.
`backend/knowledge/curation/scheduler.py:305-311` reads
`last_run.get(config.name, 0.0)`, which makes every enabled job due on the
loop's FIRST pass. `self_reflection` is registered with
`interval_seconds=86400` (`backend/factories.py:751`), so a scheduler that
started before any turn existed would spend its single daily run on an empty
table and not run again for 24 hours.

Wait for health, then give the scheduler loop a minute -- it sleeps 60 seconds
between passes (`scheduler.py:326-327`):

    docker ps --filter name=mist-backend-smoke --format "{{.Names}} {{.Status}}"

Then capture the Phase 2 log, which is where A4's evidence is:

    docker logs mist-backend-smoke > "$RUN/phase2-console.log" 2>&1

---

## Step 7 -- Assert, then prove isolation held

### 7.1 The four artifacts

    python -m scripts.smoke.assert_artifacts \
        --session-id <the uuid from Step 5> \
        --smoke-state ./smoke-state \
        --transcript "$RUN/transcript.jsonl" \
        --vault-before "$RUN/vault-before.json" \
        --backend-log "$RUN/phase1-console.log" \
        --backend-log "$RUN/phase1-debug.log" \
        --backend-log "$RUN/phase2-console.log"

Each of A1 to A4 prints PASS, FAIL or INCONCLUSIVE with its evidence. Exit code
is 1 if anything FAILed, 0 otherwise.

**If your `.env` sets a non-default `NEO4J_PASSWORD`, add `--neo4j-password
"$NEO4J_PASSWORD"`.** The flag defaults to `password`
(`assert_artifacts.py:1252`), matching the compose default
`${NEO4J_PASSWORD:-password}`. If the real credential differs, the A2
`cypher-shell` call fails authentication and A2 reports INCONCLUSIVE for an
INSTRUMENT reason rather than a pipeline one. It degrades safely -- it cannot
produce a false FAIL, and the cypher error is printed verbatim -- but you would
be reading an inconclusive A2 caused by this line rather than by MIST. Check
the printed error before concluding anything about extraction.

Read the three verdicts as they are meant:

- **PASS** -- the artifact is there.
- **FAIL** -- the pipeline was exercised and did not produce the artifact. This
  is a finding.
- **INCONCLUSIVE** -- the run cannot say. Usually because the step was never
  exercised (fewer than five turns completed), or because the evidence needed to
  tell two explanations apart was not available. **An INCONCLUSIVE is not a
  pass, and it is not a failure.** Reporting one as the other is the specific
  mistake this instrument exists to prevent.

A4 FAILing with `examined=0` is the outcome the experiment was built to detect:
it is the live stack's exact symptom, reproduced on a stack where every input is
known.

Save the output. Re-run the same command with `| tee` appended, so the verdicts
end up in `$RUN` alongside the transcript and the logs they were derived from:

    python -m scripts.smoke.assert_artifacts \
        --session-id <the uuid from Step 5> \
        --smoke-state ./smoke-state \
        --transcript "$RUN/transcript.jsonl" \
        --vault-before "$RUN/vault-before.json" \
        --backend-log "$RUN/phase1-console.log" \
        --backend-log "$RUN/phase1-debug.log" \
        --backend-log "$RUN/phase2-console.log" | tee "$RUN/assertions.txt"

### 7.2 The isolation proof

    python -m scripts.smoke.baseline --phase post --out "$RUN/baseline-post.json"
    python -m scripts.smoke.baseline --compare "$RUN/baseline-pre.json" "$RUN/baseline-post.json"

Verdicts and exit codes:

- **CLEAN** (0) -- every delta is UNCHANGED or EXPECTED.
- **CONTAMINATED** (1) -- live or dev state moved. The artifact results are not
  reportable until each CONTAMINATION has an explanation that is not the smoke
  stack.
- **UNDECIDED** (2) -- nothing looks contaminated, but a check could not run in
  one of the phases. Isolation was not proven. A check that could not run is not
  a check that passed.

One delta is genuinely ambiguous and the tool adjudicates it for you: new rows
in the LIVE `curation_job_runs` are EXPECTED, because the live scheduler runs on
its own timer. The report states that reasoning inline so you do not have to
take a bare verdict on trust. A live row with `trigger_source='manual'` is
flagged REVIEW instead -- that would mean somebody ran a curation pass by hand
during the window.

---

## Step 8 -- Teardown

Run these in order. **Never `docker compose down`**: without service names it
reconciles the whole project, including the live stack.

    docker compose -f docker-compose.yml -f docker-compose.override.yml \
      -f docker-compose.live-path-smoke.yml --profile smoke stop \
      mist-backend-smoke mist-neo4j-smoke

    docker compose -f docker-compose.yml -f docker-compose.override.yml \
      -f docker-compose.live-path-smoke.yml --profile smoke rm -fsv \
      mist-backend-smoke mist-neo4j-smoke

    docker volume rm mist.ai_mist-neo4j-smoke-data mist.ai_mist-neo4j-smoke-logs

    rm -rf smoke-state

Everything you need is already in `$RUN`. If you want to keep the smoke event
store, copy it out BEFORE the last command.

---

## Step 9 -- Verify the teardown

    docker ps -a --format "{{.Names}}" | grep smoke
    docker volume ls | grep smoke

Both must print nothing.

    ls smoke-state

Must say "No such file or directory".

    git status --porcelain

Must not list `smoke-state/`.

    docker ps --format "{{.ID}} {{.Names}}"

Compare against the `B8_docker_ps` block in `$RUN/baseline-pre.json`. Every live
container must have the SAME id it had before. A same-name, different-id
container was recreated, which a smoke command should never have done.

    python -m scripts.smoke.baseline --phase post --out "$RUN/baseline-teardown.json"
    python -m scripts.smoke.baseline --compare "$RUN/baseline-pre.json" "$RUN/baseline-teardown.json"

Expect CLEAN, now with the smoke containers gone from the `B8_docker_ps` delta.

---

## What this run does NOT cover

State this plainly in any report of the result. The run exercises the TEXT path
only, and the following are untested by it:

- **The Tauri frontend's own message construction.** The driver sends
  `{"type": "text", "text": ...}` because that is what
  `backend/server.py:856-858` reads. Whether the shipped frontend sends exactly
  that, and nothing else that matters, is not established here.
- **The Tauri shell**: its WebSocket lifecycle, reconnection, and anything it
  does around the connection.
- **Anything the real frontend sends that this script does not** -- `interrupt`,
  `reset_vad`, `log_config`, `subscribe_logs`, and the binary audio frames are
  all in the endpoint's contract (`backend/server.py:747-758`) and none is
  exercised.
- **Whisper and VAD**, and everything above `voice_processor.py:483`
  `_process_conversation_turn`. The text and voice paths converge at that
  function; this run enters below the convergence point, so the whole
  audio-ingest half is untested.

On voice specifically: the brief that commissioned this instrument states that
the shipped frontend cannot capture audio at all, there being no `getUserMedia`
or `MediaRecorder` in `mist-frontend/src/`, and so voice was not an available
input route. **That claim is UNVERIFIED from this repository** -- `mist-frontend/`
is a separately-cloned nested repository and is absent from the checkout this
runbook was written in, so it could not be checked here. Treat it as a reason
that was given, not as a fact this document establishes.

---

## Reference: the four assertions

| | Assertion | Source of truth |
|---|-----------|-----------------|
| A1 | 5 turn rows, 1 session row, `origin='real'`, `input_modality='text'`, `turn_count=5` | `smoke-state/event_store.db` |
| A2 | at least one `:__Entity__` with `EXTRACTED_FROM` to this session's `ConversationContext` | smoke Neo4j, via `docker exec mist-neo4j-smoke cypher-shell` |
| A3 | a `YYYY-MM-DD-<slug>.md` under `smoke-state/vault/sessions/` | the filesystem, plus the DEBUG log for attribution |
| A4 | a `scheduled` `self_reflection` row with `examined > 0` | `curation_job_runs` in `smoke-state/event_store.db` |

Two deliberate omissions, so nobody adds them later:

- **A1 does not assert on `ended_at`.** That column is written only by
  `EventStore.end_session` (`backend/event_store/store.py:156`), whose sole
  caller is `ConversationHandler.clear_session`
  (`backend/chat/conversation_handler.py:2874`), which has zero production
  callers (`backend/server.py:262-265`). It stays NULL after a completely normal
  disconnect, so an assertion on it would fail for a reason unrelated to the
  pipeline.
- **A2 does not grep the log for an extraction success message.** There is not
  one. `backend/knowledge/curation/graph_writer.py:198` gates its only log on
  `source_metadata is not None and result.document_provenance_edges > 0` -- the
  DOCUMENT branch. The conversational branch (`graph_writer.py:194-196`) logs
  nothing at all. The Cypher count is the only positive evidence there is.
