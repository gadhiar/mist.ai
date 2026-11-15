# Capabilities

This package is how MIST acts on the world: she keeps the time, sets alarms and timers, switches
lights through Home Assistant, and controls music through Spotify. Each of these is a
capability. One `CapabilityHost` (`host.py`) admits, decides, runs and logs every call any of
them makes, so one validation step, one policy, one redaction pass and one action log apply to
all of them, whichever path chose the call. `backend/factories.py::build_capability_host` is the
only place the host and its capabilities are assembled.

A turn reaches the host in two steps:

1. `pre_llm` answers without a model, in this order (`host.py::CapabilityHost.pre_llm`): an
   alarm or timer that is ringing takes a stop or snooze phrase; a yes or a no decides the
   session's waiting proposal; otherwise a waiting proposal is superseded; then the fast path
   runs the one capability whose grammar matches the whole utterance.
2. When `pre_llm` does not answer, `plan_turn` routes the utterance to at most two
   capabilities and builds a constrained decode over their tools, and `run_actions` runs the
   calls the decode chose.

## Adding a capability

### The base class

Subclass `Capability` (`contracts.py`). The abstract members are `tools`, `routing_hints`,
`start`, `stop`, `status` and `invoke`; the other hooks (`dynamic_keywords`, `refine_schema`,
`match_intent`, `escalate`, `active`, `on_active_utterance`, `provider_tools`) have inert
defaults, so implement only what the capability uses. Two rules hold for every capability:

- `status()` reads in-memory state and makes no network call: the host reads it at every call,
  in the status watcher's poll each second, and for `GET /capabilities`.
- `start()` returns without waiting for a provider to connect. The host starts every
  capability in a task of its own and returns at once (`CapabilityHost.start`); a `start` that
  raises makes that capability `unavailable` with the reason, and the others still run.

Then wire it:

- add its name to `KNOWN_CAPABILITIES` (`config.py`), or `MIST_CAPABILITIES` refuses it;
- build it in `build_capability_host`'s loop, passing every external system (a client, a
  clock, a transport) to its constructor, so a test can pass fakes;
- optionally add a keyword floor for it to `_MINIMUM_KEYWORDS` (`router.py`); a capability can
  add keywords through `routing_hints()` and `dynamic_keywords()` but never remove one.

### At most four tools, under a token ceiling

The decode runs on a small model, which chooses better among fewer, shorter options. So the
router (`router.py`) refuses to register a capability that offers more than four tools, and
offers each turn the tools of at most two capabilities.

The token ceiling is `MIST_CAPABILITY_TOOL_TOKEN_BUDGET` (default 1200): the decode prompt's
catalog plus its fixed instructions (`decoder.py::DECODE_INSTRUCTIONS`). It holds by
construction: registration refuses a capability whose catalog lines, plus its
`refine_allowance_tokens`, need more than half of what the ceiling leaves after the
instructions (`ToolRouter.per_capability_budget`), so any two admitted capabilities fit
together. `refine_schema` may add live detail to a tool's schema, such as the lights target
enum; the host falls back to the static schema for a turn when the refined lines cost more
than the static lines plus that allowance.

### Risk and policy

Each `ToolSpec` declares a `risk` (`read`, `reversible` or `physical`) and a `default_policy`
(`allow`, `confirm_inferred`, `confirm` or `deny`). `policy.py::PolicyTable.decide` applies
the mode by the call's source:

- `deny` refuses every call;
- a call run by an approved proposal is allowed, because the approval was the confirmation;
- `confirm` holds every other call for a yes or a no;
- `confirm_inferred` runs a fast-path call and holds a call the model chose;
- `escalate(tool, args)` can raise an allowed model or fast-path call to a confirmation, for
  arguments that need one whatever the policy.

Held calls of one turn become one proposal, answered by a spoken yes or no or by the client's
`action_decision`; it expires after `MIST_CAPABILITY_PROPOSAL_TTL_S`. `MIST_CAPABILITY_POLICY_FILE`
names a YAML file of `{tool_name: mode}` overrides; a file that does not load is logged and the
defaults apply (`build_capability_host`).

A spec's `ack` is a phrase of at most five words a voice turn may speak while a slow call runs,
and its `view` names the screen view the outcome's `data` fills. Replies about outcomes are
templates (`replies.py`), never model text, so a reply cannot say an action ran when it did not.

### The fast-path grammar and its fixtures

`match_intent(text)` matches a whole `intents.py::normalize_utterance` output against anchored
rows and returns a `FastIntent` (a tool and unvalidated arguments) or None. The fast path runs
a call only when exactly one capability matches; with none, or two or more, the utterance goes
on to routing and the decode (`CapabilityHost._fast_path`). A grammar should match only
explicit commands: each grammar module's docstring lists its rows and the slot rules that keep
questions, compound requests and scheduled requests out (for example
`lights/intents.py`, `music/intents.py`, `alarms/intents.py`).

The fixtures are each grammar's parametrized lists of utterances that must match, with their
expected intent, and utterances that must not, in `tests/unit/capabilities/<capability>/test_intents.py`
(for example `MATCHING` and `NOT_MATCHING` in the lights and music tests). Add both kinds for
every row. `tests/unit/capabilities/test_fast_path_arbitration.py` shows two capabilities
matching one utterance ("turn off the music" with a light named "Music"), so neither runs.

### The catalog line the decode shows

The decode sees each offered tool as one line, rendered by `schema.py::to_catalog_line` from
the tool's name, its arguments schema and its description:

```
lights.power(target?: bedroom|kitchen, on: true|false): turn lights on or off
```

Optional arguments are marked `?` and enum members are joined by `|`. Keep the description
short: it is what the model chooses by, and it counts against the token ceiling.

### Offline behaviour

Every capability must degrade offline. While a capability is `unavailable` or `disabled`, the
host's status gate answers each of its calls `unavailable` with the status reason, before policy
and without calling it, and the templated reply names the reason ("Lights is unavailable:
connection refused."). A capability whose provider fails during a call returns an
`unavailable` outcome rather than raising: lights catches the MCP connection's errors
(`lights/capability.py::LightsCapability.invoke`), and music marks Spotify `unreachable`,
`unavailable` for 60 s and then `degraded` so the next call can find out whether it is back
(`music/capability.py`). Alarms, timers and the clock need no network and keep working.
`tests/unit/capabilities/test_offline_degradation.py` holds this for the assembled host.

### Tests with fakes

Test a capability through its constructor with fakes, never against a live service:
`tests/unit/capabilities/fakes.py` (`FakeClock`, `FakePower`, `FakeTiers`, `RecordingEventSink`,
`ScriptedDecode`, `FakeCapability`), `tests/unit/capabilities/mcp/fake_home_assistant.py` for
Home Assistant, and an `httpx.MockTransport` for an HTTP API. `FakeClock` moves time only when a
test calls `advance`, so nothing waits on the wall clock. `tests/unit/conftest.py` sets
`MIST_EVAL_ISOLATION=1` for every unit test, which stops `build_capability_host` from building
a host; a test of the assembled host sets it to `0` (see `test_factory.py`'s `enabled` fixture).

## Why a constrained decode, not native tool calls

MIST's interactive model is a small local model. A small model chooses better among fewer,
shorter options, and a short prompt prefills quickly (`decoder.py`). So a routed turn gets a
separate decode prompt that shares nothing with the main chat prompt (no persona, retrieval or
profile text): the fixed instructions, a few recent messages, the routed tools' catalog lines
and the utterance. Its JSON schema becomes a grammar in llama-server, so tool names and enum
slots are limited to what the turn offers. A decode chooses at most three calls in at most 256
tokens, and a reply cut off at that bound is refused, never read in part. A call whose
arguments fail validation gets one repair decode. A failed decode runs no action, and the
turn goes on as it would without capabilities, with a notice.

## Records, ringing and fallback

Alarms and timers live in `capabilities.db` (`MIST_CAPABILITIES_DB_PATH`), which the backups
capture (`scripts/backup/stores.py::LIVE_STORE_FILENAMES`):

- An alarm or timer is a record, one per distinct alarm, never deleted (`alarms/records.py`).
  A one-off, a recurring alarm (a weekday set) and a tagged alarm (a label) each have their
  own identity rules, and every arm that creates an occurrence adds 1 to the record's use
  count. An occurrence that ends folds into its record's counters, and every change appends to
  an event trail.
- `AlarmScheduler` (`alarms/scheduler.py`) is the one writer of occurrence transitions. An
  occurrence found more than the grace window late (`MIST_ALARM_MISSED_GRACE_S`, never under
  60 s) is missed, reason `late`.
- `RingRouter` (`alarms/ringing.py`) rings each occurrence on the endpoints its `ring_on` names
  (`all` by default). Today the one endpoint is `desktop`, every connection of the WebSocket,
  present while a client is connected. When none of the named endpoints is present, the ring
  goes to every present endpoint instead and a `fallback` event is recorded.
- A ring counts as heard only when an endpoint acknowledges playing it (an `alarm_report` of
  `rang`). One nobody acknowledged ends missed, reason `unheard`, after
  `MIST_ALARM_RING_TIMEOUT_S`.
- `alarm_state` carries the settings an endpoint rings by (ring timeout, snooze, grace window,
  `MIST_RING_FALLBACK_AFTER_S` and the zone), so an endpoint that cannot reach the backend
  rings on its own after the fallback delay and reports the outcome when it reconnects.

Alarms, timers and the clock need `MIST_TIMEZONE`. While it is unset or unknown, no scheduler
is built and the three report `unavailable` with the reason, so an alarm never rings at an hour
from a guessed zone (`factories.py::_user_zone`, `build_capability_host`).

## Power levels

The power level is `full`, `low` or `off` (`power.py`). Capabilities read it into every record,
action-log row and `capabilities_status`, and decide nothing on it. Only the constrained decode
needs a model: `plan_turn` builds no decode while `decode_available()` is False, which is when
the `interaction` tier has no provider. Every `pre_llm` step runs at every level, off included,
so the fast path, ringing alarms and proposal decisions keep working.

Until the tier router lands, two stopgaps serve: `GpuYieldPower` reads `full` while gaming mode
is `normal` and `off` while it is `yielded` or `resume_failed`, so nothing sets `low` yet; and
`SingleModelTiers` serves both tiers from today's one model, and none at `off`.

While gaming mode is `yielded` or `resume_failed`, the WebSocket refuses spoken messages, because
Whisper is released. With a host, a typed message goes
to `ConversationHandler.handle_capability_only`, which runs only `pre_llm` and calls no model:
a message it handles is answered as a normal text turn, and any other gets the yielded reply.
With no host (capabilities disabled), every typed message gets the yielded reply
(`backend/server.py`, the `audio` and `text` gate).

## The usage view

`GET /capabilities/usage?days=N` (N from 1 to 90, default 7) reports capability use and alarm
events per local day, newest first, one entry per day even when it is empty
(`CapabilityHost.usage`). Days are local dates in `MIST_TIMEZONE`, resolved as
`GET /usage/turns-per-day` resolves it; unset or unknown counts in UTC and `tz` says `UTC`. The
body holds names, dates and numbers, never an argument, summary or secret:

- `tz`, `since` (the oldest action-log row) and `alarms_since` (the oldest alarm event);
- `days`: per day, `invocations`, and counts `by_capability`, `by_source` (`fastpath`,
  `model`, `proposal`), `by_level` (`full`, `low`, `off`) and `by_status`; `turn_ms`, the p50
  and p90 of the time from a message's receipt to each action's outcome; and `turn_ms_by_level`,
  the same p50 and p90 for each of `full`, `low` and `off`, null where that level has no row;
- `alarms`: per day, the counts of `fired`, `stopped`, `timed_out`, `snoozed`, `missed_late`,
  `missed_unheard`, `dismissed`, `reported` and `fallback`.

With no host the body is `{"enabled": false, "reason": ...}`, and a capabilities database that
cannot be read gets 503 with `{"error": "store_unavailable"}`.

## Environment variables

`load_capabilities_config` (`config.py`) parses every variable once; a blank value counts as
unset. `.env.example` documents each with its default.

| Variable | Default | Meaning |
|---|---|---|
| `MIST_CAPABILITIES_ENABLED` | `false` | Build the host at all. |
| `MIST_CAPABILITIES` | all five | Which of `clock`, `alarms`, `timers`, `lights`, `music` to build. |
| `MIST_TIMEZONE` | unset | The IANA zone of alarms, timers, the clock and the usage view. |
| `MIST_CAPABILITIES_DB_PATH` | `data/capabilities.db` | Alarms, timers and the action log. |
| `MIST_CAPABILITY_POLICY_FILE` | unset | YAML policy overrides. |
| `MIST_CAPABILITY_TOOL_TIMEOUT_S` | `8` | Bound on one tool call. |
| `MIST_CAPABILITY_PROPOSAL_TTL_S` | `60` | How long a proposal waits; at most 600. |
| `MIST_CAPABILITY_TOOL_TOKEN_BUDGET` | `1200` | The decode prompt's token ceiling. |
| `MIST_CAPABILITY_FAST_PATH` | `true` | Run the fast path. |
| `MIST_CAPABILITY_DECODE_TIMEOUT_S` | `10` | Bound on one decode. |
| `MIST_CAPABILITY_WAKE_WORDS` | `mist,hey mist,ok mist,okay mist` | Leading words stripped before matching. |
| `MIST_CAPABILITY_ROUTER_STICKY_TURNS` | `2` | Turns a capability stays routed. |
| `MIST_CAPABILITY_ROUTER_EMBEDDING_THRESHOLD` | unset | Embedding stage threshold; no embedder is wired yet, so the stage stays off. |
| `MIST_ALARM_RING_TIMEOUT_S` | `600` | How long a ring lasts; at most 3600. |
| `MIST_ALARM_SNOOZE_MINUTES` | `9` | Snooze length, 1 to 60. |
| `MIST_ALARM_MISSED_GRACE_S` | `900` | How late an alarm may still ring; never under 60 in effect. |
| `MIST_RING_FALLBACK_AFTER_S` | `30` | An endpoint's own-ring delay, 5 to 300 and below the grace window. |
| `MIST_HA_MCP_URL` | unset | Home Assistant's MCP server; lights is built only with it. |
| `MIST_HA_TOKEN`, `MIST_HA_TOKEN_FILE` | unset | Its token, or a file holding it (`secrets.py::EnvSecretSource`). |
| `MIST_LIGHTS_DEFAULT_TARGET` | unset | The light or area a command that names none acts on. |
| `MIST_SPOTIFY_CLIENT_ID` | unset | Spotify PKCE client; music is built only with it. |
| `MIST_SPOTIFY_REDIRECT_URI` | `http://127.0.0.1:8888/callback` | The authorization redirect. |
| `MIST_SPOTIFY_TOKEN_PATH` | `data/secrets/spotify_token.json` | The token file. |
| `MIST_SPOTIFY_DEVICE` | unset | Where playback goes when no device is active. |

Secret values are registered for redaction and never returned by `GET /capabilities`.

### The Origin allowlist

Capabilities share the backend's one allowlist, `MIST_ALLOWED_ORIGINS` (`backend/origins.py`).
The WebSocket, `GET /capabilities` and `GET /capabilities/usage` refuse a request whose
`Origin` the list does not hold; a request with no `Origin` is a local program and is served.
While the list holds `*`, any web page could drive lights and music, so no host is built and
`GET /capabilities` reports "MIST_ALLOWED_ORIGINS allows every origin"
(`factories.py::_capability_disabled_reason`). Eval and hydration isolation
(`MIST_EVAL_ISOLATION`, `MIST_HYDRATION_ISOLATION`) also build no host, so an eval run never
reaches real lights, music or alarms.

## The token file mode on Windows

`SpotifyTokenStore.save` (`music/token_store.py`) writes the Spotify token file with mode 0600,
because the token grants control of the account. On a Windows host the file sits on a bind mount
(`./data` in the workstation compose, `<root>/secrets` in the daily instance), and a Windows
bind mount does not keep POSIX modes, so the `chmod` has no effect there. On Windows the file is
protected by the host folder's own permissions: keep that folder readable by your account only.
