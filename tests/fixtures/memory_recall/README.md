# Synthetic memory recall fixture

This directory holds the fixture that the read path (plan mist-read-path) is measured against: the
fetcher floors (T9, T19, T21), the episode segmenter (T17), the answer probes (T27) and the
real-model eval (H4). The schemas live in `scripts/eval_harness/memory_recall.py`, along with the
loaders and the scorer. `tests/unit/eval_harness/test_memory_recall_fixture.py` checks every rule
on this page.

## Provenance

Everything here is synthetic. The user, Wren Calloway, is fictional, and so are her partner Jonah,
her sister Imogen, her mother Ada, her friends and colleagues, her dog Biscuit, and every
employer, shop, place and product named. No row is drawn from a real person, a real conversation
or a real memory store. Nothing here is about the repository's owner. There are no hostnames, IP
addresses, credentials or tokens. The data was written by hand for this fixture, with a few
mechanical expansions (the Portuguese vocabulary drill, the packing list, the field guide entries)
written out turn by turn. The four list conversations (see conversations.jsonl) are fictional in
the same way: every book, author, tool and habit in them is invented.

The history covers 14 days of conversations, Monday 2 March to Sunday 15 March 2026, with every
timestamp in UTC. Facts and notes from before that window give the history its background.

## Files

Every file is JSONL: one JSON object per line, ASCII only. Every key listed is required, nullable
keys included. The loaders refuse a missing or unknown key and name the line, so a dropped key
never turns silently into a default. Timestamps are ISO 8601 with an explicit offset.

### facts.jsonl

One row per fact version, in the shape of the graph's fact rows.

| key | meaning |
| --- | --- |
| `row_key` | unique id of this version (`fv-NNN`) |
| `fact_key` | `sha256(subject_id + "\|" + relation + "\|" + object_id)`, full hex digest; this is a fact's item id |
| `subject_id`, `subject_name` | canonical id (lowercase, hyphenated) and display name |
| `subject_type` | the subject's ontology node type, one of `EXTRACTABLE_NODE_TYPES` |
| `relation` | an ontology relationship type that is extractable and directional |
| `object_id`, `object_name` | canonical id and display name |
| `object_type` | the object's ontology node type, one of `EXTRACTABLE_NODE_TYPES` |
| `text` | the authored sentence; the index serves `render_fact`'s sentence instead (`backend/memory/fact_render.py`), and the lexical tag is judged against that one |
| `valid_from`, `valid_to` | validity interval; null means open |
| `recorded_at` | when MIST learned it |
| `confidence` | in (0, 1] |
| `source_utterance_id` | the `event_id` of the turn it was extracted from, or null for background facts |
| `state` | `current`, `superseded`, `future` or `retracted`, judged at the end of the history |
| `superseded_by` | `row_key` of the version that replaced it, for `superseded` only |
| `source_correction_id` | the correction that created this row, for replacements only |

What each state means:

- A superseded row has a `valid_to` and a successor with the same subject and relation. The
  successor's `valid_from` is at or after that `valid_to`; a gap is allowed (Portuguese lapsed in
  2023 and resumed in 2026).
- A future row starts after the history ends.
- A retracted row ended without a successor: the belief was withdrawn, so its confidence is low.

A correction never rewrites the row it corrects. That row keeps the state it was recorded with, and
the read path suppresses it at read time (see Corrections). A `correct` correction adds one
replacement row, with `source_correction_id` set and `valid_from = recorded_at =` the
correction's time.

Types. `subject_type` and `object_type` come from `EXTRACTABLE_NODE_TYPES` in
`backend/knowledge/ontologies/v1_0_0.py`. An entity has one type in every row it appears in,
as subject or object. Wren is `User`; every other person, and Biscuit, is `Person`; tastes in
food, drink and travel are `Preference`, while the cafes are `Organization`; habits are `Concept`;
practised abilities such as woodworking or sourdough baking are `Skill`. Types cut across
relations on purpose: Wren `USES` both technologies and an energy supplier (`Organization`), and
she is `LEARNING` both skills and a `Concept` (Bayesian statistics), so a type-scoped probe can
only be answered by the type.

List utterances. Facts that share `subject_id`, `relation` and `source_utterance_id` are siblings,
extracted from one turn that states a list. Four lists have sixteen siblings each, and each list's
summed `text` counts more than 300 tokens with `ApproximateTokenCounter`, so the reader's
300-token block cannot hold all of it:

| source turn | subject | relation | object type | list |
| --- | --- | --- | --- | --- |
| `evt-0303` | Imogen | `RECOMMENDS` | `Document` | books to read before autumn |
| `evt-0305` | Wren | `HAS_HABIT` | `Concept` | workday habits |
| `evt-0307` | Rafael | `RECOMMENDS` | `Technology` | tools for decorating the spare room |
| `evt-0309` | Wren | `HAS_GOAL` | `Goal` | spring birding goals |

Every sibling's `object_name` appears in its turn's `user_utterance`.

### conversations.jsonl

One row per turn, in time order.

| key | meaning |
| --- | --- |
| `event_id` | the turn's event id (`evt-NNNN`); an exchange's item id |
| `session_id` | the WebSocket session; one session can hold two conversations (`sess-12`), and a conversation can span two sessions when the client reconnects inside the gap (`conv-28` moves to `sess-30` after its sixth turn), so conversations never follow session ids |
| `conversation_id` | the conversation, by the 30-minute gap rule below |
| `turn_index` | position within the session, counted from 0 |
| `timestamp` | when the turn happened |
| `user_utterance`, `system_response` | the two halves of the turn |

The four list turns (see facts.jsonl) open four conversations of their own, added after the
first thirty: `conv-31` to `conv-34`, in sessions `sess-31` to `sess-34`, with turns `evt-0303`
to `evt-0310`. Each holds two turns: the list, then a short follow-up. They sit on the evening of
Saturday 14 March and the late morning of Sunday 15 March, after the last correction and before
the last original turn, each more than 30 minutes from its neighbours. File order is time order,
so they sit among the earlier rows and their ids do not follow file order. Placing them this way
moved no existing conversation or episode boundary, left the history's end where it was, and kept
them out of every correction's suppression window. The temporal probes for that weekend ask about
Saturday morning, Saturday afternoon, Sunday afternoon or the Sunday weekly review, so none of
them should now expect a list conversation.

### episodes.jsonl

One row per episode, in conversation order.

| key | meaning |
| --- | --- |
| `episode_id` | `e-` + the first 16 hex characters of `sha256("\|".join(event_ids))`; an episode's item id |
| `conversation_id` | the conversation it belongs to |
| `event_ids`, `first_event_id`, `last_event_id` | its turns, in order |
| `start`, `end` | timestamps of the first and the last turn |
| `turn_count`, `token_count` | the sizes the segmentation rule compares |
| `closed_by` | `turn_limit`, `token_limit` or `conversation_end` |
| `title`, `summary`, `open_threads`, `entities` | the episode note; the summary is at most 80 words |

Each list conversation is one `conversation_end` episode with no open threads.

### vault_notes.jsonl

One row per vault note section.

| key | meaning |
| --- | --- |
| `path` | path relative to the vault root, ending `.md`; a note's item id. Never under `users/`, because the profile note is never a candidate |
| `title`, `heading` | the note title and the section heading |
| `authored_by` | `user`, `user-edit` (MIST wrote it, the user edited it) or `mist` |
| `created_at`, `updated_at` | both within the history |
| `tags`, `body` | frontmatter tags and the section text |

### corrections.jsonl

One row per user correction, in the logged shape.

| key | meaning |
| --- | --- |
| `correction_id` | `corr-` + 32 hex characters |
| `recorded_at` | when the correction was made |
| `action` | `correct` or `forget` |
| `fact_key`, `subject_*`, `relation`, `object_*` | the fact corrected, copied from its row |
| `replacement_object_name` | the new object for `correct`; null for `forget` |
| `surface` | `tool` (said in a turn), `ws` (a WebSocket control message) or `cli` |
| `session_id`, `conversation_id` | the issuing context; set for `tool` and `ws`, null for `cli` |
| `turn_index`, `source_event_id` | the turn that issued it; set for `tool` only |
| `origin` | always `user` |

A `tool` correction is recorded with its turn's timestamp, so that turn falls inside the
suppression window.

### probes.jsonl

One row per probe: a user turn and the block it should and should not produce.

| key | meaning |
| --- | --- |
| `probe_id` | `<kind>-NN` |
| `kind` | `fact`, `episode`, `note`, `continuity`, `temporal`, `correction` or `noise` |
| `lexical` | whether the query shares words with its reference items (rule below) |
| `query` | the user turn |
| `now` | when the probe is asked |
| `conversation_start` | true when the probe opens a new conversation |
| `expected` | items the block must contain, as `{kind, id}` |
| `acceptable` | items that are relevant but not required; they count toward precision, not recall |
| `forbidden` | items that must not appear (correction probes only) |
| `distractors` | look-alike items a careless fetcher might return; informational, not scored |
| `correction_id` | the correction under test (correction probes only) |
| `rationale` | why the probe exists and what it tests |
| `shape` | `plain`, `list_item` or `type_scoped` (rules below) |

Item kinds are `fact` (id is the `fact_key`), `episode` (the `episode_id`), `exchange` (the turn's
`event_id`) and `note` (the `path`).

### heldout_probes.jsonl

The held-out probe set, in the same schema as probes.jsonl; see Held-out probes.

## Segmentation rule

Conversations and episodes follow Design point 11, with the Phase B defaults.

1. Conversations: sort the turns by timestamp. A new conversation starts wherever the gap since
   the previous turn is more than 30 minutes. A gap of exactly 30:00 stays inside the
   conversation; a gap of 30:01 starts a new one. The fixture holds both cases.
2. Episodes: walk each conversation's turns in order with an open episode. Before adding a turn:
   - if the open episode holds 20 turns, close it as `turn_limit`;
   - otherwise, if adding the turn would take the open episode's tokens past 6000, close it as
     `token_limit`.

   Then add the turn. The turn-count check comes first. The last episode closes as
   `conversation_end`.
3. Counting: a turn's tokens are
   `ApproximateTokenCounter().count(user_utterance + "\n" + system_response)` from
   `backend/chat/context_budget.py`. An episode's `token_count` is the sum over its turns.

The fixture hits both limits exactly at the edge: one episode closes holding exactly 20 turns,
and one holds exactly 6000 tokens. An off-by-one in either comparison therefore moves a boundary.
There are three `turn_limit` episodes and two `token_limit` episodes.

## Corrections and suppression

Design point 12: from `recorded_at` on, a correction suppresses the corrected `fact_key`. It also
suppresses every episode, exchange and note recorded at or before the correction whose text
contains the corrected object name as a whole word, case-insensitively. Each kind's text is:

- an episode: title, summary and open threads, judged by its `end`;
- an exchange: the user utterance and the response, judged by its timestamp;
- a note: title, heading and body, judged by `updated_at`.

The ten corrections cover all three surfaces and both actions. Several of them reach notes as
well as facts: the dog's breed reaches `people/biscuit.md`, Jonah's clinic reaches
`people/jonah.md`, a forgotten colleague reaches `work/team-contacts.md`, and the cello and
crossword forgets reach `ideas/someday.md`.

## Probe authoring rules

- Ask after the history. Every `now` is more than 30 minutes past the last turn, so every
  conversation has closed and every episode is indexed.
- Expect only what is available. An expected or acceptable item must exist at `now`:
  - a fact needs a version recorded by `now` and valid at `now`;
  - an episode must have ended;
  - a note must have been created;
  - an exchange must have happened.

  A probe that tests a future-dated fact moves its `now` past the fact's `valid_from`.
- Never expect a suppressed item. No probe expects, or accepts, an item that a correction made
  before its `now` suppresses.
- Fill the lists as follows:
  - `acceptable` holds the turns of each expected episode, the source turn of each expected fact,
    and any related note, episode or fact the author judged a fair answer, minus anything
    suppressed;
  - `distractors` holds superseded or future versions of the expected fact, a neighbouring
    conversation on the same day, and similar look-alikes.
- Kinds:
  - `fact`, `note` and `episode` probes expect items of their own kind.
  - `continuity` probes open a conversation (`conversation_start` is true) by resuming an open
    thread, and expect episodes.
  - `temporal` probes name a conversation by its day or time ("yesterday afternoon", "the evening
    of Thursday the 12th"), resolved against `now`, and expect episodes.
  - `correction` probes name one correction each. Their `forbidden` list equals that correction's
    suppression set exactly. A `correct` probe expects the replacement fact; a `forget` probe
    expects nothing.
  - `noise` probes are utility turns (alarms, lights, music, volume, the thermostat, the weather,
    timers) that no memory answers. They have no expected, acceptable or forbidden items and
    expect an empty block. Their `distractors` name the earlier utility turns that a tempting
    probe repeats.
- The lexical tag. `lexical: true` means a word matcher can get the probe right. Paraphrase
  probes (`lexical: false`) are reported, not floored, against the non-semantic fakes, and floored
  for the real models. Content tokens are the lowercase `[a-z0-9]+` runs of a text, minus a
  stopword list. The crude stem over-conflates on purpose ("dislike" and "dislikes", "run" and
  "running"), so a paraphrase is never one inflection away from a keyword hit.
  - A tag is judged against the text the index serves, not the authored text. For a fact that
    is `render_fact`'s sentence for every version of its key, rendered at UTC, plus its subject
    and object names: fact-01's key reads "Wren prefers cortado since 12 January 2026.", not
    "Wren's usual coffee order: a cortado.". A far-off zone can move a timestamp bound to the
    next day and change a date word, so tags are judged at UTC, the zone every fixture timestamp
    is written in.
  - Probes other than noise judge the query against their reference items: the expected items,
    or the forbidden items when nothing is expected (a probe after a forget). `lexical: true`
    means the query shares at least one exact content token with every reference item.
    `lexical: false` means no content token of the query shares even a crude stem with any
    reference item, so only a semantic match finds them.
  - A noise probe is right when its block is empty, so it is judged against every retrievable
    item: facts, notes, episode summaries and conversation turns. `lexical: true` means the query
    shares no content token, and not even a crude stem, with any item, so a word matcher returns
    nothing. `lexical: false` means the query shares at least one content token with some item,
    usually an earlier utility turn ("turn on the kitchen lights"), which tempts a word matcher
    into serving it.
- Shapes. Every probe carries a `shape`; the first 152 probes are all `plain`.
  - A `list_item` probe is a `fact` probe whose `expected` list is exactly the sibling facts of
    one list utterance, every one of them. The siblings overflow a 300-token block, so recall
    measures how much of the list the reader's cap lets through. Its `acceptable` list holds the
    list turn and its episode, plus cycling to work for the workday habits.
  - A `type_scoped` probe is a `fact` probe whose query names an entity type ("which
    organisations", "what gadgets and software"), never one of the objects it expects. Its
    expected facts share one `object_type`. For each subject and relation among them, they are
    every fact of that type that is valid at `now` and not suppressed by an earlier correction.
    Facts of another type under the same relation (Wren's energy supplier among her
    technologies) are distractors.

  There are 12 `list_item` probes, three per list, and 14 `type_scoped` probes, so each new shape
  has at least 12.

  At least 40 percent of probes are tagged lexical false, and at least ten noise probes are
  tagged lexical true, so one miss still clears T9's 0.90 empty_on_noise floor. The fixture test
  recomputes every tag.

## Held-out probes

`heldout_probes.jsonl` is a second probe set over the same corpus, in probes.jsonl's schema, with
ids `ho-<kind>-NN`. The read path's first PR is judged on it: its recall floors are measured
here. Every read-path setting is chosen on the dev set, probes.jsonl, and none on this file, so
a floor measured here says how memory works on probes nothing was tuned on.

A held-out set is spent once it is studied. Scoring a run here and reading the aggregate report
keeps it held out; reading a run probe by probe (which probe missed, which item the block served
instead) spends it, because the next change is then tuned to those probes. After that read, the
file is one more dev set, and the floors need a new held-out set.

The probes were written from the corpus files and the rules on this page, without reading
probes.jsonl. Every probe follows the Probe authoring rules above, and the fixture test runs each
per-probe rule over both files. The file-level rules hold on this file with its own kind set,
`fact`, `note`, `correction` and `noise`, the kinds the first PR gates on: every one of those
kinds is present and no other, at least 40 percent of probes are tagged lexical false, at least
ten noise probes are tagged lexical true, each new shape has at least 12 probes, and each
correction has exactly one probe. The 120-probe total belongs to the dev set alone.

The corpus fixes the targets of a list, correction or type-scoped probe, so such a probe can
differ from its dev twin only in wording. Where a held-out probe and a dev probe have the same
reference set (the expected items, else the forbidden ones), and for every pair of noise probes,
the Jaccard overlap of their queries' content stems (the crude stems above, shared over union) is
at most 0.5. A failure names the pair and the overlap, not the dev query; fix it by rewording the
held-out probe.

Minimums for the `fact` probes:

- at least 69 in all, at least 49 of them tagged lexical false;
- at least 43 `plain`, at least 27 of them lexical false;
- 12 `list_item`, three per list utterance, at least 8 of them lexical false;
- at least 14 `type_scoped`, every one lexical false.

Minimums for the other kinds:

- at least 29 `note` probes, at least 12 of them tagged lexical false;
- one `correction` probe per correction in corrections.jsonl, forbidding exactly its
  suppression set; a `correct` probe expects the replacement fact and a `forget` probe expects
  nothing;
- at least 20 `noise` probes, at least ten of them tagged lexical true.

The pooled recall classes of `PR1_GATES` (`scripts/eval_harness/memory_recall_gate.py`), the
fact, note and correction probes that expect something, hold at least 103 probes, at least 63 of
them tagged lexical false.

## Scoring a run

A run file is JSONL with one record per probe:

```json
{"probe_id": "fact-01", "items": [{"ref": "m1", "kind": "fact", "id": "...", "score": 0.82, "rank": 1}], "tokens": 212, "budget_tokens": 500}
```

`items` is the block in block order. Each item's `rank` is required, an integer of at least 1:
the fetcher's own 1-based `MemoryCandidate.rank` of the injected candidates the item stands for,
the lowest when there are several (a list's members, a note's chunks), so items may share a rank.
A run item without a valid `rank` is refused, and the error names the item. MRR is 1 / the lowest
`rank` among a probe's expected items present in the block, or 0 when none is present. It reads
the fetch rank rather than the block position because the block lays items out by section, facts
first, so a note the fetcher ranked first would otherwise score 1/2 or less behind any fact.
Score it with:

```bash
python scripts/eval_harness/memory_recall.py --run run.jsonl --json-output report.json
```

The report gives recall@block, MRR, precision@block, empty_on_noise, forgotten_leak and budget_ok.
Each is given overall, per probe kind, per lexical tag and per probe shape, with the count of
probes it averages over. The module docstring defines each metric.

## Extending the fixture

1. Edit the JSONL files by hand, keeping every key and the ASCII-only encoding. New ids follow the
   existing patterns: a `fact_key` and an `episode_id` are hashes, computed as above.
2. Adding or changing turns can move episode boundaries. Run the fixture test:

   ```bash
   python -m pytest tests/unit/eval_harness/test_memory_recall_fixture.py -q
   ```

   `test_every_boundary_matches_the_segmentation_rule` prints the boundaries, token sums and close
   reasons the rule gives. Rewrite `episodes.jsonl` to match, including the hashes and the
   summaries. To add turns without moving any existing row, add them as a new conversation more
   than 30 minutes from the turns on either side, in a new session counted from 0, and insert its
   episode rows at the conversation's place; that is how the list conversations went in. Keep new
   turns no later than the last turn, so the history's end stays put, and check that none falls
   inside a correction's suppression window or shares a stem with a lexical-true noise probe.
3. After adding a correction, recompute every probe's eligibility. The correction can suppress
   items that existing probes expect or accept, and its own probe must forbid its whole
   suppression set.
4. Tag each new probe by the lexical rule, then let the test confirm the tag. If a paraphrase is
   rejected, the failure names the shared stem and the item it matched.
5. Keep the minimums of plan task T1:
   - facts: 150 or more, with at least 15 superseded, 5 future and 5 retracted;
   - conversations: 30, spanning 14 days, with at least 300 turns;
   - episodes: at least two `turn_limit` and two `token_limit`;
   - notes: 40 or more;
   - corrections: 10 or more;
   - probes: 120 or more, covering all seven kinds, with at least 40 percent paraphrase.

   Plan task T36 adds:
   - facts: every row typed; at least three list utterances whose sibling texts sum past 300
     tokens, with `RECOMMENDS` and `HAS_HABIT` among them;
   - probes: at least 12 `list_item` and 12 `type_scoped`.
