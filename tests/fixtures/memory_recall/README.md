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
written out turn by turn.

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
| `relation` | an ontology relationship type that is extractable and directional |
| `object_id`, `object_name` | canonical id and display name |
| `text` | the sentence the fact is indexed under |
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

Item kinds are `fact` (id is the `fact_key`), `episode` (the `episode_id`), `exchange` (the turn's
`event_id`) and `note` (the `path`).

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

  At least 40 percent of probes are tagged lexical false, and at least ten noise probes are
  tagged lexical true, so one miss still clears T9's 0.90 empty_on_noise floor. The fixture test
  recomputes every tag.

## Scoring a run

A run file is JSONL with one record per probe:

```json
{"probe_id": "fact-01", "items": [{"ref": "m1", "kind": "fact", "id": "...", "score": 0.82}], "tokens": 212, "budget_tokens": 500}
```

`items` is the block in block order. Score it with:

```bash
python scripts/eval_harness/memory_recall.py --run run.jsonl --json-output report.json
```

The report gives recall@block, MRR, precision@block, empty_on_noise, forgotten_leak and budget_ok.
Each is given overall, per probe kind and per lexical tag, with the count of probes it averages
over. The module docstring defines each metric.

## Extending the fixture

1. Edit the JSONL files by hand, keeping every key and the ASCII-only encoding. New ids follow the
   existing patterns: a `fact_key` and an `episode_id` are hashes, computed as above.
2. Adding or changing turns can move episode boundaries. Run the fixture test:

   ```bash
   python -m pytest tests/unit/eval_harness/test_memory_recall_fixture.py -q
   ```

   `test_every_boundary_matches_the_segmentation_rule` prints the boundaries, token sums and close
   reasons the rule gives. Rewrite `episodes.jsonl` to match, including the hashes and the
   summaries.
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
