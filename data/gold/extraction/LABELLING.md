# Labelling guide: dev-list and held-out extraction gold sets (2026-11)

This guide is for labelling `labelling/utterances-2026-11.jsonl`: 40 dev list probes and 200
held-out probes. Two label passes write `labelling/labels-a-2026-11.jsonl` and
`labelling/labels-b-2026-11.jsonl` blind to each other. An adjudicator then merges them into
`dev-lists-2026-11.jsonl` and `heldout-2026-11.jsonl` and records each decision in
`labelling/adjudication-2026-11.md`. `tests/unit/eval_harness/test_gold_sets.py` checks the
utterance file and every label file.

The conventions below are the ones the 60-probe set (`data/ingest/extraction-gold-2026-06-14.jsonl`)
already follows. Each one cites where it comes from: the 60-probe changelog
(`data/ingest/extraction-gold-2026-06-14-changelog.md`), the production prompt
(`backend/knowledge/extraction/prompts.py`), ontology v1.4.0
(`backend/knowledge/ontologies/v1_0_0.py`, `version="1.4.0"` at line 1976) or the 60-probe design
note (`scripts/eval_harness/extraction_probe_set_design.md`). Where a new set needs a rule the
60 probes never had to make, the rule is marked "New rule", with its reason.

## The held-out rule

**Whoever edits an extraction prompt, the preprocessor, the splitter or the extraction settings
never opens `heldout-*` or `labelling/`.**

The held-out set measures extraction only while nobody has tuned against it. Tune against the
60-probe set and `dev-lists-2026-11.jsonl`. Authors and labellers of probes edit no extraction
code, so they may read both directories. Anyone who has read them does not then take on
extraction work for this goal.

## Synthetic-data rule

- The user is fictional and is never named. Utterances say "I".
- Organisations are fictional (`Driftmere Analytics`, `Kelderwick Rail`). The 60-probe set
  uses some real company names; these sets use none, and the test fails any utterance that names
  an Organization entity of the 60-probe set.
- People get first names only (`Marta`, `Teodor`). A person the utterance does not name gets no
  entity (see Third parties below).
- Nothing in any utterance, label or note is about Raj or any other real private person.
- Real technologies, real public documents (books, papers, RFCs) and real places are allowed,
  as in the 60-probe set.

## Naming rule

No file, commit, pull request text, probe, gold label or prompt names another AI assistant,
real or fictional, or the people behind one. The rule covers a word used as an assistant's
name; the same word in its ordinary sense is fine, so Friday the weekday stays. This guide
lists no names, on purpose.

## Row schema

Each label row follows the 60-probe schema exactly, so `iter_gold_probes`
(`scripts/eval_harness/score_extraction_run.py:160`) loads it unchanged:

```json
{"utterance": "...", "tag": "ho-001-list-dash-current-setup",
 "expected_entities": [{"id": "user", "type": "User"}],
 "expected_relationships": [{"source": "user", "source_type": "User", "predicate": "USES",
   "target": "zed", "target_type": "Technology"}]}
```

- `tag` and `utterance` are copied from the utterance file unchanged. The test fails a tag
  that is not in the utterance file, a repeated tag, or a changed utterance.
- `id`, `source` and `target` are canonical: lowercase, hyphenated, letters, digits and hyphens
  only (`score_extraction_run.py:65-77`; prompt Rule 2, `prompts.py:50`). Versions collapse to
  the base name (Rule 3, `prompts.py:51`).
- Types and predicates are the extractable v1.4.0 sets only: `EXTRACTABLE_NODE_TYPES`
  (`v1_0_0.py:1877-1879`, 21 types) and `EXTRACTABLE_RELATIONSHIP_TYPES`
  (`v1_0_0.py:1914-1916`, 39 predicates). The test derives both sets from the module. `Topic`
  and `Milestone` are retired (changelog lines 13-16): use `Concept` and `Event`.
- Every edge's `source` and `target` are among the probe's entities, `source_type` and
  `target_type` equal those entities' types, and the type pair is allowed for the predicate
  under the production validator's constraints (`backend/knowledge/extraction/validator.py:40-65`).
  The test checks all three. An edge the validator would drop is not gold.
- `assertion_kind` is written only for `cease` and `retract`. Absent means `assert`
  (`score_extraction_run.py:185`; design note line 27).
- A probe with nothing to extract has empty `expected_entities` and `expected_relationships`
  (prompt Rule 9, `prompts.py:57`; design note line 28).

## Conventions from the 60-probe set

### The user anchor

When the user is the subject of a fact, the probe carries the entity `{"id": "user", "type":
"User"}` and the user-sourced anchor edge for that fact (prompt Rule 1, `prompts.py:49`; design
note lines 95-106; changelog lines 32-33, which restore the `user` entity and the
`user -[KNOWS_PERSON]-> anjali` anchor on ext-29). Edges between other entities sit beside the
anchor, not in place of it (`rust -[USED_FOR]-> backend-work` in ext-01).

The `user` entity appears only when some edge uses it. ext-05, ext-12, ext-24 and ext-41 state
no user fact and carry no `user` entity. The `User` type is a singleton
(`v1_0_0.py:421-425`), so the id is always `user`.

MIST-scope facts use `{"id": "mist-identity", "type": "MistIdentity"}` as the subject, with
`USES`, `IMPLEMENTED_WITH`, `MIST_HAS_CAPABILITY`, `MIST_HAS_TRAIT` or `MIST_HAS_PREFERENCE`
(prompt lines 34 and 43; Rule 1, `prompts.py:49`; Examples 3, 4 and 6, `prompts.py:87-109`).
When one utterance states a MIST fact and a user fact, each fact keeps its own subject: never
move a MIST fact onto `user` or a user fact onto `mist-identity`. Naming MIST does not make it
the subject (`ho-174`).

### Durable third-party facts

When the user states a durable fact about another person, label both the anchor
`user -[KNOWS_PERSON]-> <person>` and the person's own fact with the person as source (prompt
Rule 19, `prompts.py:67`; changelog line 17 and lines 32-35, which add `anjali -[WORKS_AT]-> biotech`
to ext-29). A recommendation is such a fact: `<person> -[RECOMMENDS]-> <thing>` plus the
anchor (prompt Rule 17, `prompts.py:65`; ext-08, ext-31 and ext-32, audited at changelog
lines 18, 20 and 21).

An incidental encounter is not a durable fact. "I had coffee with my old manager Devin" gets
the anchor only, with no Event for the coffee (changelog lines 19 and 38).

Third-party facts are never moved onto the user (prompt line 35). A person the user does not
claim to know (a name mentioned in passing) gets no `KNOWS_PERSON` edge. New rule: a person the
utterance does not name ("my wife", "my brother") gets no entity, and that person's facts
are dropped. The 60-probe set has no such case. The rule follows prompt Example 7
(`prompts.py:111-115`), where "my coworker" produces no Person entity, and it keeps the ids
stable across the two label passes.

### Event and metric ids

An Event id names the event at the grain `OCCURRED_ON` works on, not the artifact it concerns:
`mobile-app-launch`, not `mobile-app`; `house-closing`, not `house` or `closing-on-house`
(changelog lines 24-25 and 44-50). Five of the six dated Event probes in the 60 use the bare
event noun (`team-offsite`, `birthday-party`, `mobile-app-launch`, `house-closing`,
`product-demo`); ext-23's `wedding-2026-05-30` is the one date-suffixed id (changelog line 26).
New rule: use the bare event noun, as the five do, so the two label passes agree.

A milestone-class event (shipped, launched, completed, promoted) is an `Event` anchored to its
`Date` by `OCCURRED_ON`, with no user `EXPERIENCED` edge: ext-21 and ext-22, and prompt
Rule 11 and Example 9 (`prompts.py:59`, `:123-127`). An event stated with no participant gets
`OCCURRED_ON` only (ext-60). An event the user took part in
(a meeting, a trip, a talk) also gets `user -[EXPERIENCED]-> <event>`: ext-04, ext-20 and
ext-23, and Example 12 (`prompts.py:141-145`). A decision the user made is `user -[DECIDED]->
<event>` (Rule 11). `DECIDED` takes only a `User` source, so a decision by somebody else gets
no `DECIDED` edge. The Event type's `event_type` values are at `v1_0_0.py:621-640`. Gold does
not record entity properties.

A Metric id puts the value first: `<value>-<unit>[-<quantity>]`, for example
`12000-requests-per-second` and `87-percent-coverage` (changelog lines 22-23;
`backend/knowledge/extraction/canonical_id.py:18-22`). Write the value as the utterance states
it, decimal point included (`3.1-gb-peak-memory`). The loader strips the point when it
canonicalises the id, the same way the normaliser's slug does (`canonical_id.py:10-15`). The
metric attaches through `HAS_METRIC` (`v1_0_0.py:1519`). That predicate's allowed sources
include User, Project, Technology, Skill, Concept and Goal but not Organization or Event, so a
metric about an organisation or an event gets no edge.

### The Abstraction fallback

An abstract noun takes the child type whose test clearly fits: Concept, Skill, Pattern,
Strategy, Mechanism, Convention or DataStructure (the abstract-type tests, `prompts.py:19-27`;
Rule 14, `prompts.py:62`). Only when no child's test clearly wins does it take `Abstraction`:
never pick between two siblings that fit equally well (Rule 20, `prompts.py:68`; Example 26,
`prompts.py:225-229`; the type at `v1_0_0.py:833-842`). Abstraction became the supertype of
the seven children in v1.4.0, the same taxonomy revision that retired Topic and Milestone
(changelog lines 13-16). The 60-probe set has no `Abstraction` entity; the rule above is the
prompt's and the ontology's. The held-out fill probes `ho-176` to `ho-178` were written to need
the fallback.

### Date entities

A `Date` entity exists only when an edge targets it: `<event> -[OCCURRED_ON]-> <date>`, or
`<event> -[PRECEDED_BY]-> <date>` (prompt Rule 16, `prompts.py:64`; Rule 13, `prompts.py:61`;
design note lines 84-93). Its id is the ISO date, `2026-05-20` (`v1_0_0.py:735-747`). A date
that only scopes a stative fact goes on the edge as `valid_from` or `valid_to`, with no Date
entity: ext-09, ext-33 to ext-36, ext-58 and ext-59.

`valid_from` and `valid_to` are ISO prefixes at the precision the utterance states (`2026`,
`2026-05`, `2026-05-20`). The scorer matches them by prefix (`score_extraction_run.py:434-440`;
design note line 26). A cessation with a stated stop date carries `valid_to` (ext-43).

New rule: a relative date ("yesterday", "two weeks ago", "last month", "next Friday") is
resolved as prompt Rule 5 (`prompts.py:53`) tells the model to resolve it, against the reference
date. The service gauntlet sends one pinned reference date with every probe, `2026-06-13`, a
Saturday (`MIST_FIXED_CLOCK_PIN`, `scripts/model_bench/probes/extraction.py:57`, the default of
its `--recorded-at`), so the gold is deterministic while a run keeps that default; a run with
another `--recorded-at` cannot be scored on these probes. Label at the precision the expression
states: "yesterday" is `2026-06-12`, "two weeks ago" is `2026-05-30`, "last month" is `2026-05`,
"last year" is `2025`. "Last <weekday>" is the latest such day before 2026-06-13, and "next
<weekday>" the first one after it. The resolved value goes where an absolute date would: on the
edge as `valid_from` or `valid_to` for a stative fact, or as a `Date` entity with `OCCURRED_ON`
for an event, but only when the expression resolves to a single day, because a Date id is a full
`YYYY-MM-DD` (`v1_0_0.py:735-747`); an event dated only to a month or a year gets no Date
entity. An expression too vague to resolve ("recently", "a while back") gets no date. New probes
still prefer absolute dates (design note line 26).

### Assertion kinds

`cease` means the fact stopped being true ("stopped", "quit", "left", "gave up on", "don't
... anymore"). `retract` means it was never true ("I never actually", "I misspoke", "scratch
that", "I was wrong about"). A self-correction is a retraction, not a directive (prompt Rule 6,
`prompts.py:54`; ext-43 to ext-52). A replacement in one turn carries both edges: `cease` on
the old target and an assertion on the new one (ext-53, ext-54). A negated preference is an
assertion: "I don't like X" and "I don't like X anymore" are both `DISLIKES` (Rule 6; ext-15).
A plain negation of something never stated ("I've never written PHP") is neither cease nor
retract and gets no edge.

### No-fact probes

Small talk, utility requests, questions, hypotheticals, speculation and directives are empty
(prompt Rules 8 to 10, `prompts.py:56-58`; ext-10, ext-11 and ext-37 to ext-40). A directive
is empty even when it says "I" (Rule 10).

### Habits and incidental scope

`HAS_HABIT` needs a stated cadence ("every Wednesday", "twice a day"). A continuous activity
with only a start point takes the matching stative predicate with `valid_from` (prompt Rule 17,
`prompts.py:65`; ext-55 to ext-58). A trailing phrase that only scopes the fact is context,
not a structural edge. "Recommended X for the new service" adds no `STRATEGY_FOR`, but "I use X
for Y" is itself the fact and gets `USED_FOR` (Rule 18, `prompts.py:66`; ext-01, ext-31).

## List-shaped probes (new)

The 60-probe set has no list-shaped utterance. These rules make the two label passes agree on
one:

- Label each item as if its line were a sentence that completes the lead-in or the heading
  above it. "My current setup:" then "- Zed for editing" reads "I use Zed for editing".
- A lead-in line, a heading or a parent bullet sets the subject or the predicate for the items
  under it. It is not an entity unless it names one, for example an organisation or a project.
- An item that refers back to an earlier one ("the second one", "the former", "both of them")
  resolves to that earlier item's entity and uses the same id. An item that is only an option
  someone listed, and that nobody chose, gets no edge.
- A checked box states a fact that is done. An unchecked box is a plan and gets no edge,
  except under a lead-in that names goals, where it is `HAS_GOAL`.
- In a table, the header row names the columns, and each row is one fact about its first
  column.
- A list with nothing durable (an errand list) is empty.

## Categories

The tag carries the category, so reports count categories from the tag alone. Dev tags are
`dvl-NN-<slug>` (01 to 40) and held-out tags `ho-NNN-<slug>` (001 to 200). The slug's first
word is the category:

| Prefix | Set | Count | What it tests |
|---|---|---|---|
| `list-` | dev-lists | 40 | lists; 10 start `list-long-` (10 or more items, one with 25 or more), 8 start `list-ref-` (a later line refers back) |
| `list-` | heldout | 40 | bullets (`-`, `*`, `+`, `•`), numbered (`1.`, `1)`, `a.`), nesting, headings, a pipe table, a lead-in, checkboxes, prose mixed with a list |
| `multi-` | heldout | 30 | several facts in one sentence |
| `neg-` | heldout | 25 | negation, cessation, retraction |
| `temporal-` | heldout | 25 | dates, valid-time, order of events |
| `coref-` | heldout | 15 | a pronoun or reference inside one utterance |
| `nofact-` | heldout | 25 | small talk, utility requests, directives |
| `scope-` | heldout | 15 | MIST-scope facts beside user facts |
| `fill-` | heldout | 25 | single facts that fill type and predicate coverage |

Each utterance row's `intent` (`types: ...; predicates: ...; note: ...`) records what the author
meant to cover. It is a coverage aid, not a label: label from the utterance and this guide. The
held-out intents name every extractable type at least 3 times and every extractable predicate
at least twice. T8 checks the same floor on the adjudicated labels.

## Extending the sets

- Add a probe as a new row with the next free number in its set and a slug that starts with its
  category. Then update the counts in `test_gold_sets.py`. Never renumber or retag an existing
  probe: the tag is how labels, adjudication and reports join.
- Never edit a labelled utterance. Retire it in the adjudication record and add a new probe.
- Never move a probe between the dev and held-out sets. A held-out probe that someone tuned on
  is retired, not moved.
- New held-out probes are written by someone who edits no extraction code, and they meet the
  synthetic-data rule and the naming rule. They must not repeat, or reword, a 60-probe or dev
  utterance, and they must not reuse an entity name from the production prompt's examples. The
  test checks exact repeats after case and whitespace folding, near copies by word overlap, and
  the prompt's example entity ids and names.
- Label new probes with two blind passes and adjudication, as for the first batch.
