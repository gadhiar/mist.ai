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
- People get first names only (`Marta`, `Teodor`). A relative the utterance names only by the
  relation ("my grandfather") is labelled by that relation (N16, under Durable third-party
  facts below).
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
- An id is written as the loader leaves it, so the file row and the loaded probe are the same
  (N9); written in loader form, the test can compare the file with what the loader returns.
  A Metric id keeps its decimal point as a hyphen: `3-1-gb-peak-memory`, not
  `31-gb-peak-memory` or `3.1-gb-peak-memory`, so 3.1 GB and 31 GB do not share an id (plan v11
  D16; the Metric paragraph under Event and metric ids). This replaces the 2026-11
  adjudication's dropped decimal (dvl-39, ho-015).
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
- Three keys are optional, and `iter_gold_probes` reads them: an entity's `alternatives`
  (other ids gold accepts, under N1's last sentence), an edge's `predicate_alternatives`
  (other predicates gold accepts, only where this guide names two predicates for one shape),
  and an edge's `beyond_prompt`, written only as `true`, for an edge a labelling rule beyond
  the prompt added. Each alternative is a canonical id distinct from every id in its probe.
- A probe with nothing to extract has empty `expected_entities` and `expected_relationships`
  (prompt Rule 9, `prompts.py:57`; design note line 28).

## Conventions from the 60-probe set

The rules marked N1 to N10 below came out of the 2026-11 adjudication
(`labelling/adjudication-2026-11.md`, "Rules to add"), each with the precedent that settled it.
N11 to N16 came out of the H8 check of 20 held-out probes against their gold (same file, "H8
marks"), each from the mark that asked for it. A rule marked "beyond the prompt" asks for a fact
the production prompt does not tell the model to extract; the gold labels it anyway, so the
score shows what extraction misses. In dev-lists, the six edges H8 added under such rules carry
`"beyond_prompt": true` (Relabel log).

### Ids

- **N1 Id surface.** An id is the utterance's own phrase for the thing, lowercased and
  hyphenated, with leading articles and possessive pronouns dropped and a plural kept as
  written: `regular-expressions` (`ext-28`), `in-memory-data-store` (`ext-41`), `backend-work`
  (`ext-01`). Gold invents no id (extraction contract ADR, Decision 3): an id differs from the
  utterance's phrase only where a convention with a carrier (the prompt or a rail) builds it,
  as Rule 1's `user`, the ISO Date id, the Metric rail, N2, N4, N14, N15, N16 and the Event
  rule do. An id that is a contiguous run of the utterance's words naming the entity is the
  utterance's own phrase, and a labeller neither lengthens nor shortens it.
  *Retired:* N1's habit clause, "a habit id names the activity", is retired (A30, Raj's
  Ruling 22, 2026-11-10). No prompt rule or rail carried it, and a carrier would have to turn
  verbs into activity nouns, which no deterministic rail can check. A habit id is the
  utterance's own phrase: "swim on Tuesdays" is `swim`, "I journal every night" is `journal`
  (`ext-55`, relabelled from `journaling`), "a long run every Sunday" is `long-run` (`ext-57`).
  A verb-form habit id and a later noun mention (`swim`, `swimming`) stay two nodes until a
  graph resolution step joins them. Where two ids both fit N1 and neither is the utterance's
  own phrase, gold lists both: the entity keeps one as
  `id` and the other in `alternatives` (Decision 3; this replaced the adjudication's rule that
  the base pass's stands). No dev-lists entity is such a case.
- **N15 An activity keeps its object.** An id for an activity, a capability or a habit keeps the
  object the utterance gives it, possessives dropped: "MIST can search my notes by meaning" is
  `search-notes-by-meaning`, not `search-by-meaning`, because the notes are part of the
  capability. From the H8 mark on ho-165. It asks for no new fact, only the id's surface. The
  60-probe `ext-58` wrote "mentoring junior engineers" as `mentoring`; it is relabelled to
  `mentoring-junior-engineers` under this rule (Relabel log).
- **N4 Name modifiers.** A name that only modifies another name (the Lisbon summit, the Corvane
  outage drills, the Wrenfield Bakery ordering site) gets no entity of its own, unless a list's
  lead-in carries it (List-shaped probes; dvl-35, ho-038). From the agreed rows ho-057, ho-069,
  ho-095 and ho-188. N4 still covers a team name that only modifies another name and places no
  one in it; a team that places someone is N17's.

**N6 Type by its test.** A type is chosen by its type test, never to keep an edge. An edge the
validator then refuses is dropped (Row schema). An entity left with no edge stays as an
entity-only label when the utterance states something about it, as prompt Example 7 keeps
`rust` with no relationship; a Metric left with no edge is dropped with it, as both passes did
in dvl-24 and ho-131. Settled on ho-052, ho-129 and ho-134.

### Assertion kinds

`cease` means the fact stopped being true ("stopped", "quit", "left", "gave up on", "don't
... anymore"). `retract` means it was never true ("I never actually", "I misspoke", "scratch
that", "I was wrong about"). A self-correction is a retraction, not a directive (prompt Rule 6,
`prompts.py:54`; ext-43 to ext-52). A replacement in one turn carries both edges: `cease` on
the old target and an assertion on the new one (ext-53, ext-54). A negated preference is an
assertion: "I don't like X" is `DISLIKES` (Rule 6; ext-15). A plain negation of something never
stated ("I've never written PHP") is neither cease nor retract and gets no edge.

### No-fact probes

Small talk, utility requests, questions, hypotheticals, speculation and directives are empty
(prompt Rules 8 to 10, `prompts.py:56-58`; ext-10, ext-11 and ext-37 to ext-40). A directive
is empty even when it says "I" (Rule 10).

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

## Relabel log

- **2026-11-09, dev-lists, under the extraction contract ADR (T30).**
  - The six edges H8 added beyond the prompt carry `"beyond_prompt": true`: `fish IS_A shell`
    and `helix IS_A editor` (dvl-01), `kakoune IS_A editor` (dvl-17), the two `RELATED_TO`
    edges from the planning event (dvl-30) and `user KNOWS_PERSON sister` (dvl-34). They are
    the edges the pre-H8 gold (`4f5ceed`) lacks, flagged before any id changed.
  - The 19 Events follow the Event rule above; 18 ids changed. dvl-15: `dentist-appointment`
    to `dentist-2026-11-18`, `thesis-defense` to `ilse-thesis-defense-2026-12-03`,
    `team-retro` to `team-retro-2026-11-21`. dvl-18: `lease-signing` to
    `signed-lease-in-ghent`. dvl-30: `planning` to `planning-2026-06-13`, `docs-site-move` to
    `move-docs-site-to-astro`, `feature-freeze` to `freeze-features-2026-11-28`. dvl-37:
    `flight-to-lisbon` to `fly-into-lisbon-2026-12-10`, `tram-28-ride` to
    `tram-28-2026-12-11`, `castle-visit` to `castle-2026-12-11`, `sintra-day-trip` to
    `day-trip-to-sintra-2026-12-12`, `birthday-dinner` to `marta-birthday-dinner-2026-12-13`,
    `cafe-workday` to `work-from-cafe-2026-12-14`, `surf-lesson` to
    `surf-lesson-in-cascais-2026-12-15`, `train-to-porto` to `train-to-porto-2026-12-16`,
    `port-cellar-tour` to `port-cellar-tour-2026-12-17`, `serralves-museum-visit` to
    `serralves-museum-2026-12-18`, `flight-home` to `fly-home-2026-12-19`; `lisbon-trip`
    stays. Every edge is kept with its endpoints renamed, and the retrieval queries
    (`data/gold/retrieval/queries-2026-11.jsonl`) name the new ids.
  - dvl-08's `swimming` and `calling-parents` gained `alternatives` (N1). No edge has
    `predicate_alternatives`.
  - dvl-38's `leader-election MECHANISM_OF scheduler` was not added: by its type test, and by
    prompt Rule 14, which names it, the scheduler is a `Mechanism`, and `MECHANISM_OF` admits no
    `Mechanism` target (`v1_0_0.py:1587-1588`). It goes to
    the ontology goal.
- **2026-11-09, dev-lists, gold invents no id (T30, plan v10).** Every other id that differed
  from the utterance's phrase and that no convention with a carrier builds now is that phrase,
  with its edges and retrieval query facts renamed. N1's habit clause is suspended (N1).
  dvl-08: `swimming` to `swim`, `calling-parents` to `call-parents`, both dropping the
  `alternatives` they had. dvl-23: `tests-under-a-minute` to `tests-that-run-in-under-a-minute`.
  dvl-26: `bill-splitting-cli` to `cli-for-splitting-bills`, `discord-bot` to
  `discord-bot-for-book-club`. dvl-34: `swimming` to `swim`, `calling-sister` to
  `call-with-sister`, `watering-plants` to `watering-the-plants`. dvl-40: `dark-mode` to
  `dark-mode-everywhere`. A trailing phrase stays out of an id only where it is a fact of its
  own (dvl-19's parser clauses, labelled as `OPERATES_ON`), a scope of a named thing (`sqlite`
  "for small apps", Rule 18) or a name modifier (N4).
- **2026-11-09, dev-lists, an id is neither lengthened nor shortened (T30, lead ruling).** An id
  that is a contiguous run of the utterance's words naming the entity is the utterance's own
  phrase (N1). This replaces the previous entry's trailing-phrase reading for ids that were
  already such a run: dvl-26 `discord-bot-for-book-club` back to `discord-bot`, dvl-40
  `dark-mode-everywhere` back to `dark-mode`, with their edges and the retrieval fact
  (`rq-type-scoped-07`).
- **2026-11-10, the 60 and dev-lists, named teams (N17; A32(a), Ruling 21; plan v11 T35).**
  ext-18 gains the `Organization` `payments-team` and `user MEMBER_OF payments-team`, beside
  `user WORKS_AT stripe`. ext-33's `user WORKS_AT platform-team` becomes
  `user MEMBER_OF platform-team`, `valid_from` 2026-05 kept; the retrieval fact of
  `rq-direct-15` follows it. dvl-26 gains the `Organization` `book-club` and
  `user MEMBER_OF book-club` ("a Discord bot for my book club": the speaker's possessive names
  the club), one more dev-lists gold edge. N4 no longer cites ext-18.
- **2026-11-10, dev-lists and the 60, under the id rails (plan v11 T35).**
  - dvl-39's four decimal Metric ids keep the decimal (D16): `02-percent-error-rate` to
    `0-2-percent-error-rate`, `24-seconds-cold-start` to `2-4-seconds-cold-start`,
    `31-gb-peak-memory` to `3-1-gb-peak-memory`, `380-euros-per-hour` to `3-8-euros-per-hour`,
    with their `HAS_METRIC` edges. No other dev-lists id changed.
  - ext-13 gains the `Concept` `editor` and `neovim IS_A editor` (N14). ext-58's `mentoring`
    becomes `mentoring-junior-engineers` (N15). N1's habit clause is retired (A30, Ruling 22),
    so ext-55's `journaling` becomes `journal` and ext-56's `pr-review` becomes `review-prs`
    ("I review PRs every Friday"). The retrieval facts of `rq-paraphrase-06` and
    `rq-type-scoped-04` name the new ids, and `data/golden-log/golden-log.jsonl` is regenerated
    from the relabelled 60.
