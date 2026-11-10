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
- An id is written as `canonical_id` leaves it, so the file row and the loaded probe are the same
  (N9). A decimal point is dropped: `31-gb-peak-memory`, not `3.1-gb-peak-memory`. The loader
  and the normaliser's slug strip it anyway (`canonical_id.py:10-15`), so the loaded gold is
  the same either way; written in loader form, the test can compare the file with what the
  loader returns. Settled in the 2026-11 adjudication (dvl-39, ho-015).
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
  (`ext-01`). A habit id names the activity, as `journaling` and `long-run` (`ext-55`,
  `ext-57`). Where two ids both fit this and neither is the utterance's own phrase, gold
  lists both: the entity keeps one as `id` and the other in `alternatives` (extraction
  contract ADR, Decision 3; this replaced the adjudication's rule that the base pass's stands).
  dvl-08 is the case: "swim on Tuesdays" and "call my parents" name habits only by a verb, so
  `swimming` accepts `swim` and `calling-parents` accepts `call-parents`.
- **N15 An activity keeps its object.** An id for an activity, a capability or a habit keeps the
  object the utterance gives it, possessives dropped: "MIST can search my notes by meaning" is
  `search-notes-by-meaning`, not `search-by-meaning`, because the notes are part of the
  capability. From the H8 mark on ho-165. It asks for no new fact, only the id's surface. The
  60-probe `ext-58` writes "mentoring junior engineers" as `mentoring`, so it does not follow
  this rule; that file stays unchanged.
- **N2 Goal ids.** A Goal id is the goal's verb phrase as the utterance words it, articles and
  possessive pronouns dropped: `get-pilots-license`, `cut-screen-time`, `pay-off-mortgage`. The
  60-probe set has no Goal; this follows the goal ids both passes agreed on (`ship-saltmere` in
  dvl-09, `learn-to-sail` in ho-091). The H8 check marked ho-004's goal ids good.
- **N4 Name modifiers.** A name that only modifies another name (the Lisbon summit, the Corvane
  outage drills, the Wrenfield Bakery ordering site) gets no entity of its own, unless a list's
  lead-in carries it (List-shaped probes; dvl-35, ho-038). From `ext-18`, where "on the payments
  team" adds no entity, and the agreed rows ho-057, ho-069, ho-095 and ho-188.

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

**N8 The team's tools.** A statement of which tools the user's team runs ("How our release
goes", "our data stack", "we retired our self-hosted GitLab") puts `USES` on the user. A
possessive that only names a thing ("Our API handles ...", `ext-24`) does not. From the agreed
rows ho-073 and ho-094 and the `USES` edges both passes gave ho-006.

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
claim to know (a name mentioned in passing) gets no `KNOWS_PERSON` edge.

**N16 A relative named by the relation.** A single person the user names only by their relation
("my grandfather", "my wife", "my sister") is a `Person` whose id is the relation noun
(`grandfather`, `wife`), with the anchor `user -[KNOWS_PERSON]-> <relation>` and that person's
durable facts, as for a named person ("my wife's employer" is `wife -[WORKS_AT]-> <org>`). A
group ("my parents") gets no entity, since a `Person` is one person, and neither does someone
the utterance relates to another person ("her sister") or names without a possessive ("the
baby"). From the H8 mark on ho-185, which asked for the grandfather the user's knowledge came
from; how the knowledge came from him has no predicate (adjudication, "H8 marks", class B).
This replaces the first batch's rule that an unnamed person gets no entity. Beyond the prompt:
Example 7 (`prompts.py:111-115`) gives "my coworker" no Person entity, and Rule 19's example
names the person. The 60-probe set has no unnamed relative.

**N7 Third-party events.** A named person's own event that is not milestone-class (Ilse's
thesis defense, Marta's birthday dinner, Imre's retirement) is `<person> EXPERIENCED <event>`,
beside the anchor. From prompt Rule 19 (the third party's own fact, with the person as source)
and Rule 11 (life events pair with `EXPERIENCED`); settled on dvl-15, dvl-37 and ho-121.

### Event and metric ids

An Event id follows the Event rule of the extraction contract ADR ("The extraction contract:
gold labels only what the prompt or a rail carries", adopted by Raj on 2026-11-09 with Event
id rule E-b; Decision 4):

- The id is the utterance's own phrase for the event, with articles and possessive pronouns
  dropped (N1): "work from a cafe" is `work-from-cafe`.
- A named owner is kept, its possessive `'s` dropped: "Ilse's thesis defense" is
  `ilse-thesis-defense`.
- A place is kept as the utterance words it: "surf lesson in Cascais" is
  `surf-lesson-in-cascais`.
- The id holds no date word. The date words are a closed list: a month or weekday name or its
  abbreviation, a day or year number, an ordinal day (`30th`), an ISO date, `today`,
  `tonight`, `tomorrow`, `yesterday`, and `this`, `next` or `last` before `week`, `weekend`,
  `month`, `year` or a weekday. A number that names the event itself is not a date word
  (`sprint-12`, `tram-28`).
- When the probe dates the event to exactly one day by its `OCCURRED_ON` edge, that date is
  appended as `-YYYY-MM-DD`: "dentist on 2026-11-18" is `dentist-2026-11-18`, and "today's
  planning" is `planning-2026-06-13`. An event with no single day keeps the bare phrase
  ("Lisbon trip, December" is `lisbon-trip`).

The date keeps two occurrences of one event on different days apart, since a wrong merge is
worse than a missed one, and the owner keeps one person's party apart from another's (ADR,
Rationale). Under the contract the prompt carries the owner and no-date-word clauses and a
normalizer rail appends the date (v8). This replaces the earlier rule of the bare event noun.
The 60-probe set and held-out keep their gold Event ids until their own relabel; the scorer's
event-date alignment bridges a dated produced id to them. `test_gold_sets.py` checks the rule
on every dev-lists Event.

A milestone-class event (shipped, launched, completed, promoted) is an `Event` anchored to its
`Date` by `OCCURRED_ON`, with no user `EXPERIENCED` edge: ext-21 and ext-22, and prompt
Rule 11 and Example 9 (`prompts.py:59`, `:123-127`). An event stated with no participant gets
`OCCURRED_ON` only (ext-60). An event the user took part in
(a meeting, a trip, a talk) also gets `user -[EXPERIENCED]-> <event>`: ext-04, ext-20 and
ext-23, and Example 12 (`prompts.py:141-145`). A decision the user made is `user -[DECIDED]->
<event>` (Rule 11). `DECIDED` takes only a `User` source, so a decision by somebody else gets
no `DECIDED` edge. The Event type's `event_type` values are at `v1_0_0.py:621-640`. Gold does
not record entity properties.

- **N10 Personal milestones.** Signing a lease is milestone-class, like closing on a house
  (`ext-22`): an Event with no user `EXPERIENCED` edge. Settled on dvl-18 and ho-111.
- **N3 Decisions.** A decision the user's team makes is `user DECIDED <decision event>`, and the
  decision event is linked to the thing chosen by `RELATED_TO` when no canonical predicate links
  an Event to it. The decision does not make its outcome a current fact (no `USES` for a tool
  not yet adopted, no cease for one not yet retired). A decision by someone else gets the event
  and the link but no `DECIDED`, and its owner is an entity even when the utterance names it
  only by role ("the board" is an `Organization`); the owner has no edge to the decision, since
  `DECIDED` takes only a `User` source (`v1_0_0.py:1321-1327`). From prompt Rule 11
  ("decisions (paired with `DECIDED`)") and Rule 15, which reserves `RELATED_TO` for when no
  canonical shape fits; settled on ho-007, ho-033 and dvl-30. The owner as an entity is from the
  H8 mark on ho-116.
- **N5 Stated places and organisations.** A place or organisation the utterance relates to an
  event with a preposition ("in Bruges", "at Marrowby Press", "to Porto") is an entity, linked to
  the event by `RELATED_TO`. From the agreed rows dvl-18, ho-013 and ho-027; the 60-probe set
  has no stated place.
- **N13 Topic links.** An Event the probe labels links by `RELATED_TO` to what the utterance
  says it concerned: the subject a meeting discussed or decided on, the decisions it made, and
  the object of a decision. Something chosen for a subject links to that subject, not to the
  event. "At the steering meeting the board chose Bergen for the new data centre" is
  `steering-meeting RELATED_TO new-data-centre` and `new-data-centre RELATED_TO bergen`, with no
  edge from the meeting to Bergen. An event the utterance only implies (a review where options
  "came up") is not added for this rule, and an option nobody chose still gets no edge
  (List-shaped probes).
  From the H8 mark on ho-116. Beyond the prompt: no rule tells the model to link an event to
  its topic. The 60-probe set has no event with a stated topic.

A Metric id puts the value first: `<value>-<unit>[-<quantity>]`, for example
`12000-requests-per-second` and `87-percent-coverage` (changelog lines 22-23;
`backend/knowledge/extraction/canonical_id.py:18-22`). Write the value as the utterance states
it, in loader form (N9): `31-gb-peak-memory` for "3.1 GB peak memory", with the decimal point
dropped as `canonical_id` drops it (`canonical_id.py:10-15`). The
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

**N6 Type by its test.** A type is chosen by its type test, never to keep an edge. An edge the
validator then refuses is dropped (Row schema). An entity left with no edge stays as an
entity-only label when the utterance states something about it, as prompt Example 7 keeps
`rust` with no relationship; a Metric left with no edge is dropped with it, as both passes did
in dvl-24 and ho-131. Settled on ho-052, ho-129 and ho-134.

**N14 Kind nouns.** When the utterance says what kind of thing a named entity is, beyond its
entity type, the kind is a `Concept` and the entity gets `IS_A` to it: "a Pinecil soldering
iron" is `pinecil IS_A soldering-iron`, "WezTerm as the terminal" is `wezterm IS_A terminal`, a
heading "Shells" over "zsh" is `zsh IS_A shell`, "a covers band called The Late Trains" is
`late-trains IS_A covers-band`. This is the shape of `ext-12` and `ext-41` ("X is a Y") and of
`functional-language` in ho-180 and `city` in ho-184. No edge when:

- the kind only restates the entity type, as the words of its description do (`v1_0_0.py`
  type descriptions: a Technology is "a programming language, framework, library, tool,
  platform, service, database, or protocol", so "Languages" or "the same library" adds nothing);
- the validator refuses the pair (`IS_A` joins only Abstraction, Concept, Location,
  Organization, Skill and Technology, so a Document, Project or Person gets none);
- the kind is part of the name (`zephyr-rtos`, `kestrelmoor-hiking-society`, N4), or the
  entity stands for something other than the thing the kind describes (a course's subject).

From the H8 marks on ho-121 (the Pinecil is a soldering iron) and ho-009 ("shells are zsh").
Beyond the prompt: the prompt teaches `IS_A` only by direction ("React" IS_A "Framework",
`prompts.py:39`), with no rule for a kind noun in apposition, after "as the", or in a heading.
The 60-probe `ext-13` ("Neovim as my daily editor") has no `neovim IS_A editor`, so it does not
follow this rule; that file stays unchanged.

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
entity. An expression too vague to resolve ("recently", "a while back") gets no date, and nor
does a season ("since the spring", ho-106; "by spring", ho-048), whose bounds are a convention.
New probes still prefer absolute dates (design note line 26).

**N11 Goal windows.** A goal's stated time frame is its deadline: the `HAS_GOAL` edge carries
`valid_to` at the frame's end, at the precision the frame states, and no `valid_from`, because
the utterance does not say when the goal was set. "This year I want to" is `valid_to` `2026`,
"Goals for next year" `2027`, "paid off by 2035" `2035`, "training for a 10k in March"
`2027-03` (a goal looks forward, so a bare month is the first one after 2026-06-13). A lead-in
sets the frame for every goal under it. A frame whose year is unclear ("Q1 goals" beside a
done item, ho-025) gets no date. From the H8 mark on ho-004 ("This year" for the goal), and
within prompt Rules 5 and 16 (`prompts.py:53`, `:64`): the frame is a date that scopes a
stative fact. The 60-probe set has no Goal.

### Assertion kinds

`cease` means the fact stopped being true ("stopped", "quit", "left", "gave up on", "don't
... anymore"). `retract` means it was never true ("I never actually", "I misspoke", "scratch
that", "I was wrong about"). A self-correction is a retraction, not a directive (prompt Rule 6,
`prompts.py:54`; ext-43 to ext-52). A replacement in one turn carries both edges: `cease` on
the old target and an assertion on the new one (ext-53, ext-54). A negated preference is an
assertion: "I don't like X" is `DISLIKES` (Rule 6; ext-15). A plain negation of something never
stated ("I've never written PHP") is neither cease nor retract and gets no edge.

**N12 Flipped preferences.** When the utterance states the earlier preference as well as the
new one ("I used to love X, now it slows me down", "I don't enjoy X anymore"), it carries both:
`user PREFERS X` with `assertion_kind` `cease`, and the new `user DISLIKES X` as an assertion,
the same pair as a replacement (ext-53, ext-54). Neither edge gets a date: "now" contrasts the
present with the past and states no day the preference changed. From the H8 mark on ho-081 (the
user used to love it until now; going forward it is a dislike). This reverses the first
batch's correction that kept only the new preference. Beyond the prompt: Rule 6
(`prompts.py:54`) gives "don't like X anymore" as `DISLIKES` assert only. The 60-probe set has
no flipped preference (ext-15 states no earlier one).

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
