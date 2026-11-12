# Adjudication: dev-list and held-out extraction gold sets (2026-11)

This record turns the two blind label passes over `utterances-2026-11.jsonl` into the published
sets `../dev-lists-2026-11.jsonl` (40 probes) and `../heldout-2026-11.jsonl` (200 probes). It
lists every place the passes disagreed and the rule that settled it, every row both passes got
wrong under a rule, the rules the guide does not state yet, and a sample of 20 held-out probes
for the H8 spot check. `tests/unit/eval_harness/test_gold_sets.py` checks the published files and
reads the "Dropped probes" and "Coverage gaps" sections below.

The held-out rule in `../LABELLING.md` applies to this file: whoever edits an extraction
prompt, the preprocessor, the splitter or the extraction settings never opens it.

## Method

- Inputs: `labels-a-2026-11.jsonl` (pass A) and `labels-b-2026-11.jsonl` (pass B), 240 rows each,
  labelled blind to each other from `../LABELLING.md`, ontology v1.4.0 and the production prompt.
- A probe both passes labelled the same is pass A, unless a rule of the guide says both are wrong
  ("Corrections to rows both passes agreed on").
- A probe they labelled differently is decided under `../LABELLING.md`. Where the guide is
  silent, the closest precedent in the 60-probe set (`data/ingest/extraction-gold-2026-06-14.jsonl`,
  tags `ext-NN`) or the prompt rule the guide cites settles it, and the rule is listed under
  "Rules to add". Where no rule picks between two readings that both fit (an id spelt two
  canonical ways, a type both tests allow), pass A stands, as the base pass.
- An utterance row's `intent` records what its author meant to cover. It is not a label rule:
  where it disagrees with the guide, the guide wins and the row is listed.
- Every id is written in the form the gold loader and the scorer produce
  (`canonical_id`, `scripts/eval_harness/score_extraction_run.py:65-77`), so a loaded probe
  equals its file row. The test checks this.
- No probe was dropped: every utterance could be labelled.

## Agreement before adjudication

Measured over the two passes as written, before any decision. Entities compare entity ids;
types compare the type of each id both passes have; edges compare `(source, predicate, target)`;
predicates compare the predicate on each `(source, target)` pair both passes link; dates and
kinds compare `valid_from`, `valid_to` and `assertion_kind` on each edge both passes have. Each
rate is shared over the union (entities, edges) or over the items both passes have (types,
predicates, dates and kinds).

| Field | Dev lists | Held-out | All |
|---|---|---|---|
| entities | 254/298 (85.2%) | 533/645 (82.6%) | 787/943 (83.5%) |
| types | 250/254 (98.4%) | 527/533 (98.9%) | 777/787 (98.7%) |
| edges | 234/290 (80.7%) | 344/483 (71.2%) | 578/773 (74.8%) |
| predicates | 233/233 (100.0%) | 342/344 (99.4%) | 575/577 (99.7%) |
| dates and kinds | 234/234 (100.0%) | 340/344 (98.8%) | 574/578 (99.3%) |
| whole probe | 23/40 (57.5%) | 138/200 (69.0%) | 161/240 (67.1%) |

The passes disagreed on 79 probes (17 dev, 62 held-out). Most disagreements are one pass
labelling a fact the other left out, and id spelling. Once both passes link the same two
entities they almost always pick the same predicate.

## Rule names used below

From `../LABELLING.md`:

- **Anchor**: the user anchor and durable third-party facts (prompt Rule 19; `ext-29`).
- **Events**: event and metric ids, milestone-class events (prompt Rule 11; `ext-21`, `ext-22`).
- **Metrics**: value-first metric ids and the sources `HAS_METRIC` allows.
- **Dates**: Date entities and relative dates against 2026-06-13.
- **Kinds**: assertion kinds (prompt Rule 6; `ext-15`).
- **Scope**: habits and incidental scope (prompt Rules 17 and 18; `ext-01`, `ext-13`, `ext-14`, `ext-31`).
- **Lists**: the list-shaped probe rules.
- **Validator**: an edge the validator would drop is not gold (Row schema).

New, listed under "Rules to add": **N1** id surface, **N2** goal ids, **N3** decisions,
**N4** name modifiers, **N5** stated places and organisations, **N6** type by its test,
**N7** third-party events, **N8** the team's tools, **N9** loader-form ids, **N10** personal
milestones.

## Disagreements

"A" or "B" is the pass the final probe follows; "A +" means pass A with the named change.

| Probe | What differed | Final | Rule |
|---|---|---|---|
| dvl-03 | A adds `timescaledb USED_FOR sensor-data` | A | Scope: "picked X for Y" states use for a purpose, as `ext-01` |
| dvl-04 | `zettelkasten-method` or `zettelkasten` | A | N1: the utterance says "the Zettelkasten method" |
| dvl-06 | A adds `data-team` and `marek MEMBER_OF data-team` | A | Anchor: a durable fact about Marek; the same shape as dvl-12, which both passes labelled |
| dvl-09 | goal ids | B | N2 |
| dvl-10 | `meetings-without-an-agenda` Concept or Preference | A | tie: both types are allowed and fit |
| dvl-15 | B adds `ilse EXPERIENCED thesis-defense` | B | N7 |
| dvl-18 | A adds `user EXPERIENCED lease-signing` | B | N10 |
| dvl-19 | B has the parser act on `csv` and keeps `rows` plural | B | prompt Rule 15(b), the actor acts on what it reads; N1 plural as written |
| dvl-23 | preference ids | A | N1: A keeps each item's words |
| dvl-24 | B drops `fernhollow-bank USES kotlin` | A | "is moving to Kotlin" is in progress, labelled as current as in `ext-26`; the headcount gets no edge (Metrics: an organisation is not a `HAS_METRIC` source) |
| dvl-29 | `10k` or `run-a-10k` | `training-for-10k` | N2: neither pass used the item's own verb phrase |
| dvl-30 | A dates today's planning event; B links the docs move to Astro; event ids | A + `astro`, `docs-site-move RELATED_TO astro` | Dates ("today" is 2026-06-13, one day); Events (the user took part in the planning); N3 |
| dvl-34 | `sister-call` or `calling-sister` | B | N1: the form `calling-parents` both passes gave dvl-08 |
| dvl-35 | B adds six `DEPENDS_ON` edges for tools the plan replaces | A | Lists: a checklist of plans gets no edge; the current tools are presupposed, not stated (prompt Rule 8) |
| dvl-37 | A has a trip event and splits "tram 28 and the castle"; B adds Cascais, Porto and Sintra, Marta's dinner, one combined event | A + B's three places and their `RELATED_TO` edges, `flight-to-lisbon RELATED_TO lisbon`, `marta EXPERIENCED birthday-dinner` | Lists (each item reads as a sentence, so two activities are two events); N5; N7 |
| dvl-38 | A adds `circuit-breaker MECHANISM_OF failing-dependency` and `shard-map` | A without that edge | prompt Rule 15(a): `MECHANISM_OF` says X is how Y works, which "stops calls to a failing dependency" does not; `shard-map` stays with no edge (N6), since `MECHANISM_OF` refuses a DataStructure target |
| dvl-40 | B adds Elixir, Kotlin and two `PART_OF` edges; B types three dislikes Preference | A | Scope: Rule 18 names `PART_OF` among the edges a scoping phrase ("pattern matching in Elixir") does not make; types tie |
| ho-003 | B adds `otp PART_OF erlang` | A | Scope, as dvl-40 |
| ho-004 | goal ids and the metric id | B, with `finish-marathon` | N2; Metrics: the value as the utterance states it ("two") |
| ho-006 | B adds `canary-release` with no edge | B | N6: the item states a fact about it that no edge can carry (`HAS_METRIC` refuses a Strategy source) |
| ho-007 | A: `offline-maps IMPLEMENTED_WITH openstreetmap`; B: a decision event and `user USES openstreetmap` | `user DECIDED openstreetmap-tile-bundling`, `openstreetmap-tile-bundling RELATED_TO openstreetmap` | N3; Events: object then nominalised verb, as `mobile-app-launch` |
| ho-012 | B adds `user WORKS_ON thesis` | A | Scope: "for my thesis" scopes the suggestions; Rule 18 names `WORKS_ON` |
| ho-013 | B adds `user EXPERIENCED cat-adoption` | B | Events: a life event the user took part in (prompt Rule 11) |
| ho-016 | ids singular or plural; three DataStructure or Concept | B | N1 plural as written; the abstract-type test for DataStructure fits queries, documents and posting lists |
| ho-019 | `beta-opening` or `beta-launch` | A | N1: "beta opened" |
| ho-021 | SQLite under Hearthline: `IMPLEMENTED_WITH` or `DEPENDS_ON` | A | Lists: the parent sets the predicate for its children, and Svelte takes `IMPLEMENTED_WITH` in both |
| ho-022 | B adds `user HAS_HABIT energy-drinks` cease | A | Scope: `HAS_HABIT` needs a cadence; listed under "Facts left unlabelled" |
| ho-024 | B adds guild membership and `adr-047 RELATED_TO event-sourcing` | A | Scope: "for the architecture guild" and "on event sourcing" scope the reading list |
| ho-025 | goal ids; B makes the talk an event in Rotterdam | A + B's goal ids | N2; N4 (Rotterdam is part of the meetup's name) |
| ho-028 | B adds `order-service` and three `COMPRISES`; `reliable-events` Goal or Concept | B, with `reliable-events` a Goal | Lists: a lead-in that names a project is an entity; Goal as both passes typed `zero-downtime-releases` in dvl-38 |
| ho-031 | A adds `user USES` for the four tools | A | N8 |
| ho-033 | A: `user USES valkey` and the old store ceased; B: a decision event | A's review, dates and `kafka USED_FOR events`, + `user DECIDED session-store-replacement`, `session-store-replacement RELATED_TO valkey` | N3: a review outcome is a decision, not yet a change of use; Scope: "keep Kafka for events" states the purpose |
| ho-034 | `query-tuning` or `mysql-query-tuning` | A | Scope and N1: "in MySQL" scopes the skill |
| ho-035 | B adds `user HAS_HABIT coffee` cease | A | Scope: no cadence; listed under "Facts left unlabelled" |
| ho-036 | B adds `user USES` for the table's tools | A | Anchor: the table states no link to the user, and `user` appears only when a stated fact uses it |
| ho-038 | B adds `kelderwick-rail USES ticket-gates`; gates Technology or Mechanism | B | Lists: a lead-in that names an organisation is an entity |
| ho-043 | `tallyroot-app` or `tallyroot` | A | N1 |
| ho-048 | metric id | A | Metrics: value, unit, then the quantity measured, as `87-percent-coverage` |
| ho-049 | B adds `grid-forecasting PART_OF corvane-energy` | B | "its grid forecasting" states the ownership; the same edge both passes gave dvl-35 |
| ho-052 | `ledger` Technology (A) or Concept (B), each keeping `APPLICABLE_TO` | `ledger` a Project, no edge | N6; Validator: `APPLICABLE_TO` refuses a Project target; listed under "Facts left unlabelled" |
| ho-053 | A: metrics without values; B: value-first metrics on AVIF, `DEPENDS_ON` | B, with `image-pipeline IMPLEMENTED_WITH avif` | Metrics (value first); the project-stack predicate of ho-021 |
| ho-059 | `key-value-store` or `open-source-key-value-store` | B | N1, as `in-memory-data-store` in `ext-41` |
| ho-060 | `terraform-modules` Concept or Technology | A | the team's own modules are not a tool or library |
| ho-061 | decision event id | B | Events: object then nominalised verb |
| ho-065 | `weekend-pings` or `being-pinged-on-weekends` | B | N1 |
| ho-066 | `mobile-app` or `mobile-build`; metric id | `mobile-build` (B), `62-mb-build-size` (A) | N1; Metrics: the quantity measured |
| ho-068 | B dates the Cork office `2026` | B | Dates: "this year" is 2026 |
| ho-069 | project id with or without "project" | B | N1, as `tallyroot-app` keeps its head noun |
| ho-082 | B adds the user's anchor to Kofi | B | Anchor |
| ho-089 | B adds the user's anchor to Nadia | B | Anchor |
| ho-092 | B adds the user's anchor to Hana | B | Anchor |
| ho-093 | A dates the cessation `2024` | A | Dates: "in two years" resolves to 2024; the hedge is on the self-description, the stop is stated (pass B's note) |
| ho-098 | B adds `utrecht` | A | N4 |
| ho-103 | B adds `wedding RELATED_TO bruges` | B | N5 |
| ho-104 | the event is the sunset (A) or the decision (B) | B | prompt Rule 11: the Event paired with `DECIDED` is the decision, and the date is the decision's |
| ho-107 | B adds `corvane` | A | N4 |
| ho-108 | B adds the anchor and `esme EXPERIENCED promotion` | A + `user KNOWS_PERSON esme` | Anchor; Events: a promotion is milestone-class, with no `EXPERIENCED` |
| ho-112 | A dates `HAS_METRIC`; B adds a milestone event on its Date | B | Events: passing 500 stars is milestone-class; Dates |
| ho-114 | A: the organising team; B: `hackathon RELATED_TO kelderwick-rail` | A | N4; the membership is stated. The intent's "no Date entity" is overruled (see below) |
| ho-115 | B adds `wrenfield-bakery` | A | N4 |
| ho-116 | B adds `steering-meeting RELATED_TO bergen` | B | N3 (the board decided, so no `DECIDED`); N5 |
| ho-118 | event ids | A | prompt Rule 3 collapses the version; Events: `launch`, as `ext-21` |
| ho-120 | B adds `pelloria-optics` | A | N4 |
| ho-121 | B adds `imre EXPERIENCED retirement` | B | N7; "last year" is a year, so no Date |
| ho-123 | B adds Marrowby Press and the anchor | B | Anchor; N5 ("interviewed at Marrowby Press") |
| ho-127 | `valid_from` 2025 (A) or 2026 (B) | A | Dates: "last year" is 2025 (commit ca03bc3) |
| ho-128 | B adds the user's anchor to Dario | B | Anchor |
| ho-129 | `monolith` Technology (A) or Concept (B), each keeping `APPLICABLE_TO` | `monolith` a Project, `APPLICABLE_TO` dropped | N6; Validator; listed under "Facts left unlabelled" |
| ho-134 | metric ids; both type `search-index` Technology | `search-index` a DataStructure, no metrics | N6: the index is a data shape, as `inverted-index` in ho-016; `HAS_METRIC` refuses a DataStructure source; listed under "Facts left unlabelled" |
| ho-163 | `directness` or `conciseness` | A | N1 tie: neither is the utterance's word |
| ho-164 | A adds `laptop HAS_METRIC 64-gb-ram`; preference id | A | Metrics: a laptop is a Technology, an allowed source (pass B's note); N1 "running models locally" |
| ho-167 | `podcast` Project or `podcast-recording` Concept | A | Scope: the target is the object, as `cloud-infrastructure` in `ext-14` |
| ho-170 | capability id | A | N1 tie |
| ho-181 | B splits out `kelderwick-rail` | A | N4 |
| ho-187 | goal id | B | N2 |
| ho-194 | `retrying-clients` or `network-retries` | A | N1: the strategy applies to the clients |
| ho-195 | B adds `user HAS_GOAL remembering-vocabulary` | A | prompt Rule 8: "my strategy for" presupposes the goal, it does not state it |
| ho-198 | `kafka-internals` (A) or `kafka` plus the anchor (B) | A + `user KNOWS_PERSON joaquin` | N1; Anchor |
| ho-200 | `release-train` or `thursday-release-train` | B | N1 |

## Corrections to rows both passes agreed on

| Probe | Change | Rule |
|---|---|---|
| ho-023 | adds `user KNOWS_PERSON` for Nadia, Freya and Milo | Anchor: the table states durable facts about three named people (pass B's note asked) |
| ho-081 | drops `user PREFERS vim-keybindings` cease, keeps `DISLIKES` | Kinds: a flipped preference keeps only the new one ("don't like X anymore" is `DISLIKES` assert) |
| ho-134 | `search-index` from Technology to DataStructure (also a disagreement above) | N6 |
| ho-165 | `semantic-search` becomes `search-by-meaning` | prompt Rule 14 uses `semantic-search` as its own Strategy example, so the id would be the prompt's, not the utterance's ("search my notes by meaning") |
| dvl-39, ho-015 | `0.2-percent-error-rate`, `2.4-seconds-cold-start`, `3.1-gb-peak-memory`, `3.80-euros-per-hour` and `1.8-seconds-cold-start` are written `02-percent-error-rate`, `24-seconds-cold-start`, `31-gb-peak-memory`, `380-euros-per-hour` and `18-seconds-cold-start` | N9 |

Three intents disagree with the guide, and the guide wins with no change to the labels: ho-105
and ho-119 ("no valid_from") carry the resolved start dates `2026-05-30` and `2021-06-13`, and
ho-114 ("no Date entity") carries `2026-07-14`, because the relative-date rule resolves each to a
single day.

## Pass B's notes

| Note | Decision |
|---|---|
| retypes that keep an edge the validator would refuse (ho-052, ho-129, ho-134) | not followed: N6 |
| dropped metrics (ho-131, dvl-24, ho-164) | followed for ho-131 and dvl-24 (an organisation is not a `HAS_METRIC` source); not for ho-164 (a laptop is a Technology) |
| no `MECHANISM_OF` in dvl-38 | followed: prompt Rule 15(a) |
| no user `EXPERIENCED` for lease signings (dvl-18, ho-111) | followed: N10 |
| no `KNOWS_PERSON` anchors in the ho-023 table | not followed: Anchor |
| no retract edge in ho-086 | followed: "the 3rd" names no month, so it resolves to no `YYYY-MM-DD` and there is no old `OCCURRED_ON` to retract (Dates) |
| no `valid_to` for the hedged cessation in ho-093 | not followed: Dates |
| no edges for future plans in dvl-24 and dvl-35 | followed for dvl-35; not for dvl-24, where the move is in progress |

## Rules to add

Rules this adjudication needed that `../LABELLING.md` does not state yet, each with what settled
it. The guide's owner decides whether to add them.

- **N1 Id surface.** An id is the utterance's own phrase for the thing, lowercased and
  hyphenated, with leading articles and possessive pronouns dropped and a plural kept as written:
  `regular-expressions` (`ext-28`), `in-memory-data-store` (`ext-41`), `backend-work` (`ext-01`).
  A habit id names the activity, as `journaling` and `long-run` (`ext-55`, `ext-57`). Where both
  passes' ids fit this and neither is the utterance's own phrase, pass A's stands.
- **N2 Goal ids.** A Goal id is the goal's verb phrase as the utterance words it, articles and
  possessive pronouns dropped: `get-pilots-license`, `cut-screen-time`, `pay-off-mortgage`. The
  60-probe set has no Goal; this follows the goal ids both passes agreed on (`ship-saltmere` in
  dvl-09, `learn-to-sail` in ho-091).
- **N3 Decisions.** A decision the user's team makes is `user DECIDED <decision event>`, and the
  decision event is linked to the thing chosen by `RELATED_TO` when no canonical predicate links
  an Event to it. The decision does not make its outcome a current fact (no `USES` for a tool
  not yet adopted, no cease for one not yet retired). A decision by someone else gets the event
  and the link but no `DECIDED`. From prompt Rule 11 ("decisions (paired with `DECIDED`)") and
  Rule 15, which reserves `RELATED_TO` for when no canonical shape fits.
- **N4 Name modifiers.** A name that only modifies another name (the Lisbon summit, the Corvane
  outage drills, the Wrenfield Bakery ordering site) gets no entity of its own, unless a list's
  lead-in carries it (Lists; dvl-35, ho-038). From `ext-18`, where "on the payments team" adds no
  entity, and the agreed rows ho-057, ho-069, ho-095 and ho-188.
- **N5 Stated places and organisations.** A place or organisation the utterance relates to an
  event with a preposition ("in Bruges", "at Marrowby Press", "to Porto") is an entity, linked to
  the event by `RELATED_TO`. From the agreed rows dvl-18, ho-013 and ho-027; the 60-probe set
  has no stated place.
- **N6 Type by its test.** A type is chosen by its type test, never to keep an edge. An edge the
  validator then refuses is dropped (Validator). An entity left with no edge stays as an
  entity-only label when the utterance states something about it, as prompt Example 7 keeps
  `rust` with no relationship; a Metric left with no edge is dropped with it, as both passes did
  in dvl-24 and ho-131.
- **N7 Third-party events.** A named person's own event that is not milestone-class (Ilse's
  thesis defense, Marta's birthday dinner, Imre's retirement) is `<person> EXPERIENCED <event>`,
  beside the anchor. From prompt Rule 19 (the third party's own fact, with the person as source)
  and Rule 11 (life events pair with `EXPERIENCED`).
- **N8 The team's tools.** A statement of which tools the user's team runs ("How our release
  goes", "our data stack", "we retired our self-hosted GitLab") puts `USES` on the user. A
  possessive that only names a thing ("Our API handles ...", `ext-24`) does not. From the agreed
  rows ho-073 and ho-094 and the `USES` edges both passes gave ho-006.
- **N9 Loader-form ids.** An id is written as `canonical_id` leaves it, so a decimal metric
  value loses its point in the file (`31-gb-peak-memory`), exactly as the loader and the
  normaliser's slug already strip it. This replaces the guide's "decimal point included"; the
  loaded gold is the same either way.
- **N10 Personal milestones.** Signing a lease is milestone-class, like closing on a house
  (`ext-22`): an Event with no user `EXPERIENCED` edge.

## Facts left unlabelled

Stated facts that no valid gold edge can carry, so a check of "every fact is labelled" finds
them missing by design:

- dvl-24, ho-131: an organisation's headcount (`HAS_METRIC` refuses an Organization source).
- ho-006: the canary release's share of traffic (`HAS_METRIC` refuses a Strategy source).
- ho-134: the index's size and rebuild time (`HAS_METRIC` refuses a DataStructure source).
- ho-052, ho-129: what the pattern suits or replaced (`APPLICABLE_TO` refuses a Project target).
- dvl-38: what circuit breakers, leader election and consistent hashing do (no predicate states a
  definition; `MECHANISM_OF` refuses a DataStructure or Mechanism target).
- ho-022, ho-035: giving up energy drinks and coffee (`HAS_HABIT` needs a cadence).
- ho-086: the launch's old date (unresolvable without a month).
- ho-018, ho-039, ho-121, ho-133: a third party's study, new hobby, pastime or lessons
  (`LEARNING`, `INTERESTED_IN` and `HAS_HABIT` take only a User source).
- ho-116: the board's decision (`DECIDED` takes only a User source); the board is an
  entity-only label (N3, N6).
- The class B facts listed under "H8 marks": none of them has a valid gold shape yet.

## Coverage gaps

None. Every extractable entity type appears in at least 3 held-out gold entities (the fewest:
Abstraction 3, Strategy 5, Pattern 6) and every extractable predicate in at least 2 held-out
gold edges (the fewest: `MECHANISM_OF` 2, `MIST_HAS_PREFERENCE` 2, then `APPLICABLE_TO`,
`PART_OF`, `MIST_HAS_CAPABILITY` and `MIST_HAS_TRAIT` 3 each). Recounted after the H8 marks,
which raised `IS_A` from 3 to 7.

## Dropped probes

None. The held-out set has all 200 probes and the dev-list set all 40.

## Sample for the H8 spot check

Twenty held-out probes, stratified by category in proportion to its share of the 200: list 4,
multi 3, neg 3, temporal 3, fill 3, nofact 2, coref 1, scope 1. Each category's exact share is
4, 3, 2.5, 2.5, 2.5, 2.5, 1.5 and 1.5; the three places left after rounding down go to the
25-probe categories that state facts (neg, temporal, fill). Within a category the probes are
drawn with `random.Random(20261106)` over the sorted tags.

The labels below are the ones checked, at integration commit c1e938e. "H8 marks" lists what
the check changed since.

For each probe, check:

1. every fact the utterance states is labelled, except those listed under "Facts left
   unlabelled";
2. nothing is labelled that the utterance does not state;
3. each entity type and each predicate is right;
4. each date (`valid_from`, `valid_to`, a Date entity, resolved against 2026-06-13) and each
   assertion kind (absent means assert) is right.

### `ho-004-list-dot-yearly-goals`

Utterance:

```text
This year I want to:
• finish a marathon
• earn a cloud architect certificate
• cut my screen time to two hours a day
```

Entities: `user` (User), `cut-screen-time` (Goal), `earn-cloud-architect-certificate` (Goal), `finish-marathon` (Goal), `two-hours-a-day-screen-time` (Metric)

Edges:

- `cut-screen-time` HAS_METRIC `two-hours-a-day-screen-time`
- `user` HAS_GOAL `cut-screen-time`
- `user` HAS_GOAL `earn-cloud-architect-certificate`
- `user` HAS_GOAL `finish-marathon`

### `ho-006-list-paren-release-steps`

Utterance:

```text
How our release goes:
1) Woodpecker CI builds the image
2) Argo CD rolls it out to staging
3) a canary release takes 5 percent of traffic
```

Entities: `user` (User), `argo-cd` (Technology), `canary-release` (Strategy), `woodpecker-ci` (Technology)

Edges:

- `user` USES `argo-cd`
- `user` USES `woodpecker-ci`

### `ho-009-list-heading-languages-shells`

Utterance:

```text
## Languages
- Kotlin at work
- Elixir on weekends
## Shells
- zsh
```

Entities: `user` (User), `elixir` (Technology), `kotlin` (Technology), `zsh` (Technology)

Edges:

- `user` USES `elixir`
- `user` USES `kotlin`
- `user` USES `zsh`

### `ho-016-list-star-search-internals`

Utterance:

```text
Inside the search stack:
* the tokenizer splits queries into terms
* the ranker scores candidate documents
* an inverted index maps terms to posting lists
```

Entities: `candidate-documents` (DataStructure), `inverted-index` (DataStructure), `posting-lists` (DataStructure), `queries` (DataStructure), `ranker` (Mechanism), `search-stack` (Project), `tokenizer` (Mechanism)

Edges:

- `inverted-index` COMPRISES `posting-lists`
- `ranker` OPERATES_ON `candidate-documents`
- `search-stack` COMPRISES `inverted-index`
- `search-stack` COMPRISES `ranker`
- `search-stack` COMPRISES `tokenizer`
- `tokenizer` OPERATES_ON `queries`

### `ho-041-multi-job-and-side-language`

Utterance:

```text
I'm a backend engineer at Driftmere Analytics, where most services are written in Scala, and I'm slowly picking up Kotlin on the side.
```

Entities: `user` (User), `driftmere-analytics` (Organization), `kotlin` (Technology), `scala` (Technology)

Edges:

- `driftmere-analytics` IMPLEMENTED_WITH `scala`
- `user` LEARNING `kotlin`
- `user` WORKS_AT `driftmere-analytics`

### `ho-060-multi-terraform-naming`

Utterance:

```text
Our Terraform modules follow a strict noun-verb naming scheme, and every resource tag uses PascalCase keys.
```

Entities: `noun-verb-naming` (Convention), `pascalcase` (Convention), `resource-tags` (Concept), `terraform-modules` (Concept)

Edges:

- `noun-verb-naming` NAMING_CONVENTION_OF `terraform-modules`
- `pascalcase` NAMING_CONVENTION_OF `resource-tags`

### `ho-068-multi-law-firm-offices`

Utterance:

```text
Thornquist Legal is headquartered in Leeds and opened an office in Cork this year.
```

Entities: `cork` (Location), `leeds` (Location), `thornquist-legal` (Organization)

Edges:

- `thornquist-legal` RELATED_TO `cork` (valid_from `2026`)
- `thornquist-legal` RELATED_TO `leeds`

### `ho-079-neg-off-windows-dated`

Utterance:

```text
I moved off Windows for good in June 2025; it's Fedora on everything now.
```

Entities: `user` (User), `fedora` (Technology), `windows` (Technology)

Edges:

- `user` USES `fedora`
- `user` USES `windows` (valid_to `2025-06`, assertion_kind `cease`)

### `ho-081-neg-keybindings-flip`

Utterance:

```text
I used to love Vim keybindings, now they just slow me down.
```

Entities: `user` (User), `vim-keybindings` (Concept)

Edges:

- `user` DISLIKES `vim-keybindings`

### `ho-083-neg-off-the-rewrite`

Utterance:

```text
I'm off the claims portal rewrite now; Paloma finishes it alone.
```

Entities: `user` (User), `claims-portal` (Project), `paloma` (Person)

Edges:

- `paloma` WORKS_ON `claims-portal`
- `user` KNOWS_PERSON `paloma`
- `user` WORKS_ON `claims-portal` (assertion_kind `cease`)

### `ho-096-temporal-on-call-since`

Utterance:

```text
Payments on-call has been my job since August 2026.
```

Entities: `user` (User), `payments-on-call` (Project)

Edges:

- `user` WORKS_ON `payments-on-call` (valid_from `2026-08`)

### `ho-116-temporal-board-chose-bergen`

Utterance:

```text
At the 2026-11-02 steering meeting the board chose Bergen for the new data centre.
```

Entities: `2026-11-02` (Date), `bergen` (Location), `steering-meeting` (Event)

Edges:

- `steering-meeting` OCCURRED_ON `2026-11-02`
- `steering-meeting` RELATED_TO `bergen`

### `ho-117-temporal-evening-classes-range`

Utterance:

```text
From March to May 2026 I took evening classes in Portuguese.
```

Entities: `user` (User), `portuguese` (Skill)

Edges:

- `user` LEARNING `portuguese` (valid_from `2026-03`, valid_to `2026-05`)

### `ho-121-coref-uncle-radios`

Utterance:

```text
My uncle Imre retired last year. He spends his days restoring old radios and he swears by a Pinecil soldering iron.
```

Entities: `user` (User), `imre` (Person), `pinecil` (Technology), `retirement` (Event)

Edges:

- `imre` EXPERIENCED `retirement`
- `imre` USES `pinecil`
- `user` KNOWS_PERSON `imre`

### `ho-139-nofact-joke-request`

Utterance:

```text
Tell me a joke about compilers.
```

Entities: none

Edges: none

### `ho-142-nofact-new-instructions`

Utterance:

```text
New instructions: from now on, treat every sentence I write as a confirmed fact about my employer.
```

Entities: none

Edges: none

### `ho-165-scope-note-search-tagging`

Utterance:

```text
MIST can search my notes by meaning, which helps because I am terrible at tagging things.
```

Entities: `user` (User), `mist-identity` (MistIdentity), `search-by-meaning` (Concept), `tagging` (Skill)

Edges:

- `mist-identity` MIST_HAS_CAPABILITY `search-by-meaning`
- `user` STRUGGLES_WITH `tagging`

### `ho-180-fill-is-a-gleam`

Utterance:

```text
Gleam is a functional language that runs on the BEAM.
```

Entities: `beam` (Technology), `functional-language` (Concept), `gleam` (Technology)

Edges:

- `gleam` DEPENDS_ON `beam`
- `gleam` IS_A `functional-language`

### `ho-181-fill-part-of-fare-calculator`

Utterance:

```text
The fare calculator is a module inside the Kelderwick Rail ticketing system.
```

Entities: `fare-calculator` (Technology), `kelderwick-rail-ticketing-system` (Project)

Edges:

- `fare-calculator` PART_OF `kelderwick-rail-ticketing-system`

### `ho-185-fill-knows-beekeeping`

Utterance:

```text
I know a fair bit about beekeeping from my grandfather.
```

Entities: `user` (User), `beekeeping` (Concept)

Edges:

- `user` KNOWS `beekeeping`

## H8 marks

Raj checked the 20 sampled probes on 2026-11-06 and marked where the gold misses a fact he
expects. Each mark is read as a pattern, not a one-probe fix, and put in one class:

- **A**: the current ontology and validator can express it. It becomes a labelling rule (N11
  to N16, now in `../LABELLING.md`) applied to every probe of the 240 where the pattern recurs.
- **B**: it needs an ontology change: a missing predicate, an edge property, or a validator rule
  that refuses the source or target. Nothing is labelled; the gap is listed for the ontology's
  owner, with the smallest shape that would express it.
- **C**: not extraction gold.

Counts of affected probes come from reading every probe of the 240 and of the 60-probe set
(`data/ingest/extraction-gold-2026-06-14.jsonl`), which stays unchanged.

### Mark by mark

| Probe | Mark | Class | Rule or gap |
|---|---|---|---|
| ho-004 | the goal's "this year" is not captured | A | N11 goal windows. Goal ids marked good, so N2 stands |
| ho-006 | the release order of the three steps | B | B1 step order |
| ho-009 | Kotlin at work, Elixir on weekends; shells are zsh | B, and A | B2 edge context; "shells are zsh" is N14 |
| ho-016 | much better | none | no change |
| ho-041 | "most services" is lost | B | B2 edge context (extent) |
| ho-060 | nothing missing | none | no change |
| ho-068 | headquarters against office | B | B3 location roles |
| ho-079 | good | none | no change |
| ho-081 | used to love it until now, a dislike from now | A | N12 flipped preferences |
| ho-083 | solid | none | no change |
| ho-096 | job against task | B | B8 job or role |
| ho-116 | topic links; the decision and who owns it | A, and B | N13 topic links and N3's owner entity; B6 decision ownership |
| ho-117 | the classes were in the evening | B | B2 edge context (time of day) |
| ho-121 | user knows the uncle; his ongoing pastime; Pinecil is a soldering iron | A, and B | N14 for the soldering iron; `user KNOWS_PERSON imre` was already gold; B4 kinship, B5 ongoing activity |
| ho-139 | good | none | no change |
| ho-142 | an explicit "remember this" request should be stored | C | a capability of the memory system, not an extraction label |
| ho-165 | the capability searches notes | A | N15 an activity keeps its object |
| ho-180 | a platform relation, though `DEPENDS_ON` is acceptable | B | B9 runs-on; `DEPENDS_ON` stays |
| ho-181 | good | none | no change |
| ho-185 | the knowledge came from the grandfather | A, and B | N16 for the grandfather as a person; B7 provenance, B4 kinship |

### Class A rules and the probes they changed

22 probes changed: 16 held-out, 6 dev. No probe was changed by two rules.

| Rule | Probes changed | Count | 60-probe rows that do not follow it | Beyond the prompt |
|---|---|---|---|---|
| N11 goal windows | ho-004, ho-187, dvl-09, dvl-29 | 4 | 0 (no Goal) | no: prompt Rules 5 and 16 |
| N12 flipped preferences | ho-077, ho-081 | 2 | 0 | yes: Rule 6 gives `DISLIKES` only (`prompts.py:54`) |
| N13 topic links | ho-007, ho-033, ho-104, ho-116, dvl-30 | 5 | 0 | yes: no rule links an event to its topic |
| N14 kind nouns | ho-001, ho-009, ho-113, ho-121, dvl-01, dvl-17 | 6 | 1 (`ext-13`) | yes: no rule for a kind noun in apposition, after "as the" or in a heading |
| N15 an activity keeps its object | ho-165 | 1 | 1 (`ext-58`) | no: it changes an id, not a fact |
| N16 a relative named by the relation | ho-027, ho-029, ho-185, dvl-34 | 4 | 0 | yes: Example 7 gives no Person (`prompts.py:111-115`) |

What changed:

- N11: `valid_to` on every `HAS_GOAL` edge: `2026` (ho-004, three goals), `2035` (ho-187),
  `2027` (dvl-09, three goals), `2027-03` (dvl-29). Not changed: ho-048 ("by spring", a season)
  and ho-025 ("Q1" with no clear year).
- N12: `user PREFERS <x>` with `assertion_kind` `cease` beside the existing `DISLIKES`. This
  reverses the ho-081 row under "Corrections to rows both passes agreed on".
- N13: ho-116 drops `steering-meeting RELATED_TO bergen` and gains `new-data-centre`
  (Location), `steering-meeting RELATED_TO new-data-centre`, `new-data-centre RELATED_TO bergen`
  and `board` (Organization, entity only, N3). ho-033 links `architecture-review` to
  `session-store-replacement` and `kafka`; dvl-30 links `planning` to `docs-site-move` and
  `feature-freeze`. ho-104 gains `legacy-billing-endpoints` (Technology) as the decision's
  object, and ho-007 gains `offline-maps-feature` (Project) as the subject the decision is for.
- N14: `IS_A` to a Concept: `wezterm` to `terminal`, `zsh` to `shell`, `late-trains` to
  `covers-band`, `pinecil` to `soldering-iron`, `helix` to `editor` and `fish` to `shell`
  (dvl-01), `kakoune` to `editor` (dvl-17). Not changed, because `IS_A` refuses the type:
  dvl-25 (a Document), dvl-26 (a Project), dvl-38's glossary (DataStructure and Mechanism), and
  dvl-22, whose entities are what the courses teach.
- N15: ho-165's `search-by-meaning` becomes `search-notes-by-meaning`, replacing the id the
  "Corrections" table gave it.
- N16: `wife` (ho-027, with `wife WORKS_AT juniperwell-clinic` and the `user` entity),
  `brother` (ho-029, with `brother USES elm`), `grandfather` (ho-185), `sister` (dvl-34), each
  with `user KNOWS_PERSON`. Not changed: dvl-08's "my parents" (a group), ho-074's "her
  sister", ho-078's "the baby".

None of the new held-out ids occurs in the production prompt, and the T8 checks (counts,
coverage, overlap, contamination, loading) still pass. `test_gold_sets.py` checks N11, N12, N16
and the "as the" case of N14 on both sets.

### Class B gaps

Line numbers are in `backend/knowledge/ontologies/v1_0_0.py` unless another file is named. The
validator derives its constraints from those definitions (`validator.py:40-65`). Gold records
no entity property and, on an edge, only `valid_from`, `valid_to` and `assertion_kind`
(`../LABELLING.md`, Row schema; `scripts/eval_harness/score_extraction_run.py:174-183`).

| Gap | Marks | What blocks it | Smallest shape that would express it | Probes of the 240 | Probes of the 60 |
|---|---|---|---|---|---|
| B1 step order | ho-006 | `PRECEDED_BY` (1559) takes only an Event source and an Event or Date target; process steps are not Events (621-623) | a directional `FOLLOWS` predicate between steps (Mechanism, Strategy, Concept, Technology) | 3: ho-006, ho-038, dvl-02 (a ranked list) | 0 |
| B2 edge context | ho-009, ho-041, ho-117 | the free-text `context` (220-224) and `frequency` (229-234) edge properties exist, but gold and the scorer carry neither | `setting` and `extent` edge properties and a scored `frequency`, each with a gold field | 19: ho-001, ho-009, ho-041, ho-058, ho-062, ho-066, ho-076, ho-106, ho-117, ho-161, ho-166, ho-174, dvl-10, dvl-13, dvl-21, dvl-24, dvl-25, dvl-31, dvl-40; and the cadence of 7 `HAS_HABIT` probes | 3: ext-13, ext-18, ext-24; and the cadence of ext-55 to ext-57 |
| B3 location roles | ho-068 | no extractable predicate links an Organization, Person or User to a Location except `RELATED_TO` (1400-1407) and `IS_A` (1370-1390); `location_type` (712-724) is a node property | `LOCATED_IN` (Organization, Person, User, Location to Location) with a `role` edge property (headquarters, office, site, home, birthplace) | 6: ho-027, ho-068, ho-101, ho-116, ho-130, ho-184; N5's event places add 5 (ho-013, ho-103, ho-123, dvl-18, dvl-37) | 0 |
| B4 kinship and relation to the user | ho-121, ho-185 | `KNOWS_PERSON` (1440-1445) has no relation field; `relationship_to_user` (483-488) is a node property, and the prompt puts the relation in the edge's `context` (`prompts.py`, Examples 23 and 25) | a `relationship_to_user` property on `KNOWS_PERSON`, with a closed vocabulary and a gold field | 18: ho-008, ho-011, ho-027, ho-029, ho-039, ho-042, ho-047, ho-051, ho-064, ho-103, ho-121, ho-135, ho-185, dvl-04, dvl-06, dvl-08, dvl-24, dvl-34 | 5: ext-08, ext-29, ext-30, ext-31, ext-32 |
| B5 a third party's ongoing activity | ho-121 | `HAS_HABIT` (1359-1364), `INTERESTED_IN` (1266-1272) and `LEARNING` (1306-1311) take only a User source | widen those three sources to include Person | 4: ho-018, ho-039, ho-121, ho-133 | 0 |
| B6 decision ownership | ho-116 | `DECIDED` (1321-1327) takes only a User source, so a team's decision is put on `user` (N3) and a board's has no edge | widen `DECIDED` sources to Person and Organization, the owner as source | 6: ho-116 (a board), and the team decisions of ho-007, ho-033, ho-061, ho-104, dvl-30 | 0 |
| B7 knowledge provenance | ho-185 | `LEARNED_FROM` (1454-1459) is internal, LearningEvent to ConversationContext, and excluded from extraction (1883) | a `learned_from` edge property on `KNOWS`, `INTERESTED_IN` and `EXPERT_IN` holding the source entity's id (a Person, Document, Project or Organization), with a gold field | 5: ho-069, ho-124, ho-126, ho-185, dvl-25 | 0 |
| B8 job or role | ho-096 | no type or predicate for a job; User `role` (439-444) and Person `role` (471-476) are node properties, so a job is forced into a Project with `WORKS_ON` (1251-1256) | a `Role` node type and `HAS_ROLE` (User, Person to Role) | 10: ho-005, ho-023, ho-041, ho-051, ho-096, ho-102, ho-123, ho-135, dvl-06, dvl-33 | 1: ext-31 |
| B9 runs on a platform | ho-180 | no predicate; `DEPENDS_ON` (1409-1418) is the closest and stays, as Raj accepts it | `RUNS_ON` (Technology, Project, Organization to Technology) | 5: ho-049, ho-161, ho-180, dvl-13, dvl-24 | 0 |
| B10 animals | the lead, 2026-11-06 (the read path's recall fixture types a pet dog as Person) | no animal type: the extractable types are `_EXTERNAL_NODE_TYPES` (1133-1162) plus MistIdentity (1877-1879), listed in the prompt at `prompts.py:14-17`; a Person is "a person other than the primary user" (468) | an `Animal` node type and `HAS_PET` (User, Person to Animal) | 1: ho-013 (a cat, with no entity; the adoption is the Event `cat-adoption`); none in dev | 0 |

No probe was retyped for B10.

### Class C

ho-142: a request to store something ("remember this") is an instruction to the memory system.
The extraction gold for a directive stays empty (prompt Rule 10), and nothing is labelled.
