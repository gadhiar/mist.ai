# Adjudication: the dev-general set (2026-11, plan v26 T70)

This records how the gold of `data/gold/extraction/dev-general-2026-11.jsonl` was made from the
two blind label passes in this directory, `pass-a.jsonl` and `pass-b.jsonl`. Every ruling below
rests on `data/gold/extraction/LABELLING.md` and the v1.4.0 pack's guide
(`backend/knowledge/ontologies/packs/v1_4_0/labelling.md`) with the prompt rules that carry it
(`backend/knowledge/ontologies/packs/v1_4_0/conventions.py`). The gold is labelled as said on
2026-06-13 (the pack's guide, Date entities). Neither pass was edited or re-run, the design file
was not opened, and no ruling is "pass A stands": where the gold matches one pass, it is because
the rule named picks that label.

## Agreement, measured before any ruling

`agreement.json` was written before any difference was ruled on, by
`scripts/eval_harness/selection_agreement.py` (`selection_agreement` under the v1.4.0 pack and
`run_rails`, rendered with `render`), with this set's file fixing the probes. All 60 probes are
prose (`group_of`). `test_gold_sets.py` recomputes it with `selection_agreement.main` and
compares bytes.

| Set figure | Kappa | F1 |
|---|---|---|
| All relations | 0.917 | 0.924 |
| Judgement relations | 0.917 | 0.924 |

Both kappas clear the bar of 0.61 and the target of 0.80. The judgement filter removed no
relation (every `filter` count in `agreement.json` is 0), so the two figures are equal. Pass A held 74 relations and pass B 71; 67 agree,
and 11 (7 of pass A's, 4 of pass B's) differ, all in four probes, dvg-16, dvg-17, dvg-18 and
dvg-41. Each of the four has a pass F1 of 0, under 0.61, so each is adjudicated whole below, with
both rationales. Two further probes differ in entities only: dvg-06 in one entity's type and
dvg-56 in an entity with no edge. They are ruled on too.

The 67 agreed relations and the agreed empty rows were read against the same guides and stand.

## Rulings

### dvg-06-core-negated-yaml-config: entity type

- Pass A types `yaml-config-files` as `Technology`; pass B as `Concept`. The edge,
  `user DISLIKES yaml-config-files`, is agreed.
- Ruling: `Technology`. Rule: N6 Type by its test (LABELLING.md; Rules 14 and 20). YAML is a data
  serialisation language and the phrase names configuration written in it, which the Technology
  test ("a programming language, framework, library, tool, platform, service, database, or
  protocol") covers. The Concept test, "an abstract concept such as a methodology, pattern,
  principle, domain, or theory", does not fit concrete files.

### dvg-16-metric-larchfield-indexer-throughput: adjudicated whole

- Pass A: `larchfield` (`Technology`) `IS_A` `indexer` (`Concept`), and `larchfield HAS_METRIC
  850-documents-per-minute`. Its reading: a named thing with a kind noun beside its name.
- Pass B: `larchfield-indexer` (`Project`) `HAS_METRIC 850-documents-per-minute`. Its reading:
  the whole name is the entity, a project of the user's.
- Ruling (re-ruled after the T70 pull-request review): pass B's two entities and one edge,
  `larchfield-indexer` (`Project`) `HAS_METRIC 850-documents-per-minute`. The first ruling cited
  a "Project test" that names an undertaking; the guide has no such test, so the type is ruled
  again on the tests as they are written.
  - The rule: N6, "A type is chosen by its type test, never to keep an edge."
    (`data/gold/extraction/LABELLING.md:131`). The two tests in play are the pack's type
    descriptions: Technology, "A programming language, framework, library, tool, platform,
    service, database, or protocol." (`backend/knowledge/ontologies/packs/v1_4_0/types.py:542`),
    and Project, "A work, personal, open-source, or research project."
    (`backend/knowledge/ontologies/packs/v1_4_0/types.py:594`).
  - Type: `Project`, by the best reading of those two tests. The utterance names the thing with
    the speaker's possessive and a proper name of its own, which marks a piece of software that
    is the user's own: a personal project in the Project test's words. The Technology test lists
    the kinds of technology anyone may use, and nothing in the utterance makes the indexer one
    of those rather than the user's own work. The agreed row dvg-32 reads the same shape the same
    way: both passes, blind, type the user's named route planner a `Project` with the id
    `orrin-route-planner`, and its `WORKS_ON` edge is the pack's "Indicates active involvement in
    a project." (`backend/knowledge/ontologies/packs/v1_4_0/types.py:1280`). The tests do not
    settle the type outright, so the question is listed under Decisions needed.
  - Kind: no `IS_A`. N14 gives a kind no edge when "`IS_A` joins only Abstraction, Concept,
    Location, Organization, Skill and Technology, so a Document, Project or Person gets none"
    (`backend/knowledge/ontologies/packs/v1_4_0/labelling.md:207-208`).
  - Id: `larchfield-indexer`. Rule 2: "When a contiguous run of the utterance's words names the
    entity, that run is the id, neither lengthened nor shortened"
    (`backend/knowledge/ontologies/packs/v1_4_0/conventions.py:802`). With no kind split off, the
    run that names the indexer is the name with its kind noun, as dvg-32's agreed
    `orrin-route-planner` keeps its kind noun.
  - Metric: `850-documents-per-minute` is agreed and follows the Metric id rule (Rule 27): value
    first, the unit as written, no quantity word given.
  - No user edge: the possessive only names the thing (N8).
  - Pass A's rationale, kept for the record: a proper name before a lowercase kind noun is
    N14's Pinecil shape, the name the entity and the kind a `Concept` it `IS_A`, with the
    indexer a tool under the Technology test.

### dvg-17-metric-gorsey-test-suite-time: adjudicated whole

- Pass A: `gorsey` (`Technology`) `IS_A` `budgeting-app` (`Concept`), and `gorsey HAS_METRIC
  42-seconds`.
- Pass B: `gorsey-budgeting-app` (`Project`) `HAS_METRIC 42-seconds-test-suite`.
- Ruling (re-ruled after the T70 pull-request review): `gorsey-budgeting-app` (`Project`)
  `HAS_METRIC 42-seconds-test-suite`.
  - Type: `Project`, under the same rule and tests as dvg-16 (N6,
    `data/gold/extraction/LABELLING.md:131`; Technology,
    `backend/knowledge/ontologies/packs/v1_4_0/types.py:542`; Project, "A work, personal,
    open-source, or research project.", `backend/knowledge/ontologies/packs/v1_4_0/types.py:594`).
    The speaker's possessive and the app's own proper name mark the user's own software, and the
    app's test suite says the user develops it: a personal project. This sits with the agreed
    dvg-32 as dvg-16 does. The question is listed under Decisions needed.
  - Kind and id: no `IS_A`, since N14 gives a Project none
    (`backend/knowledge/ontologies/packs/v1_4_0/labelling.md:207-208`), and the id is the run
    that names the app, `gorsey-budgeting-app` (Rule 2,
    `backend/knowledge/ontologies/packs/v1_4_0/conventions.py:802`).
  - Pass A's rationale, kept for the record: as for dvg-16, the Pinecil shape with the app a
    tool under the Technology test.
  - Metric id: the Metric id rule (Rule 27), "<value>-<unit>-<quantity>" with only empty parts
    left out. The utterance names what is measured, the test suite's run, so the quantity is
    `test-suite` and is kept; pass A's `42-seconds` leaves out a part the utterance gives (Rule 28
    states the same for list items: none of the quantity's words dropped).
  - The test suite gets no entity of its own: both passes measure it on the app, as a cold start
    is measured on its service in dev-lists (`2-4-seconds-cold-start`).

### dvg-18-metric-sqlite-query-time: adjudicated whole

- Pass A: `sqlite HAS_METRIC 3-ms`. Pass B: `sqlite HAS_METRIC 3-ms-typical-query`. Both keep
  `sqlite` (`Technology`) and neither labels the device, which only scopes the fact.
- Ruling: `sqlite HAS_METRIC 3-ms-typical-query`. Rule: the Metric id rule (Rule 27) as in
  dvg-17. What is measured is a typical query's time, so the quantity is `typical-query`, its
  article dropped (N1). The device phrase stays context (Rule 18), and the possessive only names
  it (N8).

### dvg-41-structural-planner-module-part-of-brindlecote: adjudicated whole

- Pass A: `brindlecote` (`Technology`) `COMPRISES planner-module` (`Mechanism`), and
  `brindlecote IS_A scheduling-service` (`Concept`).
- Pass B: `brindlecote-scheduling-service` (`Technology`) `COMPRISES planner-module`.
- Ruling: pass A's three entities and two edges.
  - Type: both passes type the service a `Technology`, and the Technology test names a
    "service" (`backend/knowledge/ontologies/packs/v1_4_0/types.py:542`). Unlike dvg-16 and
    dvg-17, the utterance gives it no possessive and nothing marks it as the user's own work, so
    the Project test does not compete.
  - Id and kind: N14 Kind nouns (Rule 25). Its own example, "a Pinecil soldering iron" is
    `pinecil IS_A soldering-iron` (`backend/knowledge/ontologies/packs/v1_4_0/labelling.md:198-199`),
    is this shape: a proper name before a lowercase kind noun, the name not modifying some other
    thing, so N4 Name modifiers (Rule 23) does not apply. A Technology can take `IS_A`, and
    "scheduling service" says more than the Technology test's own word "service", so it is not a
    kind that only restates the type.
  - The part: both passes type the module a `Mechanism` (Rule 14, a concrete component that
    performs a function inside a larger system). `PART_OF` admits no `Mechanism` source, so the
    composition is `COMPRISES` from the whole, "X is made up of Y" (Rule 15(e)); both passes
    agree on that edge's predicate and direction.

### dvg-56-habit-volunteer-food-bank: entity with no edge

- Pass B adds the `Organization` `kestrelford-food-bank` with no edge; pass A has no such
  entity. `user HAS_HABIT volunteer` is agreed.
- Ruling: no entity. Rule: Habits and incidental scope (Rules 17 and 18): the place phrase only
  scopes the habit and is context. N6's entity-only label is for an entity whose edge the
  validator refused, and Rule 31's place entity is for an Event; a habit is a stative fact, not
  an Event. The agreed rows give the lido in dvg-55 and the city in dvg-10 no entity either.

## The gold

60 rows; the utterances and tags are unchanged. The six `nofact` rows are empty, as both passes
had them (Rules 8 to 10 of the engine core). The gold holds 126 entities and 72 relationships:
the 67 agreed relations and five ruled ones. Every gold id is a fixed point of the id rails.

## Decisions needed

1. Technology or Project for the user's own named software (dvg-16, dvg-17). The Project test,
   "A work, personal, open-source, or research project." (`types.py:594`), and the Technology
   test, which names a "tool" (`types.py:542`), both fit an indexer or an app the speaker calls
   "my" and gives a proper name. The gold reads both as `Project`, as the agreed dvg-32 reads the
   user's named route planner. Read as `Technology`, each would take N14's Pinecil shape instead:
   the proper name as its id, and an `IS_A` edge to its kind noun (`indexer`, `budgeting-app`),
   one more entity and one more relation each.
2. The N14 or N4 reading of a proper name before a kind noun (dvg-41). The gold reads
   "Brindlecote" as the service's own name (N14's Pinecil shape), not as the owner of a thing
   named by the whole phrase (N4). Read the other way, dvg-41 would lose its `IS_A` edge and take
   the whole phrase, `brindlecote-scheduling-service`, as its id.
