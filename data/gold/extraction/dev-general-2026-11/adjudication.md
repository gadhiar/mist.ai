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
- Ruling: pass A's three entities and two edges.
  - Id and kind: N14 Kind nouns (the pack's guide; Rule 25). Its own example, "a Pinecil
    soldering iron" is `pinecil IS_A soldering-iron`, is this shape: a proper name before a
    lowercase kind noun. The kind is not part of the name (it is not capitalised with it, as
    `kestrelmoor-hiking-society` is), and the name does not modify some other thing, so N4 Name
    modifiers (Rule 23, "the Ferrowind tuning guide") does not apply. Under N1 the name is the
    utterance's own run for the entity.
  - Type: N6 Type by its test. An indexer that processes documents is a tool, which the
    Technology test names. The Project test names an undertaking, and the utterance says nothing
    of the user working on it (contrast dvg-32, where "working on" makes the planner a `Project`).
    As a Technology, the kind gets its `IS_A`; the type was not chosen to keep that edge.
  - Metric: `850-documents-per-minute` is agreed and follows the Metric id rule (Rule 27): value
    first, the unit as written, no quantity word given.
  - No user edge: the possessive only names the thing (N8).

### dvg-17-metric-gorsey-test-suite-time: adjudicated whole

- Pass A: `gorsey` (`Technology`) `IS_A` `budgeting-app` (`Concept`), and `gorsey HAS_METRIC
  42-seconds`.
- Pass B: `gorsey-budgeting-app` (`Project`) `HAS_METRIC 42-seconds-test-suite`.
- Ruling: `gorsey` (`Technology`) `IS_A budgeting-app`, and `gorsey HAS_METRIC
  42-seconds-test-suite`.
  - Id, kind and type: as dvg-16, by N14 (the Pinecil shape), N4 not applying, and N6 (an app is
    a tool; nothing says the user works on it).
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
  - Id and kind: N14 (Rule 25), the Pinecil shape, as dvg-16. "scheduling service" says more
    than the Technology type's own word "service", so it is not a kind that only restates the
    type.
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
had them (Rules 8 to 10 of the engine core). The gold holds 128 entities and 74 relationships:
the 67 agreed relations and seven ruled ones. Every gold id is a fixed point of the id rails.

## Decisions needed

None. One reading is recorded for review: in dvg-16, dvg-17 and dvg-41, the name before the
kind noun is read as the entity's own name (N14's Pinecil shape), not as the owner of a thing
named by the whole phrase (N4). Read the other way, each of the three would lose its `IS_A` edge
and take the whole phrase as its id.
