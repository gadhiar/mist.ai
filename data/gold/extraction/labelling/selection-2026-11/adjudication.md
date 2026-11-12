# Adjudication: selection set, 2026-11

The selection set's two blind label passes (`pass-a.jsonl`, `pass-b.jsonl`) are merged here
into the gold fields of `data/gold/extraction/selection-2026-11.jsonl`. Their agreement was
measured first, before any ruling, and is recorded in `agreement.json`.

## Method

- Texts. Every ruling reads the utterance under two texts and no others: the labelling guide
  as of 058e52c (`data/gold/extraction/LABELLING.md`, cited by its rule names, N1 to N17 and
  its sections) and the reviewed text of ADR-054's v14 amendment (cited as "the amendment",
  with its point or register row). No example from outside the guide is used.
- What is adjudicated. A probe on which the passes agree in every relation and every entity
  takes that row unchanged and has no entry below. On every other probe, each relation that
  one pass holds and the other does not is ruled, and its outcome is logged: kept or dropped,
  and which pass that holds. A probe whose relation F1 between the passes is under 0.61 is
  adjudicated whole: each of its relations is read again, agreed ones included, and the entry
  gives each pass's rationale beside the ruling. A relation neither pass wrote, or an agreed
  relation the ruling drops, is logged as "neither held".
- Entities. An entity's id follows its relations' ruling. Where the passes type one id
  differently, or one pass holds an entity with no edge, the entry says which holds and why.
- Attributes. On a relation both passes hold, the attributes of the pass the entry starts
  from hold unless the entry says otherwise, and a `beyond_prompt` flag either pass wrote
  stands, since the guide's rules that ask for it (N13, N14, N16) say so.
- Assumptions of the agreement figures, which the adjudication does not change:
  1. No probe records its scope, so the filter read a probe as user-scope when either pass
     labels the user, and as third-party otherwise.
  2. A gold row records no batch an entity came from, so every entry of the batch record the
     rails read carries the pass's whole payload.
  3. The batch record is built with the extraction service's own splitter, item filter and
     batcher, imported from the engine, with no model call.
  4. Ids were compared as the rails store them, run with their relation phase off, so the id
     function adds and removes no edge.

## Rulings

### sel-002-list-dash-planner-parts (F1 0.40, adjudicated whole)

- Pass A: took the proper name as the project and read 'route planner' as its kind.
- Pass B: kept the utterance's noun phrase for the project whole.
- Ruling: N14: a proper name before a kind noun ('a Pinecil soldering iron') is the named thing, and the kind is not part of its name; 'Lanternfish' is the planner's own name, not another entity's, so N4 does not apply. The id is the name, and a Project takes no IS_A to the kind. The DEPENDS_ON edges follow that id; the USED_FOR edges stand as both passes wrote them.
- `lanternfish DEPENDS_ON cassandra` (pass A only): kept, pass A held.
- `lanternfish DEPENDS_ON ktor` (pass A only): kept, pass A held.
- `lanternfish DEPENDS_ON nats` (pass A only): kept, pass A held.
- `lanternfish-route-planner DEPENDS_ON cassandra` (pass B only): dropped, pass A held.
- `lanternfish-route-planner DEPENDS_ON ktor` (pass B only): dropped, pass A held.
- `lanternfish-route-planner DEPENDS_ON nats` (pass B only): dropped, pass A held.

### sel-006-list-ref-first-broker (F1 0.80)

- Ruling: 'for dispatch events' scopes the shortlist, not a use the utterance states (Habits and incidental scope, Rule 18), so no USED_FOR. The kind noun's IS_A is beyond the prompt (N14), so its flag stands.
- `rabbitmq USED_FOR dispatch-events` (pass B only): dropped, pass A held.
- `rabbitmq IS_A message-broker` attributes differ: settled by the attribute policy in the method section.

### sel-008-list-ref-middle-asked-about (F1 1.00)

- Ruling: Attributes only: the N14 IS_A is beyond the prompt, so its flag stands.
- `emacs IS_A editor` attributes differ: settled by the attribute policy in the method section.

### sel-010-list-plus-week-outside-office (F1 1.00)

- Ruling: Attributes only: N16's relative anchor is beyond the prompt, so its flag stands.
- `user KNOWS_PERSON godmother` attributes differ: settled by the attribute policy in the method section.

### sel-012-list-checkbox-migration-week (F1 0.40, adjudicated whole)

- Pass A: labelled the tools' use and cessation, and kept 'old' in the retired endpoints' id.
- Pass B: added the API the move names, built on the new framework, and dropped 'old'.
- Ruling: The checked item states a durable fact about the tracking API, which is now built with Ktor, so B's IMPLEMENTED_WITH and its Project hold. The retired endpoints' id is the utterance's own phrase with only the article dropped (N1), so A's id holds.
- `tracking-api IMPLEMENTED_WITH ktor` (pass B only): kept, pass B held.
- `user USES old-thrift-endpoints` (pass A only): kept, pass A held.
- `user USES thrift-endpoints` (pass B only): dropped, pass A held.

### sel-014-list-heading-plate-and-parked (F1 0.67)

- Ruling: N1 with the lead ruling in the Relabel log: 'for the pager rotation' scopes the notes, and an id that is already a contiguous run naming the thing is not lengthened, as `discord-bot` was kept over its trailing phrase.
- `user WORKS_ON handover-notes` (pass A only): kept, pass A held.
- `user WORKS_ON handover-notes-for-the-pager-rotation` (pass B only): dropped, pass A held.

### sel-015-list-mixed-lyon-office-update (F1 0.40, adjudicated whole)

- Pass A: read the two sites as employers and gave the colleague WORKS_AT each.
- Pass B: read the two sites as named units the colleague and the speaker are placed in.
- Ruling: N17: an office and a depot named by their place are units the utterance places people in ('moved over to us' places the colleague, and 'us' the speaker), not named employers, so MEMBER_OF holds, with the cessation on the old unit. 'Switched the pager rotation to Opsgenie' states what the tool is used for (Rule 18), so B's USED_FOR holds beside the user's USES (N8).
- `leopold MEMBER_OF lyon-office` (pass B only): kept, pass B held.
- `leopold MEMBER_OF marseille-depot` (pass B only): kept, pass B held.
- `leopold WORKS_AT lyon-office` (pass A only): dropped, pass B held.
- `leopold WORKS_AT marseille-depot` (pass A only): dropped, pass B held.
- `opsgenie USED_FOR pager-rotation` (pass B only): kept, pass B held.
- `user MEMBER_OF lyon-office` (pass B only): kept, pass B held.

### sel-023-temporal-pilot-after-freeze (F1 0.25, adjudicated whole)

- Pass A: named the go-live event by its subject only.
- Pass B: named it by the utterance's own phrase, place included.
- Ruling: The Event rule: the id is the utterance's own phrase for the event with the place as the utterance words it ('surf lesson in Cascais'), and the agreed freeze event keeps its verb the same way. N5's RELATED_TO to the terminal stands beside it.
- `quaywise-pilot-2026-04-14 OCCURRED_ON 2026-04-14` (pass A only): dropped, pass B held.
- `quaywise-pilot-2026-04-14 PRECEDED_BY froze-schema-2026-03-24` (pass A only): dropped, pass B held.
- `quaywise-pilot-2026-04-14 RELATED_TO fos-sur-mer-terminal` (pass A only): dropped, pass B held.
- `quaywise-pilot-went-live-at-fos-sur-mer-terminal-2026-04-14 OCCURRED_ON 2026-04-14` (pass B only): kept, pass B held.
- `quaywise-pilot-went-live-at-fos-sur-mer-terminal-2026-04-14 PRECEDED_BY froze-schema-2026-03-24` (pass B only): kept, pass B held.
- `quaywise-pilot-went-live-at-fos-sur-mer-terminal-2026-04-14 RELATED_TO fos-sur-mer-terminal` (pass B only): kept, pass B held.

### sel-024-scope-notes-and-tone (F1 0.57, adjudicated whole)

- Pass A: split the tone into two traits, one per adjective.
- Pass B: kept 'dry, brief tone' as one trait.
- Ruling: N1: the utterance names one tone with two describing words, and gold invents no id, so the trait is the utterance's phrase, as B wrote it.
- `mist-identity MIST_HAS_TRAIT brief` (pass A only): dropped, pass B held.
- `mist-identity MIST_HAS_TRAIT dry` (pass A only): dropped, pass B held.
- `mist-identity MIST_HAS_TRAIT dry-brief-tone` (pass B only): kept, pass B held.

### sel-025-list-dash-dispensary-software (F1 0.67)

- Ruling: N14's pattern: 'an Access database' is the named tool with a kind noun beside it, and the kind only restates the Technology type, so the id is the name, as 'a Pinecil soldering iron' is `pinecil`.
- `access USED_FOR controlled-drugs-log` (pass A only): kept, pass A held.
- `access-database USED_FOR controlled-drugs-log` (pass B only): dropped, pass A held.
- `dispensary USES access` (pass A only): kept, pass A held.
- `dispensary USES access-database` (pass B only): dropped, pass A held.

### sel-026-list-dash-msc-term (F1 0.86)

- Ruling: 'R for biostatistics' states what R is used for, which is itself a fact (Rule 18, Habits and incidental scope), so B's USED_FOR holds.
- `r USED_FOR biostatistics` (pass B only): kept, pass B held.

### sel-027-list-dash-why-rowing (F1 0.40, adjudicated whole)

- Pass A: kept the membership only, reading the reasons as no fact.
- Pass B: read each reason as a durable preference of the user.
- Ruling: List rule: each item completes the lead-in, 'I joined the club for the early mornings on the Avon', which states what the user likes, a durable preference; B's PREFERS edges hold beside the agreed membership.
- `user PREFERS big-breakfast-afterwards` (pass B only): kept, pass B held.
- `user PREFERS crew-that-never-talks-shop` (pass B only): kept, pass B held.
- `user PREFERS early-mornings-on-the-avon` (pass B only): kept, pass B held.

### sel-028-list-numbered-teachable-topics (F1 0.00, adjudicated whole)

- Pass A: read 'could happily teach' as expert-level proficiency.
- Pass B: read the topics as things the user knows.
- Ruling: Being able to teach a topic to a junior colleague is expert-level proficiency, EXPERT_IN's gloss, so A's predicate holds. Each topic is a competency one can be better or worse at, so the Skill test fits (B's types).
- `user EXPERT_IN anticoagulant-dosing` (pass A only): kept, pass A held.
- `user EXPERT_IN medicines-reconciliation` (pass A only): kept, pass A held.
- `user EXPERT_IN reading-a-renal-function-panel` (pass A only): kept, pass A held.
- `user KNOWS anticoagulant-dosing` (pass B only): dropped, pass A held.
- `user KNOWS medicines-reconciliation` (pass B only): dropped, pass A held.
- `user KNOWS reading-a-renal-function-panel` (pass B only): dropped, pass A held.
- Entities: the two topics the passes typed apart are Skills by the abstract-type test.

### sel-029-list-ref-third-ward-app (F1 1.00)

- Ruling: Attributes only: the N14 IS_A is beyond the prompt, so its flag stands.
- `medusa IS_A reference-app` attributes differ: settled by the attribute policy in the method section.

### sel-032-list-ref-neither-ordered-erg (F1 1.00)

- Ruling: Entity only: the id is the utterance's own phrase, so it takes no alternative (N1's last sentence lists one only where neither id is the utterance's phrase).
- `air-resistance-one IS_A rowing-machine` attributes differ: settled by the attribute policy in the method section.

### sel-033-list-plus-shift-rules (F1 1.00)

- Ruling: Types only: rules the user keeps for themself are the user's preferences.
- Entities: the three rules take B's Preference type, the type PREFERS names.

### sel-034-list-checkbox-msc-year-goals (F1 1.00)

- Ruling: Entity only: the checked goal is done, a milestone-class Event with no user EXPERIENCED and, with no date, no OCCURRED_ON; N6 keeps it as an entity-only label.

### sel-037-list-paren-fixed-dates (F1 0.62)

- Ruling: The user takes part in each of the fixed dates, so each event gets user EXPERIENCED (Event rule). 'The club' is no named owner, so the head race's id keeps only its own phrase (Event rule, N1).
- `autumn-head-race-2026-10-03 OCCURRED_ON 2026-10-03` (pass A only): kept, pass A held.
- `club-autumn-head-race-2026-10-03 OCCURRED_ON 2026-10-03` (pass B only): dropped, pass A held.
- `user EXPERIENCED audit-committee-2026-09-08` (pass A only): kept, pass A held.
- `user EXPERIENCED autumn-head-race-2026-10-03` (pass A only): kept, pass A held.
- `user EXPERIENCED ottoline-leaving-do-2026-09-19` (pass A only): kept, pass A held.

### sel-039-list-lettered-friends-hospice (F1 0.80)

- Ruling: 'It still uses paper drug charts' is a durable fact about the organisation, so A's USES holds, its target a tool (Technology).
- `elmstoke-hospice USES paper-drug-charts` (pass A only): kept, pass A held.
- Entities: the charts take A's Technology type, the target USES admits.

### sel-041-list-long-week-fixed-points (F1 0.82)

- Ruling: N1 with the lead ruling: a habit id is the utterance's own phrase, and 'on the river' and 'by ten' scope the habit, so the shorter contiguous runs hold.
- `user HAS_HABIT club-outing` (pass A only): kept, pass A held.
- `user HAS_HABIT club-outing-on-the-river` (pass B only): dropped, pass A held.
- `user HAS_HABIT lights-out` (pass A only): kept, pass A held.
- `user HAS_HABIT lights-out-by-ten` (pass B only): dropped, pass A held.
- Entities: the journal club is a recurring activity like the other habits, a Concept (A).

### sel-044-list-long-audit-figures (F1 0.95)

- Ruling: 'my discharge prescriptions audit' makes the user the subject of a fact, the work on the audit, so the user anchor holds (The user anchor) beside the agreed metrics.
- `user WORKS_ON discharge-prescriptions-audit` (pass B only): kept, pass B held.

### sel-045-coref-clinic-lead (F1 0.86)

- Ruling: The clinic's move is an event the utterance states, its id the utterance's phrase (Event rule), and N5 links it to the place it moves into. With no single day it gets no Date.
- `clinic-moves-into-new-wing RELATED_TO new-wing` (pass A only): kept, pass A held.

### sel-050-list-dash-year-nine-stumbles (F1 1.00)

- Ruling: Entities only: the pupils are a group with no entity (N16), so no edge, but the utterance states something about each topic, and N6 keeps an entity-only label.

### sel-051-list-numbered-teaching-weak-spots (F1 0.67)

- Ruling: N1: the thing the user struggles with is Fourier transforms; 'anything involving' names no entity, so the id is the run that names the thing.
- `user STRUGGLES_WITH anything-involving-fourier-transforms` (pass B only): dropped, pass A held.
- `user STRUGGLES_WITH fourier-transforms` (pass A only): kept, pass A held.

### sel-052-list-numbered-concert-order (F1 0.00, adjudicated whole)

- Pass A: read each piece as a Document the user refers to.
- Pass B: labelled nothing, reading no named document and no fact of the user.
- Ruling: The items describe pieces by their kind and name none, and rehearsing a piece is no reference to a document; no predicate of the guide carries the rehearsal, so the probe is empty (No-fact probes, List rule for a list with nothing durable).
- `user REFERENCES_DOCUMENT evening-hymn` (pass A only): dropped, pass B held.
- `user REFERENCES_DOCUMENT new-carol` (pass A only): dropped, pass B held.
- `user REFERENCES_DOCUMENT unaccompanied-motet` (pass A only): dropped, pass B held.

### sel-053-list-ref-second-simulation-tool (F1 0.67)

- Ruling: N17: 'the department' is a group noun with no name of its own, so no entity and no edge from it. 'For the waves unit' scopes the shortlist, not a stated use (Rule 18), so no USED_FOR. The user's USES with its date stands.
- `algodoo USED_FOR waves-unit` (pass B only): dropped, pass A held.
- `department USES algodoo` (pass A only): dropped, pass B held.
- `algodoo IS_A simulation-tool` attributes differ: settled by the attribute policy in the method section.

### sel-059-list-checkbox-before-mocks (F1 0.00, adjudicated whole)

- Pass A: labelled nothing, reading the checklist as tasks.
- Pass B: dated the mocks by the hall booked for them.
- Ruling: The lead-in names the mocks and the checked item books their hall for one day, which dates the event; an event stated with no participant gets OCCURRED_ON only (Event rule), with the date in its id. The tasks state no durable fact.
- `year-11-mocks-2026-12-07 OCCURRED_ON 2026-12-07` (pass B only): kept, pass B held.

### sel-060-list-nested-science-cover (F1 0.80)

- Ruling: Each person covers a subject at the school, a durable fact about that person (Durable third-party facts), so B's KNOWS edges hold. The parent bullets carry no group noun, so they name no team (N17).
- `isidore KNOWS physics` (pass B only): kept, pass B held.
- `jacinta KNOWS physics` (pass B only): kept, pass B held.
- `kasimir KNOWS chemistry` (pass B only): kept, pass B held.

### sel-061-list-heading-teaching-and-outside (F1 0.33, adjudicated whole)

- Pass A: read the courses the user teaches as expertise.
- Pass B: read them as subjects the user knows.
- Ruling: The heading states what the user teaches, not a claim of expert-level proficiency (EXPERT_IN's gloss), so KNOWS holds; the choir membership is agreed.
- `user EXPERT_IN year-10-electricity` (pass A only): dropped, pass B held.
- `user EXPERT_IN year-12-mechanics` (pass A only): dropped, pass B held.
- `user KNOWS year-10-electricity` (pass B only): kept, pass B held.
- `user KNOWS year-12-mechanics` (pass B only): kept, pass B held.

### sel-062-list-mixed-parents-evening (F1 0.80)

- Ruling: The user took part in the parents' evening ('I got home after nine'), so the event gets user EXPERIENCED (Event rule).
- `user EXPERIENCED parents-evening` (pass B only): kept, pass B held.

### sel-063-list-long-used-with-a-class (F1 0.00, adjudicated whole)

- Pass A: read each tool as one the user uses.
- Pass B: read each as one the user knows.
- Ruling: 'Used ... at some point' states past use, not current use, so USES as an assertion would state a current fact the utterance does not; it states the user's familiarity with each, which is KNOWS.
- `user KNOWS algodoo` (pass B only): kept, pass B held.
- `user KNOWS audacity` (pass B only): kept, pass B held.
- `user KNOWS desmos` (pass B only): kept, pass B held.
- `user KNOWS geogebra` (pass B only): kept, pass B held.
- `user KNOWS logger-pro` (pass B only): kept, pass B held.
- `user KNOWS phyphox` (pass B only): kept, pass B held.
- `user KNOWS scratch` (pass B only): kept, pass B held.
- `user KNOWS stellarium` (pass B only): kept, pass B held.
- `user KNOWS tinkercad` (pass B only): kept, pass B held.
- `user KNOWS tracker` (pass B only): kept, pass B held.
- `user USES algodoo` (pass A only): dropped, pass B held.
- `user USES audacity` (pass A only): dropped, pass B held.
- `user USES desmos` (pass A only): dropped, pass B held.
- `user USES geogebra` (pass A only): dropped, pass B held.
- `user USES logger-pro` (pass A only): dropped, pass B held.
- `user USES phyphox` (pass A only): dropped, pass B held.
- `user USES scratch` (pass A only): dropped, pass B held.
- `user USES stellarium` (pass A only): dropped, pass B held.
- `user USES tinkercad` (pass A only): dropped, pass B held.
- `user USES tracker` (pass A only): dropped, pass B held.

### sel-068-multi-teacher-choir-site (F1 0.80)

- Ruling: Teaching a subject states no expert-level claim (EXPERT_IN's gloss), so KNOWS holds, as in the teaching heading probe.
- `user EXPERT_IN physics` (pass A only): dropped, pass B held.
- `user KNOWS physics` (pass B only): kept, pass B held.

### sel-069-neg-club-handover (F1 1.00)

- Ruling: Attributes only: the cessation's stop date is stated, so it carries valid_to.
- `user MEMBER_OF astronomy-club` attributes differ: the cessation carries the stated stop date as valid_to (Date entities), pass B held.

### sel-071-list-dash-drawing-desk (F1 0.92)

- Ruling: The item reads 'I use an iPad Pro with Procreate for sketching': the purpose covers both tools, so A's USED_FOR for the tablet holds (Rule 18).
- `ipad-pro USED_FOR sketching` (pass A only): kept, pass A held.

### sel-072-list-dash-quarter-clients (F1 1.00)

- Ruling: Attributes only: the N14 IS_A is beyond the prompt, so its flag stands.
- `pellham-rook IS_A childrens-publisher` attributes differ: settled by the attribute policy in the method section.

### sel-074-list-numbered-chargeable-skills (F1 0.00, adjudicated whole)

- Pass A: read skills the user would charge for as expertise.
- Pass B: read them as skills the user knows.
- Ruling: A skill the user would charge for is one held at a professional, expert level, EXPERT_IN's gloss, so A holds.
- `user EXPERT_IN character-design` (pass A only): kept, pass A held.
- `user EXPERT_IN gouache-painting` (pass A only): kept, pass A held.
- `user EXPERT_IN hand-lettering` (pass A only): kept, pass A held.
- `user KNOWS character-design` (pass B only): dropped, pass A held.
- `user KNOWS gouache-painting` (pass B only): dropped, pass A held.
- `user KNOWS hand-lettering` (pass B only): dropped, pass A held.

### sel-075-list-ref-second-print-shop (F1 0.40, adjudicated whole)

- Pass A: linked the user to the print shop chosen.
- Pass B: made the proofs the user's project and linked the chosen shop to them.
- Ruling: N13: something chosen for a subject links to that subject, so the shop links to the proofs, and the proofs are the user's work (The user anchor). A user RELATED_TO the shop is not a shape the guide gives.
- `moth-almanac-proofs RELATED_TO grafica-albufera` (pass B only): kept, pass B held.
- `user RELATED_TO grafica-albufera` (pass A only): dropped, pass B held.
- `user WORKS_ON moth-almanac-proofs` (pass B only): kept, pass B held.
- `grafica-albufera IS_A print-shop` attributes differ: settled by the attribute policy in the method section.

### sel-076-list-ref-latter-studio-mate (F1 0.89)

- Ruling: Sharing the named studio is a durable fact of the user; no canonical predicate joins the user to a place, so RELATED_TO (Rule 15) holds.
- `user RELATED_TO inkwell-yard-studio` (pass B only): kept, pass B held.

### sel-077-list-ref-first-brush-pack (F1 0.50, adjudicated whole)

- Pass A: gave every pack its kind and ceased the two the user no longer opens.
- Pass B: labelled the first pack only.
- Ruling: N14: the lead-in's kind noun gives each item IS_A, as a heading 'Shells' over 'zsh' does, flagged beyond the prompt. 'The only one I still open' says the other two are no longer used, a cessation (Assertion kinds).
- `gouache-dreams-megapack IS_A brush-pack` (pass A only): kept, pass A held.
- `rusty-nib-set IS_A brush-pack` (pass A only): kept, pass A held.
- `user USES gouache-dreams-megapack` (pass A only): kept, pass A held.
- `user USES rusty-nib-set` (pass A only): kept, pass A held.
- `wet-edge-watercolours IS_A brush-pack` attributes differ: settled by the attribute policy in the method section.

### sel-078-list-ref-first-two-workshops (F1 0.00, adjudicated whole)

- Pass A: read the two accepted workshops as events the user takes part in.
- Pass B: read them as skills the user knows.
- Ruling: Teaching a workshop is an event the user takes part in (Event rule, EXPERIENCED); the utterance states no skill, and the declined workshop gets nothing.
- `user EXPERIENCED linocut-for-beginners` (pass A only): kept, pass A held.
- `user EXPERIENCED zine-making` (pass A only): kept, pass A held.
- `user KNOWS linocut-for-beginners` (pass B only): dropped, pass A held.
- `user KNOWS zine-making` (pass B only): dropped, pass A held.
- Entities: the two workshops are Events (A).

### sel-079-list-plus-studio-day (F1 0.50, adjudicated whole)

- Pass A: labelled only the item with its own cadence.
- Pass B: read the lead-in as the cadence of every item.
- Ruling: List rule: the lead-in sets the predicate for the items under it, and 'my studio day' gives each item the cadence of the studio day, so each is HAS_HABIT.
- `user HAS_HABIT emails` (pass B only): kept, pass B held.
- `user HAS_HABIT walk-along-the-turia` (pass B only): kept, pass B held.

### sel-083-list-paren-agreed-deadlines (F1 0.77)

- Ruling: Deadlines are deliveries, milestone-class events with OCCURRED_ON and no user EXPERIENCED (Event rule); the opening is the named person's own event (N7).
- `user EXPERIENCED delphine-gallery-opening-2026-09-05` (pass A only): dropped, pass B held.
- `user EXPERIENCED moth-almanac-files-to-printer-2026-08-21` (pass A only): dropped, pass B held.
- `user EXPERIENCED saltgrass-cover-art-2026-07-10` (pass A only): dropped, pass B held.

### sel-084-list-dot-want-to-improve (F1 0.00, adjudicated whole)

- Pass A: read each item as something the user is learning.
- Pass B: read each as a goal of the user.
- Ruling: 'Things I'd like to get better at' states a wish, not current learning, so HAS_GOAL holds, each Goal named by the item's own phrase (N2, List rule).
- `user HAS_GOAL animating-in-toon-boom` (pass B only): kept, pass B held.
- `user HAS_GOAL perspective-in-crowded-street-scenes` (pass B only): kept, pass B held.
- `user HAS_GOAL pricing-own-work` (pass B only): kept, pass B held.
- `user LEARNING animating-in-toon-boom` (pass A only): dropped, pass B held.
- `user LEARNING perspective-in-crowded-street-scenes` (pass A only): dropped, pass B held.
- `user LEARNING pricing-own-work` (pass A only): dropped, pass B held.
- Entities: the three items are Goals (B).

### sel-091-temporal-contract-then-roughs (F1 1.00)

- Ruling: Entities only; the relations are agreed.
- Entities: the book's release is an event the utterance states, dated only by a season, so it has no Date and no edge (Date entities); N6 keeps it as an entity-only label (B).

### sel-092-coref-editor-and-deputy (F1 0.44, adjudicated whole)

- Pass A: resolved 'the magazine' to Saltgrass and gave both editors their employment there.
- Pass B: kept 'the magazine' as its own id and made the cover the user's project.
- Ruling: The magazine the editor left is the one whose cover she commissioned, so the reference resolves to that entity and takes its own name (N1, List rule on references). The commission is a past event of the editor's, and the utterance states no current work of the user on the cover, so no WORKS_ON. The anchors and the cessation are agreed.
- `greta WORKS_AT magazine` (pass B only): dropped, pass A held.
- `greta WORKS_AT saltgrass` (pass A only): kept, pass A held.
- `user WORKS_ON saltgrass-cover` (pass B only): dropped, pass A held.
- `xanthe WORKS_AT magazine` (pass B only): dropped, pass A held.
- `xanthe WORKS_AT saltgrass` (pass A only): kept, pass A held.

### sel-093-scope-memos-into-briefs (F1 0.80)

- Ruling: Beside the MIST facts, the utterance states a user fact, forgetting the deadlines, which is a difficulty with them: user STRUGGLES_WITH, with the user anchor (The user anchor); each fact keeps its own subject.
- `user STRUGGLES_WITH deadline` (pass B only): kept, pass B held.

### sel-094-multi-architect-to-illustrator (F1 0.20, adjudicated whole)

- Pass A: made the training an event in its place, and the two crafts expertise.
- Pass B: read both crafts as knowledge and tied the user to the city.
- Ruling: The training is an event the user took part in, linked to its stated place (Event rule, N5); a user RELATED_TO a city is not a shape the guide gives. Switching to illustration as a profession, worked for commissions, is the professional level of EXPERT_IN's gloss, as for a skill the user charges for. Teaching linocut states the user's knowledge of linocut, as teaching does elsewhere in this set, so KNOWS linocut. The book commissions are the user's work (WORKS_ON).
- `trained-as-architect-in-valencia RELATED_TO valencia` (pass A only): kept, pass A held.
- `user EXPERIENCED trained-as-architect-in-valencia` (pass A only): kept, pass A held.
- `user EXPERT_IN illustration` (pass A only): kept, pass A held.
- `user EXPERT_IN teaching-linocut` (pass A only): dropped, pass B held.
- `user KNOWS illustration` (pass B only): dropped, pass A held.
- `user KNOWS linocut` (pass B only): kept, pass B held.
- `user RELATED_TO valencia` (pass B only): dropped, pass A held.
- `user WORKS_ON book-commissions` (pass A only): kept, pass A held.
- Entities: linocut is a Skill (B); the teaching activity is no entity.

### sel-095-list-dash-forecast-stages (F1 0.71)

- Ruling: Data the feature builder joins in is consumed as its input, INPUT_TO's gloss ('X feeds into Y'), not data it acts on or owns, so B holds.
- `feature-builder OPERATES_ON holidays` (pass A only): dropped, pass B held.
- `feature-builder OPERATES_ON weather` (pass A only): dropped, pass B held.
- `holidays INPUT_TO feature-builder` (pass B only): kept, pass B held.
- `weather INPUT_TO feature-builder` (pass B only): kept, pass B held.
- Entities: the meter reads are concrete data a pipeline stage pulls, not an abstract idea, so the DataStructure test fits (A).

### sel-096-list-dash-monday-first-things (F1 0.40, adjudicated whole)

- Pass A: named the tool by its proper name and gave it its kind.
- Pass B: kept the kind noun in the tool's id.
- Ruling: N14: a proper name followed by a kind noun ('a Pinecil soldering iron') is the named entity IS_A the kind, flagged beyond the prompt; the kind is not part of the name. The coffee is nothing durable.
- `ampere-atlas IS_A dashboard` (pass A only): kept, pass A held.
- `user USES ampere-atlas` (pass A only): kept, pass A held.
- `user USES ampere-atlas-dashboard` (pass B only): dropped, pass A held.

### sel-097-list-dash-company-courses (F1 0.67)

- Ruling: N1: the course's own phrase is the whole run 'power systems for non-engineers', and a labeller does not shorten it.
- `user LEARNING power-systems` (pass A only): dropped, pass B held.
- `user LEARNING power-systems-for-non-engineers` (pass B only): kept, pass B held.
- `user LEARNING bayesian-time-series` attributes differ: the lead-in's 'this year' scopes each course as a stative fact, valid_from 2026 (Date entities), pass B held.
- `user LEARNING dutch` attributes differ: the lead-in's 'this year' scopes each course as a stative fact, valid_from 2026 (Date entities), pass B held.

### sel-098-list-numbered-trusted-methods (F1 0.67)

- Ruling: N14: the lead-in's kind noun gives each item IS_A, as a heading 'Shells' over 'zsh' does, flagged beyond the prompt.
- `anything-with-a-transformer-in-it IS_A forecasting-method` (pass B only): kept, pass B held.
- `exponential-smoothing IS_A forecasting-method` (pass B only): kept, pass B held.
- `gradient-boosted-trees IS_A forecasting-method` (pass B only): kept, pass B held.

### sel-099-list-numbered-weekly-numbers (F1 1.00)

- Ruling: Attributes only: N16's relative anchor is beyond the prompt, so its flag stands.
- `user KNOWS_PERSON manager` attributes differ: settled by the attribute policy in the method section.

### sel-100-list-ref-second-orchestrator (F1 1.00)

- Ruling: Attributes only: the N14 IS_A is beyond the prompt, so its flag stands.
- `airflow IS_A orchestrator` attributes differ: settled by the attribute policy in the method section.
- `dagster IS_A orchestrator` attributes differ: settled by the attribute policy in the method section.

### sel-102-list-ref-middle-weather-feed (F1 0.80)

- Ruling: 'For the wind features' scopes the comparison, not a use the utterance states (Habits and incidental scope, Rule 18), so no USED_FOR.
- `icon-eu USED_FOR wind-features` (pass B only): dropped, pass A held.
- `icon-eu IS_A weather-feed` attributes differ: settled by the attribute policy in the method section.

### sel-103-list-ref-former-talk (F1 0.00, adjudicated whole)

- Pass A: took the conference as the user's event and linked it to the talk's subject.
- Pass B: took the talk given as the user's event, linked to the conference and its subject.
- Ruling: 'The former' resolves to the first item, the talk, and 'the one I gave' makes it an event the user took part in (List rule on references, Event rule); its subject links to it (N13), not to the conference. Giving the talk there is taking part in the conference too, so A's EXPERIENCED holds beside B's; the talk links to the conference it was for by RELATED_TO, no canonical shape joining two events (Rule 15).
- `lowland-data-days RELATED_TO probabilistic-load-forecasts` (pass A only): dropped, pass B held.
- `one-on-probabilistic-load-forecasts RELATED_TO lowland-data-days` (pass B only): kept, pass B held.
- `one-on-probabilistic-load-forecasts RELATED_TO probabilistic-load-forecasts` (pass B only): kept, pass B held.
- `user EXPERIENCED lowland-data-days` (pass A only): kept, pass A held.
- `user EXPERIENCED one-on-probabilistic-load-forecasts` (pass B only): kept, pass B held.
- Entities: the talk's id is the item's own phrase (N1, Event rule), so no alternative is listed: N1 lists one only where neither id is the utterance's phrase.

### sel-104-list-plus-repeating-week (F1 0.75)

- Ruling: N1 as in the river outing: 'on the polder roads' scopes the habit, so the shorter run names it.
- `user HAS_HABIT long-ride` (pass A only): kept, pass A held.
- `user HAS_HABIT long-ride-on-the-polder-roads` (pass B only): dropped, pass A held.
- Entities: a habit is a recurring activity, a Concept (A), as Example 24's habit is.

### sel-115-multi-grid-scientist-utrecht (F1 0.75)

- Ruling: N14 as in the route planner probe: 'Gridcast' is the forecast's own name, not another entity's (N4), so the id is the name and the Project takes no IS_A.
- `user WORKS_ON gridcast` (pass A only): kept, pass A held.
- `user WORKS_ON gridcast-forecast` (pass B only): dropped, pass A held.

### sel-118-list-dash-game-made-with (F1 0.77)

- Ruling: N1: 'the sea and the wind' names two things, each its own entity with its article dropped, so the purpose gives two USED_FOR edges.
- `fmod USED_FOR sea` (pass B only): kept, pass B held.
- `fmod USED_FOR sea-and-the-wind` (pass A only): dropped, pass B held.
- `fmod USED_FOR wind` (pass B only): kept, pass B held.

### sel-119-list-dash-systems-i-wrote (F1 0.92)

- Ruling: The mixer blends rain into fog, acting on both, so each is a target of OPERATES_ON (its gloss: acts upon or manipulates).
- `weather-mixer OPERATES_ON fog` (pass B only): kept, pass B held.

### sel-120-list-dash-day-off-genres (F1 0.67)

- Ruling: N14: the lead-in's kind noun gives each item IS_A, flagged beyond the prompt, as in the trusted-methods list.
- `anything-with-a-fishing-minigame IS_A genre` (pass B only): kept, pass B held.
- `city-builders IS_A genre` (pass B only): kept, pass B held.
- `roguelike-deckbuilders IS_A genre` (pass B only): kept, pass B held.

### sel-122-list-ref-second-engine (F1 0.86)

- Ruling: The engines were weighed for the game, and the second was picked and is still the engine ('never looked back'), so the game is built with it (IMPLEMENTED_WITH) beside the user's USES.
- `tidewrack IMPLEMENTED_WITH godot` (pass B only): kept, pass B held.
- `godot IS_A engine` attributes differ: settled by the attribute policy in the method section.

### sel-124-list-ref-first-art-tool (F1 1.00)

- Ruling: Attributes only: the N14 IS_A edges are beyond the prompt, and the use is dated.
- `aseprite IS_A art-tool` attributes differ: the IS_A edges are flagged beyond the prompt (N14), and 'last week' resolves to 2026-06 at the precision a week allows, valid_from on the use (Date entities), pass B held.
- `krita IS_A art-tool` attributes differ: the IS_A edges are flagged beyond the prompt (N14), and 'last week' resolves to 2026-06 at the precision a week allows, valid_from on the use (Date entities), pass B held.
- `pixelorama IS_A art-tool` attributes differ: the IS_A edges are flagged beyond the prompt (N14), and 'last week' resolves to 2026-06 at the precision a week allows, valid_from on the use (Date entities), pass B held.
- `user USES aseprite` attributes differ: the IS_A edges are flagged beyond the prompt (N14), and 'last week' resolves to 2026-06 at the precision a week allows, valid_from on the use (Date entities), pass B held.

### sel-126-list-plus-studio-rituals (F1 0.67)

- Ruling: N1 as in the river outing: 'round Kelvingrove' scopes the habit, so the shorter run names it.
- `user HAS_HABIT walk` (pass A only): kept, pass A held.
- `user HAS_HABIT walk-round-kelvingrove` (pass B only): dropped, pass A held.
- Entities: habits are recurring activities, Concepts (A), as Example 24's habit is.

### sel-127-list-checkbox-demo-aims (F1 0.80)

- Ruling: The lead-in names the demo the aims are for, and a lead-in that names a project is an entity (List rules); the user's aims for it make it the user's work, WORKS_ON. 'Tidewrack' names another entity, the game, so the demo keeps the whole phrase (N4).
- `user WORKS_ON tidewrack-demo` (pass B only): kept, pass B held.

### sel-130-list-paren-studio-calendar (F1 0.77)

- Ruling: The showcase and the wedding are on the user's calendar, events the user takes part in (Event rule; the wedding is also the named person's own, N7). The demo freeze is milestone-class, like the deadlines elsewhere in this set, so it gets OCCURRED_ON and no user EXPERIENCED.
- `user EXPERIENCED clydeside-showcase-2026-09-12` (pass A only): kept, pass A held.
- `user EXPERIENCED demo-freeze-2026-08-14` (pass A only): dropped, pass B held.
- `user EXPERIENCED torquil-wedding-2026-10-03` (pass A only): kept, pass A held.

### sel-131-list-dot-how-projects-run (F1 1.00)

- Ruling: Types only; the relations are agreed.
- Entities: how the user likes a project run is a workflow preference, the Preference type's own test, so Preference holds (B, N6).

### sel-135-list-long-skills-for-first-game (F1 1.00)

- Ruling: Types only; the relations are agreed.
- Entities: every item is a competency the user picked up, typed Skill like the other nine (B).

### sel-139-temporal-launch-start-demo (F1 0.67)

- Ruling: Event rule: the id is the utterance's own phrase for the event, 'Lampblack launched', with its single day appended; the game's name alone names the game, not its launch.
- `lampblack-2024-11-07 OCCURRED_ON 2024-11-07` (pass A only): dropped, pass B held.
- `lampblack-launched-2024-11-07 OCCURRED_ON 2024-11-07` (pass B only): kept, pass B held.

### sel-140-coref-borrowed-handheld (F1 0.57, adjudicated whole)

- Pass A: read the loan as the user's current use of the device.
- Pass B: read each friend's device as that friend's use.
- Ruling: The loan for the port states that the user uses the device now. The friends' devices are stated as possessions, not uses, and no extractable predicate carries ownership, so they get no edge; the anchors are agreed.
- `ailsa USES steam-deck` (pass B only): dropped, pass A held.
- `struan USES steam-deck` (pass B only): dropped, pass A held.
- `user USES steam-deck` (pass A only): kept, pass A held.

### sel-142-list-dash-commis-techniques (F1 0.00, adjudicated whole)

- Pass A: read techniques the user teaches as expertise.
- Pass B: read them as techniques the user knows.
- Ruling: Teaching a technique states no expert-level claim (EXPERT_IN's gloss), so KNOWS holds, as for teaching elsewhere in this set.
- `user EXPERT_IN breaking-down-a-whole-turbot` (pass A only): dropped, pass B held.
- `user EXPERT_IN making-a-proper-beurre-blanc` (pass A only): dropped, pass B held.
- `user EXPERT_IN sharpening-on-a-whetstone` (pass A only): dropped, pass B held.
- `user KNOWS breaking-down-a-whole-turbot` (pass B only): kept, pass B held.
- `user KNOWS making-a-proper-beurre-blanc` (pass B only): kept, pass B held.
- `user KNOWS sharpening-on-a-whetstone` (pass B only): kept, pass B held.

### sel-144-list-numbered-best-cuisines (F1 0.00, adjudicated whole)

- Pass A: read the cuisines cooked best as expertise.
- Pass B: read them as cuisines the user knows, each of the lead-in's kind.
- Ruling: 'Best' ranks the user's own cuisines against each other and makes no expert-level claim (EXPERT_IN's gloss), so KNOWS holds. N14: the lead-in's kind noun gives each IS_A cuisine, flagged beyond the prompt.
- `basque IS_A cuisine` (pass B only): kept, pass B held.
- `japanese IS_A cuisine` (pass B only): kept, pass B held.
- `lebanese IS_A cuisine` (pass B only): kept, pass B held.
- `user EXPERT_IN basque` (pass A only): dropped, pass B held.
- `user EXPERT_IN japanese` (pass A only): dropped, pass B held.
- `user EXPERT_IN lebanese` (pass A only): dropped, pass B held.
- `user KNOWS basque` (pass B only): kept, pass B held.
- `user KNOWS japanese` (pass B only): kept, pass B held.
- `user KNOWS lebanese` (pass B only): kept, pass B held.
- Entities: a cuisine is a domain, the Concept test, not a competency, so the three take Concept (A).

### sel-146-list-ref-second-oven (F1 1.00)

- Ruling: Attributes only: the N14 IS_A is beyond the prompt, so its flag stands.
- `ten-tray-electric-one IS_A combi-oven` attributes differ: settled by the attribute policy in the method section.
- Entities: the oven's id is the item's own phrase, so no alternative is listed (N1 lists one only where neither id is the utterance's phrase).

### sel-147-list-ref-first-two-on-pass (F1 0.67)

- Ruling: The training is an event of each named person (N7), linked to its stated place (N5), as the user's training is in the architect probe; a person RELATED_TO a city is not a shape the guide gives.
- `jimena EXPERIENCED trained-in-san-sebastian` (pass A only): kept, pass A held.
- `jimena RELATED_TO san-sebastian` (pass B only): dropped, pass A held.
- `octavio EXPERIENCED trained-in-san-sebastian` (pass A only): kept, pass A held.
- `octavio RELATED_TO san-sebastian` (pass B only): dropped, pass A held.
- `trained-in-san-sebastian RELATED_TO san-sebastian` (pass A only): kept, pass A held.

### sel-149-list-ref-first-knife (F1 0.62)

- Ruling: N1 drops only leading articles and possessive pronouns, so the knife's id keeps its size. 'I reach for the first one for nearly everything' states the user's preference for it beside its use, PREFERS. The lead-in's kind gives it IS_A knife, flagged beyond the prompt (N14).
- `240-mm-gyuto IS_A knife` (pass B only): kept, pass B held.
- `gyuto IS_A knife` (pass A only): dropped, pass B held.
- `user PREFERS 240-mm-gyuto` (pass B only): kept, pass B held.
- `user USES 240-mm-gyuto` (pass B only): kept, pass B held.
- `user USES gyuto` (pass A only): dropped, pass B held.
- `petty IS_A knife` attributes differ: settled by the attribute policy in the method section.

### sel-150-list-plus-days-off-pattern (F1 0.33, adjudicated whole)

- Pass A: named each habit by its activity.
- Pass B: kept each habit's place in its id.
- Ruling: N1 as in the river outing: 'at La Ribera' and 'at Sopelana' scope the habits, so the shorter runs name them; the cadences are stated, so each is HAS_HABIT.
- `user HAS_HABIT market` (pass A only): kept, pass A held.
- `user HAS_HABIT market-at-la-ribera` (pass B only): dropped, pass A held.
- `user HAS_HABIT surfing` (pass A only): kept, pass A held.
- `user HAS_HABIT surfing-at-sopelana` (pass B only): dropped, pass A held.

### sel-152-list-checkbox-newsletter-month (F1 0.75)

- Ruling: N14 as in the route planner probe: 'Second Stove' is the newsletter's own name, so the id is the name, and the Project takes no IS_A.
- `user WORKS_ON second-stove` (pass A only): kept, pass A held.
- `user WORKS_ON second-stove-newsletter` (pass B only): dropped, pass A held.
- `issue-41-2026-05-08 RELATED_TO preserved-lemons` attributes differ: settled by the attribute policy in the method section.

### sel-154-list-heading-restaurant-and-side (F1 1.00)

- Ruling: Attributes only: the N14 IS_A is beyond the prompt, so its flag stands.
- `larder-club IS_A supper-club` attributes differ: settled by the attribute policy in the method section.

### sel-155-list-mixed-messy-service (F1 0.50, adjudicated whole)

- Pass A: labelled only the anchor for the named colleague.
- Pass B: made tonight's service an event the user took part in.
- Ruling: One evening's service is routine work, an incidental occasion like the coffee in the guide's example (Durable third-party facts), so it gets no Event; the colleague's anchor is agreed.
- `service-2026-06-13 OCCURRED_ON 2026-06-13` (pass B only): dropped, pass A held.
- `user EXPERIENCED service-2026-06-13` (pass B only): dropped, pass A held.

### sel-159-list-long-section-rules (F1 1.00)

- Ruling: Types only; the relations are agreed.
- Entities: rules the user keeps on the section are workflow preferences, the Preference type's own test (B, N6).

### sel-161-multi-sous-chef-club-newsletter (F1 0.75)

- Ruling: N14 as in the route planner probe: 'Second Stove' is the newsletter's own name, so the id is the name.
- `user WORKS_ON second-stove` (pass A only): kept, pass A held.
- `user WORKS_ON second-stove-newsletter` (pass B only): dropped, pass A held.
- `larder-club IS_A supper-club` attributes differ: settled by the attribute policy in the method section.

### sel-162-scope-scaling-and-spanish (F1 0.00, adjudicated whole)

- Pass A: cut each MIST fact to its head and made the language the capability and the liking.
- Pass B: kept each phrase whole and read the Spanish as a trait the user likes.
- Ruling: N15: a capability keeps what the utterance gives it, as 'search my notes by meaning' keeps its manner, so the scaling keeps its range. Speaking Spanish to the user when tired is how MIST behaves, a trait, and 'which I like' is the user's preference for that behaviour; each fact keeps its own subject.
- `mist-identity MIST_HAS_CAPABILITY scale-a-recipe` (pass A only): dropped, pass B held.
- `mist-identity MIST_HAS_CAPABILITY scale-a-recipe-from-four-covers-to-ninety` (pass B only): kept, pass B held.
- `mist-identity MIST_HAS_CAPABILITY spanish` (pass A only): dropped, pass B held.
- `mist-identity MIST_HAS_TRAIT speak-to-me-in-spanish` (pass B only): kept, pass B held.
- `user PREFERS spanish` (pass A only): dropped, pass B held.
- `user PREFERS speak-to-me-in-spanish` (pass B only): kept, pass B held.

### sel-163-fill-fishmonger-recommendation (F1 0.80)

- Ruling: The person's job at the market is a durable fact about him (Durable third-party facts), so WORKS_AT holds beside the recommendation.
- `ulises WORKS_AT la-ribera` (pass B only): kept, pass B held.

### sel-165-list-dash-dive-bag-kit (F1 0.00, adjudicated whole)

- Pass A: read the kit as used for the project's surveys.
- Pass B: read the project as depending on the kit.
- Ruling: The kit goes in the bag for every survey: the items are used for the project's work, USED_FOR's gloss, as 'I use X for Y' is (Rule 18). The utterance states no dependency.
- `compact-camera USED_FOR maerl-watch` (pass A only): kept, pass A held.
- `maerl-watch DEPENDS_ON compact-camera` (pass B only): dropped, pass A held.
- `maerl-watch DEPENDS_ON pencils` (pass B only): dropped, pass A held.
- `maerl-watch DEPENDS_ON quadrat` (pass B only): dropped, pass A held.
- `maerl-watch DEPENDS_ON slates` (pass B only): dropped, pass A held.
- `maerl-watch DEPENDS_ON transect-tapes` (pass B only): dropped, pass A held.
- `pencils USED_FOR maerl-watch` (pass A only): kept, pass A held.
- `quadrat USED_FOR maerl-watch` (pass A only): kept, pass A held.
- `slates USED_FOR maerl-watch` (pass A only): kept, pass A held.
- `transect-tapes USED_FOR maerl-watch` (pass A only): kept, pass A held.

### sel-166-list-dash-thesis-chapters (F1 0.00, adjudicated whole)

- Pass A: made the thesis the user's project and linked its chapters to it.
- Pass B: made each chapter a project of the user.
- Ruling: The lead-in names the thesis, a project, so it is an entity (List rules) and the user's work; the chapters are what it concerns, linked by RELATED_TO with no canonical shape for it (Rule 15).
- `thesis RELATED_TO how-fast-maerl-beds-grow-back-after-scallop-dredging` (pass A only): kept, pass A held.
- `thesis RELATED_TO whether-drone-photos-can-map-them-at-low-tide` (pass A only): kept, pass A held.
- `thesis RELATED_TO which-juvenile-fish-shelter-in-them` (pass A only): kept, pass A held.
- `user WORKS_ON how-fast-maerl-beds-grow-back-after-scallop-dredging` (pass B only): dropped, pass A held.
- `user WORKS_ON thesis` (pass A only): kept, pass A held.
- `user WORKS_ON whether-drone-photos-can-map-them-at-low-tide` (pass B only): dropped, pass A held.
- `user WORKS_ON which-juvenile-fish-shelter-in-them` (pass B only): dropped, pass A held.

### sel-169-list-ref-latter-supervisor (F1 0.67)

- Ruling: List rule: an option nobody chose gets no edge, so the supervisor not asked for gets neither an anchor nor a fact.
- `anselm KNOWS seagrass` (pass A only): dropped, pass B held.
- `user KNOWS_PERSON anselm` (pass A only): dropped, pass B held.

### sel-170-list-ref-undredged-first-site (F1 1.00)

- Ruling: Attributes only: the N14 IS_A is beyond the prompt, so its flag stands.
- `caol-scotnish IS_A dive-site` attributes differ: settled by the attribute policy in the method section.
- `creag-gobhainn-bed IS_A dive-site` attributes differ: settled by the attribute policy in the method section.
- `sween-narrows IS_A dive-site` attributes differ: settled by the attribute policy in the method section.

### sel-171-list-ref-top-scoring-software (F1 0.57, adjudicated whole)

- Pass A: ceased the tools that did not stick.
- Pass B: labelled only the tool still in use.
- Ruling: The user tried every item, and 'nothing stuck except the top one' says the others are no longer used, a cessation (Assertion kinds), as for the packs no longer opened elsewhere in this set. 'A spreadsheet and a stopwatch' names two things, each its own entity.
- `user USES eventmeasure` (pass A only): kept, pass A held.
- `user USES spreadsheet` (pass A only): kept, pass A held.
- `user USES stopwatch` (pass A only): kept, pass A held.

### sel-172-list-ref-last-two-officemates (F1 0.60, adjudicated whole)

- Pass A: gave each member of the cohort and the user employment at the institute.
- Pass B: gave each membership of the institute.
- Ruling: The institute is where the cohort and the user are based and work, WORKS_AT's gloss (employment or membership at an organisation). MEMBER_OF is the guide's edge for a named team or club placed under an employer (N17), which the institute is not. The topics and anchors are agreed.
- `emrys MEMBER_OF skerryvore-institute` (pass B only): dropped, pass A held.
- `emrys WORKS_AT skerryvore-institute` (pass A only): kept, pass A held.
- `ludovic MEMBER_OF skerryvore-institute` (pass B only): dropped, pass A held.
- `ludovic WORKS_AT skerryvore-institute` (pass A only): kept, pass A held.
- `marit MEMBER_OF skerryvore-institute` (pass B only): dropped, pass A held.
- `marit WORKS_AT skerryvore-institute` (pass A only): kept, pass A held.
- `user MEMBER_OF skerryvore-institute` (pass B only): dropped, pass A held.
- `user WORKS_AT skerryvore-institute` (pass A only): kept, pass A held.

### sel-173-list-plus-fortnight-repeats (F1 0.50, adjudicated whole)

- Pass A: named each habit by its activity.
- Pass B: kept each habit's place or company in its id.
- Ruling: N1 as in the river outing: 'out of Gannet Rock' and 'with the department' scope the habits, so the shorter runs name them.
- `user HAS_HABIT boat-day` (pass A only): kept, pass A held.
- `user HAS_HABIT boat-day-out-of-gannet-rock` (pass B only): dropped, pass A held.
- `user HAS_HABIT sea-swim` (pass A only): kept, pass A held.
- `user HAS_HABIT sea-swim-with-the-department` (pass B only): dropped, pass A held.
- Entities: habits are recurring activities, Concepts (A), as Example 24's habit is.

### sel-177-list-paren-survey-day (F1 0.00, adjudicated whole)

- Pass A: read the list as one day's itinerary, with nothing durable.
- Pass B: read it as the user's routine and kept the database the user enters counts in.
- Ruling: 'A survey day, start to finish' describes the user's routine on every survey day, not one day's errands, and the last step states the user's use of the station database, a durable fact (USES). The other steps name no entity.
- `user USES station-database` (pass B only): kept, pass B held.

### sel-186-temporal-phd-start-to-stipend-end (F1 0.75)

- Ruling: As in the cohort probe: the institute where the user does the PhD is where the user works, WORKS_AT, with its start date; MEMBER_OF is N17's edge for a team.
- `user MEMBER_OF skerryvore-institute` (pass B only): dropped, pass A held.
- `user WORKS_AT skerryvore-institute` (pass A only): kept, pass A held.

### sel-187-multi-supervisors-and-cottage (F1 0.86)

- Ruling: Renting a cottage on the island states where the user lives, a durable fact; no canonical predicate joins the user to a place, so RELATED_TO (Rule 15) holds, as for the shared studio.
- `user RELATED_TO cumbrae` (pass B only): kept, pass B held.

### sel-188-list-dash-android-app-parts (F1 0.50, adjudicated whole)

- Pass A: took the app's proper name as the project.
- Pass B: kept the kind noun in the project's id.
- Ruling: N14 as in the route planner probe: 'Pennywhistle' is the app's own name, so the id is the name and the Project takes no IS_A; the IMPLEMENTED_WITH edges follow it.
- `pennywhistle IMPLEMENTED_WITH jetpack-compose` (pass A only): kept, pass A held.
- `pennywhistle IMPLEMENTED_WITH retrofit` (pass A only): kept, pass A held.
- `pennywhistle IMPLEMENTED_WITH room` (pass A only): kept, pass A held.
- `pennywhistle-app IMPLEMENTED_WITH jetpack-compose` (pass B only): dropped, pass A held.
- `pennywhistle-app IMPLEMENTED_WITH retrofit` (pass B only): dropped, pass A held.
- `pennywhistle-app IMPLEMENTED_WITH room` (pass B only): dropped, pass A held.

### sel-189-list-dash-habits-from-api-author (F1 0.89)

- Ruling: 'Elowen, who writes the API my app calls' is a durable fact about her, WORKS_ON the API, beside her anchor (Durable third-party facts).
- `elowen WORKS_ON api` (pass B only): kept, pass B held.

### sel-190-list-dash-why-left-agency (F1 0.40, adjudicated whole)

- Pass A: labelled only the new employer.
- Pass B: read each reason as something the user prefers.
- Ruling: The reasons for the move state what the user values in a job, each a preference (PREFERS), beside the agreed employer.
- `user PREFERS no-timesheets` (pass B only): kept, pass B held.
- `user PREFERS one-product-instead-of-nine` (pass B only): kept, pass B held.
- `user PREFERS pension-that-actually-exists` (pass B only): kept, pass B held.
- Entities: the reasons are preferences about work, the Preference type's test (B).

### sel-192-list-numbered-job-parts-ranked (F1 0.67)

- Ruling: N1 with the lead ruling: 'for the app stores' qualifies the paperwork, and the shorter run names it, as 'dark mode' does without 'everywhere'.
- `user KNOWS release-paperwork` (pass A only): kept, pass A held.
- `user KNOWS release-paperwork-for-the-app-stores` (pass B only): dropped, pass A held.

### sel-193-list-ref-former-crash-reporter (F1 1.00)

- Ruling: Attributes only: the N14 IS_A is beyond the prompt, so its flag stands.
- `bugsnag IS_A crash-reporter` attributes differ: settled by the attribute policy in the method section.

### sel-195-list-ref-last-navigation-library (F1 0.44, adjudicated whole)

- Pass A: gave every library its kind and the user's use.
- Pass B: labelled only the library the app shipped on.
- Ruling: List rule: options nobody chose get no edge, so the two libraries only prototyped with get neither use nor kind. The app shipped on the last one, which the user built it with, so A's USES holds for it beside the agreed edges.
- `decompose IS_A navigation-library` (pass A only): dropped, pass B held.
- `user USES compose-navigation` (pass A only): kept, pass A held.
- `user USES decompose` (pass A only): dropped, pass B held.
- `user USES voyager` (pass A only): dropped, pass B held.
- `voyager IS_A navigation-library` (pass A only): dropped, pass B held.
- `compose-navigation IS_A navigation-library` attributes differ: settled by the attribute policy in the method section.

### sel-196-list-ref-accepted-b-and-c (F1 0.00, adjudicated whole)

- Pass A: took the conference as the user's event and linked the talks to it as topics.
- Pass B: took the two accepted talks as the user's events, linked to the conference.
- Ruling: The accepted talks are events the user takes part in (Event rule), each linked to the conference it is for by RELATED_TO (Rule 15), and the declined talk gets no edge (List rule). Giving them is taking part in the conference too, so A's EXPERIENCED holds, as in the former-talk probe.
- `offline-first-savings-pots RELATED_TO pocketconf` (pass B only): kept, pass B held.
- `pocketconf RELATED_TO offline-first-savings-pots` (pass A only): dropped, pass B held.
- `pocketconf RELATED_TO what-screen-readers-taught-me` (pass A only): dropped, pass B held.
- `user EXPERIENCED offline-first-savings-pots` (pass B only): kept, pass B held.
- `user EXPERIENCED pocketconf` (pass A only): kept, pass A held.
- `user EXPERIENCED what-screen-readers-taught-me` (pass B only): kept, pass B held.
- `what-screen-readers-taught-me RELATED_TO pocketconf` (pass B only): kept, pass B held.
- Entities: the two talks are Events (B).

### sel-197-list-plus-week-off-keyboard (F1 0.33, adjudicated whole)

- Pass A: named each habit by its activity.
- Pass B: kept each habit's company or place in its id.
- Ruling: N1 as in the river outing: 'with the Hollin Beck harriers' and 'round Roundhay Park' scope the habits, so the shorter runs name them.
- `user HAS_HABIT fell-running` (pass A only): kept, pass A held.
- `user HAS_HABIT fell-running-with-the-hollin-beck-harriers` (pass B only): dropped, pass A held.
- `user HAS_HABIT five-kilometres` (pass A only): kept, pass A held.
- `user HAS_HABIT five-kilometres-round-roundhay-park` (pass B only): dropped, pass A held.
- Entities: habits are recurring activities, Concepts (A), as Example 24's habit is.

### sel-198-list-star-aims-written-down (F1 0.86)

- Ruling: The user names the person the work goes to, someone the user knows, not a name in passing, so the anchor holds (Durable third-party facts).
- `user KNOWS_PERSON gethin` (pass B only): kept, pass B held.

### sel-199-list-checkbox-release-four-two (F1 0.80)

- Ruling: A checked box states a fact that is done (List rules): the biometric login is now part of the app, PART_OF.
- `biometric-login PART_OF pennywhistle` (pass B only): kept, pass B held.

### sel-200-list-nested-what-i-own (F1 0.88)

- Ruling: The parent bullets set the subject for their children (List rules), and the user owns the children: 'only the map screen' says the user does not own the whole app, so no WORKS_ON for either parent.
- `user WORKS_ON branch-finder` (pass B only): dropped, pass A held.
- `user WORKS_ON pennywhistle` (pass B only): dropped, pass A held.

### sel-201-list-heading-quarters-and-never (F1 0.50, adjudicated whole)

- Pass A: made this quarter's items the user's current work.
- Pass B: labelled only the dislike.
- Ruling: The heading sets the predicate for its items (List rules): this quarter's items are the user's current work, WORKS_ON, and next quarter's are a plan with no edge. The quarter is not resolved to dates.
- `user WORKS_ON dark-theme` (pass A only): kept, pass A held.
- `user WORKS_ON pots-that-round-up-card-payments` (pass A only): kept, pass A held.
- Entities: a chatbot is a tool or service, the Technology test (B).

### sel-205-list-long-run-through-before-release (F1 0.30, adjudicated whole)

- Pass A: made every check a habit named by its whole line.
- Pass B: named each check by its head and read the two screen readers as tools in use.
- Ruling: 'Before every release' is a stated cadence, so each check is HAS_HABIT; as in the river outing, a trailing phrase that scopes a habit stays out of its id (N1). The two screen-reader lines state the user's use of a tool, 'I use X for Y' (Rule 18), so USES; HAS_HABIT admits no Technology.
- `user HAS_HABIT captions` (pass B only): kept, pass B held.
- `user HAS_HABIT captions-on-the-help-videos` (pass A only): dropped, pass B held.
- `user HAS_HABIT colour-contrast` (pass B only): kept, pass B held.
- `user HAS_HABIT colour-contrast-on-both-themes` (pass A only): dropped, pass B held.
- `user HAS_HABIT keyboard-only` (pass B only): kept, pass B held.
- `user HAS_HABIT keyboard-only-on-a-tablet` (pass A only): dropped, pass B held.
- `user HAS_HABIT landscape` (pass B only): kept, pass B held.
- `user HAS_HABIT landscape-on-a-small-phone` (pass A only): dropped, pass B held.
- `user HAS_HABIT second-screen-reader-on-the-transfer-flow` (pass A only): dropped, pass B held.
- `user HAS_HABIT talkback-through-the-whole-sign-up` (pass A only): dropped, pass B held.
- `user HAS_HABIT touch-targets` (pass B only): kept, pass B held.
- `user HAS_HABIT touch-targets-of-48-dp` (pass A only): dropped, pass B held.
- `user USES second-screen-reader` (pass B only): kept, pass B held.
- `user USES talkback` (pass B only): kept, pass B held.

### sel-206-list-long-cv-languages (F1 0.91)

- Ruling: N1: the item's own phrase cannot be an id, and its canonical form collides with another language's name, so gold lists the spelled-out id with the others as alternatives, as N1 lists both where neither id is the utterance's phrase.
- `user KNOWS c` (pass A only): dropped, pass B held.
- `user KNOWS c-sharp` (pass B only): kept, pass B held.

### sel-207-list-long-fell-day-kit (F1 0.50, adjudicated whole)

- Pass A: typed every item but the food as a Technology the user uses.
- Pass B: typed only the instruments as Technology and kept the rest as entity-only labels.
- Ruling: N6: a type is chosen by its test, never to keep an edge. Clothing, containers, emergency cover and food are not tools, so they stay entity-only labels; a paper map is a navigation tool like the compass, so it takes Technology and USES.
- `user USES foil-blanket` (pass A only): dropped, pass B held.
- `user USES paper-map` (pass A only): kept, pass A held.
- `user USES soft-flasks` (pass A only): dropped, pass B held.
- `user USES spare-gloves` (pass A only): dropped, pass B held.
- `user USES studded-fell-shoes` (pass A only): dropped, pass B held.
- `user USES windproof-smock` (pass A only): dropped, pass B held.

### sel-212-list-dash-promises-to-guesthouses (F1 0.86)

- Ruling: The lead-in names the product the promises are made for, a project, so it is an entity, and the user's promises for it make it the user's work (WORKS_ON), as for the demo aims.
- `user WORKS_ON stile` (pass B only): kept, pass B held.

### sel-214-list-numbered-countries-sold (F1 1.00)

- Ruling: Attributes only: the N14 IS_A is beyond the prompt, so its flag stands.
- `england IS_A country` attributes differ: settled by the attribute policy in the method section.
- `portugal IS_A country` attributes differ: settled by the attribute policy in the method section.
- `scotland IS_A country` attributes differ: settled by the attribute policy in the method section.
- `slovenia IS_A country` attributes differ: settled by the attribute policy in the method section.

### sel-215-list-ref-second-agency (F1 0.40, adjudicated whole)

- Pass A: linked the user to the chosen agency.
- Pass B: made the redesign the user's project and linked the chosen agency to it.
- Ruling: N13: something chosen for a subject links to that subject, so the agency links to the redesign, dated from its stated start; the redesign is the user's work, and 'Bootprint' names another entity, so it keeps the whole phrase (N4).
- `bootprint-redesign RELATED_TO oxbow-finch` (pass B only): kept, pass B held.
- `user RELATED_TO oxbow-finch` (pass A only): dropped, pass B held.
- `user WORKS_ON bootprint-redesign` (pass B only): kept, pass B held.
- `oxbow-finch IS_A agency` attributes differ: settled by the attribute policy in the method section.

### sel-216-list-ref-first-and-other-two-reports (F1 1.00)

- Ruling: Attributes only: the newest report's anchor is dated.
- `user KNOWS_PERSON fergal` attributes differ: 'joined only last week' resolves to 2026-06 at the precision a week allows, valid_from on the anchor as the other two carry 2024 (Date entities), pass B held.

### sel-217-list-ref-middle-survey-tool (F1 0.80)

- Ruling: 'For the post-trip questionnaire' scopes the trial, not a use the utterance states (Habits and incidental scope, Rule 18), so no USED_FOR, as for the weather feeds.
- `tally USED_FOR post-trip-questionnaire` (pass B only): dropped, pass A held.
- `tally IS_A survey-tool` attributes differ: settled by the attribute policy in the method section.

### sel-219-list-plus-calendar-repeats (F1 1.00)

- Ruling: Types only; the relations are agreed.
- Entities: habits are recurring activities, Concepts (A), as Example 24's habit is.

### sel-220-list-checkbox-half-year-objectives (F1 1.00)

- Ruling: Entities only; the relations are agreed.
- Entities: the checked launch is a done fact, an event with no stated day, so it has no Date and no edge (Date entities); N6 keeps it as an entity-only label (B).

### sel-224-list-dot-learning-on-the-side (F1 1.00)

- Ruling: Attributes only: the lead-in's year dates every item.
- `user LEARNING sketchnoting` attributes differ: the lead-in's 'this year' scopes each item as a stative fact, valid_from 2026 (Date entities), pass B held.
- `user LEARNING slovenian` attributes differ: the lead-in's 'this year' scopes each item as a stative fact, valid_from 2026 (Date entities), pass B held.
- `user LEARNING sql` attributes differ: the lead-in's 'this year' scopes each item as a stative fact, valid_from 2026 (Date entities), pass B held.

### sel-227-list-long-learned-my-way-around (F1 0.00, adjudicated whole)

- Pass A: read the tools learned as tools the user knows.
- Pass B: read them as tools the user is learning, from this year.
- Ruling: 'I've had to learn my way around' states learning done, so the user now knows each tool (KNOWS), not learning in progress. The lead-in's 'this year' dates when that began, valid_from 2026, as B's dates do (Date entities).
- `user KNOWS beds24` (pass A only): kept, pass A held.
- `user KNOWS cloudbeds` (pass A only): kept, pass A held.
- `user KNOWS eviivo` (pass A only): kept, pass A held.
- `user KNOWS freetobook` (pass A only): kept, pass A held.
- `user KNOWS mailchimp` (pass A only): kept, pass A held.
- `user KNOWS plain-ical-feeds` (pass A only): kept, pass A held.
- `user KNOWS quickbooks` (pass A only): kept, pass A held.
- `user KNOWS twilio` (pass A only): kept, pass A held.
- `user KNOWS xero` (pass A only): kept, pass A held.
- `user KNOWS zendesk` (pass A only): kept, pass A held.
- `user LEARNING beds24` (pass B only): dropped, pass A held.
- `user LEARNING cloudbeds` (pass B only): dropped, pass A held.
- `user LEARNING eviivo` (pass B only): dropped, pass A held.
- `user LEARNING freetobook` (pass B only): dropped, pass A held.
- `user LEARNING mailchimp` (pass B only): dropped, pass A held.
- `user LEARNING plain-ical-feeds` (pass B only): dropped, pass A held.
- `user LEARNING quickbooks` (pass B only): dropped, pass A held.
- `user LEARNING twilio` (pass B only): dropped, pass A held.
- `user LEARNING xero` (pass B only): dropped, pass A held.
- `user LEARNING zendesk` (pass B only): dropped, pass A held.

### sel-230-list-long-product-manager-refusals (F1 1.00)

- Ruling: Types only; the relations are agreed.
- Entities: ways of working the user refuses are workflow preferences, the Preference type's own test (B, N6).

### sel-233-temporal-vouchers-redesign-role (F1 0.50, adjudicated whole)

- Pass A: named each event by its subject with its day.
- Pass B: named the launch by its clause and the kick-off by its subject.
- Ruling: Event rule: the id is the utterance's own phrase for the event, and where the utterance words it as a clause the verb is kept, as the guide's 'signed-lease-in-ghent' and 'fly-into-lisbon' are. 'Gift vouchers went live' and 'Bootprint redesign kicks off' are the events; the subject alone names the product or the project. The role names no entity.
- `gift-vouchers-2026-04-02 OCCURRED_ON 2026-04-02` (pass A only): dropped, pass B held.
- `gift-vouchers-went-live-2026-04-02 OCCURRED_ON 2026-04-02` (pass B only): kept, pass B held.
- `bootprint-redesign-kicks-off-2026-06-15 OCCURRED_ON 2026-06-15` (neither pass): kept, neither held.
- `bootprint-redesign-2026-06-15 OCCURRED_ON 2026-06-15` (both passes): dropped, neither held.

### sel-234-scope-release-notes-and-exclamations (F1 0.50, adjudicated whole)

- Pass A: cut the capability to its head.
- Pass B: kept the capability's manner.
- Ruling: N15: a capability keeps what the utterance gives it, as 'search my notes by meaning' keeps its manner, so 'in plain English' stays. The trait is agreed, and the user's overuse of exclamation marks has no predicate of the guide.
- `mist-identity MIST_HAS_CAPABILITY draft-release-notes` (pass A only): dropped, pass B held.
- `mist-identity MIST_HAS_CAPABILITY draft-release-notes-in-plain-english` (pass B only): kept, pass B held.

### sel-236-list-dash-tools-saved-from-fire (F1 0.67)

- Ruling: N1: the chisels are the tools, and 'set of' counts them as 'two' does for the flasks, so the run that names them is 'bevel-edge chisels'.
- `user USES bevel-edge-chisels` (pass A only): kept, pass A held.
- `user USES set-of-bevel-edge-chisels` (pass B only): dropped, pass A held.

### sel-239-list-numbered-ruled-out-by-knees (F1 0.00, adjudicated whole)

- Pass A: read work the user's knees rule out as a difficulty.
- Pass B: read it as knowledge the user no longer has.
- Ruling: The utterance says the user can no longer do this work, a current difficulty (STRUGGLES_WITH's gloss); it does not say the user stopped knowing it, so a KNOWS cessation states a fact it does not.
- `user KNOWS double-digging` (pass B only): dropped, pass A held.
- `user KNOWS laying-floors` (pass B only): dropped, pass A held.
- `user KNOWS roofing` (pass B only): dropped, pass A held.
- `user STRUGGLES_WITH double-digging` (pass A only): kept, pass A held.
- `user STRUGGLES_WITH laying-floors` (pass A only): kept, pass A held.
- `user STRUGGLES_WITH roofing` (pass A only): kept, pass A held.

### sel-242-list-ref-middle-bench-timber (F1 0.00, adjudicated whole)

- Pass A: kept the place in the project's id.
- Pass B: named the project by its head and gave the chosen timber its kind.
- Ruling: N1 as in the river outing: 'on the green' scopes the benches, so the shorter run names them. N13: the timber chosen for them links to them. N14: the lead-in's kind noun gives the chosen item IS_A, flagged beyond the prompt; the other two are options nobody chose.
- `benches RELATED_TO sweet-chestnut` (pass B only): kept, pass B held.
- `benches-on-the-green RELATED_TO sweet-chestnut` (pass A only): dropped, pass B held.
- `sweet-chestnut IS_A timber` (pass B only): kept, pass B held.
- `user WORKS_ON benches` (pass B only): kept, pass B held.
- `user WORKS_ON benches-on-the-green` (pass A only): dropped, pass B held.

### sel-243-list-ref-all-but-first-class (F1 0.00, adjudicated whole)

- Pass A: read the classes the user agreed to teach as expertise.
- Pass B: read them as subjects the user knows.
- Ruling: Teaching a subject states no expert-level claim (EXPERT_IN's gloss), so KNOWS holds, as for teaching elsewhere in this set; the declined class gets no edge.
- `user EXPERT_IN furniture-repair` (pass A only): dropped, pass B held.
- `user EXPERT_IN tool-sharpening` (pass A only): dropped, pass B held.
- `user KNOWS furniture-repair` (pass B only): kept, pass B held.
- `user KNOWS tool-sharpening` (pass B only): kept, pass B held.

### sel-244-list-plus-retired-month-shape (F1 1.00)

- Ruling: Types only; the relations are agreed.
- Entities: habits are recurring activities, Concepts (A), as Example 24's habit is.

### sel-245-list-star-before-eighty (F1 0.86)

- Ruling: The user names the person a goal is shared with, someone the user knows, so the anchor holds, as for the hand-over goal (Durable third-party facts).
- `user KNOWS_PERSON fionnuala` (pass B only): kept, pass B held.

### sel-247-list-nested-society-duties (F1 0.80)

- Ruling: The parent bullets name what each person looks after (List rules), a durable fact about each, so WORKS_ON the duty beside the agreed membership (Durable third-party facts). A duty heading names one undertaking, so it keeps one id.
- `clovis WORKS_ON water-and-paths` (pass A only): kept, pass A held.
- `drusilla WORKS_ON seed-swap` (pass A only): kept, pass A held.
- `peregrine WORKS_ON water-and-paths` (pass A only): kept, pass A held.

### sel-248-list-heading-mornings-afternoons (F1 0.86)

- Ruling: The heading gives the cadence for its items (List rules), and the item names two places the user's mornings go, so each is HAS_HABIT.
- `user HAS_HABIT workshop` (pass A only): kept, pass A held.

### sel-249-list-mixed-committee-night (F1 1.00)

- Ruling: Attributes only: the N13 links are beyond the prompt, so their flags stand.
- `committee-night RELATED_TO waiting-list` attributes differ: settled by the attribute policy in the method section.
- `committee-night RELATED_TO water-bill` attributes differ: settled by the attribute policy in the method section.

### sel-251-list-long-built-since-retiring (F1 1.00)

- Ruling: Attributes only: each piece is built, so the work on it has ended.
- `user WORKS_ON bird-tables` attributes differ: 'what I've built' states finished work, so WORKS_ON is no longer true of any piece and each carries cease (Assertion kinds), pass A held.
- `user WORKS_ON bookcase` attributes differ: 'what I've built' states finished work, so WORKS_ON is no longer true of any piece and each carries cease (Assertion kinds), pass A held.
- `user WORKS_ON childs-stool` attributes differ: 'what I've built' states finished work, so WORKS_ON is no longer true of any piece and each carries cease (Assertion kinds), pass A held.
- `user WORKS_ON cold-frame` attributes differ: 'what I've built' states finished work, so WORKS_ON is no longer true of any piece and each carries cease (Assertion kinds), pass A held.
- `user WORKS_ON compost-bay` attributes differ: 'what I've built' states finished work, so WORKS_ON is no longer true of any piece and each carries cease (Assertion kinds), pass A held.
- `user WORKS_ON garden-gate` attributes differ: 'what I've built' states finished work, so WORKS_ON is no longer true of any piece and each carries cease (Assertion kinds), pass A held.
- `user WORKS_ON kitchen-dresser` attributes differ: 'what I've built' states finished work, so WORKS_ON is no longer true of any piece and each carries cease (Assertion kinds), pass A held.
- `user WORKS_ON lectern` attributes differ: 'what I've built' states finished work, so WORKS_ON is no longer true of any piece and each carries cease (Assertion kinds), pass A held.
- `user WORKS_ON rocking-horse` attributes differ: 'what I've built' states finished work, so WORKS_ON is no longer true of any piece and each carries cease (Assertion kinds), pass A held.
- `user WORKS_ON tool-chest` attributes differ: 'what I've built' states finished work, so WORKS_ON is no longer true of any piece and each carries cease (Assertion kinds), pass A held.

### sel-259-list-dash-episode-software (F1 0.50, adjudicated whole)

- Pass A: read the software as what the podcast is made with.
- Pass B: read it as what the podcast depends on.
- Ruling: The episode passes through each tool as it is made, so the podcast is built with them (IMPLEMENTED_WITH), as 'made with' and 'built from' read elsewhere in this set; the purposes are agreed.
- `undertow DEPENDS_ON descript` (pass B only): dropped, pass A held.
- `undertow DEPENDS_ON hindenburg` (pass B only): dropped, pass A held.
- `undertow DEPENDS_ON reaper` (pass B only): dropped, pass A held.
- `undertow IMPLEMENTED_WITH descript` (pass A only): kept, pass A held.
- `undertow IMPLEMENTED_WITH hindenburg` (pass A only): kept, pass A held.
- `undertow IMPLEMENTED_WITH reaper` (pass A only): kept, pass A held.

### sel-260-list-dash-stories-being-chased (F1 0.18, adjudicated whole)

- Pass A: read the stories as the user's interests, each part of the podcast.
- Pass B: read each story as work the user is doing.
- Ruling: Chasing a story is working on it, WORKS_ON's gloss (active involvement), so each is a Project of the user. They are for a season not yet made, so the utterance states no PART_OF yet, and no interest beyond the work.
- `dredging-records-that-went-missing PART_OF undertow` (pass A only): dropped, pass B held.
- `ferry-contract-signed-without-a-tender PART_OF undertow` (pass A only): dropped, pass B held.
- `user INTERESTED_IN dredging-records-that-went-missing` (pass A only): dropped, pass B held.
- `user INTERESTED_IN ferry-contract-signed-without-a-tender` (pass A only): dropped, pass B held.
- `user INTERESTED_IN who-really-owns-the-lyttelton-coal-berths` (pass A only): dropped, pass B held.
- `user WORKS_ON dredging-records-that-went-missing` (pass B only): kept, pass B held.
- `user WORKS_ON ferry-contract-signed-without-a-tender` (pass B only): kept, pass B held.
- `user WORKS_ON who-really-owns-the-lyttelton-coal-berths` (pass B only): kept, pass B held.
- `who-really-owns-the-lyttelton-coal-berths PART_OF undertow` (pass A only): dropped, pass B held.
- Entities: the stories are Projects (B).

### sel-261-list-numbered-interviewing-skills (F1 0.67)

- Ruling: N14: the lead-in's kind noun gives each item IS_A, flagged beyond the prompt, as in the trusted-methods list.
- `doorstepping IS_A interviewing-skill` (pass A only): kept, pass A held.
- `reading-documents-upside-down IS_A interviewing-skill` (pass A only): kept, pass A held.
- `staying-quiet-after-a-hard-question IS_A interviewing-skill` (pass A only): kept, pass A held.

### sel-264-list-ref-second-microphone (F1 1.00)

- Ruling: Attributes only: the N14 IS_A is beyond the prompt, so its flag stands.
- `broadcast-dynamic IS_A microphone` attributes differ: settled by the attribute policy in the method section.

### sel-265-list-ref-third-application (F1 0.00, adjudicated whole)

- Pass A: made the grant awarded an event the user took part in.
- Pass B: labelled nothing.
- Ruling: A grant coming through is milestone-class, like a promotion, so it gets no user EXPERIENCED, and with no stated day it gets no Date (Event rule, Date entities). The applications that failed get no edge (List rule).
- `user EXPERIENCED reporting-grant-for-pacific-shipping` (pass A only): dropped, pass B held.
- Entities: N6 keeps the grant as an entity-only label, since the utterance states it.

### sel-266-list-plus-standing-dates (F1 1.00)

- Ruling: Types only; the relations are agreed.
- Entities: habits are recurring activities, Concepts (A), as Example 24's habit is.

### sel-271-list-dot-interview-languages (F1 0.67)

- Ruling: N14: the lead-in's kind noun gives each item IS_A, flagged beyond the prompt, as in the trusted-methods list.
- `english IS_A language` (pass B only): kept, pass B held.
- `samoan IS_A language` (pass B only): kept, pass B held.
- `te-reo-maori IS_A language` (pass B only): kept, pass B held.

### sel-274-list-long-pieces-proud-of (F1 0.00, adjudicated whole)

- Pass A: typed each piece as a Document the user refers to.
- Pass B: typed each as a Project the user works on.
- Ruling: N6: a type is chosen by its test, never to keep an edge. Each piece is a finished article named by its title, the Document test, and the user discusses each here, REFERENCES_DOCUMENT's gloss; WORKS_ON (active involvement) is not true of finished work and admits no Document.
- `user REFERENCES_DOCUMENT ferry-tender` (pass A only): kept, pass A held.
- `user REFERENCES_DOCUMENT harbourmaster-who-kept-two-logbooks` (pass A only): kept, pass A held.
- `user REFERENCES_DOCUMENT last-woolshed-dance` (pass A only): kept, pass A held.
- `user REFERENCES_DOCUMENT lighthouse-keepers-tapes` (pass A only): kept, pass A held.
- `user REFERENCES_DOCUMENT night-with-the-oyster-fleet` (pass A only): kept, pass A held.
- `user REFERENCES_DOCUMENT profile-of-a-shearing-gang-cook` (pass A only): kept, pass A held.
- `user REFERENCES_DOCUMENT school-with-one-pupil` (pass A only): kept, pass A held.
- `user REFERENCES_DOCUMENT six-hours-in-a-magistrates-waiting-room` (pass A only): kept, pass A held.
- `user REFERENCES_DOCUMENT wharf-strike-told-by-the-tea-lady` (pass A only): kept, pass A held.
- `user REFERENCES_DOCUMENT what-the-dredger-found` (pass A only): kept, pass A held.
- `user WORKS_ON ferry-tender` (pass B only): dropped, pass A held.
- `user WORKS_ON harbourmaster-who-kept-two-logbooks` (pass B only): dropped, pass A held.
- `user WORKS_ON last-woolshed-dance` (pass B only): dropped, pass A held.
- `user WORKS_ON lighthouse-keepers-tapes` (pass B only): dropped, pass A held.
- `user WORKS_ON night-with-the-oyster-fleet` (pass B only): dropped, pass A held.
- `user WORKS_ON profile-of-a-shearing-gang-cook` (pass B only): dropped, pass A held.
- `user WORKS_ON school-with-one-pupil` (pass B only): dropped, pass A held.
- `user WORKS_ON six-hours-in-a-magistrates-waiting-room` (pass B only): dropped, pass A held.
- `user WORKS_ON wharf-strike-told-by-the-tea-lady` (pass B only): dropped, pass A held.
- `user WORKS_ON what-the-dredger-found` (pass B only): dropped, pass A held.

### sel-275-list-long-recorded-wherever-i-travel (F1 0.00, adjudicated whole)

- Pass A: read each sound as a habit with the cadence of every trip.
- Pass B: read each sound as an interest.
- Ruling: 'Wherever I travel' gives the recording a cadence, each trip, so each item is HAS_HABIT under its own phrase (List rules, N1); the utterance states no interest beyond the habit.
- `user HAS_HABIT cattle-grids` (pass A only): kept, pass A held.
- `user HAS_HABIT church-bells` (pass A only): kept, pass A held.
- `user HAS_HABIT ferry-horns` (pass A only): kept, pass A held.
- `user HAS_HABIT kettle-whistles` (pass A only): kept, pass A held.
- `user HAS_HABIT market-traders` (pass A only): kept, pass A held.
- `user HAS_HABIT rain-on-tin-roofs` (pass A only): kept, pass A held.
- `user HAS_HABIT screen-doors` (pass A only): kept, pass A held.
- `user HAS_HABIT shearing-handpieces` (pass A only): kept, pass A held.
- `user HAS_HABIT station-announcements` (pass A only): kept, pass A held.
- `user HAS_HABIT tui-at-dawn` (pass A only): kept, pass A held.
- `user INTERESTED_IN cattle-grids` (pass B only): dropped, pass A held.
- `user INTERESTED_IN church-bells` (pass B only): dropped, pass A held.
- `user INTERESTED_IN ferry-horns` (pass B only): dropped, pass A held.
- `user INTERESTED_IN kettle-whistles` (pass B only): dropped, pass A held.
- `user INTERESTED_IN market-traders` (pass B only): dropped, pass A held.
- `user INTERESTED_IN rain-on-tin-roofs` (pass B only): dropped, pass A held.
- `user INTERESTED_IN screen-doors` (pass B only): dropped, pass A held.
- `user INTERESTED_IN shearing-handpieces` (pass B only): dropped, pass A held.
- `user INTERESTED_IN station-announcements` (pass B only): dropped, pass A held.
- `user INTERESTED_IN tui-at-dawn` (pass B only): dropped, pass A held.

### sel-280-temporal-seasons-and-tenure (F1 1.00)

- Ruling: Entities only; the relations are agreed.
- Entities: the season the utterance calls 'the second' has no id that is its own phrase, so N1 lists both forms, B's alternative held.

## M66: Rule 23's form

Lead ruling M66 (12 Nov): where the production prompt is explicit, it governs the gold. Rule 23
reads: "A name that only modifies another name gets no entity of its own, unless a list's lead-in
carries it: in "the Ferrowind tuning guide" the entity is "ferrowind-tuning-guide", and Ferrowind
gets no entity." So on that shape Rule 23's form, the name kept together with the noun it
modifies in one id, replaces the N14 reading that rulings above took (the id is the name alone).
Each id below is renamed in the probe's entities and in every relation endpoint that names it;
nothing else in the probe changes, and the probe's ruling above stands except for the id.

- `sel-025-list-dash-dispensary-software`: `access` -> `access-database`, in its entity and 2 relation endpoints. Rule 23: the name only modifies the noun after it, and no list lead-in carries it.
- `sel-096-list-dash-monday-first-things`: `ampere-atlas` -> `ampere-atlas-dashboard`, in its entity and 2 relation endpoints. Rule 23: the name only modifies the noun after it, and no list lead-in carries it.
- `sel-115-multi-grid-scientist-utrecht`: `gridcast` -> `gridcast-forecast`, in its entity and 1 relation endpoint. Rule 23: the name only modifies the noun after it, and no list lead-in carries it.
- `sel-161-multi-sous-chef-club-newsletter`: `second-stove` -> `second-stove-newsletter`, in its entity and 1 relation endpoint. Rule 23: the name only modifies the noun after it, and no list lead-in carries it.
- `sel-096-list-dash-monday-first-things`: `ampere-atlas-dashboard IS_A dashboard` dropped. Rule 25: no IS_A edge when the kind is part of the name, and the kind is now part of the entity's own id.

Kept as ruled, counted unsure, where Rule 23 is not explicit:

- `sel-002-list-dash-planner-parts`: a list's lead-in carries the name, and Rule 23 does not say which id the name takes there.
- `sel-019-list-long-soak-run-figures`: a list's lead-in carries the name, and Rule 23 does not say which id the name takes there.
- `sel-099-list-numbered-weekly-numbers`: a list's lead-in carries the name, and Rule 23 does not say which id the name takes there.
- `sel-106-list-checkbox-release-checklist`: a list's lead-in carries the name, and Rule 23 does not say which id the name takes there.
- `sel-112-list-long-monthly-backtest-error`: a list's lead-in carries the name, and Rule 23 does not say which id the name takes there.
- `sel-152-list-checkbox-newsletter-month`: a list's lead-in carries the name, and Rule 23 does not say which id the name takes there.
- `sel-165-list-dash-dive-bag-kit`: a list's lead-in carries the name, and Rule 23 does not say which id the name takes there.
- `sel-188-list-dash-android-app-parts`: a list's lead-in carries the name, and Rule 23 does not say which id the name takes there.
- `sel-024-scope-notes-and-tone`: the noun after the name is plural and names files of a format, not one thing as in Rule 23's example.
- `sel-092-coref-editor-and-deputy`: the name does not only modify: the utterance refers back to it as an organisation, the target of both WORKS_AT edges.

Ids renamed: 4, in 4 probes; relation endpoints renamed: 6; relations dropped under Rule 25: 1. Kept and counted unsure: 10.

## Totals

- Probes with a difference adjudicated: 139; still to adjudicate: 0.
- Relations the passes differed on, adjudicated: 425.
- Probes adjudicated whole (F1 under 0.61): 53.
- Outcomes: pass A held 227, pass B held 198, neither held 2.
- Shared relations whose attributes differed: 54.
