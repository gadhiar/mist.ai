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
- Ruling: N1 and N4: 'Lanternfish' only modifies 'route planner', and the project is the route planner, so its id is the utterance's whole phrase; the DEPENDS_ON edges follow that id. The USED_FOR edges stand as both passes wrote them.
- `lanternfish DEPENDS_ON cassandra` (pass A only): dropped, pass B held.
- `lanternfish DEPENDS_ON ktor` (pass A only): dropped, pass B held.
- `lanternfish DEPENDS_ON nats` (pass A only): dropped, pass B held.
- `lanternfish-route-planner DEPENDS_ON cassandra` (pass B only): kept, pass B held.
- `lanternfish-route-planner DEPENDS_ON ktor` (pass B only): kept, pass B held.
- `lanternfish-route-planner DEPENDS_ON nats` (pass B only): kept, pass B held.

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

## Totals

- Probes with a difference adjudicated: 42; still to adjudicate: 97.
- Relations the passes differed on, adjudicated: 121.
- Probes adjudicated whole (F1 under 0.61): 17.
- Outcomes: pass A held 44, pass B held 77, neither held 0.
- Shared relations whose attributes differed: 9.
