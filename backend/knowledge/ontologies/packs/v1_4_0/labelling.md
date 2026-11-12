- **N2 Goal ids.** A Goal id is the goal's verb phrase as the utterance words it, articles and
  possessive pronouns dropped: `get-pilots-license`, `cut-screen-time`, `pay-off-mortgage`. The
  60-probe set has no Goal; this follows the goal ids both passes agreed on (`ship-saltmere` in
  dvl-09, `learn-to-sail` in ho-091). The H8 check marked ho-004's goal ids good.

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

**N17 Named teams.** A named team, club or community that the utterance places a person in is
an `Organization`. A person is placed in it when:

- they are on it, are in it, joined it, or run or lead it;
- the speaker's possessive names it ("our billing team", "my billing team"), which places the
  speaker;
- they hold a title that names the team with its group noun ("the billing team lead"); a title
  with no group noun ("our design lead") names no team;
- they are listed under a heading, lead-in or parent bullet that is its name.

*Named:* a team is named when the utterance gives it a word that tells it apart from other
teams, what it does or its proper name. A heading, lead-in or parent bullet also names a team
when it carries its own group noun ("Tooling team:") or sits under a lead-in that carries a
group noun ("My team at <employer>:" over "Tooling"). A describing word (new, old, small,
whole, current, remote, other) does not name a team. Nor does the employer's name alone: "the
<employer> team" is the employer, with `WORKS_AT`, and no team entity.

*Id:* the phrase under N1. The group word is kept when the utterance writes it
(`billing-team`); a heading's own word stands alone (`tooling`).

*Edges:* each person placed in it gets `MEMBER_OF` it, including the one who runs or leads it.
A named employer is a separate `Organization` that each of them `WORKS_AT`. The team gets no
`PART_OF` to the employer.

*No entity and no `MEMBER_OF` for:* a group noun with no name of its own ("my team", "the
group", "my new team", "my team lead"), as an item or as a lead-in; a role or job title with no
group noun; a relationship category (colleagues, friends, family, household), which N16 already
leaves without an entity.

*What stays elsewhere:* a team's tools and decisions stay on the user (N8, N3). N4 still covers
a team name that only modifies another name and places no one in it. An employer that only
modifies a team's name is still the employer, with `WORKS_AT`.

From A32(a) (Raj's Ruling 21, 2026-11-10): the ontology types a team as an `Organization`
("a company, team, institution, community, or open-source project", `v1_0_0.py:494`), and
dev-lists already labels team membership this way (dvl-06, dvl-12, dvl-33). Settled on
`ext-18` ("at Stripe on the payments team": `user MEMBER_OF payments-team` beside
`user WORKS_AT stripe`), `ext-33` and dvl-26 (Relabel log). Beyond the prompt until the
prompt's group 3 carries it.

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
The held-out set is relabelled under the contract's owner and own-phrase clauses and, like the
60-probe set, whose Event ids are not yet relabelled, leaves Event dates to the scorer's
event-date alignment. `test_gold_sets.py` checks the rule
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
`canonical_metric_id` in `backend/knowledge/extraction/canonical_id.py`). Write the value as the
utterance states it, spelled as the Metric rail spells it (plan v11 D16,
`canonical_metric_value`): a digit separator is dropped (`12,000` is `12000`), a decimal point
is kept as a hyphen and trailing fractional zeros are dropped, so "3.1 GB peak memory" is
`3-1-gb-peak-memory`, "0.2 percent" is `0-2-percent-...` and "3.80 euros" is `3-8-euros-...`.
The unit takes gold's one spelling (D16, `UNIT_SPELLINGS`): `percent` (for `%`, `pct`),
`ms` (for `msec`, `milliseconds`), `seconds` (for `s`, `sec`), `requests-per-second` (for
`rps`, `requests/sec`), `tokens-per-second` (for `tokens/sec`) and `gb` (for `gigabytes`);
any other unit keeps its slug (`euros-per-hour`, `matches-per-minute`). The
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
The 60-probe `ext-13` ("Neovim as my daily editor") had no `neovim IS_A editor`; it is
relabelled under this rule with the `Concept` `editor` and `neovim IS_A editor`, since "daily"
qualifies the use and is not part of the kind (Relabel log).

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

**N12 Flipped preferences.** When the utterance states the earlier preference as well as the
new one ("I used to love X, now it slows me down", "I don't enjoy X anymore"), it carries both:
`user PREFERS X` with `assertion_kind` `cease`, and the new `user DISLIKES X` as an assertion,
the same pair as a replacement (ext-53, ext-54). Neither edge gets a date: "now" contrasts the
present with the past and states no day the preference changed. From the H8 mark on ho-081 (the
user used to love it until now; going forward it is a dislike). This reverses the first
batch's correction that kept only the new preference. Beyond the prompt: Rule 6
(`prompts.py:54`) gives "don't like X anymore" as `DISLIKES` assert only. The 60-probe set has
no flipped preference (ext-15 states no earlier one).

### Habits and incidental scope

`HAS_HABIT` needs a stated cadence ("every Wednesday", "twice a day"). A continuous activity
with only a start point takes the matching stative predicate with `valid_from` (prompt Rule 17,
`prompts.py:65`; ext-55 to ext-58). A trailing phrase that only scopes the fact is context,
not a structural edge. "Recommended X for the new service" adds no `STRATEGY_FOR`, but "I use X
for Y" is itself the fact and gets `USED_FOR` (Rule 18, `prompts.py:66`; ext-01, ext-31).

  except under a lead-in that names goals, where it is `HAS_GOAL`.

## Register

Each convention above, by its label, and what carries it in this pack: a prompt rule by its number
and the id of the prompt fragment that holds it, any other rule by its id in `conventions.py`.
Beside a rule of the pack, a row may name a rule of the prompt's engine core by its number, or a
rule of the engine by the name of its function. The first four rows are the first rows of the
extraction contract ADR's v14 amendment (ADR-054, 2026-11-11): their Convention cells are the
amendment's, word for word. Each of them states its convention in full: a labeller applies the
row, never a rule's code, and where the row is narrower than a line above it, the row governs.
The row governs too where a line above says a convention is beyond the prompt and the row names a
prompt rule that carries it. The last seven rows list the generic conventions of the gold guide
(`data/gold/extraction/LABELLING.md`) that a rule of this pack carries, each under the key of its
row in that guide's register.

| Convention | Carrier | Class |
|---|---|---|
| Named teams (N17; Rule 34): each person placed in a named team also `WORKS_AT` the employer the utterance names for that team | Rule 34 in `prompt-rules-11-to-37` | prompt rule |
| Durable third-party facts (Rule 19): in a user-scope turn, the user `KNOWS_PERSON` each person the utterance places in a named team, whether or not the user is in it, since a placement is a durable fact about that person. The user gets no edge to themself | Rule 19 in `prompt-rules-11-to-37` | prompt rule |
| Named teams (N17; Rule 34): a team's `PART_OF` its employer is not stored. This covers a team `Organization` to its employer only; any other `PART_OF` is kept | Rule 34 in `prompt-rules-11-to-37` | prompt rule |
| List rules, unchecked box: a `[ ]` item yields no edge, except `HAS_GOAL` when its lead-in, or any heading or parent bullet above it, contains one of these goal words, as a whole word in any case, singular or plural: goal, objective, resolution, aim, aspiration, target, and the phrase "bucket list". An unchecked itinerary item is a plan and yields no edge | Rule 29 in `prompt-rules-11-to-37`, and `unchecked-goals` | prompt rule and pre-LLM rule |
| N2 Goal ids | Rule 22 in `prompt-rules-11-to-37` | prompt rule |
| The user anchor | Rule 1 in `prompt-rules-1-to-3`, and `reserved-mist-identity` | prompt rule and payload rail |
| N8 The team's tools | Rule 35 in `prompt-rules-11-to-37` | prompt rule |
| Durable third-party facts | Rules 17 and 19 in `prompt-rules-11-to-37` | prompt rule |
| N16 A relative named by the relation | Rule 24 in `prompt-rules-11-to-37` | prompt rule |
| N7 Third-party events | Rule 32 in `prompt-rules-11-to-37` | prompt rule |
| N17 Named teams | Rule 34 in `prompt-rules-11-to-37` | prompt rule |
| Event and metric ids | Rules 11, 26 and 27 in `prompt-rules-11-to-37`, `event-date` and `metric-id` | prompt rule and id rail |
| N10 Personal milestones | Rule 36 in `prompt-rules-11-to-37` | prompt rule |
| N3 Decisions | Rule 30 in `prompt-rules-11-to-37` | prompt rule |
| N5 Stated places and organisations | Rule 31 in `prompt-rules-11-to-37` | prompt rule |
| N13 Topic links | Rule 33 in `prompt-rules-11-to-37` | prompt rule |
| The Abstraction fallback | Rules 14 and 20 in `prompt-rules-11-to-37`, and `prompt-constraints` | prompt rule |
| N14 Kind nouns | Rule 25 in `prompt-rules-11-to-37` | prompt rule |
| Date entities | Rules 13 and 16 in `prompt-rules-11-to-37`, Rule 5 of the engine core, and `backend.knowledge.extraction.temporal.TemporalResolver.resolve` | prompt rule and engine rule |
| N11 Goal windows | Rule 37 in `prompt-rules-11-to-37` | prompt rule |
| N12 Flipped preferences | Rule 6 in `prompt-rule-6` | prompt rule |
| Habits and incidental scope | Rules 17 and 18 in `prompt-rules-11-to-37` | prompt rule |
| Row schema | Rules 2 and 3 in `prompt-rules-1-to-3`, `prompt-constraints`, `technology-version`, `metric-id`, `pre-canon-aliases` and `static-aliases` | prompt rule and id rail |
| N1 Id surface | Rule 2 in `prompt-rules-1-to-3`, and `n1-exempt` | prompt rule and id rail |
| N15 An activity keeps its object | Rule 21 in `prompt-rules-11-to-37` | prompt rule |
| N4 Name modifiers | Rule 23 in `prompt-rules-11-to-37` | prompt rule |
| N6 Type by its test | Rules 14 and 20 in `prompt-rules-11-to-37`, `prompt-constraints` and `prompt-examples` | prompt rule |
| Assertion kinds | Rule 6 in `prompt-rule-6` | prompt rule |
| List-shaped probes (new) | Rule 29 in `prompt-rules-11-to-37` | prompt rule |
