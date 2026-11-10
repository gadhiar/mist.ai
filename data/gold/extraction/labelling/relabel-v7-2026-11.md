# Held-out relabel under the extraction contract (T31, v7), 2026-11-09

This file is for Raj. It lists every id the held-out relabel changed in
`heldout-2026-11.jsonl`, and every id it left alone that a rule might have changed. The rules are the
extraction contract ADR (adopted 2026-11-09 with Event id rule E-b; Decisions 3 and 4), LABELLING.md
(N1, "Event and metric ids"), and the sub-lead's plan v10 course correction. That correction applies
the id rule to every held-out id, not only to Events:

1. An id that a convention with a carrier decides stays as that convention builds it (`user`, ISO
   Date ids, value-first Metric ids, Rule 3, N16, N14, N2, N15, N4, the Event rule). N1's habit
   clause is not one of these and decides no id in this phase.
2. Any other id that differs from the utterance's own phrase is relabelled to that phrase, and its
   edges follow.
3. `alternatives` only where two forms fit and neither is the utterance's own phrase.

Event dates are left to the scorer's event-date alignment kind. No `-YYYY-MM-DD` suffix was added
or removed. Every edge was kept with its endpoints renamed (71 endpoints, the `OCCURRED_ON` edges
included). Touched rows were re-sorted the way every gold row is sorted (`user` first, then by id;
edges by source, predicate, target). No tag, utterance, type, predicate or edge count changed. The
worker opened nothing under `.local/` or `data/ceiling/`.

## How the phrase was read

- "The utterance's phrase" is read as T30's Event-rule check (`test_gold_sets.py`,
  `_event_rule_faults`) reads it: a run of the utterance's words. An id that is already such a run
  was kept, even where a longer phrase also fits (for example a head noun without its modifier).
  Whether a shorter run counts as "differing from the own phrase" is a question for Raj (below).
- Where an id was not a run, the new id is the run that carries the old id's content words,
  extended to the owner, the place and any number that identifies the event.
- Events drop every article (the Event rule, as T30 did: `work-from-cafe`). Other ids drop only a
  leading article or possessive pronoun (N1, and the N1 rail), so a mid-phrase article stays, as
  `read-one-paper-a-week` already had it.
- Quantifier determiners (`every`, `any`, `one` before a noun) were treated like articles.
- A non-possessive name before an event noun (an organisation that holds the event) was kept as
  the event's named owner. N4 keeps it out of the entity list, not out of the id; `lisbon-summit`
  already kept it.

## Events (40 in held-out; 25 changed)

Owner added (8): ho-018 `alumni-dinner` to `ashcombe-university-alumni-dinner`; ho-103 `wedding`
to `freya-gets-married-in-bruges`; ho-107 `outage-drills` to `corvane-outage-drills`; ho-114
`hackathon` to `kelderwick-rail-hackathon`; ho-115 `holiday-market` to
`wrenfield-bakery-holiday-market`; ho-120 `audit` to `pelloria-optics-audit`; ho-121 `retirement`
to `imre-retired`; ho-123 `interview` to `saoirse-interviewed-at-marrowby-press`. ho-103, ho-121
and ho-123 also changed their phrase.

Own phrase changed (19, the three above included): ho-007 `openstreetmap-tile-bundling` to
`bundle-openstreetmap-tiles`; ho-013 `cat-adoption` to `adopted-cat`, `move` to
`moved-to-utrecht`; ho-019 `beta-opening` to `beta-opened`; ho-033 `next-architecture-review` to
`next-review`, `session-store-replacement` to `valkey-replaces-in-house-session-store`; ho-061
`database-replica-addition` to `add-second-database-replica`; ho-097
`tallyroot-general-availability` to `general-availability-for-tallyroot`; ho-103 (above); ho-104
`sunset-decision` to `sunset-legacy-billing-endpoints`; ho-108 `promotion` to `was-promoted`;
ho-111 `lease-signing` to `signed-lease`, `loan-approval` to `approved-loan`; ho-112
`ledgerwick-500-stars` to `ledgerwick-repo-passed-500-stars`; ho-118 `hearthline-app-launch` to
`version-2-of-hearthline-app-shipped`, `hardware-kit-launch` to `hardware-kit`; ho-121 (above);
ho-123 (above), `job-offer` to `offered-editor-role`.

Date word removed (1): ho-200 `thursday-release-train` to `release-train` (a weekday name).

Notes:
- ho-108: `promoted` alone is a word the production prompt spells (Rule 11's milestone list), which
  `test_no_heldout_entity_id_occurs_in_the_production_prompt` refuses, so the id is the verb group
  as worded. "Promoted" is milestone-class, so N7's owner does not apply, and the owner is not in
  the phrase's run.
- ho-123 `offered-editor-role`: the owner appears in the phrase only as a pronoun, so no named
  owner is in the run.
- ho-103: the place follows "on <ISO date>". The id drops the whole date phrase and keeps the place
  as worded. The T30 Event-rule check drops the ISO date but not its preposition, so it would call
  this id "not the utterance's phrase". That check runs on dev-lists only.
- Unchanged (15): ho-019 `public-launch`, ho-025 `rotterdam-meetup`, ho-033 `architecture-review`,
  ho-061 `outage` and `postmortem`, ho-080 `open-night`, ho-086 `launch`, ho-095 `lisbon-summit`,
  ho-098 `public-talk`, ho-099 `data-migration`, `schema-freeze` and `planning-session`, ho-100
  `rebrand`, ho-116 `steering-meeting`; ho-035 `physio`. 16 of the 40 carry a single-day
  `OCCURRED_ON` and no suffix, which the event-date alignment kind bridges.

## Other ids relabelled by clause 2 (23)

ho-002 `xml-config` to `config-written-in-xml`; ho-032 `sqlite-internals-guide` to
`guide-to-sqlite-internals`; ho-035 `swimming` to `swim`; ho-040 `commit-messages` to `messages`;
ho-054 `mandarin-practice` to `practise-mandarin`; ho-058 `azure-devops-boards` to `boards`; ho-060
`resource-tags` to `resource-tag`; ho-063 `rollback` to `roll-back`; ho-126
`crdt-garbage-collection-paper` to `paper-on-crdt-garbage-collection`; ho-161 `voice-transcription`
to `transcribes-voice`; ho-163 `directness` to `to-the-point`, `dryness` to `dry`; ho-169
`curiosity` to `curious`; ho-170 `whiteboard-transcription` to
`read-a-whiteboard-photo-and-turn-it-into-notes`; ho-171 `vault-layer-adr` to
`adr-on-the-vault-layer`; ho-172 `invoicing-app` to `tauri-app-for-invoicing`; ho-179
`branch-names` to `branch-name`, `ticket-number-prefix` to `starts-with-the-ticket-number`; ho-182
`crash-recovery` to `survives-a-crash-mid-transaction`; ho-183 `black-coffee` to `coffee-black`,
`short-commit-messages` to `commit-messages-short`; ho-194 `retrying-clients` to
`client-that-retries-network-calls`; ho-196 `kubernetes-namespaces` to `kubernetes-namespace`.

ho-035 and ho-054 are the habit cases that dev-lists handles with `alternatives` (dvl-08). Under
the course correction the habit clause decides no id, so the verb as worded is the id, with no
alternative.

## Relabelled to its clause-1 convention (1)

ho-048 Goal `5k-under-25-minutes` to `get-5k-under-25-minutes`: N2 builds a Goal id from the goal's
verb phrase, and the old id lacked the verb.

## Alternatives

Added (1): ho-130 `tromso` gained `troms`. The utterance's own word, "Tromso" with a slashed o, has
no canonical-id form of its own: `canonical_id` drops the letter and gives `troms`, while a model
that transliterates writes `tromso`. Neither is the utterance's phrase, so both are listed.
Removed (0): held-out had none. No edge has `predicate_alternatives`, and no edge has
`beyond_prompt`.

## Kept by clause 1 though they differ from the utterance's phrase (15)

- Value-first Metric ids (10): ho-004 `two-hours-a-day-screen-time`; ho-015
  `14-ms-median-tile-render`, `18-seconds-cold-start`, `210-mb-peak-ram`; ho-048
  `25-minutes-5k-time`; ho-053 `300-ms-first-paint-reduction`,
  `40-percent-page-weight-reduction`; ho-066 `62-mb-build-size`, `900-ms-cold-start`; ho-164
  `64-gb-ram`.
- N2 Goal ids (3): ho-004 `cut-screen-time`, ho-091 `learn-to-sail` and ho-187 `pay-off-mortgage`,
  the forms LABELLING's N2 names as its own examples.
- N14 kind noun (1): ho-009 `shell`, the form N14 names for a heading "Shells" over zsh.
- N4 name modifier (1): ho-038 `ticket-gates`, where the lead-in carries the organisation as its
  own entity (N4's cited case).

## Left unchanged: no rule decides (1)

ho-196 Convention `owning-team-naming`. The utterance words the convention only as a split phrase
("name ... after the team that owns it"), around the entity it applies to. No run of the
utterance's words carries it, and N1 does not say which part to keep. The id is unchanged until
Raj rules.

## Questions for Raj

1. Should a non-Event id that is a run of the utterance's words, but shorter than its full noun
   phrase (a head without a modifier), count as differing from the own phrase? This relabel kept
   such ids, as T30's check does.
2. ho-196's convention: which form should it have, if any?
3. Should Events also keep mid-phrase articles, as other ids do here? They follow T30, which drops
   them.
