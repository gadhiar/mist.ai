# Labelling rows of the pet fixture pack

### Crews

**N1 Serving under the skipper.** A deckhand who crews with a skipper also serves under that
skipper. The prompt states it as "F1. Each deckhand placed in a crew also SERVES_UNDER its
skipper."

**N2 A skipper's rank.** A skipper outranking their own crew is not stored.

### Unchecked items

An unchecked `[ ]` item is a plan and yields no edge, unless its lead-in or a heading above it
names a drill or a watch list.

### Pets

**N18 Pet rosters.** Under a roster lead-in that names pets, the user has each pet, unless its
line names another owner.

## Register

Each convention above, by its label, and the rules of this pack that carry it.

| Convention | Carrier | Class |
|---|---|---|
| N1 Serving under the skipper | Rule F1 in `crew-rule-text`, and `crew-fan-out` | prompt rule and relation rail |
| N2 A skipper's rank | `skipper-outranks` | relation rail |
| Unchecked items: a `[ ]` item yields no edge, unless its lead-in, or any heading or parent bullet above it, contains one of these words, as a whole word in any case, singular or plural: drill, and the phrase "watch list" | `unchecked-items` | pre-LLM rule |
| N18 Pet rosters | Rule N18 in `pet-roster-text`, and `pet-roster` | prompt rule and relation rail |
