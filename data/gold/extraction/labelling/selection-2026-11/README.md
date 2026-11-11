# Selection set 2026-11: labelling files

The selection set is the probe set on which the extraction recall verdict is read. Its utterances
were written before any label, and its gold comes from two blind label passes and an adjudication
(plan v14, T53 to T56). This directory holds the files of that process. The label rules are in
`../../LABELLING.md`; nothing here restates them.

## Files

| File | What it is | Written by |
| --- | --- | --- |
| `../../selection-2026-11.jsonl` | The 280 probes, one per line in the gold row schema. `expected_entities` and `expected_relationships` stay empty until adjudication writes them. | Generation; adjudication fills the gold fields |
| `design.jsonl` | One row per probe: `tag`, `shape` (list probes) or `category` (prose probes), `persona`, `intended_relations`. The author's notes on how each probe was built. | Generation |
| `pass-a.jsonl` | The first blind label pass. | Pass A |
| `pass-b.jsonl` | The second blind label pass. | Pass B |
| `agreement.json` | Relation F1 and Cohen's kappa between the two passes, per probe and per group, from `scripts/eval_harness/gold_agreement.py`. | Adjudication |
| `adjudication.md` | The decision on each probe where the passes differ. | Adjudication |

Only `design.jsonl` and this file exist until the passes are written.

## The probes

- 240 list probes and 40 prose probes, tagged `sel-NNN-<slug>` and numbered from 001 without gaps.
- A list probe's slug starts `list-<shape>-`. The shapes and their counts are six times the split
  of `dev-lists-2026-11.jsonl`: dash 36, numbered 18, ref 48, plus 12, star 6, checkbox 12,
  nested 6, heading 12, table 6, paren 6, dot 6, mixed 6, lettered 6, long 60.
- A prose probe's slug starts with a `LABELLING.md` category other than `list`: multi, neg,
  temporal, coref, nofact, scope, fill.
- The utterances come from twelve synthetic personas. No persona, person or organisation in them
  is real, and none appears in the tuning sets or the held-out set; the guards in
  `tests/unit/eval_harness/test_gold_sets.py` check the overlap.

## Who may open what

- **A labeller (pass A or pass B)** opens `../../selection-2026-11.jsonl`, `../../LABELLING.md`
  and their own pass file. A labeller never opens `design.jsonl`: `intended_relations` is the
  author's expectation, and reading it would anchor the label. A labeller does not open the other
  pass, `agreement.json` or `adjudication.md`.
- **The adjudicator** opens both passes once both are handed back, and may then open
  `design.jsonl`.
- **A session that edits the extraction prompt, the extraction code or an extraction setting**
  opens neither the selection file nor anything in this directory, and reads no raw or scored
  file of a selection run. The set stays unseen by tuning so that its verdict is not fitted.
- Reports about this set give counts and agreement figures only: no utterance, name, id or tag
  slug.

## Pass-file format

`pass-a.jsonl` and `pass-b.jsonl` are JSON Lines files in this directory, one row per probe, in
the gold row schema of `LABELLING.md`:

```json
{"utterance": "...", "tag": "sel-001-...", "expected_entities": [...], "expected_relationships": [...]}
```

- `tag` and `utterance` are copied unchanged from the selection file.
- A pass holds all 280 tags, each once. Rows are keyed by tag and may be in any order.
- `expected_entities` and `expected_relationships` follow `LABELLING.md`, including its optional
  keys. A probe with no fact keeps both lists empty.
- The gold-set tests read a pass file as soon as it exists: schema, tags, ontology types and
  predicates, edge endpoints and type pairs. A failure prints tags and counts only.

## `design.jsonl`

- `shape` is one of the fourteen list shapes above; `category` is one of the seven prose
  categories. A row has one or the other.
- `persona` is an id such as `p01-...`; it groups the probes written from one invented person.
- `intended_relations` is the number of gold relations the author built the probe to carry, at
  most 15. It is a design bound, not a label: the gold count comes from the passes.
