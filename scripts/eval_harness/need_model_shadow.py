"""Fit the shadow-only need model and write its record (plan mist-read-path v10, A55).

H23 ran this script on 16 Nov, in image mist-read-path-h4:be7be966, to fit the record shipped as
`backend/memory/need_model.json`; tests/unit/memory/test_need_model.py pins that record.

H18 (16 Nov, image mist-read-path-h4:be7be966) found no eligible gate variant and wrote no need
model. Raj's ruling of 16 Nov was to ship the gate in shadow mode: each fetch decided and traced,
then served as open. A shadow gate still needs a record for its key in need_model.json. At H4 (17
Nov) the shadow gate put 121 of 178 dev idle fetches past their deadline, so the gate ships `off`
(Raj, 17 Nov) and the record ships unread until a run sets `shadow` or `on`. This fits the
one variant plan v10 names: the `shared` gate embedder (the fact embedder, whose query vector the
fetch shares), the form H18's cross-validation chose for it, recall target 0.95 and band_k 1. No
eligibility rule chose it, and its record's `source` starts with `SHADOW_ONLY`.

The fit is the grid's own `fit_gate` on the same bank, dev probes and follow-ups, so its form and
thresholds are H18's up to float summation order. It refuses to write a record whose embedder or
form differs from H18's report, or whose thresholds differ from H18's by more than a relative
1e-6.

Reads the dev probes (probes.jsonl, through Fixture.load), the dev follow-ups, the bank and H18's
report; it never opens heldout_probes.jsonl. The fact embedder runs on the CPU.

Usage, from the repository root (`/app` in the image), so `backend` imports:
    python -m scripts.eval_harness.need_model_shadow --h18 PATH --output PATH

Exit 0: the record is written and loads by its key. Exit 2: an input is missing or malformed, or
H18's report holds a choice. Exit 3: the fit is not H18's, and nothing is written.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

from backend.errors import MemoryConfigError
from backend.memory.need_gate import SHADOW_ONLY, NeedKey, load_bank, load_models, select_model
from scripts.check_models_lock import LockError, parse_lock
from scripts.eval_harness.memory_recall import FIXTURE_DIR
from scripts.eval_harness.memory_recall_run import (
    DEFAULT_MODELS_LOCK,
    Fixture,
    SentenceTransformerEmbedder,
)
from scripts.eval_harness.memory_selection_grid import (
    FOLLOWUP_FILE,
    SHARED,
    GateEmbedder,
    fit_gate,
    load_followups,
    need_model_record,
)

FACT_EMBEDDER = "/opt/models/all-MiniLM-L6-v2"
RECALL_TARGET = 0.95
BAND_K = 1
REL_TOL = 1e-6
H18_DATE = "2026-11-16"


def h18_expected(report: dict) -> tuple[dict, str, dict]:
    """H18's `shared` embedder, its form, and its variant at RECALL_TARGET and BAND_K."""
    if report.get("choice") is not None:
        raise ValueError("H18's report holds a choice; the shadow record is for a run with none")
    embedder = report["cv"][SHARED]["embedder"]
    gate = report["gates"][SHARED]
    variant = next(
        (
            v
            for v in gate["variants"]
            if v["recall_target"] == RECALL_TARGET and v["band_k"] == BAND_K
        ),
        None,
    )
    if variant is None:
        raise ValueError(f"H18's report has no shared variant at {RECALL_TARGET}, k {BAND_K}")
    return embedder, gate["form"], variant


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--h18", type=Path, required=True, help="H18's selection grid report")
    parser.add_argument("--output", type=Path, required=True, help="the need model file to write")
    args = parser.parse_args()
    try:
        report = json.loads(args.h18.read_text(encoding="utf-8"))
        h18_embedder, h18_form, h18_variant = h18_expected(report)
        fixture = Fixture.load(FIXTURE_DIR)
        followups = load_followups(FIXTURE_DIR / FOLLOWUP_FILE)
        bank = load_bank()
        entries = parse_lock(DEFAULT_MODELS_LOCK)
    except (OSError, ValueError, KeyError, TypeError, MemoryConfigError, LockError) as exc:
        print(f"h23 exit 2, input: {type(exc).__name__}: {exc}")
        return 2
    entry = next((e for e in entries if e.name == Path(FACT_EMBEDDER).name), None)
    if entry is None:
        print(f"h23 exit 2, {DEFAULT_MODELS_LOCK} has no entry for {FACT_EMBEDDER}")
        return 2
    lock_embedder = {"name": entry.name, "revision": entry.revision}
    print(f"h23 embedder {lock_embedder} (H18 {h18_embedder})")
    if lock_embedder != h18_embedder:
        print("h23 exit 3, the image's fact embedder is not the one H18 fit; nothing written")
        return 3

    fact = SentenceTransformerEmbedder(FACT_EMBEDDER, device="cpu")
    embedder = GateEmbedder(SHARED, entry.name, entry.revision, fact)
    fit = fit_gate(embedder, bank, fixture.probes, followups)
    form = fit.form.describe() if fit.form is not None else None
    print(f"h23 bank {bank.sha256} rows {len(bank.exemplars)}")
    print(f"h23 form {form} (H18 {h18_form})")
    if form != h18_form:
        print("h23 exit 3, the fit's form is not H18's; nothing written")
        return 3
    variant = next(
        v for v in fit.variants if v.recall_target == RECALL_TARGET and v.band_k == BAND_K
    )
    for name in ("theta_low", "theta_high"):
        got = getattr(variant, name)
        want = float(h18_variant[name])
        print(f"h23 {name} {got!r} (H18 {want!r})")
        if not math.isclose(got, want, rel_tol=REL_TOL, abs_tol=0.0):
            print(f"h23 exit 3, {name} is more than a relative {REL_TOL:g} from H18's")
            return 3

    today = dt.datetime.now(dt.UTC).date().isoformat()
    source = (
        f"{SHADOW_ONLY} H23 {today}: shared {form} at recall {RECALL_TARGET}, band_k {BAND_K}; "
        f"no gate met the eligibility rule at H18 ({H18_DATE}) and no rule chose this one "
        "(plan mist-read-path v10)"
    )
    record = need_model_record(fit, variant, source=source)
    args.output.write_text(json.dumps({"records": [record]}, indent=2) + "\n", encoding="utf-8")
    key = NeedKey(bank.sha256, entry.name, entry.revision, embedder.query_prefix)
    model = select_model(load_models(args.output), key)
    print(f"h23 key {model.key} w {len(model.params['w'])} b {model.params['b']!r}")
    print(f"h23 source {model.source}")
    print(f"h23 wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
