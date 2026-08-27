"""MIS-131: pin the canonical-form exclusion set, before anyone sees a red diff.

Every field `canonical_serialize` excludes is a field the `live == rebuilt`
gate cannot see. That is correct for wall-clock stamps and necessary for
embeddings, and it is also the single lever that can turn a failing gate green
without fixing anything.

The closure design names the pressure explicitly: the curation scheduler,
Stage-9 derivation and seed-apply defects all push the gate RED, and the cheap
way to make a red diff go away is to widen this set. That is how a gate quietly
stops proving anything. This test exists so widening it is a deliberate,
reviewed act rather than a one-line edit during a debugging session.

**These tests are tripwires, not correctness checks.** They pin accidents on
purpose. A legitimate change to the exclusion set SHOULD fail them; the
required response is to update the expectation here in the same commit, with a
sentence saying what the gate can no longer see and why that is acceptable.

The union test is derived rather than enumerated, per `tests/CLAUDE.md`
("Derived beats enumerated whenever the enumeration can go stale"): it
introspects the module for exclusion frozensets, so a NEW one added later fails
until it is pinned here. An enumerated test would silently ignore it -- which is
exactly the hole a pinning test exists to close.
"""

from __future__ import annotations

from backend.knowledge import canonical_serialize as cs


class TestPinnedByExactEquality:
    """Each set, by value. Widening any of them weakens the closure proof."""

    def test_audit_fields(self):
        assert (
            frozenset({"created_at", "updated_at", "derived_at", "first_seen_at", "last_seen_at"})
            == cs.AUDIT_FIELDS
        )

    def test_epoch_stamp_fields(self):
        assert (
            frozenset({"ontology_version", "extraction_version", "model_hash"})
            == cs.EPOCH_STAMP_FIELDS
        )

    def test_derived_artifact_fields(self):
        """`embedding` is excluded because it is huge and float-noisy.

        The cost is documented in `seed/gates.py:264-268`: a canonical form is
        byte-identical whether embeddings are present, absent, or all-zero, so
        a seed-apply that skips the backfill produces a graph nothing can
        retrieve from AND certifies clean. That is why MIS-130 carries a
        separate presence-and-dimension assertion instead of relying on this.
        """
        assert frozenset({"embedding"}) == cs.DERIVED_ARTIFACT_FIELDS

    def test_node_only_excluded_fields(self):
        """Node `confidence` only. Edge `confidence` is deliberately compared.

        Edge confidence is reinforce-only (a monotonic max on write) and
        therefore log-deterministic. Node confidence is additionally written by
        `ConfidenceDecayJob` off the wall clock, which is the whole reason for
        the asymmetry -- and why it becomes un-excludable once the scheduler is
        off (MIS-131's own exclusion decision, closed 2026-08-26).
        """
        assert frozenset({"confidence"}) == cs.NODE_ONLY_EXCLUDED_FIELDS


class TestTheSetIsWhollyPinned:
    def test_no_exclusion_set_escapes_this_file(self):
        """Derived, so a NEW exclusion frozenset cannot slip in unpinned.

        Enumerating the four sets above would pass unchanged if a fifth were
        added tomorrow -- the gate would lose visibility of a field and no test
        would notice. Introspecting the module instead means the addition
        itself fails here until someone writes down what it costs.
        """
        found = {
            name
            for name, value in vars(cs).items()
            if isinstance(value, frozenset) and not name.startswith("_")
        }
        assert found == {
            "AUDIT_FIELDS",
            "EPOCH_STAMP_FIELDS",
            "DERIVED_ARTIFACT_FIELDS",
            "NODE_ONLY_EXCLUDED_FIELDS",
        }, (
            "A canonical-form exclusion set was added or renamed. Pin it above "
            "with a docstring saying what the live == rebuilt gate can no "
            "longer see, and why that is acceptable."
        )

    def test_every_pinned_set_is_actually_consulted(self):
        """A constant nothing reads is documentation, not a guard.

        Pins that the exclusion sets are wired into `_canon_props` rather than
        merely declared -- the failure mode where a refactor stops consulting
        one and the pinning test above still passes.
        """
        for field in cs.AUDIT_FIELDS | cs.EPOCH_STAMP_FIELDS | cs.DERIVED_ARTIFACT_FIELDS:
            assert cs._canon_props({field: "x", "keep": 1}, is_node=True) == {
                "keep": 1
            }, f"{field} is pinned as excluded but survived _canon_props"

    def test_node_only_fields_are_excluded_on_nodes_and_kept_on_edges(self):
        """The asymmetry is load-bearing and easy to lose in a refactor."""
        for field in cs.NODE_ONLY_EXCLUDED_FIELDS:
            assert cs._canon_props({field: 0.9}, is_node=True) == {}
            assert cs._canon_props({field: 0.9}, is_node=False) == {field: 0.9}
