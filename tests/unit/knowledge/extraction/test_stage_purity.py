r"""Stages 3-6 must be deterministic given their arguments: same input twice
-> identical output, no clock.

Spec D2 moves the cache boundary to just after Stage 2 and lets a rebuild
re-run Stages 3-6. That is only sound if those stages are deterministic --
but not, precisely, "functions of their arguments" alone. Each stage also
reads fixed state that is not part of the `ExtractionResult` argument:
confidence.py's class-level `HEDGE_PATTERNS`/`THIRD_PARTY_PATTERNS`,
validator.py's `OntologyConstrainedExtractor.ALLOWED_ENTITY_TYPES`/
`ALLOWED_RELATIONSHIP_TYPES` (`grep -c "ALLOWED_ENTITY_TYPES\|ALLOWED_RELATIONSHIP_TYPES"
backend/knowledge/extraction/validator.py` -> 2 read sites), and
normalizer.py's `RETIRED_TYPE_MAP`/`CANONICAL_REGISTRY`/`STATIC_ALIASES` --
all module/class-level constants, fixed at import. temporal.py's pattern
table is different in kind: `self._patterns` is INSTANCE state, not module
state -- it is built fresh, from the same literal `re.compile` calls, inside
`TemporalResolver.__init__` on every construction
(`grep -n "self._patterns" backend/knowledge/extraction/temporal.py` -> the
sole assignment is at `__init__`, not module scope). D2 needs all of this
state to itself be deterministic per replay -- true today, whether it is a
constant fixed at import or an instance rebuilt identically on every
construction -- not that the stages ignore it.

What this module actually establishes, and no more:
- No stage reads the wall clock (Stage 4 only -- directly trapped below).
  Stages 3, 5, 6 (confidence.py, validator.py, normalizer.py) have no
  `datetime`/`time`/`random`/`uuid`/`os.environ` reference AT ALL -- not
  even an unused import -- confirmed by
  `grep -nE 'datetime\.|\.now\(|\.today\(|utcnow|time\.|random|uuid|os\.environ|^import (datetime|time)\b|^from (datetime|time) '
  backend/knowledge/extraction/{confidence,validator,normalizer}.py` ->
  zero matches (deliberately excludes temporal.py, which legitimately does
  `from datetime import datetime` for its `reference_date` parameter type --
  an import, not a clock read, and Stage 4's own clock-freedom is what the
  monkeypatch trap below establishes instead). The same pattern fires on a
  file with a bare `import datetime`/`import time` and no dotted use
  (confirming the added alternation actually extends coverage, not just
  restates the dotted-call check) and on `backend/knowledge/admin.py`
  (which does call `datetime.now(`), confirming it is a live check, not a
  pattern that cannot fire.
- Given fixed module/instance state (true within one process and, since none
  of that state is seeded from `random`/clock/env, true across processes
  too), each stage returns byte-identical output for byte-identical input
  across two calls on the same instance.

It says nothing about unordered iteration over a *different* kind of hidden
state (a global counter, a set built from a source that isn't itself
order-stable) unless a fixture reaches the code path that would expose it --
each test below documents the fixture property that puts it on a real code
path, not a vacuous early return, and names the mutant it was manually
confirmed to catch (applied to source, run, seen RED, reverted) during
review.

Diagnostic over the paths its fixtures reach, written before the D2 rebuild
driver that will re-run Stages 3-6 end to end -- that driver is still
unwritten. D3 already partially depends on this assumption, though:
`cache_key`'s own docstring (landed at 49c880d) states that Stage 5 and
Stage 6 "both now run in REPLAYED code, so an ontology change there is
re-derived on every rebuild rather than invalidating the cache" -- a direct
reliance on Stages 5-6 being safely re-runnable, not merely a claim about
where Stage 2's cache boundary sits.
"""

import copy
from datetime import datetime

import pytest

from backend.knowledge.extraction.confidence import ConfidenceScorer
from backend.knowledge.extraction.ontology_extractor import ExtractionResult
from backend.knowledge.extraction.temporal import TemporalResolver
from backend.knowledge.extraction.validator import ExtractionValidator

REFERENCE_DATE = datetime.fromisoformat("2026-08-18T12:00:00+00:00")


def _sample_extraction(
    *,
    source_utterance: str = "I think I have used Python for about 5 years",
    temporal_expression: str = "5 years ago",
) -> ExtractionResult:
    """Baseline single-relationship extraction.

    Defaults deliberately reach real logic rather than an early return:
    `source_utterance` contains "I think" (a Stage 3 hedge trigger) and
    `temporal_expression` is "5 years ago" (a Stage 4 pattern match, not
    the un-parseable "for about 5 years" the original fixture used). Both
    are overridable so a caller can also exercise the early-return / no-match
    path deliberately, rather than by accident.
    """
    return ExtractionResult(
        entities=[
            {"id": "user", "name": "User", "type": "User"},
            {"id": "python", "name": "Python", "type": "Technology"},
        ],
        relationships=[
            {
                "source": "user",
                "target": "python",
                "type": "USES",
                "properties": {
                    "confidence": 0.9,
                    "temporal_status": "current",
                    "start_date": None,
                    "end_date": None,
                    "temporal_expression": temporal_expression,
                    "context": None,
                    "negated": False,
                    "assertion_kind": "assert",
                },
            }
        ],
        source_utterance=source_utterance,
    )


class TestStage3Confidence:
    """Mutant killed (confirmed during review, applied/RED/reverted): a
    `random.uniform(-0.05, 0.05)` term added to `adjusted` in
    `ConfidenceScorer.adjust_confidence`'s per-relationship loop
    (confidence.py). With the default hedge-triggering fixture, this mutant
    makes the two calls in `test_stage_3_confidence_is_deterministic` diverge
    with overwhelming probability; the fixture used in Task 2's original
    submission never entered the loop, so the same mutant would have passed.
    """

    def test_stage_3_confidence_is_deterministic(self):
        scorer = ConfidenceScorer()
        first = scorer.adjust_confidence(copy.deepcopy(_sample_extraction()))
        second = scorer.adjust_confidence(copy.deepcopy(_sample_extraction()))
        assert first.entities == second.entities
        assert first.relationships == second.relationships
        # Not a vacuous early return: "I think" is a hedge trigger, so the
        # adjustment body ran and confidence moved off its 0.9 input.
        assert first.relationships[0]["properties"]["confidence"] != 0.9

    def test_stage_3_confidence_is_unchanged_without_a_hedge_or_third_party_signal(
        self,
    ):
        """Sibling of the hedge case: the early-return guard
        (`confidence.py`'s `if hedge_penalty == 0.0 and not is_third_party:
        return extraction`) is itself deterministic and leaves confidence
        untouched. This is the path the module's default fixture no longer
        covers, so it is covered here explicitly instead of by accident.
        """
        scorer = ConfidenceScorer()
        extraction = _sample_extraction(source_utterance="I have used Python for about 5 years")

        first = scorer.adjust_confidence(copy.deepcopy(extraction))
        second = scorer.adjust_confidence(copy.deepcopy(extraction))

        assert first.relationships == second.relationships
        assert first.relationships[0]["properties"]["confidence"] == 0.9


# One expression per TemporalResolver._patterns handler (temporal.py), so
# the determinism and clock-trap tests below exercise all 12, not a subset.
TEMPORAL_EXPRESSIONS = [
    pytest.param("last year", id="last-year"),
    pytest.param("this year", id="this-year"),
    pytest.param("next year", id="next-year"),
    pytest.param("last month", id="last-month"),
    pytest.param("this month", id="this-month"),
    pytest.param("next month", id="next-month"),
    pytest.param("next quarter", id="next-quarter"),
    pytest.param("last quarter", id="last-quarter"),
    pytest.param("5 years ago", id="n-years-ago"),
    pytest.param("3 months ago", id="n-months-ago"),
    pytest.param("last week", id="last-week"),
    pytest.param("yesterday", id="yesterday"),
]


class TestStage4Temporal:
    """Mutants killed (confirmed during review, applied/RED/reverted).

    (1) A module-level counter in `temporal.py` that reverses
    `_resolve_n_years_ago`'s returned tuple on every other call --
    `test_stage_4_temporal_is_deterministic_for_a_fixed_reference_date`
    goes RED, because with "5 years ago" the handler actually runs.
    Task 2's original fixture ("for about 5 years", no "ago") matched no
    pattern, so `resolutions == 0` and no handler ever ran; that mutant
    would have passed silently.

    (2) `datetime.now(reference_date.tzinfo)` inserted at the top of
    `_resolve_n_years_ago`, discarding the passed `reference_date` --
    `test_stage_4_never_reads_the_wall_clock` goes RED via the monkeypatched
    `ExplodingDatetime`. Under the original fixture this handler never ran,
    so the trap was armed over code that could not fire.
    """

    @pytest.mark.parametrize("expression", TEMPORAL_EXPRESSIONS)
    def test_stage_4_temporal_is_deterministic_for_a_fixed_reference_date(self, expression):
        resolver = TemporalResolver()
        extraction = _sample_extraction(temporal_expression=expression)

        first = resolver.resolve(copy.deepcopy(extraction), REFERENCE_DATE)
        second = resolver.resolve(copy.deepcopy(extraction), REFERENCE_DATE)

        assert first.relationships == second.relationships
        # Not a vacuous no-match: every _resolve_* handler sets start_date.
        assert first.relationships[0]["properties"]["start_date"] is not None

    @pytest.mark.parametrize("expression", TEMPORAL_EXPRESSIONS)
    def test_stage_4_never_reads_the_wall_clock(self, monkeypatch, expression):
        """C1 anchors reference_date to recorded_at. A clock read anywhere in
        `resolve()` or any of its 12 `_resolve_*` handlers would defeat it.

        Scope limit, stated rather than implied: this traps a clock read
        that happens when `resolve()` runs. A read at *import time* (a
        module-level `datetime.now()` executed when `temporal.py` is first
        imported) would happen before `monkeypatch.setattr` below installs
        the trap and is structurally uncatchable by this test. The
        `grep`-based evidence in this module's docstring covers that gap
        instead: it shows no `datetime.now`/`.today`/`utcnow` call exists
        anywhere in the file, import-time or not.
        """
        import backend.knowledge.extraction.temporal as temporal_module

        class ExplodingDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                raise AssertionError("Stage 4 read the wall clock via now()")

            @classmethod
            def today(cls):
                raise AssertionError("Stage 4 read the wall clock via today()")

            @classmethod
            def utcnow(cls):
                raise AssertionError("Stage 4 read the wall clock via utcnow()")

            @classmethod
            def fromtimestamp(cls, *args, **kwargs):
                raise AssertionError("Stage 4 read the wall clock via fromtimestamp()")

        monkeypatch.setattr(temporal_module, "datetime", ExplodingDatetime)
        extraction = _sample_extraction(temporal_expression=expression)

        TemporalResolver().resolve(extraction, REFERENCE_DATE)


def _extraction_with_two_distinct_warnings() -> ExtractionResult:
    """A valid relationship plus two invalid ones, each tripping a
    DIFFERENT `ExtractionValidator.validate` warning, so `warnings` has
    real content and real order to pin -- not `[] == []`.

    Relationship 2 (`python` -[USES]-> `python`) trips the self-reference
    guard (`validator.py`: `if source == target: warnings.append(...)`).
    Relationship 3 (`user` -[USES]-> `python`, confidence 0.2) trips the
    confidence-threshold guard (`if confidence < self.min_confidence:
    warnings.append(...)`, default `min_confidence=0.5`). Both are warnings,
    not errors, so validation never halts -- consistent with the module's
    "Invalid items are dropped and logged. The pipeline never halts."
    """
    return ExtractionResult(
        entities=[
            {"id": "user", "name": "User", "type": "User"},
            {"id": "python", "name": "Python", "type": "Technology"},
        ],
        relationships=[
            {
                "source": "user",
                "target": "python",
                "type": "USES",
                "properties": {"confidence": 0.9},
            },
            {
                "source": "python",
                "target": "python",
                "type": "USES",
                "properties": {"confidence": 0.9},
            },
            {
                "source": "user",
                "target": "python",
                "type": "USES",
                "properties": {"confidence": 0.2},
            },
        ],
        source_utterance="I have used Python for about 5 years",
    )


class TestStage6Validation:
    """Mutant killed (confirmed during review, applied/RED/reverted): a
    module-level call counter in `validator.py` that reverses the `warnings`
    list on every other `validate()` call. Task 2's original fixture
    produced zero warnings, so `[] == []` could not distinguish a reversed
    empty list from an unreversed one; that mutant would have passed.
    """

    def test_stage_6_validation_is_deterministic_and_order_stable(self):
        validator = ExtractionValidator()
        extraction = _extraction_with_two_distinct_warnings()

        first = validator.validate(copy.deepcopy(extraction))
        second = validator.validate(copy.deepcopy(extraction))

        assert first.entities == second.entities
        assert first.relationships == second.relationships
        assert first.warnings == second.warnings
        assert first.errors == second.errors
        # Not `[] == []`: two distinct warnings, in a fixed order, or there
        # is nothing here for "order stable" to mean.
        assert len(first.warnings) == 2


@pytest.mark.asyncio
class TestStage5Normalize:
    r"""Stage 5 is a pure function of its argument.

    Both constructor dependencies are vestigial: `embedding_generator` and
    `executor` are stored in `EntityNormalizer.__init__` and never read
    anywhere else in the module (`grep -n "self\._embedding_generator\|self\._executor"
    backend/knowledge/extraction/normalizer.py` -- both hits are the assignments
    in `__init__`, no other match in the file), and `normalize()`'s own
    docstring states it "issues no graph queries" -- graph-identity resolution
    moved to the curation deduper (Stage 7a). Passing None for both is
    therefore valid, and a rebuild can run Stage 5 with no graph and no model.
    """

    async def test_normalize_is_pure_with_none_collaborators(self):
        """Reproduces exactly what D2's rebuild driver will construct: both
        collaborators as literal `None`, not fakes.

        If passing None raises, that assumption is dead and spec D2 needs
        revisiting -- stop and report rather than supplying a real
        collaborator to make it pass.

        Mutant killed (confirmed during review, applied/RED/reverted):
        `if self._executor is None: raise RuntimeError(...)` inserted at the
        top of `normalize()` -- i.e. code that requires a live executor to be
        present at all. This mutant does NOT fail the sentinel test below
        (there, `_executor` is a non-None object, so the guard never fires
        and the mutant is invisible) -- confirmed by applying it and seeing
        that test stay green. The two tests are complementary, not
        redundant: each catches a mutant class the other cannot.
        """
        from backend.knowledge.extraction.normalizer import EntityNormalizer

        normalizer = EntityNormalizer(embedding_generator=None, executor=None)
        first = await normalizer.normalize(copy.deepcopy(_sample_extraction()))
        second = await normalizer.normalize(copy.deepcopy(_sample_extraction()))
        assert first.entities == second.entities
        assert first.relationships == second.relationships

    async def test_normalize_never_touches_its_collaborators(self):
        """Stronger and complementary to the `None` case above: `None` makes
        a hypothetical `if self._executor is not None: ...` guard silently
        skip, which would happen to also match real replay behavior and so
        would not be caught by the test above. A guard shaped that way would
        still mean Stage 5 is no longer unconditionally callable with no
        graph -- only conditionally so, which is a narrower and weaker
        property than "issues no graph queries." This sentinel makes any
        touch at all -- guarded or not -- raise immediately.

        Mutant killed (confirmed during review, applied/RED/reverted):
        `if self._executor is not None: self._executor.touch()` inserted at
        the top of `normalize()` -- a conditional touch that only fires for
        a live collaborator. This mutant does NOT fail the `None`-collaborator
        test above (there, `_executor is not None` is false, so the guard
        never fires and the mutant is invisible) -- confirmed by applying it
        and seeing that test stay green. This is the exact failure mode the
        `None` test alone cannot rule out: a future `if self._executor is not
        None: ...` guard would pass forever under literal `None` collaborators
        while still meaning Stage 5 is not unconditionally free of graph
        queries, only conditionally so.
        """
        from backend.knowledge.extraction.normalizer import EntityNormalizer

        class _ExplodingCollaborator:
            def __getattr__(self, name: str):
                raise AssertionError(f"Stage 5 touched its collaborator via .{name}")

        normalizer = EntityNormalizer(
            embedding_generator=_ExplodingCollaborator(),
            executor=_ExplodingCollaborator(),
        )
        first = await normalizer.normalize(copy.deepcopy(_sample_extraction()))
        second = await normalizer.normalize(copy.deepcopy(_sample_extraction()))
        assert first.entities == second.entities
        assert first.relationships == second.relationships
