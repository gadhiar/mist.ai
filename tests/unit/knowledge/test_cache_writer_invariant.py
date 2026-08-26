"""D10: the production cache records what the MODEL produced. Nothing else.

The golden-log generator writes AUTHORED entities. If its cache root ever
resolves to the production root, a rebuild reads the answer key and scores
itself perfect -- with no symptom anywhere.

`KnowledgeConfig.from_env()` is used here rather than `build_test_config()`
(tests/CLAUDE.md's usual rule) deliberately: the property under test IS the
real, environment-derived production path, and `assert_not_production_root`
itself calls `KnowledgeConfig.from_env()` internally (never an injected
config). Substituting a test config would verify a fiction instead of the
guard that actually runs in production.
"""

from pathlib import Path

import pytest

from backend.factories import production_cache_path
from backend.knowledge.config import KnowledgeConfig
from scripts.golden_log.generate import golden_log_cache_path


def test_golden_log_and_production_cache_roots_can_never_coincide(tmp_path):
    """Mutant this kills: `golden_log_cache_path` deriving its path from
    `production_cache_path`/config instead of from its own `root` argument
    (e.g. a copy-paste that forgot to swap the source) -- both sides would
    then resolve under the real production root and this equality would fail
    where it should never even be close.
    """
    config = KnowledgeConfig.from_env()
    prod = Path(production_cache_path(config)).resolve()
    golden = Path(golden_log_cache_path(tmp_path)).resolve()
    assert prod != golden


def test_golden_log_refuses_to_write_into_the_production_root():
    """Mutant this kills: deleting the `raise` (or the equality check) inside
    `assert_not_production_root` -- the call would return silently instead of
    refusing, and a golden-log materialize pointed at the production root
    would proceed to overwrite the answer key into the live cache.
    """
    from scripts.golden_log import generate as gen

    config = KnowledgeConfig.from_env()
    prod_root = Path(production_cache_path(config)).parent

    with pytest.raises(gen.GoldenLogError, match="production"):
        gen.assert_not_production_root(prod_root)


def test_golden_log_permits_writing_to_an_ordinary_root(tmp_path):
    """Review round 1, Minor 3: the refuse direction is proven above; this
    proves the guard is not an unconditional raise. The 24 tests in
    tests/unit/golden_log/ (test_generate.py, test_replay.py) already
    exercise `materialize_isolated` end-to-end against ordinary `tmp_path`
    roots and would catch an always-raise regression, but this pins the
    permit direction locally, in the file someone reads when modifying the
    guard.

    Mutant this kills: `if True or Path(root).resolve() == prod:` (always
    raise, over-eager).
    """
    from scripts.golden_log import generate as gen

    gen.assert_not_production_root(tmp_path)  # must not raise


def test_golden_log_does_not_misfire_when_the_production_cache_is_in_memory(monkeypatch):
    """Review round 1, Minor 1: `production_cache_path` returns the bare
    ":memory:" sentinel when the production event store is configured
    in-memory (`backend/factories.py:production_cache_path`). Before this
    fix, `assert_not_production_root` took `.parent` of that string
    unconditionally -- `Path(":memory:").parent` is a relative "." whose
    `.resolve()` is the process CWD, not a production root -- silently
    retargeting the guard onto an unrelated directory instead of refusing
    to compare at all. `EVENT_STORE_DB_PATH` is unreachable at ":memory:"
    through the running container's real config (`.env:52`,
    `docker-compose.yml:39` both set a real path), so this test drives the
    sentinel branch directly via monkeypatch.

    Mutant this kills: removing the ":memory:" early-return and letting
    execution fall through to the unconditional `.parent.resolve()` branch
    -- this test would then raise `GoldenLogError` for a root that merely
    happens to equal the process CWD, which has nothing to do with any real
    production cache.
    """
    monkeypatch.setenv("EVENT_STORE_DB_PATH", ":memory:")
    from scripts.golden_log import generate as gen

    misresolved_root = Path(":memory:").parent.resolve()  # what the old bug computed as "prod"

    gen.assert_not_production_root(misresolved_root)  # must not raise


def test_golden_log_and_production_cache_use_different_filenames():
    """The root check above is D10's real, sufficient invariant: two
    different roots can never produce the same path regardless of filename.
    The two caches ALSO happen to use different filenames -- hyphen
    (`extraction-cache.db`) for the golden log, underscore
    (`extraction_cache.db`, see `backend/factories.py:production_cache_path`
    and `scripts/mist_admin.py:_build_log_regenerator`) for production. That
    divergence was never chosen as a safety property and nothing else pinned
    it -- exactly what a naming-convention cleanup would "fix" without
    noticing it deletes a second, independent barrier against the hazard
    this file exists to prevent.

    This test does not add protection by itself; it makes the accident
    visible so a future edit has to make it a deliberate choice rather than
    a silent one.

    Mutant this kills: renaming the golden log's cache file to
    `extraction_cache.db` (matching production) for naming consistency.
    """
    config = KnowledgeConfig.from_env()
    prod_name = Path(production_cache_path(config)).name
    golden_name = Path(golden_log_cache_path(Path("irrelevant"))).name

    assert golden_name != prod_name
