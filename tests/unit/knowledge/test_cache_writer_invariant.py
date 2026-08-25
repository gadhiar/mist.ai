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
