---
type: meta-changelog
created: 2026-01-01
updated: 2026-01-01
---

# MIST Vault Changelog (Test Baseline)

## 2026-01-01 -- Test-vault baseline created
- Mirror of real mist-memory/ structure with deterministic seed content.
- Copied per gauntlet run by the isolated_test_vault pytest fixture
  (tests/conftest.py).
- Seed content: one identity, one test user, two seed sessions, one
  seed decision. Sufficient for retrieval / extraction gauntlets.
