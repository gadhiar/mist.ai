"""Mutation-proof guard: no production path may write a VaultNote node.

R1.3/R1.4 retired the `DERIVED_FROM -> VaultNote` seed-provenance edge --
seed facts now carry a `seed_version` property instead (R1.4 Tasks 1-5).
`emit_seed_vault_provenance` (backend/knowledge/admin.py) was the last
surviving write path that MERGE-created a `VaultNote` node; R1.4 Task 6
deletes it.

The `DERIVED_FROM` edge TYPE survives -- `graph_writer.py` still writes it
for entity->chunk edges when `source_metadata.synthesis` is true (a wholly
separate, legitimate use). This guard therefore asserts on the `VaultNote`
TARGET, never on `DERIVED_FROM`; a guard keyed on `DERIVED_FROM` would go
red on that legitimate synthesis code and be wrong.
"""

import pathlib


def test_no_production_path_writes_a_vaultnote_node():
    """`DERIVED_FROM -> VaultNote` was retired in R1.3/R1.4.

    The edge TYPE survives for synthesis->chunk edges (graph_writer.py), so
    this asserts on the VaultNote TARGET, not on DERIVED_FROM.
    """
    roots = [pathlib.Path("/app/backend"), pathlib.Path("/app/scripts")]
    offenders = []
    for root in roots:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#") or '"""' in stripped:
                    continue
                if "VaultNote" in line and ("MERGE" in line or "CREATE" in line):
                    offenders.append(f"{path}:{lineno}: {stripped}")

    assert not offenders, "VaultNote write path reintroduced:\n" + "\n".join(offenders)
