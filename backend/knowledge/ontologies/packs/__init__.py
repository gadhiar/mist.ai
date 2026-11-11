"""The ontology packs: one package per ontology version.

A pack package holds `types.py`, `manifest.py`, `conventions.py`, `plugins.py` and
`labelling.md`, and `backend.knowledge.ontologies.pack.load_pack` reads it by its dotted
name. This package imports nothing, and neither does a pack's own `__init__`: the old
import path of the types (`backend.knowledge.ontologies.v1_0_0`) loads both before the
pack loader can be imported.
"""
