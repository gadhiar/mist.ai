"""The v1.4.0 ontology pack: the types, the manifest, the rules, the plug-ins and the guide.

`backend.knowledge.ontologies.pack.load_pack` builds the pack from this package's dotted
name. Nothing is imported here: `types.py` is loaded through the old import path
(`backend.knowledge.ontologies.v1_0_0`) while the pack loader is still being imported, so
this file and `types.py` must not reach the loader. `manifest.py`, `conventions.py` and
`plugins.py` do import it, and are read by the loader alone.
"""
