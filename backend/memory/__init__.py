"""Memory read path: recall of facts, episodes, exchanges and notes for each turn.

The package depends on the event store, the graph (through GraphExecutor), the vault sidecar,
an embedder, a reranker and a token counter, each passed into a constructor; real wiring lives
in backend/factories.py. This module re-exports nothing, so importing one submodule (for
example the config, which backend/knowledge/config.py nests) loads no other.
"""
