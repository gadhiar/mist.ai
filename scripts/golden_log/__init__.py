"""R1.4.5 golden log: the authored `log` half of `graph = f(seed@version, log, epoch)`.

`native_shape` holds the one authority for the gold-corpus <-> extractor-native field
mapping, `translate` converts gold records into native payloads, and `generate` builds the
deterministic artifact and materializes it into an ISOLATED event store + extraction cache.
Nothing here ever touches the live event store or the live graph.
"""
