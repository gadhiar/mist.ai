"""Container-side speech probes.

These modules run inside the speech bench containers, with the repository mounted at `/work:ro`.
Engine packages are imported inside functions only, so the worker image's tests can import every
module here without loading an engine.
"""
