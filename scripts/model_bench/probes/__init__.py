"""Stdlib-only probes used by bench_host.py.

Every module here is importable stand-alone with no third-party
dependencies, except `voice_vram.py`, which runs inside the mist-backend
container (via `docker exec -i mist-backend python -`, source on stdin) and
may import torch and backend/src modules lazily.
"""
