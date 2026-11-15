"""Ceiling runs for MIST's extraction service: an answerer in the model's seat.

`store` keeps the answerer's replies in an append-only, hash-chained file, and `relay`
serves them to the unchanged extraction service as llama-server would, so a run replays
from committed files alone.
"""
