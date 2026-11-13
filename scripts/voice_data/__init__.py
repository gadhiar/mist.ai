"""Voice-data pipeline: private root, records, gates and the `python -m scripts.voice_data` CLI.

Everything here that runs on the host uses the standard library only; container-side stages live
in their own folders (see `tests/unit/voice_data/test_host_stdlib.py`).
"""
