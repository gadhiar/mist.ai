"""Image-side model bindings: the protocols every stage is written against, and their real models.

This folder runs only inside the voice-data images, so it may import numpy and the model
libraries; `tests/unit/voice_data/test_host_stdlib.py` exempts it from the host's stdlib rule.
"""
