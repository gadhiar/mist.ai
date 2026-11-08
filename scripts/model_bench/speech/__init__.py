"""Speech bench: the suites behind `python -m scripts.model_bench.speech_bench`.

Modules directly in this package run on the host and are stdlib-only, like `bench_host.py`.
Modules under `probes/` run inside the speech bench containers and import their engine packages
only inside functions, so importing any module here loads no engine.
"""
