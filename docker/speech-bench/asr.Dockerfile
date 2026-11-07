# mist-speech-bench-asr: the NeMo and onnx-asr speech-to-text engines of run 3
# (`parakeet-gpu`, `nemotron-160`, `parakeet-cpu`; mist-model-host plan v2).
#
# Built by the lead in host step H2 with this directory as the context:
#   docker build -f docker/speech-bench/asr.Dockerfile -t mist-speech-bench-asr docker/speech-bench/
# It holds packages only: the probe code is the repository, mounted at /work:ro when a container
# runs, so a code change needs no rebuild. The torch constraints keep the base image's
# torch 2.6.0+cu124 in place while NeMo installs.
ARG BASE_IMAGE=mist-speech-bench-base
FROM ${BASE_IMAGE}

COPY requirements-asr.txt constraints-torch.txt /tmp/speech-bench/
RUN python -m pip install --no-cache-dir \
        -r /tmp/speech-bench/requirements-asr.txt \
        -c /tmp/speech-bench/constraints-torch.txt

# nemo_toolkit 2.7.3 installs a top-level `scripts` package into site-packages. The repository's
# `scripts` has no `__init__.py`, so it is a namespace package, and a regular package anywhere on
# `sys.path` wins over it: `scripts.model_bench` would not import, and none of the three engines
# would load. The step removes NeMo's copy and fails the build if any `scripts` is still
# importable. `-I` drops the working directory (/app, which holds the backend's `scripts/`) and
# `PYTHONPATH` from `sys.path`, so only the installed packages are judged.
RUN rm -rf /opt/venv/lib/python3.11/site-packages/scripts \
    && python -I -c "import importlib.util, sys; sys.exit(importlib.util.find_spec('scripts') is not None)"
