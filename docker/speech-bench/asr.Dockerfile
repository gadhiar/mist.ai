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
