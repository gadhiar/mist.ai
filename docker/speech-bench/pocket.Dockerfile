# mist-speech-bench-pocket: Kyutai's Pocket TTS on the base bench image, run on the CPU.
#
# Its own image so its pins never meet chatterbox's. torch, torchaudio and torchvision stay at
# the backend's versions through the constraints file. Only the two pin files beside this
# Dockerfile are copied; the probe code is mounted at run time.
#
#   docker build -f docker/speech-bench/pocket.Dockerfile -t mist-speech-bench-pocket docker/speech-bench/
ARG BASE_IMAGE=mist-speech-bench-base
FROM ${BASE_IMAGE}

COPY requirements-pocket.txt constraints-torch.txt /tmp/speech-bench/
RUN python -m pip install --no-cache-dir \
        -r /tmp/speech-bench/requirements-pocket.txt \
        -c /tmp/speech-bench/constraints-torch.txt
