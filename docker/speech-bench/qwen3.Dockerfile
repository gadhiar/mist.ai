# mist-speech-bench-qwen3: faster-qwen3-tts on the base bench image.
#
# Its own image so its transformers 5 pin never meets chatterbox's transformers 4. torch,
# torchaudio and torchvision stay at the backend's versions through the constraints file. Only
# the two pin files beside this Dockerfile are copied; the probe code is mounted at run time.
#
#   docker build -f docker/speech-bench/qwen3.Dockerfile -t mist-speech-bench-qwen3 docker/speech-bench/
ARG BASE_IMAGE=mist-speech-bench-base
FROM ${BASE_IMAGE}

COPY requirements-qwen3.txt constraints-torch.txt /tmp/speech-bench/
RUN python -m pip install --no-cache-dir \
        -r /tmp/speech-bench/requirements-qwen3.txt \
        -c /tmp/speech-bench/constraints-torch.txt
