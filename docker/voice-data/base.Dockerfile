# mist-voice-base: the CUDA runtime, Python 3.11, the shared scoring venv and the pod entry.
#
# Every family image is FROM this image built at the same commit. It holds `/opt/venv-score`
# (speaker similarity, CER, arousal, pyin), ffmpeg, espeak-ng, git for pinned upstream clones,
# openssh-server for rented pods, and the repository code at `/opt/mist-voice/code`.
# Built by `python -m scripts.voice_data images build` from a `git archive` export, so paths are
# relative to the repository root:
#   docker build -f docker/voice-data/base.Dockerfile --build-arg GIT_SHA=<sha>
#     -t mist-voice-base:<first 12 hex of sha> <export>
FROM docker.io/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04@sha256:17e2934e1fa96152b14f78078bfbafd0f00f391df995dc6c641a720fce1202bb

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Ubuntu 22.04 ships Python 3.10, so 3.11 comes from the deadsnakes PPA. The host keys the
# openssh-server package generates are removed: remote-entry.sh makes fresh ones on each pod.
RUN apt-get update \
 && apt-get install -y --no-install-recommends software-properties-common gpg-agent ca-certificates \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update \
 && apt-get install -y --no-install-recommends python3.11 python3.11-venv python3.11-dev build-essential ffmpeg espeak-ng openssh-server git \
 && rm -f /etc/ssh/ssh_host_* \
 && rm -rf /var/lib/apt/lists/*

COPY docker/voice-data/requirements-score.txt /opt/mist-voice/requirements/requirements-score.txt
RUN python3.11 -m venv /opt/venv-score \
 && /opt/venv-score/bin/pip install --no-cache-dir -r /opt/mist-voice/requirements/requirements-score.txt

COPY docker/voice-data/remote-entry.sh /usr/local/bin/remote-entry.sh
RUN chmod 0755 /usr/local/bin/remote-entry.sh
CMD ["/usr/local/bin/remote-entry.sh"]

# Declared late so a new commit does not invalidate the layers above.
ARG GIT_SHA
RUN test -n "$GIT_SHA" \
 && python3.11 -I -c 'import hashlib, json, pathlib, sys; reqs = sorted(pathlib.Path("/opt/mist-voice/requirements").glob("*.txt")); record = {"family": sys.argv[1], "git_sha": sys.argv[2], "requirements": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in reqs}}; pathlib.Path("/etc/mist-voice-image.json").write_text(json.dumps(record, indent=1) + "\n")' score "$GIT_SHA"

COPY scripts/voice_data/ /opt/mist-voice/code/scripts/voice_data/
COPY scripts/model_bench/ /opt/mist-voice/code/scripts/model_bench/
RUN printf '%s\n' /opt/mist-voice/code > /opt/venv-score/lib/python3.11/site-packages/mist-voice-code.pth
RUN /opt/venv-score/bin/python -I -c 'import torch, torchaudio, transformers, huggingface_hub, speechbrain, librosa, soundfile, numpy, scipy, safetensors, accelerate, sentencepiece, scripts.voice_data'
