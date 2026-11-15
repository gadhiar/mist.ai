# mist-voice-convert: kNN-VC and Seed-VC on torch 2.4.0 (cu124).
#
# FROM the code-free `runtime` stage of base.Dockerfile: `images build` tags it
# mist-voice-runtime:<first 12 hex> and passes it as BASE_IMAGE, so a new commit reuses the
# venv layers below and rebuilds only the last four steps.
# Neither code base is pip-installable: both are cloned at their pinned commits.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY docker/voice-data/requirements-convert.txt /opt/mist-voice/requirements/requirements-convert.txt
RUN python3.11 -m venv /opt/venv-convert \
 && /opt/venv-convert/bin/pip install --no-cache-dir -r /opt/mist-voice/requirements/requirements-convert.txt
RUN printf '%s\n' /opt/mist-voice/code > /opt/venv-convert/lib/python3.11/site-packages/mist-voice-code.pth

RUN git clone --filter=blob:none https://github.com/bshall/knn-vc /opt/mist-voice/upstream/knn-vc \
 && git -C /opt/mist-voice/upstream/knn-vc checkout --detach c616845c4e309e24d5927f15adbdf277a3d65358 \
 && git clone --filter=blob:none https://github.com/Plachtaa/seed-vc /opt/mist-voice/upstream/seed-vc \
 && git -C /opt/mist-voice/upstream/seed-vc checkout --detach 51383efd921027683c89e5348211d93ff12ac2a8

ARG GIT_SHA
RUN test -n "$GIT_SHA" \
 && python3.11 -I -c 'import hashlib, json, pathlib, sys; reqs = sorted(pathlib.Path("/opt/mist-voice/requirements").glob("*.txt")); record = {"family": sys.argv[1], "git_sha": sys.argv[2], "requirements": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in reqs}}; pathlib.Path("/etc/mist-voice-image.json").write_text(json.dumps(record, indent=1) + "\n")' convert "$GIT_SHA"

COPY scripts/voice_data/ /opt/mist-voice/code/scripts/voice_data/
COPY scripts/model_bench/ /opt/mist-voice/code/scripts/model_bench/
RUN /opt/venv-convert/bin/python -I -c 'import torch, torchaudio, torchvision, scipy, librosa, huggingface_hub, munch, einops, dac, pydub, jiwer, transformers, soundfile, modelscope, funasr, numpy, hydra, yaml, accelerate, scripts.voice_data' \
 && /opt/venv-score/bin/python -I -c 'import scripts.voice_data'
