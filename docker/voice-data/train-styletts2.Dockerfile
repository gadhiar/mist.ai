# mist-voice-train-styletts2: StyleTTS2 fine-tuning. StyleTTS2 is not pip-installable, so it is
# cloned at its pinned commit; its Utils/ASR, Utils/JDC and Utils/PLBERT checkpoints come with it.
#
# FROM the code-free `runtime` stage of base.Dockerfile: `images build` tags it
# mist-voice-runtime:<first 12 hex> and passes it as BASE_IMAGE, so a new commit reuses the
# venv layers below and rebuilds only the last four steps.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY docker/voice-data/requirements-styletts2.txt /opt/mist-voice/requirements/requirements-styletts2.txt
RUN python3.11 -m venv /opt/venv-train-styletts2 \
 && /opt/venv-train-styletts2/bin/pip install --no-cache-dir -r /opt/mist-voice/requirements/requirements-styletts2.txt
RUN printf '%s\n' /opt/mist-voice/code > /opt/venv-train-styletts2/lib/python3.11/site-packages/mist-voice-code.pth

RUN git clone --filter=blob:none https://github.com/yl4579/StyleTTS2 /opt/mist-voice/upstream/StyleTTS2 \
 && git -C /opt/mist-voice/upstream/StyleTTS2 checkout --detach 5cedc71c333f8d8b8551ca59378bdcc7af4c9529

ARG GIT_SHA
RUN test -n "$GIT_SHA" \
 && python3.11 -I -c 'import hashlib, json, pathlib, sys; reqs = sorted(pathlib.Path("/opt/mist-voice/requirements").glob("*.txt")); record = {"family": sys.argv[1], "git_sha": sys.argv[2], "requirements": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in reqs}}; pathlib.Path("/etc/mist-voice-image.json").write_text(json.dumps(record, indent=1) + "\n")' train-styletts2 "$GIT_SHA"

COPY scripts/voice_data/ /opt/mist-voice/code/scripts/voice_data/
COPY scripts/model_bench/ /opt/mist-voice/code/scripts/model_bench/
RUN /opt/venv-train-styletts2/bin/python -I -c 'import torch, torchaudio, soundfile, munch, pydub, yaml, librosa, nltk, matplotlib, accelerate, transformers, einops, einops_exts, tqdm, phonemizer, numpy, monotonic_align, scripts.voice_data' \
 && /opt/venv-score/bin/python -I -c 'import scripts.voice_data'
