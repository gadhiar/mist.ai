# mist-voice-teacher-cosyvoice3: Fun-CosyVoice3 0.5B on torch 2.3.1 (cu121).
#
# FROM the code-free `runtime` stage of base.Dockerfile: `images build` tags it
# mist-voice-runtime:<first 12 hex> and passes it as BASE_IMAGE, so a new commit reuses the
# venv layers below and rebuilds only the last four steps.
# CosyVoice is not pip-installable: it is cloned at its pinned commit, with its Matcha-TTS
# submodule cloned at the pinned commit into third_party/Matcha-TTS.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY docker/voice-data/requirements-cosyvoice3.txt /opt/mist-voice/requirements/requirements-cosyvoice3.txt
RUN python3.11 -m venv /opt/venv-teacher-cosyvoice3 \
 && /opt/venv-teacher-cosyvoice3/bin/pip install --no-cache-dir -r /opt/mist-voice/requirements/requirements-cosyvoice3.txt
RUN printf '%s\n' /opt/mist-voice/code > /opt/venv-teacher-cosyvoice3/lib/python3.11/site-packages/mist-voice-code.pth

RUN git clone --filter=blob:none https://github.com/FunAudioLLM/CosyVoice /opt/mist-voice/upstream/CosyVoice \
 && git -C /opt/mist-voice/upstream/CosyVoice checkout --detach 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc \
 && rm -rf /opt/mist-voice/upstream/CosyVoice/third_party/Matcha-TTS \
 && git clone --filter=blob:none https://github.com/shivammehta25/Matcha-TTS /opt/mist-voice/upstream/CosyVoice/third_party/Matcha-TTS \
 && git -C /opt/mist-voice/upstream/CosyVoice/third_party/Matcha-TTS checkout --detach dd9105b34bf2be2230f4aa1e4769fb586a3c824e

ARG GIT_SHA
RUN test -n "$GIT_SHA" \
 && python3.11 -I -c 'import hashlib, json, pathlib, sys; reqs = sorted(pathlib.Path("/opt/mist-voice/requirements").glob("*.txt")); record = {"family": sys.argv[1], "git_sha": sys.argv[2], "requirements": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in reqs}}; pathlib.Path("/etc/mist-voice-image.json").write_text(json.dumps(record, indent=1) + "\n")' teacher-cosyvoice3 "$GIT_SHA"

COPY scripts/voice_data/ /opt/mist-voice/code/scripts/voice_data/
COPY scripts/model_bench/ /opt/mist-voice/code/scripts/model_bench/
RUN /opt/venv-teacher-cosyvoice3/bin/python -I -c 'import torch, torchaudio, conformer, diffusers, hydra, hyperpyyaml, inflect, librosa, lightning, modelscope, networkx, numpy, omegaconf, onnx, onnxruntime, whisper, pyarrow, pydantic, pyworld, soundfile, transformers, x_transformers, wetext, scripts.voice_data' \
 && /opt/venv-score/bin/python -I -c 'import scripts.voice_data'
