# mist-voice-teacher-cosyvoice3: Fun-CosyVoice3 0.5B on torch 2.3.1 (cu121).
#
# FROM the base image of the same commit: `images build` tags it mist-voice-base:<first 12 hex>
# and passes only GIT_SHA, so BASE_IMAGE drops the last 28 characters of the 40-hex sha.
# CosyVoice is not pip-installable: it is cloned at its pinned commit, with its Matcha-TTS
# submodule cloned at the pinned commit into third_party/Matcha-TTS.
ARG GIT_SHA
ARG BASE_IMAGE=mist-voice-base:${GIT_SHA%????????????????????????????}
FROM ${BASE_IMAGE}

COPY docker/voice-data/requirements-cosyvoice3.txt /opt/mist-voice/requirements/requirements-cosyvoice3.txt
RUN python3.11 -m venv /opt/venv-teacher-cosyvoice3 \
 && /opt/venv-teacher-cosyvoice3/bin/pip install --no-cache-dir -r /opt/mist-voice/requirements/requirements-cosyvoice3.txt

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
RUN printf '%s\n' /opt/mist-voice/code > /opt/venv-teacher-cosyvoice3/lib/python3.11/site-packages/mist-voice-code.pth
RUN /opt/venv-teacher-cosyvoice3/bin/python -I -c 'import torch, torchaudio, conformer, diffusers, hydra, hyperpyyaml, inflect, librosa, lightning, modelscope, networkx, numpy, omegaconf, onnx, onnxruntime, whisper, pyarrow, pydantic, pyworld, soundfile, transformers, x_transformers, wetext, scripts.voice_data'
