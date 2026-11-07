# mist-speech-bench-base: the running mist-backend image, plus onnxruntime 1.31.1 with numpy
# held at 1.26.4 (requirements-base.txt).
#
# The backend image already holds chatterbox-tts 0.1.7 (with Turbo), openai-whisper and ffmpeg,
# so the speech bench's `turbo` and `original` engines and whisper-base run on it as built. It
# does not hold onnxruntime, which Smart Turn and onnx-asr need, so this image installs it,
# holding numpy at the backend's 1.26.4 and torch at its versions through the constraints file.
# The probe code is never copied in: it is mounted at run time as `-v <repo>:/work:ro`, so a code
# change needs no rebuild.
#
# Built by the lead with this directory as the context, given the image id the pin file records.
# BuildKit reads a bare image id in FROM as a registry name, so the id is tagged first:
#   docker tag <mist-backend image id> mist-backend-pinned:<first 12 hex of the id>
#   docker build -f docker/speech-bench/base.Dockerfile
#     --build-arg BACKEND_IMAGE=mist-backend-pinned:<first 12 hex of the id>
#     -t mist-speech-bench-base docker/speech-bench/
# (two commands, the second wrapped here).
ARG BACKEND_IMAGE
FROM ${BACKEND_IMAGE}

COPY requirements-base.txt constraints-torch.txt /tmp/speech-bench/
RUN python -m pip install --no-cache-dir \
        -r /tmp/speech-bench/requirements-base.txt \
        -c /tmp/speech-bench/constraints-torch.txt
