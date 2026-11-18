"""Voice VRAM probe: load the realtime speech engines, run STT and TTS once, report peak CUDA memory.

This is the only file in scripts/model_bench that is NOT stdlib-only (see
CLAUDE.md and the T3a brief): it runs INSIDE the mist-backend container,
piped in via `docker exec -i mist-backend python -` (see bench_host.py's
`voice` subcommand), and imports torch and the backend's own realtime
modules -- lazily, after `sys.path` is set up, so importing this module on
the host (where torch is not installed) does not fail.

It is deliberately a single self-contained script, not a package import
from bench_host.py: `docker exec -i ... python -` only has this file's
source on stdin, nothing else from the repo is on that stream.

The engines are the ones the live path runs: the settings come from
`load_voice_settings(os.environ)`, as `lifespan` reads them
(`backend/server.py:1547`), and `build_speech_engines` (`backend/factories.py`)
walks the same STT and TTS chains with the same adapters `build_mist_runtime`
loads. So the probe measures whatever `MIST_VOICE_STT_CHAIN` and
`MIST_VOICE_TTS_CHAIN` name in the container's environment.

Eager-loading note (see README.md "Voice probe double-counting"): production
loads its engines at mist-backend startup, in `build_mist_runtime`, which
`lifespan` runs through `_build_runtime` (`backend/server.py:1090`). This
probe loads its OWN engines in its own process rather than reusing the running
backend's, so its peak-memory reading double-counts whatever the backend
already holds resident -- the vram_steps.json fit estimate this feeds is
therefore conservative (overstates VRAM headroom pressure), not exact.

The peak is read once, over the whole run: the probe resets torch's peak
statistics before loading and passes `build_speech_engines` a meter that does
not reset them again, where the default meter (`TorchGpuMeter`) resets them
before each CUDA load.

Prints exactly one JSON line on stdout:
    {"max_memory_reserved_mib": float|null, "max_memory_allocated_mib": float|null,
     "stt_ok": bool, "tts_ok": bool, "error": str|null}
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any

PROBE_TEXT = "Voice VRAM probe."


def _mib(num_bytes: int) -> float:
    return round(num_bytes / (1024 * 1024), 3)


class _WholeRunMeter:
    """A `GpuMeter` that leaves torch's peak statistics alone, so one reading covers every load."""

    def measure(self, load: Any) -> tuple[Any, int]:
        return load(), 0


def main() -> None:
    result: dict[str, object] = {
        "max_memory_reserved_mib": None,
        "max_memory_allocated_mib": None,
        "stt_ok": False,
        "tts_ok": False,
        "error": None,
    }
    errors: list[str] = []

    try:
        import numpy as np
        import torch

        # Production's project root inside the container is fixed by
        # docker/backend/Dockerfile's `WORKDIR /app`. Overridable for
        # container layouts other than mist-backend's own.
        project_root = os.environ.get("MIST_PROJECT_ROOT", "/app")
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from backend.factories import build_speech_engines
        from backend.realtime.budget import load_voice_settings
        from backend.realtime.speech import CancelToken, TextSegment

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        settings = load_voice_settings(os.environ)
        engines = build_speech_engines(settings, gpu_meter=_WholeRunMeter())

        try:
            transcriber = engines.transcriber
            if transcriber is None:
                raise RuntimeError(f"no STT engine loaded: {engines.statuses.stt}")
            stream = transcriber.open_stream()
            try:
                stream.push(np.zeros(16000 * 2, dtype=np.float32))  # 2s of silence @ 16kHz
                stream.decode_final(CancelToken())
            finally:
                stream.close()
            result["stt_ok"] = True
        except Exception as exc:  # noqa: BLE001 -- probe reports both halves, never crashes on one
            errors.append(f"stt: {exc!r}")

        try:
            synthesizer = engines.synthesizer
            if synthesizer is None:
                raise RuntimeError(f"no TTS engine loaded: {engines.statuses.tts}")
            segment = TextSegment(PROBE_TEXT, 0, len(PROBE_TEXT))
            for _chunk in synthesizer.synthesize(segment, CancelToken()):
                pass
            result["tts_ok"] = True
        except Exception as exc:  # noqa: BLE001 -- see stt comment above
            errors.append(f"tts: {exc!r}")

        if torch.cuda.is_available():
            result["max_memory_reserved_mib"] = _mib(torch.cuda.max_memory_reserved())
            result["max_memory_allocated_mib"] = _mib(torch.cuda.max_memory_allocated())

        if errors:
            result["error"] = "; ".join(errors)

    except Exception:  # noqa: BLE001 -- top-level: always print one JSON line, never a traceback
        result["error"] = traceback.format_exc()

    print(json.dumps(result))


if __name__ == "__main__":
    main()
