"""The four speech synthesis adapters of run 1, each meeting `engines.SpeechEngine`.

Every engine speaks with a stock voice of its own and nothing is cloned: `voices()` returns only
the stock names `speech_arms.json` lists, `load` refuses any other name before importing
anything, and no adapter hands its engine an audio file or an audio argument. Engine packages and
torch are imported lazily through `importer`, inside the bench container only, so importing this
module loads no engine and the unit tests pass fake modules instead.

Weights are read from the read-only mount at `CONTAINER_MODELS_DIR` (layout `<org>/<repo>/`)
with `HF_HUB_OFFLINE=1`, so nothing is downloaded.

The calls for faster-qwen3-tts and pocket-tts are written from the projects' documentation (the
mist-model-host plan v2, engine facts); neither package is installed anywhere these tests run.
Host step H2 confirms, against the installed packages, the faster-qwen3-tts 0.5.4 signatures and
speaker ids, and Pocket TTS 4.0.1's streaming call and its offline load path. The Chatterbox
calls were read from chatterbox-tts 0.1.7 in the worker image.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from .. import core

if TYPE_CHECKING:
    import numpy

Importer = Callable[[str], ModuleType]

# Fixed container path of the host's speech models directory, mounted read-only.
CONTAINER_MODELS_DIR = "/models/speech"

# Setting keys that describe how a setting is used rather than what the engine is called with.
_SETTING_META = ("lines", "timed")

# The streaming call pocket-tts 4.0.1 is expected to carry; the README names audio streaming but
# shows no call, so the adapter streams only if the loaded model has this method and otherwise
# synthesizes whole segments. H2 confirms the name.
POCKET_STREAM_METHOD = "generate_audio_stream"


class VoiceError(core.SpeechBenchError):
    """A voice name the engine does not list as one of its stock voices."""


class SettingError(core.SpeechBenchError):
    """A setting name the engine has no entry for in `speech_arms.json`."""


def to_mono(audio: object, importer: Importer) -> numpy.ndarray:
    """A 1-D float32 array from an engine's numpy array or torch tensor."""
    np = importer("numpy")
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().float().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


class StockVoiceEngine:
    """The shared part of every adapter: stock voices, settings and the load-once rule."""

    def __init__(
        self,
        name: str,
        cfg: Mapping[str, object],
        *,
        models_dir: Path = Path(CONTAINER_MODELS_DIR),
        importer: Importer = importlib.import_module,
    ):
        self.name = name
        self.device = str(cfg["device"])
        self.streaming = bool(cfg.get("streaming", False))
        self.sample_rate = 24000
        self.voice: str | None = None
        self.setting = "default"
        self._cfg = cfg
        self._voices = [str(v) for v in cfg["voices"]]
        self._settings = dict(cfg["settings"])
        self._models_dir = models_dir
        self._import = importer
        self._model: object | None = None

    def voices(self) -> list[str]:
        """Return the engine's stock voice names."""
        return list(self._voices)

    def select_setting(self, name: str) -> None:
        """Use the named setting from `speech_arms.json` for the following calls."""
        if name not in self._settings:
            raise SettingError(
                f"{self.name} has no setting {name!r}; it has {list(self._settings)}"
            )
        self.setting = name

    def params(self) -> dict[str, object]:
        """The current setting's engine parameters."""
        raw = self._settings[self.setting]
        return {k: v for k, v in raw.items() if k not in _SETTING_META}

    def weights_dir(self) -> Path:
        """`<models dir>/<org>/<repo>` for the engine's repository."""
        repository = str(self._cfg["model"]["repository"])
        return self._models_dir.joinpath(*repository.split("/"))

    def load(self, voice: str) -> None:
        """Load the model once, then select the stock voice `voice`."""
        if voice not in self._voices:
            raise VoiceError(
                f"{self.name} has no stock voice {voice!r}; its voices are {self._voices}"
            )
        if self._model is None:
            self._model = self._load_model()
        self._select_voice(voice)
        self.voice = voice

    def stream(self, text: str) -> Iterator[numpy.ndarray]:
        """Synthesize `text` in the loaded voice."""
        if self._model is None or self.voice is None:
            raise core.SpeechBenchError(f"{self.name}: load a voice before synthesizing")
        return self._stream(text)

    def _load_model(self) -> object:
        raise NotImplementedError

    def _select_voice(self, voice: str) -> None:
        """Per-voice preparation; the default needs none."""

    def _stream(self, text: str) -> Iterator[numpy.ndarray]:
        raise NotImplementedError


class Qwen3Engine(StockVoiceEngine):
    """Qwen3-TTS 0.6B CustomVoice through faster-qwen3-tts, streaming with a preset speaker.

    No `instruct` text is sent: the 0.6B CustomVoice model has no instruction control, so its
    delivery comes from the text and the speaker only.
    """

    def _load_model(self) -> object:
        torch = self._import("torch")
        package = self._import("faster_qwen3_tts")
        params = self.params()
        model = package.FasterQwen3TTS.from_pretrained(
            str(self.weights_dir()), device=self.device, dtype=getattr(torch, str(params["dtype"]))
        )
        # Captures the CUDA graphs, so it belongs to load time, not to the first timed line.
        model.warmup()
        return model

    def _stream(self, text: str) -> Iterator[numpy.ndarray]:
        params = self.params()
        for chunk, sample_rate, _timing in self._model.generate_custom_voice_streaming(
            text,
            speaker=self.voice,
            language=params["language"],
            chunk_size=params["chunk_size"],
        ):
            self.sample_rate = int(sample_rate)
            yield to_mono(chunk, self._import)


class ChatterboxEngine(StockVoiceEngine):
    """A Chatterbox model with the built-in voice its checkpoint ships in `conds.pt`.

    `from_local` loads `conds.pt` as the built-in voice, and `generate` without an audio
    argument speaks with it, so the adapter never passes one. Whole-segment: it yields once.
    """

    module = ""
    class_name = ""

    def _load_model(self) -> object:
        package = self._import(self.module)
        model = getattr(package, self.class_name).from_local(self.weights_dir(), self.device)
        self.sample_rate = int(model.sr)
        return model

    def _stream(self, text: str) -> Iterator[numpy.ndarray]:
        yield to_mono(self._model.generate(text, **self.params()), self._import)


class OriginalEngine(ChatterboxEngine):
    """`chatterbox.tts.ChatterboxTTS`, at MIST's settings or the `expressive` setting."""

    module = "chatterbox.tts"
    class_name = "ChatterboxTTS"


class TurboEngine(ChatterboxEngine):
    """`chatterbox.tts_turbo.ChatterboxTurboTTS`: text only, its tags are its emotion control."""

    module = "chatterbox.tts_turbo"
    class_name = "ChatterboxTurboTTS"


class PocketEngine(StockVoiceEngine):
    """Pocket TTS on the CPU with a pre-made voice, passed by name only."""

    def _load_model(self) -> object:
        torch = self._import("torch")
        torch.set_num_threads(int(self._cfg["threads"]))
        package = self._import("pocket_tts")
        # H2 confirms that `load_model()` reads the mounted weights offline, and which
        # repository 4.0.1 loads.
        model = package.TTSModel.load_model()
        self.sample_rate = int(model.sample_rate)
        self.streaming = callable(getattr(model, POCKET_STREAM_METHOD, None))
        return model

    def _select_voice(self, voice: str) -> None:
        self._voice_state = self._model.get_state_for_audio_prompt(voice)

    def _stream(self, text: str) -> Iterator[numpy.ndarray]:
        if self.streaming:
            for chunk in getattr(self._model, POCKET_STREAM_METHOD)(self._voice_state, text):
                yield to_mono(chunk, self._import)
            return
        yield to_mono(self._model.generate_audio(self._voice_state, text), self._import)


ADAPTERS: dict[str, type[StockVoiceEngine]] = {
    "qwen3": Qwen3Engine,
    "turbo": TurboEngine,
    "original": OriginalEngine,
    "pocket": PocketEngine,
}


def make_engine(
    engine_id: str,
    *,
    models_dir: Path = Path(CONTAINER_MODELS_DIR),
    arms: Mapping[str, object] | None = None,
    importer: Importer = importlib.import_module,
) -> StockVoiceEngine:
    """The adapter for `engine_id`, configured from the `tts` section of `speech_arms.json`."""
    tts = (arms if arms is not None else core.load_json(core.SPEECH_ARMS_PATH))["tts"]
    if engine_id not in ADAPTERS or engine_id not in tts:
        raise core.SpeechBenchError(
            f"unknown speech engine {engine_id!r}; known: {sorted(set(ADAPTERS) & set(tts))}"
        )
    return ADAPTERS[engine_id](engine_id, tts[engine_id], models_dir=models_dir, importer=importer)
