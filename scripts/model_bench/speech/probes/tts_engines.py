"""The speech synthesis adapters of run 1, each meeting `engines.SpeechEngine`.

`voices()` returns only the names `speech_arms.json` lists, and `load` refuses any other name
before importing anything. The four run-1 engines speak with a stock voice of their own, and none
of them hands its engine an audio file or an audio argument. Two arms do not (mist-realtime-voice
plan v6, T41): `pocket-clone` clones the one identity clip `--voice-clip` mounts read-only at
`CONTAINER_VOICE_CLIP`, its only voice being `identity`, and `qwen3-design` designs its voice from
the fixed description and seed of its arm. Engine packages and torch are imported lazily through
`importer`, inside the bench container only, so importing this module loads no engine and the
unit tests pass fake modules instead.

Weights are read from the read-only mount at `CONTAINER_MODELS_DIR` (layout `<org>/<repo>/`)
with `HF_HUB_OFFLINE=1`, so nothing is downloaded.

The faster-qwen3-tts calls and Pocket TTS's streaming call are written from the projects'
documentation (the mist-model-host plan v2, engine facts); neither package is installed anywhere
these tests run. Host step H2 confirms, against the installed packages, the faster-qwen3-tts
0.5.4 signatures and speaker ids, and Pocket TTS 4.0.1's streaming call. The Chatterbox calls
were read from chatterbox-tts 0.1.7 in the worker image.

Pocket TTS's offline load path was read from the installed 4.0.1 package. Its English config
names its files as `hf://` URIs, which resolve only through a Hugging Face cache, so the adapter
loads a copy of that config in which the model and tokenizer are the mounted files of the
ungated repository. Each stock Pocket voice is passed as the path of its precomputed voice state,
a safetensors file from the same repository and revision as the weights, which holds no audio.

`pocket-clone` loads the same installed config with all three of its `hf://` URIs mapped to the
mounted snapshots (host step H13 printed them from the pocket image): the cloning model from
`kyutai/pocket-tts`, the tokenizer and the fallback model from the ungated repository each URI
names. Its voice is `get_state_for_audio_prompt(<the mounted clip>)`. `qwen3-design` loads and
calls the VoiceDesign model through `design_probe.Designer`, the loading path T40's voice-design
tool uses, with the entry point H13 read from the qwen3 image.

`pocket-clone` can instead speak as an identity directory `--voice-dir` mounts at
`/voice/<voice_id>` (mist-voice-fidelity plan v5, M17): the runtime's `load_identity` reads it,
and its trained weights and tokenizer reach the model through the runtime adapter's own
`local_config`, which `tests/unit/realtime/test_engine_bench_parity.py` holds equal to the bench's.
"""

from __future__ import annotations

import copy
import importlib
import tempfile
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import yaml

from .. import core
from . import design_probe

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

# The installed config the adapter copies, from `pocket_tts.utils.config.CONFIGS_DIR`.
POCKET_CONFIG = "english.yaml"

# The two `hf://` URIs of Pocket TTS 4.0.1's `english.yaml` that name the ungated repository, each
# mapped to its file under the engine's weights directory. Each is matched whole (repository,
# path and revision), so another release's config is refused rather than loaded.
POCKET_MODEL_URI = (
    "hf://kyutai/pocket-tts-without-voice-cloning/languages/english/model.safetensors"
    "@a7056f505a4a8ca72d8b5fb1440071d6f1a35739"
)
POCKET_TOKENIZER_URI = (
    "hf://kyutai/pocket-tts-without-voice-cloning/languages/english/tokenizer.json"
    "@00eac05ed3d16bdc3f6b5d598874019c34a89214"
)
POCKET_LOCAL_FILES = {
    POCKET_MODEL_URI: "languages/english/model.safetensors",
    POCKET_TOKENIZER_URI: "languages/english/tokenizer.json",
}

# The stock voices' precomputed states, under the engine's weights directory.
POCKET_VOICE_STATES = "languages/english/embeddings"

# The installed config's `weights_path`: the cloning model of the gated repository, at the
# revision the `pocket-clone` arm records. `pocket-clone` maps exactly these three URIs.
POCKET_CLONE_MODEL_URI = (
    "hf://kyutai/pocket-tts/languages/english/model.safetensors"
    "@db04f0e653473e5c7333c69686ac92b7460b0c1d"
)
POCKET_CLONE_URIS = (POCKET_CLONE_MODEL_URI, POCKET_MODEL_URI, POCKET_TOKENIZER_URI)

# Fixed container path of the identity clip `--voice-clip` mounts read-only for `pocket-clone`.
CONTAINER_VOICE_CLIP = "/voice/identity.wav"

# Where `--voice-dir` mounts an identity directory read-only for `pocket-clone`: at
# `<root>/<voice_id>`, since `load_identity` refuses an identity whose `voice_id` is not its
# directory's name.
CONTAINER_VOICE_ROOT = "/voice"

# The runtime adapter's modules, imported through the adapter's importer inside the bench
# container only, so importing this module on the host loads neither (nor numpy).
VOICE_ASSETS_MODULE = "backend.realtime.voice_assets"
RUNTIME_POCKET_MODULE = "backend.realtime.engines.pocket_tts"
# The runtime adapter reads its weights from `<models_dir>/speech/<org>/<repo>/`; the bench
# mounts that `speech` directory at `CONTAINER_MODELS_DIR`.
RUNTIME_SPEECH_DIR = "speech"


def container_voice_dir(voice_id: str) -> str:
    """The container path an identity directory is mounted at."""
    return f"{CONTAINER_VOICE_ROOT}/{voice_id}"


# Values `pocket_tts.utils.utils.download_if_necessary` fetches rather than opens.
_REMOTE_PREFIXES = ("hf://", "http://", "https://")


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


def _strings(value: object) -> Iterator[str]:
    """Every string in a parsed YAML document, mapping keys included."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


class PocketEngine(StockVoiceEngine):
    """Pocket TTS on the CPU, from the mounted ungated English model, with a stock voice state.

    4.0.1 loads its model weights only from `weights_path`, the gated repository's URI, and falls
    back to `weights_path_without_voice_cloning` only when that download raises. A local path
    never raises, so the copy sets `weights_path` to the mounted ungated file and the fallback to
    null: a failed load then raises instead of loading a second file.
    """

    def _load_model(self) -> object:
        model = self._load_local_model()
        # The state upstream leaves after loading this same file through its fallback.
        model.has_voice_cloning = False
        return model

    def _load_local_model(self) -> object:
        """`TTSModel.load_model` on `_local_config`'s copy of the installed config."""
        torch = self._import("torch")
        torch.set_num_threads(int(self._cfg["threads"]))
        package = self._import("pocket_tts")
        configs_dir = Path(self._import("pocket_tts.utils.config").CONFIGS_DIR)
        installed = yaml.safe_load((configs_dir / POCKET_CONFIG).read_text(encoding="utf-8"))
        config = self._local_config(installed)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / POCKET_CONFIG
            with path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
            # `load_config` reads the file during this call. The model keeps its path as
            # `origin`, read only for a voice passed by name, which this adapter never does.
            model = package.TTSModel.load_model(config=path)
        self.sample_rate = int(model.sample_rate)
        self.streaming = callable(getattr(model, POCKET_STREAM_METHOD, None))
        return model

    def _local_config(self, installed: object) -> dict:
        """A copy of the installed config whose files are all under the mounted weights.

        Raises:
            SpeechBenchError: the installed config does not have 4.0.1's URIs and layout, a
                mapped file is missing, or the copy still names a remote file.
        """
        name = f"{self.name}: the installed {POCKET_CONFIG}"
        if not isinstance(installed, dict):
            raise core.SpeechBenchError(f"{name} is not a mapping")
        flow_lm, mimi = installed.get("flow_lm"), installed.get("mimi")
        if not isinstance(flow_lm, dict) or not isinstance(mimi, dict):
            raise core.SpeechBenchError(f"{name} has no `flow_lm` or `mimi` section")
        lookup_table = flow_lm.get("lookup_table")
        tokenizer = lookup_table.get("tokenizer_path") if isinstance(lookup_table, dict) else None
        for key, found, expected in (
            (
                "weights_path_without_voice_cloning",
                installed.get("weights_path_without_voice_cloning"),
                POCKET_MODEL_URI,
            ),
            ("flow_lm.lookup_table.tokenizer_path", tokenizer, POCKET_TOKENIZER_URI),
        ):
            if found != expected:
                raise core.SpeechBenchError(
                    f"{name} has {key} {found!r}; this adapter loads only {expected!r}, "
                    "the value of Pocket TTS 4.0.1"
                )
        for section, values in (("flow_lm", flow_lm), ("mimi", mimi)):
            if values.get("weights_path") is not None:
                raise core.SpeechBenchError(
                    f"{name} sets {section}.weights_path; Pocket TTS 4.0.1 leaves it unset"
                )
        model_file, tokenizer_file = (
            self.weights_dir() / POCKET_LOCAL_FILES[uri]
            for uri in (POCKET_MODEL_URI, POCKET_TOKENIZER_URI)
        )
        for path in (model_file, tokenizer_file):
            if not path.is_file():
                raise core.SpeechBenchError(f"{self.name}: {path} is not in the mounted weights")
        config = copy.deepcopy(installed)
        # Replaced by key: the installed value, the gated repository's URI, is never read.
        config["weights_path"] = str(model_file)
        config["weights_path_without_voice_cloning"] = None
        config["flow_lm"]["lookup_table"]["tokenizer_path"] = str(tokenizer_file)
        remote = sorted({s for s in _strings(config) if s.startswith(_REMOTE_PREFIXES)})
        if remote:
            raise core.SpeechBenchError(f"{name} still names remote files: {remote}")
        return config

    def _select_voice(self, voice: str) -> None:
        # A `Path` ending in `.safetensors` is opened as a voice state; a name would resolve
        # through an `hf://` URI instead.
        state = self.weights_dir() / POCKET_VOICE_STATES / f"{voice}.safetensors"
        if not state.is_file():
            raise VoiceError(f"{self.name}: stock voice {voice!r} has no state file at {state}")
        self._voice_state = self._model.get_state_for_audio_prompt(state)

    def _stream(self, text: str) -> Iterator[numpy.ndarray]:
        if self.streaming:
            for chunk in getattr(self._model, POCKET_STREAM_METHOD)(self._voice_state, text):
                yield to_mono(chunk, self._import)
            return
        yield to_mono(self._model.generate_audio(self._voice_state, text), self._import)


def _hf_local_path(uri: str, models_dir: Path) -> Path:
    """`hf://<org>/<repo>/<path>@<revision>` as `<models dir>/<org>/<repo>/<path>`."""
    org, repo, *path = uri.removeprefix("hf://").rsplit("@", 1)[0].split("/")
    return models_dir.joinpath(org, repo, *path)


class PocketCloneEngine(PocketEngine):
    """Pocket TTS on the CPU with voice cloning on, speaking as the mounted identity clip, or as
    a mounted identity directory with its trained weights.

    The gated repository's cloning model is the one 4.0.1 loads from `weights_path`, so the copy
    maps that URI and the other two to their mounted snapshots and keeps cloning on. A model that
    loads without cloning anyway is refused: it cannot speak as the clip.

    With `voice_id` (`--voice-dir`, plan v5 M17), the identity at `<voice_root>/<voice_id>` is
    loaded by the runtime's `load_identity`, checked by the runtime adapter's own rules, and the
    model config is the runtime adapter's `local_config` with the identity's weights and
    tokenizer, so the bench loads exactly the files the runtime would. Its voice is the
    identity's neutral clip, the state every other one falls back to.
    """

    def __init__(
        self,
        name: str,
        cfg: Mapping[str, object],
        *,
        models_dir: Path = Path(CONTAINER_MODELS_DIR),
        importer: Importer = importlib.import_module,
        voice_clip: Path = Path(CONTAINER_VOICE_CLIP),
        voice_root: Path = Path(CONTAINER_VOICE_ROOT),
        voice_id: str | None = None,
    ):
        super().__init__(name, cfg, models_dir=models_dir, importer=importer)
        self.voice_clip = voice_clip
        self.voice_root = voice_root
        self.voice_id = voice_id
        self._identity: object | None = None

    def _load_model(self) -> object:
        model = self._load_local_model()
        if getattr(model, "has_voice_cloning", None) is not True:
            raise core.SpeechBenchError(f"{self.name}: the model loaded without voice cloning")
        return model

    def _load_identity(self) -> object:
        """The identity `voice_id` names, loaded once and checked as the runtime adapter checks it.

        Raises:
            VoiceError: the identity is unusable, or its trained weights are not for this
                engine, base or tokenizer; the message names a manifest field or a relative
                path, never a path under the private root.
        """
        if self._identity is not None:
            return self._identity
        assets = self._import(VOICE_ASSETS_MODULE)
        runtime = self._import(RUNTIME_POCKET_MODULE)
        try:
            identity = assets.load_identity(self.voice_root, self.voice_id)
            # The runtime's own check of trained weights against this engine, its base and the
            # stock tokenizer; it runs before the model loads, as it does at runtime.
            runtime._model_pin(identity, self._runtime_models_dir())
        except runtime.SpeechEngineError as exc:
            raise VoiceError(f"{self.name}: {exc}") from exc
        self._identity = identity
        return identity

    def _runtime_models_dir(self) -> Path:
        """The runtime's `models_dir`, whose `speech` directory is the bench's mounted one."""
        if self._models_dir.name != RUNTIME_SPEECH_DIR:
            raise core.SpeechBenchError(
                f"{self.name}: an identity needs the models directory mounted as "
                f"`<models>/{RUNTIME_SPEECH_DIR}`, the runtime adapter's layout"
            )
        return self._models_dir.parent

    def _local_config(self, installed: object) -> dict:
        """The runtime adapter's config for the identity, or else the clip's config."""
        if self.voice_id is None:
            return self._clip_config(installed)
        models_dir = self._runtime_models_dir()
        trained = self._load_identity().model
        runtime = self._import(RUNTIME_POCKET_MODULE)
        try:
            return runtime.local_config(
                installed,
                models_dir,
                None if trained is None else trained.weights_path,
                None if trained is None else trained.tokenizer_path,
            )
        except runtime.SpeechEngineError as exc:
            raise core.SpeechBenchError(str(exc)) from exc

    def _clip_config(self, installed: object) -> dict:
        """A copy of the installed config whose three `hf://` URIs are the mounted snapshots.

        Raises:
            SpeechBenchError: the installed config does not name 4.0.1's three URIs where 4.0.1
                does, a mapped file is missing, or the copy still names a remote file.
        """
        name = f"{self.name}: the installed {POCKET_CONFIG}"
        if not isinstance(installed, dict):
            raise core.SpeechBenchError(f"{name} is not a mapping")
        flow_lm, mimi = installed.get("flow_lm"), installed.get("mimi")
        if not isinstance(flow_lm, dict) or not isinstance(mimi, dict):
            raise core.SpeechBenchError(f"{name} has no `flow_lm` or `mimi` section")
        lookup_table = flow_lm.get("lookup_table")
        if not isinstance(lookup_table, dict):
            raise core.SpeechBenchError(f"{name} has no `flow_lm.lookup_table` section")
        for key, found, expected in (
            ("weights_path", installed.get("weights_path"), POCKET_CLONE_MODEL_URI),
            (
                "weights_path_without_voice_cloning",
                installed.get("weights_path_without_voice_cloning"),
                POCKET_MODEL_URI,
            ),
            (
                "flow_lm.lookup_table.tokenizer_path",
                lookup_table.get("tokenizer_path"),
                POCKET_TOKENIZER_URI,
            ),
        ):
            if found != expected:
                raise core.SpeechBenchError(
                    f"{name} has {key} {found!r}; this adapter loads only {expected!r}, "
                    "the value of Pocket TTS 4.0.1"
                )
        for section, values in (("flow_lm", flow_lm), ("mimi", mimi)):
            if values.get("weights_path") is not None:
                raise core.SpeechBenchError(
                    f"{name} sets {section}.weights_path; Pocket TTS 4.0.1 leaves it unset"
                )
        local = {uri: _hf_local_path(uri, self._models_dir) for uri in POCKET_CLONE_URIS}
        for path in local.values():
            if not path.is_file():
                raise core.SpeechBenchError(f"{self.name}: {path} is not in the mounted weights")
        config = copy.deepcopy(installed)
        config["weights_path"] = str(local[POCKET_CLONE_MODEL_URI])
        config["weights_path_without_voice_cloning"] = str(local[POCKET_MODEL_URI])
        config["flow_lm"]["lookup_table"]["tokenizer_path"] = str(local[POCKET_TOKENIZER_URI])
        remote = sorted({s for s in _strings(config) if s.startswith(_REMOTE_PREFIXES)})
        if remote:
            raise core.SpeechBenchError(f"{name} still names remote files: {remote}")
        return config

    def _select_voice(self, voice: str) -> None:
        # A `Path` not ending in `.safetensors` is read as audio to clone.
        if self.voice_id is not None:
            identity = self._load_identity()
            neutral = self._import(VOICE_ASSETS_MODULE).NEUTRAL_STATE
            self._voice_state = self._model.get_state_for_audio_prompt(identity.clip(neutral).audio)
            return
        if not self.voice_clip.is_file():
            raise VoiceError(
                f"{self.name}: no identity clip is mounted at {self.voice_clip}; "
                "run with --voice-clip"
            )
        self._voice_state = self._model.get_state_for_audio_prompt(self.voice_clip)


class Qwen3DesignEngine(StockVoiceEngine):
    """Qwen3-TTS 1.7B VoiceDesign through `qwen_tts`, voicing every text from one description.

    Each call designs its voice afresh, so torch is seeded with the arm's seed before every call,
    as the voice-design tool seeds each candidate. Whole-segment: it yields once per text.
    """

    def _load_model(self) -> object:
        return design_probe.Designer(self.weights_dir(), self._import)

    def _stream(self, text: str) -> Iterator[numpy.ndarray]:
        params = self.params()
        audio, rate = self._model.design(
            description=str(params["instruct"]), text=text, seed=int(params["seed"])
        )
        self.sample_rate = int(rate)
        yield to_mono(audio, self._import)


ADAPTERS: dict[str, type[StockVoiceEngine]] = {
    "qwen3": Qwen3Engine,
    "turbo": TurboEngine,
    "original": OriginalEngine,
    "pocket": PocketEngine,
    "pocket-clone": PocketCloneEngine,
    "qwen3-design": Qwen3DesignEngine,
}


def make_engine(
    engine_id: str,
    *,
    models_dir: Path = Path(CONTAINER_MODELS_DIR),
    arms: Mapping[str, object] | None = None,
    importer: Importer = importlib.import_module,
    voice_id: str | None = None,
) -> StockVoiceEngine:
    """The adapter for `engine_id`, configured from the `tts` section of `speech_arms.json`.

    `voice_id` names the identity mounted under `CONTAINER_VOICE_ROOT`, for `pocket-clone` only.
    """
    tts = (arms if arms is not None else core.load_json(core.SPEECH_ARMS_PATH))["tts"]
    if engine_id not in ADAPTERS or engine_id not in tts:
        raise core.SpeechBenchError(
            f"unknown speech engine {engine_id!r}; known: {sorted(set(ADAPTERS) & set(tts))}"
        )
    if voice_id is None:
        return ADAPTERS[engine_id](
            engine_id, tts[engine_id], models_dir=models_dir, importer=importer
        )
    if ADAPTERS[engine_id] is not PocketCloneEngine:
        raise core.SpeechBenchError(
            f"{engine_id} speaks a voice of its own; an identity is for {core.VOICE_CLIP_ENGINE}"
        )
    return PocketCloneEngine(
        engine_id, tts[engine_id], models_dir=models_dir, importer=importer, voice_id=voice_id
    )
