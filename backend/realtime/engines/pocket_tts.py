"""Kyutai Pocket TTS with voice cloning behind `SpeechSynthesizer`: MIST speaking her own voice.

Engine id `pocket-clone`, on the CPU with 4 threads, the arm of the same name in
`scripts/model_bench/speech/speech_arms.json` (the parity test pins both to it). The voice is the
identity `PlacementConfig.tts_voice` names: one clip per speaking state, each turned into a
Pocket voice state by `PocketVoiceBuilder` and cached by `voice_assets.ensure_artefacts`.

Weights. pocket-tts 4.0.1's installed `english.yaml` names its files as three `hf://` URIs, which
would resolve through a Hugging Face download. `local_config` copies that config with each URI
mapped to its staged snapshot under `<models_dir>/speech/<org>/<repo>/`, as the bench's
`pocket-clone` adapter does: the cloning model from `kyutai/pocket-tts`, the tokenizer and the
fallback model from `kyutai/pocket-tts-without-voice-cloning`. Any other URI, or a mapped file
that is missing, is refused, so the config the model loads names no remote file.

Voice states. For each clip, `build` computes `get_state_for_audio_prompt(<clip>)` and writes it
with `pocket_tts.export_model_state`; `load` reads it back by passing the `.safetensors` path to
the same call, which opens a path with that suffix as a voice state rather than as audio to
clone. The transcript is never read: Pocket clones from audio alone. The adapter keeps the
states `ensure_artefacts` returns and never reads the cache again.

pocket-tts and torch are imported only by `import_pocket`, which `load` is handed, so importing
this module loads neither and tests pass a fake library.
"""

import copy
import importlib
import logging
import tempfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import yaml

from ..budget import PlacementConfig
from ..errors import SpeechEngineError
from ..speech import (
    NEUTRAL_STATE,
    SPEAKING_STATES,
    CancelToken,
    SynthChunk,
    TextSegment,
    VoiceInfo,
)
from ..voice_assets import VoiceAssetError, VoiceClip, ensure_artefacts, load_identity
from .audio import (
    OUTPUT_RATE,
    LoudnessNormalizer,
    pcm16_chunks,
    resample,
    speakable_text,
    to_mono_float,
    to_pcm16,
)

logger = logging.getLogger(__name__)

ENGINE_ID = "pocket-clone"
REPOSITORY = "kyutai/pocket-tts"
REVISION = "db04f0e653473e5c7333c69686ac92b7460b0c1d"
MODEL_PIN = f"{REPOSITORY}@{REVISION}"
DEVICE = "cpu"
THREADS = 4
ARTEFACT_VERSION = 1
VOICE_STATE_FILE = "voice_state.safetensors"

# The installed config `local_config` copies, from `pocket_tts.utils.config.CONFIGS_DIR`.
CONFIG_NAME = "english.yaml"
# The three `hf://` URIs pocket-tts 4.0.1's `english.yaml` names (host step H13), each matched
# whole, so another release's config is refused rather than loaded.
CLONE_MODEL_URI = f"hf://{REPOSITORY}/languages/english/model.safetensors@{REVISION}"
MODEL_URI = (
    "hf://kyutai/pocket-tts-without-voice-cloning/languages/english/model.safetensors"
    "@a7056f505a4a8ca72d8b5fb1440071d6f1a35739"
)
TOKENIZER_URI = (
    "hf://kyutai/pocket-tts-without-voice-cloning/languages/english/tokenizer.json"
    "@00eac05ed3d16bdc3f6b5d598874019c34a89214"
)
CONFIG_URIS = (CLONE_MODEL_URI, MODEL_URI, TOKENIZER_URI)
_REMOTE_PREFIXES = ("hf://", "http://", "https://")

# The streaming call the bench's Pocket adapter uses when the model has it; otherwise a segment
# is synthesized whole (`probes/tts_engines.py`, `POCKET_STREAM_METHOD`).
STREAM_METHOD = "generate_audio_stream"

# What pocket-tts and torch raise for a bad input, file or state. Converted to
# `SpeechEngineError`, so no library error crosses the lane (`lanes.py`) and none is bare
# `Exception`. AssertionError is here because the library checks shapes with `assert`.
_LIBRARY_ERRORS = (
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    OSError,
    AssertionError,
)


class VoiceState(Protocol):
    """A Pocket voice state, opaque here: `dict[str, dict[str, torch.Tensor]]` in 4.0.1."""


class PocketModel(Protocol):
    """The calls the adapter makes on a loaded `pocket_tts.TTSModel` (4.0.1 signatures)."""

    sample_rate: int
    has_voice_cloning: bool

    def get_state_for_audio_prompt(
        self, audio_conditioning: Path, truncate: bool = False
    ) -> VoiceState:
        """A voice state from a clip, or from a `.safetensors` path that holds one."""
        ...

    def generate_audio(self, model_state: VoiceState, text_to_generate: str) -> object:
        """The whole segment's audio, a 1-D tensor at `sample_rate`."""
        ...


@dataclass(frozen=True, slots=True)
class PocketLibrary:
    """What the adapter needs from pocket-tts and torch, bound once by `import_pocket`.

    `load_model` is the model factory: `TTSModel.load_model(config=<path>)`.
    `set_num_threads` is `torch.set_num_threads`, which is process-wide.
    """

    load_model: Callable[[Path], PocketModel]
    export_model_state: Callable[[VoiceState, Path], None]
    configs_dir: Path
    set_num_threads: Callable[[int], None]


def import_pocket() -> PocketLibrary:
    """Import pocket-tts and torch and bind what the adapter calls.

    Raises:
        SpeechEngineError: pocket-tts or torch is not installed.
    """
    try:
        torch = importlib.import_module("torch")
        package = importlib.import_module("pocket_tts")
        config = importlib.import_module("pocket_tts.utils.config")
    except ImportError as exc:
        raise SpeechEngineError(
            f"pocket-tts is not installed ({exc.name}); requirements-voice.txt pins it"
        ) from exc

    def load_model(path: Path) -> PocketModel:
        return package.TTSModel.load_model(config=path)

    return PocketLibrary(
        load_model=load_model,
        export_model_state=package.export_model_state,
        configs_dir=Path(config.CONFIGS_DIR),
        set_num_threads=torch.set_num_threads,
    )


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


def _local_path(uri: str, models_dir: Path) -> Path:
    """`hf://<org>/<repo>/<path>@<revision>` as `<models_dir>/speech/<org>/<repo>/<path>`."""
    org, repo, *path = uri.removeprefix("hf://").rsplit("@", 1)[0].split("/")
    return models_dir.joinpath("speech", org, repo, *path)


def local_config(installed: object, models_dir: Path) -> dict:
    """A copy of the installed config whose three `hf://` URIs are the staged files.

    Voice cloning stays on: `weights_path` is the cloning model.

    Raises:
        SpeechEngineError: The config does not name 4.0.1's three URIs where 4.0.1 does, it
            names another remote file, or a mapped file is missing; the message names the URI
            or the path.
    """
    name = f"{ENGINE_ID}: the installed {CONFIG_NAME}"
    if not isinstance(installed, dict):
        raise SpeechEngineError(f"{name} is not a mapping")
    flow_lm, mimi = installed.get("flow_lm"), installed.get("mimi")
    if not isinstance(flow_lm, dict) or not isinstance(mimi, dict):
        raise SpeechEngineError(f"{name} has no `flow_lm` or `mimi` section")
    lookup_table = flow_lm.get("lookup_table")
    if not isinstance(lookup_table, dict):
        raise SpeechEngineError(f"{name} has no `flow_lm.lookup_table` section")
    for key, found, expected in (
        ("weights_path", installed.get("weights_path"), CLONE_MODEL_URI),
        (
            "weights_path_without_voice_cloning",
            installed.get("weights_path_without_voice_cloning"),
            MODEL_URI,
        ),
        ("flow_lm.lookup_table.tokenizer_path", lookup_table.get("tokenizer_path"), TOKENIZER_URI),
    ):
        if found != expected:
            raise SpeechEngineError(
                f"{name} has {key} {found!r}; this adapter loads only {expected!r}, "
                "the value of pocket-tts 4.0.1"
            )
    local = {uri: _local_path(uri, Path(models_dir)) for uri in CONFIG_URIS}
    for path in local.values():
        if not path.is_file():
            raise SpeechEngineError(f"{ENGINE_ID}: weight file {path} is missing")
    config = copy.deepcopy(installed)
    config["weights_path"] = str(local[CLONE_MODEL_URI])
    config["weights_path_without_voice_cloning"] = str(local[MODEL_URI])
    config["flow_lm"]["lookup_table"]["tokenizer_path"] = str(local[TOKENIZER_URI])
    remote = sorted({s for s in _strings(config) if s.startswith(_REMOTE_PREFIXES)})
    if remote:
        raise SpeechEngineError(f"{name} names remote files this adapter refuses: {remote}")
    return config


def _library_error(action: str, exc: BaseException) -> SpeechEngineError:
    """A `SpeechEngineError` naming the action and the library error's type, never its text.

    A library message can echo its input, which for synthesis is reply text.
    """
    return SpeechEngineError(f"{ENGINE_ID}: {action} failed ({type(exc).__name__})")


class PocketVoiceBuilder:
    """Turns an identity clip into a Pocket voice state file, and reads it back.

    Holds the loaded model, since computing a voice state runs the model's audio encoder.
    """

    engine_id = ENGINE_ID
    model_pin = MODEL_PIN
    artefact_version = ARTEFACT_VERSION

    __slots__ = ("_export", "_model")

    def __init__(
        self, model: PocketModel, export_model_state: Callable[[VoiceState, Path], None]
    ) -> None:
        self._model = model
        self._export = export_model_state

    def build(self, clip: VoiceClip, out_dir: Path) -> None:
        """Compute the clip's voice state and export it; the transcript is not read.

        Raises:
            SpeechEngineError: pocket-tts could not compute or write the state.
        """
        try:
            state = self._model.get_state_for_audio_prompt(clip.audio)
            self._export(state, out_dir / VOICE_STATE_FILE)
        except _LIBRARY_ERRORS as exc:
            raise _library_error(f"voice state for state {clip.state!r}", exc) from exc

    def load(self, artefact_dir: Path) -> VoiceState:
        """Read the exported voice state.

        Raises:
            SpeechEngineError: The file is missing or pocket-tts could not read it.
        """
        path = artefact_dir / VOICE_STATE_FILE
        if not path.is_file():
            raise SpeechEngineError(f"{ENGINE_ID}: {VOICE_STATE_FILE} is missing")
        try:
            return self._model.get_state_for_audio_prompt(path)
        except _LIBRARY_ERRORS as exc:
            raise _library_error(f"reading {VOICE_STATE_FILE}", exc) from exc


class PocketCloneSynthesizer:
    """Pocket TTS speaking MIST's voice identity, one voice state per speaking state."""

    engine_id = ENGINE_ID
    model_id = MODEL_PIN
    sample_rate = OUTPUT_RATE

    __slots__ = (
        "_fallbacks_logged",
        "_model",
        "_normalizer",
        "_prompts",
        "_released",
        "states",
        "voice",
    )

    def __init__(
        self, model: PocketModel, prompts: Mapping[str, VoiceState], voice: VoiceInfo
    ) -> None:
        if NEUTRAL_STATE not in prompts:
            raise SpeechEngineError(f"{ENGINE_ID}: no voice state for {NEUTRAL_STATE!r}")
        self._model: PocketModel | None = model
        self._prompts: dict[str, VoiceState] = dict(prompts)
        self.voice = voice
        self.states: frozenset[str] = frozenset(self._prompts) & SPEAKING_STATES
        self._normalizer = LoudnessNormalizer()
        self._fallbacks_logged: set[str] = set()
        self._released = False

    @classmethod
    def load(
        cls,
        placement: PlacementConfig,
        profile: str,
        *,
        library_loader: Callable[[], PocketLibrary],
        device: str = DEVICE,
    ) -> "PocketCloneSynthesizer":
        """Load the identity, the model and every state's voice state.

        The identity loads first, so a bad identity fails before the model loads. Its voice
        states are built into `placement.voice_cache_dir` where missing, then read back under
        the cache lock; nothing is written under the identity.

        Raises:
            VoiceAssetError: No identity is configured, or it or its voice states are unusable;
                the message names the field, the relative path or the state.
            SpeechEngineError: The device is not `cpu`, pocket-tts is not installed, a weight
                file is missing, or the model failed to load or loaded without cloning.
        """
        if device != DEVICE:
            raise SpeechEngineError(f"{ENGINE_ID} runs on {DEVICE} only, not {device!r}")
        if placement.tts_voice is None:
            raise VoiceAssetError(
                f"{ENGINE_ID}: no voice identity is configured (MIST_VOICE_TTS_VOICE)"
            )
        identity = load_identity(Path(placement.voice_root), placement.tts_voice)
        library = library_loader()
        model = _load_model(library, Path(placement.models_dir))
        builder = PocketVoiceBuilder(model, library.export_model_state)
        prompts = ensure_artefacts(identity, builder, Path(placement.voice_cache_dir))
        voice = VoiceInfo(ENGINE_ID, identity.voice_id, profile)
        synthesizer = cls(model, prompts, voice)
        logger.info(
            "%s loaded voice %r on %s with %d threads: states %s",
            ENGINE_ID,
            identity.voice_id,
            DEVICE,
            THREADS,
            ", ".join(sorted(synthesizer.states)),
        )
        return synthesizer

    def synthesize(self, segment: TextSegment, token: CancelToken) -> Iterator[SynthChunk]:
        """Synthesize one segment in its speaking state, or in `neutral` when it has no clip.

        Raises:
            SpeechEngineError: The adapter was released, or the engine failed; the message
                holds no segment text.
        """
        if self._released:
            raise SpeechEngineError(f"{ENGINE_ID} was released")
        return self._synthesize(segment, token)

    def release(self) -> None:
        """Drop the model and the voice states; a second call does nothing."""
        if self._released:
            return
        self._released = True
        self._model = None
        self._prompts = {}
        logger.info("%s released voice %r", ENGINE_ID, self.voice.voice)

    def _voiced_state(self, requested: str) -> str:
        if requested in self._prompts:
            return requested
        if requested not in self._fallbacks_logged:
            self._fallbacks_logged.add(requested)
            if requested in SPEAKING_STATES:
                level, reason = logging.INFO, "has no clip for"
            else:
                level, reason = logging.WARNING, "was asked for a state outside the vocabulary,"
            logger.log(
                level,
                "%s voice %r %s state %r; voicing %r",
                ENGINE_ID,
                self.voice.voice,
                reason,
                requested,
                NEUTRAL_STATE,
            )
        return NEUTRAL_STATE

    def _synthesize(self, segment: TextSegment, token: CancelToken) -> Iterator[SynthChunk]:
        state = self._voiced_state(segment.state)
        text = speakable_text(segment.text, keep_tags=False)
        if not text or token.is_set():
            return
        model, prompt = self._model, self._prompts[state]
        pieces = []
        try:
            stream = getattr(model, STREAM_METHOD, None)
            if callable(stream):
                for piece in stream(prompt, text):
                    pieces.append(to_mono_float(piece))
                    if token.is_set():
                        return
            else:
                pieces.append(to_mono_float(model.generate_audio(prompt, text)))
        except _LIBRARY_ERRORS as exc:
            # `from None`: the library's message may echo the segment text.
            raise _library_error("synthesis", exc) from None
        if token.is_set() or not pieces:
            return
        samples = resample(np.concatenate(pieces), int(model.sample_rate))
        pcm = to_pcm16(self._normalizer(samples, OUTPUT_RATE))
        for chunk in pcm16_chunks(pcm, OUTPUT_RATE):
            if token.is_set():
                return
            yield SynthChunk(pcm16=chunk, samples=len(chunk) // 2, segment=segment, state=state)


def _load_model(library: PocketLibrary, models_dir: Path) -> PocketModel:
    """`library.load_model` on `local_config`'s copy of the installed config, on 4 threads.

    Raises:
        SpeechEngineError: A weight file is missing, the model failed to load, or it loaded
            without voice cloning.
    """
    config_path = library.configs_dir / CONFIG_NAME
    try:
        installed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SpeechEngineError(f"{ENGINE_ID}: cannot read {config_path}") from exc
    config = local_config(installed, models_dir)
    library.set_num_threads(THREADS)
    try:
        with tempfile.TemporaryDirectory(prefix="pocket-config-") as tmp:
            path = Path(tmp) / CONFIG_NAME
            path.write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
            )
            # The library reads the file during this call and never again for a voice state
            # passed as a path.
            model = library.load_model(path)
    except _LIBRARY_ERRORS as exc:
        raise _library_error("loading the model", exc) from exc
    if getattr(model, "has_voice_cloning", None) is not True:
        raise SpeechEngineError(f"{ENGINE_ID}: the model loaded without voice cloning")
    return model
