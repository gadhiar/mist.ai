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

Trained weights. A schema-2 identity carries a student trained on MIST's voice. Its weights
replace the cloning model in `weights_path`, and its tokenizer, when it carries one, replaces the
stock tokenizer; without one, the stock tokenizer's sha256 must be the one the student was
trained with. The identity must name this engine and `MODEL_PIN` as its base, and the model must
run at 24 kHz. The pin then becomes `MODEL_PIN+<sha256>`, so the voice states rebuild for new
weights, and the turn record and the engine status name the trained weights.

Voice states. For each clip, `build` computes `get_state_for_audio_prompt(<clip>)` and writes it
with `pocket_tts.export_model_state`; `load` reads it back by passing the `.safetensors` path to
the same call, which opens a path with that suffix as a voice state rather than as audio to
clone. The transcript is never read: Pocket clones from audio alone. The adapter keeps the
states `ensure_artefacts` returns and never reads the cache again.

Loudness. Each voice state has one gain, measured when its artefacts are built and written
beside it as `gain.json` (`state`, `gain_db`, `calibration_sha`): the state renders the fixed
`CALIBRATION_SENTENCES` once per seed in `CALIBRATION_SEEDS`, torch seeded before each render
set, each set's integrated loudness is measured, and the gain brings the median of the three to
-20 LUFS, capped at +20 dB (`audio.target_gain_db`). One measured gain per state, rather than one
per segment, is what lets a segment leave before the engine has finished it. `CALIBRATION_SHA`
pins the recipe; changing it means bumping `ARTEFACT_VERSION`, so every cached gain rebuilds,
and `load` refuses a `gain.json` from another recipe.

Streaming. With the model's streaming call, each piece it yields is resampled, scaled by the
state's gain, passed through a `character.PeakLimiter` and cut into chunks of at most 200 ms
before the next piece is pulled, so the first chunk leaves after Pocket's first piece. The
limiter is the one the character stage ends with, one per segment, its state carried across
the segment's pieces, so a line louder than its state's calibration is limited, not clipped.
The cancel token is checked after every piece and between chunks, and the engine's stream is
closed when synthesis stops early. Without the streaming call, `generate_audio` returns the
whole segment as one piece, which goes through the same steps.

pocket-tts and torch are imported only by `import_pocket`, which `load` is handed, so importing
this module loads neither and tests pass a fake library.
"""

import copy
import hashlib
import importlib
import json
import logging
import math
import tempfile
from collections.abc import Callable, Generator, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import yaml

from ..budget import LimiterConfig, PlacementConfig
from ..character import PeakLimiter
from ..errors import SpeechEngineError
from ..speech import (
    NEUTRAL_STATE,
    SPEAKING_STATES,
    CancelToken,
    SynthChunk,
    TextSegment,
    VoiceInfo,
)
from ..voice_assets import (
    SHA_PREFIX,
    VoiceAssetError,
    VoiceClip,
    VoiceIdentity,
    ensure_artefacts,
    load_identity,
)
from .audio import (
    MAX_GAIN_DB,
    OUTPUT_RATE,
    TARGET_LUFS,
    integrated_loudness,
    pcm16_chunks,
    resample,
    speakable_text,
    target_gain_db,
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
# 2: `gain.json`, the state's calibrated gain, beside the voice state.
ARTEFACT_VERSION = 2
VOICE_STATE_FILE = "voice_state.safetensors"
GAIN_FILE = "gain.json"

# What every voice state is measured on: plain, public sentences of ordinary length, rendered
# once per seed. Sampling makes each render's level vary, and three seeds' median holds the
# gain steady against one loud or quiet render.
CALIBRATION_SENTENCES = (
    "The morning train leaves the station at a quarter past eight.",
    "Please remember to water the plants on the kitchen window sill.",
    "A cup of warm tea and a quiet hour can make a long evening feel slower.",
)
CALIBRATION_SEEDS = (0, 1, 2)
CALIBRATION_SHA = hashlib.sha256(
    json.dumps(
        {
            "sentences": CALIBRATION_SENTENCES,
            "seeds": CALIBRATION_SEEDS,
            "target_lufs": TARGET_LUFS,
            "max_gain_db": MAX_GAIN_DB,
        },
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()
"""The calibration recipe's pin, written into each `gain.json` and checked by `load`."""

# The limiter that ends the character stage, at its default settings: a soft knee at -1 dBFS,
# no look-ahead, so it adds no latency to the first chunk.
LIMITER = LimiterConfig()
# Pocket's codec rate. Trained weights at another rate are not the base's architecture.
MODEL_RATE = 24000

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
    `set_num_threads` is `torch.set_num_threads`, which is process-wide. `manual_seed` is
    `torch.manual_seed`, which seeds the generator Pocket samples from; the calibration seeds it.
    """

    load_model: Callable[[Path], PocketModel]
    export_model_state: Callable[[VoiceState, Path], None]
    configs_dir: Path
    set_num_threads: Callable[[int], None]
    manual_seed: Callable[[int], object]


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
        manual_seed=torch.manual_seed,
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


def local_config(
    installed: object,
    models_dir: Path,
    weights: Path | None = None,
    tokenizer: Path | None = None,
) -> dict:
    """A copy of the installed config whose three `hf://` URIs are the staged files.

    Voice cloning stays on: `weights_path` is the cloning model, or `weights`, an identity's
    trained weights. `tokenizer`, when given, replaces the stock tokenizer the same way.
    `weights_path_without_voice_cloning` is always the stock file.

    Raises:
        SpeechEngineError: The config does not name 4.0.1's three URIs where 4.0.1 does, it
            names another remote file, or a file it maps to is missing; the message names the
            URI or the staged path, and never the path of a given file, which is private.
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
    given = {
        uri: Path(path)
        for uri, path in ((CLONE_MODEL_URI, weights), (TOKENIZER_URI, tokenizer))
        if path is not None
    }
    for uri, path in given.items():
        if not path.is_file():
            raise SpeechEngineError(f"{ENGINE_ID}: a given file for {uri} is missing")
    staged = {uri: _local_path(uri, Path(models_dir)) for uri in CONFIG_URIS if uri not in given}
    for path in staged.values():
        if not path.is_file():
            raise SpeechEngineError(f"{ENGINE_ID}: weight file {path} is missing")
    local = staged | given
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


@dataclass(frozen=True, slots=True)
class PocketVoice:
    """One speaking state's artefacts: its voice state and the gain it is spoken at."""

    state: VoiceState
    gain_db: float


def _pieces(model: PocketModel, prompt: VoiceState, text: str) -> Generator[np.ndarray, None, None]:
    """The model's audio for `text` as mono float pieces at its own rate, as it produces them.

    The streaming call's pieces when the model has it, else `generate_audio`'s whole segment
    as one piece. Closing this closes the engine's stream.
    """
    stream = getattr(model, STREAM_METHOD, None)
    if not callable(stream):
        yield to_mono_float(model.generate_audio(prompt, text))
        return
    pieces = iter(stream(prompt, text))
    try:
        for piece in pieces:
            yield to_mono_float(piece)
    finally:
        close = getattr(pieces, "close", None)
        if callable(close):
            close()


class PocketVoiceBuilder:
    """Turns an identity clip into a Pocket voice state file and its gain, and reads them back.

    Holds the loaded model, since computing a voice state runs the model's audio encoder and
    the gain renders speech with it. `model_pin` is `MODEL_PIN`, or `MODEL_PIN+<sha256>` for an
    identity's trained weights. `manual_seed` seeds the model's sampling before each set of
    calibration renders.
    """

    engine_id = ENGINE_ID
    artefact_version = ARTEFACT_VERSION

    __slots__ = ("_export", "_manual_seed", "_model", "_model_pin")

    def __init__(
        self,
        model: PocketModel,
        export_model_state: Callable[[VoiceState, Path], None],
        *,
        manual_seed: Callable[[int], object],
        model_pin: str = MODEL_PIN,
    ) -> None:
        self._model = model
        self._export = export_model_state
        self._manual_seed = manual_seed
        self._model_pin = model_pin

    @property
    def model_pin(self) -> str:
        """The weights the voice states are computed with; part of their cache key."""
        return self._model_pin

    def build(self, clip: VoiceClip, out_dir: Path) -> None:
        """Compute the clip's voice state and export it, then calibrate its gain.

        The transcript is not read. `gain.json` is written last, so a build that fails leaves
        no gain behind.

        Raises:
            SpeechEngineError: pocket-tts could not compute or write the state, a calibration
                render failed, or every render is silent; the message names the state.
        """
        try:
            state = self._model.get_state_for_audio_prompt(clip.audio)
            self._export(state, out_dir / VOICE_STATE_FILE)
        except _LIBRARY_ERRORS as exc:
            raise _library_error(f"voice state for state {clip.state!r}", exc) from exc
        try:
            levels = [self._calibration_loudness(state, seed) for seed in CALIBRATION_SEEDS]
        except _LIBRARY_ERRORS as exc:
            raise _library_error(f"calibrating state {clip.state!r}", exc) from exc
        loudness = float(np.median(levels))
        if not math.isfinite(loudness):
            raise SpeechEngineError(
                f"{ENGINE_ID}: the calibration renders of state {clip.state!r} are silent"
            )
        gain_db = target_gain_db(loudness)
        gain = {"state": clip.state, "gain_db": gain_db, "calibration_sha": CALIBRATION_SHA}
        (out_dir / GAIN_FILE).write_text(json.dumps(gain, indent=2) + "\n", encoding="utf-8")
        logger.info(
            "%s calibrated state %r at %.1f LUFS: gain %+.1f dB",
            ENGINE_ID,
            clip.state,
            loudness,
            gain_db,
        )

    def _calibration_loudness(self, state: VoiceState, seed: int) -> float:
        """The integrated loudness of every calibration sentence rendered after one seed."""
        self._manual_seed(seed)
        rate = int(self._model.sample_rate)
        renders = [
            resample(piece, rate)
            for sentence in CALIBRATION_SENTENCES
            for piece in _pieces(self._model, state, sentence)
            if piece.size
        ]
        if not renders:
            return float("-inf")
        return integrated_loudness(np.concatenate(renders), OUTPUT_RATE)

    def load(self, artefact_dir: Path) -> PocketVoice:
        """Read the exported voice state and its gain.

        Raises:
            SpeechEngineError: A file is missing, `gain.json` is unusable or from another
                calibration, or pocket-tts could not read the state; the message names the file.
        """
        path = artefact_dir / VOICE_STATE_FILE
        if not path.is_file():
            raise SpeechEngineError(f"{ENGINE_ID}: {VOICE_STATE_FILE} is missing")
        gain_db = _read_gain(artefact_dir / GAIN_FILE)
        try:
            state = self._model.get_state_for_audio_prompt(path)
        except _LIBRARY_ERRORS as exc:
            raise _library_error(f"reading {VOICE_STATE_FILE}", exc) from exc
        return PocketVoice(state, gain_db)


def _read_gain(path: Path) -> float:
    """The `gain_db` of a `gain.json` that `build` wrote with this calibration.

    Raises:
        SpeechEngineError: The file is missing, is not JSON, lacks a field, holds a gain that is
            not a finite number at most `MAX_GAIN_DB`, or names another calibration.
    """
    where = f"{ENGINE_ID}: {GAIN_FILE}"
    if not path.is_file():
        raise SpeechEngineError(f"{where} is missing")
    try:
        gain = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpeechEngineError(f"{where} is unreadable ({type(exc).__name__})") from exc
    if not isinstance(gain, dict):
        raise SpeechEngineError(f"{where} is not an object")
    if gain.get("calibration_sha") != CALIBRATION_SHA:
        raise SpeechEngineError(
            f"{where} was measured with another calibration; bump ARTEFACT_VERSION when the "
            "calibration changes, so every gain is rebuilt"
        )
    if not isinstance(gain.get("state"), str):
        raise SpeechEngineError(f"{where} has no `state` string")
    gain_db = gain.get("gain_db")
    if (
        isinstance(gain_db, bool)
        or not isinstance(gain_db, int | float)
        or not math.isfinite(gain_db)
        or gain_db > MAX_GAIN_DB
    ):
        raise SpeechEngineError(
            f"{where} has `gain_db` {gain_db!r}, not a gain of at most {MAX_GAIN_DB:+g} dB"
        )
    return float(gain_db)


class PocketCloneSynthesizer:
    """Pocket TTS speaking MIST's voice identity, one voice state per speaking state."""

    engine_id = ENGINE_ID
    sample_rate = OUTPUT_RATE

    __slots__ = (
        "_fallbacks_logged",
        "_gains",
        "_model",
        "_model_id",
        "_prompts",
        "_released",
        "states",
        "voice",
    )

    def __init__(
        self,
        model: PocketModel,
        prompts: Mapping[str, PocketVoice],
        voice: VoiceInfo,
        *,
        model_id: str = MODEL_PIN,
    ) -> None:
        if NEUTRAL_STATE not in prompts:
            raise SpeechEngineError(f"{ENGINE_ID}: no voice state for {NEUTRAL_STATE!r}")
        self._model_id = model_id
        self._model: PocketModel | None = model
        self._prompts: dict[str, VoiceState] = {
            state: prompt.state for state, prompt in prompts.items()
        }
        self._gains: dict[str, float] = {
            state: 10.0 ** (prompt.gain_db / 20.0) for state, prompt in prompts.items()
        }
        self.voice = voice
        self.states: frozenset[str] = frozenset(self._prompts) & SPEAKING_STATES
        self._fallbacks_logged: set[str] = set()
        self._released = False

    @property
    def model_id(self) -> str:
        """The weights this instance speaks with: the builder's pin, which the turn records."""
        return self._model_id

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
        the cache lock; nothing is written under the identity. An identity's trained weights
        are checked against this engine before the model loads, and load in place of the
        stock clone weights.

        Raises:
            VoiceAssetError: No identity is configured, or it or its voice states are unusable,
                or its trained weights are not for this engine, base, tokenizer or rate; the
                message names the field, the relative path or the state.
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
        models_dir = Path(placement.models_dir)
        pin = _model_pin(identity, models_dir)
        trained = identity.model
        library = library_loader()
        model = _load_model(
            library,
            models_dir,
            weights=None if trained is None else trained.weights_path,
            tokenizer=None if trained is None else trained.tokenizer_path,
        )
        if trained is not None and int(model.sample_rate) != MODEL_RATE:
            raise VoiceAssetError(
                f"voice identity {identity.voice_id!r}: model: the trained weights run at "
                f"{int(model.sample_rate)} Hz, not {MODEL_RATE} Hz"
            )
        builder = PocketVoiceBuilder(
            model, library.export_model_state, manual_seed=library.manual_seed, model_pin=pin
        )
        prompts = ensure_artefacts(identity, builder, Path(placement.voice_cache_dir))
        voice = VoiceInfo(ENGINE_ID, identity.voice_id, profile)
        synthesizer = cls(model, prompts, voice, model_id=builder.model_pin)
        logger.info(
            "%s loaded voice %r on %s with %d threads: states %s",
            ENGINE_ID,
            identity.voice_id,
            DEVICE,
            THREADS,
            ", ".join(sorted(synthesizer.states)),
        )
        if trained is not None:
            logger.info(
                "%s voice %r speaks trained weights sha256 %s",
                ENGINE_ID,
                identity.voice_id,
                trained.sha256[:SHA_PREFIX],
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
        self._gains = {}
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
        model, prompt, gain = self._model, self._prompts[state], self._gains[state]
        rate = int(model.sample_rate)
        limiter = PeakLimiter(LIMITER, sample_rate=OUTPUT_RATE)
        pieces = _pieces(model, prompt, text)
        try:
            while True:
                # Only the pull is guarded: an error thrown in at a `yield` is the caller's.
                try:
                    piece = next(pieces, None)
                except _LIBRARY_ERRORS as exc:
                    # `from None`: the library's message may echo the segment text.
                    raise _library_error("synthesis", exc) from None
                if piece is None or token.is_set():
                    return
                if not piece.size:
                    continue
                samples = limiter.process(resample(piece, rate) * gain)
                for chunk in pcm16_chunks(to_pcm16(samples), OUTPUT_RATE):
                    if token.is_set():
                        return
                    yield SynthChunk(
                        pcm16=chunk, samples=len(chunk) // 2, segment=segment, state=state
                    )
        finally:
            pieces.close()


def _model_pin(identity: VoiceIdentity, models_dir: Path) -> str:
    """The pin of the weights the identity is voiced with; checks trained weights fit this engine.

    Runs before the library loads, so an identity this adapter cannot voice fails without
    loading a model.

    Raises:
        VoiceAssetError: The trained weights name another engine or base, or carry no
            tokenizer while the stock one is not the tokenizer they were trained with.
        SpeechEngineError: The stock tokenizer to check is missing; the message names its
            staged path.
    """
    trained = identity.model
    if trained is None:
        return MODEL_PIN
    where = f"voice identity {identity.voice_id!r}: model"
    if trained.engine != ENGINE_ID:
        raise VoiceAssetError(f"{where}.engine: {trained.engine!r} is not {ENGINE_ID!r}")
    if trained.base != MODEL_PIN:
        raise VoiceAssetError(f"{where}.base: {trained.base!r} is not {MODEL_PIN!r}")
    if trained.tokenizer_path is None:
        stock = _local_path(TOKENIZER_URI, models_dir)
        if not stock.is_file():
            raise SpeechEngineError(f"{ENGINE_ID}: weight file {stock} is missing")
        with stock.open("rb") as handle:
            stock_sha = hashlib.file_digest(handle, "sha256").hexdigest()
        if stock_sha != trained.tokenizer_sha256:
            raise VoiceAssetError(
                f"{where}.tokenizer_sha256: is not the stock tokenizer's "
                f"({stock_sha[:SHA_PREFIX]}), and the identity carries no tokenizer"
            )
    return f"{MODEL_PIN}+{trained.sha256}"


def _load_model(
    library: PocketLibrary,
    models_dir: Path,
    weights: Path | None = None,
    tokenizer: Path | None = None,
) -> PocketModel:
    """`library.load_model` on `local_config`'s copy of the installed config, on 4 threads.

    `weights` and `tokenizer` are an identity's trained files, which replace the stock clone
    weights and tokenizer.

    Raises:
        SpeechEngineError: A weight file is missing, the model failed to load, or it loaded
            without voice cloning.
    """
    config_path = library.configs_dir / CONFIG_NAME
    try:
        installed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SpeechEngineError(f"{ENGINE_ID}: cannot read {config_path}") from exc
    config = local_config(installed, models_dir, weights, tokenizer)
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
