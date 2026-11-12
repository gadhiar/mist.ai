"""Tests for the voice latency budget, the voice configuration dataclasses and their env loader.

The budget presets are the Design doc's Latency budget table: a figure that drifts from the table
makes `turn_metrics.over_budget` judge turns against a target nobody agreed to, so each preset is
pinned figure by figure and by its sum rule. `load_voice_settings` is the only reader of the
`MIST_VOICE_*` variables, so each variable is checked to reach its field and each malformed value
to fail at startup with the variable's name, instead of surfacing later as a confusing engine
error. The variables the GPU yield and the cloned voice used are pinned as ignored, so a stale
`.env` entry neither fails startup nor changes a setting.
"""

import dataclasses
import importlib.util
import json
import sys

import pytest

from backend.realtime import budget as budget_module
from backend.realtime.budget import (
    BUDGET_FIELDS,
    BUDGET_PRESETS,
    DEFAULT_ADDRESS_STARTS,
    DEFAULT_GREETINGS,
    DEFAULT_STOP_BEFORE,
    DEFAULT_WAKE_PHRASES,
    STAGE_FIELDS,
    BargeInConfig,
    EndpointingConfig,
    PlacementConfig,
    PlaybackConfig,
    SegmenterConfig,
    TurnConfig,
    VoiceLatencyBudget,
    VoiceSettings,
    WakeConfig,
    load_voice_settings,
    parse_budget,
)
from src.multimodal import voice_profile

STAGE_ROWS = (
    "end_of_turn_ms",
    "stt_final_ms",
    "context_build_ms",
    "llm_first_token_ms",
    "first_segment_ms",
    "tts_first_chunk_ms",
    "transport_playback_ms",
)

TABLE_ROWS = (
    *STAGE_ROWS,
    "total_p50_ms",
    "total_p90_ms",
    "tool_call_ms",
    "barge_in_reaction_ms",
    "wake_screen_ms",
    "caption_lag_ms",
    "speech_screen_skew_ms",
    "screen_command_ms",
    "event_to_pixel_ms",
)

# The five screen and caption targets, the same in every preset. Wake to screen is the server
# side only, the name's end to `attend` or `listen` sent, so it is 205, not the 300 to the pixel.
SCREEN_TARGETS = (205, 500, 150, 600, 50)

# The Design doc's Latency budget table, column by column, in TABLE_ROWS order.
TABLE = {
    "floor": (250, 250, 0, 250, 120, 500, 100, 1470, 2100, 450, 700, *SCREEN_TARGETS),
    "memo": (250, 50, 0, 200, 100, 150, 100, 850, 1200, 270, 700, *SCREEN_TARGETS),
}

OUTSIDE_SUM = TABLE_ROWS[8:]


def _rows(budget: VoiceLatencyBudget) -> tuple[int, ...]:
    return tuple(getattr(budget, name) for name in TABLE_ROWS)


def _assert_sum_rule(budget: VoiceLatencyBudget) -> None:
    stages = [getattr(budget, name) for name in STAGE_FIELDS]
    assert budget.total_p50_ms == sum(stages)
    for name in OUTSIDE_SUM:
        assert name not in STAGE_FIELDS


class TestBudgetShape:
    def test_budget_has_sixteen_rows_and_a_preset_name(self):
        names = [field.name for field in dataclasses.fields(VoiceLatencyBudget)]

        assert names == ["preset", *TABLE_ROWS]
        assert BUDGET_FIELDS == TABLE_ROWS
        assert len(BUDGET_FIELDS) == 16

    def test_stage_fields_are_the_seven_rows_from_end_of_turn_to_transport(self):
        assert STAGE_FIELDS == STAGE_ROWS

    def test_presets_are_floor_and_memo(self):
        assert set(BUDGET_PRESETS) == {"floor", "memo"}

    def test_budget_is_frozen_and_slotted(self):
        budget = BUDGET_PRESETS["floor"]

        assert not hasattr(budget, "__dict__")
        with pytest.raises(dataclasses.FrozenInstanceError):
            budget.stt_final_ms = 1  # type: ignore[misc]


class TestPresets:
    def test_floor_matches_the_table_and_its_total_is_the_seven_stage_sum(self):
        budget = BUDGET_PRESETS["floor"]

        assert _rows(budget) == TABLE["floor"]
        assert budget.preset == "floor"
        assert budget.total_p50_ms == 1470
        _assert_sum_rule(budget)

    def test_memo_matches_the_table_and_its_total_is_the_seven_stage_sum(self):
        budget = BUDGET_PRESETS["memo"]

        assert _rows(budget) == TABLE["memo"]
        assert budget.preset == "memo"
        assert budget.total_p50_ms == 850
        _assert_sum_rule(budget)

    def test_default_budget_is_floor_not_memo(self):
        settings = load_voice_settings({})

        assert settings.budget == BUDGET_PRESETS["floor"]
        assert settings.budget.total_p50_ms == 1470

    def test_preset_name_travels_in_the_budget_turn_metrics_reports(self):
        budget = parse_budget("memo,tts_first_chunk_ms=420")

        reported = dataclasses.asdict(budget)

        assert reported["preset"] == "memo"
        assert reported["tts_first_chunk_ms"] == 420

    @pytest.mark.parametrize("name", sorted(BUDGET_PRESETS))
    def test_every_preset_judges_wake_to_screen_on_the_server_side_bound(self, name):
        budget = BUDGET_PRESETS[name]

        assert budget.wake_screen_ms == 205
        assert budget.event_to_pixel_ms == 50


class TestParseBudget:
    def test_a_bare_preset_name_returns_that_preset(self):
        assert parse_budget("memo") == BUDGET_PRESETS["memo"]

    def test_a_stage_override_replaces_the_stage_and_recomputes_the_total(self):
        budget = parse_budget("floor,stt_final_ms=300")

        assert budget.stt_final_ms == 300
        assert budget.total_p50_ms == 1520
        assert budget.preset == "floor"
        assert budget.total_p90_ms == BUDGET_PRESETS["floor"].total_p90_ms

    def test_a_row_outside_the_sum_can_be_overridden_without_changing_the_total(self):
        budget = parse_budget("memo,tool_call_ms=1000")

        assert budget.tool_call_ms == 1000
        assert budget.total_p50_ms == 850

    def test_a_screen_target_can_be_overridden_without_changing_the_total(self):
        budget = parse_budget("floor,caption_lag_ms=700")

        assert budget.caption_lag_ms == 700
        assert budget.total_p50_ms == 1470

    def test_whitespace_around_items_is_ignored(self):
        assert parse_budget(" floor , stt_final_ms = 300 ") == parse_budget(
            "floor,stt_final_ms=300"
        )

    def test_overriding_the_total_itself_raises_naming_it(self):
        with pytest.raises(ValueError, match="total_p50_ms"):
            parse_budget("floor,total_p50_ms=900")

    def test_an_unknown_preset_raises_naming_it(self):
        with pytest.raises(ValueError, match="fastest"):
            parse_budget("fastest")

    def test_the_removed_yield_preset_is_unknown(self):
        with pytest.raises(ValueError, match="floor_yield"):
            parse_budget("floor_yield")

    def test_an_unknown_row_raises_naming_it(self):
        with pytest.raises(ValueError, match="stt_partial_ms"):
            parse_budget("floor,stt_partial_ms=100")

    @pytest.mark.parametrize(
        "spec",
        [
            pytest.param("floor,stt_final_ms=fast", id="non-numeric"),
            pytest.param("floor,stt_final_ms=-5", id="negative"),
            pytest.param("floor,stt_final_ms", id="missing-equals"),
        ],
    )
    def test_a_malformed_override_raises_naming_the_row(self, spec):
        with pytest.raises(ValueError, match="stt_final_ms"):
            parse_budget(spec)


CONFIG_DEFAULTS = [
    pytest.param(
        EndpointingConfig,
        {
            "vad_threshold": 0.5,
            "speech_start_windows": 3,
            "pre_roll_ms": 300,
            "stt_partial_interval_ms": 250,
            "stt_partial_interval_speaking_ms": 250,
            "endpoint_silence_ms": 200,
            "eot_threshold": 0.5,
            "eot_recheck_ms": 300,
            "eot_max_silence_ms": 1500,
            "eot_window_ms": 8000,
            "min_utterance_ms": 250,
            "max_backlog_ms": 2000,
            "speculative_start": True,
            "no_speech_threshold": 0.6,
            "logprob_threshold": -1.0,
        },
        id="endpointing",
    ),
    pytest.param(
        BargeInConfig,
        {
            "keywords": ("stop", "wait", "hold on", "cancel", "never mind"),
            "arm_delay_ms": 1000,
            "keyword_window_words": 3,
            "echo_guard_ms": 1500,
            "wake_arbitration_ms": 250,
            "alarm_stop_keywords": ("stop",),
        },
        id="barge-in",
    ),
    pytest.param(
        PlaybackConfig,
        {
            "client_prebuffer_ms": 80,
            "max_lead_ms": 500,
            "fade_ms": 30,
            "progress_interval_ms": 100,
            "frame_ms": 40,
        },
        id="playback",
    ),
    pytest.param(
        SegmenterConfig,
        {"first_segment_min_chars": 40, "max_segment_chars": 200},
        id="segmenter",
    ),
    pytest.param(
        PlacementConfig,
        {
            "vad_device": "cpu",
            "eot_device": "cpu",
            "eot_threads": 4,
            "stt_device": ("cuda", "cpu"),
            "tts_device": ("cuda", "off"),
            "vad_model": (
                "onnx-community/silero-vad@e71cae966052b992a7eca6b17738916ce0eca4ec"
                ":onnx/model.onnx"
            ),
            "eot_model": (
                "pipecat-ai/smart-turn-v3@f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
                ":smart-turn-v3.2-cpu.onnx"
            ),
            "stt_engine": None,
            "stt_model": None,
            "tts_engine": None,
            "tts_model": None,
            "tts_voice": None,
            "vram_budget_mib": 4500,
            "release_timeout_s": 10.0,
            "models_dir": "/models",
            "kws_model": None,
            "kws_threads": 1,
        },
        id="placement",
    ),
    pytest.param(TurnConfig, {"tool_cancel_grace_ms": 2000}, id="turn"),
    pytest.param(
        WakeConfig,
        {
            "names": ("mist",),
            "phrases": (
                "hey {name}",
                "{name}, how's it going",
                "{name}, what's up",
                "are you there {name}",
            ),
            "address_starts": (
                "what",
                "what's",
                "how",
                "how's",
                "are",
                "can",
                "could",
                "would",
                "will",
                "do",
                "did",
                "set",
                "play",
                "turn",
                "tell",
                "show",
                "open",
                "close",
                "stop",
                "remind",
                "start",
                "put",
                "add",
                "call",
            ),
            "stop_before": (
                "the",
                "a",
                "an",
                "this",
                "that",
                "some",
                "of",
                "in",
                "into",
                "through",
                "morning",
                "thick",
                "light",
                "sea",
                "mountain",
            ),
            "greetings": ("hey", "hi", "hello", "morning", "evening", "yo"),
            "kws_threshold": 0.5,
            "endpoint_threshold": 0.5,
            "lookback_ms": 6000,
            "join_gap_ms": 1000,
            "gate_tail_ms": 300,
            "reference_tolerance_ms": 400,
            "endpoint_min_backlog_ms": 1500,
            "followup_window_ms": 0,
            "attend_cue": True,
        },
        id="wake",
    ),
]


class TestConfigDataclasses:
    @pytest.mark.parametrize(("config_cls", "defaults"), CONFIG_DEFAULTS)
    def test_defaults_are_the_design_docs(self, config_cls, defaults):
        config = config_cls()

        assert dataclasses.asdict(config) == defaults

    @pytest.mark.parametrize(("config_cls", "defaults"), CONFIG_DEFAULTS)
    def test_config_is_frozen_and_slotted(self, config_cls, defaults):
        config = config_cls()
        first_field = dataclasses.fields(config_cls)[0].name

        assert not hasattr(config, "__dict__")
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(config, first_field, getattr(config, first_field))

    def test_voice_settings_is_frozen_and_slotted(self):
        settings = VoiceSettings()

        assert not hasattr(settings, "__dict__")
        with pytest.raises(dataclasses.FrozenInstanceError):
            settings.turn = TurnConfig()  # type: ignore[misc]

    def test_voice_settings_fields_and_the_mist_profile_default(self):
        names = [field.name for field in dataclasses.fields(VoiceSettings)]

        assert names == [
            "budget",
            "endpointing",
            "barge_in",
            "playback",
            "segmenter",
            "placement",
            "turn",
            "wake",
            "profile",
        ]
        assert VoiceSettings().profile == "mist"

    def test_the_four_wake_word_sets_are_the_wake_configs_defaults(self):
        config = WakeConfig()

        assert config.phrases == DEFAULT_WAKE_PHRASES
        assert config.address_starts == DEFAULT_ADDRESS_STARTS
        assert config.stop_before == DEFAULT_STOP_BEFORE
        assert config.greetings == DEFAULT_GREETINGS

    def test_is_is_not_an_address_start(self):
        # "mist is a good name" talks about the name, not to her.
        assert "is" not in DEFAULT_ADDRESS_STARTS

    def test_every_default_template_holds_the_name_slot_once(self):
        for template in DEFAULT_WAKE_PHRASES:
            assert template.count("{name}") == 1

    def test_the_profile_default_follows_the_voice_profile_constant(self, monkeypatch):
        # A fresh copy of the module, built while the constant reads "probe": a default that
        # comes from `DEFAULT_VOICE_PROFILE` follows it, and a literal copy of it would not. The
        # copy is named inside `backend.realtime`, so its relative imports resolve.
        monkeypatch.setattr(voice_profile, "DEFAULT_VOICE_PROFILE", "probe")
        name = "backend.realtime.budget_probe"
        spec = importlib.util.spec_from_file_location(name, budget_module.__file__)
        probe = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, probe)

        spec.loader.exec_module(probe)

        assert probe.VoiceSettings().profile == "probe"

    @pytest.mark.parametrize(
        "name",
        [
            "VoiceReferenceConfig",
            "GpuYieldConfig",
            "TTS_MODEL_PINS",
            "DEFAULT_YIELD_BUDGET",
        ],
    )
    def test_the_gpu_yield_and_cloned_voice_names_are_gone(self, name):
        assert not hasattr(budget_module, name)


class TestLoadVoiceSettingsDefaults:
    def test_nothing_set_gives_every_design_doc_default(self):
        settings = load_voice_settings({})

        assert settings == VoiceSettings(
            budget=BUDGET_PRESETS["floor"],
            endpointing=EndpointingConfig(),
            barge_in=BargeInConfig(),
            playback=PlaybackConfig(),
            segmenter=SegmenterConfig(),
            placement=PlacementConfig(),
            turn=TurnConfig(),
            wake=WakeConfig(),
            profile="mist",
        )
        assert settings.placement.kws_model is None
        assert settings.placement.kws_threads == 1
        assert settings.placement.stt_device == ("cuda", "cpu")
        assert settings.placement.tts_device == ("cuda", "off")
        assert settings.placement.stt_engine is None
        assert settings.placement.tts_engine is None
        assert settings.placement.tts_voice is None
        assert settings.endpointing.stt_partial_interval_ms == 250
        assert settings.placement.release_timeout_s == 10.0
        assert settings.placement.models_dir == "/models"
        assert settings.barge_in.alarm_stop_keywords == ("stop",)

    def test_every_removed_variable_is_ignored(self):
        removed = {
            "MIST_VOICE_YIELD_BUDGET": "floor_yield,total_p50_ms=1",
            "MIST_VOICE_STT_YIELD_DEVICE": "remote,cpu",
            "MIST_VOICE_TTS_YIELD_DEVICE": "tpu",
            "MIST_VOICE_TTS_VARIANT": "multilingual",
            "MIST_VOICE_GPU_YIELD_TRIGGERS": "manual,game_detected",
            "MIST_VOICE_GPU_YIELD_FREE_MIB": "-1",
            "MIST_VOICE_GPU_YIELD_AFTER_S": "soon",
            "MIST_VOICE_GPU_RESUME_AFTER_S": "nan",
            "MIST_VOICE_PROFILES_DIR": "",
            "MIST_SPEECH_SERVICE_URL": "ws://speech.example:8091",
            # No mount uses it any more; it was only ever Docker Compose's.
            "MIST_VOICE_PROFILES_HOST_DIR": "/srv/profiles",
        }

        assert load_voice_settings(removed) == load_voice_settings({})

    @pytest.mark.parametrize(
        "name",
        [
            pytest.param("MIST_VOICE_THINKING_BUDGET_TOKENS", id="llm-config-thinking"),
            pytest.param("MIST_VOICE_FIRST_SEGMENT_MIN_CHARS", id="llm-config-first-segment"),
        ],
    )
    def test_variables_read_elsewhere_leave_the_settings_unchanged(self, name):
        settings = load_voice_settings({name: "17"})

        assert settings == load_voice_settings({})


# (variable, value, path to the field, expected value)
VARIABLES = [
    ("MIST_VOICE_BUDGET", "memo", "budget", BUDGET_PRESETS["memo"]),
    ("MIST_VOICE_CLIENT_PREBUFFER_MS", "120", "playback.client_prebuffer_ms", 120),
    ("MIST_VOICE_VAD_THRESHOLD", "0.65", "endpointing.vad_threshold", 0.65),
    ("MIST_VOICE_ENDPOINT_SILENCE_MS", "350", "endpointing.endpoint_silence_ms", 350),
    ("MIST_VOICE_EOT_THRESHOLD", "0.7", "endpointing.eot_threshold", 0.7),
    ("MIST_VOICE_EOT_MAX_SILENCE_MS", "2000", "endpointing.eot_max_silence_ms", 2000),
    (
        "MIST_VOICE_BARGE_IN_KEYWORDS",
        "Stop, hang on ,enough",
        "barge_in.keywords",
        ("stop", "hang on", "enough"),
    ),
    ("MIST_VOICE_BARGE_IN_ARM_DELAY_MS", "800", "barge_in.arm_delay_ms", 800),
    ("MIST_VOICE_ECHO_GUARD_MS", "900", "barge_in.echo_guard_ms", 900),
    (
        "MIST_VOICE_ALARM_STOP_KEYWORDS",
        " Stop ,cancel",
        "barge_in.alarm_stop_keywords",
        ("stop", "cancel"),
    ),
    ("MIST_VOICE_SPECULATIVE_START", "false", "endpointing.speculative_start", False),
    ("MIST_VOICE_MAX_SEGMENT_CHARS", "150", "segmenter.max_segment_chars", 150),
    ("MIST_VOICE_VAD_DEVICE", "cuda", "placement.vad_device", "cuda"),
    ("MIST_VOICE_EOT_DEVICE", "cuda:1", "placement.eot_device", "cuda:1"),
    ("MIST_VOICE_EOT_THREADS", "8", "placement.eot_threads", 8),
    (
        "MIST_VOICE_STT_DEVICE",
        "cuda:0, cpu, off",
        "placement.stt_device",
        ("cuda:0", "cpu", "off"),
    ),
    ("MIST_VOICE_TTS_DEVICE", "off", "placement.tts_device", ("off",)),
    (
        "MIST_VOICE_VAD_MODEL",
        "org/vad@abc123:model.onnx",
        "placement.vad_model",
        "org/vad@abc123:model.onnx",
    ),
    (
        "MIST_VOICE_EOT_MODEL",
        "org/eot@def456:onnx/x.onnx",
        "placement.eot_model",
        "org/eot@def456:onnx/x.onnx",
    ),
    ("MIST_VOICE_STT_ENGINE", "whisper-turbo", "placement.stt_engine", "whisper-turbo"),
    ("MIST_VOICE_STT_MODEL", "org/stt@0123abc", "placement.stt_model", "org/stt@0123abc"),
    ("MIST_VOICE_TTS_ENGINE", "kokoro2", "placement.tts_engine", "kokoro2"),
    ("MIST_VOICE_TTS_MODEL", "org/tts@0123abc", "placement.tts_model", "org/tts@0123abc"),
    ("MIST_VOICE_TTS_VOICE", "af_heart", "placement.tts_voice", "af_heart"),
    ("MIST_VOICE_VRAM_BUDGET_MIB", "6000", "placement.vram_budget_mib", 6000),
    ("MIST_VOICE_RELEASE_TIMEOUT_S", "2.5", "placement.release_timeout_s", 2.5),
    ("MIST_VOICE_MODELS_DIR", "/srv/weights", "placement.models_dir", "/srv/weights"),
    ("VOICE_PROFILE", "studio", "profile", "studio"),
    ("MIST_VOICE_NO_SPEECH_THRESHOLD", "0.8", "endpointing.no_speech_threshold", 0.8),
    ("MIST_VOICE_LOGPROB_THRESHOLD", "-0.5", "endpointing.logprob_threshold", -0.5),
    ("MIST_VOICE_TOOL_CANCEL_GRACE_MS", "3000", "turn.tool_cancel_grace_ms", 3000),
    # At most the 300 ms gate tail, so that arbitration overlaps it.
    ("MIST_VOICE_WAKE_ARBITRATION_MS", "280", "barge_in.wake_arbitration_ms", 280),
    ("MIST_VOICE_WAKE_NAMES", " mist, nova ", "wake.names", ("mist", "nova")),
    ("MIST_VOICE_KWS_THRESHOLD", "0.62", "wake.kws_threshold", 0.62),
    ("MIST_VOICE_WAKE_ENDPOINT_THRESHOLD", "0.4", "wake.endpoint_threshold", 0.4),
    ("MIST_VOICE_WAKE_LOOKBACK_MS", "9000", "wake.lookback_ms", 9000),
    ("MIST_VOICE_WAKE_JOIN_GAP_MS", "800", "wake.join_gap_ms", 800),
    ("MIST_VOICE_WAKE_GATE_TAIL_MS", "350", "wake.gate_tail_ms", 350),
    ("MIST_VOICE_WAKE_REFERENCE_TOLERANCE_MS", "500", "wake.reference_tolerance_ms", 500),
    ("MIST_VOICE_WAKE_ENDPOINT_MIN_BACKLOG_MS", "2000", "wake.endpoint_min_backlog_ms", 2000),
    ("MIST_VOICE_FOLLOWUP_WINDOW_MS", "4000", "wake.followup_window_ms", 4000),
    ("MIST_VOICE_WAKE_ATTEND_CUE", "false", "wake.attend_cue", False),
    ("MIST_VOICE_KWS_MODEL", "mist-wake-2", "placement.kws_model", "mist-wake-2"),
    ("MIST_VOICE_KWS_THREADS", "2", "placement.kws_threads", 2),
]

# Read from a file the variable names, so the file tests cover them.
FILE_VARIABLES = {"MIST_VOICE_WAKE_PHRASES_FILE"}


def _get_path(settings: VoiceSettings, path: str) -> object:
    value: object = settings
    for part in path.split("."):
        value = getattr(value, part)
    return value


def _with_path(settings: VoiceSettings, path: str, value: object) -> VoiceSettings:
    head, _, rest = path.partition(".")
    if not rest:
        return dataclasses.replace(settings, **{head: value})
    child = getattr(settings, head)
    return dataclasses.replace(settings, **{head: dataclasses.replace(child, **{rest: value})})


class TestLoadVoiceSettingsVariables:
    @pytest.mark.parametrize(
        ("name", "raw", "path", "expected"),
        [pytest.param(*case, id=case[0]) for case in VARIABLES],
    )
    def test_each_variable_reaches_its_field_and_nothing_else(self, name, raw, path, expected):
        settings = load_voice_settings({name: raw})

        assert _get_path(settings, path) == expected
        assert settings == _with_path(load_voice_settings({}), path, expected)

    def test_the_budget_variable_accepts_stage_overrides(self):
        settings = load_voice_settings({"MIST_VOICE_BUDGET": "memo,stt_final_ms=150"})

        assert settings.budget.stt_final_ms == 150
        assert settings.budget.total_p50_ms == 950

    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param("true", id="true"),
            pytest.param("1", id="one"),
            pytest.param("YES", id="yes-upper"),
            pytest.param("on", id="on"),
        ],
    )
    def test_speculative_start_accepts_true_spellings(self, raw):
        settings = load_voice_settings({"MIST_VOICE_SPECULATIVE_START": raw})

        assert settings.endpointing.speculative_start is True

    def test_alarm_stop_keywords_are_checked_against_the_configured_barge_in_keywords(self):
        settings = load_voice_settings(
            {
                "MIST_VOICE_BARGE_IN_KEYWORDS": "halt,stop",
                "MIST_VOICE_ALARM_STOP_KEYWORDS": "halt",
            }
        )

        assert settings.barge_in.alarm_stop_keywords == ("halt",)

    def test_every_variable_in_the_configuration_list_is_covered(self):
        read = {case[0] for case in VARIABLES} | FILE_VARIABLES

        assert read == {
            "MIST_VOICE_WAKE_NAMES",
            "MIST_VOICE_WAKE_PHRASES_FILE",
            "MIST_VOICE_KWS_THRESHOLD",
            "MIST_VOICE_WAKE_ENDPOINT_THRESHOLD",
            "MIST_VOICE_WAKE_LOOKBACK_MS",
            "MIST_VOICE_WAKE_JOIN_GAP_MS",
            "MIST_VOICE_WAKE_GATE_TAIL_MS",
            "MIST_VOICE_WAKE_REFERENCE_TOLERANCE_MS",
            "MIST_VOICE_WAKE_ENDPOINT_MIN_BACKLOG_MS",
            "MIST_VOICE_FOLLOWUP_WINDOW_MS",
            "MIST_VOICE_WAKE_ATTEND_CUE",
            "MIST_VOICE_KWS_MODEL",
            "MIST_VOICE_KWS_THREADS",
            "MIST_VOICE_BUDGET",
            "MIST_VOICE_CLIENT_PREBUFFER_MS",
            "MIST_VOICE_VAD_THRESHOLD",
            "MIST_VOICE_ENDPOINT_SILENCE_MS",
            "MIST_VOICE_EOT_THRESHOLD",
            "MIST_VOICE_EOT_MAX_SILENCE_MS",
            "MIST_VOICE_BARGE_IN_KEYWORDS",
            "MIST_VOICE_BARGE_IN_ARM_DELAY_MS",
            "MIST_VOICE_ECHO_GUARD_MS",
            "MIST_VOICE_ALARM_STOP_KEYWORDS",
            "MIST_VOICE_SPECULATIVE_START",
            "MIST_VOICE_MAX_SEGMENT_CHARS",
            "MIST_VOICE_VAD_DEVICE",
            "MIST_VOICE_EOT_DEVICE",
            "MIST_VOICE_EOT_THREADS",
            "MIST_VOICE_STT_DEVICE",
            "MIST_VOICE_TTS_DEVICE",
            "MIST_VOICE_VAD_MODEL",
            "MIST_VOICE_EOT_MODEL",
            "MIST_VOICE_STT_ENGINE",
            "MIST_VOICE_STT_MODEL",
            "MIST_VOICE_TTS_ENGINE",
            "MIST_VOICE_TTS_MODEL",
            "MIST_VOICE_TTS_VOICE",
            "MIST_VOICE_VRAM_BUDGET_MIB",
            "MIST_VOICE_RELEASE_TIMEOUT_S",
            "MIST_VOICE_MODELS_DIR",
            "MIST_VOICE_NO_SPEECH_THRESHOLD",
            "MIST_VOICE_LOGPROB_THRESHOLD",
            "MIST_VOICE_TOOL_CANCEL_GRACE_MS",
            "MIST_VOICE_WAKE_ARBITRATION_MS",
            "VOICE_PROFILE",
        }


def _phrases_file(tmp_path, content: object) -> dict[str, str]:
    path = tmp_path / "phrases.json"
    text = content if isinstance(content, str) else json.dumps(content)
    path.write_text(text, encoding="utf-8")
    return {"MIST_VOICE_WAKE_PHRASES_FILE": str(path)}


class TestWakePhrasesFile:
    def test_a_file_adds_a_template_and_a_greeting(self, tmp_path):
        environ = _phrases_file(
            tmp_path,
            {
                "phrases": [*DEFAULT_WAKE_PHRASES, "yo {name}"],
                "greetings": [*DEFAULT_GREETINGS, "howdy"],
            },
        )

        wake = load_voice_settings(environ).wake

        assert wake.phrases == (*DEFAULT_WAKE_PHRASES, "yo {name}")
        assert wake.greetings == (*DEFAULT_GREETINGS, "howdy")

    def test_the_keys_a_file_omits_keep_their_defaults(self, tmp_path):
        environ = _phrases_file(tmp_path, {"phrases": ["yo {name}"]})

        wake = load_voice_settings(environ).wake

        assert wake.phrases == ("yo {name}",)
        assert wake.address_starts == DEFAULT_ADDRESS_STARTS
        assert wake.stop_before == DEFAULT_STOP_BEFORE
        assert wake.greetings == DEFAULT_GREETINGS
        assert wake == dataclasses.replace(WakeConfig(), phrases=("yo {name}",))

    def test_a_file_sets_the_address_starts_and_the_stop_words(self, tmp_path):
        environ = _phrases_file(
            tmp_path, {"address_starts": ["what", "find"], "stop_before": ["the"]}
        )

        wake = load_voice_settings(environ).wake

        assert wake.address_starts == ("what", "find")
        assert wake.stop_before == ("the",)


class TestLoadVoiceSettingsErrors:
    @pytest.mark.parametrize(
        "template",
        [
            pytest.param("hey there", id="no-name"),
            pytest.param("{name} {name}", id="two-names"),
        ],
    )
    def test_a_template_without_exactly_one_name_slot_raises_naming_the_file_variable(
        self, tmp_path, template
    ):
        environ = _phrases_file(tmp_path, {"phrases": ["hey {name}", template]})

        with pytest.raises(ValueError, match="MIST_VOICE_WAKE_PHRASES_FILE"):
            load_voice_settings(environ)

    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param("mist ai", id="space"),
            pytest.param("Mist", id="upper-case"),
            pytest.param(" , ", id="empty"),
        ],
    )
    def test_a_name_that_is_not_one_lower_case_word_raises_naming_the_variable(self, raw):
        with pytest.raises(ValueError, match="MIST_VOICE_WAKE_NAMES"):
            load_voice_settings({"MIST_VOICE_WAKE_NAMES": raw})

    def test_a_lookback_that_leaves_no_second_inside_the_wire_bound_raises_naming_it(self):
        # 9500 + 1000 > 10000, the most a wake may carry.
        with pytest.raises(ValueError, match="MIST_VOICE_WAKE_LOOKBACK_MS"):
            load_voice_settings({"MIST_VOICE_WAKE_LOOKBACK_MS": "9500"})

    @pytest.mark.parametrize(
        "name",
        [
            "MIST_VOICE_WAKE_LOOKBACK_MS",
            "MIST_VOICE_WAKE_JOIN_GAP_MS",
            "MIST_VOICE_WAKE_GATE_TAIL_MS",
            "MIST_VOICE_WAKE_REFERENCE_TOLERANCE_MS",
        ],
    )
    def test_a_zero_wake_interval_raises_naming_the_variable(self, name):
        with pytest.raises(ValueError, match=name):
            load_voice_settings({name: "0"})

    def test_a_relative_phrases_path_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_WAKE_PHRASES_FILE"):
            load_voice_settings({"MIST_VOICE_WAKE_PHRASES_FILE": "wake/phrases.json"})

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param('{"phrases": ["hey {name}"', id="truncated-json"),
            pytest.param(["hey {name}"], id="not-an-object"),
            pytest.param({"phrases": "hey {name}"}, id="not-a-list"),
            pytest.param({"phrases": ["hey {name}", 3]}, id="not-a-string"),
            pytest.param({"greetings": ["hey", " "]}, id="empty-string"),
            pytest.param({"phrase": ["hey {name}"]}, id="unknown-key"),
        ],
    )
    def test_a_malformed_phrases_file_raises_naming_the_variable(self, tmp_path, content):
        environ = _phrases_file(tmp_path, content)

        with pytest.raises(ValueError, match="MIST_VOICE_WAKE_PHRASES_FILE"):
            load_voice_settings(environ)

    def test_a_missing_phrases_file_raises_naming_the_variable(self, tmp_path):
        missing = str(tmp_path / "absent.json")

        with pytest.raises(ValueError, match="MIST_VOICE_WAKE_PHRASES_FILE"):
            load_voice_settings({"MIST_VOICE_WAKE_PHRASES_FILE": missing})

    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param("wake/mist-2", id="slash"),
            pytest.param("Mist-Wake", id="upper-case"),
            pytest.param("", id="empty"),
        ],
    )
    def test_a_kws_model_outside_the_id_form_raises_naming_the_variable(self, raw):
        with pytest.raises(ValueError, match="MIST_VOICE_KWS_MODEL"):
            load_voice_settings({"MIST_VOICE_KWS_MODEL": raw})

    @pytest.mark.parametrize("raw", ["0", "-1", "two"])
    def test_kws_threads_must_be_a_positive_integer(self, raw):
        with pytest.raises(ValueError, match="MIST_VOICE_KWS_THREADS"):
            load_voice_settings({"MIST_VOICE_KWS_THREADS": raw})

    @pytest.mark.parametrize("raw", ["400", "10001"])
    def test_an_endpoint_min_backlog_outside_its_bounds_raises_naming_it(self, raw):
        with pytest.raises(ValueError, match="MIST_VOICE_WAKE_ENDPOINT_MIN_BACKLOG_MS"):
            load_voice_settings({"MIST_VOICE_WAKE_ENDPOINT_MIN_BACKLOG_MS": raw})

    @pytest.mark.parametrize("raw", ["500", "10000"])
    def test_an_endpoint_min_backlog_at_its_bounds_is_accepted(self, raw):
        settings = load_voice_settings({"MIST_VOICE_WAKE_ENDPOINT_MIN_BACKLOG_MS": raw})

        assert settings.wake.endpoint_min_backlog_ms == int(raw)

    def test_wake_arbitration_above_the_gate_tail_raises_naming_both(self):
        environ = {"MIST_VOICE_WAKE_ARBITRATION_MS": "301", "MIST_VOICE_WAKE_GATE_TAIL_MS": "300"}

        with pytest.raises(
            ValueError, match="MIST_VOICE_WAKE_ARBITRATION_MS.*MIST_VOICE_WAKE_GATE_TAIL_MS"
        ):
            load_voice_settings(environ)

    def test_a_gate_tail_shorter_than_the_default_arbitration_raises(self):
        with pytest.raises(ValueError, match="MIST_VOICE_WAKE_ARBITRATION_MS"):
            load_voice_settings({"MIST_VOICE_WAKE_GATE_TAIL_MS": "200"})

    @pytest.mark.parametrize(
        "name", ["MIST_VOICE_KWS_THRESHOLD", "MIST_VOICE_WAKE_ENDPOINT_THRESHOLD"]
    )
    def test_a_wake_threshold_above_one_raises_naming_the_variable(self, name):
        with pytest.raises(ValueError, match=name):
            load_voice_settings({name: "1.2"})

    def test_a_negative_followup_window_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_FOLLOWUP_WINDOW_MS"):
            load_voice_settings({"MIST_VOICE_FOLLOWUP_WINDOW_MS": "-1"})

    def test_a_negative_number_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_ENDPOINT_SILENCE_MS"):
            load_voice_settings({"MIST_VOICE_ENDPOINT_SILENCE_MS": "-200"})

    def test_a_non_numeric_number_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_CLIENT_PREBUFFER_MS"):
            load_voice_settings({"MIST_VOICE_CLIENT_PREBUFFER_MS": "eighty"})

    def test_a_non_numeric_float_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_LOGPROB_THRESHOLD"):
            load_voice_settings({"MIST_VOICE_LOGPROB_THRESHOLD": "low"})

    def test_a_non_finite_float_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_NO_SPEECH_THRESHOLD"):
            load_voice_settings({"MIST_VOICE_NO_SPEECH_THRESHOLD": "nan"})

    def test_a_probability_threshold_above_one_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_VAD_THRESHOLD"):
            load_voice_settings({"MIST_VOICE_VAD_THRESHOLD": "1.5"})

    def test_an_unknown_placement_location_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_STT_DEVICE"):
            load_voice_settings({"MIST_VOICE_STT_DEVICE": "cuda,tpu"})

    @pytest.mark.parametrize("name", ["MIST_VOICE_STT_DEVICE", "MIST_VOICE_TTS_DEVICE"])
    def test_remote_is_no_longer_a_location(self, name):
        with pytest.raises(ValueError, match=name):
            load_voice_settings({name: "remote,off"})

    def test_a_malformed_cuda_index_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_TTS_DEVICE"):
            load_voice_settings({"MIST_VOICE_TTS_DEVICE": "cuda:x,off"})

    def test_an_empty_placement_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_TTS_DEVICE"):
            load_voice_settings({"MIST_VOICE_TTS_DEVICE": ""})

    def test_a_placement_of_only_separators_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_STT_DEVICE"):
            load_voice_settings({"MIST_VOICE_STT_DEVICE": " , "})

    def test_a_repeated_placement_location_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_STT_DEVICE"):
            load_voice_settings({"MIST_VOICE_STT_DEVICE": "cpu,cuda,cpu"})

    def test_off_before_the_last_place_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_TTS_DEVICE"):
            load_voice_settings({"MIST_VOICE_TTS_DEVICE": "cuda,off,cpu"})

    def test_a_vad_device_list_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_VAD_DEVICE"):
            load_voice_settings({"MIST_VOICE_VAD_DEVICE": "cuda,cpu"})

    def test_a_remote_vad_device_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_VAD_DEVICE"):
            load_voice_settings({"MIST_VOICE_VAD_DEVICE": "remote"})

    def test_an_off_end_of_turn_device_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_EOT_DEVICE"):
            load_voice_settings({"MIST_VOICE_EOT_DEVICE": "off"})

    @pytest.mark.parametrize("raw", ["0", "-2", "four"])
    def test_end_of_turn_threads_must_be_a_positive_integer(self, raw):
        with pytest.raises(ValueError, match="MIST_VOICE_EOT_THREADS"):
            load_voice_settings({"MIST_VOICE_EOT_THREADS": raw})

    @pytest.mark.parametrize("name", ["MIST_VOICE_STT_ENGINE", "MIST_VOICE_TTS_ENGINE"])
    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param("", id="empty"),
            pytest.param("Whisper", id="upper-case"),
            pytest.param("-kokoro", id="leading-hyphen"),
            pytest.param("faster_whisper", id="underscore"),
            pytest.param("qwen3 tts", id="space"),
            pytest.param("kokoro/v1", id="slash"),
        ],
    )
    def test_an_engine_id_outside_the_id_form_raises_naming_the_variable(self, name, raw):
        with pytest.raises(ValueError, match=name):
            load_voice_settings({name: raw})

    def test_an_stt_model_without_a_revision_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_STT_MODEL"):
            load_voice_settings({"MIST_VOICE_STT_MODEL": "base"})

    def test_a_tts_model_without_a_revision_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_TTS_MODEL"):
            load_voice_settings({"MIST_VOICE_TTS_MODEL": "ResembleAI/chatterbox-turbo"})

    def test_a_tts_model_with_a_file_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_TTS_MODEL"):
            load_voice_settings({"MIST_VOICE_TTS_MODEL": "org/tts@abc:model.bin"})

    def test_an_onnx_model_without_a_file_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_VAD_MODEL"):
            load_voice_settings({"MIST_VOICE_VAD_MODEL": "onnx-community/silero-vad@e71cae9"})

    def test_an_onnx_model_without_a_revision_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_EOT_MODEL"):
            load_voice_settings({"MIST_VOICE_EOT_MODEL": "pipecat-ai/smart-turn-v3:model.onnx"})

    def test_an_empty_tts_voice_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_TTS_VOICE"):
            load_voice_settings({"MIST_VOICE_TTS_VOICE": " "})

    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param("0", id="zero"),
            pytest.param("-1", id="negative"),
            pytest.param("ten", id="non-numeric"),
            pytest.param("inf", id="infinite"),
        ],
    )
    def test_a_release_timeout_that_is_not_a_positive_number_raises_naming_it(self, raw):
        with pytest.raises(ValueError, match="MIST_VOICE_RELEASE_TIMEOUT_S"):
            load_voice_settings({"MIST_VOICE_RELEASE_TIMEOUT_S": raw})

    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param("", id="empty"),
            pytest.param("models", id="relative"),
            pytest.param("./models", id="dot-relative"),
        ],
    )
    def test_a_models_dir_that_is_not_an_absolute_path_raises_naming_it(self, raw):
        with pytest.raises(ValueError, match="MIST_VOICE_MODELS_DIR"):
            load_voice_settings({"MIST_VOICE_MODELS_DIR": raw})

    def test_an_alarm_stop_word_that_is_not_a_barge_in_keyword_raises_naming_it(self):
        with pytest.raises(ValueError, match="MIST_VOICE_ALARM_STOP_KEYWORDS.*halt"):
            load_voice_settings({"MIST_VOICE_ALARM_STOP_KEYWORDS": "stop,halt"})

    def test_an_empty_alarm_stop_keyword_list_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_ALARM_STOP_KEYWORDS"):
            load_voice_settings({"MIST_VOICE_ALARM_STOP_KEYWORDS": " , "})

    def test_an_unreadable_boolean_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_SPECULATIVE_START"):
            load_voice_settings({"MIST_VOICE_SPECULATIVE_START": "maybe"})

    def test_an_empty_keyword_list_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_BARGE_IN_KEYWORDS"):
            load_voice_settings({"MIST_VOICE_BARGE_IN_KEYWORDS": " , "})

    def test_an_unknown_budget_preset_raises_naming_the_variable_and_preset(self):
        with pytest.raises(ValueError, match="MIST_VOICE_BUDGET.*fastest"):
            load_voice_settings({"MIST_VOICE_BUDGET": "fastest"})

    def test_a_budget_total_override_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_BUDGET.*total_p50_ms"):
            load_voice_settings({"MIST_VOICE_BUDGET": "floor,total_p50_ms=1"})

    def test_an_empty_voice_profile_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="VOICE_PROFILE"):
            load_voice_settings({"VOICE_PROFILE": ""})
