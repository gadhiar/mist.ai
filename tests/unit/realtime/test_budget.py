"""Tests for the voice latency budget, the voice configuration dataclasses and their env loader.

The budget presets are the Design doc's Latency budget table: a figure that drifts from the table
makes `turn_metrics.over_budget` judge turns against a target nobody agreed to, so each preset is
pinned figure by figure and by its sum rule. `load_voice_settings` is the only reader of the
`MIST_VOICE_*` variables, so each variable is checked to reach its field and each malformed value
to fail at startup with the variable's name, instead of surfacing later as a confusing engine
error.
"""

import dataclasses

import pytest

from backend.realtime.budget import (
    BUDGET_FIELDS,
    BUDGET_PRESETS,
    STAGE_FIELDS,
    TTS_MODEL_PINS,
    BargeInConfig,
    EndpointingConfig,
    GpuYieldConfig,
    PlacementConfig,
    PlaybackConfig,
    SegmenterConfig,
    TurnConfig,
    VoiceLatencyBudget,
    VoiceReferenceConfig,
    VoiceSettings,
    load_voice_settings,
    parse_budget,
)

TABLE_ROWS = (
    "end_of_turn_ms",
    "stt_final_ms",
    "context_build_ms",
    "llm_first_token_ms",
    "first_segment_ms",
    "tts_first_chunk_ms",
    "transport_playback_ms",
    "total_p50_ms",
    "total_p90_ms",
    "tool_call_ms",
    "barge_in_reaction_ms",
)

# The Design doc's Latency budget table, column by column, in TABLE_ROWS order.
TABLE = {
    "floor": (250, 250, 0, 250, 120, 500, 100, 1470, 2100, 450, 700),
    "floor_yield": (250, 600, 0, 500, 240, 1200, 130, 2920, 4100, 900, 1200),
    "memo": (250, 50, 0, 200, 100, 150, 100, 850, 1200, 270, 700),
}

OUTSIDE_SUM = ("total_p90_ms", "tool_call_ms", "barge_in_reaction_ms")


def _rows(budget: VoiceLatencyBudget) -> tuple[int, ...]:
    return tuple(getattr(budget, name) for name in TABLE_ROWS)


def _assert_sum_rule(budget: VoiceLatencyBudget) -> None:
    stages = [getattr(budget, name) for name in STAGE_FIELDS]
    assert budget.total_p50_ms == sum(stages)
    for name in OUTSIDE_SUM:
        assert name not in STAGE_FIELDS


class TestBudgetShape:
    def test_budget_has_the_tables_eleven_rows_and_a_preset_name(self):
        names = [field.name for field in dataclasses.fields(VoiceLatencyBudget)]

        assert names == ["preset", *TABLE_ROWS]
        assert BUDGET_FIELDS == TABLE_ROWS

    def test_stage_fields_are_the_seven_rows_from_end_of_turn_to_transport(self):
        assert TABLE_ROWS[:7] == STAGE_FIELDS

    def test_presets_are_floor_floor_yield_and_memo(self):
        assert set(BUDGET_PRESETS) == {"floor", "floor_yield", "memo"}

    def test_budget_is_frozen_and_slotted(self):
        budget = BUDGET_PRESETS["floor"]

        assert not hasattr(budget, "__dict__")
        with pytest.raises(dataclasses.FrozenInstanceError):
            budget.stt_final_ms = 1  # type: ignore[misc]


class TestPresets:
    def test_floor_matches_the_table_and_its_total_is_the_stage_sum(self):
        budget = BUDGET_PRESETS["floor"]

        assert _rows(budget) == TABLE["floor"]
        assert budget.preset == "floor"
        _assert_sum_rule(budget)

    def test_floor_yield_matches_the_table_and_its_total_is_the_stage_sum(self):
        budget = BUDGET_PRESETS["floor_yield"]

        assert _rows(budget) == TABLE["floor_yield"]
        assert budget.preset == "floor_yield"
        _assert_sum_rule(budget)

    def test_memo_matches_the_table_and_its_total_is_the_stage_sum(self):
        budget = BUDGET_PRESETS["memo"]

        assert _rows(budget) == TABLE["memo"]
        assert budget.preset == "memo"
        _assert_sum_rule(budget)

    def test_default_budget_is_floor_not_memo_and_yield_is_floor_yield(self):
        settings = load_voice_settings({})

        assert settings.budget == BUDGET_PRESETS["floor"]
        assert settings.budget.total_p50_ms == 1470
        assert settings.yield_budget == BUDGET_PRESETS["floor_yield"]
        assert settings.yield_budget.total_p50_ms == 2920

    def test_preset_name_travels_in_the_budget_turn_metrics_reports(self):
        budget = parse_budget("memo,tts_first_chunk_ms=420")

        reported = dataclasses.asdict(budget)

        assert reported["preset"] == "memo"
        assert reported["tts_first_chunk_ms"] == 420


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
        budget = parse_budget("floor_yield,tool_call_ms=1000")

        assert budget.tool_call_ms == 1000
        assert budget.total_p50_ms == 2920

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
            "stt_partial_interval_ms": 400,
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
            "stt_primary": ("cuda", "cpu"),
            "stt_yield": ("remote", "cpu"),
            "tts_primary": ("cuda", "remote", "off"),
            "tts_yield": ("remote", "off"),
            "vad_model": (
                "onnx-community/silero-vad@e71cae966052b992a7eca6b17738916ce0eca4ec"
                ":onnx/model.onnx"
            ),
            "eot_model": (
                "pipecat-ai/smart-turn-v3@f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
                ":smart-turn-v3.2-cpu.onnx"
            ),
            "stt_model": "base",
            "tts_variant": "turbo",
            "tts_model": None,
            "speech_service_url": None,
            "vram_budget_mib": 4500,
        },
        id="placement",
    ),
    pytest.param(
        VoiceReferenceConfig,
        {"profiles_dir": "data/voice_profiles", "profile": "jarvis"},
        id="voice-reference",
    ),
    pytest.param(
        GpuYieldConfig,
        {
            "triggers": frozenset({"manual", "fullscreen"}),
            "free_mib": 512,
            "yield_after_s": 3.0,
            "resume_after_s": 30.0,
            "switch_timeout_s": 10.0,
            "poll_s": 1.0,
        },
        id="gpu-yield",
    ),
    pytest.param(TurnConfig, {"tool_cancel_grace_ms": 2000}, id="turn"),
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

    def test_tts_model_pins_cover_both_variants(self):
        assert {
            "turbo": "ResembleAI/chatterbox-turbo@749d1c1a46eb10492095d68fbcf55691ccf137cd",
            "original": "ResembleAI/chatterbox@5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",
        } == TTS_MODEL_PINS


class TestLoadVoiceSettingsDefaults:
    def test_nothing_set_gives_every_design_doc_default(self):
        settings = load_voice_settings({})

        assert settings == VoiceSettings(
            budget=BUDGET_PRESETS["floor"],
            yield_budget=BUDGET_PRESETS["floor_yield"],
            endpointing=EndpointingConfig(),
            barge_in=BargeInConfig(),
            playback=PlaybackConfig(),
            segmenter=SegmenterConfig(),
            placement=PlacementConfig(),
            voice_reference=VoiceReferenceConfig(),
            gpu_yield=GpuYieldConfig(),
            turn=TurnConfig(),
        )
        assert settings.placement.stt_primary == ("cuda", "cpu")
        assert settings.placement.stt_yield == ("remote", "cpu")
        assert settings.placement.tts_primary == ("cuda", "remote", "off")
        assert settings.placement.tts_yield == ("remote", "off")
        assert settings.placement.tts_variant == "turbo"
        assert settings.placement.tts_model is None
        assert settings.gpu_yield.triggers == frozenset({"manual", "fullscreen"})
        assert settings.voice_reference.profiles_dir == "data/voice_profiles"
        assert settings.voice_reference.profile == "jarvis"

    @pytest.mark.parametrize(
        "name",
        [
            pytest.param("MIST_VOICE_PROFILES_HOST_DIR", id="compose-only-host-dir"),
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
    ("MIST_VOICE_YIELD_BUDGET", "floor", "yield_budget", BUDGET_PRESETS["floor"]),
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
    ("MIST_VOICE_SPECULATIVE_START", "false", "endpointing.speculative_start", False),
    ("MIST_VOICE_MAX_SEGMENT_CHARS", "150", "segmenter.max_segment_chars", 150),
    ("MIST_VOICE_VAD_DEVICE", "cuda", "placement.vad_device", "cuda"),
    ("MIST_VOICE_EOT_DEVICE", "cuda:1", "placement.eot_device", "cuda:1"),
    (
        "MIST_VOICE_STT_DEVICE",
        "cuda:0, remote, cpu",
        "placement.stt_primary",
        ("cuda:0", "remote", "cpu"),
    ),
    ("MIST_VOICE_STT_YIELD_DEVICE", "cpu", "placement.stt_yield", ("cpu",)),
    (
        "MIST_VOICE_TTS_DEVICE",
        "remote,cuda,off",
        "placement.tts_primary",
        ("remote", "cuda", "off"),
    ),
    ("MIST_VOICE_TTS_YIELD_DEVICE", "off", "placement.tts_yield", ("off",)),
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
    ("MIST_VOICE_STT_MODEL", "small.en", "placement.stt_model", "small.en"),
    ("MIST_VOICE_TTS_VARIANT", "original", "placement.tts_variant", "original"),
    ("MIST_VOICE_TTS_MODEL", "org/tts@0123abc", "placement.tts_model", "org/tts@0123abc"),
    ("MIST_VOICE_VRAM_BUDGET_MIB", "6000", "placement.vram_budget_mib", 6000),
    (
        "MIST_SPEECH_SERVICE_URL",
        "http://speech.example:8091",
        "placement.speech_service_url",
        "http://speech.example:8091",
    ),
    ("MIST_VOICE_PROFILES_DIR", "/srv/profiles", "voice_reference.profiles_dir", "/srv/profiles"),
    ("VOICE_PROFILE", "cortana", "voice_reference.profile", "cortana"),
    (
        "MIST_VOICE_GPU_YIELD_TRIGGERS",
        "manual,pressure",
        "gpu_yield.triggers",
        frozenset({"manual", "pressure"}),
    ),
    ("MIST_VOICE_GPU_YIELD_FREE_MIB", "1024", "gpu_yield.free_mib", 1024),
    ("MIST_VOICE_GPU_YIELD_AFTER_S", "5.5", "gpu_yield.yield_after_s", 5.5),
    ("MIST_VOICE_GPU_RESUME_AFTER_S", "60", "gpu_yield.resume_after_s", 60.0),
    ("MIST_VOICE_NO_SPEECH_THRESHOLD", "0.8", "endpointing.no_speech_threshold", 0.8),
    ("MIST_VOICE_LOGPROB_THRESHOLD", "-0.5", "endpointing.logprob_threshold", -0.5),
    ("MIST_VOICE_TOOL_CANCEL_GRACE_MS", "3000", "turn.tool_cancel_grace_ms", 3000),
    ("MIST_VOICE_WAKE_ARBITRATION_MS", "400", "barge_in.wake_arbitration_ms", 400),
]


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

    def test_budget_variables_accept_stage_overrides(self):
        settings = load_voice_settings({"MIST_VOICE_YIELD_BUDGET": "floor_yield,stt_final_ms=400"})

        assert settings.yield_budget.stt_final_ms == 400
        assert settings.yield_budget.total_p50_ms == 2720

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

    def test_https_speech_service_url_is_accepted(self):
        settings = load_voice_settings({"MIST_SPEECH_SERVICE_URL": "https://speech.example/base"})

        assert settings.placement.speech_service_url == "https://speech.example/base"

    def test_every_variable_in_the_configuration_list_is_covered(self):
        read = {case[0] for case in VARIABLES}

        assert read == {
            "MIST_VOICE_BUDGET",
            "MIST_VOICE_YIELD_BUDGET",
            "MIST_VOICE_CLIENT_PREBUFFER_MS",
            "MIST_VOICE_VAD_THRESHOLD",
            "MIST_VOICE_ENDPOINT_SILENCE_MS",
            "MIST_VOICE_EOT_THRESHOLD",
            "MIST_VOICE_EOT_MAX_SILENCE_MS",
            "MIST_VOICE_BARGE_IN_KEYWORDS",
            "MIST_VOICE_BARGE_IN_ARM_DELAY_MS",
            "MIST_VOICE_ECHO_GUARD_MS",
            "MIST_VOICE_SPECULATIVE_START",
            "MIST_VOICE_MAX_SEGMENT_CHARS",
            "MIST_VOICE_VAD_DEVICE",
            "MIST_VOICE_EOT_DEVICE",
            "MIST_VOICE_STT_DEVICE",
            "MIST_VOICE_STT_YIELD_DEVICE",
            "MIST_VOICE_TTS_DEVICE",
            "MIST_VOICE_TTS_YIELD_DEVICE",
            "MIST_VOICE_VAD_MODEL",
            "MIST_VOICE_EOT_MODEL",
            "MIST_VOICE_STT_MODEL",
            "MIST_VOICE_TTS_VARIANT",
            "MIST_VOICE_TTS_MODEL",
            "MIST_VOICE_PROFILES_DIR",
            "MIST_SPEECH_SERVICE_URL",
            "MIST_VOICE_GPU_YIELD_TRIGGERS",
            "MIST_VOICE_GPU_YIELD_FREE_MIB",
            "MIST_VOICE_GPU_YIELD_AFTER_S",
            "MIST_VOICE_GPU_RESUME_AFTER_S",
            "MIST_VOICE_NO_SPEECH_THRESHOLD",
            "MIST_VOICE_LOGPROB_THRESHOLD",
            "MIST_VOICE_VRAM_BUDGET_MIB",
            "MIST_VOICE_TOOL_CANCEL_GRACE_MS",
            "MIST_VOICE_WAKE_ARBITRATION_MS",
            "VOICE_PROFILE",
        }


class TestLoadVoiceSettingsErrors:
    def test_a_negative_number_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_ENDPOINT_SILENCE_MS"):
            load_voice_settings({"MIST_VOICE_ENDPOINT_SILENCE_MS": "-200"})

    def test_a_negative_float_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_GPU_YIELD_AFTER_S"):
            load_voice_settings({"MIST_VOICE_GPU_YIELD_AFTER_S": "-1.5"})

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

    def test_a_malformed_cuda_index_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_TTS_DEVICE"):
            load_voice_settings({"MIST_VOICE_TTS_DEVICE": "cuda:x,off"})

    def test_an_empty_placement_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_TTS_YIELD_DEVICE"):
            load_voice_settings({"MIST_VOICE_TTS_YIELD_DEVICE": ""})

    def test_a_placement_of_only_separators_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_STT_YIELD_DEVICE"):
            load_voice_settings({"MIST_VOICE_STT_YIELD_DEVICE": " , "})

    def test_a_repeated_placement_location_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_STT_YIELD_DEVICE"):
            load_voice_settings({"MIST_VOICE_STT_YIELD_DEVICE": "cpu,remote,cpu"})

    def test_off_before_the_last_place_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_TTS_DEVICE"):
            load_voice_settings({"MIST_VOICE_TTS_DEVICE": "cuda,off,remote"})

    def test_a_vad_device_list_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_VAD_DEVICE"):
            load_voice_settings({"MIST_VOICE_VAD_DEVICE": "cuda,cpu"})

    def test_a_remote_vad_device_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_VAD_DEVICE"):
            load_voice_settings({"MIST_VOICE_VAD_DEVICE": "remote"})

    def test_an_off_end_of_turn_device_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_EOT_DEVICE"):
            load_voice_settings({"MIST_VOICE_EOT_DEVICE": "off"})

    def test_an_unknown_gpu_yield_trigger_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_GPU_YIELD_TRIGGERS"):
            load_voice_settings({"MIST_VOICE_GPU_YIELD_TRIGGERS": "manual,game_detected"})

    def test_an_unknown_tts_variant_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_TTS_VARIANT"):
            load_voice_settings({"MIST_VOICE_TTS_VARIANT": "multilingual"})

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

    def test_a_ws_speech_service_url_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_SPEECH_SERVICE_URL"):
            load_voice_settings({"MIST_SPEECH_SERVICE_URL": "ws://speech.example:8091"})

    def test_a_speech_service_url_without_a_host_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_SPEECH_SERVICE_URL"):
            load_voice_settings({"MIST_SPEECH_SERVICE_URL": "http://"})

    def test_a_speech_service_url_with_a_query_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_SPEECH_SERVICE_URL"):
            load_voice_settings({"MIST_SPEECH_SERVICE_URL": "http://speech.example?x=1"})

    def test_an_unreadable_boolean_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_SPECULATIVE_START"):
            load_voice_settings({"MIST_VOICE_SPECULATIVE_START": "maybe"})

    def test_an_empty_keyword_list_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_BARGE_IN_KEYWORDS"):
            load_voice_settings({"MIST_VOICE_BARGE_IN_KEYWORDS": " , "})

    def test_an_unknown_budget_preset_raises_naming_the_variable_and_preset(self):
        with pytest.raises(ValueError, match="MIST_VOICE_BUDGET.*fastest"):
            load_voice_settings({"MIST_VOICE_BUDGET": "fastest"})

    def test_a_yield_budget_total_override_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_YIELD_BUDGET.*total_p50_ms"):
            load_voice_settings({"MIST_VOICE_YIELD_BUDGET": "floor_yield,total_p50_ms=1"})

    def test_an_empty_stt_model_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="MIST_VOICE_STT_MODEL"):
            load_voice_settings({"MIST_VOICE_STT_MODEL": " "})

    def test_an_empty_voice_profile_raises_naming_the_variable(self):
        with pytest.raises(ValueError, match="VOICE_PROFILE"):
            load_voice_settings({"VOICE_PROFILE": ""})
