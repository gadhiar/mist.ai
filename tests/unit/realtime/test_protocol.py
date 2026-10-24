"""Tests for the realtime protocol v2 frame codec and control message schemas.

`tests/fixtures/realtime_protocol_v2/vectors.json` is the contract the mist-frontend client copies
and asserts parity against, so every vector here is checked in both directions: decoding the
recorded bytes or JSON gives the recorded fields, and encoding those fields gives the recorded
bytes or JSON back.
"""

import json
import struct
from pathlib import Path

import pytest

from backend.errors import MistError
from backend.realtime.errors import ProtocolError, RealtimeError, SpeechEngineError
from backend.realtime.protocol import (
    CLIENT_MESSAGE_TYPES,
    CONTRACT,
    DOWNLINK_SAMPLE_RATE,
    HEADER_SIZE,
    MAGIC,
    PROTOCOL_VERSION,
    SERVER_MESSAGE_TYPES,
    UPLINK_SAMPLE_RATE,
    VOICE_DECISION_KINDS,
    DegradedDetail,
    Direction,
    Frame,
    MsgType,
    PlaybackStopMessage,
    StateCycleMessage,
    StreamCancelledMessage,
    StreamStartMessage,
    TranscriptionMessage,
    VoiceDecisionDropped,
    VoiceDecisionPlacement,
    VoiceDecisionWakeLost,
    VoiceStartMessage,
    VoiceStartPeek,
    VoiceStateMessage,
    decode_frame,
    encode_frame,
    encode_message,
    next_turn_tag,
    parse_client_message,
    parse_server_message,
    peek_voice_start,
)

VECTORS_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures" / "realtime_protocol_v2" / "vectors.json"
)
VECTORS = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))
BINARY_VECTORS = VECTORS["binary"]
CLIENT_VECTORS = VECTORS["control"]["client_to_server"]
SERVER_VECTORS = VECTORS["control"]["server_to_client"]


def _header(
    magic: int = MAGIC,
    msg_type: int = MsgType.MIC_AUDIO,
    turn_tag: int = 0,
    seq: int = 0,
    sample_rate: int = UPLINK_SAMPLE_RATE,
) -> bytes:
    return struct.pack("<IHHII", magic, msg_type, turn_tag, seq, sample_rate)


def _vector_id(vector: dict) -> str:
    return vector["name"]


def _vector_message(vectors: list[dict], name: str) -> dict:
    """A deep copy of the named vector's message, safe to mutate."""
    vector = next(v for v in vectors if v["name"] == name)
    return json.loads(json.dumps(vector["message"]))


# -- errors ------------------------------------------------------------------


def test_error_hierarchy_roots_at_mist_error():
    assert issubclass(RealtimeError, MistError)
    assert issubclass(ProtocolError, RealtimeError)
    assert issubclass(SpeechEngineError, RealtimeError)


def test_protocol_error_names_the_field():
    error = ProtocolError("magic", "expected 0x4D495354")
    assert error.field == "magic"
    assert "magic" in str(error)


# -- vectors file ------------------------------------------------------------


def test_vectors_declare_the_contract():
    assert VECTORS["contract"] == "MIST realtime protocol 2.0.0"
    assert VECTORS["contract"] == CONTRACT
    assert PROTOCOL_VERSION == "2.0.0"
    assert VECTORS["protocol_version"] == PROTOCOL_VERSION


def test_vectors_cover_every_binary_type():
    covered = {vector["msg_type"] for vector in BINARY_VECTORS}
    assert covered == {int(member) for member in MsgType}


def test_vectors_cover_every_client_message():
    covered = {vector["message"]["type"] for vector in CLIENT_VECTORS}
    assert covered == CLIENT_MESSAGE_TYPES


def test_vectors_cover_every_server_message_and_decision_kind():
    covered = {vector["message"]["type"] for vector in SERVER_VECTORS}
    assert covered == SERVER_MESSAGE_TYPES
    kinds = {
        vector["message"]["kind"]
        for vector in SERVER_VECTORS
        if vector["message"]["type"] == "voice_decision"
    }
    assert kinds == VOICE_DECISION_KINDS


def test_client_message_set_matches_design_doc():
    assert {
        "voice_start",
        "voice_stop",
        "ptt",
        "playback_progress",
        "playback_done",
        "barge_in",
        "wake",
        "gpu_yield",
        "text",
        "interrupt",
        "log_config",
        "subscribe_logs",
    } == CLIENT_MESSAGE_TYPES


def test_server_message_set_matches_design_doc():
    assert {
        "voice_state",
        "transcript_partial",
        "transcription",
        "vad_status",
        "voice_decision",
        "playback_stop",
        "playback_duck",
        "turn_metrics",
        "stream_start",
        "stream_cancelled",
        "state_cycle",
        "error",
        "session_started",
    } == SERVER_MESSAGE_TYPES
    assert {
        "endpoint",
        "hold",
        "release",
        "supersede",
        "barge_in",
        "barge_in_ignored",
        "dropped",
        "degraded",
        "overload",
        "wake_lost",
        "placement",
    } == VOICE_DECISION_KINDS


# -- binary frames: round trip -----------------------------------------------


@pytest.mark.parametrize("vector", BINARY_VECTORS, ids=_vector_id)
def test_binary_vector_decodes_to_recorded_fields(vector):
    frame = decode_frame(bytes.fromhex(vector["frame_hex"]))
    assert frame.msg_type == MsgType(vector["msg_type"])
    assert frame.direction == Direction(vector["direction"])
    assert frame.turn_tag == vector["turn_tag"]
    assert frame.seq == vector["seq"]
    assert frame.sample_rate == vector["sample_rate"]
    assert frame.payload == bytes.fromhex(vector["payload_hex"])


@pytest.mark.parametrize("vector", BINARY_VECTORS, ids=_vector_id)
def test_binary_vector_encodes_byte_exactly(vector):
    frame = Frame(
        msg_type=MsgType(vector["msg_type"]),
        turn_tag=vector["turn_tag"],
        seq=vector["seq"],
        sample_rate=vector["sample_rate"],
        payload=bytes.fromhex(vector["payload_hex"]),
    )
    encoded = encode_frame(frame)
    assert encoded.hex() == vector["frame_hex"]
    assert decode_frame(encoded) == frame


def test_header_is_sixteen_bytes():
    assert HEADER_SIZE == 16


# -- binary frames: the five rejections the plan names -----------------------


def test_decode_rejects_bad_magic():
    with pytest.raises(ProtocolError) as excinfo:
        decode_frame(_header(magic=0x4D495355) + b"\x00\x00")
    assert excinfo.value.field == "magic"


def test_decode_rejects_unknown_type():
    with pytest.raises(ProtocolError) as excinfo:
        decode_frame(_header(msg_type=0x11) + b"\x00\x00")
    assert excinfo.value.field == "msg_type"


def test_decode_rejects_odd_pcm_byte_count():
    with pytest.raises(ProtocolError) as excinfo:
        decode_frame(_header() + b"\x00\x00\x00")
    assert excinfo.value.field == "payload"
    assert "odd" in str(excinfo.value)


def test_decode_rejects_uplink_rate_other_than_16000():
    with pytest.raises(ProtocolError) as excinfo:
        decode_frame(_header(sample_rate=48000) + b"\x00\x00")
    assert excinfo.value.field == "sample_rate"


def test_decode_rejects_payload_over_one_second():
    one_second = UPLINK_SAMPLE_RATE * 2
    assert decode_frame(_header() + bytes(one_second)).payload == bytes(one_second)
    with pytest.raises(ProtocolError) as excinfo:
        decode_frame(_header() + bytes(one_second + 2))
    assert excinfo.value.field == "payload"
    assert "one second" in str(excinfo.value)


# -- binary frames: further header rules -------------------------------------


def test_decode_rejects_truncated_header():
    with pytest.raises(ProtocolError) as excinfo:
        decode_frame(_header()[:15])
    assert excinfo.value.field == "header"


def test_decode_rejects_downlink_rate_other_than_24000():
    header = _header(msg_type=MsgType.AUDIO_CHUNK, turn_tag=1, sample_rate=UPLINK_SAMPLE_RATE)
    with pytest.raises(ProtocolError) as excinfo:
        decode_frame(header + b"\x00\x00")
    assert excinfo.value.field == "sample_rate"


def test_downlink_payload_limit_is_one_second_at_24000():
    one_second = DOWNLINK_SAMPLE_RATE * 2
    header = _header(msg_type=MsgType.AUDIO_CHUNK, turn_tag=1, sample_rate=DOWNLINK_SAMPLE_RATE)
    assert len(decode_frame(header + bytes(one_second)).payload) == one_second
    with pytest.raises(ProtocolError) as excinfo:
        decode_frame(header + bytes(one_second + 2))
    assert excinfo.value.field == "payload"


@pytest.mark.parametrize("msg_type", [MsgType.AUDIO_COMPLETE, MsgType.INTERRUPT_FADE])
def test_decode_rejects_payload_on_empty_types(msg_type):
    header = _header(msg_type=msg_type, turn_tag=1, sample_rate=DOWNLINK_SAMPLE_RATE)
    with pytest.raises(ProtocolError) as excinfo:
        decode_frame(header + b"\x00\x00")
    assert excinfo.value.field == "payload"


def test_decode_rejects_nonzero_uplink_turn_tag():
    with pytest.raises(ProtocolError) as excinfo:
        decode_frame(_header(turn_tag=3) + b"\x00\x00")
    assert excinfo.value.field == "turn_tag"


def test_decode_rejects_zero_downlink_turn_tag():
    header = _header(msg_type=MsgType.AUDIO_COMPLETE, turn_tag=0, sample_rate=DOWNLINK_SAMPLE_RATE)
    with pytest.raises(ProtocolError) as excinfo:
        decode_frame(header)
    assert excinfo.value.field == "turn_tag"


def test_decode_with_expected_direction_rejects_the_other_direction():
    downlink = bytes.fromhex(BINARY_VECTORS[0]["frame_hex"])
    with pytest.raises(ProtocolError) as excinfo:
        decode_frame(downlink, direction=Direction.UP)
    assert excinfo.value.field == "msg_type"
    uplink = _header() + b"\x00\x00"
    assert decode_frame(uplink, direction=Direction.UP).msg_type == MsgType.MIC_AUDIO


def test_encode_applies_the_same_rules():
    with pytest.raises(ProtocolError) as excinfo:
        encode_frame(
            Frame(msg_type=MsgType.MIC_AUDIO, turn_tag=0, seq=0, sample_rate=16000, payload=b"\x00")
        )
    assert excinfo.value.field == "payload"
    with pytest.raises(ProtocolError) as excinfo:
        encode_frame(Frame(msg_type=0x7F, turn_tag=0, seq=0, sample_rate=16000))
    assert excinfo.value.field == "msg_type"
    with pytest.raises(ProtocolError) as excinfo:
        encode_frame(Frame(msg_type=MsgType.MIC_AUDIO, turn_tag=0, seq=2**32, sample_rate=16000))
    assert excinfo.value.field == "seq"


def test_next_turn_tag_wraps_past_zero():
    assert next_turn_tag(1) == 2
    assert next_turn_tag(65534) == 65535
    assert next_turn_tag(65535) == 1


# -- control messages: vectors -----------------------------------------------


@pytest.mark.parametrize("vector", CLIENT_VECTORS, ids=_vector_id)
def test_client_vector_round_trips(vector):
    message = parse_client_message(json.dumps(vector["message"]))
    assert message.type == vector["message"]["type"]
    assert json.loads(encode_message(message)) == vector["message"]


@pytest.mark.parametrize("vector", SERVER_VECTORS, ids=_vector_id)
def test_server_vector_round_trips(vector):
    message = parse_server_message(json.dumps(vector["message"]))
    assert message.type == vector["message"]["type"]
    assert json.loads(encode_message(message)) == vector["message"]


def test_server_message_built_in_code_serializes_to_the_vector():
    vector = next(v for v in SERVER_VECTORS if v["name"] == "playback_stop")
    message = PlaybackStopMessage(turn_id="turn-1", audio_tag=7, fade_ms=30, reason="barge_in")
    assert json.loads(encode_message(message)) == vector["message"]


def test_optional_fields_are_omitted_when_unset():
    start = StreamStartMessage(turn_id="turn-3", channel="text", origin="text")
    assert "audio_tag" not in json.loads(encode_message(start))
    cycle = StateCycleMessage(state="listen")
    assert json.loads(encode_message(cycle)) == {"type": "state_cycle", "state": "listen"}


def test_placement_decision_has_null_turn_id():
    decision = VoiceDecisionPlacement(
        reason="manual",
        detail={"mode": "yield", "engines": {"stt": "cpu", "tts": "cpu"}, "released_mib": None},
    )
    assert json.loads(encode_message(decision))["turn_id"] is None
    with pytest.raises(ProtocolError) as excinfo:
        parse_server_message(
            json.dumps(dict(json.loads(encode_message(decision)), turn_id="turn-1"))
        )
    assert excinfo.value.field == "turn_id"


# -- control messages: rejections --------------------------------------------


def test_unknown_client_type_raises_protocol_error():
    with pytest.raises(ProtocolError) as excinfo:
        parse_client_message(json.dumps({"type": "dance"}))
    assert excinfo.value.field == "type"


@pytest.mark.parametrize("removed", ["audio", "reset_vad"])
def test_removed_v1_messages_are_unknown(removed):
    with pytest.raises(ProtocolError) as excinfo:
        parse_client_message(json.dumps({"type": removed, "audio": [], "sample_rate": 16000}))
    assert excinfo.value.field == "type"


def test_missing_type_raises_protocol_error():
    with pytest.raises(ProtocolError) as excinfo:
        parse_client_message(json.dumps({"pressed": True}))
    assert excinfo.value.field == "type"


def test_unknown_server_type_raises_protocol_error():
    with pytest.raises(ProtocolError) as excinfo:
        parse_server_message(json.dumps({"type": "dance"}))
    assert excinfo.value.field == "type"


def test_missing_required_field_raises_protocol_error():
    message = dict(CLIENT_VECTORS[0]["message"])
    del message["mode"]
    with pytest.raises(ProtocolError) as excinfo:
        parse_client_message(json.dumps(message))
    assert excinfo.value.field == "mode"


def test_missing_nested_field_names_the_path():
    vector = next(v for v in SERVER_VECTORS if v["name"] == "voice_decision_endpoint")
    message = json.loads(json.dumps(vector["message"]))
    del message["detail"]["utterance_id"]
    with pytest.raises(ProtocolError) as excinfo:
        parse_server_message(json.dumps(message))
    assert excinfo.value.field == "detail.utterance_id"


def test_unknown_voice_decision_kind_raises_protocol_error():
    message = {"type": "voice_decision", "kind": "shrug", "turn_id": None, "reason": "x"}
    with pytest.raises(ProtocolError) as excinfo:
        parse_server_message(json.dumps(message))
    assert excinfo.value.field == "kind"


def test_voice_start_rejects_uplink_rate_other_than_16000():
    message = dict(CLIENT_VECTORS[0]["message"], sample_rate=48000)
    with pytest.raises(ProtocolError) as excinfo:
        parse_client_message(json.dumps(message))
    assert excinfo.value.field == "sample_rate"


def test_wrong_field_type_is_not_coerced():
    with pytest.raises(ProtocolError) as excinfo:
        parse_client_message(json.dumps({"type": "ptt", "pressed": "yes"}))
    assert excinfo.value.field == "pressed"


def test_invalid_json_raises_protocol_error():
    with pytest.raises(ProtocolError) as excinfo:
        parse_client_message("{not json")
    assert excinfo.value.field == "message"


def test_non_object_json_raises_protocol_error():
    with pytest.raises(ProtocolError) as excinfo:
        parse_client_message("[1, 2]")
    assert excinfo.value.field == "message"


def test_playback_stop_rejects_audio_tag_outside_range():
    with pytest.raises(ProtocolError) as excinfo:
        parse_server_message(
            json.dumps(
                {
                    "type": "playback_stop",
                    "turn_id": "t",
                    "audio_tag": 0,
                    "fade_ms": 30,
                    "reason": "barge_in",
                }
            )
        )
    assert excinfo.value.field == "audio_tag"


def test_engine_location_accepts_indexed_cuda_and_rejects_other_strings():
    vector = next(v for v in SERVER_VECTORS if v["name"] == "voice_state")
    message = json.loads(json.dumps(vector["message"]))
    message["engines"]["stt"]["location"] = "cuda:12"
    parse_server_message(json.dumps(message))
    message["engines"]["stt"]["location"] = "gpu"
    with pytest.raises(ProtocolError) as excinfo:
        parse_server_message(json.dumps(message))
    assert excinfo.value.field == "engines.stt.location"


def test_budget_stage_values_must_be_integers():
    vector = next(v for v in SERVER_VECTORS if v["name"] == "turn_metrics")
    message = json.loads(json.dumps(vector["message"]))
    message["budget"]["end_of_turn_ms"] = "fast"
    with pytest.raises(ProtocolError) as excinfo:
        parse_server_message(json.dumps(message))
    assert excinfo.value.field == "budget.end_of_turn_ms"


# The Design doc's Latency budget section names these; the frontend copies the vectors, so a
# vector key outside them would teach the client a name the server never sends.
DESIGN_BUDGET_KEYS = {
    "preset",
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
}
DESIGN_TIMING_KEYS = {
    "eot_decided",
    "stt_final",
    "context_built",
    "request_sent",
    "first_token",
    "first_segment",
    "first_tts_chunk",
    "first_frame_sent",
    "first_played",
    "tool_call_parsed",
    "tool_done",
}


@pytest.mark.parametrize("name", ["voice_state", "turn_metrics"])
def test_vector_budget_keys_come_from_the_design_doc(name):
    vector = next(v for v in SERVER_VECTORS if v["name"] == name)
    budget = vector["message"]["budget"]
    assert "preset" in budget
    assert set(budget) <= DESIGN_BUDGET_KEYS


# The Design doc's budget table (v1 plan, Latency budget): every preset the server can send, with
# its eleven rows. A vector carrying another preset or other figures teaches the client a budget
# the server never reports.
DESIGN_BUDGET_PRESETS = {
    "floor": {
        "end_of_turn_ms": 250,
        "stt_final_ms": 250,
        "context_build_ms": 0,
        "llm_first_token_ms": 250,
        "first_segment_ms": 120,
        "tts_first_chunk_ms": 500,
        "transport_playback_ms": 100,
        "total_p50_ms": 1470,
        "total_p90_ms": 2100,
        "tool_call_ms": 450,
        "barge_in_reaction_ms": 700,
    },
    "floor_yield": {
        "end_of_turn_ms": 250,
        "stt_final_ms": 600,
        "context_build_ms": 0,
        "llm_first_token_ms": 500,
        "first_segment_ms": 240,
        "tts_first_chunk_ms": 1200,
        "transport_playback_ms": 130,
        "total_p50_ms": 2920,
        "total_p90_ms": 4100,
        "tool_call_ms": 900,
        "barge_in_reaction_ms": 1200,
    },
    "memo": {
        "end_of_turn_ms": 250,
        "stt_final_ms": 50,
        "context_build_ms": 0,
        "llm_first_token_ms": 200,
        "first_segment_ms": 100,
        "tts_first_chunk_ms": 150,
        "transport_playback_ms": 100,
        "total_p50_ms": 850,
        "total_p90_ms": 1200,
        "tool_call_ms": 270,
        "barge_in_reaction_ms": 700,
    },
}


@pytest.mark.parametrize(
    "vector", [v for v in SERVER_VECTORS if "budget" in v["message"]], ids=_vector_id
)
def test_vector_budget_is_a_design_preset_with_its_figures(vector):
    budget = dict(vector["message"]["budget"])

    preset = budget.pop("preset")

    assert budget == DESIGN_BUDGET_PRESETS[preset]


def test_vector_timing_and_over_budget_keys_come_from_the_design_doc():
    vector = next(v for v in SERVER_VECTORS if v["name"] == "turn_metrics")
    assert vector["message"]["timings"]
    assert set(vector["message"]["timings"]) <= DESIGN_TIMING_KEYS
    assert set(vector["message"]["over_budget"]) <= DESIGN_BUDGET_KEYS - {"preset"}


# Each stage is its timestamp minus the previous stage's, floored at 0; `request_sent` and
# `first_frame_sent` are informational, and `total_p50_ms` is judged against `first_played`.
STAGE_TIMESTAMPS = [
    ("end_of_turn_ms", "eot_decided"),
    ("stt_final_ms", "stt_final"),
    ("context_build_ms", "context_built"),
    ("llm_first_token_ms", "first_token"),
    ("first_segment_ms", "first_segment"),
    ("tts_first_chunk_ms", "first_tts_chunk"),
    ("transport_playback_ms", "first_played"),
]


def test_vector_over_budget_follows_from_its_timings():
    message = next(v for v in SERVER_VECTORS if v["name"] == "turn_metrics")["message"]
    timings, budget = message["timings"], message["budget"]
    over = []
    previous = 0
    for stage, timestamp in STAGE_TIMESTAMPS:
        if max(timings[timestamp] - previous, 0) > budget[stage]:
            over.append(stage)
        previous = timings[timestamp]
    if timings["first_played"] > budget["total_p50_ms"]:
        over.append("total_p50_ms")
    assert over == ["stt_final_ms", "context_build_ms", "tts_first_chunk_ms", "total_p50_ms"]
    assert message["over_budget"] == over


# -- control messages: pinned contract choices --------------------------------


@pytest.mark.parametrize("played_ms", [480.5, "480"])
def test_integer_fields_are_strict(played_ms):
    """The client must round: a float or a numeric string is refused, not coerced."""
    message = {"type": "playback_done", "turn_id": "turn-1", "played_ms": played_ms}
    with pytest.raises(ProtocolError) as excinfo:
        parse_client_message(json.dumps(message))
    assert excinfo.value.field == "played_ms"


def test_unknown_field_on_client_message_is_dropped():
    """A field a newer client adds must not break this server."""
    message = parse_client_message(
        json.dumps({"type": "ptt", "pressed": True, "added_in_minor": 1})
    )
    assert json.loads(encode_message(message)) == {"type": "ptt", "pressed": True}


def test_unknown_field_on_server_message_is_dropped():
    message = parse_server_message(
        json.dumps({"type": "vad_status", "status": "speech_ended", "added_in_minor": 1})
    )
    assert json.loads(encode_message(message)) == {"type": "vad_status", "status": "speech_ended"}


def test_voice_state_mode_null_round_trips():
    """A broadcast reaches sessions that never sent `voice_start`, so `mode` is null."""
    message = _vector_message(SERVER_VECTORS, "voice_state_broadcast_before_voice_start")

    parsed = parse_server_message(json.dumps(message))

    assert parsed.mode is None
    assert parsed.active is False
    assert json.loads(encode_message(parsed)) == message


# -- second commit: utterance ids, answer correlation, the tts switch --------

_MISSING = object()


def _without(message: dict, path: tuple[str, ...]) -> dict:
    target = message
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]
    return message


@pytest.mark.parametrize(
    ("direction", "name", "path"),
    [
        pytest.param(
            "server", "voice_decision_barge_in", ("detail", "utterance_id"), id="barge_in-utterance"
        ),
        pytest.param(
            "server",
            "voice_decision_barge_in_ignored",
            ("detail", "utterance_id"),
            id="barge_in_ignored-utterance",
        ),
        pytest.param(
            "server",
            "voice_decision_wake_lost",
            ("detail", "utterance_id"),
            id="wake_lost-utterance",
        ),
        pytest.param(
            "server", "voice_decision_degraded", ("detail", "voiced_text"), id="degraded-voiced"
        ),
        pytest.param(
            "server",
            "voice_decision_degraded",
            ("detail", "text_from_token"),
            id="degraded-text-from-token",
        ),
        pytest.param("server", "voice_state", ("cause",), id="voice_state-cause"),
        pytest.param("server", "voice_state", ("req",), id="voice_state-req"),
        pytest.param("client", "voice_start", ("req",), id="voice_start-req"),
    ],
)
def test_each_new_key_is_required(direction, name, path):
    """A nullable new key is still required: a sender that omits it breaks the contract."""
    vectors, parse = (
        (SERVER_VECTORS, parse_server_message)
        if direction == "server"
        else (CLIENT_VECTORS, parse_client_message)
    )
    message = _without(_vector_message(vectors, name), path)

    with pytest.raises(ProtocolError) as excinfo:
        parse(json.dumps(message))

    assert excinfo.value.field == ".".join(path)


@pytest.mark.parametrize("name", ["voice_decision_barge_in", "voice_decision_barge_in_ignored"])
def test_barge_in_utterance_id_may_be_null(name):
    message = _vector_message(SERVER_VECTORS, name)
    message["detail"]["utterance_id"] = None

    parsed = parse_server_message(json.dumps(message))

    assert parsed.detail.utterance_id is None


def test_wake_lost_utterance_id_is_never_null():
    message = _vector_message(SERVER_VECTORS, "voice_decision_wake_lost")
    message["detail"]["utterance_id"] = None

    with pytest.raises(ProtocolError) as excinfo:
        parse_server_message(json.dumps(message))

    assert excinfo.value.field == "detail.utterance_id"


def test_degraded_text_from_token_rejects_a_negative_index():
    message = _vector_message(SERVER_VECTORS, "voice_decision_degraded")
    message["detail"]["text_from_token"] = -1

    with pytest.raises(ProtocolError) as excinfo:
        parse_server_message(json.dumps(message))

    assert excinfo.value.field == "detail.text_from_token"


def test_degraded_text_from_token_accepts_zero():
    """0 is a switch before any `stream_token` was sent: the whole text is unspoken."""
    message = _vector_message(SERVER_VECTORS, "voice_decision_degraded")
    message["detail"]["text_from_token"] = 0
    message["detail"]["voiced_text"] = ""

    parsed = parse_server_message(json.dumps(message))

    assert parsed.detail.text_from_token == 0


@pytest.mark.parametrize("key", ["voiced_text", "text_from_token"])
def test_degraded_tts_failed_with_a_null_switch_key_raises(key):
    """Without both keys the client cannot tell the spoken part of the turn from the rest."""
    message = _vector_message(SERVER_VECTORS, "voice_decision_degraded")
    message["detail"][key] = None

    with pytest.raises(ProtocolError) as excinfo:
        parse_server_message(json.dumps(message))

    assert excinfo.value.field == f"detail.{key}"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        pytest.param("voiced_text", "", id="voiced_text"),
        pytest.param("text_from_token", 0, id="tf"),
    ],
)
def test_degraded_other_reason_with_a_switch_key_set_raises(key, value):
    """Only `tts_failed` switches a turn to text, so another reason setting a key misleads."""
    message = _vector_message(SERVER_VECTORS, "voice_decision_degraded_tts_unavailable")
    message["detail"][key] = value

    with pytest.raises(ProtocolError) as excinfo:
        parse_server_message(json.dumps(message))

    assert excinfo.value.field == f"detail.{key}"


def test_voice_state_cause_outside_the_two_values_raises():
    message = dict(_vector_message(SERVER_VECTORS, "voice_state"), cause="reply")

    with pytest.raises(ProtocolError) as excinfo:
        parse_server_message(json.dumps(message))

    assert excinfo.value.field == "cause"


_BAD_REQS = [
    pytest.param(0, id="zero"),
    pytest.param(4294967296, id="over-u32"),
    pytest.param(1.0, id="float"),
    pytest.param("1", id="string"),
]


@pytest.mark.parametrize("req", [*_BAD_REQS, pytest.param(_MISSING, id="missing")])
def test_voice_start_req_must_be_a_u32_from_one(req):
    message = _vector_message(CLIENT_VECTORS, "voice_start")
    if req is _MISSING:
        del message["req"]
    else:
        message["req"] = req

    with pytest.raises(ProtocolError) as excinfo:
        parse_client_message(json.dumps(message))

    assert excinfo.value.field == "req"


@pytest.mark.parametrize("req", [1, 4294967295])
def test_voice_start_req_bounds_parse(req):
    message = dict(_vector_message(CLIENT_VECTORS, "voice_start"), req=req)

    parsed = parse_client_message(json.dumps(message))

    assert parsed.req == req


def test_voice_state_broadcast_with_req_set_raises():
    """A broadcast answers no request, so a `req` on it would be matched to the wrong one."""
    message = _vector_message(SERVER_VECTORS, "voice_state_broadcast_before_voice_start")
    message["req"] = 3

    with pytest.raises(ProtocolError) as excinfo:
        parse_server_message(json.dumps(message))

    assert excinfo.value.field == "req"


@pytest.mark.parametrize("req", _BAD_REQS)
def test_voice_state_answer_req_must_be_a_u32_from_one(req):
    message = dict(_vector_message(SERVER_VECTORS, "voice_state"), req=req)

    with pytest.raises(ProtocolError) as excinfo:
        parse_server_message(json.dumps(message))

    assert excinfo.value.field == "req"


@pytest.mark.parametrize("req", [1, 4294967295, None])
def test_voice_state_answer_req_bounds_and_null_parse(req):
    """An answer's `req` is null only when the `voice_start` it answers had no readable `req`."""
    message = dict(_vector_message(SERVER_VECTORS, "voice_state"), req=req)

    parsed = parse_server_message(json.dumps(message))

    assert parsed.cause == "answer"
    assert parsed.req == req


def _voice_start_text(**changes: object) -> str:
    return json.dumps(dict(_vector_message(CLIENT_VECTORS, "voice_start"), **changes))


_DEEPLY_NESTED = '{"type": "voice_start", "req": 7, "x": ' + "[" * 100_000 + "]" * 100_000 + "}"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(_voice_start_text(), VoiceStartPeek(req=1), id="valid"),
        pytest.param(_voice_start_text(mode="loud", req=5), VoiceStartPeek(req=5), id="bad-mode"),
        pytest.param(_voice_start_text(req="x"), VoiceStartPeek(req=None), id="req-string"),
        pytest.param(json.dumps({"type": "ptt"}), None, id="other-type"),
        pytest.param("{not json", None, id="not-json"),
        pytest.param(json.dumps({"type": "voice_start"}), VoiceStartPeek(req=None), id="no-req"),
        pytest.param(_voice_start_text(req=True), VoiceStartPeek(req=None), id="req-bool"),
        pytest.param(_voice_start_text(req=5.0), VoiceStartPeek(req=None), id="req-float"),
        pytest.param(_voice_start_text(req=0), VoiceStartPeek(req=None), id="req-zero"),
        pytest.param(
            _voice_start_text(req=4294967296), VoiceStartPeek(req=None), id="req-over-u32"
        ),
        pytest.param(
            _voice_start_text(req=4294967295), VoiceStartPeek(req=4294967295), id="req-max"
        ),
        pytest.param(json.dumps(["voice_start"]), None, id="json-array"),
        pytest.param(_DEEPLY_NESTED, None, id="nested-past-recursion-limit"),
    ],
)
def test_peek_voice_start(raw, expected):
    """The session answers a `voice_start` that failed validation with the `req` peeked here."""
    assert peek_voice_start(raw) == expected


def test_peek_reads_req_from_a_voice_start_that_fails_validation():
    raw = _voice_start_text(mode="loud", req=5)

    with pytest.raises(ProtocolError):
        parse_client_message(raw)

    assert peek_voice_start(raw) == VoiceStartPeek(req=5)


@pytest.mark.parametrize("reason", ["keyword", "voice_stop", "ignored"])
def test_dropped_closing_reasons_round_trip(reason):
    message = dict(_vector_message(SERVER_VECTORS, "voice_decision_dropped"), reason=reason)

    parsed = parse_server_message(json.dumps(message))

    assert json.loads(encode_message(parsed)) == message


def test_dropped_unknown_reason_still_raises():
    message = dict(_vector_message(SERVER_VECTORS, "voice_decision_dropped"), reason="bored")

    with pytest.raises(ProtocolError) as excinfo:
        parse_server_message(json.dumps(message))

    assert excinfo.value.field == "reason"


_EMPTY_MARKER_BAD_RATES = [
    pytest.param(msg_type, rate, id=f"{msg_type.name.lower()}-{rate}")
    for msg_type in (MsgType.AUDIO_COMPLETE, MsgType.INTERRUPT_FADE)
    for rate in (0, UPLINK_SAMPLE_RATE)
]


@pytest.mark.parametrize(("msg_type", "rate"), _EMPTY_MARKER_BAD_RATES)
def test_decode_rejects_empty_marker_at_rate_other_than_24000(msg_type, rate):
    data = _header(msg_type=msg_type, turn_tag=1, sample_rate=rate)

    with pytest.raises(ProtocolError) as excinfo:
        decode_frame(data)

    assert excinfo.value.field == "sample_rate"


@pytest.mark.parametrize(("msg_type", "rate"), _EMPTY_MARKER_BAD_RATES)
def test_encode_rejects_empty_marker_at_rate_other_than_24000(msg_type, rate):
    frame = Frame(msg_type=msg_type, turn_tag=1, seq=0, sample_rate=rate)

    with pytest.raises(ProtocolError) as excinfo:
        encode_frame(frame)

    assert excinfo.value.field == "sample_rate"


@pytest.mark.parametrize(
    ("documented", "terms"),
    [
        pytest.param(Frame, ("gap", "rebase", "restart"), id="Frame.seq"),
        pytest.param(
            VoiceStartMessage, ("req", "from 1", "strictly increasing", "`cause`"), id="req"
        ),
        pytest.param(
            VoiceStateMessage,
            (
                "answer",
                "broadcast",
                "engine",
                "trigger",
                "order",
                "req",
                "active",
                "superseded",
                "voice_stop",
                "invalid",
                "switch_failed",
            ),
            id="VoiceStateMessage",
        ),
        pytest.param(peek_voice_start, ("recursion limit",), id="peek_voice_start"),
        pytest.param(StreamCancelledMessage, ("null", "`text`"), id="StreamCancelledMessage"),
        pytest.param(TranscriptionMessage, ("closes",), id="TranscriptionMessage"),
        pytest.param(VoiceDecisionDropped, ("closes",), id="VoiceDecisionDropped"),
        pytest.param(VoiceDecisionWakeLost, ("closes",), id="VoiceDecisionWakeLost"),
        pytest.param(
            DegradedDetail,
            ("voiced_text", "text_from_token", "tts_failed", "rejects"),
            id="DegradedDetail",
        ),
    ],
)
def test_docstrings_state_the_v2_rules(documented, terms):
    """Fires when a v2 rule's key term is edited out of the docstring that states it.

    It checks that the terms are present, not that the sentences around them are right; that is
    the reviewer's job.
    """
    doc = documented.__doc__ or ""

    missing = [term for term in terms if term not in doc]

    assert missing == []
