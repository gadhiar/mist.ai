"""Tests for the realtime protocol v2 frame codec and control message schemas.

`tests/fixtures/realtime_protocol_v2/vectors.json` is the contract the mist-frontend client copies
and asserts parity against, so every vector here is checked in both directions: decoding the
recorded bytes or JSON gives the recorded fields, and encoding those fields gives the recorded
bytes or JSON back.
"""

import json
import re
import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import get_args

import pytest

from backend.capabilities import events as capability_events
from backend.errors import MistError
from backend.realtime import protocol
from backend.realtime.errors import ProtocolError, RealtimeError, SpeechEngineError
from backend.realtime.protocol import (
    CLIENT_MESSAGE_TYPES,
    CONTRACT,
    DOWNLINK_SAMPLE_RATE,
    HEADER_SIZE,
    MAGIC,
    MAX_WAKE_BACKLOG_MS,
    PROTOCOL_VERSION,
    ROOM_CLIENT_MESSAGE_TYPES,
    ROOM_SERVER_MESSAGE_TYPES,
    SERVER_MESSAGE_TYPES,
    UPLINK_SAMPLE_RATE,
    VOICE_DECISION_KINDS,
    ActionDecisionMessage,
    AlarmActionMessage,
    AlarmReportMessage,
    BackendMistState,
    DegradedDetail,
    Direction,
    Frame,
    GpuYieldMessage,
    MsgType,
    MuteMessage,
    PlaybackStopMessage,
    Spotter,
    StateCycleMessage,
    StreamCancelledMessage,
    StreamStartMessage,
    TranscriptionMessage,
    TtsVoice,
    TurnMetricsMessage,
    VoiceDecisionDropped,
    VoiceDecisionPlacement,
    VoiceDecisionWakeLost,
    VoiceDecisionWakeRejected,
    VoiceMode,
    VoiceStartMessage,
    VoiceStartPeek,
    VoiceStateMessage,
    WakeLostDetail,
    WakeMessage,
    WakeRejectedDetail,
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
# The invalid arrays take the frontend fixture's entry shape: `binary_invalid` entries are
# `{name, direction, frame_hex, field}` and `control_invalid` entries `{name, direction, json,
# rule, field}`, where `field` is the field the `ProtocolError` must name.
BINARY_INVALID_VECTORS = VECTORS["binary_invalid"]
CONTROL_INVALID_VECTORS = VECTORS["control_invalid"]


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


def test_vectors_have_both_invalid_arrays():
    """An empty array would let `test_invalid_vectors_fail_naming_their_field` pass vacuously."""
    assert BINARY_INVALID_VECTORS
    assert CONTROL_INVALID_VECTORS
    for vector in BINARY_INVALID_VECTORS:
        assert set(vector) == {"name", "direction", "frame_hex", "field"}
    for vector in CONTROL_INVALID_VECTORS:
        assert set(vector) == {"name", "direction", "json", "rule", "field"}


def _decode_invalid(vector: dict) -> None:
    decode_frame(bytes.fromhex(vector["frame_hex"]), direction=Direction(vector["direction"]))


def _parse_invalid(vector: dict) -> None:
    parse = {
        "client_to_server": parse_client_message,
        "server_to_client": parse_server_message,
    }[vector["direction"]]
    parse(json.dumps(vector["json"]))


@pytest.mark.parametrize(
    ("check", "vector"),
    [
        *(
            pytest.param(_decode_invalid, v, id=f"binary-{v['name']}")
            for v in BINARY_INVALID_VECTORS
        ),
        *(
            pytest.param(_parse_invalid, v, id=f"control-{v['name']}")
            for v in CONTROL_INVALID_VECTORS
        ),
    ],
)
def test_invalid_vectors_fail_naming_their_field(check, vector):
    with pytest.raises(ProtocolError) as excinfo:
        check(vector)

    assert excinfo.value.field == vector["field"]


# The removed push-to-talk and open-mode names. `control_invalid` must name them to prove they
# fail, so it is the one path the walk skips.
_REMOVED_MODES = {"open", "ptt"}


def _removed_names_in(node: object, path: str) -> list[str]:
    found: list[str] = []
    if isinstance(node, dict):
        if node.get("type") == "ptt":
            found.append(f"{path}: type ptt")
        for key, value in node.items():
            if key in ("local_kws", "pressed"):
                found.append(f"{path}.{key}")
            if key == "mode" and value in _REMOVED_MODES:
                found.append(f"{path}.mode: {value}")
            if key == "modes" and isinstance(value, list):
                found.extend(f"{path}.modes: {m}" for m in value if m in _REMOVED_MODES)
            found.extend(_removed_names_in(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for index, item in enumerate(node):
            found.extend(_removed_names_in(item, f"{path}[{index}]"))
    return found


def _removed_names_outside_control_invalid(vectors: dict) -> list[str]:
    return [
        *_removed_names_in(vectors["binary"], "binary"),
        *_removed_names_in(vectors["control"]["client_to_server"], "control.client_to_server"),
        *_removed_names_in(vectors["control"]["server_to_client"], "control.server_to_client"),
    ]


def test_removed_names_absent_outside_control_invalid():
    assert _removed_names_outside_control_invalid(VECTORS) == []

    # The walk must see the removed names where they would be: a copy with the old `ptt`
    # vector put back fails it, and one with a `ptt` vector only under `control_invalid` does not.
    restored = json.loads(json.dumps(VECTORS))
    restored["control"]["client_to_server"].append(
        {"name": "ptt", "message": {"type": "ptt", "pressed": True}}
    )
    restored_path = f"control.client_to_server[{len(CLIENT_VECTORS)}].message"
    assert _removed_names_outside_control_invalid(restored) == [
        f"{restored_path}: type ptt",
        f"{restored_path}.pressed",
    ]
    invalid_only = json.loads(json.dumps(VECTORS))
    invalid_only["control_invalid"].append(
        {"name": "ptt", "direction": "client_to_server", "json": {"type": "ptt"}}
    )
    assert _removed_names_outside_control_invalid(invalid_only) == []


def test_voice_mode_is_wake_only():
    assert get_args(VoiceMode) == ("wake",)


# -- wake by name: spotter, reference, backlog, kws, attend, wake_rejected ----


def test_spotter_is_server_or_endpoint():
    assert get_args(Spotter) == ("server", "endpoint")


def test_voice_start_reference_and_muted_default_to_false():
    message = _vector_message(CLIENT_VECTORS, "voice_start")
    del message["reference"]
    del message["muted"]

    parsed = parse_client_message(json.dumps(message))

    assert parsed.spotter == "server"
    assert parsed.reference is False
    assert parsed.muted is False


def test_voice_start_reference_with_endpoint_spotter_is_invalid_but_its_req_is_read():
    """An endpoint spots on the device, so the server has no name hit to score 0x11 against.

    The session answers it `refused: "invalid"`, which needs the `req` peeked from it.
    """
    vector = next(
        v for v in CONTROL_INVALID_VECTORS if v["name"].startswith("voice_start: reference true")
    )
    raw = json.dumps(vector["json"])

    with pytest.raises(ProtocolError) as excinfo:
        parse_client_message(raw)

    assert excinfo.value.field == "reference"
    assert peek_voice_start(raw) == VoiceStartPeek(req=vector["json"]["req"])


@pytest.mark.parametrize("backlog_ms", [0, 1500, MAX_WAKE_BACKLOG_MS])
def test_wake_backlog_ms_bounds_parse(backlog_ms):
    """0 parses: the gate's minimum is a server setting, which rejects it `no_backlog`."""
    message = dict(_vector_message(CLIENT_VECTORS, "wake"), backlog_ms=backlog_ms)

    parsed = parse_client_message(json.dumps(message))

    assert isinstance(parsed, WakeMessage)
    assert parsed.backlog_ms == backlog_ms


@pytest.mark.parametrize("backlog_ms", [-1, MAX_WAKE_BACKLOG_MS + 1, 1500.0, "1500"])
def test_wake_backlog_ms_outside_the_wire_bound_raises(backlog_ms):
    message = dict(_vector_message(CLIENT_VECTORS, "wake"), backlog_ms=backlog_ms)

    with pytest.raises(ProtocolError) as excinfo:
        parse_client_message(json.dumps(message))

    assert excinfo.value.field == "backlog_ms"


def test_max_wake_backlog_is_ten_seconds():
    assert MAX_WAKE_BACKLOG_MS == 10000


def test_state_cycle_attend_is_a_backend_state():
    assert "attend" in get_args(BackendMistState)
    message = _vector_message(SERVER_VECTORS, "state_cycle_attend_sent_at_the_name_hit")

    parsed = parse_server_message(json.dumps(message))

    assert isinstance(parsed, StateCycleMessage)
    assert parsed.state == "attend"
    assert json.loads(encode_message(parsed)) == message


WAKE_REJECTED_REASONS = [
    "self_speech",
    "self_media",
    "not_addressed",
    "no_backlog",
]


def test_wake_rejected_reasons_are_exactly_the_four():
    schema = VoiceDecisionWakeRejected.model_json_schema()

    assert schema["properties"]["reason"]["enum"] == WAKE_REJECTED_REASONS


def test_wake_rejected_refuses_lost_arbitration_which_wake_lost_replaced():
    """An arbitration loser gets `wake_lost` (A52 option A), so the rejection reason is gone."""
    message = dict(
        _vector_message(SERVER_VECTORS, "voice_decision_wake_rejected"), reason="lost_arbitration"
    )

    with pytest.raises(ProtocolError) as excinfo:
        parse_server_message(json.dumps(message))

    assert excinfo.value.field == "reason"


def test_the_wake_rejected_reason_vector_names_the_four_reasons():
    (vector,) = (
        v
        for v in CONTROL_INVALID_VECTORS
        if v["name"].startswith("voice_decision wake_rejected: reason")
    )

    assert vector["rule"] == "wake_rejected reasons: " + ", ".join(WAKE_REJECTED_REASONS)


@pytest.mark.parametrize("reason", WAKE_REJECTED_REASONS)
def test_wake_rejected_round_trips_with_each_reason(reason):
    message = dict(_vector_message(SERVER_VECTORS, "voice_decision_wake_rejected"), reason=reason)

    parsed = parse_server_message(json.dumps(message))

    assert isinstance(parsed, VoiceDecisionWakeRejected)
    assert json.loads(encode_message(parsed)) == message


def test_wake_rejected_detail_cannot_carry_an_utterance_or_text():
    """A rejected candidate was never published, so nothing about its speech travels."""
    assert set(WakeRejectedDetail.model_fields) == {"source", "score", "reference_score"}
    message = _vector_message(SERVER_VECTORS, "voice_decision_wake_rejected")
    sent = json.loads(json.dumps(message))
    sent["detail"].update(utterance_id="utt-9", text="hey mist")

    parsed = parse_server_message(json.dumps(sent))

    assert json.loads(encode_message(parsed))["detail"] == message["detail"]


@pytest.mark.parametrize("source", ["server", "endpoint"])
def test_wake_rejected_source_is_a_spotter(source):
    message = _vector_message(SERVER_VECTORS, "voice_decision_wake_rejected")
    message["detail"]["source"] = source

    parsed = parse_server_message(json.dumps(message))

    assert parsed.detail.source == source


def test_capabilities_announce_spotters_and_reference():
    message = _vector_message(SERVER_VECTORS, "session_started")

    voice = parse_server_message(json.dumps(message)).capabilities.voice

    assert voice.modes == ["wake"]
    assert voice.spotters == ["endpoint", "server"]
    assert voice.reference is True


def test_removed_push_to_talk_names_are_gone_from_the_source():
    """T31's acceptance grep over `backend/realtime`, pinned for `protocol.py`."""
    source = PROTOCOL_SOURCE.read_text(encoding="utf-8")

    assert re.findall(r"\bptt\b|Ptt|ptt_release|local_kws", source) == []


def test_client_message_set_matches_design_doc():
    assert {
        "voice_start",
        "voice_stop",
        "playback_progress",
        "playback_done",
        "barge_in",
        "wake",
        "gpu_yield",
        "mute",
        "text",
        "interrupt",
        "log_config",
        "subscribe_logs",
        "action_decision",
        "alarm_action",
        "alarm_report",
    } == CLIENT_MESSAGE_TYPES


def test_room_client_message_set_is_the_device_types_within_the_client_set():
    """A room session accepts these nine; the firmware test pins them to what the device sends."""
    assert {
        "voice_start",
        "wake",
        "barge_in",
        "mute",
        "interrupt",
        "playback_progress",
        "playback_done",
        "alarm_action",
        "alarm_report",
    } == ROOM_CLIENT_MESSAGE_TYPES
    assert ROOM_CLIENT_MESSAGE_TYPES < CLIENT_MESSAGE_TYPES


def _capability_event_types() -> frozenset[str]:
    """The `type` tags of capabilities' outbound events, the `...Event` models of `events.py`."""
    types: set[str] = set()
    for name, value in vars(capability_events).items():
        if name.endswith("Event") and isinstance(value, type) and "type" in value.model_fields:
            types.update(get_args(value.model_fields["type"].annotation))
    return frozenset(types)


def test_room_server_message_set_is_the_types_the_device_acts_on():
    """A room session is sent these eleven; the firmware test pins them to what the codec reads."""
    assert {
        "session_started",
        "voice_state",
        "state_cycle",
        "voice_decision",
        "stream_start",
        "playback_stop",
        "playback_duck",
        "error",
        "alarm_state",
        "alarm_fired",
        "alarm_action_result",
    } == ROOM_SERVER_MESSAGE_TYPES
    events = _capability_event_types()
    assert {"alarm_state", "alarm_fired", "alarm_action_result", "action_proposed"} <= events
    assert SERVER_MESSAGE_TYPES | events >= ROOM_SERVER_MESSAGE_TYPES


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
        "wake_rejected",
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
        decode_frame(_header(msg_type=0x12) + b"\x00\x00")
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


def test_ref_audio_is_an_uplink_type_0x11():
    assert MsgType.REF_AUDIO == 0x11
    assert Frame(MsgType.REF_AUDIO, 0, 0, UPLINK_SAMPLE_RATE).direction is Direction.UP


def test_ref_audio_640_byte_frame_round_trips_byte_exactly():
    """One 20 ms frame of the PC's output, PCM16 mono at 16 kHz, as the desktop sends it."""
    vector = next(v for v in BINARY_VECTORS if v["msg_type"] == MsgType.REF_AUDIO)
    data = bytes.fromhex(vector["frame_hex"])

    frame = decode_frame(data, direction=Direction.UP)

    assert frame.msg_type is MsgType.REF_AUDIO
    assert frame.turn_tag == 0
    assert frame.sample_rate == UPLINK_SAMPLE_RATE
    assert len(frame.payload) == 640
    assert encode_frame(frame) == data


def test_ref_audio_invalid_vectors_cover_the_four_rules():
    """Rate 24000, an odd payload, a downlink decode and a nonzero tag each have a vector."""
    ref_audio = [v for v in BINARY_INVALID_VECTORS if v["frame_hex"][8:12] == "1100"]
    covered = {(v["field"], v["direction"]) for v in ref_audio}

    assert covered == {
        ("sample_rate", "up"),
        ("payload", "up"),
        ("msg_type", "down"),
        ("turn_tag", "up"),
    }


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
        reason="power_off",
        detail={"power_level": "off", "engines": {"stt": "cpu", "tts": "off"}, "released_mib": 0},
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
        parse_client_message(json.dumps({"active": True}))
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
        parse_client_message(json.dumps({"type": "mute", "active": "yes"}))
    assert excinfo.value.field == "active"


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
    # v3's D4: the caption's P90 lag over the turn's partials, and the wake's screen latency.
    "caption_lag_ms",
    "wake_screen_ms",
    # v5: the hit frame's arrival to the addressee verdict.
    "wake_accept_ms",
}


@pytest.mark.parametrize("name", ["voice_state", "turn_metrics"])
def test_vector_budget_keys_come_from_the_design_doc(name):
    vector = next(v for v in SERVER_VECTORS if v["name"] == name)
    budget = vector["message"]["budget"]
    assert "preset" in budget
    assert set(budget) <= DESIGN_BUDGET_KEYS


# The Design doc's budget table (v1 plan, Latency budget): every preset the server can send, with
# its eleven rows. v3 keeps only `floor` and `memo` (its Settings section). A vector carrying
# another preset or other figures teaches the client a budget the server never reports.
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
        json.dumps({"type": "mute", "active": True, "added_in_minor": 1})
    )
    assert json.loads(encode_message(message)) == {"type": "mute", "active": True}


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
        pytest.param("server", "voice_state", ("power_level",), id="voice_state-power_level"),
        pytest.param("server", "voice_state", ("muted",), id="voice_state-muted"),
        pytest.param("server", "turn_metrics", ("power_level",), id="turn_metrics-power_level"),
        pytest.param(
            "server",
            "voice_decision_placement",
            ("detail", "power_level"),
            id="placement-power_level",
        ),
        pytest.param("server", "voice_state", ("tts_voice", "engine"), id="tts_voice-engine"),
        pytest.param("server", "voice_state", ("tts_voice", "voice"), id="tts_voice-voice"),
        pytest.param("server", "voice_state", ("tts_voice", "profile"), id="tts_voice-profile"),
        pytest.param("client", "voice_start", ("spotter",), id="voice_start-spotter"),
        pytest.param("client", "wake", ("backlog_ms",), id="wake-backlog_ms"),
        pytest.param("server", "voice_state", ("engines", "kws"), id="engines-kws"),
        pytest.param(
            "server",
            "session_started",
            ("capabilities", "voice", "spotters"),
            id="capabilities-spotters",
        ),
        pytest.param(
            "server",
            "session_started",
            ("capabilities", "voice", "reference"),
            id="capabilities-reference",
        ),
        pytest.param(
            "server",
            "voice_decision_wake_rejected",
            ("detail", "reference_score"),
            id="wake_rejected-reference_score",
        ),
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
        pytest.param(json.dumps({"type": "mute"}), None, id="other-type"),
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
                "order",
                "req",
                "active",
                "superseded",
                "voice_stop",
                "invalid",
                "power_off",
                "power_level",
                "muted",
                "per session",
                "reload_failed",
            ),
            id="VoiceStateMessage",
        ),
        pytest.param(GpuYieldMessage, ("frontend v1", "ignores"), id="GpuYieldMessage"),
        pytest.param(MuteMessage, ("req", "broadcast", "dropped"), id="MuteMessage"),
        pytest.param(
            VoiceStartMessage, ("muted", "false", "advisory"), id="VoiceStartMessage.muted"
        ),
        pytest.param(
            VoiceDecisionPlacement,
            ("power_off", "power_restored", "reload_failed", "release_timeout"),
            id="VoiceDecisionPlacement",
        ),
        pytest.param(VoiceDecisionDropped, ("muted", "power_off"), id="VoiceDecisionDropped.v3"),
        pytest.param(peek_voice_start, ("recursion limit",), id="peek_voice_start"),
        pytest.param(StreamCancelledMessage, ("null", "`text`"), id="StreamCancelledMessage"),
        pytest.param(TranscriptionMessage, ("closes",), id="TranscriptionMessage"),
        pytest.param(VoiceDecisionDropped, ("closes",), id="VoiceDecisionDropped"),
        pytest.param(VoiceDecisionWakeLost, ("closes",), id="VoiceDecisionWakeLost"),
        pytest.param(
            VoiceDecisionWakeLost,
            ("arbitration", "endpoint", "backlog", "fresh"),
            id="VoiceDecisionWakeLost.fresh_id",
        ),
        pytest.param(
            WakeLostDetail, ("winner_score", "fresh", "backlog"), id="WakeLostDetail.fresh_id"
        ),
        pytest.param(
            VoiceDecisionWakeRejected,
            ("self_speech", "self_media", "not_addressed", "no_backlog", "wake_lost"),
            id="VoiceDecisionWakeRejected",
        ),
        pytest.param(
            DegradedDetail,
            ("voiced_text", "text_from_token", "tts_failed", "rejects"),
            id="DegradedDetail",
        ),
        pytest.param(
            VoiceStartMessage,
            ("spotter", "`server`", "`endpoint`", "reference", "0x11", "invalid"),
            id="VoiceStartMessage.spotter",
        ),
        pytest.param(WakeMessage, ("backlog_ms", "no_backlog", "score"), id="WakeMessage"),
        pytest.param(WakeRejectedDetail, ("never published",), id="WakeRejectedDetail"),
        pytest.param(
            TurnMetricsMessage,
            ("wake_accept_ms", "wake_screen_ms", "hit_lag_ms", "`attend`", "server side"),
            id="TurnMetricsMessage",
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


# -- third commit: the power level, mute and stock voices --------------------

PROTOCOL_SOURCE = Path(__file__).resolve().parents[3] / "backend" / "realtime" / "protocol.py"


def test_gpu_yield_from_frontend_v1_still_parses():
    """Frontend v1's C1 sends `gpu_yield`; an unknown type would end its session's voice."""
    sent_by_frontend_v1 = {"type": "gpu_yield", "active": True, "source": "fullscreen"}

    parsed = parse_client_message(json.dumps(sent_by_frontend_v1))

    assert isinstance(parsed, GpuYieldMessage)
    assert parsed.active is True


@pytest.mark.parametrize("retired", ["gpu", "speech_service"])
def test_voice_state_ignores_the_retired_gpu_keys(retired):
    """A v2.0 sender's `gpu` or `speech_service` is an unknown key, dropped like any other."""
    message = _vector_message(SERVER_VECTORS, "voice_state")
    sent = dict(message, **{retired: {"mode": "primary", "triggers": []}})

    parsed = parse_server_message(json.dumps(sent))

    assert json.loads(encode_message(parsed)) == message


@pytest.mark.parametrize("engine", ["vad", "eot", "stt", "tts", "kws"])
def test_engine_location_remote_is_refused(engine):
    message = _vector_message(SERVER_VECTORS, "voice_state")
    message["engines"][engine]["location"] = "remote"

    with pytest.raises(ProtocolError) as excinfo:
        parse_server_message(json.dumps(message))

    assert excinfo.value.field == f"engines.{engine}.location"


@pytest.mark.parametrize("location", ["cpu", "cuda", "cuda:1", "off"])
def test_engine_location_keeps_the_local_values(location):
    message = _vector_message(SERVER_VECTORS, "voice_state")
    message["engines"]["tts"]["location"] = location

    parsed = parse_server_message(json.dumps(message))

    assert parsed.engines.tts.location == location


def test_tts_voice_takes_exactly_engine_voice_and_profile():
    """`voice` is the voice identity id; its audio is private, so no provenance fields."""
    assert set(TtsVoice.model_fields) == {"engine", "voice", "profile"}
    message = _vector_message(SERVER_VECTORS, "voice_state")
    sent = json.loads(json.dumps(message))
    sent["tts_voice"].update(variant="turbo", source="builtin", seconds=None, sha256_prefix=None)

    parsed = parse_server_message(json.dumps(sent))

    assert json.loads(encode_message(parsed))["tts_voice"] == message["tts_voice"]


@pytest.mark.parametrize("key", ["engine", "voice", "profile"])
def test_tts_voice_fields_are_never_null(key):
    message = _vector_message(SERVER_VECTORS, "voice_state")
    message["tts_voice"][key] = None

    with pytest.raises(ProtocolError) as excinfo:
        parse_server_message(json.dumps(message))

    assert excinfo.value.field == f"tts_voice.{key}"


POWER_PLACEMENT_REASONS = ["power_off", "power_restored", "reload_failed", "release_timeout"]


def test_placement_reasons_are_exactly_the_power_reasons():
    schema = VoiceDecisionPlacement.model_json_schema()

    assert schema["properties"]["reason"]["enum"] == POWER_PLACEMENT_REASONS


@pytest.mark.parametrize("reason", POWER_PLACEMENT_REASONS)
def test_placement_power_reasons_round_trip(reason):
    message = dict(_vector_message(SERVER_VECTORS, "voice_decision_placement"), reason=reason)

    parsed = parse_server_message(json.dumps(message))

    assert json.loads(encode_message(parsed)) == message


@pytest.mark.parametrize(
    "retired",
    [
        "manual",
        "fullscreen",
        "pressure",
        "resume",
        "switch_failed",
        "remote_lost",
        "remote_restored",
    ],
)
def test_placement_retired_reasons_raise(retired):
    message = dict(_vector_message(SERVER_VECTORS, "voice_decision_placement"), reason=retired)

    with pytest.raises(ProtocolError) as excinfo:
        parse_server_message(json.dumps(message))

    assert excinfo.value.field == "reason"


def test_placement_detail_carries_the_level_not_the_gpu_mode():
    message = _vector_message(SERVER_VECTORS, "voice_decision_placement")
    message["detail"]["mode"] = "yield"

    parsed = parse_server_message(json.dumps(message))

    assert "mode" not in json.loads(encode_message(parsed))["detail"]
    assert parsed.detail.power_level == "off"


@pytest.mark.parametrize("reason", ["muted", "power_off"])
def test_dropped_gains_muted_and_power_off(reason):
    message = dict(_vector_message(SERVER_VECTORS, "voice_decision_dropped"), reason=reason)

    parsed = parse_server_message(json.dumps(message))

    assert json.loads(encode_message(parsed)) == message


def test_voice_state_refused_power_off_round_trips():
    message = _vector_message(SERVER_VECTORS, "voice_state_answer_refused_power_off")

    parsed = parse_server_message(json.dumps(message))

    assert parsed.refused == "power_off"
    assert parsed.active is False
    assert json.loads(encode_message(parsed)) == message


@pytest.mark.parametrize("name", ["voice_state", "turn_metrics"])
@pytest.mark.parametrize("level", ["full", "low", "off"])
def test_power_level_takes_the_capability_levels(name, level):
    message = dict(_vector_message(SERVER_VECTORS, name), power_level=level)

    parsed = parse_server_message(json.dumps(message))

    assert parsed.power_level == level


@pytest.mark.parametrize("name", ["voice_state", "turn_metrics"])
@pytest.mark.parametrize("level", ["primary", "yield", "Full"])
def test_power_level_rejects_other_values(name, level):
    message = dict(_vector_message(SERVER_VECTORS, name), power_level=level)

    with pytest.raises(ProtocolError) as excinfo:
        parse_server_message(json.dumps(message))

    assert excinfo.value.field == "power_level"


def test_turn_metrics_ignores_the_retired_placement_keys():
    message = _vector_message(SERVER_VECTORS, "turn_metrics")
    sent = json.loads(json.dumps(message))
    sent["placement_mode"] = "primary"
    sent["lanes"]["remote_rtt_ms"] = None

    parsed = parse_server_message(json.dumps(sent))

    assert json.loads(encode_message(parsed)) == message


def test_voice_state_muted_must_be_a_bool():
    message = dict(_vector_message(SERVER_VECTORS, "voice_state"), muted="yes")

    with pytest.raises(ProtocolError) as excinfo:
        parse_server_message(json.dumps(message))

    assert excinfo.value.field == "muted"


@pytest.mark.parametrize("active", [True, False])
def test_mute_round_trips(active):
    message = {"type": "mute", "active": active}

    parsed = parse_client_message(json.dumps(message))

    assert isinstance(parsed, MuteMessage)
    assert parsed.active is active
    assert json.loads(encode_message(parsed)) == message


def test_mute_carries_no_req():
    """The answer cause stays tied to `voice_start`, so a `mute` has no request to answer."""
    assert "req" not in MuteMessage.model_fields

    parsed = parse_client_message(json.dumps({"type": "mute", "active": True, "req": 4}))

    assert json.loads(encode_message(parsed)) == {"type": "mute", "active": True}


def test_mute_active_is_required_and_strict():
    with pytest.raises(ProtocolError) as excinfo:
        parse_client_message(json.dumps({"type": "mute"}))
    assert excinfo.value.field == "active"
    with pytest.raises(ProtocolError) as excinfo:
        parse_client_message(json.dumps({"type": "mute", "active": 1}))
    assert excinfo.value.field == "active"


def test_voice_start_without_muted_reads_false():
    """Frontend v1's C1 has no mute and sends no `muted`."""
    message = _vector_message(CLIENT_VECTORS, "voice_start")
    del message["muted"]

    parsed = parse_client_message(json.dumps(message))

    assert parsed.muted is False


def test_voice_start_muted_true_round_trips():
    message = _vector_message(CLIENT_VECTORS, "voice_start_muted")

    parsed = parse_client_message(json.dumps(message))

    assert parsed.muted is True
    assert json.loads(encode_message(parsed)) == message


def test_voice_start_muted_is_strict():
    message = dict(_vector_message(CLIENT_VECTORS, "voice_start"), muted="true")

    with pytest.raises(ProtocolError) as excinfo:
        parse_client_message(json.dumps(message))

    assert excinfo.value.field == "muted"


@pytest.mark.parametrize(
    "name",
    [
        "voice_state_broadcast_before_voice_start",
        "voice_state_broadcast_muted",
        "voice_state_broadcast_power_off_capability_only",
    ],
)
def test_voice_state_broadcast_vectors_answer_no_request(name):
    message = _vector_message(SERVER_VECTORS, name)

    parsed = parse_server_message(json.dumps(message))

    assert parsed.cause == "broadcast"
    assert parsed.req is None


def test_retired_gpu_names_are_gone_from_the_contract():
    """The T1c acceptance grep: no retired GPU yield or voice-cloning name is left."""
    retired = [
        "GpuMode",
        "GpuState",
        "SpeechServiceState",
        "remote_rtt_ms",
        "placement_mode",
        "sha256_prefix",
    ]
    source = PROTOCOL_SOURCE.read_text(encoding="utf-8")

    assert [name for name in retired if name in source] == []


def test_vectors_keep_the_gpu_yield_vector():
    names = {vector["name"] for vector in CLIENT_VECTORS}

    assert {"gpu_yield", "mute", "voice_start", "voice_start_muted"} <= names


# -- main's 1.x control messages ---------------------------------------------

# Main's own example payloads, from `tests/unit/capabilities/test_events.py`: the inbound
# parametrization for `action_decision` and `alarm_action`, and the rang report that
# `test_a_rang_report_validates_and_round_trips` builds for `alarm_report`.
MAIN_CONTROL_EXAMPLES = {
    "action_decision": {"type": "action_decision", "proposal_id": "p1", "decision": "approve"},
    "alarm_action": {"type": "alarm_action", "alarm_id": "a1", "action": "snooze", "minutes": 5},
    "alarm_report": {
        "type": "alarm_report",
        "alarm_id": "a1",
        "outcome": "rang",
        "at": "2026-01-05T06:59:00Z",
        "rang_at": "2026-01-05T06:59:00Z",
        "request_id": "q1",
    },
}
# The instant main's capabilities tests validate the example report against.
MAIN_REPORT_NOW = datetime(2026, 1, 5, 7, 0, tzinfo=UTC)

MAIN_CONTROL_MODELS = {
    "action_decision": capability_events.ActionDecisionMessage,
    "alarm_action": capability_events.AlarmActionMessage,
    "alarm_report": capability_events.AlarmReportMessage,
}

ADDED_CONTROL_MODELS = [
    pytest.param(ActionDecisionMessage, MAIN_CONTROL_MODELS["action_decision"], id="decision"),
    pytest.param(AlarmActionMessage, MAIN_CONTROL_MODELS["alarm_action"], id="alarm_action"),
    pytest.param(AlarmReportMessage, MAIN_CONTROL_MODELS["alarm_report"], id="alarm_report"),
]


def _field_contract(model: type) -> dict[str, tuple]:
    """Each field's type, constraints and default; `type` by its tag only.

    The protocol's `type` defaults to its tag, which `CLIENT_MESSAGE_TYPES` reads, where
    capabilities' is required, so only its annotation is compared.
    """
    contract: dict[str, tuple] = {}
    for name, info in model.model_fields.items():
        if name == "type":
            contract[name] = (info.annotation,)
        else:
            contract[name] = (info.annotation, info.metadata, info.is_required(), info.default)
    return contract


@pytest.mark.parametrize(("ours", "mains"), ADDED_CONTROL_MODELS)
def test_added_control_message_fields_are_mains(ours, mains):
    """Fires when a field drifts between the protocol model and capabilities' model."""
    assert list(ours.model_fields) == list(mains.model_fields)
    assert _field_contract(ours) == _field_contract(mains)
    assert ours.model_config.get("extra") == mains.model_config.get("extra")


@pytest.mark.parametrize("name", sorted(MAIN_CONTROL_EXAMPLES))
def test_added_control_message_parses_from_mains_example_and_round_trips(name):
    example = MAIN_CONTROL_EXAMPLES[name]

    parsed = parse_client_message(json.dumps(example))

    assert parsed.type == name
    assert json.loads(encode_message(parsed)) == example
    assert _vector_message(CLIENT_VECTORS, name) == example
    main = MAIN_CONTROL_MODELS[name].model_validate(example, context={"now": MAIN_REPORT_NOW})
    assert main.to_payload() == example


def test_alarm_report_refuses_an_endpoint_key():
    """The server attaches the connection's endpoint, so no report may name its own."""
    message = dict(MAIN_CONTROL_EXAMPLES["alarm_report"], endpoint="room")

    with pytest.raises(ProtocolError) as excinfo:
        parse_client_message(json.dumps(message))

    assert excinfo.value.field == "endpoint"
    assert "endpoint" not in AlarmReportMessage.model_fields


@pytest.mark.parametrize("name", ["action_decision", "alarm_action"])
def test_other_added_control_messages_ignore_an_unknown_key(name):
    message = dict(MAIN_CONTROL_EXAMPLES[name], endpoint="room")

    parsed = parse_client_message(json.dumps(message))

    assert json.loads(encode_message(parsed)) == MAIN_CONTROL_EXAMPLES[name]


def test_alarm_action_omits_request_id_only_when_absent():
    """The same payload capabilities' `to_payload` gives: `minutes` stays, null or not."""
    stop = {"type": "alarm_action", "alarm_id": "a1", "action": "stop"}

    parsed = parse_client_message(json.dumps(stop))
    assert json.loads(encode_message(parsed)) == dict(stop, minutes=None)

    with_request = dict(stop, request_id="q1")
    parsed = parse_client_message(json.dumps(with_request))
    assert json.loads(encode_message(parsed)) == dict(with_request, minutes=None)


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        pytest.param({"minutes": 61}, "minutes", id="minutes-over-60"),
        pytest.param({"minutes": "5"}, "minutes", id="minutes-not-coerced"),
        pytest.param({"request_id": ""}, "request_id", id="request-id-empty"),
        pytest.param({"action": "pause"}, "action", id="unknown-action"),
    ],
)
def test_alarm_action_keeps_mains_constraints(changes, field):
    message = dict(MAIN_CONTROL_EXAMPLES["alarm_action"], **changes)

    with pytest.raises(ProtocolError) as excinfo:
        parse_client_message(json.dumps(message))

    assert excinfo.value.field == field


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        pytest.param({"outcome": "snoozed"}, "snoozed_until", id="snoozed-without-until"),
        pytest.param(
            {"outcome": "stopped", "snoozed_until": "2026-01-05T07:08:00Z"},
            "snoozed_until",
            id="until-with-stopped",
        ),
        pytest.param({"at": "2026-01-05T06:59:00"}, "at", id="naive-at"),
        pytest.param({"alarm_id": ""}, "alarm_id", id="empty-alarm-id"),
        pytest.param({"request_id": "q" * 65}, "request_id", id="request-id-65"),
    ],
)
def test_alarm_report_keeps_mains_constraints(changes, field):
    message = dict(MAIN_CONTROL_EXAMPLES["alarm_report"], **changes)

    with pytest.raises(ProtocolError) as excinfo:
        parse_client_message(json.dumps(message))

    assert excinfo.value.field == field


def test_alarm_report_snoozed_round_trips_and_omits_what_it_did_not_carry():
    message = {
        "type": "alarm_report",
        "alarm_id": "a1",
        "outcome": "snoozed",
        "at": "2026-01-05T06:59:00Z",
        "rang_at": "2026-01-05T06:30:00Z",
        "snoozed_until": "2026-01-05T07:08:00Z",
    }

    parsed = parse_client_message(json.dumps(message))

    assert json.loads(encode_message(parsed)) == message


def test_an_unknown_type_beside_mains_control_messages_still_names_type():
    with pytest.raises(ProtocolError) as excinfo:
        parse_client_message(json.dumps({"type": "alarm_snooze", "alarm_id": "a1"}))

    assert excinfo.value.field == "type"


def test_module_docstring_states_what_2_0_0_carries():
    doc = protocol.__doc__ or ""
    terms = ("2.0.0", "1.x control messages", "protocol 2's own", "server messages", "not modelled")

    assert [term for term in terms if term not in doc] == []
    assert PROTOCOL_VERSION == "2.0.0"
