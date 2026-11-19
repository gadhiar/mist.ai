// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.
//
// Host tests for the codec and `json_lite`: parity with the protocol 2 vectors file
// (`MIST_PROTOCOL_VECTORS` names it), the JSON builders and readers, and the ISO 8601 times.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "check.h"
#include "codec.h"
#include "json_lite.h"

using mist_core::Direction;
using mist_core::Frame;
using mist_core::FrameError;
using mist_core::Inbound;
using mist_core::InboundType;
using mist_core::MsgType;
using mist_core::ReadError;
namespace json = mist_core::json;

namespace {

constexpr uint32_t kPoolSize = 32768;
json::Token g_pool[kPoolSize];
char g_text[1 << 20];
size_t g_text_len = 0;

bool load_vectors(json::Doc* doc) {
  const char* path = std::getenv("MIST_PROTOCOL_VECTORS");
  if (path == nullptr) {
    std::printf("MIST_PROTOCOL_VECTORS is not set\n");
    return false;
  }
  std::FILE* file = std::fopen(path, "rb");
  if (file == nullptr) {
    std::printf("cannot open %s\n", path);
    return false;
  }
  g_text_len = std::fread(g_text, 1, sizeof(g_text) - 1, file);
  std::fclose(file);
  g_text[g_text_len] = '\0';
  return doc->parse(g_text, g_text_len) == json::ParseError::None;
}

int hex_digit(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

// Decodes the hex string `value` into `out`; returns the byte count, or -1 on bad hex.
long from_hex(json::Value value, uint8_t* out, size_t cap) {
  static char text[8192];
  size_t len = 0;
  if (!value.get_string(text, sizeof(text), &len) || len % 2 != 0 || len / 2 > cap) {
    return -1;
  }
  for (size_t i = 0; i < len; i += 2) {
    const int hi = hex_digit(text[i]);
    const int lo = hex_digit(text[i + 1]);
    if (hi < 0 || lo < 0) return -1;
    out[i / 2] = static_cast<uint8_t>(hi * 16 + lo);
  }
  return static_cast<long>(len / 2);
}

Direction direction_of(json::Value vector) {
  return vector.get("direction").equals("up") ? Direction::Up : Direction::Down;
}

int64_t int_field(json::Value object, const char* key) {
  int64_t value = -1;
  object.get(key).get_int(&value);
  return value;
}

// Structural JSON equality: object members in any order, numbers by value.
bool json_equal(json::Value a, json::Value b) {
  if (a.type() != b.type()) return false;
  switch (a.type()) {
    case json::Type::Null:
    case json::Type::True:
    case json::Type::False:
      return true;
    case json::Type::Number: {
      double x = 0;
      double y = 0;
      return a.get_double(&x) && b.get_double(&y) && x == y;
    }
    case json::Type::String: {
      char x[512];
      char y[512];
      return a.get_string(x, sizeof(x)) && b.get_string(y, sizeof(y)) && std::strcmp(x, y) == 0;
    }
    case json::Type::Array: {
      if (a.size() != b.size()) return false;
      for (uint32_t i = 0; i < a.size(); ++i) {
        if (!json_equal(a.at(i), b.at(i))) return false;
      }
      return true;
    }
    case json::Type::Object: {
      if (a.size() != b.size()) return false;
      for (uint32_t i = 0; i < a.size(); ++i) {
        char key[128];
        if (!a.key_at(i).get_string(key, sizeof(key))) return false;
        if (!json_equal(a.at(i), b.get(key))) return false;
      }
      return true;
    }
  }
  return false;
}

json::Value find_vector(json::Value list, const char* name) {
  for (uint32_t i = 0; i < list.size(); ++i) {
    if (list.at(i).get("name").equals(name)) return list.at(i);
  }
  return json::Value();
}

// Parses `built` and compares it with the `message` of the named client vector.
bool matches_client_vector(json::Value root, const char* name, const char* built) {
  static json::Token pool[512];
  json::Doc doc(pool, 512);
  if (doc.parse(built, std::strlen(built)) != json::ParseError::None) {
    std::printf("built JSON does not parse: %s\n", built);
    return false;
  }
  json::Value vector = find_vector(root.get("control").get("client_to_server"), name);
  if (!vector.ok()) {
    std::printf("no client vector %s\n", name);
    return false;
  }
  if (!json_equal(doc.root(), vector.get("message"))) {
    std::printf("built %s differs from the vector\n", built);
    return false;
  }
  return true;
}

}  // namespace

// -- Binary frames: parity with vectors.json -------------------------------------------------

MIST_TEST(every_binary_vector_decodes_to_its_fields) {
  json::Doc doc(g_pool, kPoolSize);
  CHECK(load_vectors(&doc));
  json::Value binary = doc.root().get("binary");
  CHECK(binary.size() >= 6);
  for (uint32_t i = 0; i < binary.size(); ++i) {
    json::Value vector = binary.at(i);
    static uint8_t bytes[4096];
    static uint8_t payload[4096];
    const long len = from_hex(vector.get("frame_hex"), bytes, sizeof(bytes));
    const long payload_len = from_hex(vector.get("payload_hex"), payload, sizeof(payload));
    CHECK(len >= 16);
    CHECK(payload_len >= 0);
    Frame frame{};
    const FrameError error =
        mist_core::decode_frame(bytes, static_cast<size_t>(len), direction_of(vector), &frame);
    CHECK_EQ(static_cast<int>(error), static_cast<int>(FrameError::None));
    CHECK_EQ(static_cast<int>(frame.type), int_field(vector, "msg_type"));
    CHECK_EQ(frame.turn_tag, int_field(vector, "turn_tag"));
    CHECK_EQ(frame.seq, int_field(vector, "seq"));
    CHECK_EQ(frame.sample_rate, int_field(vector, "sample_rate"));
    CHECK_EQ(frame.payload_len, payload_len);
    CHECK(payload_len == 0 || std::memcmp(frame.payload, payload, frame.payload_len) == 0);
  }
}

MIST_TEST(every_binary_vector_encodes_byte_for_byte) {
  json::Doc doc(g_pool, kPoolSize);
  CHECK(load_vectors(&doc));
  json::Value binary = doc.root().get("binary");
  for (uint32_t i = 0; i < binary.size(); ++i) {
    json::Value vector = binary.at(i);
    static uint8_t expected[4096];
    static uint8_t payload[4096];
    static uint8_t out[4096];
    const long len = from_hex(vector.get("frame_hex"), expected, sizeof(expected));
    const long payload_len = from_hex(vector.get("payload_hex"), payload, sizeof(payload));
    CHECK(len >= 16);
    Frame frame{};
    frame.type = static_cast<MsgType>(int_field(vector, "msg_type"));
    frame.turn_tag = static_cast<uint16_t>(int_field(vector, "turn_tag"));
    frame.seq = static_cast<uint32_t>(int_field(vector, "seq"));
    frame.sample_rate = static_cast<uint32_t>(int_field(vector, "sample_rate"));
    frame.payload = payload;
    frame.payload_len = static_cast<size_t>(payload_len);
    size_t written = 0;
    CHECK_EQ(static_cast<int>(mist_core::encode_frame(frame, out, sizeof(out), &written)),
             static_cast<int>(FrameError::None));
    CHECK_EQ(written, len);
    CHECK(std::memcmp(out, expected, written) == 0);
  }
}

MIST_TEST(every_invalid_binary_vector_is_refused_naming_its_field) {
  json::Doc doc(g_pool, kPoolSize);
  CHECK(load_vectors(&doc));
  json::Value invalid = doc.root().get("binary_invalid");
  CHECK(invalid.size() >= 10);
  for (uint32_t i = 0; i < invalid.size(); ++i) {
    json::Value vector = invalid.at(i);
    static uint8_t bytes[4096];
    const long len = from_hex(vector.get("frame_hex"), bytes, sizeof(bytes));
    CHECK(len >= 0);
    Frame frame{};
    const FrameError error =
        mist_core::decode_frame(bytes, static_cast<size_t>(len), direction_of(vector), &frame);
    CHECK(error != FrameError::None);
    char field[32];
    CHECK(vector.get("field").get_string(field, sizeof(field)));
    CHECK_STREQ(mist_core::frame_error_field(error), field);
  }
}

MIST_TEST(mic_frame_encoder_packs_little_endian_pcm_after_the_header) {
  const int16_t samples[2] = {100, -100};
  uint8_t out[64];

  const size_t written = mist_core::encode_mic_frame(7, samples, 2, out, sizeof(out));

  const uint8_t expected[] = {0x54, 0x53, 0x49, 0x4d, 0x10, 0x00, 0x00, 0x00, 0x07, 0x00,
                              0x00, 0x00, 0x80, 0x3e, 0x00, 0x00, 0x64, 0x00, 0x9c, 0xff};
  CHECK_EQ(written, sizeof(expected));
  CHECK(std::memcmp(out, expected, sizeof(expected)) == 0);
}

MIST_TEST(mic_frame_encoder_refuses_a_buffer_too_small_and_over_one_second) {
  static int16_t samples[16001];
  static uint8_t out[40000];

  CHECK_EQ(mist_core::encode_mic_frame(0, samples, 320, out, 16 + 639), 0);
  CHECK_EQ(mist_core::encode_mic_frame(0, samples, 16001, out, sizeof(out)), 0);
  CHECK_EQ(mist_core::encode_mic_frame(0, samples, 16000, out, sizeof(out)), 16 + 32000);
}

MIST_TEST(encoder_refuses_what_the_decoder_would_refuse) {
  const uint8_t two[2] = {0, 0};
  uint8_t out[64];
  size_t written = 0;
  Frame frame{};
  frame.type = MsgType::MicAudio;
  frame.sample_rate = 16000;
  frame.turn_tag = 1;
  CHECK(mist_core::encode_frame(frame, out, sizeof(out), &written) == FrameError::TurnTag);
  frame.turn_tag = 0;
  frame.sample_rate = 24000;
  CHECK(mist_core::encode_frame(frame, out, sizeof(out), &written) == FrameError::SampleRate);
  frame = Frame{};
  frame.type = MsgType::AudioComplete;
  frame.turn_tag = 3;
  frame.sample_rate = 24000;
  frame.payload = two;
  frame.payload_len = 2;
  CHECK(mist_core::encode_frame(frame, out, sizeof(out), &written) == FrameError::Payload);
  frame.payload_len = 0;
  frame.turn_tag = 0;
  CHECK(mist_core::encode_frame(frame, out, sizeof(out), &written) == FrameError::TurnTag);
  frame.type = static_cast<MsgType>(0x04);
  frame.turn_tag = 3;
  CHECK(mist_core::encode_frame(frame, out, sizeof(out), &written) == FrameError::MsgType);
  frame.type = MsgType::AudioComplete;
  CHECK(mist_core::encode_frame(frame, out, 15, &written) == FrameError::Capacity);
}

MIST_TEST(decoder_refuses_a_downlink_payload_over_one_second) {
  static uint8_t bytes[16 + 48002];
  uint8_t header[16] = {0x54, 0x53, 0x49, 0x4d, 0x01, 0x00, 0x01, 0x00,
                        0x00, 0x00, 0x00, 0x00, 0xc0, 0x5d, 0x00, 0x00};
  std::memcpy(bytes, header, 16);
  Frame frame{};

  CHECK(mist_core::decode_frame(bytes, 16 + 48000, Direction::Down, &frame) == FrameError::None);
  CHECK(mist_core::decode_frame(bytes, 16 + 48002, Direction::Down, &frame) ==
        FrameError::Payload);
}

// -- Control messages: builders against the client vectors -----------------------------------

MIST_TEST(voice_start_matches_the_room_unit_vector) {
  json::Doc doc(g_pool, kPoolSize);
  CHECK(load_vectors(&doc));
  char buf[512];

  CHECK(mist_core::build_voice_start(buf, sizeof(buf), 1, false) > 0);

  CHECK(matches_client_vector(doc.root(), "voice_start_endpoint_spotter_room_unit", buf));
}

MIST_TEST(wake_matches_its_vectors) {
  json::Doc doc(g_pool, kPoolSize);
  CHECK(load_vectors(&doc));
  char buf[256];

  CHECK(mist_core::build_wake(buf, sizeof(buf), "hey mist", 0.87f, 1500) > 0);
  CHECK(matches_client_vector(doc.root(), "wake", buf));
  CHECK(mist_core::build_wake(buf, sizeof(buf), "mist", 0.74f, 0) > 0);
  CHECK(matches_client_vector(doc.root(), "wake_backlog_zero_parses_and_the_server_rejects_it_no_backlog", buf));
}

MIST_TEST(wake_builder_refuses_values_outside_the_wire_bounds) {
  char buf[256];

  CHECK_EQ(mist_core::build_wake(buf, sizeof(buf), "mist", 0.5f, 10001), 0);
  CHECK(mist_core::build_wake(buf, sizeof(buf), "mist", 0.5f, 10000) > 0);
  CHECK_EQ(mist_core::build_wake(buf, sizeof(buf), "mist", NAN, 100), 0);
  CHECK_EQ(mist_core::build_wake(buf, sizeof(buf), "mist", INFINITY, 100), 0);
}

MIST_TEST(wake_score_is_written_as_a_json_float) {
  char buf[256];

  CHECK(mist_core::build_wake(buf, sizeof(buf), "mist", 1.0f, 100) > 0);

  CHECK(std::strstr(buf, "\"score\":1.0") != nullptr);
}

MIST_TEST(short_client_messages_match_their_vectors) {
  json::Doc doc(g_pool, kPoolSize);
  CHECK(load_vectors(&doc));
  char buf[256];

  CHECK(mist_core::build_mute(buf, sizeof(buf), true) > 0);
  CHECK(matches_client_vector(doc.root(), "mute", buf));
  CHECK(mist_core::build_interrupt(buf, sizeof(buf)) > 0);
  CHECK(matches_client_vector(doc.root(), "interrupt", buf));
  CHECK(mist_core::build_barge_in(buf, sizeof(buf), "turn-1", "stop") > 0);
  CHECK(matches_client_vector(doc.root(), "barge_in", buf));
  CHECK(mist_core::build_playback_progress(buf, sizeof(buf), "turn-1", 480) > 0);
  CHECK(matches_client_vector(doc.root(), "playback_progress", buf));
  CHECK(mist_core::build_playback_done(buf, sizeof(buf), "turn-1", 2160) > 0);
  CHECK(matches_client_vector(doc.root(), "playback_done", buf));
}

MIST_TEST(builders_report_overflow_as_zero_and_never_write_past_the_buffer) {
  char buf[24];
  std::memset(buf, 'x', sizeof(buf));

  CHECK_EQ(mist_core::build_voice_start(buf, 20, 1, false), 0);
  CHECK_EQ(buf[20], 'x');
  CHECK_EQ(mist_core::build_interrupt(buf, sizeof(buf)), std::strlen("{\"type\":\"interrupt\"}"));
}

MIST_TEST(strings_are_escaped_in_built_messages) {
  char buf[256];

  CHECK(mist_core::build_barge_in(buf, sizeof(buf), "a\"b\\c\n\x01", "stop") > 0);

  CHECK(std::strstr(buf, "\"turn_id\":\"a\\\"b\\\\c\\n\\u0001\"") != nullptr);
}

MIST_TEST(alarm_action_omits_minutes_when_zero_and_bounds_its_fields) {
  char buf[256];

  CHECK(mist_core::build_alarm_action(buf, sizeof(buf), "occ-1", mist_core::AlarmActionKind::Stop,
                                      0, "r1") > 0);
  CHECK_STREQ(buf,
              "{\"type\":\"alarm_action\",\"alarm_id\":\"occ-1\",\"action\":\"stop\","
              "\"request_id\":\"r1\"}");
  CHECK(mist_core::build_alarm_action(buf, sizeof(buf), "occ-1",
                                      mist_core::AlarmActionKind::Snooze, 9, "r2") > 0);
  CHECK(std::strstr(buf, "\"minutes\":9") != nullptr);
  CHECK_EQ(mist_core::build_alarm_action(buf, sizeof(buf), "occ-1",
                                         mist_core::AlarmActionKind::Snooze, 61, "r2"),
           0);
  CHECK_EQ(mist_core::build_alarm_action(buf, sizeof(buf), "occ-1",
                                         mist_core::AlarmActionKind::Stop, 0, ""),
           0);
}

MIST_TEST(alarm_report_follows_the_owner_rules_for_its_times) {
  char buf[512];
  mist_core::AlarmReport report{};
  std::strcpy(report.alarm_id, "occ-1");
  report.outcome = mist_core::ReportOutcome::Rang;
  report.at_ms = 1794898800000;
  report.rang_at_ms = 1794898800000;

  CHECK(mist_core::build_alarm_report(buf, sizeof(buf), report) > 0);
  CHECK_STREQ(buf,
              "{\"type\":\"alarm_report\",\"alarm_id\":\"occ-1\",\"outcome\":\"rang\","
              "\"at\":\"2026-11-17T07:00:00.000Z\",\"rang_at\":\"2026-11-17T07:00:00.000Z\"}");
  report.at_ms += 1;
  CHECK_EQ(mist_core::build_alarm_report(buf, sizeof(buf), report), 0);
  report.outcome = mist_core::ReportOutcome::Stopped;
  CHECK(mist_core::build_alarm_report(buf, sizeof(buf), report) > 0);
  report.rang_at_ms = report.at_ms + 1;
  CHECK_EQ(mist_core::build_alarm_report(buf, sizeof(buf), report), 0);
  report.rang_at_ms = report.at_ms;
  report.outcome = mist_core::ReportOutcome::Snoozed;
  CHECK_EQ(mist_core::build_alarm_report(buf, sizeof(buf), report), 0);
  report.has_snoozed_until = true;
  report.snoozed_until_ms = report.at_ms + 540000;
  std::strcpy(report.request_id, "r9");
  CHECK(mist_core::build_alarm_report(buf, sizeof(buf), report) > 0);
  CHECK(std::strstr(buf, "\"snoozed_until\":\"2026-11-17T07:09:00.001Z\"") != nullptr);
  CHECK(std::strstr(buf, "\"request_id\":\"r9\"") != nullptr);
  report.outcome = mist_core::ReportOutcome::Stopped;
  CHECK_EQ(mist_core::build_alarm_report(buf, sizeof(buf), report), 0);
}

// -- Control messages: readers against the server vectors ------------------------------------

MIST_TEST(every_server_vector_reads_without_error) {
  json::Doc doc(g_pool, kPoolSize);
  CHECK(load_vectors(&doc));
  json::Value list = doc.root().get("control").get("server_to_client");
  CHECK(list.size() >= 30);
  static json::Token pool[4096];
  static char text[16384];
  for (uint32_t i = 0; i < list.size(); ++i) {
    json::Value message = list.at(i).get("message");
    const uint32_t start = doc.token(message.index()).start;
    const uint32_t end = doc.token(message.index()).end;
    std::memcpy(text, g_text + start, end - start);
    text[end - start] = '\0';
    json::Doc inbound_doc(pool, 4096);
    Inbound inbound{};

    const ReadError error = mist_core::read_inbound(&inbound_doc, text, end - start, &inbound);

    if (error != ReadError::None) {
      char name[128];
      list.at(i).get("name").get_string(name, sizeof(name));
      std::printf("vector %s: read error %d\n", name, static_cast<int>(error));
    }
    CHECK(error == ReadError::None);
    const bool acted_on = message.get("type").equals("voice_state") ||
                          message.get("type").equals("state_cycle") ||
                          message.get("type").equals("voice_decision") ||
                          message.get("type").equals("stream_start") ||
                          message.get("type").equals("playback_stop") ||
                          message.get("type").equals("playback_duck") ||
                          message.get("type").equals("session_started") ||
                          message.get("type").equals("error");
    CHECK_EQ(inbound.type != InboundType::Unknown, acted_on);
  }
}

MIST_TEST(server_vector_fields_reach_the_reader) {
  json::Doc doc(g_pool, kPoolSize);
  CHECK(load_vectors(&doc));
  json::Value list = doc.root().get("control").get("server_to_client");
  static json::Token pool[4096];
  // Each vector is read again from its own text, so the expectations name the vector values.
  static char text[16384];
  auto read = [&](const char* name, Inbound* out) -> bool {
    json::Value message = find_vector(list, name).get("message");
    if (!message.ok()) return false;
    const json::Token& token = doc.token(message.index());
    std::memcpy(text, g_text + token.start, token.end - token.start);
    text[token.end - token.start] = '\0';
    json::Doc inbound_doc(pool, 4096);
    return mist_core::read_inbound(&inbound_doc, text, token.end - token.start, out) ==
           ReadError::None;
  };
  Inbound in{};

  CHECK(read("voice_state", &in));
  CHECK(in.type == InboundType::VoiceState);
  CHECK(in.voice_state.answer);
  CHECK(in.voice_state.has_req);
  CHECK_EQ(in.voice_state.req, 1);
  CHECK(in.voice_state.active);
  CHECK(!in.voice_state.muted);
  CHECK(!in.voice_state.refused);
  CHECK_EQ(in.voice_state.playback.prebuffer_ms, 80);
  CHECK_EQ(in.voice_state.playback.fade_ms, 30);
  CHECK_EQ(in.voice_state.playback.progress_interval_ms, 240);
  CHECK(read("voice_state_broadcast_before_voice_start", &in));
  CHECK(!in.voice_state.answer);
  CHECK(!in.voice_state.has_req);
  CHECK(!in.voice_state.active);
  CHECK(read("voice_state_answer_superseded", &in));
  CHECK(in.voice_state.refused);
  CHECK(read("voice_state_broadcast_muted", &in));
  CHECK(in.voice_state.muted);

  CHECK(read("state_cycle", &in));
  CHECK(in.type == InboundType::StateCycle);
  CHECK(in.state_cycle.state == mist_core::MistState::Speak);
  CHECK_STREQ(in.state_cycle.turn_id, "turn-1");
  CHECK(read("state_cycle_listen", &in));
  CHECK(in.state_cycle.state == mist_core::MistState::Listen);
  CHECK_STREQ(in.state_cycle.turn_id, "");
  CHECK(read("state_cycle_attend_sent_at_the_name_hit", &in));
  CHECK(in.state_cycle.state == mist_core::MistState::Attend);

  CHECK(read("voice_decision_wake_rejected", &in));
  CHECK(in.type == InboundType::VoiceDecision);
  CHECK(in.decision.kind == mist_core::DecisionKind::WakeRejected);
  CHECK(read("voice_decision_wake_lost", &in));
  CHECK(in.decision.kind == mist_core::DecisionKind::WakeLost);
  CHECK(read("voice_decision_dropped_muted", &in));
  CHECK(in.decision.kind == mist_core::DecisionKind::Dropped);
  CHECK(read("voice_decision_endpoint", &in));
  CHECK(in.decision.kind == mist_core::DecisionKind::Endpoint);
  CHECK(read("voice_decision_hold", &in));
  CHECK(in.decision.kind == mist_core::DecisionKind::Hold);
  CHECK(read("voice_decision_placement", &in));
  CHECK(in.decision.kind == mist_core::DecisionKind::Placement);
  CHECK_STREQ(in.decision.turn_id, "");

  CHECK(read("stream_start", &in));
  CHECK(in.type == InboundType::StreamStart);
  CHECK(in.stream_start.has_audio_tag);
  json::Value start = find_vector(list, "stream_start").get("message");
  CHECK_EQ(in.stream_start.audio_tag, int_field(start, "audio_tag"));
  CHECK(start.get("turn_id").equals(in.stream_start.turn_id));
  CHECK(read("stream_start_text", &in));
  CHECK(!in.stream_start.has_audio_tag);
  CHECK_STREQ(in.stream_start.turn_id, "turn-3");

  CHECK(read("playback_stop", &in));
  CHECK(in.type == InboundType::PlaybackStop);
  json::Value stop = find_vector(list, "playback_stop").get("message");
  CHECK_EQ(in.playback_stop.audio_tag, int_field(stop, "audio_tag"));
  CHECK_EQ(in.playback_stop.fade_ms, int_field(stop, "fade_ms"));
  CHECK(stop.get("turn_id").equals(in.playback_stop.turn_id));

  CHECK(read("playback_duck", &in));
  CHECK(in.type == InboundType::PlaybackDuck);
  CHECK(in.playback_duck.db == -12.0f);
  CHECK_STREQ(in.playback_duck.turn_id, "turn-1");

  CHECK(read("session_started", &in));
  CHECK(in.type == InboundType::SessionStarted);
  CHECK_STREQ(in.session_started.protocol_version, "2.0.0");

  CHECK(read("error", &in));
  CHECK(in.type == InboundType::Error);
}

MIST_TEST(reader_refuses_a_known_type_with_a_bad_field_and_ignores_unknown_types) {
  static json::Token pool[256];
  json::Doc doc(pool, 256);
  Inbound in{};
  const char* bad_tag = "{\"type\":\"stream_start\",\"turn_id\":\"t\",\"channel\":\"tts\","
                        "\"origin\":\"voice\",\"audio_tag\":0}";
  const char* bad_state = "{\"type\":\"state_cycle\",\"state\":\"dance\"}";
  const char* unknown = "{\"type\":\"stream_token\",\"text\":\"hi\"}";
  const char* not_json = "{\"type\":";
  const char* no_type = "{\"kind\":\"x\"}";

  CHECK(mist_core::read_inbound(&doc, bad_tag, std::strlen(bad_tag), &in) == ReadError::Field);
  CHECK(mist_core::read_inbound(&doc, bad_state, std::strlen(bad_state), &in) ==
        ReadError::Field);
  CHECK(mist_core::read_inbound(&doc, unknown, std::strlen(unknown), &in) == ReadError::None);
  CHECK(in.type == InboundType::Unknown);
  CHECK(mist_core::read_inbound(&doc, not_json, std::strlen(not_json), &in) == ReadError::Json);
  CHECK(mist_core::read_inbound(&doc, no_type, std::strlen(no_type), &in) == ReadError::NoType);
}

MIST_TEST(reader_takes_alarm_messages_and_their_action_results) {
  static json::Token pool[256];
  json::Doc doc(pool, 256);
  Inbound in{};
  const char* result = "{\"type\":\"alarm_action_result\",\"alarm_id\":\"occ-1\",\"action\":"
                       "\"report\",\"request_id\":\"r4\",\"outcome\":\"ok\",\"reason\":null,"
                       "\"detail\":\"applied\",\"at\":\"2026-11-17T07:00:00Z\"}";
  const char* state = "{\"type\":\"alarm_state\",\"rev\":1}";
  const char* fired = "{\"type\":\"alarm_fired\",\"alarm_id\":\"a\"}";

  CHECK(mist_core::read_inbound(&doc, result, std::strlen(result), &in) == ReadError::None);
  CHECK(in.type == InboundType::AlarmActionResult);
  CHECK_STREQ(in.action_result.alarm_id, "occ-1");
  CHECK_STREQ(in.action_result.action, "report");
  CHECK_STREQ(in.action_result.request_id, "r4");
  CHECK_STREQ(in.action_result.outcome, "ok");
  CHECK(mist_core::read_inbound(&doc, state, std::strlen(state), &in) == ReadError::None);
  CHECK(in.type == InboundType::AlarmState);
  CHECK(mist_core::read_inbound(&doc, fired, std::strlen(fired), &in) == ReadError::None);
  CHECK(in.type == InboundType::AlarmFired);
}

// -- json_lite -------------------------------------------------------------------------------

MIST_TEST(json_reader_refuses_malformed_text) {
  static json::Token pool[64];
  json::Doc doc(pool, 64);
  const char* bad[] = {"",         "{",        "{\"a\":}",  "[1,]",        "{\"a\":1,}",
                       "01",       "1.",       "-",         "\"\\x\"",     "\"\x01\"",
                       "tru",      "{} {}",    "[1 2]",     "{\"a\" 1}",   "\"\\ud800\"",
                       "{1:2}",    "nul",      ".5",        "1e",          "\"abc"};
  for (const char* text : bad) {
    if (doc.parse(text, std::strlen(text)) == json::ParseError::None) {
      std::printf("accepted %s\n", text);
    }
    CHECK(doc.parse(text, std::strlen(text)) != json::ParseError::None);
  }
}

MIST_TEST(json_reader_bounds_depth_and_tokens) {
  static json::Token pool[8];
  json::Doc small(pool, 8);
  static json::Token big_pool[256];
  json::Doc doc(big_pool, 256);
  char deep[80];
  for (int i = 0; i < 40; ++i) {
    deep[i] = '[';
    deep[79 - i] = ']';
  }

  CHECK(small.parse("[1,2,3,4,5,6,7,8]", 17) == json::ParseError::Capacity);
  CHECK(small.parse("[1,2,3,4,5,6,7]", 15) == json::ParseError::None);
  CHECK(doc.parse(deep, sizeof(deep)) == json::ParseError::Depth);
}

MIST_TEST(json_reader_unescapes_strings_and_utf16_pairs) {
  static json::Token pool[16];
  json::Doc doc(pool, 16);
  const char* text = "{\"s\":\"a\\n\\u00e9\\ud83d\\ude00\\/\"}";
  CHECK(doc.parse(text, std::strlen(text)) == json::ParseError::None);
  char out[32];
  size_t len = 0;

  CHECK(doc.root().get("s").get_string(out, sizeof(out), &len));

  CHECK_STREQ(out, "a\n\xc3\xa9\xf0\x9f\x98\x80/");
  // 1 + 1 + 2 (U+00E9) + 4 (U+1F600) + 1 bytes.
  CHECK_EQ(len, 9);
  CHECK(doc.root().get("s").equals("a\n\xc3\xa9\xf0\x9f\x98\x80/"));
  CHECK(!doc.root().get("s").equals("a\n"));
  // Nine bytes and the NUL need ten.
  CHECK(!doc.root().get("s").get_string(out, 9));
  CHECK(doc.root().get("s").get_string(out, 10));
}

MIST_TEST(json_reader_reads_numbers_strictly) {
  static json::Token pool[16];
  json::Doc doc(pool, 16);
  const char* text = "[9223372036854775807,9223372036854775808,1.5,-2,1e3,true]";
  CHECK(doc.parse(text, std::strlen(text)) == json::ParseError::None);
  int64_t value = 0;
  double real = 0;
  bool flag = false;

  CHECK(doc.root().at(0).get_int(&value));
  CHECK(value == INT64_MAX);
  CHECK(!doc.root().at(1).get_int(&value));
  CHECK(!doc.root().at(2).get_int(&value));
  CHECK(doc.root().at(2).get_double(&real));
  CHECK(real == 1.5);
  CHECK(doc.root().at(3).get_int(&value));
  CHECK_EQ(value, -2);
  CHECK(!doc.root().at(4).get_int(&value));
  CHECK(doc.root().at(4).get_double(&real));
  CHECK(real == 1000.0);
  CHECK(!doc.root().at(5).get_int(&value));
  CHECK(doc.root().at(5).get_bool(&flag));
  CHECK(flag);
  CHECK(!doc.root().at(6).ok());
}

MIST_TEST(json_writer_writes_shortest_round_trip_numbers) {
  char buf[128];
  json::Writer writer(buf, sizeof(buf));

  writer.begin_array();
  writer.number(0.1);
  writer.number(2.0);
  writer.number(-1e300);
  writer.number_float(0.87f);
  writer.integer(-5);
  writer.boolean(false);
  writer.null();
  writer.end_array();

  CHECK(writer.ok());
  CHECK_STREQ(buf, "[0.1,2.0,-1e+300,0.87,-5,false,null]");
}

// -- ISO 8601 times --------------------------------------------------------------------------

MIST_TEST(iso_times_format_in_utc_with_milliseconds) {
  char out[32];

  CHECK(mist_core::format_iso8601_ms(0, out, sizeof(out)));
  CHECK_STREQ(out, "1970-01-01T00:00:00.000Z");
  CHECK(mist_core::format_iso8601_ms(1709164800123, out, sizeof(out)));
  CHECK_STREQ(out, "2024-02-29T00:00:00.123Z");
  CHECK(!mist_core::format_iso8601_ms(-1, out, sizeof(out)));
  CHECK(!mist_core::format_iso8601_ms(0, out, 24));
}

MIST_TEST(iso_times_parse_offsets_and_fractions) {
  int64_t ms = 0;

  CHECK(mist_core::parse_iso8601_ms("2026-11-17T07:00:00Z", &ms));
  CHECK(ms == 1794898800000);
  CHECK(mist_core::parse_iso8601_ms("2026-11-17T12:30:00.123456+05:30", &ms));
  CHECK(ms == 1794898800123);
  CHECK(mist_core::parse_iso8601_ms("2026-11-17T02:00:00-05:00", &ms));
  CHECK(ms == 1794898800000);
  CHECK(mist_core::parse_iso8601_ms("2024-02-29T23:59:59.9Z", &ms));
  CHECK(ms == 1709251199900);
}

MIST_TEST(iso_times_refuse_naive_and_impossible_values) {
  int64_t ms = 0;
  const char* bad[] = {"2026-11-17T07:00:00",     "2026-02-29T00:00:00Z", "2026-13-01T00:00:00Z",
                       "2026-11-17T24:00:00Z",    "2026-11-17T07:60:00Z", "2026-11-17 07:00:00Z",
                       "2026-11-17T07:00:00+5:30", "2026-11-17T07:00:00.Z", "2026-11-17T07:00:00Zx"};
  for (const char* text : bad) {
    CHECK(!mist_core::parse_iso8601_ms(text, &ms));
  }
}
