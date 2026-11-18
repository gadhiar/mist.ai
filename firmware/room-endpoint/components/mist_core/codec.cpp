// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.

#include "codec.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace mist_core {

namespace {

constexpr size_t kPcmSampleBytes = 2;
constexpr int64_t kMsPerDay = 86400000;
constexpr int64_t kMaxIdLen = 64;

bool is_known(uint16_t type) {
  return type == 0x01 || type == 0x02 || type == 0x03 || type == 0x10 || type == 0x11;
}

Direction direction_of(MsgType type) {
  return (type == MsgType::MicAudio || type == MsgType::RefAudio) ? Direction::Up
                                                                  : Direction::Down;
}

bool carries_pcm(MsgType type) {
  return type == MsgType::AudioChunk || type == MsgType::MicAudio || type == MsgType::RefAudio;
}

FrameError check_frame(MsgType type, uint16_t turn_tag, uint32_t sample_rate,
                       size_t payload_len) {
  const Direction direction = direction_of(type);
  const uint32_t expected_rate =
      direction == Direction::Up ? kUplinkSampleRate : kDownlinkSampleRate;
  if (sample_rate != expected_rate) return FrameError::SampleRate;
  if (direction == Direction::Up && turn_tag != 0) return FrameError::TurnTag;
  if (direction == Direction::Down && turn_tag == 0) return FrameError::TurnTag;
  if (!carries_pcm(type)) return payload_len == 0 ? FrameError::None : FrameError::Payload;
  if (payload_len % kPcmSampleBytes != 0) return FrameError::Payload;
  if (payload_len > static_cast<size_t>(expected_rate) * kPcmSampleBytes) {
    return FrameError::Payload;
  }
  return FrameError::None;
}

void put_u16(uint8_t* p, uint16_t v) {
  p[0] = static_cast<uint8_t>(v & 0xFF);
  p[1] = static_cast<uint8_t>(v >> 8);
}

void put_u32(uint8_t* p, uint32_t v) {
  p[0] = static_cast<uint8_t>(v & 0xFF);
  p[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
  p[2] = static_cast<uint8_t>((v >> 16) & 0xFF);
  p[3] = static_cast<uint8_t>(v >> 24);
}

uint16_t get_u16(const uint8_t* p) {
  return static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8));
}

uint32_t get_u32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

void put_header(uint8_t* out, MsgType type, uint16_t turn_tag, uint32_t seq, uint32_t rate) {
  put_u32(out, kMagic);
  put_u16(out + 4, static_cast<uint16_t>(type));
  put_u16(out + 6, turn_tag);
  put_u32(out + 8, seq);
  put_u32(out + 12, rate);
}

// Days since 1970-01-01 of a civil date, and back (proleptic Gregorian; H. Hinnant's algorithms).
int64_t days_from_civil(int64_t y, unsigned m, unsigned d) {
  y -= m <= 2 ? 1 : 0;
  const int64_t era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = static_cast<unsigned>(y - era * 400);
  const unsigned doy = (153u * (m > 2 ? m - 3 : m + 9) + 2u) / 5u + d - 1u;
  const unsigned doe = yoe * 365u + yoe / 4u - yoe / 100u + doy;
  return era * 146097 + static_cast<int64_t>(doe) - 719468;
}

void civil_from_days(int64_t z, int64_t* y, unsigned* m, unsigned* d) {
  z += 719468;
  const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
  const unsigned doe = static_cast<unsigned>(z - era * 146097);
  const unsigned yoe = (doe - doe / 1460u + doe / 36524u - doe / 146096u) / 365u;
  const unsigned doy = doe - (365u * yoe + yoe / 4u - yoe / 100u);
  const unsigned mp = (5u * doy + 2u) / 153u;
  *d = doy - (153u * mp + 2u) / 5u + 1u;
  *m = mp < 10u ? mp + 3u : mp - 9u;
  *y = static_cast<int64_t>(yoe) + era * 400 + (*m <= 2u ? 1 : 0);
}

bool is_leap(int64_t y) { return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0; }

unsigned days_in_month(int64_t y, unsigned m) {
  static const unsigned kDays[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  return m == 2 && is_leap(y) ? 29u : kDays[m - 1];
}

// Reads exactly `n` decimal digits.
bool fixed_digits(const char* p, int n, unsigned* out) {
  unsigned value = 0;
  for (int i = 0; i < n; ++i) {
    if (p[i] < '0' || p[i] > '9') return false;
    value = value * 10u + static_cast<unsigned>(p[i] - '0');
  }
  *out = value;
  return true;
}

bool id_ok(const char* id) {
  if (id == nullptr) return false;
  const size_t len = std::strlen(id);
  return len >= 1 && static_cast<int64_t>(len) <= kMaxIdLen;
}

size_t finish(const json::Writer& writer) { return writer.ok() ? writer.size() : 0; }

bool write_time(json::Writer* writer, const char* key, int64_t epoch_ms) {
  char text[32];
  if (!format_iso8601_ms(epoch_ms, text, sizeof(text))) return false;
  writer->key(key).string(text);
  return true;
}

}  // namespace

const char* frame_error_field(FrameError error) {
  switch (error) {
    case FrameError::None: return "";
    case FrameError::Header: return "header";
    case FrameError::Magic: return "magic";
    case FrameError::MsgType: return "msg_type";
    case FrameError::SampleRate: return "sample_rate";
    case FrameError::TurnTag: return "turn_tag";
    case FrameError::Payload: return "payload";
    case FrameError::Capacity: return "capacity";
  }
  return "";
}

FrameError encode_frame(const Frame& frame, uint8_t* out, size_t cap, size_t* written) {
  if (!is_known(static_cast<uint16_t>(frame.type))) return FrameError::MsgType;
  const FrameError error =
      check_frame(frame.type, frame.turn_tag, frame.sample_rate, frame.payload_len);
  if (error != FrameError::None) return error;
  if (cap < kHeaderSize || cap - kHeaderSize < frame.payload_len) return FrameError::Capacity;
  put_header(out, frame.type, frame.turn_tag, frame.seq, frame.sample_rate);
  if (frame.payload_len > 0) std::memcpy(out + kHeaderSize, frame.payload, frame.payload_len);
  *written = kHeaderSize + frame.payload_len;
  return FrameError::None;
}

FrameError decode_frame(const uint8_t* data, size_t len, Direction direction, Frame* out) {
  if (len < kHeaderSize) return FrameError::Header;
  if (get_u32(data) != kMagic) return FrameError::Magic;
  const uint16_t raw_type = get_u16(data + 4);
  if (!is_known(raw_type)) return FrameError::MsgType;
  const MsgType type = static_cast<MsgType>(raw_type);
  if (direction_of(type) != direction) return FrameError::MsgType;
  const uint16_t turn_tag = get_u16(data + 6);
  const uint32_t sample_rate = get_u32(data + 12);
  const FrameError error = check_frame(type, turn_tag, sample_rate, len - kHeaderSize);
  if (error != FrameError::None) return error;
  out->type = type;
  out->turn_tag = turn_tag;
  out->seq = get_u32(data + 8);
  out->sample_rate = sample_rate;
  out->payload = data + kHeaderSize;
  out->payload_len = len - kHeaderSize;
  return FrameError::None;
}

size_t encode_mic_frame(uint32_t seq, const int16_t* samples, size_t count, uint8_t* out,
                        size_t cap) {
  const size_t payload_len = count * kPcmSampleBytes;
  if (check_frame(MsgType::MicAudio, 0, kUplinkSampleRate, payload_len) != FrameError::None) {
    return 0;
  }
  if (cap < kHeaderSize || cap - kHeaderSize < payload_len) return 0;
  put_header(out, MsgType::MicAudio, 0, seq, kUplinkSampleRate);
  uint8_t* p = out + kHeaderSize;
  for (size_t i = 0; i < count; ++i) {
    put_u16(p + i * kPcmSampleBytes, static_cast<uint16_t>(samples[i]));
  }
  return kHeaderSize + payload_len;
}

bool format_iso8601_ms(int64_t epoch_ms, char* out, size_t cap) {
  if (epoch_ms < 0 || cap < 25) return false;
  const int64_t days = epoch_ms / kMsPerDay;
  const int64_t in_day = epoch_ms % kMsPerDay;
  int64_t year = 0;
  unsigned month = 0;
  unsigned day = 0;
  civil_from_days(days, &year, &month, &day);
  if (year > 9999) return false;
  const int64_t seconds = in_day / 1000;
  std::snprintf(out, cap, "%04d-%02u-%02uT%02d:%02d:%02d.%03dZ", static_cast<int>(year), month,
                day, static_cast<int>(seconds / 3600), static_cast<int>((seconds / 60) % 60),
                static_cast<int>(seconds % 60), static_cast<int>(in_day % 1000));
  return true;
}

bool parse_iso8601_ms(const char* text, int64_t* epoch_ms) {
  // YYYY-MM-DDTHH:MM:SS, then an optional fraction, then Z or +HH:MM or -HH:MM.
  unsigned year = 0;
  unsigned month = 0;
  unsigned day = 0;
  unsigned hour = 0;
  unsigned minute = 0;
  unsigned second = 0;
  if (std::strlen(text) < 20) return false;
  if (!fixed_digits(text, 4, &year) || text[4] != '-' || !fixed_digits(text + 5, 2, &month) ||
      text[7] != '-' || !fixed_digits(text + 8, 2, &day) || (text[10] != 'T' && text[10] != 't') ||
      !fixed_digits(text + 11, 2, &hour) || text[13] != ':' ||
      !fixed_digits(text + 14, 2, &minute) || text[16] != ':' ||
      !fixed_digits(text + 17, 2, &second)) {
    return false;
  }
  if (month < 1 || month > 12 || day < 1 || day > days_in_month(year, month) || hour > 23 ||
      minute > 59 || second > 59) {
    return false;
  }
  const char* p = text + 19;
  int64_t millis = 0;
  if (*p == '.') {
    ++p;
    int digits = 0;
    while (*p >= '0' && *p <= '9') {
      if (digits < 3) millis = millis * 10 + (*p - '0');
      ++digits;
      ++p;
    }
    if (digits == 0) return false;
    for (int i = digits; i < 3; ++i) millis *= 10;
  }
  int64_t offset_ms = 0;
  if (*p == 'Z' || *p == 'z') {
    ++p;
  } else if (*p == '+' || *p == '-') {
    const int64_t sign = *p == '+' ? 1 : -1;
    unsigned off_hour = 0;
    unsigned off_minute = 0;
    if (!fixed_digits(p + 1, 2, &off_hour) || p[3] != ':' ||
        !fixed_digits(p + 4, 2, &off_minute) || off_hour > 23 || off_minute > 59) {
      return false;
    }
    offset_ms = sign * (static_cast<int64_t>(off_hour) * 3600 + off_minute * 60) * 1000;
    p += 6;
  } else {
    return false;
  }
  if (*p != '\0') return false;
  const int64_t days = days_from_civil(year, month, day);
  *epoch_ms = days * kMsPerDay + (static_cast<int64_t>(hour) * 3600 + minute * 60 + second) * 1000 +
              millis - offset_ms;
  return true;
}

size_t build_voice_start(char* buf, size_t cap, uint32_t req, bool muted) {
  if (req == 0) return 0;
  json::Writer w(buf, cap);
  w.begin_object()
      .key("type").string("voice_start")
      .key("req").integer(req)
      .key("mode").string("wake")
      .key("sample_rate").integer(kUplinkSampleRate)
      .key("frame_ms").integer(kFrameMs)
      .key("aec").string("device")
      .key("playback").boolean(true)
      .key("endpoint").string("room")
      .key("spotter").string("endpoint")
      .key("reference").boolean(false)
      .key("muted").boolean(muted)
      .end_object();
  return finish(w);
}

size_t build_wake(char* buf, size_t cap, const char* keyword, float score, uint32_t backlog_ms) {
  if (keyword == nullptr || backlog_ms > kMaxWakeBacklogMs || !std::isfinite(score)) return 0;
  json::Writer w(buf, cap);
  w.begin_object()
      .key("type").string("wake")
      .key("keyword").string(keyword)
      .key("score").number_float(score)
      .key("backlog_ms").integer(backlog_ms)
      .end_object();
  return finish(w);
}

size_t build_barge_in(char* buf, size_t cap, const char* turn_id, const char* keyword) {
  if (turn_id == nullptr || keyword == nullptr) return 0;
  json::Writer w(buf, cap);
  w.begin_object()
      .key("type").string("barge_in")
      .key("turn_id").string(turn_id)
      .key("keyword").string(keyword)
      .end_object();
  return finish(w);
}

size_t build_mute(char* buf, size_t cap, bool active) {
  json::Writer w(buf, cap);
  w.begin_object().key("type").string("mute").key("active").boolean(active).end_object();
  return finish(w);
}

size_t build_interrupt(char* buf, size_t cap) {
  json::Writer w(buf, cap);
  w.begin_object().key("type").string("interrupt").end_object();
  return finish(w);
}

namespace {

size_t build_playback(char* buf, size_t cap, const char* type, const char* turn_id,
                      uint32_t played_ms) {
  if (turn_id == nullptr) return 0;
  json::Writer w(buf, cap);
  w.begin_object()
      .key("type").string(type)
      .key("turn_id").string(turn_id)
      .key("played_ms").integer(played_ms)
      .end_object();
  return finish(w);
}

}  // namespace

size_t build_playback_progress(char* buf, size_t cap, const char* turn_id, uint32_t played_ms) {
  return build_playback(buf, cap, "playback_progress", turn_id, played_ms);
}

size_t build_playback_done(char* buf, size_t cap, const char* turn_id, uint32_t played_ms) {
  return build_playback(buf, cap, "playback_done", turn_id, played_ms);
}

size_t build_alarm_action(char* buf, size_t cap, const char* alarm_id, AlarmActionKind action,
                          uint32_t minutes, const char* request_id) {
  if (alarm_id == nullptr || !id_ok(request_id) || minutes > 60) return 0;
  const char* name = action == AlarmActionKind::Stop     ? "stop"
                     : action == AlarmActionKind::Snooze ? "snooze"
                                                         : "dismiss";
  json::Writer w(buf, cap);
  w.begin_object()
      .key("type").string("alarm_action")
      .key("alarm_id").string(alarm_id)
      .key("action").string(name);
  if (minutes > 0) w.key("minutes").integer(minutes);
  w.key("request_id").string(request_id).end_object();
  return finish(w);
}

size_t build_alarm_report(char* buf, size_t cap, const AlarmReport& report) {
  const bool snoozed = report.outcome == ReportOutcome::Snoozed;
  if (!id_ok(report.alarm_id) || report.has_snoozed_until != snoozed ||
      report.rang_at_ms > report.at_ms ||
      (report.outcome == ReportOutcome::Rang && report.rang_at_ms != report.at_ms)) {
    return 0;
  }
  if (report.request_id[0] != '\0' && !id_ok(report.request_id)) return 0;
  const char* outcome = report.outcome == ReportOutcome::Rang      ? "rang"
                        : report.outcome == ReportOutcome::Stopped ? "stopped"
                                                                   : "snoozed";
  json::Writer w(buf, cap);
  w.begin_object()
      .key("type").string("alarm_report")
      .key("alarm_id").string(report.alarm_id)
      .key("outcome").string(outcome);
  if (!write_time(&w, "at", report.at_ms) || !write_time(&w, "rang_at", report.rang_at_ms)) {
    return 0;
  }
  if (snoozed && !write_time(&w, "snoozed_until", report.snoozed_until_ms)) return 0;
  if (report.request_id[0] != '\0') w.key("request_id").string(report.request_id);
  w.end_object();
  return finish(w);
}

size_t build_device_stats(char* buf, size_t cap, const DeviceStats& stats) {
  json::Writer w(buf, cap);
  w.begin_object()
      .key("type").string("device_stats")
      .key("heap_free_min").integer(stats.heap_free_min)
      .key("psram_free").integer(stats.psram_free)
      .key("mic_overruns").integer(stats.mic_overruns)
      .key("ring_overruns").integer(stats.ring_overruns)
      .key("playback_underruns").integer(stats.playback_underruns)
      .key("uptime_s").integer(stats.uptime_s)
      .key("reconnects").integer(stats.reconnects)
      .end_object();
  return finish(w);
}

bool read_id(json::Value value, char* out, size_t cap) {
  return value.is_string() && value.get_string(out, cap);
}

bool read_time(json::Value value, int64_t* epoch_ms) {
  char text[48];
  return read_id(value, text, sizeof(text)) && parse_iso8601_ms(text, epoch_ms);
}

namespace {

// An optional id: absent or null reads as empty.
bool read_optional_id(json::Value value, char* out, size_t cap) {
  if (!value.ok() || value.is_null()) {
    out[0] = '\0';
    return true;
  }
  return read_id(value, out, cap);
}

bool read_uint(json::Value value, uint64_t max, uint32_t* out) {
  int64_t raw = 0;
  if (!value.get_int(&raw) || raw < 0 || static_cast<uint64_t>(raw) > max) return false;
  *out = static_cast<uint32_t>(raw);
  return true;
}

bool read_state(json::Value value, MistState* out) {
  static const struct {
    const char* name;
    MistState state;
  } kStates[] = {{"idle", MistState::Idle},
                 {"attend", MistState::Attend},
                 {"listen", MistState::Listen},
                 {"think", MistState::Think},
                 {"speak", MistState::Speak}};
  for (const auto& entry : kStates) {
    if (value.equals(entry.name)) {
      *out = entry.state;
      return true;
    }
  }
  return false;
}

DecisionKind decision_kind(json::Value value) {
  static const struct {
    const char* name;
    DecisionKind kind;
  } kKinds[] = {{"endpoint", DecisionKind::Endpoint},
                {"hold", DecisionKind::Hold},
                {"release", DecisionKind::Release},
                {"supersede", DecisionKind::Supersede},
                {"barge_in", DecisionKind::BargeIn},
                {"barge_in_ignored", DecisionKind::BargeInIgnored},
                {"dropped", DecisionKind::Dropped},
                {"degraded", DecisionKind::Degraded},
                {"overload", DecisionKind::Overload},
                {"wake_lost", DecisionKind::WakeLost},
                {"wake_rejected", DecisionKind::WakeRejected},
                {"placement", DecisionKind::Placement}};
  for (const auto& entry : kKinds) {
    if (value.equals(entry.name)) return entry.kind;
  }
  return DecisionKind::Other;
}

bool read_voice_state(json::Value root, VoiceStateInfo* out) {
  const json::Value cause = root.get("cause");
  if (cause.equals("answer")) {
    out->answer = true;
  } else if (cause.equals("broadcast")) {
    out->answer = false;
  } else {
    return false;
  }
  const json::Value req = root.get("req");
  out->has_req = !req.is_null();
  if (out->has_req && !read_uint(req, 0xFFFFFFFFu, &out->req)) return false;
  const json::Value refused = root.get("refused");
  if (!refused.is_null() && !refused.is_string()) return false;
  out->refused = refused.is_string();
  const json::Value playback = root.get("playback");
  return root.get("active").get_bool(&out->active) && root.get("muted").get_bool(&out->muted) &&
         read_uint(playback.get("client_prebuffer_ms"), 0xFFFFFFFFu,
                   &out->playback.prebuffer_ms) &&
         read_uint(playback.get("fade_ms"), 0xFFFFFFFFu, &out->playback.fade_ms) &&
         read_uint(playback.get("progress_interval_ms"), 0xFFFFFFFFu,
                   &out->playback.progress_interval_ms);
}

bool read_stream_start(json::Value root, StreamStartInfo* out) {
  if (!read_id(root.get("turn_id"), out->turn_id, kIdCap)) return false;
  const json::Value tag = root.get("audio_tag");
  out->has_audio_tag = tag.ok() && !tag.is_null();
  if (!out->has_audio_tag) return true;
  uint32_t value = 0;
  if (!read_uint(tag, kMaxTurnTag, &value) || value == 0) return false;
  out->audio_tag = static_cast<uint16_t>(value);
  return true;
}

bool read_playback_stop(json::Value root, PlaybackStopInfo* out) {
  uint32_t tag = 0;
  if (!read_id(root.get("turn_id"), out->turn_id, kIdCap) ||
      !read_uint(root.get("audio_tag"), kMaxTurnTag, &tag) || tag == 0 ||
      !read_uint(root.get("fade_ms"), 0xFFFFFFFFu, &out->fade_ms)) {
    return false;
  }
  out->audio_tag = static_cast<uint16_t>(tag);
  return true;
}

bool read_playback_duck(json::Value root, PlaybackDuckInfo* out) {
  double db = 0;
  if (!read_id(root.get("turn_id"), out->turn_id, kIdCap) || !root.get("db").get_double(&db)) {
    return false;
  }
  out->db = static_cast<float>(db);
  return true;
}

bool read_action_result(json::Value root, ActionResultInfo* out) {
  return read_id(root.get("alarm_id"), out->alarm_id, kIdCap) &&
         read_id(root.get("action"), out->action, sizeof(out->action)) &&
         read_optional_id(root.get("request_id"), out->request_id, kIdCap) &&
         read_id(root.get("outcome"), out->outcome, sizeof(out->outcome));
}

}  // namespace

ReadError read_inbound(json::Doc* doc, const char* text, size_t len, Inbound* out) {
  *out = Inbound{};
  if (doc->parse(text, len) != json::ParseError::None) return ReadError::Json;
  const json::Value root = doc->root();
  if (!root.is_object()) return ReadError::NotObject;
  const json::Value type = root.get("type");
  if (!type.is_string()) return ReadError::NoType;
  bool ok = true;
  if (type.equals("voice_state")) {
    out->type = InboundType::VoiceState;
    ok = read_voice_state(root, &out->voice_state);
  } else if (type.equals("state_cycle")) {
    out->type = InboundType::StateCycle;
    ok = read_state(root.get("state"), &out->state_cycle.state) &&
         read_optional_id(root.get("turn_id"), out->state_cycle.turn_id, kIdCap);
  } else if (type.equals("voice_decision")) {
    out->type = InboundType::VoiceDecision;
    const json::Value kind = root.get("kind");
    out->decision.kind = decision_kind(kind);
    ok = kind.is_string() && read_optional_id(root.get("turn_id"), out->decision.turn_id, kIdCap);
  } else if (type.equals("stream_start")) {
    out->type = InboundType::StreamStart;
    ok = read_stream_start(root, &out->stream_start);
  } else if (type.equals("playback_stop")) {
    out->type = InboundType::PlaybackStop;
    ok = read_playback_stop(root, &out->playback_stop);
  } else if (type.equals("playback_duck")) {
    out->type = InboundType::PlaybackDuck;
    ok = read_playback_duck(root, &out->playback_duck);
  } else if (type.equals("session_started")) {
    out->type = InboundType::SessionStarted;
    ok = read_id(root.get("protocol_version"), out->session_started.protocol_version,
                 sizeof(out->session_started.protocol_version));
  } else if (type.equals("alarm_state")) {
    out->type = InboundType::AlarmState;
  } else if (type.equals("alarm_fired")) {
    out->type = InboundType::AlarmFired;
  } else if (type.equals("alarm_action_result")) {
    out->type = InboundType::AlarmActionResult;
    ok = read_action_result(root, &out->action_result);
  } else if (type.equals("error")) {
    out->type = InboundType::Error;
    ok = read_id(root.get("kind"), out->error.kind, sizeof(out->error.kind)) &&
         root.get("retriable").get_bool(&out->error.retriable);
  }
  return ok ? ReadError::None : ReadError::Field;
}

}  // namespace mist_core
