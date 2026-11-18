// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.
//
// The room endpoint's half of MIST.AI realtime protocol 2.0.0 (`backend/realtime/protocol.py`,
// vectors in `tests/fixtures/realtime_protocol_v2/vectors.json`): the 16-byte `<IHHII>` binary
// frame header, the JSON builders for every message the device sends, and the readers for the
// messages it acts on.
//
// Encoding applies the same header and payload rules as decoding, as the backend's codec does, so
// the device cannot emit a frame the server would refuse. A builder returns the length written,
// or 0 when the buffer is too small or a value is outside what the server accepts. No function
// here allocates or throws.

#pragma once

#include <cstddef>
#include <cstdint>

#include "json_lite.h"

namespace mist_core {

constexpr uint32_t kMagic = 0x4D495354u;  // "MIST", little-endian on the wire
constexpr size_t kHeaderSize = 16;
constexpr uint32_t kUplinkSampleRate = 16000;
constexpr uint32_t kDownlinkSampleRate = 24000;
constexpr uint32_t kFrameMs = 20;
constexpr uint32_t kMaxWakeBacklogMs = 10000;
constexpr uint32_t kMaxTurnTag = 0xFFFF;
// Wire ids (turn, alarm, request) up to 64 bytes, plus the terminating NUL.
constexpr size_t kIdCap = 65;

enum class MsgType : uint16_t {
  AudioChunk = 0x01,
  AudioComplete = 0x02,
  InterruptFade = 0x03,
  MicAudio = 0x10,
  RefAudio = 0x11,
};

enum class Direction : uint8_t { Up, Down };

// Why a frame was refused; `frame_error_field` names it as the backend's `ProtocolError.field`.
enum class FrameError : uint8_t {
  None,
  Header,
  Magic,
  MsgType,
  SampleRate,
  TurnTag,
  Payload,
  Capacity,
};

const char* frame_error_field(FrameError error);

// One binary frame. `payload` points into the caller's buffer and is not copied.
struct Frame {
  MsgType type;
  uint16_t turn_tag;
  uint32_t seq;
  uint32_t sample_rate;
  const uint8_t* payload;
  size_t payload_len;
};

// Writes the frame's wire bytes; `written` is set only on success.
FrameError encode_frame(const Frame& frame, uint8_t* out, size_t cap, size_t* written);

// Decodes and validates one whole binary WebSocket message travelling in `direction`; a frame
// of the other direction is refused as `MsgType`, as the server refuses an injected downlink one.
FrameError decode_frame(const uint8_t* data, size_t len, Direction direction, Frame* out);

// Encodes one `0x10` microphone frame (16 kHz, `turn_tag` 0) from native PCM16 samples. Returns
// the byte count, or 0 when `out` is too small or the payload is over one second.
size_t encode_mic_frame(uint32_t seq, const int16_t* samples, size_t count, uint8_t* out,
                        size_t cap);

// -- Times -----------------------------------------------------------------------------------

// Writes `epoch_ms` as `YYYY-MM-DDTHH:MM:SS.mmmZ` (24 characters and a NUL); false for a time
// before 1970 or after 9999, or a buffer under 25 bytes.
bool format_iso8601_ms(int64_t epoch_ms, char* out, size_t cap);

// Reads an aware ISO 8601 time (`Z` or `+HH:MM`, any fraction, truncated to milliseconds) as
// milliseconds since the epoch. A naive time is refused: the owner sends every time aware.
bool parse_iso8601_ms(const char* text, int64_t* epoch_ms);

// -- Messages the device sends ---------------------------------------------------------------

// `voice_start` with the room unit's fixed fields (plan Design item 3): mode `wake`, 16 kHz,
// 20 ms frames, device AEC, playback on, endpoint `room`, spotter `endpoint`.
size_t build_voice_start(char* buf, size_t cap, uint32_t req, bool muted);
size_t build_wake(char* buf, size_t cap, const char* keyword, float score, uint32_t backlog_ms);
size_t build_barge_in(char* buf, size_t cap, const char* turn_id, const char* keyword);
size_t build_mute(char* buf, size_t cap, bool active);
size_t build_interrupt(char* buf, size_t cap);
size_t build_playback_progress(char* buf, size_t cap, const char* turn_id, uint32_t played_ms);
size_t build_playback_done(char* buf, size_t cap, const char* turn_id, uint32_t played_ms);

enum class AlarmActionKind : uint8_t { Stop, Snooze, Dismiss };

// `minutes` 0 leaves the field out (the owner's default snooze); otherwise 1 to 60.
// `request_id` is 1 to 64 bytes.
size_t build_alarm_action(char* buf, size_t cap, const char* alarm_id, AlarmActionKind action,
                          uint32_t minutes, const char* request_id);

enum class ReportOutcome : uint8_t { Rang, Stopped, Snoozed };

// An `alarm_report` (`backend/capabilities/events.py`, `AlarmReportMessage`). `rang_at` is when
// the ring started and `at` when its outcome happened: equal for `rang`, `rang_at` never later.
// `snoozed_until` comes with `snoozed` and with nothing else. An empty `request_id` is left out.
struct AlarmReport {
  char alarm_id[kIdCap];
  ReportOutcome outcome;
  int64_t at_ms;
  int64_t rang_at_ms;
  bool has_snoozed_until;
  int64_t snoozed_until_ms;
  char request_id[kIdCap];
};

size_t build_alarm_report(char* buf, size_t cap, const AlarmReport& report);

// The spike build's counters, sent to the spike listener only; no backend model reads it.
struct DeviceStats {
  uint32_t heap_free_min;
  uint32_t psram_free;
  uint32_t mic_overruns;
  uint32_t ring_overruns;
  uint32_t playback_underruns;
  uint32_t uptime_s;
  uint32_t reconnects;
};

size_t build_device_stats(char* buf, size_t cap, const DeviceStats& stats);

// -- Messages the device acts on -------------------------------------------------------------

// The device ignores every other type, so `Unknown` is not an error.
enum class InboundType : uint8_t {
  Unknown,
  SessionStarted,
  VoiceState,
  StateCycle,
  VoiceDecision,
  StreamStart,
  PlaybackStop,
  PlaybackDuck,
  AlarmState,
  AlarmFired,
  AlarmActionResult,
  Error,
};

enum class ReadError : uint8_t { None, Json, NotObject, NoType, Field };

enum class MistState : uint8_t { Idle, Attend, Listen, Think, Speak };

enum class DecisionKind : uint8_t {
  Other,
  Endpoint,
  Hold,
  Release,
  Supersede,
  BargeIn,
  BargeInIgnored,
  Dropped,
  Degraded,
  Overload,
  WakeLost,
  WakeRejected,
  Placement,
};

// The three fields of `PlaybackSettings` (`protocol.py`); the defaults are the plan's item 5.
struct PlaybackParams {
  uint32_t prebuffer_ms = 80;
  uint32_t fade_ms = 30;
  uint32_t progress_interval_ms = 100;
};

struct VoiceStateInfo {
  bool answer;  // `cause` is `answer`, else `broadcast`
  bool has_req;
  uint32_t req;
  bool active;
  bool muted;
  bool refused;
  PlaybackParams playback;
};

struct StateCycleInfo {
  MistState state;
  char turn_id[kIdCap];  // empty when absent
};

struct DecisionInfo {
  DecisionKind kind;
  char turn_id[kIdCap];  // empty when null
};

struct StreamStartInfo {
  char turn_id[kIdCap];
  bool has_audio_tag;
  uint16_t audio_tag;
};

struct PlaybackStopInfo {
  char turn_id[kIdCap];
  uint16_t audio_tag;
  uint32_t fade_ms;
};

struct PlaybackDuckInfo {
  char turn_id[kIdCap];
  float db;
};

struct SessionStartedInfo {
  char protocol_version[16];
};

struct ActionResultInfo {
  char alarm_id[kIdCap];
  char action[16];
  char request_id[kIdCap];  // empty when absent
  char outcome[16];
};

struct ErrorInfo {
  char kind[16];
  bool retriable;
};

// The fields of the one message read; only the member matching `type` is set. For `alarm_state`
// and `alarm_fired` only `type` is set: `alarm_rules` reads them from the same `Doc`.
struct Inbound {
  InboundType type;
  VoiceStateInfo voice_state;
  StateCycleInfo state_cycle;
  DecisionInfo decision;
  StreamStartInfo stream_start;
  PlaybackStopInfo playback_stop;
  PlaybackDuckInfo playback_duck;
  SessionStartedInfo session_started;
  ActionResultInfo action_result;
  ErrorInfo error;
};

// Parses one JSON text frame into `doc` and reads it into `out`. A known type with a missing or
// ill-typed field is `Field`; the message is then dropped rather than half-applied.
ReadError read_inbound(json::Doc* doc, const char* text, size_t len, Inbound* out);

// Helpers shared with `alarm_rules`: a string field into a fixed buffer, and an aware time.
bool read_id(json::Value value, char* out, size_t cap);
bool read_time(json::Value value, int64_t* epoch_ms);

}  // namespace mist_core
