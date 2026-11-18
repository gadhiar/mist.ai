// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.
//
// The wake uplink (plan Design item 4): the one place that decides when microphone audio leaves
// the device.
//
// It opens only on a name detection while connected, unmuted and with `voice_state.active` true.
// Opening sends `wake {keyword, score, backlog_ms}`, then the newest `backlog_ms` of the pre-wake
// ring oldest first, then live frames, all `0x10` with `turn_tag` 0 and one `seq` sequence per
// connection. `backlog_ms` is min(5000, buffered), in whole 20 ms frames.
//
// It closes on the first of: `state_cycle` `speak` or `idle`; `voice_decision` `wake_rejected`,
// `wake_lost` or `dropped`; 30 s after the wake; mute; disconnection. It stays open through
// `endpoint`, `hold` and `think`, so the server's hold and supersede logic still hears a sentence
// that goes on.
//
// "Stop" during her speech sends `barge_in {turn_id, keyword: "stop"}` and opens nothing.
//
// Threading: every method runs on one task, the ring's consumer; the ring's producer (the
// microphone tap) runs anywhere else.

#pragma once

#include <cstddef>
#include <cstdint>

#include "codec.h"
#include "ring.h"

namespace mist_core {

// Where the uplink's messages go; on the device, the link's outbound queue. A `false` return
// means the queue had no room: a text message is then not sent, and a frame is retried by the
// next `pump` with the same `seq`.
class UplinkSink {
 public:
  virtual ~UplinkSink() = default;
  virtual bool send_text(const char* text, size_t len) = 0;
  virtual bool send_binary(const uint8_t* data, size_t len) = 0;
};

struct UplinkConfig {
  // The name the device spots, sent as `wake.keyword`. A spike build passes its stand-in.
  const char* keyword = "mist";
  // The most pre-wake audio a wake sends; held to at most 5000 ms (plan item 4).
  uint32_t backlog_ms = 5000;
  // How long an uplink stays open after its wake.
  uint32_t cap_ms = 30000;
};

enum class WakeOutcome : uint8_t {
  Opened,
  NotConnected,
  Muted,
  VoiceInactive,
  AlreadyOpen,
  SendFailed,
};

enum class StopOutcome : uint8_t {
  BargeInSent,
  AlreadySent,  // a barge-in for this turn went out already
  NotSpeaking,
  NotConnected,
  Muted,
  SendFailed,
};

enum class CloseReason : uint8_t { None, StateCycle, Decision, Cap, Mute, Disconnect };

class Uplink {
 public:
  static constexpr uint32_t kMaxBacklogMs = 5000;

  Uplink(PcmRing& ring, UplinkSink& sink, const UplinkConfig& config = UplinkConfig());

  // A new connection: `seq` restarts at 0 and voice is inactive until a `voice_state` says so.
  // Its `voice_start` already carried the current mute state, so no `mute` is owed.
  void on_connected();
  void on_disconnected();
  // `voice_state.active` from either cause.
  void on_voice_state(bool active);
  // The hardware mute switch. Mute closes the uplink and clears the ring; unmute clears it
  // again, so nothing captured before or during the mute reaches a later backlog. While
  // connected, a change the server does not know yet is sent as `mute {active}`; returns false
  // only when that send was refused. A refused one is sent again by the next `pump` or `on_wake`
  // (a wake never reaches the server ahead of the unmute before it), unless the switch went back
  // first or a new connection's `voice_start` carries the state instead.
  bool on_mute(bool muted);
  void on_state_cycle(MistState state, const char* turn_id);
  void on_voice_decision(DecisionKind kind);

  // A name detection. `now_ms` is a free-running millisecond clock that may wrap.
  WakeOutcome on_wake(float score, uint32_t now_ms);
  // A "stop" detection.
  StopOutcome on_stop_keyword();

  // Sends a mute change the sink refused before; then sends the frames the ring holds past the
  // cursor, at most `max_frames`, and closes the uplink at its cap. Returns the frames handed to
  // the sink.
  uint32_t pump(uint32_t now_ms, uint32_t max_frames = 0xFFFFFFFFu);

  bool is_open() const { return open_; }
  bool connected() const { return connected_; }
  bool muted() const { return muted_; }
  uint32_t next_seq() const { return seq_; }
  uint32_t last_backlog_ms() const { return last_backlog_ms_; }
  CloseReason last_close() const { return last_close_; }

 private:
  void close(CloseReason reason);
  bool send_pending();
  // Sends `mute {active}` when the server's mute state differs from the switch's; true when they
  // agree afterwards.
  bool sync_mute();

  PcmRing& ring_;
  UplinkSink& sink_;
  UplinkConfig config_;

  bool connected_ = false;
  bool voice_active_ = false;
  bool muted_ = false;
  // The mute state the server holds: set from `voice_start` on each connection, then by each
  // `mute` the sink took.
  bool server_muted_ = false;
  bool open_ = false;
  uint32_t opened_at_ms_ = 0;
  uint32_t seq_ = 0;
  uint32_t last_backlog_ms_ = 0;
  CloseReason last_close_ = CloseReason::None;

  MistState state_ = MistState::Idle;
  char speaking_turn_[kIdCap] = {};
  char barged_turn_[kIdCap] = {};

  // A frame the sink refused, kept with its `seq` for the next pump.
  bool has_pending_ = false;
  size_t pending_len_ = 0;
  uint8_t pending_[kHeaderSize + PcmRing::kFrameBytes] = {};
};

}  // namespace mist_core
