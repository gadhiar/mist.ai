// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.
//
// Host tests for the uplink state machine (plan Design item 4): nothing leaves the device before
// a wake, while muted or after a rejection; the backlog goes oldest first with contiguous `seq`;
// `backlog_ms` is min(5000, buffered) in whole 20 ms frames; each close condition; the 30 s cap.

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "check.h"
#include "codec.h"
#include "json_lite.h"
#include "ring.h"
#include "uplink.h"

using mist_core::CloseReason;
using mist_core::DecisionKind;
using mist_core::MistState;
using mist_core::PcmRing;
using mist_core::StopOutcome;
using mist_core::Uplink;
using mist_core::UplinkConfig;
using mist_core::WakeOutcome;
namespace json = mist_core::json;

namespace {

constexpr size_t kFrame = PcmRing::kFrameSamples;

struct SentFrame {
  uint32_t seq;
  uint32_t index;
};

class RecordingSink : public mist_core::UplinkSink {
 public:
  bool send_text(const char* text, size_t len) override {
    if (refuse_text) return false;
    texts.emplace_back(text, len);
    return true;
  }

  bool send_binary(const uint8_t* data, size_t len) override {
    if (refuse_binary) return false;
    mist_core::Frame frame{};
    if (mist_core::decode_frame(data, len, mist_core::Direction::Up, &frame) !=
        mist_core::FrameError::None) {
      ++malformed;
      return true;
    }
    const uint8_t* p = frame.payload;
    const uint32_t index = static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
                           (static_cast<uint32_t>(p[2]) << 16) |
                           (static_cast<uint32_t>(p[3]) << 24);
    if (frame.payload_len != kFrame * 2) ++malformed;
    frames.push_back(SentFrame{frame.seq, index});
    return true;
  }

  std::vector<std::string> texts;
  std::vector<SentFrame> frames;
  int malformed = 0;
  bool refuse_text = false;
  bool refuse_binary = false;
};

// A ring sized like the device's (6 s), in static storage.
alignas(4) uint8_t g_storage[PcmRing::bytes_for_ms(6000)];

struct Rig {
  explicit Rig(const UplinkConfig& config = UplinkConfig())
      : ring(g_storage, sizeof(g_storage)), uplink(ring, sink, config) {}

  // Writes `count` frames whose first two samples carry their running index.
  void write(uint32_t count) {
    int16_t frame[kFrame] = {};
    for (uint32_t k = 0; k < count; ++k) {
      frame[0] = static_cast<int16_t>(static_cast<uint16_t>(next_index & 0xFFFFu));
      frame[1] = static_cast<int16_t>(static_cast<uint16_t>(next_index >> 16));
      ring.write(frame, kFrame);
      ++next_index;
    }
  }

  void ready() {
    uplink.on_connected();
    uplink.on_voice_state(true);
  }

  PcmRing ring;
  RecordingSink sink;
  Uplink uplink;
  uint32_t next_index = 0;
};

bool field_equals(const std::string& text, const char* key, const char* expected) {
  static json::Token pool[64];
  json::Doc doc(pool, 64);
  if (doc.parse(text.data(), text.size()) != json::ParseError::None) return false;
  return doc.root().get(key).equals(expected);
}

int64_t int_field(const std::string& text, const char* key) {
  static json::Token pool[64];
  json::Doc doc(pool, 64);
  int64_t value = -1;
  if (doc.parse(text.data(), text.size()) != json::ParseError::None) return -1;
  doc.root().get(key).get_int(&value);
  return value;
}

bool bool_field(const std::string& text, const char* key, bool* out) {
  static json::Token pool[64];
  json::Doc doc(pool, 64);
  if (doc.parse(text.data(), text.size()) != json::ParseError::None) return false;
  return doc.root().get(key).get_bool(out);
}

}  // namespace

MIST_TEST(nothing_is_sent_before_a_wake) {
  Rig rig;
  rig.ready();

  for (int i = 0; i < 50; ++i) {
    rig.write(1);
    CHECK_EQ(rig.uplink.pump(static_cast<uint32_t>(i) * 20u), 0);
  }

  CHECK(rig.sink.texts.empty());
  CHECK(rig.sink.frames.empty());
  CHECK(!rig.uplink.is_open());
}

MIST_TEST(a_wake_sends_the_wake_then_the_backlog_oldest_first_then_live_frames) {
  Rig rig;
  rig.ready();
  rig.write(10);

  CHECK(rig.uplink.on_wake(0.9f, 1000) == WakeOutcome::Opened);

  CHECK_EQ(rig.sink.texts.size(), 1);
  CHECK(field_equals(rig.sink.texts[0], "type", "wake"));
  CHECK(field_equals(rig.sink.texts[0], "keyword", "mist"));
  CHECK_EQ(int_field(rig.sink.texts[0], "backlog_ms"), 200);
  CHECK_EQ(rig.sink.frames.size(), 10);
  for (uint32_t k = 0; k < 10; ++k) {
    CHECK_EQ(rig.sink.frames[k].seq, k);
    CHECK_EQ(rig.sink.frames[k].index, k);
  }
  rig.write(2);
  CHECK_EQ(rig.uplink.pump(1040), 2);
  CHECK_EQ(rig.sink.frames.size(), 12);
  CHECK_EQ(rig.sink.frames[10].seq, 10);
  CHECK_EQ(rig.sink.frames[10].index, 10);
  CHECK_EQ(rig.sink.frames[11].seq, 11);
  CHECK_EQ(rig.sink.malformed, 0);
}

MIST_TEST(backlog_is_capped_at_5000_ms_of_the_newest_frames) {
  Rig rig;
  rig.ready();
  rig.write(300);

  CHECK(rig.uplink.on_wake(0.9f, 0) == WakeOutcome::Opened);

  CHECK_EQ(int_field(rig.sink.texts[0], "backlog_ms"), 5000);
  CHECK_EQ(rig.uplink.last_backlog_ms(), 5000);
  CHECK_EQ(rig.sink.frames.size(), 250);
  CHECK_EQ(rig.sink.frames.front().index, 50);
  CHECK_EQ(rig.sink.frames.back().index, 299);
  CHECK_EQ(rig.sink.frames.back().seq, 249);
}

MIST_TEST(backlog_counts_only_whole_buffered_frames) {
  Rig rig;
  rig.ready();
  rig.write(249);
  int16_t half[kFrame / 2] = {};
  rig.ring.write(half, kFrame / 2);

  CHECK(rig.uplink.on_wake(0.9f, 0) == WakeOutcome::Opened);

  CHECK_EQ(int_field(rig.sink.texts[0], "backlog_ms"), 4980);
  CHECK_EQ(rig.sink.frames.size(), 249);
}

MIST_TEST(backlog_one_frame_over_the_cap_still_sends_5000_ms) {
  Rig rig;
  rig.ready();
  rig.write(251);

  CHECK(rig.uplink.on_wake(0.9f, 0) == WakeOutcome::Opened);

  CHECK_EQ(int_field(rig.sink.texts[0], "backlog_ms"), 5000);
  CHECK_EQ(rig.sink.frames.front().index, 1);
}

MIST_TEST(a_configured_backlog_above_the_plan_is_held_to_5000_ms) {
  UplinkConfig config;
  config.backlog_ms = 9000;
  Rig rig(config);
  rig.ready();
  rig.write(300);

  CHECK(rig.uplink.on_wake(0.9f, 0) == WakeOutcome::Opened);

  CHECK_EQ(int_field(rig.sink.texts[0], "backlog_ms"), 5000);
}

MIST_TEST(the_wake_keyword_is_a_parameter) {
  UplinkConfig config;
  config.keyword = "stand-in";
  Rig rig(config);
  rig.ready();

  CHECK(rig.uplink.on_wake(0.5f, 0) == WakeOutcome::Opened);

  CHECK(field_equals(rig.sink.texts[0], "keyword", "stand-in"));
}

MIST_TEST(a_wake_is_refused_while_disconnected) {
  Rig rig;
  rig.write(10);

  CHECK(rig.uplink.on_wake(0.9f, 0) == WakeOutcome::NotConnected);

  CHECK(rig.sink.texts.empty());
  CHECK(rig.sink.frames.empty());
}

MIST_TEST(a_wake_is_refused_until_voice_is_active) {
  Rig rig;
  rig.uplink.on_connected();
  rig.write(10);

  CHECK(rig.uplink.on_wake(0.9f, 0) == WakeOutcome::VoiceInactive);
  rig.uplink.on_voice_state(true);
  rig.uplink.on_voice_state(false);
  CHECK(rig.uplink.on_wake(0.9f, 0) == WakeOutcome::VoiceInactive);

  CHECK(rig.sink.texts.empty());
  CHECK(rig.sink.frames.empty());
}

MIST_TEST(nothing_is_sent_while_muted) {
  Rig rig;
  rig.ready();
  CHECK(rig.uplink.on_mute(true));
  rig.sink.texts.clear();
  rig.write(100);

  CHECK(rig.uplink.on_wake(0.9f, 0) == WakeOutcome::Muted);
  CHECK(rig.uplink.on_stop_keyword() == StopOutcome::Muted);
  CHECK_EQ(rig.uplink.pump(100), 0);

  CHECK(rig.sink.texts.empty());
  CHECK(rig.sink.frames.empty());
}

MIST_TEST(mute_closes_the_uplink_clears_the_ring_and_sends_mute) {
  Rig rig;
  rig.ready();
  rig.write(20);
  CHECK(rig.uplink.on_wake(0.9f, 0) == WakeOutcome::Opened);
  const size_t sent = rig.sink.frames.size();

  CHECK(rig.uplink.on_mute(true));
  rig.write(5);
  rig.uplink.pump(100);

  CHECK(!rig.uplink.is_open());
  CHECK(rig.uplink.last_close() == CloseReason::Mute);
  CHECK_EQ(rig.sink.frames.size(), sent);
  CHECK_EQ(rig.sink.texts.size(), 2);
  CHECK(field_equals(rig.sink.texts[1], "type", "mute"));
  bool active = false;
  CHECK(bool_field(rig.sink.texts[1], "active", &active));
  CHECK(active);
}

MIST_TEST(audio_from_before_or_during_mute_never_reaches_a_later_backlog) {
  Rig rig;
  rig.ready();
  rig.write(50);
  rig.uplink.on_mute(true);
  rig.write(50);

  CHECK(rig.uplink.on_mute(false));
  rig.write(3);
  CHECK(rig.uplink.on_wake(0.9f, 0) == WakeOutcome::Opened);

  bool active = true;
  CHECK(bool_field(rig.sink.texts[1], "active", &active));
  CHECK(!active);
  CHECK_EQ(int_field(rig.sink.texts[2], "backlog_ms"), 60);
  CHECK_EQ(rig.sink.frames.size(), 3);
  CHECK_EQ(rig.sink.frames[0].index, 100);
}

MIST_TEST(mute_while_disconnected_sends_nothing_and_is_kept) {
  Rig rig;

  CHECK(rig.uplink.on_mute(true));

  CHECK(rig.sink.texts.empty());
  CHECK(rig.uplink.muted());
  rig.ready();
  CHECK(rig.uplink.on_wake(0.9f, 0) == WakeOutcome::Muted);
}

MIST_TEST(a_refused_mute_is_sent_by_the_next_pump) {
  Rig rig;
  rig.ready();
  rig.sink.refuse_text = true;

  CHECK(!rig.uplink.on_mute(true));
  CHECK(rig.uplink.muted());
  CHECK_EQ(rig.uplink.pump(20), 0);
  rig.sink.refuse_text = false;
  rig.write(5);
  CHECK_EQ(rig.uplink.pump(40), 0);

  CHECK_EQ(rig.sink.texts.size(), 1);
  CHECK(field_equals(rig.sink.texts[0], "type", "mute"));
  bool active = false;
  CHECK(bool_field(rig.sink.texts[0], "active", &active));
  CHECK(active);
  CHECK(rig.sink.frames.empty());
  rig.uplink.pump(60);
  CHECK_EQ(rig.sink.texts.size(), 1);
}

MIST_TEST(a_mute_undone_before_its_retry_sends_nothing) {
  Rig rig;
  rig.ready();
  rig.sink.refuse_text = true;
  CHECK(!rig.uplink.on_mute(true));

  rig.sink.refuse_text = false;
  CHECK(rig.uplink.on_mute(false));
  rig.uplink.pump(20);

  CHECK(rig.sink.texts.empty());
  CHECK(!rig.uplink.muted());
}

MIST_TEST(a_new_connection_drops_an_unsent_mute_because_voice_start_carries_it) {
  Rig rig;
  rig.ready();
  rig.sink.refuse_text = true;
  CHECK(!rig.uplink.on_mute(true));
  rig.uplink.on_disconnected();
  rig.sink.refuse_text = false;

  rig.ready();
  rig.uplink.pump(20);

  CHECK(rig.sink.texts.empty());
  CHECK(rig.uplink.muted());
  CHECK(rig.uplink.on_wake(0.9f, 40) == WakeOutcome::Muted);
}

MIST_TEST(an_unsent_unmute_goes_before_the_next_wake) {
  Rig rig;
  rig.ready();
  CHECK(rig.uplink.on_mute(true));
  rig.sink.refuse_text = true;
  CHECK(!rig.uplink.on_mute(false));
  rig.sink.refuse_text = false;
  rig.write(3);

  CHECK(rig.uplink.on_wake(0.9f, 0) == WakeOutcome::Opened);

  CHECK_EQ(rig.sink.texts.size(), 3);
  CHECK(field_equals(rig.sink.texts[1], "type", "mute"));
  bool active = true;
  CHECK(bool_field(rig.sink.texts[1], "active", &active));
  CHECK(!active);
  CHECK(field_equals(rig.sink.texts[2], "type", "wake"));
  CHECK_EQ(rig.sink.frames.size(), 3);
}

MIST_TEST(a_wake_waits_while_an_unmute_cannot_be_sent) {
  Rig rig;
  rig.ready();
  CHECK(rig.uplink.on_mute(true));
  rig.sink.refuse_text = true;
  CHECK(!rig.uplink.on_mute(false));
  rig.write(3);

  CHECK(rig.uplink.on_wake(0.9f, 0) == WakeOutcome::SendFailed);

  CHECK(!rig.uplink.is_open());
  CHECK_EQ(rig.sink.texts.size(), 1);
  CHECK(rig.sink.frames.empty());
}

MIST_TEST(each_rejecting_decision_closes_the_uplink_and_nothing_follows) {
  const DecisionKind closing[] = {DecisionKind::WakeRejected, DecisionKind::WakeLost,
                                  DecisionKind::Dropped};
  for (DecisionKind kind : closing) {
    Rig rig;
    rig.ready();
    rig.write(10);
    CHECK(rig.uplink.on_wake(0.9f, 0) == WakeOutcome::Opened);
    const size_t sent = rig.sink.frames.size();

    rig.uplink.on_voice_decision(kind);
    rig.write(10);
    rig.uplink.pump(200);

    CHECK(!rig.uplink.is_open());
    CHECK(rig.uplink.last_close() == CloseReason::Decision);
    CHECK_EQ(rig.sink.frames.size(), sent);
    CHECK_EQ(rig.sink.texts.size(), 1);
  }
}

MIST_TEST(other_decisions_keep_the_uplink_open) {
  const DecisionKind kept[] = {DecisionKind::Endpoint,  DecisionKind::Hold,
                               DecisionKind::Release,   DecisionKind::Supersede,
                               DecisionKind::BargeIn,   DecisionKind::BargeInIgnored,
                               DecisionKind::Degraded,  DecisionKind::Overload,
                               DecisionKind::Placement, DecisionKind::Other};
  Rig rig;
  rig.ready();
  CHECK(rig.uplink.on_wake(0.9f, 0) == WakeOutcome::Opened);

  for (DecisionKind kind : kept) rig.uplink.on_voice_decision(kind);
  rig.write(3);

  CHECK(rig.uplink.is_open());
  CHECK_EQ(rig.uplink.pump(60), 3);
}

MIST_TEST(speak_and_idle_close_the_uplink) {
  const MistState closing[] = {MistState::Speak, MistState::Idle};
  for (MistState state : closing) {
    Rig rig;
    rig.ready();
    CHECK(rig.uplink.on_wake(0.9f, 0) == WakeOutcome::Opened);

    rig.uplink.on_state_cycle(state, "turn-1");
    rig.write(3);

    CHECK(!rig.uplink.is_open());
    CHECK(rig.uplink.last_close() == CloseReason::StateCycle);
    CHECK_EQ(rig.uplink.pump(60), 0);
  }
}

MIST_TEST(attend_listen_and_think_keep_the_uplink_open) {
  Rig rig;
  rig.ready();
  CHECK(rig.uplink.on_wake(0.9f, 0) == WakeOutcome::Opened);

  rig.uplink.on_state_cycle(MistState::Attend, "");
  rig.uplink.on_state_cycle(MistState::Listen, "");
  rig.uplink.on_state_cycle(MistState::Think, "turn-1");
  rig.write(2);

  CHECK(rig.uplink.is_open());
  CHECK_EQ(rig.uplink.pump(40), 2);
}

MIST_TEST(the_uplink_closes_30_s_after_the_wake) {
  Rig rig;
  rig.ready();
  CHECK(rig.uplink.on_wake(0.9f, 1000) == WakeOutcome::Opened);

  rig.write(1);
  CHECK_EQ(rig.uplink.pump(30999), 1);
  CHECK(rig.uplink.is_open());
  rig.write(1);
  CHECK_EQ(rig.uplink.pump(31000), 0);

  CHECK(!rig.uplink.is_open());
  CHECK(rig.uplink.last_close() == CloseReason::Cap);
}

MIST_TEST(the_cap_holds_across_the_millisecond_clock_wrap) {
  Rig rig;
  rig.ready();
  const uint32_t start = 0xFFFFF000u;
  CHECK(rig.uplink.on_wake(0.9f, start) == WakeOutcome::Opened);

  rig.uplink.pump(start + 29999u);
  CHECK(rig.uplink.is_open());
  rig.uplink.pump(start + 30000u);

  CHECK(!rig.uplink.is_open());
}

MIST_TEST(a_disconnection_closes_the_uplink_and_the_next_connection_restarts_seq) {
  Rig rig;
  rig.ready();
  rig.write(5);
  CHECK(rig.uplink.on_wake(0.9f, 0) == WakeOutcome::Opened);
  CHECK_EQ(rig.uplink.next_seq(), 5);

  rig.uplink.on_disconnected();
  CHECK(!rig.uplink.is_open());
  CHECK(rig.uplink.last_close() == CloseReason::Disconnect);
  CHECK(rig.uplink.on_wake(0.9f, 0) == WakeOutcome::NotConnected);
  rig.ready();
  rig.write(2);
  rig.sink.frames.clear();
  CHECK(rig.uplink.on_wake(0.9f, 0) == WakeOutcome::Opened);

  CHECK_EQ(rig.sink.frames.front().seq, 0);
}

MIST_TEST(seq_runs_on_across_uplinks_of_one_connection) {
  Rig rig;
  rig.ready();
  rig.write(3);
  CHECK(rig.uplink.on_wake(0.9f, 0) == WakeOutcome::Opened);
  rig.uplink.on_voice_decision(DecisionKind::WakeRejected);
  rig.write(1);
  rig.sink.frames.clear();

  CHECK(rig.uplink.on_wake(0.9f, 100) == WakeOutcome::Opened);

  CHECK_EQ(rig.sink.frames.front().seq, 3);
}

MIST_TEST(a_second_wake_while_open_sends_nothing_more) {
  Rig rig;
  rig.ready();
  rig.write(4);
  CHECK(rig.uplink.on_wake(0.9f, 0) == WakeOutcome::Opened);

  CHECK(rig.uplink.on_wake(0.95f, 100) == WakeOutcome::AlreadyOpen);

  CHECK_EQ(rig.sink.texts.size(), 1);
  CHECK_EQ(rig.sink.frames.size(), 4);
}

MIST_TEST(a_wake_during_her_speech_opens_a_new_uplink) {
  Rig rig;
  rig.ready();
  rig.uplink.on_state_cycle(MistState::Speak, "turn-1");
  rig.write(4);

  CHECK(rig.uplink.on_wake(0.9f, 0) == WakeOutcome::Opened);

  CHECK(rig.uplink.is_open());
  CHECK_EQ(rig.sink.frames.size(), 4);
}

MIST_TEST(stop_during_her_speech_sends_barge_in_and_opens_nothing) {
  Rig rig;
  rig.ready();
  rig.write(10);
  rig.uplink.on_state_cycle(MistState::Speak, "turn-7");

  CHECK(rig.uplink.on_stop_keyword() == StopOutcome::BargeInSent);
  CHECK(rig.uplink.on_stop_keyword() == StopOutcome::AlreadySent);

  CHECK(!rig.uplink.is_open());
  CHECK(rig.sink.frames.empty());
  CHECK_EQ(rig.sink.texts.size(), 1);
  CHECK(field_equals(rig.sink.texts[0], "type", "barge_in"));
  CHECK(field_equals(rig.sink.texts[0], "turn_id", "turn-7"));
  CHECK(field_equals(rig.sink.texts[0], "keyword", "stop"));
  rig.uplink.on_state_cycle(MistState::Speak, "turn-8");
  CHECK(rig.uplink.on_stop_keyword() == StopOutcome::BargeInSent);
}

MIST_TEST(stop_outside_her_speech_sends_nothing) {
  Rig rig;
  rig.ready();

  CHECK(rig.uplink.on_stop_keyword() == StopOutcome::NotSpeaking);
  rig.uplink.on_state_cycle(MistState::Speak, "");
  CHECK(rig.uplink.on_stop_keyword() == StopOutcome::NotSpeaking);
  rig.uplink.on_state_cycle(MistState::Speak, "turn-1");
  rig.uplink.on_state_cycle(MistState::Listen, "");
  CHECK(rig.uplink.on_stop_keyword() == StopOutcome::NotSpeaking);

  CHECK(rig.sink.texts.empty());
}

MIST_TEST(a_refused_frame_is_retried_with_the_same_seq) {
  Rig rig;
  rig.ready();
  rig.write(3);
  rig.sink.refuse_binary = true;
  CHECK(rig.uplink.on_wake(0.9f, 0) == WakeOutcome::Opened);
  CHECK(rig.sink.frames.empty());

  rig.sink.refuse_binary = false;
  CHECK_EQ(rig.uplink.pump(20), 3);

  CHECK_EQ(rig.sink.frames[0].seq, 0);
  CHECK_EQ(rig.sink.frames[0].index, 0);
  CHECK_EQ(rig.sink.frames[2].seq, 2);
}

MIST_TEST(a_wake_the_link_cannot_take_opens_nothing) {
  Rig rig;
  rig.ready();
  rig.write(3);
  rig.sink.refuse_text = true;

  CHECK(rig.uplink.on_wake(0.9f, 0) == WakeOutcome::SendFailed);

  CHECK(!rig.uplink.is_open());
  CHECK(rig.sink.frames.empty());
}

MIST_TEST(pump_honours_its_frame_budget) {
  Rig rig;
  rig.ready();
  rig.write(10);
  rig.sink.refuse_binary = true;
  CHECK(rig.uplink.on_wake(0.9f, 0) == WakeOutcome::Opened);
  rig.sink.refuse_binary = false;

  CHECK_EQ(rig.uplink.pump(20, 4), 4);
  CHECK_EQ(rig.uplink.pump(20, 4), 4);
  CHECK_EQ(rig.uplink.pump(20, 4), 2);
}
