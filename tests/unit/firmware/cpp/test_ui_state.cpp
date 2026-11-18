// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.
//
// Host tests for `ui_state` (plan Design item 7): the LED pattern and model enablement from
// connection, mute, `state_cycle` and ringing; the button's actions; the dial's volume. No button
// event ever opens the microphone: there is no push-to-talk.

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "check.h"
#include "codec.h"
#include "ring.h"
#include "ui_state.h"
#include "uplink.h"

using mist_core::ButtonAction;
using mist_core::ButtonEvent;
using mist_core::LedPattern;
using mist_core::MistState;
using mist_core::PcmRing;
using mist_core::UiState;

namespace {

const MistState kStates[] = {MistState::Idle, MistState::Attend, MistState::Listen,
                             MistState::Think, MistState::Speak};
const ButtonEvent kEvents[] = {ButtonEvent::SinglePress, ButtonEvent::DoublePress,
                               ButtonEvent::LongPress};

class RecordingSink : public mist_core::UplinkSink {
 public:
  bool send_text(const char* text, size_t len) override {
    texts.emplace_back(text, len);
    return true;
  }
  bool send_binary(const uint8_t*, size_t) override {
    ++binaries;
    return true;
  }
  std::vector<std::string> texts;
  int binaries = 0;
};

UiState connected_in(MistState state) {
  UiState ui;
  ui.set_connected(true);
  ui.set_state(state);
  return ui;
}

}  // namespace

MIST_TEST(leds_follow_her_state_while_connected) {
  CHECK(connected_in(MistState::Idle).led() == LedPattern::Idle);
  CHECK(connected_in(MistState::Attend).led() == LedPattern::Attend);
  CHECK(connected_in(MistState::Listen).led() == LedPattern::Listen);
  CHECK(connected_in(MistState::Think).led() == LedPattern::Think);
  CHECK(connected_in(MistState::Speak).led() == LedPattern::Speak);
}

MIST_TEST(a_lost_connection_shows_and_forgets_her_state) {
  UiState ui = connected_in(MistState::Speak);

  ui.set_connected(false);
  CHECK(ui.led() == LedPattern::Disconnected);
  ui.set_connected(true);

  CHECK(ui.led() == LedPattern::Idle);
}

MIST_TEST(mute_shows_over_every_state_and_over_a_lost_connection) {
  for (MistState state : kStates) {
    UiState ui = connected_in(state);
    ui.set_muted(true);
    CHECK(ui.led() == LedPattern::Muted);
  }
  UiState offline;
  offline.set_muted(true);

  CHECK(offline.led() == LedPattern::Muted);
}

MIST_TEST(ringing_shows_over_everything) {
  UiState ui = connected_in(MistState::Speak);
  ui.set_muted(true);
  ui.set_ringing(true);
  CHECK(ui.led() == LedPattern::Ringing);
  ui.set_connected(false);
  CHECK(ui.led() == LedPattern::Ringing);
  ui.set_ringing(false);

  CHECK(ui.led() == LedPattern::Muted);
}

MIST_TEST(the_name_model_runs_only_while_connected_and_unmuted) {
  UiState ui;
  CHECK(!ui.wake_model_enabled());
  ui.set_connected(true);
  CHECK(ui.wake_model_enabled());
  ui.set_muted(true);
  CHECK(!ui.wake_model_enabled());
  ui.set_muted(false);
  ui.set_state(MistState::Speak);

  CHECK(ui.wake_model_enabled());
}

MIST_TEST(the_stop_model_runs_only_while_she_speaks_or_a_ring_plays) {
  for (MistState state : kStates) {
    UiState ui = connected_in(state);
    CHECK_EQ(ui.stop_model_enabled(), state == MistState::Speak);
  }
  UiState ringing;
  ringing.set_ringing(true);
  CHECK(ringing.stop_model_enabled());
  ringing.set_muted(true);
  CHECK(!ringing.stop_model_enabled());
  UiState speaking_muted = connected_in(MistState::Speak);
  speaking_muted.set_muted(true);

  CHECK(!speaking_muted.stop_model_enabled());
}

MIST_TEST(a_single_press_stops_a_ring_and_a_double_press_snoozes_it) {
  UiState ui = connected_in(MistState::Speak);
  ui.set_ringing(true);

  CHECK(ui.on_button(ButtonEvent::SinglePress) == ButtonAction::StopRing);
  CHECK(ui.on_button(ButtonEvent::DoublePress) == ButtonAction::SnoozeRing);
  CHECK(ui.on_button(ButtonEvent::LongPress) == ButtonAction::None);
}

MIST_TEST(a_single_press_while_she_speaks_interrupts) {
  UiState ui = connected_in(MistState::Speak);

  CHECK(ui.on_button(ButtonEvent::SinglePress) == ButtonAction::Interrupt);
  CHECK(ui.on_button(ButtonEvent::DoublePress) == ButtonAction::None);
  ui.set_connected(false);
  CHECK(ui.on_button(ButtonEvent::SinglePress) == ButtonAction::None);
}

MIST_TEST(a_press_outside_her_speech_and_a_ring_does_nothing) {
  for (MistState state : kStates) {
    if (state == MistState::Speak) continue;
    UiState ui = connected_in(state);
    for (ButtonEvent event : kEvents) {
      CHECK(ui.on_button(event) == ButtonAction::None);
    }
  }
}

MIST_TEST(no_button_event_ever_reaches_the_uplink) {
  alignas(4) static uint8_t storage[PcmRing::kFrameBytes * 300];
  PcmRing ring(storage, sizeof(storage));
  RecordingSink sink;
  mist_core::Uplink uplink(ring, sink);
  uplink.on_connected();
  uplink.on_voice_state(true);
  int16_t frame[PcmRing::kFrameSamples] = {};
  for (int i = 0; i < 100; ++i) ring.write(frame, PcmRing::kFrameSamples);

  for (int connected = 0; connected < 2; ++connected) {
    for (int muted = 0; muted < 2; ++muted) {
      for (int ringing = 0; ringing < 2; ++ringing) {
        for (MistState state : kStates) {
          UiState ui;
          ui.set_connected(connected != 0);
          ui.set_muted(muted != 0);
          ui.set_ringing(ringing != 0);
          ui.set_state(state);
          for (ButtonEvent event : kEvents) {
            // The wiring T6 adds: an interrupt goes out as a control message, and nothing a
            // button does is handed to the uplink.
            const ButtonAction action = ui.on_button(event);
            CHECK(action == ButtonAction::None || action == ButtonAction::StopRing ||
                  action == ButtonAction::SnoozeRing || action == ButtonAction::Interrupt);
            if (action == ButtonAction::Interrupt) {
              char buf[64];
              const size_t len = mist_core::build_interrupt(buf, sizeof(buf));
              sink.send_text(buf, len);
            }
            uplink.pump(0);
          }
        }
      }
    }
  }

  CHECK(!uplink.is_open());
  CHECK_EQ(sink.binaries, 0);
  for (const std::string& text : sink.texts) {
    CHECK(text == "{\"type\":\"interrupt\"}");
  }
}

MIST_TEST(the_dial_moves_the_volume_within_its_range) {
  UiState ui;
  ui.set_volume(0.5f);

  CHECK(ui.on_dial(0.25f) == 0.75f);
  CHECK(ui.on_dial(1.0f) == 1.0f);
  CHECK(ui.on_dial(-3.0f) == 0.0f);
  ui.set_volume(2.0f);
  CHECK(ui.volume() == 1.0f);
  ui.set_volume(-1.0f);
  CHECK(ui.volume() == 0.0f);
}
