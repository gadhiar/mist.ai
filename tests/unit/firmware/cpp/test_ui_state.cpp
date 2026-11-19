// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.
//
// Host tests for `ui_state` (plan Design item 7): the LED pattern and model enablement from
// connection, mute, `state_cycle` and ringing; the button's and the stop word's actions; the
// dial's volume and the speaker level it sets (plan v4, A13), checked against the DAC driver's own
// formula. No button event ever opens the microphone: there is no push-to-talk.

#include <cmath>
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

constexpr float kDialStep = 0.05f;

bool near(float a, float b) { return std::fabs(a - b) < 1e-5f; }

// The byte the AIC3204 driver writes for a speaker level: `AIC3204::set_volume` clamps it to
// [0, 1], and `write_volume_` computes `-127 + volume * (48 - -127)` in float and stores it in an
// `int8_t`, which truncates toward zero (esphome/components/aic3204/aic3204.cpp). 0 is 0 dB of
// digital gain; each step is 0.5 dB.
int aic3204_byte(float volume) {
  const float clamped = volume < 0.0f ? 0.0f : (volume > 1.0f ? 1.0f : volume);
  const int8_t min_byte = -127;
  const int8_t max_byte = 48;
  return static_cast<int8_t>(min_byte + (clamped * (max_byte - min_byte)));
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

MIST_TEST(set_volume_holds_the_volume_to_its_range) {
  UiState ui;

  ui.set_volume(2.0f);
  CHECK(ui.volume() == 1.0f);
  ui.set_volume(-1.0f);
  CHECK(ui.volume() == 0.0f);
  ui.set_volume(std::nanf(""));
  CHECK(ui.volume() == 0.0f);
}

// -- The output level (plan v4, A13) -----------------------------------------------------------

MIST_TEST(with_nothing_saved_the_level_writes_dac_byte_zero) {
  UiState ui;

  CHECK(ui.volume() >= 0.711f && ui.volume() <= 0.737f);
  CHECK_EQ(aic3204_byte(ui.speaker_level()), 0);
}

MIST_TEST(the_level_is_the_stock_remap_of_the_volume) {
  UiState ui;
  ui.set_volume(0.5f);
  CHECK(near(ui.speaker_level(), 0.625f));
  ui.set_volume(1.0f);

  CHECK(near(ui.speaker_level(), 0.85f));
  CHECK_EQ(aic3204_byte(ui.speaker_level()), 21);
  ui.set_volume(kDialStep);
  CHECK(near(ui.speaker_level(), 0.4f + 0.45f * kDialStep));
  CHECK_EQ(aic3204_byte(ui.speaker_level()), -53);
}

MIST_TEST(volume_zero_is_silent) {
  UiState ui;
  ui.set_volume(0.0f);

  // The speaker treats a level under 0.001 as silence and mutes the DAC (speaker.h).
  CHECK(ui.speaker_level() == 0.0f);
}

MIST_TEST(the_dial_moves_the_volume_in_steps_of_0_05) {
  UiState ui;
  ui.set_volume(0.5f);

  CHECK(near(ui.on_dial(1), 0.55f));
  CHECK(near(ui.on_dial(2), 0.65f));
  CHECK(near(ui.on_dial(-3), 0.5f));
  CHECK(near(ui.volume(), 0.5f));
  CHECK(near(ui.speaker_level(), 0.625f));
}

MIST_TEST(the_dial_stops_at_both_ends) {
  UiState ui;

  CHECK(ui.on_dial(100) == 1.0f);
  CHECK(ui.on_dial(1) == 1.0f);
  CHECK_EQ(aic3204_byte(ui.speaker_level()), 21);
  CHECK(ui.on_dial(-100) == 0.0f);
  CHECK(ui.on_dial(-1) == 0.0f);
  CHECK(ui.speaker_level() == 0.0f);
  CHECK(near(ui.on_dial(1), kDialStep));
}

MIST_TEST(from_the_default_the_dial_reaches_full_and_silent) {
  UiState up;
  int steps = 0;
  while (up.volume() < 1.0f && steps < 100) {
    up.on_dial(1);
    ++steps;
  }
  CHECK_EQ(steps, 6);
  UiState down;
  steps = 0;
  while (down.volume() > 0.0f && steps < 100) {
    down.on_dial(-1);
    ++steps;
  }

  CHECK_EQ(steps, 15);
  CHECK(down.speaker_level() == 0.0f);
}

MIST_TEST(while_a_ring_plays_its_floor_applies) {
  UiState ui;
  ui.set_volume(0.2f);
  ui.set_ringing(true);

  CHECK(near(ui.speaker_level(), 0.4f + 0.45f * 0.6f));
  ui.set_volume(0.0f);
  CHECK(near(ui.speaker_level(), 0.4f + 0.45f * 0.6f));
  ui.set_volume(0.7f);
  CHECK(near(ui.speaker_level(), 0.4f + 0.45f * 0.7f));
  ui.set_ringing(false);
  ui.set_volume(0.2f);
  CHECK(near(ui.speaker_level(), 0.4f + 0.45f * 0.2f));
}

MIST_TEST(a_ring_never_plays_above_dac_byte_zero) {
  UiState ui;
  ui.set_ringing(true);

  for (int step = 0; step <= 20; ++step) {
    ui.set_volume(static_cast<float>(step) / 20.0f);
    CHECK(aic3204_byte(ui.speaker_level()) <= 0);
  }
  ui.set_volume(1.0f);
  CHECK_EQ(aic3204_byte(ui.speaker_level()), 0);
  ui.set_ringing(false);
  CHECK_EQ(aic3204_byte(ui.speaker_level()), 21);
}

MIST_TEST(the_ring_floor_is_set_and_held_to_the_range) {
  UiState ui;
  CHECK(near(ui.ring_floor(), 0.6f));
  ui.set_volume(0.0f);
  ui.set_ringing(true);

  ui.set_ring_floor(0.3f);
  CHECK(near(ui.speaker_level(), 0.4f + 0.45f * 0.3f));
  ui.set_ring_floor(0.95f);
  CHECK_EQ(aic3204_byte(ui.speaker_level()), 0);
  ui.set_ring_floor(-1.0f);
  CHECK(ui.ring_floor() == 0.0f);
  CHECK(ui.speaker_level() == 0.0f);
}

MIST_TEST(a_saved_volume_restores) {
  UiState before;
  before.on_dial(-5);
  const float saved = before.volume();

  UiState after;
  after.restore_volume(true, saved);

  CHECK(after.volume() == saved);
  CHECK(after.speaker_level() == before.speaker_level());
}

MIST_TEST(an_unusable_saved_volume_keeps_the_default) {
  const float fallback = UiState().volume();
  UiState missing;
  missing.set_volume(0.1f);
  missing.restore_volume(false, 0.3f);
  CHECK(missing.volume() == fallback);
  UiState garbled;
  garbled.restore_volume(true, std::nanf(""));
  CHECK(garbled.volume() == fallback);
  UiState huge;
  huge.restore_volume(true, 7.0f);

  CHECK(huge.volume() == 1.0f);
}

// -- The stop word -------------------------------------------------------------------------------

MIST_TEST(the_stop_word_stops_a_ring) {
  UiState ui = connected_in(MistState::Speak);
  ui.set_ringing(true);

  CHECK(ui.on_stop_word() == ButtonAction::StopRing);
}

MIST_TEST(the_stop_word_outside_a_ring_is_left_to_the_uplink) {
  for (MistState state : kStates) {
    UiState ui = connected_in(state);
    CHECK(ui.on_stop_word() == ButtonAction::None);
  }
}
