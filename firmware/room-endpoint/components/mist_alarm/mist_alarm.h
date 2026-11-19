// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.
//
// MIST's alarms on the room device (plan Design item 6, T7; capabilities v2 D3 and A14): the glue
// between `mist_link`, `mist_core`'s `AlarmRules`, the alarm speaker and the player.
//
// - The link. `alarm_state` updates the schedule copy (the entries whose `ring_on` is `all` or
//   names `endpoint_name`) and ends a server ring no longer `ringing`; `alarm_fired` rings and
//   reports `rang` as the tone starts; `alarm_action_result` retires an answered report. Both
//   `alarm_state.at` and `alarm_fired.fired_at` set the clock while SNTP has not synced.
// - Without the link, the schedule copy rings an occurrence by A14's rules, once, and its outcome
//   is reported on the next connection.
// - The ring. `AlarmTone`'s 880 Hz beeps, 16 kHz PCM16 mono, go to `speaker` (the alarm
//   resampler, which feeds its own mixer input). As a ring starts the component hands the player
//   the ring floor and turns its ringing on: the LEDs, the stop model and the level follow (v4
//   A13). The level is the player's alone; nothing here sets a volume.
// - Stop and snooze. `packages/controls.yaml` hands the player's answer to a press or a "stop"
//   (`StopRing`, `SnoozeRing`) to `on_ring_action`; online they go out as `alarm_action`, offline
//   they become the local ring's outcome.
// - Memory. The rules' `AlarmMemory` (the schedule copy, the occurrences rung, the unanswered
//   outcomes) is kept in preferences under `kMemoryKey` and written to flash at once whenever it
//   changes, so a reboot or a power cut loses neither the schedule nor an outcome.
//
// Threading: everything runs on the main loop.

#pragma once

#include <cstddef>
#include <cstdint>

#include "esphome/components/mist_core/alarm_rules.h"
#include "esphome/components/mist_core/codec.h"
#include "esphome/components/mist_core/json_lite.h"
#include "esphome/components/mist_core/ui_state.h"
#include "esphome/components/mist_link/mist_link.h"
#include "esphome/components/mist_player/mist_player.h"
#include "esphome/components/speaker/speaker.h"
#include "esphome/components/time/real_time_clock.h"
#include "esphome/core/component.h"
#include "esphome/core/preferences.h"

namespace esphome {
namespace mist_alarm {

class MistAlarm : public Component, public mist_link::LinkListener, public mist_core::AlarmSink {
 public:
  // The preference the rules' memory is kept under; ESPHome stores it in NVS namespace
  // `esphome` under the decimal `fnv1_hash` of this name.
  static constexpr const char *kMemoryKey = "mist_alarm_schedule";
  // How often the rules look for a due local ring, a ring's timeout and an unsent report.
  static constexpr uint32_t kTickMs = 250;
  // The tone is made 20 ms at a time.
  static constexpr size_t kChunkSamples = mist_core::AlarmTone::kSampleRate / 50;
  // At most this many chunks go to the speaker per pass of the loop; the resampler's buffer
  // refuses the rest.
  static constexpr int kMaxChunksPerLoop = 8;

  void set_link(mist_link::MistLink *link) { this->link_ = link; }
  void set_player(mist_player::MistPlayer *player) { this->player_ = player; }
  void set_speaker(speaker::Speaker *output) { this->speaker_ = output; }
  void set_time(time::RealTimeClock *clock) { this->time_ = clock; }
  void set_endpoint_name(const char *name) { this->endpoint_name_ = name; }
  void set_ring_floor(float floor) { this->ring_floor_ = floor; }

  void setup() override;
  void loop() override;
  void dump_config() override;
  // After the player, whose ring entries it calls, and the link.
  float get_setup_priority() const override { return setup_priority::AFTER_CONNECTION; }

  // `packages/controls.yaml`: what the player answered to a press or a "stop". `StopRing` and
  // `SnoozeRing` act on the ring; anything else is not the alarm's. True when a ring ended.
  bool on_ring_action(mist_core::ButtonAction action);

  // mist_link::LinkListener
  void on_link_connected() override;
  void on_link_disconnected() override;
  void on_link_message(const mist_core::Inbound &message, const mist_core::json::Doc &doc) override;

  // mist_core::AlarmSink
  bool send_text(const char *text, size_t len) override;
  void start_ring(const char *alarm_id) override;
  void stop_ring() override;

 protected:
  int64_t mono_ms_() const;
  int64_t now_ms_() const;
  void persist_();
  void drive_speaker_();
  void feed_tone_();

  mist_link::MistLink *link_{nullptr};
  mist_player::MistPlayer *player_{nullptr};
  speaker::Speaker *speaker_{nullptr};
  time::RealTimeClock *time_{nullptr};
  const char *endpoint_name_{nullptr};
  float ring_floor_{mist_core::UiState::kDefaultRingFloor};

  mist_core::AlarmRules rules_{*this};
  mist_core::AlarmClock clock_;
  mist_core::AlarmTone tone_;

  // The memory as last handed to preferences, in PSRAM, and the rules' revision it holds.
  mist_core::AlarmMemory *memory_{nullptr};
  ESPPreferenceObject memory_pref_;
  uint32_t saved_revision_{0};
  bool memory_ready_{false};

  // A parsed `alarm_state`, kept off the stack.
  mist_core::ScheduleCopy *incoming_{nullptr};

  uint32_t last_tick_ms_{0};
  bool ringing_{false};         // the rules want the tone
  bool speaker_started_{false};  // the speaker was started for the current ring
  int16_t chunk_[kChunkSamples]{};
  size_t chunk_offset_{0};  // bytes of `chunk_` already handed to the speaker
  size_t chunk_bytes_{0};
};

}  // namespace mist_alarm
}  // namespace esphome
