// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.
//
// The room build's player, controls and lights (plan Design items 5 and 7; v4 A13): the glue
// between `mist_link`, the speaker chain, the wake models, the mute switch, the LED ring and
// `mist_core`'s `TagQueue` and `UiState`.
//
// - Replies. `stream_start.audio_tag` names the turn whose `0x01` frames play (24 kHz PCM16
//   mono). They wait in a PSRAM FIFO until the turn's prebuffer is queued, then go to the reply
//   resampler, which feeds the mixer and the I2S DAC. The player keeps at most `kLeadMs` of audio
//   in the speaker chain ahead of the DAC, so a stop's fade is heard soon after it arrives. The
//   resampler's output callback counts the samples the DAC has played; from it the player sends
//   `playback_progress` and, once a completed turn has played out, `playback_done`.
//   `playback_stop` and `0x03` keep the next `fade_ms` of queued audio, ramp it to silence
//   (`mist_core::apply_fade_out`) and drop the rest. `playback_duck` is the mixer's
//   `apply_ducking` on the reply's mixer input, over the session's `fade_ms`; a new turn
//   restores it.
// - Level. The user volume is kept in preferences (NVS). The level `UiState` computes from it is
//   applied only through the I2S speaker's `set_volume`, which sets the DAC: at boot from
//   `packages/player.yaml`'s `on_boot` below priority -100 (after `packages/device.yaml`'s own
//   level at -100), at each `stream_start` before its audio, on every dial change and on every
//   change of ringing. No sample is ever scaled for volume.
// - Models and LEDs follow `UiState`: the name model while connected and unmuted, the stop model
//   while she speaks or a ring plays, the ring's effect per `UiState::led()`.
// - Inputs come from `packages/controls.yaml`: the button's presses (`on_button`, which sends
//   `interrupt` itself and never touches the uplink), the dial (`on_dial`) and the stop model's
//   detections (`on_stop_detected`, beside `MistPrewake::on_stop_detected`).
//
// For the alarm component (plan T7): `set_ringing`, `set_ring_floor`, and the `ButtonAction`
// that `on_button` and `on_stop_detected` return (`StopRing`, `SnoozeRing`).
//
// Threading: everything runs on the main loop except the speaker's output callback, which only
// adds to an atomic counter.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "esphome/components/binary_sensor/binary_sensor.h"
#include "esphome/components/light/light_state.h"
#include "esphome/components/micro_wake_word/micro_wake_word.h"
#include "esphome/components/mist_core/codec.h"
#include "esphome/components/mist_core/json_lite.h"
#include "esphome/components/mist_core/tag_queue.h"
#include "esphome/components/mist_core/ui_state.h"
#include "esphome/components/mist_link/mist_link.h"
#include "esphome/components/mixer/speaker/mixer_speaker.h"
#include "esphome/components/speaker/speaker.h"
#include "esphome/core/component.h"
#include "esphome/core/preferences.h"

namespace esphome {
namespace mist_player {

class MistPlayer : public Component, public mist_link::LinkListener {
 public:
  static constexpr uint32_t kSampleRate = mist_core::kDownlinkSampleRate;
  // The server keeps at most `max_lead_ms` (500 ms) of a reply ahead of the position the device
  // reports, so 2 s holds the prebuffer, the lead and network jitter.
  static constexpr size_t kFifoSamples = kSampleRate * 2;
  // The most audio the player keeps in the speaker chain ahead of the DAC: enough to ride out a
  // slow pass of the main loop, little enough that a stop's fade is heard soon.
  static constexpr uint32_t kLeadMs = 200;
  static constexpr uint32_t kLeadSamples = kSampleRate * kLeadMs / 1000;
  // The preference the user volume is kept under.
  static constexpr const char *kVolumeKey = "mist_player_volume";

  void set_link(mist_link::MistLink *link) { this->link_ = link; }
  void set_speaker(speaker::Speaker *output) { this->speaker_ = output; }
  void set_ducking_speaker(mixer_speaker::SourceSpeaker *input) { this->ducking_ = input; }
  void set_volume_speaker(speaker::Speaker *output) { this->volume_speaker_ = output; }
  void set_wake_model(micro_wake_word::WakeWordModel *model) { this->wake_model_ = model; }
  void set_stop_model(micro_wake_word::WakeWordModel *model) { this->stop_model_ = model; }
  void set_mute_switch(binary_sensor::BinarySensor *mute_switch) { this->mute_switch_ = mute_switch; }
  void set_leds(light::LightState *leds) { this->leds_ = leds; }

  void setup() override;
  void loop() override;
  void dump_config() override;
  // After the speakers (whose output callback it registers) and the link.
  float get_setup_priority() const override { return setup_priority::AFTER_CONNECTION; }

  // `packages/player.yaml`'s `on_boot` below -100: the level, both models and the LEDs.
  void apply_boot_state();

  // `packages/controls.yaml`'s inputs.
  mist_core::ButtonAction on_button(mist_core::ButtonEvent event);
  mist_core::ButtonAction on_stop_detected();
  void on_dial(int steps);

  // The alarm component's entries: the LEDs, the stop model and the level follow the ring.
  void set_ringing(bool ringing);
  void set_ring_floor(float floor);
  bool is_ringing() const { return this->ui_.ringing(); }
  float volume() const { return this->ui_.volume(); }

  void on_link_connected() override;
  void on_link_disconnected() override;
  void on_link_message(const mist_core::Inbound &message, const mist_core::json::Doc &doc) override;
  void on_link_frame(const mist_core::Frame &frame) override;

 protected:
  void apply_level_();
  void refresh_ui_(bool force);
  void set_model_(micro_wake_word::WakeWordModel *model, bool enabled, bool force);
  void show_leds_(bool force);

  void start_turn_(const mist_core::StreamStartInfo &info);
  void on_audio_(const mist_core::Frame &frame);
  void begin_stop_(const mist_core::StopDecision &decision);
  void halt_output_();
  size_t push_(const uint8_t *payload, size_t samples);
  void feed_speaker_();
  void send_report_(const mist_core::PlaybackReport &report);

  mist_link::MistLink *link_{nullptr};
  speaker::Speaker *speaker_{nullptr};
  mixer_speaker::SourceSpeaker *ducking_{nullptr};
  speaker::Speaker *volume_speaker_{nullptr};
  micro_wake_word::WakeWordModel *wake_model_{nullptr};
  micro_wake_word::WakeWordModel *stop_model_{nullptr};
  binary_sensor::BinarySensor *mute_switch_{nullptr};
  light::LightState *leds_{nullptr};

  mist_core::UiState ui_;
  mist_core::TagQueue queue_;
  ESPPreferenceObject volume_pref_;
  bool led_shown_{false};
  mist_core::LedPattern led_pattern_{mist_core::LedPattern::Idle};

  // The FIFO of the current turn's samples, in PSRAM.
  int16_t *fifo_{nullptr};
  size_t head_{0};  // the next sample to hand the speaker
  size_t used_{0};
  // The speaker took only the low byte of the sample at `head_`.
  bool half_sample_{false};

  bool speaker_active_{false};  // started for the current turn and not yet stopped
  bool speaker_ran_{false};     // seen running since that start
  bool ending_{false};          // no more audio comes for this output: complete, or stopping
  bool finishing_{false};       // `finish()` sent; waiting for the speaker to stop
  uint32_t in_speaker_{0};      // samples handed to the speaker and not yet played
  uint32_t overflow_samples_{0};
  std::atomic<uint32_t> played_frames_{0};
};

}  // namespace mist_player
}  // namespace esphome
