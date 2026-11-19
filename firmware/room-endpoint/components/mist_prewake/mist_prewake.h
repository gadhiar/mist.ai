// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.
//
// The pre-wake ring and the wake uplink on the device (plan Design item 4): the glue between
// ESPHome's microphone, `micro_wake_word` and mute switch, `mist_core`'s ring and uplink state
// machine, and `mist_link`.
//
// - A `MicrophoneSource` tap (channel 0, gain 1) writes every sample into a `ring_ms` ring in
//   PSRAM. The tap runs on the microphone's task and is the ring's only producer.
// - A detection whose word is the name model's opens the uplink, which sends `wake`, the backlog
//   oldest first, then live frames through the link. A detection by any other model opens
//   nothing. Before a detection nothing leaves the device (rule 5).
// - `voice_state`, `state_cycle` and `voice_decision` from the link, the cap, mute and
//   disconnection close it, as `mist_core::Uplink` decides.
// - The mute switch clears the ring, closes any uplink and sends `mute`; the link's next
//   `voice_start` carries the state too.
//
// Threading: everything here except the tap runs on the main loop, the ring's only consumer.

#pragma once

#include <cstdint>
#include <string>

#include "esphome/components/binary_sensor/binary_sensor.h"
#include "esphome/components/micro_wake_word/micro_wake_word.h"
#include "esphome/components/microphone/microphone_source.h"
#include "esphome/components/mist_core/codec.h"
#include "esphome/components/mist_core/json_lite.h"
#include "esphome/components/mist_core/ring.h"
#include "esphome/components/mist_core/uplink.h"
#include "esphome/components/mist_link/mist_link.h"
#include "esphome/core/component.h"

namespace esphome {
namespace mist_prewake {

class MistPrewake : public Component, public mist_link::LinkListener {
 public:
  void set_link(mist_link::MistLink *link) { this->link_ = link; }
  void set_microphone_source(microphone::MicrophoneSource *source) { this->microphone_source_ = source; }
  void set_wake_model(micro_wake_word::WakeWordModel *model) { this->wake_model_ = model; }
  void set_mute_switch(binary_sensor::BinarySensor *mute_switch) { this->mute_switch_ = mute_switch; }
  void set_ring_ms(uint32_t ring_ms) { this->ring_ms_ = ring_ms; }
  void set_backlog_ms(uint32_t backlog_ms) { this->backlog_ms_ = backlog_ms; }
  void set_uplink_cap_ms(uint32_t cap_ms) { this->uplink_cap_ms_ = cap_ms; }

  void setup() override;
  void loop() override;
  void dump_config() override;
  // After the microphone (which `setup` starts) and the link (which it registers with).
  float get_setup_priority() const override { return setup_priority::AFTER_CONNECTION; }

  // `micro_wake_word`'s `on_wake_word_detected`, which passes only the detected model's word.
  void on_wake_word_detected(const std::string &wake_word);
  // A detection by the stop model (`packages/controls.yaml`). While she speaks the uplink sends
  // `barge_in {turn_id, keyword: "stop"}` at most once per turn; it never opens the uplink.
  void on_stop_detected();

  void on_link_connected() override;
  void on_link_disconnected() override;
  void on_link_message(const mist_core::Inbound &message, const mist_core::json::Doc &doc) override;

 protected:
  void on_mute_(bool muted);
  void log_close_(bool was_open);

  mist_link::MistLink *link_{nullptr};
  microphone::MicrophoneSource *microphone_source_{nullptr};
  micro_wake_word::WakeWordModel *wake_model_{nullptr};
  binary_sensor::BinarySensor *mute_switch_{nullptr};
  uint32_t ring_ms_{6000};
  uint32_t backlog_ms_{5000};
  uint32_t uplink_cap_ms_{30000};

  mist_core::PcmRing *ring_{nullptr};
  mist_core::Uplink *uplink_{nullptr};
  uint32_t ring_bytes_{0};
  uint32_t reported_overruns_{0};
};

}  // namespace mist_prewake
}  // namespace esphome
