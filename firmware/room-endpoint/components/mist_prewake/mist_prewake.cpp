// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.

#include "mist_prewake.h"

#include <vector>

#include <esp_heap_caps.h>

#include "esphome/core/hal.h"
#include "esphome/core/log.h"

namespace esphome {
namespace mist_prewake {

static const char *const TAG = "mist_prewake";

static const char *close_reason_name(mist_core::CloseReason reason) {
  switch (reason) {
    case mist_core::CloseReason::StateCycle:
      return "state_cycle";
    case mist_core::CloseReason::Decision:
      return "voice_decision";
    case mist_core::CloseReason::Cap:
      return "cap";
    case mist_core::CloseReason::Mute:
      return "mute";
    case mist_core::CloseReason::Disconnect:
      return "disconnection";
    default:
      return "none";
  }
}

static const char *wake_outcome_name(mist_core::WakeOutcome outcome) {
  switch (outcome) {
    case mist_core::WakeOutcome::Opened:
      return "opened";
    case mist_core::WakeOutcome::NotConnected:
      return "not connected";
    case mist_core::WakeOutcome::Muted:
      return "muted";
    case mist_core::WakeOutcome::VoiceInactive:
      return "voice inactive";
    case mist_core::WakeOutcome::AlreadyOpen:
      return "already open";
    case mist_core::WakeOutcome::SendFailed:
      return "link queue full";
  }
  return "unknown";
}

void MistPrewake::setup() {
  if (this->link_ == nullptr || this->microphone_source_ == nullptr || this->wake_model_ == nullptr) {
    ESP_LOGE(TAG, "Missing the link, microphone or wake model");
    this->mark_failed();
    return;
  }

  // The ring in PSRAM: 16 kHz mono PCM16, `ring_ms` of whole 20 ms frames.
  this->ring_bytes_ = static_cast<uint32_t>(mist_core::PcmRing::bytes_for_ms(this->ring_ms_));
  void *storage = heap_caps_malloc(this->ring_bytes_, MALLOC_CAP_SPIRAM);
  if (storage == nullptr) {
    ESP_LOGE(TAG, "Could not allocate the %u-byte ring in PSRAM", (unsigned) this->ring_bytes_);
    this->mark_failed();
    return;
  }
  this->ring_ = new mist_core::PcmRing(storage, this->ring_bytes_);  // NOLINT: lives as long as the device
  if (!this->ring_->valid()) {
    ESP_LOGE(TAG, "The ring's storage is unusable");
    this->mark_failed();
    return;
  }

  mist_core::UplinkConfig config;
  config.keyword = "mist";
  config.backlog_ms = this->backlog_ms_;
  config.cap_ms = this->uplink_cap_ms_;
  this->uplink_ = new mist_core::Uplink(*this->ring_, *this->link_, config);  // NOLINT: as the ring

  // The tap: the ring's only producer, on the microphone's task. The source delivers 16-bit mono.
  mist_core::PcmRing *ring = this->ring_;
  this->microphone_source_->add_data_callback([ring](const std::vector<uint8_t> &data) {
    ring->write(reinterpret_cast<const int16_t *>(data.data()), data.size() / sizeof(int16_t));
  });
  this->microphone_source_->start();

  this->link_->add_listener(this);

  if (this->mute_switch_ != nullptr) {
    if (this->mute_switch_->has_state()) {
      this->on_mute_(this->mute_switch_->state);
    }
    this->mute_switch_->add_on_state_callback([this](bool state) { this->on_mute_(state); });
  }
}

void MistPrewake::loop() {
  if (this->uplink_ == nullptr) {
    return;
  }
  const bool was_open = this->uplink_->is_open();
  this->uplink_->pump(millis());
  this->log_close_(was_open);

  const uint32_t overruns = this->ring_->overruns();
  if (overruns != this->reported_overruns_) {
    this->link_->add_ring_overruns(overruns - this->reported_overruns_);
    this->reported_overruns_ = overruns;
  }
}

void MistPrewake::dump_config() {
  ESP_LOGCONFIG(TAG,
                "MIST pre-wake ring and uplink:\n"
                "  Ring: %u ms, %u bytes in PSRAM\n"
                "  Backlog: %u ms\n"
                "  Uplink cap: %u ms\n"
                "  Mute switch: %s",
                (unsigned) this->ring_ms_, (unsigned) this->ring_bytes_, (unsigned) this->backlog_ms_,
                (unsigned) this->uplink_cap_ms_, YESNO(this->mute_switch_ != nullptr));
}

void MistPrewake::on_wake_word_detected(const std::string &wake_word) {
  if (this->uplink_ == nullptr || wake_word != this->wake_model_->get_wake_word()) {
    return;  // only the name model opens the uplink
  }
  // The trigger carries no probability, so `score` is the model's cutoff (README, "wake.score").
  const float score = static_cast<float>(this->wake_model_->get_probability_cutoff()) / 255.0f;
  const mist_core::WakeOutcome outcome = this->uplink_->on_wake(score, millis());
  if (outcome == mist_core::WakeOutcome::Opened) {
    ESP_LOGI(TAG, "Wake: uplink open, backlog %u ms", (unsigned) this->uplink_->last_backlog_ms());
  } else {
    ESP_LOGW(TAG, "Wake: uplink not opened (%s)", wake_outcome_name(outcome));
  }
}

void MistPrewake::on_link_connected() {
  if (this->uplink_ != nullptr) {
    this->uplink_->on_connected();
  }
}

void MistPrewake::on_link_disconnected() {
  if (this->uplink_ == nullptr) {
    return;
  }
  const bool was_open = this->uplink_->is_open();
  this->uplink_->on_disconnected();
  this->log_close_(was_open);
}

void MistPrewake::on_link_message(const mist_core::Inbound &message, const mist_core::json::Doc & /*doc*/) {
  if (this->uplink_ == nullptr) {
    return;
  }
  const bool was_open = this->uplink_->is_open();
  switch (message.type) {
    case mist_core::InboundType::VoiceState:
      this->uplink_->on_voice_state(message.voice_state.active);
      break;
    case mist_core::InboundType::StateCycle:
      this->uplink_->on_state_cycle(message.state_cycle.state, message.state_cycle.turn_id);
      break;
    case mist_core::InboundType::VoiceDecision:
      this->uplink_->on_voice_decision(message.decision.kind);
      break;
    default:
      break;
  }
  this->log_close_(was_open);
}

void MistPrewake::on_mute_(bool muted) {
  if (this->uplink_ == nullptr) {
    return;
  }
  // Set first, so a `voice_start` queued from here on carries the new state.
  this->link_->set_muted(muted);
  const bool was_open = this->uplink_->is_open();
  if (!this->uplink_->on_mute(muted)) {
    ESP_LOGW(TAG, "Could not queue mute; retrying");
  }
  ESP_LOGI(TAG, "Mute switch %s; ring cleared", muted ? "on" : "off");
  this->log_close_(was_open);
}

void MistPrewake::log_close_(bool was_open) {
  if (was_open && !this->uplink_->is_open()) {
    ESP_LOGI(TAG, "Uplink closed (%s), next seq %u", close_reason_name(this->uplink_->last_close()),
             (unsigned) this->uplink_->next_seq());
  }
}

}  // namespace mist_prewake
}  // namespace esphome
