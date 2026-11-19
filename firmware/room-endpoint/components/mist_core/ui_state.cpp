// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.

#include "ui_state.h"

#include <cmath>

namespace mist_core {

void UiState::set_connected(bool connected) {
  connected_ = connected;
  if (!connected) state_ = MistState::Idle;
}

LedPattern UiState::led() const {
  if (ringing_) return LedPattern::Ringing;
  if (muted_) return LedPattern::Muted;
  if (!connected_) return LedPattern::Disconnected;
  switch (state_) {
    case MistState::Attend:
      return LedPattern::Attend;
    case MistState::Listen:
      return LedPattern::Listen;
    case MistState::Think:
      return LedPattern::Think;
    case MistState::Speak:
      return LedPattern::Speak;
    case MistState::Idle:
      break;
  }
  return LedPattern::Idle;
}

ButtonAction UiState::on_button(ButtonEvent event) const {
  if (ringing_) {
    if (event == ButtonEvent::SinglePress) return ButtonAction::StopRing;
    if (event == ButtonEvent::DoublePress) return ButtonAction::SnoozeRing;
    return ButtonAction::None;
  }
  if (event == ButtonEvent::SinglePress && connected_ && state_ == MistState::Speak) {
    return ButtonAction::Interrupt;
  }
  return ButtonAction::None;
}

ButtonAction UiState::on_stop_word() const {
  return ringing_ ? ButtonAction::StopRing : ButtonAction::None;
}

void UiState::restore_volume(bool found, float saved) {
  volume_ = found && !std::isnan(saved) ? clamp(saved) : kDefaultVolume;
}

float UiState::on_dial(int steps) {
  volume_ = clamp(volume_ + static_cast<float>(steps) * kDialStep);
  return volume_;
}

float UiState::speaker_level() const {
  if (!ringing_) return level_for(volume_);
  const float ring_volume = volume_ > ring_floor_ ? volume_ : ring_floor_;
  const float level = level_for(ring_volume);
  return level < kUnityLevel ? level : kUnityLevel;
}

float UiState::clamp(float volume) {
  if (!(volume > 0.0f)) return 0.0f;  // also NaN
  if (volume > 1.0f) return 1.0f;
  return volume;
}

float UiState::level_for(float volume) {
  // The speaker mutes the DAC below a level of 0.001 (speaker.h), so 0 is silence.
  if (!(volume > 0.0f)) return 0.0f;
  return kLevelFloor + kLevelSpan * volume;
}

}  // namespace mist_core
