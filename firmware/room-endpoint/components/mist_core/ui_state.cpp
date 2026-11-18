// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.

#include "ui_state.h"

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

float UiState::on_dial(float delta) {
  volume_ = clamp(volume_ + delta);
  return volume_;
}

float UiState::clamp(float volume) {
  if (!(volume > 0.0f)) return 0.0f;  // also NaN
  if (volume > 1.0f) return 1.0f;
  return volume;
}

}  // namespace mist_core
