// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.
//
// Controls and lights (plan Design item 7). The LEDs follow, by priority, a ring, the mute
// switch, a lost connection, then her `state_cycle` state. The name model runs while connected
// and unmuted; the "stop" model while unmuted and either a ring plays or she speaks. A single
// press stops a ring or, while she speaks, interrupts her; a double press snoozes a ring. No
// button event opens the microphone: there is no push-to-talk. The dial sets the volume.
//
// Threading: one task calls every method.

#pragma once

#include <cstdint>

#include "codec.h"

namespace mist_core {

enum class LedPattern : uint8_t {
  Disconnected,
  Muted,
  Idle,
  Attend,
  Listen,
  Think,
  Speak,
  Ringing,
};

enum class ButtonEvent : uint8_t { SinglePress, DoublePress, LongPress };

// What a press asks for. `Interrupt` goes out as the `interrupt` control message; none of these
// touches the uplink.
enum class ButtonAction : uint8_t { None, StopRing, SnoozeRing, Interrupt };

class UiState {
 public:
  // A lost connection forgets her state: a new session starts idle.
  void set_connected(bool connected);
  void set_muted(bool muted) { muted_ = muted; }
  void set_state(MistState state) { state_ = state; }
  void set_ringing(bool ringing) { ringing_ = ringing; }

  LedPattern led() const;
  bool wake_model_enabled() const { return connected_ && !muted_; }
  bool stop_model_enabled() const {
    return !muted_ && (ringing_ || (connected_ && state_ == MistState::Speak));
  }

  ButtonAction on_button(ButtonEvent event) const;

  // Volume from 0 to 1; anything outside is held to the range.
  void set_volume(float volume) { volume_ = clamp(volume); }
  float volume() const { return volume_; }
  // Moves the volume by `delta` and returns the new value.
  float on_dial(float delta);

  bool connected() const { return connected_; }
  bool muted() const { return muted_; }
  bool ringing() const { return ringing_; }
  MistState state() const { return state_; }

 private:
  static float clamp(float volume);

  bool connected_ = false;
  bool muted_ = false;
  bool ringing_ = false;
  MistState state_ = MistState::Idle;
  float volume_ = 0.5f;
};

}  // namespace mist_core
