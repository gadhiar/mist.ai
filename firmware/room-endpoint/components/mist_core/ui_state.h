// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.
//
// Controls and lights (plan Design item 7). The LEDs follow, by priority, a ring, the mute
// switch, a lost connection, then her `state_cycle` state. The name model runs while connected
// and unmuted; the "stop" model while unmuted and either a ring plays or she speaks. A single
// press stops a ring or, while she speaks, interrupts her; a double press snoozes a ring. "Stop"
// said during a ring stops it; said during her speech it is the uplink's `barge_in`, not an
// action here. No button event opens the microphone: there is no push-to-talk.
//
// The dial (plan v4, A13) sets the user volume `v`, from 0 to 1, one step of 0.05 per detent,
// held to the range; with nothing saved `v` is `kDefaultVolume`. The speaker level the player
// hands the I2S speaker's `set_volume` is `0.4 + 0.45 * v`, the published Voice PE config's
// remap (-28.5 dB to +10.5 dB on the AIC3204, whose byte is `-127 + level * 175` in 0.5 dB
// steps), and 0, which mutes the DAC, when `v` is 0. While a ring plays the level is that of
// `max(v, ring floor)`, capped at `kUnityLevel` (byte 0, 0 dB), so a near-full-scale tone never
// clips. `kDefaultVolume` is the `v` whose level writes byte 0: the loudest level at which no
// PCM16 input clips in the DAC.
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

// What a press, or the stop word, asks for. `Interrupt` goes out as the `interrupt` control
// message; none of these touches the uplink.
enum class ButtonAction : uint8_t { None, StopRing, SnoozeRing, Interrupt };

class UiState {
 public:
  // One dial detent.
  static constexpr float kDialStep = 0.05f;
  // For `v` above 0 the level is `kLevelFloor + kLevelSpan * v`.
  static constexpr float kLevelFloor = 0.4f;
  static constexpr float kLevelSpan = 0.45f;
  // The level that writes AIC3204 byte 0 (0 dB): 127 / 175.
  static constexpr float kUnityLevel = 127.0f / 175.0f;
  // With nothing saved: level 0.7258, byte 0. Any `v` from 0.711 to 0.737 writes byte 0; this
  // one sits inside that span with room for float error on either side.
  static constexpr float kDefaultVolume = 0.724f;
  // `${ring_volume_floor}`'s default, on the same scale as `v`.
  static constexpr float kDefaultRingFloor = 0.6f;

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
  // A detection by the stop model. During a ring it stops the ring; otherwise nothing here (the
  // uplink sends `barge_in` while she speaks).
  ButtonAction on_stop_word() const;

  // The user volume `v`, from 0 to 1; anything outside, or NaN, is held to the range.
  void set_volume(float volume) { volume_ = clamp(volume); }
  float volume() const { return volume_; }
  // The volume read back from preferences at boot: `found` false or a NaN keeps the default.
  void restore_volume(bool found, float saved);
  // Moves the volume by `steps` dial detents and returns the new value.
  float on_dial(int steps);

  // The quietest a ring plays, on the scale of `v`; held to 0 to 1.
  void set_ring_floor(float floor) { ring_floor_ = clamp(floor); }
  float ring_floor() const { return ring_floor_; }

  // The level for the I2S speaker's `set_volume`, from the volume and, during a ring, the floor.
  float speaker_level() const;

  bool connected() const { return connected_; }
  bool muted() const { return muted_; }
  bool ringing() const { return ringing_; }
  MistState state() const { return state_; }

 private:
  static float clamp(float volume);
  static float level_for(float volume);

  bool connected_ = false;
  bool muted_ = false;
  bool ringing_ = false;
  MistState state_ = MistState::Idle;
  float volume_ = kDefaultVolume;
  float ring_floor_ = kDefaultRingFloor;
};

}  // namespace mist_core
