// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.
//
// The playback tag rules (plan Design item 5), in front of the player's resampler.
//
// `stream_start.audio_tag` names the turn whose `0x01` frames play; frames of any other tag, and
// frames before any `stream_start`, are dropped. A turn starts playing once `prebuffer_ms` of it
// is queued, or at its `0x02` if it is shorter. While it plays the device reports
// `playback_progress` every `progress_interval_ms`, and `playback_done {turn_id, played_ms}` once
// its audio after `0x02` has all been played. `playback_stop` or `0x03` for the current tag fades
// and flushes it, reports the position reached, and never reports it done. `playback_duck` lowers
// the current turn by its `db`, and a new turn restores the level.
//
// The queue counts samples; the player holds the PCM, renders the fade with `apply_fade_out`, and
// hands the duck to the mixer as `duck_reduction_db`. Threading: one task calls every method.

#pragma once

#include <cstddef>
#include <cstdint>

#include "codec.h"

namespace mist_core {

// Scales `count` samples in place by a linear fade to silence: the samples sit at `offset` within
// a fade of `total` samples, and a sample at position `p` keeps `(total - p) / total` of its
// level, so the fade starts at full level and its last sample is the quietest step above
// silence. Samples at or past `total` become silence. A fade may be applied in pieces.
void apply_fade_out(int16_t* pcm, size_t count, size_t offset, size_t total);

enum class TagVerdict : uint8_t { Play, Drop };

struct StopDecision {
  bool fade;  // fade over `fade_ms`, then flush the turn's queued audio
  uint32_t fade_ms;
};

enum class ReportKind : uint8_t { None, Progress, Done };

// A report to send: `playback_progress` or `playback_done` with these fields.
struct PlaybackReport {
  ReportKind kind;
  char turn_id[kIdCap];
  uint32_t played_ms;
};

class TagQueue {
 public:
  static constexpr uint32_t kSampleRate = kDownlinkSampleRate;

  // The session's `voice_state.playback`; applies from the next decision on.
  void set_params(const PlaybackParams& params) { params_ = params; }
  const PlaybackParams& params() const { return params_; }

  // Forgets the current turn without any report: the connection that carried it is gone.
  void reset();

  // A new turn. Returns true when an earlier turn had audio not yet played out, which the player
  // flushes: the new turn takes over.
  bool on_stream_start(const char* turn_id, bool has_audio_tag, uint16_t audio_tag);

  // A `0x01` frame of `samples` samples.
  TagVerdict on_audio_chunk(uint16_t tag, uint32_t samples);
  // The player had no room for the last `samples` samples of a chunk it was told to play; they
  // are not queued, so the turn can still play out. Never unqueues what was already played.
  void on_overflow(uint32_t samples);
  // A `0x02`: the turn's last frame is queued. Returns whether it named the current turn.
  bool on_audio_complete(uint16_t tag);
  // A `0x03`, which fades by the session's `fade_ms`.
  StopDecision on_interrupt_fade(uint16_t tag);
  // A `playback_stop`, which fades by its own `fade_ms`.
  StopDecision on_playback_stop(uint16_t tag, uint32_t fade_ms);
  // A `playback_duck`; returns whether it named the current turn. A positive `db` never raises
  // the level above the user's volume.
  bool on_playback_duck(const char* turn_id, float db);

  // Whether the player should be rendering the current turn.
  bool playing() const { return phase_ == Phase::Playing || phase_ == Phase::Draining; }

  // The player rendered `samples` more samples of the current turn at `now_ms`, a free-running
  // millisecond clock that may wrap.
  void on_played(uint32_t samples, uint32_t now_ms);
  // The output has played everything it was given after the `0x02`: every queued sample of the
  // completed turn counts as played, even where the output's own count fell short (a resampler
  // may round frames away). Does nothing before the `0x02`.
  void on_output_drained(uint32_t now_ms);

  // The report due at `now_ms`, if any.
  PlaybackReport poll(uint32_t now_ms);

  float duck_db() const { return duck_db_; }
  // The duck as the mixer's `apply_ducking` takes it: whole decibels of reduction, 0 to 255.
  uint8_t duck_reduction_db() const;
  uint64_t queued_samples() const { return queued_; }
  uint64_t played_samples() const { return played_; }

 private:
  enum class Phase : uint8_t { None, Buffering, Playing, Draining, Finished, Stopped };

  bool current(uint16_t tag) const {
    return tag == tag_ && (phase_ == Phase::Buffering || phase_ == Phase::Playing ||
                           phase_ == Phase::Draining);
  }
  StopDecision stop(uint16_t tag, uint32_t fade_ms);
  uint32_t played_ms() const;
  PlaybackReport report(ReportKind kind) const;

  PlaybackParams params_;
  Phase phase_ = Phase::None;
  char turn_id_[kIdCap] = {};
  uint16_t tag_ = 0;
  uint64_t queued_ = 0;
  uint64_t played_ = 0;
  bool started_ = false;
  uint32_t last_report_ms_ = 0;
  bool final_report_ = false;
  float duck_db_ = 0.0f;
};

}  // namespace mist_core
