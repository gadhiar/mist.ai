// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.

#include "tag_queue.h"

#include <cstring>

namespace mist_core {

bool TagQueue::on_stream_start(const char* turn_id, bool has_audio_tag, uint16_t audio_tag) {
  const bool flush =
      phase_ == Phase::Buffering || phase_ == Phase::Playing || phase_ == Phase::Draining;
  const size_t len = turn_id == nullptr ? kIdCap : std::strlen(turn_id);
  if (len < kIdCap) {
    std::memcpy(turn_id_, turn_id, len + 1);
  } else {
    turn_id_[0] = '\0';
  }
  phase_ = has_audio_tag && audio_tag != 0 ? Phase::Buffering : Phase::None;
  tag_ = phase_ == Phase::Buffering ? audio_tag : 0;
  queued_ = 0;
  played_ = 0;
  started_ = false;
  final_report_ = false;
  duck_db_ = 0.0f;
  return flush;
}

TagVerdict TagQueue::on_audio_chunk(uint16_t tag, uint32_t samples) {
  if (tag != tag_ || (phase_ != Phase::Buffering && phase_ != Phase::Playing)) {
    return TagVerdict::Drop;
  }
  queued_ += samples;
  if (phase_ == Phase::Buffering &&
      queued_ * 1000u >= static_cast<uint64_t>(params_.prebuffer_ms) * kSampleRate) {
    phase_ = Phase::Playing;
  }
  return TagVerdict::Play;
}

bool TagQueue::on_audio_complete(uint16_t tag) {
  if (tag != tag_ || (phase_ != Phase::Buffering && phase_ != Phase::Playing)) return false;
  phase_ = Phase::Draining;
  return true;
}

StopDecision TagQueue::on_interrupt_fade(uint16_t tag) { return stop(tag, params_.fade_ms); }

StopDecision TagQueue::on_playback_stop(uint16_t tag, uint32_t fade_ms) {
  return stop(tag, fade_ms);
}

StopDecision TagQueue::stop(uint16_t tag, uint32_t fade_ms) {
  if (!current(tag)) return StopDecision{false, 0};
  phase_ = Phase::Stopped;
  final_report_ = started_;
  return StopDecision{true, fade_ms};
}

bool TagQueue::on_playback_duck(const char* turn_id, float db) {
  if (turn_id == nullptr || turn_id_[0] == '\0' || std::strcmp(turn_id, turn_id_) != 0) {
    return false;
  }
  duck_db_ = db < 0.0f ? db : 0.0f;
  return true;
}

void TagQueue::on_played(uint32_t samples, uint32_t now_ms) {
  if (!playing()) return;
  if (!started_) {
    started_ = true;
    last_report_ms_ = now_ms;
  }
  played_ += samples;
  if (played_ > queued_) played_ = queued_;
}

PlaybackReport TagQueue::poll(uint32_t now_ms) {
  if (final_report_) {
    final_report_ = false;
    return report(ReportKind::Progress);
  }
  if (phase_ == Phase::Draining && played_ >= queued_) {
    phase_ = Phase::Finished;
    return report(ReportKind::Done);
  }
  if (playing() && started_ && now_ms - last_report_ms_ >= params_.progress_interval_ms) {
    last_report_ms_ = now_ms;
    return report(ReportKind::Progress);
  }
  return report(ReportKind::None);
}

uint32_t TagQueue::played_ms() const {
  return static_cast<uint32_t>(played_ * 1000u / kSampleRate);
}

PlaybackReport TagQueue::report(ReportKind kind) const {
  PlaybackReport out{};
  out.kind = kind;
  if (kind != ReportKind::None) {
    std::memcpy(out.turn_id, turn_id_, sizeof(out.turn_id));
    out.played_ms = played_ms();
  }
  return out;
}

}  // namespace mist_core
