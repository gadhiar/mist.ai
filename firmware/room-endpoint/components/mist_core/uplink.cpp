// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.

#include "uplink.h"

#include <cstring>

namespace mist_core {

namespace {

// An id too long for the wire buffers is dropped rather than cut: a cut id names another turn.
void copy_id(char* out, const char* id) {
  const size_t len = id == nullptr ? kIdCap : std::strlen(id);
  if (len >= kIdCap) {
    out[0] = '\0';
    return;
  }
  std::memcpy(out, id, len + 1);
}

}  // namespace

Uplink::Uplink(PcmRing& ring, UplinkSink& sink, const UplinkConfig& config)
    : ring_(ring), sink_(sink), config_(config) {
  if (config_.keyword == nullptr) config_.keyword = "mist";
  if (config_.backlog_ms > kMaxBacklogMs) config_.backlog_ms = kMaxBacklogMs;
}

void Uplink::on_connected() {
  close(CloseReason::Disconnect);
  connected_ = true;
  voice_active_ = false;
  server_muted_ = muted_;
  seq_ = 0;
  state_ = MistState::Idle;
  speaking_turn_[0] = '\0';
  barged_turn_[0] = '\0';
}

void Uplink::on_disconnected() {
  if (open_) close(CloseReason::Disconnect);
  connected_ = false;
  voice_active_ = false;
  state_ = MistState::Idle;
  speaking_turn_[0] = '\0';
}

void Uplink::on_voice_state(bool active) { voice_active_ = active; }

bool Uplink::on_mute(bool muted) {
  if (muted == muted_) return true;
  muted_ = muted;
  if (muted && open_) close(CloseReason::Mute);
  ring_.clear();
  return sync_mute();
}

bool Uplink::sync_mute() {
  if (!connected_ || server_muted_ == muted_) return true;
  char buf[64];
  const size_t len = build_mute(buf, sizeof(buf), muted_);
  if (len == 0 || !sink_.send_text(buf, len)) return false;
  server_muted_ = muted_;
  return true;
}

void Uplink::on_state_cycle(MistState state, const char* turn_id) {
  state_ = state;
  if (state == MistState::Speak) {
    copy_id(speaking_turn_, turn_id);
  } else {
    speaking_turn_[0] = '\0';
  }
  if (open_ && (state == MistState::Speak || state == MistState::Idle)) {
    close(CloseReason::StateCycle);
  }
}

void Uplink::on_voice_decision(DecisionKind kind) {
  if (open_ && (kind == DecisionKind::WakeRejected || kind == DecisionKind::WakeLost ||
                kind == DecisionKind::Dropped)) {
    close(CloseReason::Decision);
  }
}

WakeOutcome Uplink::on_wake(float score, uint32_t now_ms) {
  if (!connected_) return WakeOutcome::NotConnected;
  if (muted_) return WakeOutcome::Muted;
  if (!voice_active_) return WakeOutcome::VoiceInactive;
  if (open_) return WakeOutcome::AlreadyOpen;
  if (!sync_mute()) return WakeOutcome::SendFailed;
  const uint32_t frames = ring_.read_last(config_.backlog_ms);
  const uint32_t backlog_ms = frames * PcmRing::kFrameMs;
  char buf[160];
  const size_t len = build_wake(buf, sizeof(buf), config_.keyword, score, backlog_ms);
  if (len == 0 || !sink_.send_text(buf, len)) return WakeOutcome::SendFailed;
  open_ = true;
  opened_at_ms_ = now_ms;
  last_backlog_ms_ = backlog_ms;
  has_pending_ = false;
  pump(now_ms);
  return WakeOutcome::Opened;
}

StopOutcome Uplink::on_stop_keyword() {
  if (!connected_) return StopOutcome::NotConnected;
  if (muted_) return StopOutcome::Muted;
  if (state_ != MistState::Speak || speaking_turn_[0] == '\0') return StopOutcome::NotSpeaking;
  if (std::strcmp(barged_turn_, speaking_turn_) == 0) return StopOutcome::AlreadySent;
  char buf[160];
  const size_t len = build_barge_in(buf, sizeof(buf), speaking_turn_, "stop");
  if (len == 0 || !sink_.send_text(buf, len)) return StopOutcome::SendFailed;
  copy_id(barged_turn_, speaking_turn_);
  return StopOutcome::BargeInSent;
}

uint32_t Uplink::pump(uint32_t now_ms, uint32_t max_frames) {
  sync_mute();
  if (!open_) return 0;
  // Unsigned subtraction keeps the cap right across the clock's wrap.
  if (now_ms - opened_at_ms_ >= config_.cap_ms) {
    close(CloseReason::Cap);
    return 0;
  }
  uint32_t sent = 0;
  int16_t samples[PcmRing::kFrameSamples];
  while (sent < max_frames) {
    if (!has_pending_) {
      if (!ring_.read_frame(samples)) break;
      pending_len_ =
          encode_mic_frame(seq_, samples, PcmRing::kFrameSamples, pending_, sizeof(pending_));
      has_pending_ = pending_len_ > 0;
      if (!has_pending_) break;
    }
    if (!send_pending()) break;
    ++sent;
  }
  return sent;
}

bool Uplink::send_pending() {
  if (!sink_.send_binary(pending_, pending_len_)) return false;
  has_pending_ = false;
  ++seq_;
  return true;
}

void Uplink::close(CloseReason reason) {
  if (open_) last_close_ = reason;
  open_ = false;
  has_pending_ = false;
}

}  // namespace mist_core
