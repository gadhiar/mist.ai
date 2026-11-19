// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.

#include "mist_player.h"

#include <algorithm>
#include <cstring>

#include <esp_heap_caps.h>

#include "esphome/core/hal.h"
#include "esphome/core/helpers.h"
#include "esphome/core/log.h"

namespace esphome {
namespace mist_player {

static const char *const TAG = "mist_player";

static constexpr size_t kSampleBytes = sizeof(int16_t);

// How the ring shows each pattern: an effect of `mist_leds` (packages/controls.yaml), a colour and
// a brightness. Idle turns the ring off.
struct LedLook {
  const char *effect;
  float red;
  float green;
  float blue;
  float brightness;
};

// MIST's states in the published config's default ring colour; mute and a lost link in red.
static constexpr float kBlueR = 0.094f;
static constexpr float kBlueG = 0.733f;
static constexpr float kBlueB = 0.949f;

static LedLook look_for(mist_core::LedPattern pattern) {
  switch (pattern) {
    case mist_core::LedPattern::Disconnected:
      return {"Disconnected", 1.0f, 0.0f, 0.0f, 0.66f};
    case mist_core::LedPattern::Muted:
      return {"Muted", 1.0f, 0.0f, 0.0f, 0.66f};
    case mist_core::LedPattern::Attend:
      return {"Attend", kBlueR, kBlueG, kBlueB, 0.66f};
    case mist_core::LedPattern::Listen:
      return {"Listen", kBlueR, kBlueG, kBlueB, 0.66f};
    case mist_core::LedPattern::Think:
      return {"Think", kBlueR, kBlueG, kBlueB, 0.66f};
    case mist_core::LedPattern::Speak:
      return {"Speak", kBlueR, kBlueG, kBlueB, 0.66f};
    case mist_core::LedPattern::Ringing:
      return {"Ringing", kBlueR, kBlueG, kBlueB, 1.0f};
    case mist_core::LedPattern::Idle:
      break;
  }
  return {nullptr, 0.0f, 0.0f, 0.0f, 0.0f};
}

static const char *action_name(mist_core::ButtonAction action) {
  switch (action) {
    case mist_core::ButtonAction::None:
      return "nothing";
    case mist_core::ButtonAction::StopRing:
      return "stop the ring";
    case mist_core::ButtonAction::SnoozeRing:
      return "snooze the ring";
    case mist_core::ButtonAction::Interrupt:
      return "interrupt";
  }
  return "unknown";
}

void MistPlayer::setup() {
  if (this->link_ == nullptr || this->speaker_ == nullptr || this->ducking_ == nullptr ||
      this->volume_speaker_ == nullptr || this->wake_model_ == nullptr || this->stop_model_ == nullptr ||
      this->leds_ == nullptr) {
    ESP_LOGE(TAG, "Missing the link, a speaker, a wake model or the LEDs");
    this->mark_failed();
    return;
  }
  this->fifo_ = static_cast<int16_t *>(heap_caps_malloc(kFifoSamples * kSampleBytes, MALLOC_CAP_SPIRAM));
  if (this->fifo_ == nullptr) {
    ESP_LOGE(TAG, "Could not allocate the reply FIFO in PSRAM");
    this->mark_failed();
    return;
  }

  this->volume_pref_ = global_preferences->make_preference<float>(fnv1_hash(kVolumeKey));
  float saved = 0.0f;
  const bool found = this->volume_pref_.load(&saved);
  this->ui_.restore_volume(found, saved);
  ESP_LOGI(TAG, "Volume %.2f (%s)", this->ui_.volume(), found ? "saved" : "default");

  // The DAC's played frames, converted to the reply's 24 kHz by the resampler; runs on the
  // speaker's task.
  this->speaker_->add_audio_output_callback(
      [this](uint32_t frames, int64_t /*timestamp*/) { this->played_frames_.fetch_add(frames); });

  this->link_->add_listener(this);

  if (this->mute_switch_ != nullptr) {
    if (this->mute_switch_->has_state()) {
      this->ui_.set_muted(this->mute_switch_->state);
    }
    this->mute_switch_->add_on_state_callback([this](bool state) {
      this->ui_.set_muted(state);
      this->refresh_ui_(false);
    });
  }
}

void MistPlayer::loop() {
  if (this->fifo_ == nullptr) {
    return;
  }
  const uint32_t now = millis();

  // Only an output started for the current turn counts. After `halt_output_()` the mixer source
  // keeps reporting the halted turn's mixed tail until it has played, and a new turn whose
  // prebuffer is already queued (so `playing()`) would otherwise take it as its own.
  const uint32_t played = this->played_frames_.exchange(0);
  if (played > 0 && this->speaker_active_) {
    this->in_speaker_ = played < this->in_speaker_ ? this->in_speaker_ - played : 0;
    this->queue_.on_played(played, now);
  }

  if (!this->speaker_active_ && this->used_ > 0 && this->queue_.playing()) {
    this->played_frames_.store(0);
    this->in_speaker_ = 0;
    this->speaker_->set_audio_stream_info(audio::AudioStreamInfo(16, 1, kSampleRate));
    this->speaker_->start();
    this->speaker_active_ = true;
    this->speaker_ran_ = false;
    this->finishing_ = false;
  }

  if (this->speaker_active_) {
    // The speaker leaves STOPPED on its own loop, after `start()`; until it has run, a stopped
    // state says nothing about the turn.
    if (this->speaker_->is_running()) {
      this->speaker_ran_ = true;
    }
    // An empty chain holds nothing to play, whatever the output callback has counted.
    if (this->in_speaker_ > 0 && this->speaker_ran_ && !this->speaker_->has_buffered_data()) {
      this->in_speaker_ = 0;
    }
    this->feed_speaker_();
    if (this->ending_ && this->used_ == 0 && !this->half_sample_ && !this->finishing_) {
      this->speaker_->finish();
      this->finishing_ = true;
    }
    if (this->finishing_ && this->speaker_ran_ && this->speaker_->is_stopped()) {
      this->speaker_active_ = false;
      this->finishing_ = false;
      this->ending_ = false;
      this->in_speaker_ = 0;
      this->queue_.on_output_drained(now);
    }
  }

  if (this->overflow_samples_ > 0) {
    ESP_LOGW(TAG, "Reply FIFO full: dropped %u samples", (unsigned) this->overflow_samples_);
    this->overflow_samples_ = 0;
  }

  this->send_report_(this->queue_.poll(now));
}

void MistPlayer::dump_config() {
  ESP_LOGCONFIG(TAG,
                "MIST player:\n"
                "  Volume: %.2f (speaker level %.3f), ring floor %.2f\n"
                "  Reply FIFO: %u samples in PSRAM, lead %u ms\n"
                "  Mute switch: %s",
                this->ui_.volume(), this->ui_.speaker_level(), this->ui_.ring_floor(), (unsigned) kFifoSamples,
                (unsigned) kLeadMs, YESNO(this->mute_switch_ != nullptr));
}

// -- Level, models and LEDs --------------------------------------------------------------------

void MistPlayer::apply_boot_state() {
  this->apply_level_();
  this->refresh_ui_(true);
  ESP_LOGI(TAG, "Boot: volume %.2f, speaker level %.3f; name model %s, stop model %s", this->ui_.volume(),
           this->ui_.speaker_level(), ONOFF(this->ui_.wake_model_enabled()), ONOFF(this->ui_.stop_model_enabled()));
}

void MistPlayer::apply_level_() {
  if (this->volume_speaker_ == nullptr) {
    return;
  }
  // The DAC's digital volume is the chain's one adjustable gain; samples are never scaled.
  this->volume_speaker_->set_volume(this->ui_.speaker_level());
}

void MistPlayer::refresh_ui_(bool force) {
  this->set_model_(this->wake_model_, this->ui_.wake_model_enabled(), force);
  this->set_model_(this->stop_model_, this->ui_.stop_model_enabled(), force);
  this->show_leds_(force);
}

void MistPlayer::set_model_(micro_wake_word::WakeWordModel *model, bool enabled, bool force) {
  if (model == nullptr || (!force && model->is_enabled() == enabled)) {
    return;
  }
  // Both models are `internal`, so neither call saves to flash.
  if (enabled) {
    model->enable();
  } else {
    model->disable();
  }
  ESP_LOGD(TAG, "Model %s %s", model->get_id().c_str(), enabled ? "enabled" : "disabled");
}

void MistPlayer::show_leds_(bool force) {
  if (this->leds_ == nullptr) {
    return;
  }
  const mist_core::LedPattern pattern = this->ui_.led();
  if (!force && this->led_shown_ && pattern == this->led_pattern_) {
    return;
  }
  this->led_shown_ = true;
  this->led_pattern_ = pattern;
  const LedLook look = look_for(pattern);
  if (look.effect == nullptr) {
    this->leds_->turn_off().perform();
    return;
  }
  auto call = this->leds_->turn_on();
  call.set_red(look.red);
  call.set_green(look.green);
  call.set_blue(look.blue);
  call.set_brightness(look.brightness);
  call.set_effect(look.effect);
  call.perform();
}

// -- Inputs ------------------------------------------------------------------------------------

mist_core::ButtonAction MistPlayer::on_button(mist_core::ButtonEvent event) {
  const mist_core::ButtonAction action = this->ui_.on_button(event);
  if (action == mist_core::ButtonAction::Interrupt) {
    char buf[32];
    const size_t len = mist_core::build_interrupt(buf, sizeof(buf));
    if (len == 0 || !this->link_->send_text(buf, len)) {
      ESP_LOGW(TAG, "Button: could not queue interrupt");
      return action;
    }
  }
  ESP_LOGI(TAG, "Button: %s", action_name(action));
  return action;
}

mist_core::ButtonAction MistPlayer::on_stop_detected() {
  const mist_core::ButtonAction action = this->ui_.on_stop_word();
  ESP_LOGI(TAG, "Stop word: %s", action_name(action));
  return action;
}

void MistPlayer::on_dial(int steps) {
  const float before = this->ui_.volume();
  const float volume = this->ui_.on_dial(steps);
  if (volume != before) {
    // ESPHome writes a changed preference to flash at its next flush.
    this->volume_pref_.save(&volume);
  }
  this->apply_level_();
  ESP_LOGI(TAG, "Volume %.2f, speaker level %.3f", volume, this->ui_.speaker_level());
}

void MistPlayer::set_ringing(bool ringing) {
  if (ringing == this->ui_.ringing()) {
    return;
  }
  this->ui_.set_ringing(ringing);
  this->apply_level_();
  this->refresh_ui_(false);
}

void MistPlayer::set_ring_floor(float floor) {
  this->ui_.set_ring_floor(floor);
  if (this->ui_.ringing()) {
    this->apply_level_();
  }
}

// -- The link ----------------------------------------------------------------------------------

void MistPlayer::on_link_connected() {
  this->ui_.set_connected(true);
  this->refresh_ui_(false);
}

void MistPlayer::on_link_disconnected() {
  // The turn went with the connection: nothing more of it arrives and nobody hears a report.
  this->queue_.reset();
  this->halt_output_();
  this->ui_.set_connected(false);
  this->refresh_ui_(false);
}

void MistPlayer::on_link_message(const mist_core::Inbound &message, const mist_core::json::Doc & /*doc*/) {
  switch (message.type) {
    case mist_core::InboundType::VoiceState:
      this->queue_.set_params(message.voice_state.playback);
      break;
    case mist_core::InboundType::StateCycle:
      this->ui_.set_state(message.state_cycle.state);
      this->refresh_ui_(false);
      break;
    case mist_core::InboundType::StreamStart:
      this->start_turn_(message.stream_start);
      break;
    case mist_core::InboundType::PlaybackStop:
      this->begin_stop_(
          this->queue_.on_playback_stop(message.playback_stop.audio_tag, message.playback_stop.fade_ms));
      break;
    case mist_core::InboundType::PlaybackDuck:
      if (this->queue_.on_playback_duck(message.playback_duck.turn_id, message.playback_duck.db)) {
        this->ducking_->apply_ducking(this->queue_.duck_reduction_db(), this->queue_.params().fade_ms);
      }
      break;
    default:
      break;
  }
}

void MistPlayer::on_link_frame(const mist_core::Frame &frame) {
  switch (frame.type) {
    case mist_core::MsgType::AudioChunk:
      this->on_audio_(frame);
      break;
    case mist_core::MsgType::AudioComplete:
      if (this->queue_.on_audio_complete(frame.turn_tag)) {
        this->ending_ = true;
      }
      break;
    case mist_core::MsgType::InterruptFade:
      this->begin_stop_(this->queue_.on_interrupt_fade(frame.turn_tag));
      break;
    default:
      break;
  }
}

// -- Playback ----------------------------------------------------------------------------------

void MistPlayer::start_turn_(const mist_core::StreamStartInfo &info) {
  if (this->queue_.on_stream_start(info.turn_id, info.has_audio_tag, info.audio_tag)) {
    ESP_LOGD(TAG, "A new turn flushes the one playing");
  }
  this->halt_output_();
  // Before the turn's audio: the level, and the duck of an earlier turn lifted.
  this->apply_level_();
  this->ducking_->apply_ducking(0, 0);
}

void MistPlayer::on_audio_(const mist_core::Frame &frame) {
  const size_t samples = frame.payload_len / kSampleBytes;
  if (this->queue_.on_audio_chunk(frame.turn_tag, static_cast<uint32_t>(samples)) != mist_core::TagVerdict::Play) {
    return;
  }
  const size_t stored = this->push_(frame.payload, samples);
  if (stored < samples) {
    const uint32_t lost = static_cast<uint32_t>(samples - stored);
    this->queue_.on_overflow(lost);
    this->overflow_samples_ += lost;
  }
}

void MistPlayer::begin_stop_(const mist_core::StopDecision &decision) {
  if (!decision.fade) {
    return;
  }
  const size_t fade_samples = static_cast<size_t>(decision.fade_ms) * kSampleRate / 1000;
  if (!this->speaker_active_ || fade_samples == 0) {
    this->halt_output_();
    return;
  }
  // Keep the next `fade_ms` of audio, ramped to silence, and drop the rest. Audio already in the
  // speaker chain (at most `kLeadMs`) plays before it.
  const size_t kept = std::min(this->used_, fade_samples);
  const size_t first = std::min(kept, kFifoSamples - this->head_);
  mist_core::apply_fade_out(this->fifo_ + this->head_, first, 0, kept);
  mist_core::apply_fade_out(this->fifo_, kept - first, first, kept);
  this->used_ = kept;
  this->ending_ = true;
}

void MistPlayer::halt_output_() {
  if (this->speaker_active_) {
    this->speaker_->stop();
  }
  this->speaker_active_ = false;
  this->finishing_ = false;
  this->ending_ = false;
  this->head_ = 0;
  this->used_ = 0;
  this->half_sample_ = false;
  this->in_speaker_ = 0;
}

size_t MistPlayer::push_(const uint8_t *payload, size_t samples) {
  const size_t count = std::min(samples, kFifoSamples - this->used_);
  size_t tail = (this->head_ + this->used_) % kFifoSamples;
  size_t copied = 0;
  while (copied < count) {
    const size_t chunk = std::min(count - copied, kFifoSamples - tail);
    // The payload is PCM16 little-endian, the ESP32's own order; it may sit at any alignment.
    std::memcpy(this->fifo_ + tail, payload + copied * kSampleBytes, chunk * kSampleBytes);
    copied += chunk;
    tail = (tail + chunk) % kFifoSamples;
  }
  this->used_ += count;
  return count;
}

void MistPlayer::feed_speaker_() {
  if (this->half_sample_) {
    // The speaker took the low byte of this sample last time; give it the high byte.
    const uint8_t *bytes = reinterpret_cast<const uint8_t *>(this->fifo_ + this->head_);
    if (this->speaker_->play(bytes + 1, 1, 0) == 0) {
      return;
    }
    this->half_sample_ = false;
    this->head_ = (this->head_ + 1) % kFifoSamples;
    --this->used_;
    ++this->in_speaker_;
  }
  while (this->used_ > 0 && this->in_speaker_ < kLeadSamples) {
    const size_t chunk =
        std::min({this->used_, kFifoSamples - this->head_, static_cast<size_t>(kLeadSamples - this->in_speaker_)});
    const size_t bytes =
        this->speaker_->play(reinterpret_cast<const uint8_t *>(this->fifo_ + this->head_), chunk * kSampleBytes, 0);
    const size_t whole = bytes / kSampleBytes;
    this->head_ = (this->head_ + whole) % kFifoSamples;
    this->used_ -= whole;
    this->in_speaker_ += static_cast<uint32_t>(whole);
    if (bytes % kSampleBytes != 0) {
      this->half_sample_ = true;
      return;
    }
    if (whole < chunk) {
      return;
    }
  }
}

void MistPlayer::send_report_(const mist_core::PlaybackReport &report) {
  if (report.kind == mist_core::ReportKind::None) {
    return;
  }
  char buf[160];
  size_t len;
  if (report.kind == mist_core::ReportKind::Done) {
    len = mist_core::build_playback_done(buf, sizeof(buf), report.turn_id, report.played_ms);
    ESP_LOGI(TAG, "Played turn %s: %u ms", report.turn_id, (unsigned) report.played_ms);
  } else {
    len = mist_core::build_playback_progress(buf, sizeof(buf), report.turn_id, report.played_ms);
  }
  if (len == 0 || !this->link_->send_text(buf, len)) {
    ESP_LOGW(TAG, "Could not queue a playback report for turn %s", report.turn_id);
  }
}

}  // namespace mist_player
}  // namespace esphome
