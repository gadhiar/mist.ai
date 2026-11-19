// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.

#include "mist_alarm.h"

#include <cinttypes>
#include <cstring>
#include <new>

#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <sys/time.h>

#include "esphome/core/hal.h"
#include "esphome/core/helpers.h"
#include "esphome/core/log.h"

namespace esphome {
namespace mist_alarm {

static const char *const TAG = "mist_alarm";

static constexpr size_t kSampleBytes = sizeof(int16_t);

static const char *source_name(mist_core::RingSource source) {
  switch (source) {
    case mist_core::RingSource::Server:
      return "server";
    case mist_core::RingSource::Local:
      return "local";
    case mist_core::RingSource::None:
      break;
  }
  return "none";
}

void MistAlarm::setup() {
  if (this->link_ == nullptr || this->player_ == nullptr || this->speaker_ == nullptr || this->time_ == nullptr ||
      this->endpoint_name_ == nullptr) {
    ESP_LOGE(TAG, "Missing the link, the player, the speaker, the clock or the endpoint name");
    this->mark_failed();
    return;
  }
  void *memory = heap_caps_calloc(1, sizeof(mist_core::AlarmMemory), MALLOC_CAP_SPIRAM);
  void *incoming = heap_caps_calloc(1, sizeof(mist_core::ScheduleCopy), MALLOC_CAP_SPIRAM);
  if (memory == nullptr || incoming == nullptr) {
    ESP_LOGE(TAG, "Could not allocate the alarm memory in PSRAM");
    heap_caps_free(memory);
    heap_caps_free(incoming);
    this->mark_failed();
    return;
  }
  this->memory_ = new (memory) mist_core::AlarmMemory();
  this->incoming_ = new (incoming) mist_core::ScheduleCopy();

  // The schedule copy, the occurrences rung and the unanswered outcomes, from before the reboot.
  this->memory_pref_ =
      global_preferences->make_preference(sizeof(mist_core::AlarmMemory), fnv1_hash(kMemoryKey));
  if (!this->memory_pref_.load(reinterpret_cast<uint8_t *>(this->memory_), sizeof(mist_core::AlarmMemory))) {
    ESP_LOGI(TAG, "No saved alarm memory");
  } else if (!this->rules_.restore(*this->memory_)) {
    ESP_LOGW(TAG, "Saved alarm memory has another layout; starting without it");
  } else {
    ESP_LOGI(TAG, "Restored %u schedule entries and %u unanswered reports", (unsigned) this->rules_.schedule().count,
             (unsigned) this->rules_.pending_reports());
  }
  // The restore itself may change what is saved (a ring the reboot cut short becomes a report),
  // so the first pass of the loop writes the memory back.
  this->saved_revision_ = this->rules_.revision() - 1;
  this->memory_ready_ = true;

  this->time_->add_on_time_sync_callback([this]() {
    if (!this->clock_.from_sntp()) {
      ESP_LOGI(TAG, "SNTP synced: alarms run by the system clock");
    }
    this->clock_.on_sntp_synced();
  });

  this->player_->set_ring_floor(this->ring_floor_);
  this->link_->add_listener(this);
}

void MistAlarm::loop() {
  if (!this->memory_ready_) {
    return;
  }
  const uint32_t now = millis();
  if (now - this->last_tick_ms_ >= kTickMs) {
    this->last_tick_ms_ = now;
    this->rules_.tick(this->now_ms_());
  }
  this->drive_speaker_();
  this->persist_();
}

void MistAlarm::dump_config() {
  ESP_LOGCONFIG(TAG,
                "MIST alarms:\n"
                "  Endpoint name: %s\n"
                "  Ring floor: %.2f (the player's level)\n"
                "  Tone: %u Hz, %u beeps of %u ms every %u ms, peak %d of 32767\n"
                "  Memory: preference '%s' (NVS key %" PRIu32 "), %u bytes",
                this->endpoint_name_, this->ring_floor_, (unsigned) mist_core::AlarmTone::kFrequencyHz,
                (unsigned) mist_core::AlarmTone::kBeeps, (unsigned) mist_core::AlarmTone::kBeepMs,
                (unsigned) mist_core::AlarmTone::kCycleMs, (int) mist_core::AlarmTone::kPeak, kMemoryKey,
                fnv1_hash(kMemoryKey), (unsigned) sizeof(mist_core::AlarmMemory));
}

// -- Stop and snooze ---------------------------------------------------------------------------

bool MistAlarm::on_ring_action(mist_core::ButtonAction action) {
  bool ended = false;
  switch (action) {
    case mist_core::ButtonAction::StopRing:
      ended = this->rules_.on_stop(this->now_ms_());
      ESP_LOGI(TAG, "Stop: %s", ended ? "ring stopped" : "nothing rings");
      break;
    case mist_core::ButtonAction::SnoozeRing:
      ended = this->rules_.on_snooze(this->now_ms_());
      ESP_LOGI(TAG, "Snooze: %s", ended ? "ring snoozed" : "nothing rings");
      break;
    default:
      break;
  }
  this->persist_();
  return ended;
}

// -- The link ----------------------------------------------------------------------------------

void MistAlarm::on_link_connected() {
  this->rules_.on_connected(this->now_ms_());
  ESP_LOGD(TAG, "Connected: %u reports to send", (unsigned) this->rules_.pending_reports());
}

void MistAlarm::on_link_disconnected() { this->rules_.on_disconnected(this->now_ms_()); }

void MistAlarm::on_link_message(const mist_core::Inbound &message, const mist_core::json::Doc &doc) {
  switch (message.type) {
    case mist_core::InboundType::AlarmState: {
      if (!mist_core::parse_alarm_state(doc.root(), this->endpoint_name_, this->incoming_)) {
        ESP_LOGW(TAG, "alarm_state with a missing or ill-typed setting; schedule kept");
        break;
      }
      this->clock_.on_owner_time(this->incoming_->at_ms, this->mono_ms_());
      this->rules_.on_alarm_state(*this->incoming_);
      ESP_LOGI(TAG, "Schedule rev %lld: %u entries ring here", (long long) this->incoming_->rev,
               (unsigned) this->incoming_->count);
      break;
    }
    case mist_core::InboundType::AlarmFired: {
      mist_core::FiredAlarm fired{};
      if (!mist_core::parse_alarm_fired(doc.root(), &fired)) {
        ESP_LOGW(TAG, "alarm_fired with a missing or ill-typed field; not rung");
        break;
      }
      this->clock_.on_owner_time(fired.fired_at_ms, this->mono_ms_());
      this->rules_.on_alarm_fired(fired, this->now_ms_());
      break;
    }
    case mist_core::InboundType::AlarmActionResult:
      this->rules_.on_action_result(message.action_result.action, message.action_result.request_id);
      ESP_LOGD(TAG, "alarm_action_result %s %s: %s", message.action_result.action,
               message.action_result.request_id, message.action_result.outcome);
      break;
    default:
      return;
  }
  this->persist_();
}

// -- The rules' sink ---------------------------------------------------------------------------

bool MistAlarm::send_text(const char *text, size_t len) { return this->link_->send_text(text, len); }

void MistAlarm::start_ring(const char *alarm_id) {
  ESP_LOGI(TAG, "Ringing %s (%s)", alarm_id, source_name(this->rules_.ring_source()));
  // The player owns the level: the floor first, so the ring's first sample plays at it.
  this->player_->set_ring_floor(this->ring_floor_);
  this->player_->set_ringing(true);
  this->tone_.reset();
  this->chunk_offset_ = 0;
  this->chunk_bytes_ = 0;
  // The loop starts the speaker; a ring that replaces another keeps it running.
  this->ringing_ = true;
}

void MistAlarm::stop_ring() {
  ESP_LOGI(TAG, "Ring ended");
  // The loop stops the speaker on its next pass, unless a new ring starts first.
  this->ringing_ = false;
  this->player_->set_ringing(false);
}

// -- Time and memory ---------------------------------------------------------------------------

int64_t MistAlarm::mono_ms_() const { return esp_timer_get_time() / 1000; }

int64_t MistAlarm::now_ms_() const {
  struct timeval tv {};
  gettimeofday(&tv, nullptr);
  const int64_t system_ms = static_cast<int64_t>(tv.tv_sec) * 1000 + tv.tv_usec / 1000;
  return this->clock_.now(system_ms, this->mono_ms_());
}

void MistAlarm::persist_() {
  if (!this->memory_ready_ || this->rules_.revision() == this->saved_revision_) {
    return;
  }
  this->rules_.save(this->memory_);
  if (!this->memory_pref_.save(reinterpret_cast<const uint8_t *>(this->memory_), sizeof(mist_core::AlarmMemory))) {
    ESP_LOGW(TAG, "Could not queue the alarm memory for flash");
    return;
  }
  // At once rather than at ESPHome's next flush: a local ring's outcome must survive a power cut.
  // Preferences write only a blob whose bytes changed.
  if (!global_preferences->sync()) {
    ESP_LOGW(TAG, "Could not write the alarm memory to flash");
    return;
  }
  this->saved_revision_ = this->rules_.revision();
}

// -- The tone ----------------------------------------------------------------------------------

void MistAlarm::drive_speaker_() {
  if (!this->ringing_) {
    if (this->speaker_started_) {
      this->speaker_->stop();
      this->speaker_started_ = false;
    }
    return;
  }
  if (!this->speaker_started_) {
    this->speaker_->set_audio_stream_info(audio::AudioStreamInfo(16, 1, mist_core::AlarmTone::kSampleRate));
    this->speaker_->start();
    this->speaker_started_ = true;
  }
  this->feed_tone_();
}

void MistAlarm::feed_tone_() {
  for (int chunks = 0; chunks < kMaxChunksPerLoop;) {
    if (this->chunk_offset_ >= this->chunk_bytes_) {
      this->tone_.fill(this->chunk_, kChunkSamples);
      this->chunk_offset_ = 0;
      this->chunk_bytes_ = kChunkSamples * kSampleBytes;
      ++chunks;
    }
    const uint8_t *bytes = reinterpret_cast<const uint8_t *>(this->chunk_);
    const size_t written =
        this->speaker_->play(bytes + this->chunk_offset_, this->chunk_bytes_ - this->chunk_offset_, 0);
    if (written == 0) {
      return;  // the resampler's buffer is full, or it is still starting
    }
    this->chunk_offset_ += written;
  }
}

}  // namespace mist_alarm
}  // namespace esphome
