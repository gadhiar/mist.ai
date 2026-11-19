// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.

#include "alarm_rules.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace mist_core {

namespace {

// A positive number of seconds as milliseconds; a day of a billion seconds is far past any
// setting the owner sends, and keeps the sums below in range.
bool read_seconds(json::Value value, int64_t* out_ms) {
  double seconds = 0.0;
  if (!value.is_number() || !value.get_double(&seconds)) return false;
  if (!(seconds > 0.0) || seconds > 1e9) return false;
  *out_ms = static_cast<int64_t>(std::llround(seconds * 1000.0));
  return true;
}

// An id that is a string; `*fits` is false when it is too long for the device's buffers.
bool read_alarm_id(json::Value value, char* out, bool* fits) {
  if (!value.is_string()) return false;
  *fits = value.get_string(out, kIdCap);
  return true;
}

// Whether `ring_on` (`"all"` or a list of endpoint names) includes this device.
bool read_ring_on(json::Value value, const char* device_name, bool* rings_here) {
  if (value.is_string()) {
    *rings_here = value.equals("all");
    return true;
  }
  if (!value.is_array()) return false;
  *rings_here = false;
  for (uint32_t i = 0; i < value.size(); ++i) {
    const json::Value name = value.at(i);
    if (!name.is_string()) return false;
    if (device_name != nullptr && name.equals(device_name)) *rings_here = true;
  }
  return true;
}

// Copies a NUL-terminated id into a `kIdCap` buffer, cut to fit, with every byte after it zero;
// a saved memory then holds no stale bytes, and a restored one is always terminated.
void copy_id(char* out, const char* id) {
  std::strncpy(out, id, kIdCap - 1);
  out[kIdCap - 1] = '\0';
}

void copy_schedule(ScheduleCopy* out, const ScheduleCopy& in) {
  out->valid = in.valid;
  out->rev = in.rev;
  out->at_ms = in.at_ms;
  out->fallback_after_ms = in.fallback_after_ms;
  out->missed_grace_ms = in.missed_grace_ms;
  out->ring_timeout_ms = in.ring_timeout_ms;
  out->snooze_minutes = in.snooze_minutes;
  std::strncpy(out->tz, in.tz, ScheduleCopy::kTzCap - 1);
  out->tz[ScheduleCopy::kTzCap - 1] = '\0';
  out->count = in.count;
  for (size_t i = 0; i < in.count; ++i) {
    copy_id(out->entries[i].alarm_id, in.entries[i].alarm_id);
    out->entries[i].next_fire_at_ms = in.entries[i].next_fire_at_ms;
  }
  out->ringing_count = in.ringing_count;
  for (size_t i = 0; i < in.ringing_count; ++i) copy_id(out->ringing[i], in.ringing[i]);
}

void copy_report(AlarmReport* out, const AlarmReport& in) {
  copy_id(out->alarm_id, in.alarm_id);
  out->outcome = in.outcome;
  out->at_ms = in.at_ms;
  out->rang_at_ms = in.rang_at_ms;
  out->has_snoozed_until = in.outcome == ReportOutcome::Snoozed;
  out->snoozed_until_ms = out->has_snoozed_until ? in.snoozed_until_ms : 0;
  copy_id(out->request_id, in.request_id);
}

void keep_entry(ScheduleCopy* copy, const char* alarm_id, int64_t next_fire_at_ms) {
  size_t slot = copy->count;
  if (copy->count == ScheduleCopy::kMaxEntries) {
    // Full: the new entry replaces the latest one when it fires earlier.
    slot = 0;
    for (size_t i = 1; i < copy->count; ++i) {
      if (copy->entries[i].next_fire_at_ms > copy->entries[slot].next_fire_at_ms) slot = i;
    }
    if (copy->entries[slot].next_fire_at_ms <= next_fire_at_ms) return;
  } else {
    ++copy->count;
  }
  std::memcpy(copy->entries[slot].alarm_id, alarm_id, kIdCap);
  copy->entries[slot].next_fire_at_ms = next_fire_at_ms;
}

}  // namespace

bool parse_alarm_state(json::Value root, const char* device_name, ScheduleCopy* out) {
  static ScheduleCopy copy;  // one task parses; kept off the stack
  copy = ScheduleCopy{};
  if (!root.is_object()) return false;
  if (!root.get("rev").get_int(&copy.rev) || copy.rev < 0) return false;
  if (!read_seconds(root.get("fallback_after_s"), &copy.fallback_after_ms)) return false;
  if (!read_seconds(root.get("missed_grace_s"), &copy.missed_grace_ms)) return false;
  if (!read_seconds(root.get("ring_timeout_s"), &copy.ring_timeout_ms)) return false;
  int64_t snooze = 0;
  if (!root.get("snooze_minutes").get_int(&snooze) || snooze < 1 || snooze > 60) return false;
  copy.snooze_minutes = static_cast<uint32_t>(snooze);
  if (!read_id(root.get("tz"), copy.tz, sizeof(copy.tz)) || copy.tz[0] == '\0') return false;
  if (!read_time(root.get("at"), &copy.at_ms)) return false;

  const json::Value alarms = root.get("alarms");
  if (!alarms.is_array()) return false;
  for (uint32_t i = 0; i < alarms.size(); ++i) {
    const json::Value alarm = alarms.at(i);
    if (!alarm.is_object()) return false;
    char alarm_id[kIdCap];
    bool fits = false;
    if (!read_alarm_id(alarm.get("alarm_id"), alarm_id, &fits)) return false;
    bool rings_here = false;
    if (!read_ring_on(alarm.get("ring_on"), device_name, &rings_here)) return false;
    const json::Value next = alarm.get("next_fire_at");
    if (!next.ok()) return false;
    if (next.is_null() || !rings_here || !fits) continue;
    int64_t next_fire_at_ms = 0;
    if (!read_time(next, &next_fire_at_ms)) return false;
    keep_entry(&copy, alarm_id, next_fire_at_ms);
  }

  const json::Value ringing = root.get("ringing");
  if (!ringing.is_array()) return false;
  for (uint32_t i = 0; i < ringing.size(); ++i) {
    const json::Value item = ringing.at(i);
    if (!item.is_object()) return false;
    char alarm_id[kIdCap];
    bool fits = false;
    if (!read_alarm_id(item.get("alarm_id"), alarm_id, &fits)) return false;
    if (!fits || copy.ringing_count == ScheduleCopy::kMaxRinging) continue;
    std::memcpy(copy.ringing[copy.ringing_count++], alarm_id, kIdCap);
  }

  copy.valid = true;
  *out = copy;
  return true;
}

bool parse_alarm_fired(json::Value root, FiredAlarm* out) {
  FiredAlarm fired{};
  if (!root.is_object()) return false;
  if (!read_id(root.get("alarm_id"), fired.alarm_id, sizeof(fired.alarm_id))) return false;
  if (fired.alarm_id[0] == '\0') return false;
  if (!read_time(root.get("scheduled_for"), &fired.scheduled_for_ms)) return false;
  if (!read_time(root.get("fired_at"), &fired.fired_at_ms)) return false;
  if (!read_seconds(root.get("ring_timeout_s"), &fired.ring_timeout_ms)) return false;
  *out = fired;
  return true;
}

void AlarmRules::on_alarm_state(const ScheduleCopy& copy) {
  if (!copy.valid) return;
  schedule_ = copy;
  ++revision_;
  if (source_ != RingSource::Server) return;
  for (size_t i = 0; i < copy.ringing_count; ++i) {
    if (std::strcmp(copy.ringing[i], ring_id_) == 0) return;
  }
  // Stopped or snoozed elsewhere: the owner already knows the outcome.
  finish_ring();
}

void AlarmRules::on_alarm_fired(const FiredAlarm& fired, int64_t now_ms) {
  if (fired.alarm_id[0] == '\0') return;
  if (ringing() && std::strcmp(ring_id_, fired.alarm_id) == 0) return;
  if (ringing()) finish_ring();
  start_ring(RingSource::Server, fired.alarm_id, now_ms, fired.ring_timeout_ms);
  queue_report(fired.alarm_id, ReportOutcome::Rang, now_ms, now_ms, 0);
  ring_reported_ = true;
  flush(now_ms);
}

void AlarmRules::on_action_result(const char* action, const char* request_id) {
  if (action == nullptr || request_id == nullptr || request_id[0] == '\0') return;
  if (std::strcmp(action, "report") != 0) return;
  for (size_t i = 0; i < pending_count_; ++i) {
    if (std::strcmp(pending_[i].report.request_id, request_id) != 0) continue;
    for (size_t j = i + 1; j < pending_count_; ++j) pending_[j - 1] = pending_[j];
    --pending_count_;
    ++revision_;
    return;
  }
}

void AlarmRules::on_connected(int64_t now_ms) {
  connected_ = true;
  for (size_t i = 0; i < pending_count_; ++i) pending_[i].sent = false;
  if (source_ == RingSource::Local && !ring_reported_) {
    queue_report(ring_id_, ReportOutcome::Rang, ring_started_ms_, ring_started_ms_, 0);
    ring_reported_ = true;
  }
  flush(now_ms);
}

void AlarmRules::on_disconnected(int64_t now_ms) {
  connected_ = false;
  down_since_ms_ = now_ms;
}

bool AlarmRules::on_stop(int64_t now_ms) { return end_ring(now_ms, AlarmActionKind::Stop); }

bool AlarmRules::on_snooze(int64_t now_ms) { return end_ring(now_ms, AlarmActionKind::Snooze); }

bool AlarmRules::end_ring(int64_t now_ms, AlarmActionKind action) {
  if (!ringing()) return false;
  bool sent = false;
  // Online, and the owner knows this ring: the owner applies the action everywhere.
  if (connected_ && ring_reported_) {
    char request_id[kIdCap];
    next_request_id(request_id);
    char buf[256];
    const size_t len = build_alarm_action(buf, sizeof(buf), ring_id_, action, 0, request_id);
    sent = len > 0 && sink_.send_text(buf, len);
  }
  if (!sent) {
    if (action == AlarmActionKind::Snooze) {
      const int64_t until = now_ms + static_cast<int64_t>(snooze_minutes()) * 60 * 1000;
      queue_report(ring_id_, ReportOutcome::Snoozed, now_ms, ring_started_ms_, until);
    } else {
      queue_report(ring_id_, ReportOutcome::Stopped, now_ms, ring_started_ms_, 0);
    }
  }
  finish_ring();
  flush(now_ms);
  return true;
}

void AlarmRules::tick(int64_t now_ms) {
  if (now_ms <= 0) return;
  if (ringing() && now_ms - ring_started_ms_ >= ring_timeout_ms_) {
    if (source_ == RingSource::Local && !ring_reported_) {
      queue_report(ring_id_, ReportOutcome::Rang, ring_started_ms_, ring_started_ms_, 0);
    }
    finish_ring();
  }
  if (!ringing() && !connected_) start_due_local_ring(now_ms);
  flush(now_ms);
}

void AlarmRules::start_ring(RingSource source, const char* alarm_id, int64_t now_ms,
                            int64_t timeout_ms) {
  source_ = source;
  std::memcpy(ring_id_, alarm_id, kIdCap);
  ring_started_ms_ = now_ms;
  ring_timeout_ms_ = timeout_ms;
  ring_reported_ = false;
  remember(alarm_id);
  ++revision_;
  sink_.start_ring(ring_id_);
}

void AlarmRules::finish_ring() {
  if (!ringing()) return;
  sink_.stop_ring();
  source_ = RingSource::None;
  ring_id_[0] = '\0';
  ring_reported_ = false;
  ++revision_;
}

void AlarmRules::start_due_local_ring(int64_t now_ms) {
  if (!schedule_.valid) return;
  const ScheduleCopy::Entry* due = nullptr;
  for (size_t i = 0; i < schedule_.count; ++i) {
    const ScheduleCopy::Entry& entry = schedule_.entries[i];
    const int64_t fire = entry.next_fire_at_ms;
    if (entry.alarm_id[0] == '\0' || fire <= 0) continue;
    if (down_since_ms_ > fire) continue;  // the link was up at some point after the fire time
    if (now_ms < fire + schedule_.fallback_after_ms) continue;
    if (now_ms >= fire + schedule_.missed_grace_ms) continue;
    if (rung(entry.alarm_id)) continue;
    if (due == nullptr || fire < due->next_fire_at_ms) due = &entry;
  }
  if (due != nullptr) {
    start_ring(RingSource::Local, due->alarm_id, now_ms, schedule_.ring_timeout_ms);
  }
}

bool AlarmRules::rung(const char* alarm_id) const {
  for (size_t i = 0; i < ledger_count_; ++i) {
    if (std::strcmp(ledger_[i], alarm_id) == 0) return true;
  }
  return false;
}

void AlarmRules::remember(const char* alarm_id) {
  if (rung(alarm_id)) return;
  std::memcpy(ledger_[ledger_next_], alarm_id, kIdCap);
  ledger_next_ = (ledger_next_ + 1) % kLedgerSize;
  if (ledger_count_ < kLedgerSize) ++ledger_count_;
}

void AlarmRules::queue_report(const char* alarm_id, ReportOutcome outcome, int64_t at_ms,
                              int64_t rang_at_ms, int64_t snoozed_until_ms) {
  // Without a set clock the times would be refused; nothing is queued.
  if (at_ms <= 0 || rang_at_ms <= 0) return;
  if (pending_count_ == kMaxPending) {
    for (size_t j = 1; j < pending_count_; ++j) pending_[j - 1] = pending_[j];
    --pending_count_;
  }
  Pending& slot = pending_[pending_count_++];
  slot = Pending{};
  std::memcpy(slot.report.alarm_id, alarm_id, kIdCap);
  slot.report.outcome = outcome;
  slot.report.at_ms = at_ms;
  slot.report.rang_at_ms = rang_at_ms;
  slot.report.has_snoozed_until = outcome == ReportOutcome::Snoozed;
  slot.report.snoozed_until_ms = snoozed_until_ms;
  next_request_id(slot.report.request_id);
  slot.sent = false;
  ++revision_;
}

void AlarmRules::flush(int64_t now_ms) {
  if (!connected_) return;
  size_t i = 0;
  while (i < pending_count_) {
    Pending& pending = pending_[i];
    if (now_ms > 0 && pending.report.rang_at_ms < now_ms - kReportMaxAgeMs) {
      for (size_t j = i + 1; j < pending_count_; ++j) pending_[j - 1] = pending_[j];
      --pending_count_;
      ++revision_;
      continue;
    }
    if (!pending.sent) {
      char buf[384];
      const size_t len = build_alarm_report(buf, sizeof(buf), pending.report);
      if (len == 0) {
        // A report the codec cannot write never will be: drop it.
        for (size_t j = i + 1; j < pending_count_; ++j) pending_[j - 1] = pending_[j];
        --pending_count_;
        ++revision_;
        continue;
      }
      if (!sink_.send_text(buf, len)) return;  // the link is full; the next tick retries
      pending.sent = true;
    }
    ++i;
  }
}

void AlarmRules::next_request_id(char* out) {
  ++request_counter_;
  ++revision_;
  std::snprintf(out, kIdCap, "r%u", static_cast<unsigned>(request_counter_));
}

uint32_t AlarmRules::snooze_minutes() const {
  return schedule_.valid ? schedule_.snooze_minutes : kDefaultSnoozeMinutes;
}

void AlarmRules::save(AlarmMemory* out) const {
  std::memset(static_cast<void*>(out), 0, sizeof(*out));
  out->version = AlarmMemory::kVersion;
  out->request_counter = request_counter_;
  copy_schedule(&out->schedule, schedule_);
  out->ledger_count = static_cast<uint32_t>(ledger_count_);
  out->ledger_next = static_cast<uint32_t>(ledger_next_);
  for (size_t i = 0; i < ledger_count_; ++i) copy_id(out->ledger[i], ledger_[i]);
  out->pending_count = static_cast<uint32_t>(pending_count_);
  for (size_t i = 0; i < pending_count_; ++i) copy_report(&out->pending[i], pending_[i].report);
  if (source_ == RingSource::Local && !ring_reported_) {
    copy_id(out->local_ring_id, ring_id_);
    out->local_ring_started_ms = ring_started_ms_;
  }
}

bool AlarmRules::restore(const AlarmMemory& memory) {
  if (memory.version != AlarmMemory::kVersion) return false;
  if (memory.schedule.count > ScheduleCopy::kMaxEntries) return false;
  if (memory.schedule.ringing_count > ScheduleCopy::kMaxRinging) return false;
  if (memory.ledger_count > kLedgerSize || memory.ledger_next >= kLedgerSize) return false;
  if (memory.pending_count > kMaxPending) return false;
  for (uint32_t i = 0; i < memory.pending_count; ++i) {
    // Read as a byte: a damaged value is not a valid enumerator.
    uint8_t outcome = 0;
    std::memcpy(&outcome, &memory.pending[i].outcome, sizeof(outcome));
    if (outcome > static_cast<uint8_t>(ReportOutcome::Snoozed)) return false;
  }

  // Every id is copied through `copy_id`, so a memory whose bytes were damaged still yields
  // terminated strings.
  ScheduleCopy schedule{};
  copy_schedule(&schedule, memory.schedule);
  schedule_ = schedule;
  for (size_t i = 0; i < kLedgerSize; ++i) {
    if (i < memory.ledger_count) {
      copy_id(ledger_[i], memory.ledger[i]);
    } else {
      std::memset(ledger_[i], 0, kIdCap);
    }
  }
  ledger_count_ = memory.ledger_count;
  ledger_next_ = memory.ledger_next;
  pending_count_ = memory.pending_count;
  for (size_t i = 0; i < pending_count_; ++i) {
    pending_[i] = Pending{};
    copy_report(&pending_[i].report, memory.pending[i]);
  }
  request_counter_ = memory.request_counter;
  source_ = RingSource::None;
  ring_id_[0] = '\0';
  ring_reported_ = false;
  ++revision_;

  // A local ring the reboot cut short: it rang, from when it started, and is not rung again.
  char cut_short[kIdCap];
  copy_id(cut_short, memory.local_ring_id);
  if (cut_short[0] != '\0') {
    remember(cut_short);
    queue_report(cut_short, ReportOutcome::Rang, memory.local_ring_started_ms,
                 memory.local_ring_started_ms, 0);
  }
  return true;
}

// -- The clock -------------------------------------------------------------------------------

void AlarmClock::on_owner_time(int64_t owner_ms, int64_t mono_ms) {
  if (owner_ms < kEarliestMs) return;
  owner_offset_ms_ = owner_ms - mono_ms;
  owner_set_ = true;
}

int64_t AlarmClock::now(int64_t system_ms, int64_t mono_ms) const {
  if (sntp_synced_ && system_ms >= kEarliestMs) return system_ms;
  if (owner_set_) return mono_ms + owner_offset_ms_;
  return 0;
}

// -- The tone --------------------------------------------------------------------------------

void AlarmTone::fill(int16_t* out, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    out[i] = sample_at(position_);
    position_ = (position_ + 1) % static_cast<uint32_t>(kCycleSamples);
  }
}

int16_t AlarmTone::sample_at(uint32_t position) const {
  constexpr float kPi = 3.14159265358979f;
  constexpr uint32_t kBeepSamples = kSampleRate * kBeepMs / 1000;
  constexpr uint32_t kSlotSamples = kBeepSamples + kSampleRate * kGapMs / 1000;
  constexpr uint32_t kRampSamples = kSampleRate * kRampMs / 1000;
  static_assert(kBeeps * kSlotSamples <= kCycleSamples, "the beeps must fit in one cycle");

  const uint32_t beep = position / kSlotSamples;
  const uint32_t within = position % kSlotSamples;
  if (beep >= kBeeps || within >= kBeepSamples) return 0;
  // The phase in whole samples of a cycle, exact in integers, so no error builds up.
  const uint32_t step = (within * kFrequencyHz) % kSampleRate;
  const float wave = std::sin(2.0f * kPi * static_cast<float>(step) / kSampleRate);
  const uint32_t edge = within < kBeepSamples - 1 - within ? within : kBeepSamples - 1 - within;
  float envelope = 1.0f;
  if (edge < kRampSamples) {
    envelope = 0.5f * (1.0f - std::cos(kPi * static_cast<float>(edge) / kRampSamples));
  }
  return static_cast<int16_t>(std::lround(static_cast<float>(kPeak) * envelope * wave));
}

}  // namespace mist_core
