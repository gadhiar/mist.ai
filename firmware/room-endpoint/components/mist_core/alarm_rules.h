// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.
//
// The device's alarm rules (plan Design item 6; capabilities v2 D3 and A14).
//
// A ring starts two ways. The server's `alarm_fired` rings at once and reports `rang` as the tone
// starts; stop and snooze then send `alarm_action`, an `alarm_state` without the occurrence in
// `ringing` ends it (stopped elsewhere), and it ends itself at its `ring_timeout_s`. Without the
// server, the schedule copy rings an occurrence locally when the connection was down for the whole
// span from `next_fire_at` to `next_fire_at + fallback_after_s` and it is still before
// `next_fire_at + missed_grace_s`. An occurrence never rings twice on this device, and a local
// ring's outcome (`rang`, `stopped`, or `snoozed` with `snoozed_until`) is reported on
// reconnection, sent again on each later connection until `alarm_action_result` answers it.
//
// What must outlive a reboot (the schedule copy, the occurrences rung, the outcomes not yet
// answered, a local ring cut short, and the request counter) goes into an `AlarmMemory`, which the
// device keeps in preferences (NVS) and restores at boot. A local ring cut short by a reboot is
// not rung again; it is reported `rang` from the time it started.
//
// Times are milliseconds since the epoch; 0 or less means the clock is not set yet, and nothing
// rings locally until it is. `AlarmClock` gives that time: SNTP's once it has synced, else the
// owner's own from the last `alarm_state.at` or `alarm_fired.fired_at`, run on from the device's
// monotonic clock. A message reaches the device after it was written, so a clock set from it
// runs behind the owner's, inside the 60 s ahead the owner allows a report
// (`REPORT_MAX_AHEAD`, `backend/capabilities/events.py`).
//
// `AlarmTone` is the ring's sound, synthesised, since the guards forbid audio files.
//
// Threading: one task calls every method.

#pragma once

#include <cstddef>
#include <cstdint>

#include "codec.h"
#include "json_lite.h"

namespace mist_core {

// The last `alarm_state`, cut to what this device rings by.
struct ScheduleCopy {
  static constexpr size_t kMaxEntries = 16;
  static constexpr size_t kMaxRinging = 8;
  static constexpr size_t kTzCap = 65;

  struct Entry {
    char alarm_id[kIdCap];
    int64_t next_fire_at_ms;
  };

  bool valid;
  int64_t rev;
  int64_t at_ms;
  int64_t fallback_after_ms;
  int64_t missed_grace_ms;
  int64_t ring_timeout_ms;
  uint32_t snooze_minutes;
  char tz[kTzCap];
  size_t count;
  Entry entries[kMaxEntries];
  size_t ringing_count;
  char ringing[kMaxRinging][kIdCap];
};

// Reads an `alarm_state` object. It keeps the entries whose `ring_on` is `all` or names
// `device_name` and whose `next_fire_at` is set, the earliest 16 when there are more, and the
// ids in `ringing`. False, with `out` untouched, when a setting is missing or ill-typed.
bool parse_alarm_state(json::Value root, const char* device_name, ScheduleCopy* out);

struct FiredAlarm {
  char alarm_id[kIdCap];
  int64_t scheduled_for_ms;
  int64_t fired_at_ms;  // the owner's clock as it fired
  int64_t ring_timeout_ms;
};

// Reads an `alarm_fired` object; false when a field the device rings by is missing or ill-typed.
bool parse_alarm_fired(json::Value root, FiredAlarm* out);

// Where the rules' messages and the tone go. A `false` from `send_text` means the link had no
// room: an `alarm_report` stays queued and is retried by the next `tick`.
class AlarmSink {
 public:
  virtual ~AlarmSink() = default;
  virtual bool send_text(const char* text, size_t len) = 0;
  virtual void start_ring(const char* alarm_id) = 0;
  virtual void stop_ring() = 0;
};

enum class RingSource : uint8_t { None, Server, Local };

struct AlarmMemory;

class AlarmRules {
 public:
  static constexpr size_t kMaxPending = 16;
  static constexpr size_t kLedgerSize = 32;
  // The owner's own default, used only before any `alarm_state` gave the setting.
  static constexpr uint32_t kDefaultSnoozeMinutes = 9;
  // `REPORT_MAX_AGE` in `backend/capabilities/events.py`: an older report is refused.
  static constexpr int64_t kReportMaxAgeMs = 24LL * 3600 * 1000;

  explicit AlarmRules(AlarmSink& sink) : sink_(sink) {}

  void on_alarm_state(const ScheduleCopy& copy);
  void on_alarm_fired(const FiredAlarm& fired, int64_t now_ms);
  // An `alarm_action_result`; a `report` answer retires the report it names.
  void on_action_result(const char* action, const char* request_id);
  void on_connected(int64_t now_ms);
  void on_disconnected(int64_t now_ms);

  // "Stop" or a single press; false when nothing rings.
  bool on_stop(int64_t now_ms);
  // A double press; false when nothing rings.
  bool on_snooze(int64_t now_ms);

  // Ends a ring at its timeout, starts a due local ring, and retries unsent reports.
  void tick(int64_t now_ms);

  RingSource ring_source() const { return source_; }
  bool ringing() const { return source_ != RingSource::None; }
  const char* ringing_id() const { return ring_id_; }
  size_t pending_reports() const { return pending_count_; }
  const ScheduleCopy& schedule() const { return schedule_; }

  // What survives a reboot. `save` writes every byte of `out`, padding included, so the same
  // state always gives the same bytes and preferences write only a changed memory.
  void save(AlarmMemory* out) const;
  // Takes a memory saved by `save`, at boot, before anything else; false, with nothing changed,
  // for a memory of another layout or one whose counts are out of range.
  bool restore(const AlarmMemory& memory);
  // Advances on every change `save` would write; the device saves when it moves.
  uint32_t revision() const { return revision_; }

 private:
  struct Pending {
    AlarmReport report;
    bool sent;  // sent on the current connection
  };

  bool end_ring(int64_t now_ms, AlarmActionKind action);
  void start_ring(RingSource source, const char* alarm_id, int64_t now_ms, int64_t timeout_ms);
  void finish_ring();
  void start_due_local_ring(int64_t now_ms);
  bool rung(const char* alarm_id) const;
  void remember(const char* alarm_id);
  void queue_report(const char* alarm_id, ReportOutcome outcome, int64_t at_ms,
                    int64_t rang_at_ms, int64_t snoozed_until_ms);
  void flush(int64_t now_ms);
  void next_request_id(char* out);
  uint32_t snooze_minutes() const;

  AlarmSink& sink_;
  ScheduleCopy schedule_{};

  bool connected_ = false;
  // When the connection was last lost; a device that has not connected since boot counts as down
  // since forever.
  int64_t down_since_ms_ = INT64_MIN;

  RingSource source_ = RingSource::None;
  char ring_id_[kIdCap] = {};
  int64_t ring_started_ms_ = 0;
  int64_t ring_timeout_ms_ = 0;
  bool ring_reported_ = false;  // its `rang` report is queued or sent

  // Occurrences rung on this device, oldest first, so none rings twice.
  char ledger_[kLedgerSize][kIdCap] = {};
  size_t ledger_count_ = 0;
  size_t ledger_next_ = 0;

  Pending pending_[kMaxPending] = {};
  size_t pending_count_ = 0;
  uint32_t request_counter_ = 0;
  uint32_t revision_ = 0;
};

// The rules' state across a reboot: plain data, written whole to preferences.
struct AlarmMemory {
  // Raised whenever the layout changes, so an older memory is refused rather than misread.
  static constexpr uint32_t kVersion = 1;

  uint32_t version;
  uint32_t request_counter;
  ScheduleCopy schedule;
  uint32_t ledger_count;
  uint32_t ledger_next;
  char ledger[AlarmRules::kLedgerSize][kIdCap];
  uint32_t pending_count;
  AlarmReport pending[AlarmRules::kMaxPending];
  // A local ring still playing, not yet reported: empty when none.
  char local_ring_id[kIdCap];
  int64_t local_ring_started_ms;
};

// The time the rules run by, in milliseconds since the epoch.
class AlarmClock {
 public:
  // 2024-01-01T00:00:00Z: an earlier time is a clock that was never set.
  static constexpr int64_t kEarliestMs = 1704067200000LL;

  // SNTP has set the system clock; from now on it is the time.
  void on_sntp_synced() { sntp_synced_ = true; }
  // The owner's clock read `owner_ms` (an `alarm_state.at` or `alarm_fired.fired_at`) when the
  // device's monotonic clock read `mono_ms`. Each later one sets the clock again.
  void on_owner_time(int64_t owner_ms, int64_t mono_ms);
  // The time: the system clock once SNTP has synced, else the owner's run on from `mono_ms`,
  // else 0 (not set).
  int64_t now(int64_t system_ms, int64_t mono_ms) const;
  bool from_sntp() const { return sntp_synced_; }

 private:
  bool sntp_synced_ = false;
  bool owner_set_ = false;
  int64_t owner_offset_ms_ = 0;  // owner time minus monotonic time
};

// The ring's sound: four 100 ms beeps of a 880 Hz sine, 100 ms apart, every 1.4 s, as 16 kHz
// PCM16 mono. Its peak is -1 dBFS (29204), so the player's level, the ring floor and its 0 dB cap
// set how loud it plays, not the tone. Each beep starts at phase 0 and ramps in and out over
// 5 ms (a raised cosine), so its edges do not click.
class AlarmTone {
 public:
  static constexpr uint32_t kSampleRate = 16000;
  static constexpr uint32_t kFrequencyHz = 880;
  static constexpr int16_t kPeak = 29204;  // 32767 * 10^(-1/20)
  static constexpr uint32_t kBeeps = 4;
  static constexpr uint32_t kBeepMs = 100;
  static constexpr uint32_t kGapMs = 100;
  static constexpr uint32_t kCycleMs = 1400;
  static constexpr uint32_t kRampMs = 5;
  static constexpr size_t kCycleSamples = kSampleRate * kCycleMs / 1000;

  // The next `count` samples of the endless pattern.
  void fill(int16_t* out, size_t count);
  // Back to the start of the first beep, for the next ring.
  void reset() { position_ = 0; }

 private:
  int16_t sample_at(uint32_t position) const;

  uint32_t position_ = 0;  // within the cycle
};

}  // namespace mist_core
