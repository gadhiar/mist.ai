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
// Times are milliseconds since the epoch; 0 or less means the clock is not set yet, and nothing
// rings locally until it is. Threading: one task calls every method.

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
};

}  // namespace mist_core
