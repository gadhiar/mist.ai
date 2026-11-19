// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.
//
// Host tests for the alarm rules (plan Design item 6; capabilities v2 D3 and A14): the schedule
// copy from `alarm_state`, the local fallback ring and each of its boundaries, the outcome reports
// on reconnection, the rings the server starts with `alarm_fired`, what survives a reboot, the
// clock the rules run by, and the synthesised tone.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "alarm_rules.h"
#include "check.h"
#include "codec.h"
#include "json_lite.h"

using mist_core::AlarmRules;
using mist_core::FiredAlarm;
using mist_core::RingSource;
using mist_core::ScheduleCopy;
namespace json = mist_core::json;

namespace {

constexpr int64_t kT = 1794898800000;  // 2026-11-17T07:00:00Z
constexpr int64_t kSecond = 1000;
constexpr int64_t kFallback = 30 * kSecond;
constexpr int64_t kGrace = 900 * kSecond;
constexpr int64_t kTimeout = 120 * kSecond;

class RecordingSink : public mist_core::AlarmSink {
 public:
  bool send_text(const char* text, size_t len) override {
    if (refuse) return false;
    texts.emplace_back(text, len);
    return true;
  }
  void start_ring(const char* alarm_id) override { starts.emplace_back(alarm_id); }
  void stop_ring() override { ++stops; }

  std::vector<std::string> texts;
  std::vector<std::string> starts;
  int stops = 0;
  bool refuse = false;
};

ScheduleCopy schedule_of(std::initializer_list<std::pair<const char*, int64_t>> entries) {
  ScheduleCopy copy{};
  copy.valid = true;
  copy.at_ms = kT - 3600 * kSecond;
  copy.fallback_after_ms = kFallback;
  copy.missed_grace_ms = kGrace;
  copy.ring_timeout_ms = kTimeout;
  copy.snooze_minutes = 9;
  std::strcpy(copy.tz, "UTC");
  for (const auto& entry : entries) {
    std::strcpy(copy.entries[copy.count].alarm_id, entry.first);
    copy.entries[copy.count].next_fire_at_ms = entry.second;
    ++copy.count;
  }
  return copy;
}

std::string field(const std::string& text, const char* key) {
  static json::Token pool[64];
  json::Doc doc(pool, 64);
  char out[128] = {};
  if (doc.parse(text.data(), text.size()) != json::ParseError::None) return "<bad json>";
  json::Value value = doc.root().get(key);
  if (!value.ok()) return "<absent>";
  if (!value.get_string(out, sizeof(out))) return "<not a string>";
  return out;
}

std::string iso(int64_t ms) {
  char out[32];
  mist_core::format_iso8601_ms(ms, out, sizeof(out));
  return out;
}

// Acknowledges every report the sink has seen, as the server's `alarm_action_result` would.
void acknowledge_reports(AlarmRules* rules, const RecordingSink& sink) {
  for (const std::string& text : sink.texts) {
    if (field(text, "type") == "alarm_report") {
      rules->on_action_result("report", field(text, "request_id").c_str());
    }
  }
}

}  // namespace

// -- The schedule copy -----------------------------------------------------------------------

MIST_TEST(the_schedule_copy_keeps_this_devices_entries_with_a_fire_time) {
  const char* text =
      "{\"type\":\"alarm_state\",\"rev\":7,\"alarms\":["
      "{\"alarm_id\":\"a1\",\"next_fire_at\":\"2026-11-17T07:00:00Z\",\"ring_on\":\"all\"},"
      "{\"alarm_id\":\"a2\",\"next_fire_at\":\"2026-11-17T08:00:00Z\",\"ring_on\":[\"desk\",\"room-a\"]},"
      "{\"alarm_id\":\"a3\",\"next_fire_at\":\"2026-11-17T09:00:00Z\",\"ring_on\":[\"desk\"]},"
      "{\"alarm_id\":\"a4\",\"next_fire_at\":null,\"ring_on\":\"all\"}],"
      "\"ringing\":[{\"alarm_id\":\"a9\",\"late_by_s\":0.5}],\"missed\":[],"
      "\"fallback_after_s\":30.0,\"missed_grace_s\":900.0,\"ring_timeout_s\":120.5,"
      "\"snooze_minutes\":9,\"tz\":\"Europe/London\",\"at\":\"2026-11-17T06:00:00Z\"}";
  static json::Token pool[256];
  json::Doc doc(pool, 256);
  CHECK(doc.parse(text, std::strlen(text)) == json::ParseError::None);
  ScheduleCopy copy{};

  CHECK(mist_core::parse_alarm_state(doc.root(), "room-a", &copy));

  CHECK(copy.valid);
  CHECK_EQ(copy.rev, 7);
  CHECK_EQ(copy.count, 2);
  CHECK_STREQ(copy.entries[0].alarm_id, "a1");
  CHECK(copy.entries[0].next_fire_at_ms == kT);
  CHECK_STREQ(copy.entries[1].alarm_id, "a2");
  CHECK(copy.entries[1].next_fire_at_ms == kT + 3600 * kSecond);
  CHECK_EQ(copy.ringing_count, 1);
  CHECK_STREQ(copy.ringing[0], "a9");
  CHECK(copy.fallback_after_ms == kFallback);
  CHECK(copy.missed_grace_ms == kGrace);
  CHECK(copy.ring_timeout_ms == 120500);
  CHECK_EQ(copy.snooze_minutes, 9);
  CHECK_STREQ(copy.tz, "Europe/London");
  CHECK(copy.at_ms == kT - 3600 * kSecond);
}

MIST_TEST(an_alarm_state_missing_a_setting_is_refused) {
  const char* text =
      "{\"type\":\"alarm_state\",\"rev\":1,\"alarms\":[],\"ringing\":[],\"missed\":[],"
      "\"fallback_after_s\":30.0,\"missed_grace_s\":900.0,\"snooze_minutes\":9,"
      "\"tz\":\"UTC\",\"at\":\"2026-11-17T06:00:00Z\"}";
  static json::Token pool[64];
  json::Doc doc(pool, 64);
  CHECK(doc.parse(text, std::strlen(text)) == json::ParseError::None);
  ScheduleCopy copy{};

  CHECK(!mist_core::parse_alarm_state(doc.root(), "room-a", &copy));
}

MIST_TEST(alarm_fired_is_read_with_its_time_and_timeout) {
  const char* text =
      "{\"type\":\"alarm_fired\",\"alarm_id\":\"occ-1\",\"record_id\":\"rec-1\",\"kind\":\"alarm\","
      "\"category\":\"one_off\",\"label\":null,\"scheduled_for\":\"2026-11-17T07:00:00Z\","
      "\"fired_at\":\"2026-11-17T07:00:01Z\",\"late_by_s\":1.0,\"ring_timeout_s\":300.0,"
      "\"ring_on\":\"all\",\"fallback\":false}";
  static json::Token pool[64];
  json::Doc doc(pool, 64);
  CHECK(doc.parse(text, std::strlen(text)) == json::ParseError::None);
  FiredAlarm fired{};

  CHECK(mist_core::parse_alarm_fired(doc.root(), &fired));

  CHECK_STREQ(fired.alarm_id, "occ-1");
  CHECK(fired.scheduled_for_ms == kT);
  CHECK(fired.fired_at_ms == kT + kSecond);
  CHECK(fired.ring_timeout_ms == 300 * kSecond);
}

// -- The local fallback ring -----------------------------------------------------------------

MIST_TEST(a_device_that_never_connected_rings_alone_in_the_window) {
  RecordingSink sink;
  AlarmRules rules(sink);
  rules.on_alarm_state(schedule_of({{"a1", kT}}));

  rules.tick(kT + kFallback);

  CHECK(rules.ring_source() == RingSource::Local);
  CHECK_EQ(sink.starts.size(), 1);
  CHECK(sink.starts[0] == "a1");
}

MIST_TEST(no_local_ring_before_fallback_after_s) {
  RecordingSink sink;
  AlarmRules rules(sink);
  rules.on_alarm_state(schedule_of({{"a1", kT}}));

  rules.tick(kT + kFallback - 1);

  CHECK(rules.ring_source() == RingSource::None);
  CHECK(sink.starts.empty());
}

MIST_TEST(no_local_ring_once_missed_grace_s_has_passed) {
  RecordingSink sink;
  AlarmRules rules(sink);
  rules.on_alarm_state(schedule_of({{"a1", kT}}));

  rules.tick(kT + kGrace);

  CHECK(sink.starts.empty());
}

MIST_TEST(a_local_ring_one_millisecond_inside_the_grace_window_still_rings) {
  RecordingSink sink;
  AlarmRules rules(sink);
  rules.on_alarm_state(schedule_of({{"a1", kT}}));

  rules.tick(kT + kGrace - 1);

  CHECK_EQ(sink.starts.size(), 1);
}

MIST_TEST(a_connection_lost_exactly_at_the_fire_time_counts_as_down_for_the_span) {
  RecordingSink sink;
  AlarmRules rules(sink);
  rules.on_connected(kT - 60 * kSecond);
  rules.on_alarm_state(schedule_of({{"a1", kT}}));

  rules.on_disconnected(kT);
  rules.tick(kT + kFallback);

  CHECK_EQ(sink.starts.size(), 1);
}

MIST_TEST(a_connection_lost_after_the_fire_time_never_rings_alone) {
  RecordingSink sink;
  AlarmRules rules(sink);
  rules.on_connected(kT - 60 * kSecond);
  rules.on_alarm_state(schedule_of({{"a1", kT}}));

  rules.on_disconnected(kT + 1);
  rules.tick(kT + kFallback);
  rules.tick(kT + kFallback + 60 * kSecond);

  CHECK(sink.starts.empty());
}

MIST_TEST(a_connected_device_never_rings_alone) {
  RecordingSink sink;
  AlarmRules rules(sink);
  rules.on_connected(kT - 60 * kSecond);
  rules.on_alarm_state(schedule_of({{"a1", kT}}));

  rules.tick(kT + kFallback);
  rules.tick(kT + kGrace - 1);

  CHECK(sink.starts.empty());
}

MIST_TEST(no_local_ring_without_a_set_clock) {
  RecordingSink sink;
  AlarmRules rules(sink);
  rules.on_alarm_state(schedule_of({{"a1", kT}}));

  rules.tick(0);
  rules.tick(-5);

  CHECK(sink.starts.empty());
}

MIST_TEST(an_occurrence_never_rings_twice_locally) {
  RecordingSink sink;
  AlarmRules rules(sink);
  rules.on_alarm_state(schedule_of({{"a1", kT}}));
  rules.tick(kT + kFallback);
  CHECK(rules.on_stop(kT + kFallback + 5 * kSecond));

  rules.tick(kT + kFallback + 10 * kSecond);
  rules.on_alarm_state(schedule_of({{"a1", kT}}));
  rules.tick(kT + kFallback + 20 * kSecond);

  CHECK_EQ(sink.starts.size(), 1);
}

MIST_TEST(the_local_ring_stops_itself_at_ring_timeout_s) {
  RecordingSink sink;
  AlarmRules rules(sink);
  rules.on_alarm_state(schedule_of({{"a1", kT}}));
  const int64_t start = kT + kFallback;
  rules.tick(start);

  rules.tick(start + kTimeout - 1);
  CHECK(rules.ringing());
  rules.tick(start + kTimeout);

  CHECK(!rules.ringing());
  CHECK_EQ(sink.stops, 1);
}

MIST_TEST(two_due_occurrences_ring_one_after_the_other) {
  RecordingSink sink;
  AlarmRules rules(sink);
  rules.on_alarm_state(schedule_of({{"a2", kT + kSecond}, {"a1", kT}}));
  rules.tick(kT + kFallback + kSecond);
  CHECK_EQ(sink.starts.size(), 1);
  CHECK(sink.starts[0] == "a1");

  CHECK(rules.on_stop(kT + kFallback + 2 * kSecond));
  rules.tick(kT + kFallback + 3 * kSecond);

  CHECK_EQ(sink.starts.size(), 2);
  CHECK(sink.starts[1] == "a2");
}

// -- Outcome reports on reconnection ---------------------------------------------------------

MIST_TEST(a_timed_out_local_ring_is_reported_rang_on_reconnection) {
  RecordingSink sink;
  AlarmRules rules(sink);
  rules.on_alarm_state(schedule_of({{"a1", kT}}));
  const int64_t start = kT + kFallback;
  rules.tick(start);
  rules.tick(start + kTimeout);
  CHECK(sink.texts.empty());

  rules.on_connected(start + kTimeout + 10 * kSecond);

  CHECK_EQ(sink.texts.size(), 1);
  CHECK(field(sink.texts[0], "type") == "alarm_report");
  CHECK(field(sink.texts[0], "alarm_id") == "a1");
  CHECK(field(sink.texts[0], "outcome") == "rang");
  CHECK(field(sink.texts[0], "at") == iso(start));
  CHECK(field(sink.texts[0], "rang_at") == iso(start));
  CHECK(field(sink.texts[0], "request_id") != "<absent>");
}

MIST_TEST(a_stopped_local_ring_is_reported_stopped_with_both_times) {
  RecordingSink sink;
  AlarmRules rules(sink);
  rules.on_alarm_state(schedule_of({{"a1", kT}}));
  const int64_t start = kT + kFallback;
  rules.tick(start);

  CHECK(rules.on_stop(start + 7 * kSecond));
  rules.on_connected(start + 60 * kSecond);

  CHECK_EQ(sink.stops, 1);
  CHECK_EQ(sink.texts.size(), 1);
  CHECK(field(sink.texts[0], "outcome") == "stopped");
  CHECK(field(sink.texts[0], "rang_at") == iso(start));
  CHECK(field(sink.texts[0], "at") == iso(start + 7 * kSecond));
}

MIST_TEST(a_snoozed_local_ring_is_reported_with_snoozed_until) {
  RecordingSink sink;
  AlarmRules rules(sink);
  rules.on_alarm_state(schedule_of({{"a1", kT}}));
  const int64_t start = kT + kFallback;
  rules.tick(start);

  CHECK(rules.on_snooze(start + 3 * kSecond));
  rules.on_connected(start + 60 * kSecond);

  CHECK(field(sink.texts[0], "outcome") == "snoozed");
  CHECK(field(sink.texts[0], "snoozed_until") == iso(start + 3 * kSecond + 9 * 60 * kSecond));
}

MIST_TEST(an_unacknowledged_report_is_sent_again_on_the_next_connection_only) {
  RecordingSink sink;
  AlarmRules rules(sink);
  rules.on_alarm_state(schedule_of({{"a1", kT}}));
  rules.tick(kT + kFallback);
  rules.on_stop(kT + kFallback + kSecond);
  rules.on_connected(kT + 100 * kSecond);
  rules.tick(kT + 101 * kSecond);
  CHECK_EQ(sink.texts.size(), 1);

  rules.on_disconnected(kT + 102 * kSecond);
  rules.on_connected(kT + 103 * kSecond);

  CHECK_EQ(sink.texts.size(), 2);
  CHECK(field(sink.texts[1], "request_id") == field(sink.texts[0], "request_id"));
  CHECK(field(sink.texts[1], "at") == field(sink.texts[0], "at"));
}

MIST_TEST(an_acknowledged_report_is_never_sent_again) {
  RecordingSink sink;
  AlarmRules rules(sink);
  rules.on_alarm_state(schedule_of({{"a1", kT}}));
  rules.tick(kT + kFallback);
  rules.on_stop(kT + kFallback + kSecond);
  rules.on_connected(kT + 100 * kSecond);

  acknowledge_reports(&rules, sink);
  rules.on_disconnected(kT + 102 * kSecond);
  rules.on_connected(kT + 103 * kSecond);

  CHECK_EQ(sink.texts.size(), 1);
  CHECK_EQ(rules.pending_reports(), 0);
}

MIST_TEST(a_report_the_link_refused_is_retried_while_connected) {
  RecordingSink sink;
  AlarmRules rules(sink);
  rules.on_alarm_state(schedule_of({{"a1", kT}}));
  rules.tick(kT + kFallback);
  rules.on_stop(kT + kFallback + kSecond);
  sink.refuse = true;
  rules.on_connected(kT + 100 * kSecond);
  CHECK(sink.texts.empty());

  sink.refuse = false;
  rules.tick(kT + 101 * kSecond);

  CHECK_EQ(sink.texts.size(), 1);
}

MIST_TEST(a_report_older_than_a_day_is_dropped_unsent) {
  RecordingSink sink;
  AlarmRules rules(sink);
  rules.on_alarm_state(schedule_of({{"a1", kT}}));
  rules.tick(kT + kFallback);
  rules.on_stop(kT + kFallback + kSecond);

  rules.on_connected(kT + kFallback + kSecond + 24 * 3600 * kSecond + 1);

  CHECK(sink.texts.empty());
  CHECK_EQ(rules.pending_reports(), 0);
}

MIST_TEST(a_local_ring_still_playing_at_reconnection_is_reported_rang_then_acts_online) {
  RecordingSink sink;
  AlarmRules rules(sink);
  rules.on_alarm_state(schedule_of({{"a1", kT}}));
  const int64_t start = kT + kFallback;
  rules.tick(start);

  rules.on_connected(start + 5 * kSecond);
  CHECK(rules.ringing());
  CHECK(rules.on_stop(start + 6 * kSecond));

  CHECK_EQ(sink.texts.size(), 2);
  CHECK(field(sink.texts[0], "outcome") == "rang");
  CHECK(field(sink.texts[0], "at") == iso(start));
  CHECK(field(sink.texts[1], "type") == "alarm_action");
  CHECK(field(sink.texts[1], "action") == "stop");
  CHECK(field(sink.texts[1], "alarm_id") == "a1");
}

// -- Rings the server starts -----------------------------------------------------------------

MIST_TEST(alarm_fired_rings_and_reports_rang_as_the_tone_starts) {
  RecordingSink sink;
  AlarmRules rules(sink);
  rules.on_connected(kT - kSecond);
  FiredAlarm fired{};
  std::strcpy(fired.alarm_id, "occ-1");
  fired.scheduled_for_ms = kT;
  fired.ring_timeout_ms = kTimeout;

  rules.on_alarm_fired(fired, kT + 500);

  CHECK(rules.ring_source() == RingSource::Server);
  CHECK_EQ(sink.starts.size(), 1);
  CHECK_EQ(sink.texts.size(), 1);
  CHECK(field(sink.texts[0], "outcome") == "rang");
  CHECK(field(sink.texts[0], "at") == iso(kT + 500));
  CHECK(field(sink.texts[0], "rang_at") == iso(kT + 500));
}

MIST_TEST(stop_and_snooze_of_a_server_ring_send_alarm_actions) {
  RecordingSink sink;
  AlarmRules rules(sink);
  rules.on_connected(kT - kSecond);
  FiredAlarm fired{};
  std::strcpy(fired.alarm_id, "occ-1");
  fired.scheduled_for_ms = kT;
  fired.ring_timeout_ms = kTimeout;
  rules.on_alarm_fired(fired, kT);

  CHECK(rules.on_snooze(kT + kSecond));
  CHECK(!rules.on_stop(kT + 2 * kSecond));
  std::strcpy(fired.alarm_id, "occ-2");
  rules.on_alarm_fired(fired, kT + 3 * kSecond);
  CHECK(rules.on_stop(kT + 4 * kSecond));

  CHECK(field(sink.texts[1], "type") == "alarm_action");
  CHECK(field(sink.texts[1], "action") == "snooze");
  CHECK(field(sink.texts[1], "alarm_id") == "occ-1");
  CHECK(field(sink.texts[3], "action") == "stop");
  CHECK(field(sink.texts[3], "alarm_id") == "occ-2");
  CHECK_EQ(sink.stops, 2);
}

MIST_TEST(an_alarm_state_without_the_occurrence_ringing_stops_the_ring) {
  RecordingSink sink;
  AlarmRules rules(sink);
  rules.on_connected(kT - kSecond);
  FiredAlarm fired{};
  std::strcpy(fired.alarm_id, "occ-1");
  fired.scheduled_for_ms = kT;
  fired.ring_timeout_ms = kTimeout;
  rules.on_alarm_fired(fired, kT);
  ScheduleCopy still = schedule_of({});
  std::strcpy(still.ringing[0], "occ-1");
  still.ringing_count = 1;

  rules.on_alarm_state(still);
  CHECK(rules.ringing());
  rules.on_alarm_state(schedule_of({}));

  CHECK(!rules.ringing());
  CHECK_EQ(sink.stops, 1);
  CHECK_EQ(sink.texts.size(), 1);
}

MIST_TEST(a_server_ring_stops_itself_at_its_timeout) {
  RecordingSink sink;
  AlarmRules rules(sink);
  rules.on_connected(kT - kSecond);
  FiredAlarm fired{};
  std::strcpy(fired.alarm_id, "occ-1");
  fired.scheduled_for_ms = kT;
  fired.ring_timeout_ms = 300 * kSecond;
  rules.on_alarm_fired(fired, kT);

  rules.tick(kT + 300 * kSecond - 1);
  CHECK(rules.ringing());
  rules.tick(kT + 300 * kSecond);

  CHECK(!rules.ringing());
}

MIST_TEST(a_server_ring_stopped_while_offline_is_reported_on_reconnection) {
  RecordingSink sink;
  AlarmRules rules(sink);
  rules.on_connected(kT - kSecond);
  FiredAlarm fired{};
  std::strcpy(fired.alarm_id, "occ-1");
  fired.scheduled_for_ms = kT;
  fired.ring_timeout_ms = kTimeout;
  rules.on_alarm_fired(fired, kT);
  acknowledge_reports(&rules, sink);
  rules.on_disconnected(kT + kSecond);

  CHECK(rules.on_stop(kT + 2 * kSecond));
  rules.on_connected(kT + 10 * kSecond);

  CHECK_EQ(sink.texts.size(), 2);
  CHECK(field(sink.texts[1], "outcome") == "stopped");
  CHECK(field(sink.texts[1], "rang_at") == iso(kT));
  CHECK(field(sink.texts[1], "at") == iso(kT + 2 * kSecond));
}

MIST_TEST(an_occurrence_the_server_rang_never_rings_again_locally) {
  RecordingSink sink;
  AlarmRules rules(sink);
  rules.on_alarm_state(schedule_of({{"occ-1", kT}}));
  rules.on_connected(kT - kSecond);
  FiredAlarm fired{};
  std::strcpy(fired.alarm_id, "occ-1");
  fired.scheduled_for_ms = kT;
  fired.ring_timeout_ms = kTimeout;
  rules.on_alarm_fired(fired, kT);
  rules.on_stop(kT + kSecond);

  rules.on_disconnected(kT);
  rules.tick(kT + kFallback);

  CHECK_EQ(sink.starts.size(), 1);
}

MIST_TEST(stop_and_snooze_without_a_ring_do_nothing) {
  RecordingSink sink;
  AlarmRules rules(sink);
  rules.on_connected(kT);

  CHECK(!rules.on_stop(kT));
  CHECK(!rules.on_snooze(kT));

  CHECK(sink.texts.empty());
  CHECK_EQ(sink.stops, 0);
}

// -- What survives a reboot ------------------------------------------------------------------

namespace {

// The device's memory between two boots, as the alarm component keeps it in preferences.
mist_core::AlarmMemory g_memory;

bool reboot(const AlarmRules& before, AlarmRules* after) {
  before.save(&g_memory);
  return after->restore(g_memory);
}

}  // namespace

MIST_TEST(a_restored_schedule_copy_rings_locally_after_a_reboot) {
  RecordingSink sink;
  AlarmRules before(sink);
  before.on_connected(kT - 3600 * kSecond);
  before.on_alarm_state(schedule_of({{"a1", kT}}));
  AlarmRules after(sink);

  CHECK(reboot(before, &after));
  after.tick(kT + kFallback);

  CHECK_STREQ(after.schedule().entries[0].alarm_id, "a1");
  CHECK_EQ(after.schedule().snooze_minutes, 9);
  CHECK(after.ring_source() == RingSource::Local);
  CHECK_EQ(sink.starts.size(), 1);
}

MIST_TEST(an_occurrence_rung_before_a_reboot_never_rings_again_after_it) {
  RecordingSink sink;
  AlarmRules before(sink);
  before.on_alarm_state(schedule_of({{"a1", kT}}));
  before.tick(kT + kFallback);
  CHECK(before.on_stop(kT + kFallback + kSecond));
  AlarmRules after(sink);

  CHECK(reboot(before, &after));
  after.tick(kT + kFallback + 10 * kSecond);

  CHECK_EQ(sink.starts.size(), 1);
}

MIST_TEST(an_outcome_unsent_at_a_reboot_is_reported_once_on_the_next_connection) {
  RecordingSink sink;
  AlarmRules before(sink);
  before.on_alarm_state(schedule_of({{"a1", kT}}));
  before.tick(kT + kFallback);
  CHECK(before.on_snooze(kT + kFallback + 2 * kSecond));
  CHECK(sink.texts.empty());
  AlarmRules after(sink);

  CHECK(reboot(before, &after));
  after.on_connected(kT + 300 * kSecond);
  after.tick(kT + 301 * kSecond);

  CHECK_EQ(sink.texts.size(), 1);
  CHECK(field(sink.texts[0], "outcome") == "snoozed");
  CHECK(field(sink.texts[0], "rang_at") == iso(kT + kFallback));
  CHECK(field(sink.texts[0], "at") == iso(kT + kFallback + 2 * kSecond));
  acknowledge_reports(&after, sink);
  CHECK_EQ(after.pending_reports(), 0);
}

MIST_TEST(a_local_ring_cut_short_by_a_reboot_is_reported_rang) {
  RecordingSink sink;
  AlarmRules before(sink);
  before.on_alarm_state(schedule_of({{"a1", kT}}));
  const int64_t start = kT + kFallback;
  before.tick(start);
  CHECK(before.ringing());
  AlarmRules after(sink);

  CHECK(reboot(before, &after));
  after.tick(start + 5 * kSecond);
  after.on_connected(start + 60 * kSecond);

  CHECK(!after.ringing());
  CHECK_EQ(sink.starts.size(), 1);
  CHECK_EQ(sink.texts.size(), 1);
  CHECK(field(sink.texts[0], "outcome") == "rang");
  CHECK(field(sink.texts[0], "at") == iso(start));
  CHECK(field(sink.texts[0], "rang_at") == iso(start));
}

MIST_TEST(request_ids_after_a_reboot_never_repeat_one_from_before_it) {
  RecordingSink sink;
  AlarmRules before(sink);
  before.on_alarm_state(schedule_of({{"a1", kT}, {"a2", kT + kSecond}}));
  before.tick(kT + kFallback + kSecond);
  CHECK(before.on_stop(kT + kFallback + 2 * kSecond));
  AlarmRules after(sink);

  CHECK(reboot(before, &after));
  after.tick(kT + kFallback + 3 * kSecond);
  CHECK(after.on_stop(kT + kFallback + 4 * kSecond));
  after.on_connected(kT + 300 * kSecond);

  CHECK_EQ(sink.texts.size(), 2);
  CHECK(field(sink.texts[0], "alarm_id") == "a1");
  CHECK(field(sink.texts[1], "alarm_id") == "a2");
  CHECK(field(sink.texts[0], "request_id") != field(sink.texts[1], "request_id"));
}

MIST_TEST(a_memory_of_another_layout_or_out_of_range_is_refused) {
  RecordingSink sink;
  AlarmRules before(sink);
  before.on_alarm_state(schedule_of({{"a1", kT}}));
  AlarmRules after(sink);

  before.save(&g_memory);
  g_memory.version = mist_core::AlarmMemory::kVersion + 1;
  CHECK(!after.restore(g_memory));
  before.save(&g_memory);
  g_memory.pending_count = AlarmRules::kMaxPending + 1;
  CHECK(!after.restore(g_memory));
  before.save(&g_memory);
  g_memory.schedule.count = ScheduleCopy::kMaxEntries + 1;
  CHECK(!after.restore(g_memory));

  CHECK(!after.schedule().valid);
  after.tick(kT + kFallback);
  CHECK(sink.starts.empty());
}

MIST_TEST(a_saved_memory_is_the_same_bytes_for_the_same_state) {
  RecordingSink sink;
  AlarmRules rules(sink);
  rules.on_alarm_state(schedule_of({{"a1", kT}}));
  static mist_core::AlarmMemory first;
  static mist_core::AlarmMemory second;
  std::memset(static_cast<void*>(&first), 0xA5, sizeof(first));
  std::memset(static_cast<void*>(&second), 0x5A, sizeof(second));

  rules.save(&first);
  rules.save(&second);

  // Preferences write only a changed blob, so padding must not differ between two saves.
  CHECK(std::memcmp(&first, &second, sizeof(first)) == 0);
}

MIST_TEST(every_change_that_survives_a_reboot_advances_the_revision) {
  RecordingSink sink;
  AlarmRules rules(sink);
  uint32_t seen = rules.revision();
  auto advanced = [&]() {
    const bool moved = rules.revision() != seen;
    seen = rules.revision();
    return moved;
  };

  rules.on_alarm_state(schedule_of({{"a1", kT}}));
  CHECK(advanced());
  rules.tick(kT + kSecond);
  CHECK(!advanced());
  rules.tick(kT + kFallback);
  CHECK(advanced());  // a local ring, remembered
  rules.on_stop(kT + kFallback + kSecond);
  CHECK(advanced());  // its outcome, queued
  rules.on_connected(kT + 100 * kSecond);
  advanced();
  acknowledge_reports(&rules, sink);
  CHECK(advanced());  // the outcome retired
}

// -- The clock -------------------------------------------------------------------------------

MIST_TEST(the_clock_is_unset_until_sntp_or_the_owner_sets_it) {
  mist_core::AlarmClock clock;

  CHECK_EQ(clock.now(0, 5000), 0);
  CHECK_EQ(clock.now(kT, 5000), 0);  // a system time SNTP has not confirmed
}

MIST_TEST(without_sntp_the_clock_runs_from_the_owners_alarm_state_at) {
  mist_core::AlarmClock clock;

  clock.on_owner_time(kT, 5000);

  CHECK(clock.now(0, 5000) == kT);
  CHECK(clock.now(0, 65000) == kT + 60 * kSecond);
  clock.on_owner_time(kT + 120 * kSecond, 100000);  // a later `at` sets it again
  CHECK(clock.now(0, 100500) == kT + 120 * kSecond + 500);
}

MIST_TEST(a_synced_sntp_clock_wins_over_the_owners_time) {
  mist_core::AlarmClock clock;
  clock.on_owner_time(kT, 5000);

  clock.on_sntp_synced();

  CHECK(clock.now(kT + 7 * kSecond, 6000) == kT + 7 * kSecond);
  // A system time before 2024 is a clock never set, whatever SNTP said.
  CHECK(clock.now(1000, 6000) == kT + kSecond);
}

MIST_TEST(an_owner_time_before_2024_is_ignored) {
  mist_core::AlarmClock clock;

  clock.on_owner_time(1000, 5000);

  CHECK_EQ(clock.now(0, 5000), 0);
}

// -- The tone --------------------------------------------------------------------------------

namespace {

std::vector<int16_t> one_cycle() {
  mist_core::AlarmTone tone;
  std::vector<int16_t> samples(mist_core::AlarmTone::kCycleSamples);
  // In uneven pieces, as the speaker takes them.
  size_t at = 0;
  size_t piece = 1;
  while (at < samples.size()) {
    const size_t count = std::min(piece, samples.size() - at);
    tone.fill(samples.data() + at, count);
    at += count;
    piece = piece * 3 % 997 + 1;
  }
  return samples;
}

constexpr size_t kBeepSamples =
    mist_core::AlarmTone::kSampleRate * mist_core::AlarmTone::kBeepMs / 1000;
constexpr size_t kSlotSamples =
    kBeepSamples + mist_core::AlarmTone::kSampleRate * mist_core::AlarmTone::kGapMs / 1000;

}  // namespace

MIST_TEST(the_tone_peaks_within_3_db_of_full_scale_and_never_clips) {
  const std::vector<int16_t> samples = one_cycle();
  int peak = 0;
  for (int16_t sample : samples) peak = std::max(peak, std::abs(static_cast<int>(sample)));

  const double dbfs = 20.0 * std::log10(peak / 32767.0);
  CHECK(dbfs >= -3.0);
  CHECK(dbfs < 0.0);
}

MIST_TEST(the_tone_beeps_at_880_hz_four_times_a_cycle) {
  const std::vector<int16_t> samples = one_cycle();
  CHECK_EQ(mist_core::AlarmTone::kBeeps, 4);
  CHECK(samples.size() > 4 * kSlotSamples);

  for (size_t beep = 0; beep < 4; ++beep) {
    const size_t start = beep * kSlotSamples;
    int crossings = 0;
    for (size_t i = start + 1; i < start + kBeepSamples; ++i) {
      if ((samples[i - 1] < 0) != (samples[i] < 0)) ++crossings;
    }
    // 880 Hz for 100 ms: 88 cycles, two sign changes each.
    CHECK(std::abs(crossings - 176) <= 2);
    for (size_t i = start + kBeepSamples; i < start + kSlotSamples; ++i) CHECK_EQ(samples[i], 0);
  }
  for (size_t i = 4 * kSlotSamples; i < samples.size(); ++i) CHECK_EQ(samples[i], 0);
}

MIST_TEST(each_beep_ramps_in_and_out_without_a_click) {
  const std::vector<int16_t> samples = one_cycle();

  CHECK(std::abs(static_cast<int>(samples[0])) < 200);
  CHECK(std::abs(static_cast<int>(samples[kBeepSamples - 1])) < 200);
  // The first 1 ms stays far under the peak: the ramp, not the sine, sets the onset.
  for (size_t i = 0; i < 16; ++i) CHECK(std::abs(static_cast<int>(samples[i])) < 4000);
}

MIST_TEST(the_tone_starts_again_at_its_first_beep_after_a_reset) {
  mist_core::AlarmTone tone;
  int16_t first[64];
  int16_t later[64];
  static int16_t skip[5000];
  tone.fill(first, 64);
  tone.fill(skip, 5000);

  tone.reset();
  tone.fill(later, 64);

  CHECK(std::memcmp(first, later, sizeof(first)) == 0);
}
