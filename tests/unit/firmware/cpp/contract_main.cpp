// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.
//
// Drives the core through scripted scenarios and prints every message the device would send, so
// `tests/unit/firmware/test_core.py` can validate each one with the backend's own models.
//
// Output, one item per line:
//   SCENARIO <name>
//   TEXT <device clock in ms since the epoch> <JSON text frame>
//   BINARY <hex of the whole binary frame>

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "alarm_rules.h"
#include "codec.h"
#include "ring.h"
#include "tag_queue.h"
#include "ui_state.h"
#include "uplink.h"

namespace {

constexpr int64_t kT = 1794898800000;  // 2026-11-17T07:00:00Z
constexpr int64_t kSecond = 1000;

int64_t g_now = kT;
int g_failures = 0;

void scenario(const char* name) { std::printf("SCENARIO %s\n", name); }

void emit_text(const char* text, size_t len) {
  if (len == 0) {
    ++g_failures;
    std::printf("ERROR a builder refused its input\n");
    return;
  }
  std::printf("TEXT %lld %.*s\n", static_cast<long long>(g_now), static_cast<int>(len), text);
}

void emit_binary(const uint8_t* data, size_t len) {
  std::printf("BINARY ");
  for (size_t i = 0; i < len; ++i) std::printf("%02x", data[i]);
  std::printf("\n");
}

class PrintingUplinkSink : public mist_core::UplinkSink {
 public:
  bool send_text(const char* text, size_t len) override {
    emit_text(text, len);
    return true;
  }
  bool send_binary(const uint8_t* data, size_t len) override {
    emit_binary(data, len);
    return true;
  }
};

class PrintingAlarmSink : public mist_core::AlarmSink {
 public:
  bool send_text(const char* text, size_t len) override {
    emit_text(text, len);
    return true;
  }
  void start_ring(const char*) override {}
  void stop_ring() override {}
};

alignas(4) uint8_t g_storage[mist_core::PcmRing::kFrameBytes * 300];

void write_speech(mist_core::PcmRing* ring, uint32_t frames) {
  int16_t frame[mist_core::PcmRing::kFrameSamples];
  for (uint32_t k = 0; k < frames; ++k) {
    for (size_t i = 0; i < mist_core::PcmRing::kFrameSamples; ++i) {
      frame[i] = static_cast<int16_t>((static_cast<int>(i) * 37 + static_cast<int>(k)) % 2000 - 1000);
    }
    ring->write(frame, mist_core::PcmRing::kFrameSamples);
  }
}

void connect_scenario() {
  scenario("connect");
  char buf[512];
  emit_text(buf, mist_core::build_voice_start(buf, sizeof(buf), 1, false));
  emit_text(buf, mist_core::build_voice_start(buf, sizeof(buf), 2, true));
}

void uplink_scenario() {
  scenario("uplink");
  mist_core::PcmRing ring(g_storage, sizeof(g_storage));
  PrintingUplinkSink sink;
  mist_core::Uplink uplink(ring, sink);
  uplink.on_connected();
  uplink.on_voice_state(true);
  write_speech(&ring, 120);

  uplink.on_wake(0.82f, 0);
  write_speech(&ring, 3);
  uplink.pump(60);
  uplink.on_voice_decision(mist_core::DecisionKind::WakeRejected);
  write_speech(&ring, 3);
  uplink.pump(120);

  scenario("barge_in");
  uplink.on_state_cycle(mist_core::MistState::Speak, "turn-1");
  uplink.on_stop_keyword();

  scenario("mute");
  uplink.on_mute(true);
  uplink.on_mute(false);
}

void player_scenario() {
  scenario("player");
  mist_core::TagQueue queue;
  queue.on_stream_start("turn-2", true, 4);
  queue.on_audio_chunk(4, 24 * 400);
  queue.on_audio_complete(4);
  char buf[256];
  uint32_t now = 0;
  for (int step = 0; step < 40; ++step) {
    queue.on_played(24 * 20, now);
    const mist_core::PlaybackReport report = queue.poll(now);
    if (report.kind == mist_core::ReportKind::Progress) {
      emit_text(buf, mist_core::build_playback_progress(buf, sizeof(buf), report.turn_id,
                                                        report.played_ms));
    } else if (report.kind == mist_core::ReportKind::Done) {
      emit_text(buf,
                mist_core::build_playback_done(buf, sizeof(buf), report.turn_id, report.played_ms));
      break;
    }
    now += 20;
  }
}

void button_scenario() {
  scenario("button");
  mist_core::UiState ui;
  ui.set_connected(true);
  ui.set_state(mist_core::MistState::Speak);
  if (ui.on_button(mist_core::ButtonEvent::SinglePress) == mist_core::ButtonAction::Interrupt) {
    char buf[64];
    emit_text(buf, mist_core::build_interrupt(buf, sizeof(buf)));
  }
}

mist_core::ScheduleCopy schedule(const char* alarm_id, int64_t next_fire_at) {
  mist_core::ScheduleCopy copy{};
  copy.valid = true;
  copy.at_ms = next_fire_at - 3600 * kSecond;
  copy.fallback_after_ms = 30 * kSecond;
  copy.missed_grace_ms = 900 * kSecond;
  copy.ring_timeout_ms = 120 * kSecond;
  copy.snooze_minutes = 9;
  std::strcpy(copy.tz, "UTC");
  std::strcpy(copy.entries[0].alarm_id, alarm_id);
  copy.entries[0].next_fire_at_ms = next_fire_at;
  copy.count = 1;
  return copy;
}

void local_ring(const char* alarm_id, int64_t fire_at, int outcome) {
  PrintingAlarmSink sink;
  mist_core::AlarmRules rules(sink);
  rules.on_alarm_state(schedule(alarm_id, fire_at));
  g_now = fire_at + 30 * kSecond;
  rules.tick(g_now);
  g_now += 20 * kSecond;
  if (outcome == 0) {
    g_now += 120 * kSecond;
    rules.tick(g_now);
  } else if (outcome == 1) {
    rules.on_stop(g_now);
  } else {
    rules.on_snooze(g_now);
  }
  g_now += 10 * 60 * kSecond;
  rules.on_connected(g_now);
}

void alarm_scenarios() {
  scenario("local_ring_timed_out");
  local_ring("occ-local-1", kT, 0);
  scenario("local_ring_stopped");
  local_ring("occ-local-2", kT + 3600 * kSecond, 1);
  scenario("local_ring_snoozed");
  local_ring("occ-local-3", kT + 7200 * kSecond, 2);

  scenario("server_ring");
  PrintingAlarmSink sink;
  mist_core::AlarmRules rules(sink);
  g_now = kT + 10 * 3600 * kSecond;
  rules.on_connected(g_now);
  mist_core::FiredAlarm fired{};
  std::strcpy(fired.alarm_id, "occ-server-1");
  fired.scheduled_for_ms = g_now;
  fired.ring_timeout_ms = 120 * kSecond;
  rules.on_alarm_fired(fired, g_now);
  g_now += 4 * kSecond;
  rules.on_snooze(g_now);
  g_now += 9 * 60 * kSecond;
  std::strcpy(fired.alarm_id, "occ-server-2");
  fired.scheduled_for_ms = g_now;
  rules.on_alarm_fired(fired, g_now);
  g_now += 2 * kSecond;
  rules.on_stop(g_now);
}

// A local ring snoozed offline, then a reboot before the link returns: the outcome kept in the
// device's memory is reported on the first connection after it.
void reboot_scenario() {
  scenario("local_ring_across_a_reboot");
  static mist_core::AlarmMemory memory;
  PrintingAlarmSink sink;
  const int64_t fire_at = kT + 4 * 3600 * kSecond;
  {
    mist_core::AlarmRules before(sink);
    before.on_alarm_state(schedule("occ-reboot-1", fire_at));
    g_now = fire_at + 30 * kSecond;
    before.tick(g_now);
    g_now += 3 * kSecond;
    before.on_snooze(g_now);
    before.save(&memory);
  }
  mist_core::AlarmRules after(sink);
  if (!after.restore(memory)) {
    ++g_failures;
    return;
  }
  g_now += 5 * 60 * kSecond;
  after.on_connected(g_now);
}

// No SNTP: the device's clock runs from `alarm_state.at`, which reached it `kLatencyMs` after the
// owner wrote it, on a crystal `kFastPpm` fast. Each report is printed at the owner's own time,
// which `test_core.py` validates it against.
void owner_clock_scenario() {
  scenario("owner_clock");
  constexpr int64_t kLatencyMs = 250;
  constexpr int64_t kFastPpm = 100;
  const int64_t at = kT + 6 * 3600 * kSecond;
  const int64_t boot_mono = 10 * kSecond;
  // The device's monotonic time when the owner's clock reads `owner_ms`.
  auto mono_at = [&](int64_t owner_ms) {
    const int64_t elapsed = owner_ms - (at + kLatencyMs);
    return boot_mono + elapsed + elapsed * kFastPpm / 1000000;
  };

  mist_core::AlarmClock clock;
  clock.on_owner_time(at, boot_mono);
  PrintingAlarmSink sink;
  mist_core::AlarmRules rules(sink);
  rules.on_alarm_state(schedule("occ-clock-1", at + 60 * kSecond));
  rules.on_disconnected(clock.now(0, mono_at(at + 2 * kSecond)));

  for (g_now = at + 2 * kSecond; g_now <= at + 20 * 60 * kSecond; g_now += kSecond) {
    rules.tick(clock.now(0, mono_at(g_now)));
  }
  rules.on_connected(clock.now(0, mono_at(g_now)));
}

void stats_scenario() {
  scenario("device_stats");
  mist_core::DeviceStats stats{};
  stats.heap_free_min = 81234;
  stats.psram_free = 6123456;
  stats.mic_overruns = 0;
  stats.ring_overruns = 2;
  stats.playback_underruns = 1;
  stats.uptime_s = 3600;
  stats.reconnects = 1;
  char buf[256];
  emit_text(buf, mist_core::build_device_stats(buf, sizeof(buf), stats));
}

}  // namespace

int main() {
  // The start marker, as in check.h: the pytest driver may restart a run that crashed before it.
  std::printf("MIST_TEST_START\n");
  std::fflush(stdout);
  connect_scenario();
  uplink_scenario();
  player_scenario();
  button_scenario();
  alarm_scenarios();
  reboot_scenario();
  owner_clock_scenario();
  stats_scenario();
  return g_failures == 0 ? 0 : 1;
}
