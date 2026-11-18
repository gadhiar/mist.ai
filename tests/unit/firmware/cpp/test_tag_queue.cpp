// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.
//
// Host tests for the playback tag rules (plan Design item 5): `stream_start.audio_tag` names the
// turn whose `0x01` frames play and every other tag is dropped; the prebuffer; progress at its
// interval; `playback_done` once; fade and flush on `playback_stop` and `0x03`; ducking.

#include <cstdint>
#include <cstring>

#include "check.h"
#include "tag_queue.h"

using mist_core::PlaybackParams;
using mist_core::PlaybackReport;
using mist_core::ReportKind;
using mist_core::StopDecision;
using mist_core::TagQueue;
using mist_core::TagVerdict;

namespace {

// 24 kHz: 24 samples per millisecond.
constexpr uint32_t ms(uint32_t value) { return value * 24u; }

}  // namespace

MIST_TEST(frames_before_any_stream_start_are_dropped) {
  TagQueue queue;

  CHECK(queue.on_audio_chunk(1, ms(40)) == TagVerdict::Drop);
  CHECK(!queue.on_audio_complete(1));
  CHECK(!queue.on_interrupt_fade(1).fade);
}

MIST_TEST(only_frames_of_the_current_tag_play) {
  TagQueue queue;
  CHECK(!queue.on_stream_start("turn-1", true, 5));

  CHECK(queue.on_audio_chunk(5, ms(40)) == TagVerdict::Play);
  CHECK(queue.on_audio_chunk(4, ms(40)) == TagVerdict::Drop);
  CHECK(queue.on_audio_chunk(6, ms(40)) == TagVerdict::Drop);
  CHECK_EQ(queue.queued_samples(), ms(40));
}

MIST_TEST(a_turn_without_an_audio_tag_plays_nothing) {
  TagQueue queue;
  queue.on_stream_start("turn-1", true, 5);
  queue.on_audio_chunk(5, ms(200));

  CHECK(queue.on_stream_start("turn-2", false, 0));

  CHECK(queue.on_audio_chunk(5, ms(40)) == TagVerdict::Drop);
  CHECK(queue.on_audio_chunk(0, ms(40)) == TagVerdict::Drop);
  CHECK(!queue.playing());
}

MIST_TEST(a_new_turn_takes_over_and_flushes_an_unfinished_one) {
  TagQueue queue;
  queue.on_stream_start("turn-1", true, 65535);
  queue.on_audio_chunk(65535, ms(200));

  CHECK(queue.on_stream_start("turn-2", true, 1));

  CHECK(queue.on_audio_chunk(65535, ms(40)) == TagVerdict::Drop);
  CHECK(queue.on_audio_chunk(1, ms(40)) == TagVerdict::Play);
  CHECK_EQ(queue.queued_samples(), ms(40));
}

MIST_TEST(a_new_turn_after_a_finished_one_flushes_nothing) {
  TagQueue queue;
  queue.on_stream_start("turn-1", true, 2);
  queue.on_audio_chunk(2, ms(100));
  queue.on_audio_complete(2);
  queue.on_played(ms(100), 0);
  CHECK(queue.poll(0).kind == ReportKind::Done);

  CHECK(!queue.on_stream_start("turn-2", true, 3));
}

MIST_TEST(playback_waits_for_the_prebuffer) {
  TagQueue queue;
  queue.on_stream_start("turn-1", true, 9);

  queue.on_audio_chunk(9, ms(80) - 1);
  CHECK(!queue.playing());
  queue.on_audio_chunk(9, 1);

  CHECK(queue.playing());
}

MIST_TEST(the_prebuffer_follows_the_server_settings) {
  TagQueue queue;
  PlaybackParams params;
  params.prebuffer_ms = 200;
  queue.set_params(params);
  queue.on_stream_start("turn-1", true, 9);

  queue.on_audio_chunk(9, ms(199));
  CHECK(!queue.playing());
  queue.on_audio_chunk(9, ms(1));

  CHECK(queue.playing());
}

MIST_TEST(a_reply_shorter_than_the_prebuffer_plays_once_complete) {
  TagQueue queue;
  queue.on_stream_start("turn-1", true, 9);
  queue.on_audio_chunk(9, ms(40));
  CHECK(!queue.playing());

  CHECK(queue.on_audio_complete(9));

  CHECK(queue.playing());
}

MIST_TEST(progress_is_reported_at_its_interval_from_the_first_played_sample) {
  TagQueue queue;
  queue.on_stream_start("turn-1", true, 9);
  queue.on_audio_chunk(9, ms(1000));
  CHECK(queue.poll(5000).kind == ReportKind::None);

  queue.on_played(ms(20), 10000);
  CHECK(queue.poll(10099).kind == ReportKind::None);
  queue.on_played(ms(80), 10099);
  const PlaybackReport report = queue.poll(10100);

  CHECK(report.kind == ReportKind::Progress);
  CHECK_STREQ(report.turn_id, "turn-1");
  CHECK_EQ(report.played_ms, 100);
  CHECK(queue.poll(10199).kind == ReportKind::None);
  CHECK(queue.poll(10200).kind == ReportKind::Progress);
}

MIST_TEST(done_is_reported_once_when_the_completed_turn_has_played_out) {
  TagQueue queue;
  queue.on_stream_start("turn-1", true, 9);
  queue.on_audio_chunk(9, ms(300));
  CHECK(queue.on_audio_complete(9));
  queue.on_played(ms(300) - 1, 0);
  CHECK(queue.poll(50).kind == ReportKind::None);

  queue.on_played(1, 60);
  const PlaybackReport report = queue.poll(60);

  CHECK(report.kind == ReportKind::Done);
  CHECK_STREQ(report.turn_id, "turn-1");
  CHECK_EQ(report.played_ms, 300);
  CHECK(queue.poll(1000).kind == ReportKind::None);
  CHECK(queue.on_audio_chunk(9, ms(20)) == TagVerdict::Drop);
}

MIST_TEST(no_done_before_the_audio_complete_marker) {
  TagQueue queue;
  queue.on_stream_start("turn-1", true, 9);
  queue.on_audio_chunk(9, ms(100));

  queue.on_played(ms(100), 0);

  CHECK(queue.poll(50).kind == ReportKind::None);
  CHECK(!queue.on_audio_complete(8));
  CHECK(queue.poll(60).kind == ReportKind::None);
}

MIST_TEST(playback_stop_for_the_current_tag_fades_flushes_and_reports_the_position) {
  TagQueue queue;
  queue.on_stream_start("turn-1", true, 9);
  queue.on_audio_chunk(9, ms(1000));
  queue.on_played(ms(250), 0);

  const StopDecision stop = queue.on_playback_stop(9, 45);

  CHECK(stop.fade);
  CHECK_EQ(stop.fade_ms, 45);
  CHECK(!queue.playing());
  CHECK(queue.on_audio_chunk(9, ms(20)) == TagVerdict::Drop);
  const PlaybackReport report = queue.poll(10);
  CHECK(report.kind == ReportKind::Progress);
  CHECK_EQ(report.played_ms, 250);
  CHECK(queue.poll(1000).kind == ReportKind::None);
  CHECK(!queue.on_audio_complete(9));
  CHECK(queue.poll(2000).kind == ReportKind::None);
}

MIST_TEST(playback_stop_for_another_tag_changes_nothing) {
  TagQueue queue;
  queue.on_stream_start("turn-1", true, 9);
  queue.on_audio_chunk(9, ms(100));

  CHECK(!queue.on_playback_stop(8, 30).fade);

  CHECK(queue.playing());
  CHECK(queue.on_audio_chunk(9, ms(20)) == TagVerdict::Play);
}

MIST_TEST(interrupt_fade_for_the_current_tag_fades_by_the_session_setting) {
  TagQueue queue;
  PlaybackParams params;
  params.fade_ms = 30;
  queue.set_params(params);
  queue.on_stream_start("turn-1", true, 9);
  queue.on_audio_chunk(9, ms(100));

  CHECK(!queue.on_interrupt_fade(10).fade);
  const StopDecision stop = queue.on_interrupt_fade(9);

  CHECK(stop.fade);
  CHECK_EQ(stop.fade_ms, 30);
  CHECK(queue.on_audio_chunk(9, ms(20)) == TagVerdict::Drop);
}

MIST_TEST(duck_applies_to_the_current_turn_only_and_zero_restores) {
  TagQueue queue;
  queue.on_stream_start("turn-1", true, 9);

  CHECK(!queue.on_playback_duck("turn-0", -12.0f));
  CHECK(queue.duck_db() == 0.0f);
  CHECK(queue.on_playback_duck("turn-1", -12.0f));
  CHECK(queue.duck_db() == -12.0f);
  CHECK(queue.on_playback_duck("turn-1", 0.0f));
  CHECK(queue.duck_db() == 0.0f);
  queue.on_playback_duck("turn-1", -6.0f);
  queue.on_stream_start("turn-2", true, 10);
  CHECK(queue.duck_db() == 0.0f);
}

MIST_TEST(duck_never_raises_the_volume) {
  TagQueue queue;
  queue.on_stream_start("turn-1", true, 9);

  queue.on_playback_duck("turn-1", 6.0f);

  CHECK(queue.duck_db() == 0.0f);
}
