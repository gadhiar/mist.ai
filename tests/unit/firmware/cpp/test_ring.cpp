// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.
//
// Host tests for the pre-wake PCM ring: wrap-around, `read_last`, overwrite-oldest, overrun
// counting, clearing, and one producer thread against one consumer thread.

#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>

#include "check.h"
#include "ring.h"

using mist_core::PcmRing;

namespace {

constexpr size_t kFrame = PcmRing::kFrameSamples;

int16_t sample_of(uint32_t index, size_t i) {
  if (i == 0) return static_cast<int16_t>(static_cast<uint16_t>(index & 0xFFFFu));
  if (i == 1) return static_cast<int16_t>(static_cast<uint16_t>(index >> 16));
  return static_cast<int16_t>(static_cast<uint16_t>(index * 7u + static_cast<uint32_t>(i)));
}

// The first two samples of frame `index` carry the index and every other sample a value derived
// from it, so a torn frame (part of one frame and part of another) is detectable.
void fill_frame(int16_t* frame, uint32_t index) {
  for (size_t i = 0; i < kFrame; ++i) frame[i] = sample_of(index, i);
}

bool frame_is(const int16_t* frame, uint32_t index) {
  for (size_t i = 0; i < kFrame; ++i) {
    if (frame[i] != sample_of(index, i)) return false;
  }
  return true;
}

uint32_t index_of(const int16_t* frame) {
  return static_cast<uint32_t>(static_cast<uint16_t>(frame[0])) |
         (static_cast<uint32_t>(static_cast<uint16_t>(frame[1])) << 16);
}

void write_frames(PcmRing* ring, uint32_t first, uint32_t count) {
  int16_t frame[kFrame];
  for (uint32_t k = 0; k < count; ++k) {
    fill_frame(frame, first + k);
    ring->write(frame, kFrame);
  }
}

}  // namespace

MIST_TEST(storage_size_and_alignment_are_checked) {
  alignas(4) static uint8_t storage[PcmRing::kFrameBytes * 5 + 8];

  PcmRing ring(storage, sizeof(storage));
  PcmRing too_small(storage, PcmRing::kFrameBytes * 2 - 1);
  PcmRing misaligned(storage + 2, PcmRing::kFrameBytes * 2);

  // Five slots hold four readable frames: the fifth is the one the producer is filling.
  CHECK(ring.valid());
  CHECK_EQ(ring.capacity_frames(), 4);
  CHECK(!too_small.valid());
  CHECK(!misaligned.valid());
  CHECK_EQ(PcmRing::bytes_for_ms(6000), 301 * PcmRing::kFrameBytes);
  alignas(4) static uint8_t device[PcmRing::bytes_for_ms(6000)];
  CHECK_EQ(PcmRing(device, sizeof(device)).capacity_frames(), 300);
}

MIST_TEST(an_empty_ring_reads_nothing) {
  alignas(4) static uint8_t storage[PcmRing::kFrameBytes * 4];
  PcmRing ring(storage, sizeof(storage));
  int16_t out[kFrame];

  CHECK_EQ(ring.buffered_frames(), 0);
  CHECK_EQ(ring.read_last(5000), 0);
  CHECK(!ring.read_frame(out));
}

MIST_TEST(partial_writes_publish_only_whole_frames) {
  alignas(4) static uint8_t storage[PcmRing::kFrameBytes * 4];
  PcmRing ring(storage, sizeof(storage));
  int16_t frame[kFrame];
  fill_frame(frame, 3);
  int16_t out[kFrame];

  ring.write(frame, 101);
  CHECK_EQ(ring.buffered_frames(), 0);
  ring.write(frame + 101, 77);
  ring.write(frame + 178, kFrame - 178 - 1);
  CHECK_EQ(ring.buffered_frames(), 0);
  ring.write(frame + kFrame - 1, 1);

  CHECK_EQ(ring.buffered_frames(), 1);
  CHECK_EQ(ring.read_last(20), 1);
  CHECK(ring.read_frame(out));
  CHECK(frame_is(out, 3));
}

MIST_TEST(read_last_counts_whole_frames_and_is_bounded_by_what_is_buffered) {
  alignas(4) static uint8_t storage[PcmRing::kFrameBytes * 8];
  PcmRing ring(storage, sizeof(storage));
  write_frames(&ring, 0, 5);
  int16_t out[kFrame];

  CHECK_EQ(ring.read_last(59), 2);
  CHECK(ring.read_frame(out));
  CHECK(frame_is(out, 3));
  CHECK_EQ(ring.read_last(19), 0);
  CHECK(!ring.read_frame(out));
  CHECK_EQ(ring.read_last(1000), 5);
  CHECK(ring.read_frame(out));
  CHECK(frame_is(out, 0));
}

MIST_TEST(writing_past_capacity_overwrites_the_oldest_frames) {
  alignas(4) static uint8_t storage[PcmRing::kFrameBytes * 5];
  PcmRing ring(storage, sizeof(storage));
  write_frames(&ring, 0, 10);
  int16_t out[kFrame];

  CHECK_EQ(ring.buffered_frames(), 4);
  CHECK_EQ(ring.read_last(1000), 4);
  for (uint32_t index = 6; index < 10; ++index) {
    CHECK(ring.read_frame(out));
    CHECK(frame_is(out, index));
  }
  CHECK(!ring.read_frame(out));
  CHECK_EQ(ring.overruns(), 0);
}

MIST_TEST(reads_follow_live_writes_across_the_wrap) {
  alignas(4) static uint8_t storage[PcmRing::kFrameBytes * 3];
  PcmRing ring(storage, sizeof(storage));
  int16_t out[kFrame];
  CHECK_EQ(ring.read_last(0), 0);

  for (uint32_t index = 0; index < 20; ++index) {
    write_frames(&ring, index, 1);
    CHECK_EQ(ring.available_frames(), 1);
    CHECK(ring.read_frame(out));
    CHECK(frame_is(out, index));
  }
  CHECK_EQ(ring.overruns(), 0);
}

MIST_TEST(a_reader_overtaken_by_the_writer_skips_to_the_oldest_frame_and_counts_the_loss) {
  alignas(4) static uint8_t storage[PcmRing::kFrameBytes * 5];
  PcmRing ring(storage, sizeof(storage));
  write_frames(&ring, 0, 3);
  CHECK_EQ(ring.read_last(1000), 3);
  int16_t out[kFrame];
  CHECK(ring.read_frame(out));
  CHECK(frame_is(out, 0));

  write_frames(&ring, 3, 5);

  CHECK(ring.read_frame(out));
  CHECK(frame_is(out, 4));
  CHECK_EQ(ring.overruns(), 3);
}

MIST_TEST(clear_drops_every_buffered_frame_and_the_frame_in_progress) {
  alignas(4) static uint8_t storage[PcmRing::kFrameBytes * 4];
  PcmRing ring(storage, sizeof(storage));
  write_frames(&ring, 0, 3);
  int16_t frame[kFrame];
  fill_frame(frame, 3);
  ring.write(frame, 100);
  int16_t out[kFrame];

  ring.clear();

  CHECK_EQ(ring.buffered_frames(), 0);
  CHECK_EQ(ring.read_last(1000), 0);
  CHECK(!ring.read_frame(out));
  // The 100 samples written before the clear are dropped, so 220 more do not complete a frame.
  ring.write(frame + 100, kFrame - 100);
  CHECK_EQ(ring.buffered_frames(), 0);
  CHECK(!ring.read_frame(out));
  ring.write(frame, 100);
  CHECK_EQ(ring.buffered_frames(), 1);
  write_frames(&ring, 4, 2);
  CHECK_EQ(ring.buffered_frames(), 3);
  CHECK(ring.read_frame(out));
  CHECK(ring.read_frame(out));
  CHECK(frame_is(out, 4));
  CHECK_EQ(ring.overruns(), 0);
}

MIST_TEST(one_producer_and_one_consumer_thread_lose_nothing_but_counted_overruns) {
  alignas(4) static uint8_t storage[PcmRing::kFrameBytes * 4];
  PcmRing ring(storage, sizeof(storage));
  constexpr uint32_t kFrames = 20000;
  std::atomic<bool> done{false};
  uint32_t received = 0;
  uint32_t torn = 0;
  uint32_t out_of_order = 0;
  uint32_t last = 0;
  bool any = false;

  std::thread producer([&ring, &done] {
    int16_t frame[kFrame];
    for (uint32_t index = 0; index < kFrames; ++index) {
      fill_frame(frame, index);
      // Two uneven writes per frame, so the consumer also races partial frames.
      ring.write(frame, 123);
      ring.write(frame + 123, kFrame - 123);
      if (index % 64 == 0) std::this_thread::yield();
    }
    done.store(true, std::memory_order_release);
  });
  std::thread consumer([&] {
    int16_t out[kFrame];
    for (;;) {
      const bool finished = done.load(std::memory_order_acquire);
      bool got = false;
      while (ring.read_frame(out)) {
        got = true;
        const uint32_t found = index_of(out);
        if (found >= kFrames || !frame_is(out, found)) {
          ++torn;
          continue;
        }
        if (any && found <= last) ++out_of_order;
        last = found;
        any = true;
        ++received;
      }
      if (finished && !got) break;
    }
  });
  producer.join();
  consumer.join();

  CHECK_EQ(torn, 0);
  CHECK_EQ(out_of_order, 0);
  CHECK(any);
  CHECK_EQ(last, kFrames - 1);
  CHECK_EQ(received + ring.overruns(), kFrames);
}
