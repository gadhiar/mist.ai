// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.
//
// The pre-wake PCM ring (plan Design item 4): 16 kHz mono PCM16 in 20 ms frames over memory the
// caller provides (PSRAM on the device), with one producer (the microphone tap) and one consumer
// (the uplink's task).
//
// The producer never blocks: when the ring is full it overwrites the oldest frame. A consumer that
// falls more than the ring's length behind skips to the oldest frame still held and adds the
// frames it lost to `overruns()`. A frame the producer overwrote while the consumer was copying it
// is detected after the copy (the consumer re-reads the producer's position, as a seqlock reader
// does) and counted the same way, so a torn frame is never returned.
//
// Samples are stored two to a `std::atomic<uint32_t>` word with relaxed loads and stores, so the
// concurrent copy is free of data races in the C++ memory model; on the ESP32-S3 and on x86 these
// compile to plain 32-bit loads and stores. Frame counters run modulo a multiple of the capacity
// below 2^31, so they wrap without a 64-bit atomic (which the ESP32-S3 lacks).

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>

namespace mist_core {

class PcmRing {
 public:
  static constexpr uint32_t kFrameMs = 20;
  static constexpr size_t kFrameSamples = 320;  // 20 ms at 16 kHz
  static constexpr size_t kFrameBytes = kFrameSamples * 2;
  static constexpr size_t kFrameWords = kFrameSamples / 2;

  // Bytes of storage that hold `ms` of readable audio in whole frames: one slot more than the
  // frames, because the slot the producer is filling is never readable.
  static constexpr size_t bytes_for_ms(uint32_t ms) {
    return (ms / kFrameMs + 1) * kFrameBytes;
  }

  // `storage` must be 4-byte aligned and hold at least two frame slots; it must outlive the ring.
  PcmRing(void* storage, size_t bytes) {
    static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t), "packed sample words");
    static_assert(std::atomic<uint32_t>::is_always_lock_free, "lock-free sample words");
    const bool aligned = reinterpret_cast<uintptr_t>(storage) % alignof(std::atomic<uint32_t>) == 0;
    const size_t slots = bytes / kFrameBytes;
    if (storage == nullptr || !aligned || slots < 2 || slots > 0x7FFFFFFFu) return;
    slots_ = static_cast<uint32_t>(slots);
    capacity_ = slots_ - 1;
    modulus_ = slots_ * (0x80000000u / slots_);
    words_ = reinterpret_cast<std::atomic<uint32_t>*>(storage);
    for (size_t i = 0; i < slots * kFrameWords; ++i) new (&words_[i]) std::atomic<uint32_t>(0);
  }

  PcmRing(const PcmRing&) = delete;
  PcmRing& operator=(const PcmRing&) = delete;

  bool valid() const { return capacity_ > 0; }
  // The most frames the ring holds readable: one fewer than its slots.
  uint32_t capacity_frames() const { return capacity_; }

  // -- Producer ------------------------------------------------------------------------------

  // Appends samples; each completed 20 ms frame is published at once. Any count works: a partial
  // frame waits for the next call. A `clear()` drops the partial frame at the next call.
  void write(const int16_t* samples, size_t count) {
    if (!valid()) return;
    const uint32_t generation = clear_generation_.load(std::memory_order_acquire);
    if (generation != seen_generation_) {
      seen_generation_ = generation;
      fill_ = 0;
      has_low_ = false;
    }
    for (size_t i = 0; i < count; ++i) {
      if (fill_ == 0 && !has_low_) {
        // Orders the stores of the frame below after the store that published the frame
        // before it, so a consumer that sees any of them also sees the producer's position.
        std::atomic_thread_fence(std::memory_order_release);
      }
      const uint16_t sample = static_cast<uint16_t>(samples[i]);
      if (!has_low_) {
        low_ = sample;
        has_low_ = true;
        continue;
      }
      const size_t slot = head_ % slots_;
      words_[slot * kFrameWords + fill_].store(
          static_cast<uint32_t>(low_) | (static_cast<uint32_t>(sample) << 16),
          std::memory_order_relaxed);
      has_low_ = false;
      if (++fill_ == kFrameWords) {
        fill_ = 0;
        head_ = add(head_, 1);
        published_.store(head_, std::memory_order_release);
      }
    }
  }

  // -- Consumer: every method below runs on the consumer's thread only ------------------------

  // Whole frames held and readable: at most the capacity, none from before the last `clear()`.
  uint32_t buffered_frames() const { return readable(published_.load(std::memory_order_acquire)); }

  // Points the read cursor at the newest whole frames covering at most `ms`, and returns how
  // many frames that is: min(ms / 20, buffered_frames()).
  uint32_t read_last(uint32_t ms) {
    if (!valid()) return 0;
    const uint32_t head = published_.load(std::memory_order_acquire);
    const uint32_t wanted = ms / kFrameMs;
    const uint32_t held = readable(head);
    const uint32_t frames = wanted < held ? wanted : held;
    cursor_ = sub(head, frames);
    return frames;
  }

  // Frames from the cursor to the newest, at most `buffered_frames()`.
  uint32_t available_frames() const {
    if (!valid()) return 0;
    const uint32_t head = published_.load(std::memory_order_acquire);
    const uint32_t behind = sub(head, cursor_);
    if (behind > modulus_ / 2) return 0;
    const uint32_t held = readable(head);
    return behind < held ? behind : held;
  }

  // Copies the frame at the cursor into `out` (`kFrameSamples` samples) and advances. Returns
  // false when no frame is ready.
  bool read_frame(int16_t* out) {
    if (!valid()) return false;
    for (int attempt = 0; attempt < 8; ++attempt) {
      const uint32_t head = published_.load(std::memory_order_acquire);
      const uint32_t behind = sub(head, cursor_);
      if (behind == 0 || behind > modulus_ / 2) return false;
      const uint32_t held = readable(head);
      if (behind > held) {
        add_overruns(behind - held);
        cursor_ = sub(head, held);
        if (held == 0) return false;
      }
      const size_t base = (cursor_ % slots_) * kFrameWords;
      for (size_t w = 0; w < kFrameWords; ++w) {
        const uint32_t word = words_[base + w].load(std::memory_order_relaxed);
        out[2 * w] = static_cast<int16_t>(static_cast<uint16_t>(word & 0xFFFFu));
        out[2 * w + 1] = static_cast<int16_t>(static_cast<uint16_t>(word >> 16));
      }
      std::atomic_thread_fence(std::memory_order_acquire);
      const uint32_t after = published_.load(std::memory_order_relaxed);
      // The producer fills frame `after` in slot `after % slots`; the copy is intact while it has
      // not reached the frame one ring-length past the cursor, which reuses the cursor's slot.
      if (sub(after, cursor_) < slots_) {
        cursor_ = add(cursor_, 1);
        return true;
      }
      add_overruns(1);
      cursor_ = add(cursor_, 1);
    }
    return false;
  }

  // Drops every frame published so far, and the producer's partial frame at its next write. The
  // uplink clears on mute and again on unmute, so no audio from before or during a mute reaches a
  // later backlog.
  void clear() {
    if (!valid()) return;
    clear_generation_.fetch_add(1, std::memory_order_acq_rel);
    const uint32_t head = published_.load(std::memory_order_acquire);
    floor_ = head;
    cursor_ = head;
  }

  // Frames the consumer lost to the producer (read from any thread).
  uint32_t overruns() const { return overruns_.load(std::memory_order_relaxed); }

 private:
  uint32_t add(uint32_t a, uint32_t n) const { return (a + n) % modulus_; }
  uint32_t sub(uint32_t a, uint32_t b) const { return (a + modulus_ - b) % modulus_; }

  // Frames readable at `head`: since the floor, at most the capacity. A floor so old that the
  // counter has wrapped past it reads short, never stale.
  uint32_t readable(uint32_t head) const {
    if (!valid()) return 0;
    const uint32_t since = sub(head, floor_);
    if (since > modulus_ / 2) return 0;
    return since < capacity_ ? since : capacity_;
  }

  void add_overruns(uint32_t n) {
    overruns_.store(overruns_.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
  }

  std::atomic<uint32_t>* words_ = nullptr;
  uint32_t slots_ = 0;
  uint32_t capacity_ = 0;
  uint32_t modulus_ = 1;

  // Shared.
  std::atomic<uint32_t> published_{0};
  std::atomic<uint32_t> clear_generation_{0};
  std::atomic<uint32_t> overruns_{0};

  // Producer only.
  uint32_t head_ = 0;
  size_t fill_ = 0;
  uint16_t low_ = 0;
  bool has_low_ = false;
  uint32_t seen_generation_ = 0;

  // Consumer only.
  uint32_t cursor_ = 0;
  uint32_t floor_ = 0;
};

}  // namespace mist_core
