// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.
//
// The room endpoint's one link to MIST (plan Design item 3): a persistent `wss://` connection to
// `/v2/device` through `esp_websocket_client`, with the private CA pinned and `Authorization:
// Bearer <device>.<secret>`.
//
// - On every connection it sends `voice_start` first, with the current mute state.
// - Outbound messages go through a PSRAM ring buffer that a sender task, its stack in PSRAM,
//   drains in order. A message queued on one connection is never sent on the next, so a `seq` sequence
//   never crosses a reconnection.
// - Inbound messages are read with `mist_core::read_inbound` and handed, in arrival order and on
//   the main loop, to each registered `LinkListener`; each owner acts on its own types. Unknown
//   types are ignored. Binary downlink frames (`0x01`, `0x02`, `0x03`) are validated and handed
//   on in the same order, so a `stream_start` always reaches its owner before its audio.
// - Reconnection backs off from 1 s, doubling, to 15 s, and returns to 1 s once connected.
//
// Threading: `send_text` and `send_binary` may be called from any task. Listener callbacks run on
// the main loop. The client's own task only copies inbound messages into the inbound queue.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>
#include <freertos/ringbuf.h>

#include "esp_websocket_client.h"
#include "esphome/components/mist_core/codec.h"
#include "esphome/components/mist_core/json_lite.h"
#include "esphome/components/mist_core/uplink.h"
#include "esphome/core/component.h"
#include "esphome/core/defines.h"
#include "esphome/core/static_task.h"

#ifdef USE_MIST_LINK_SPIKE_PLAYER
#include "esphome/components/speaker/speaker.h"
#endif

namespace esphome {
namespace mist_link {

#ifdef USE_MIST_LINK_SPIKE_PLAYER
class SpikePlayer;
#endif

// The owner of some inbound messages. Every callback runs on the main loop.
class LinkListener {
 public:
  virtual ~LinkListener() = default;
  // `voice_start` is already queued when this runs.
  virtual void on_link_connected() {}
  virtual void on_link_disconnected() {}
  // One JSON message of a type the device acts on. `doc` holds its parse, for readers that need
  // more than `Inbound` carries (`alarm_state` and `alarm_fired`, read by `alarm_rules`).
  virtual void on_link_message(const mist_core::Inbound &message, const mist_core::json::Doc &doc) {}
  // One validated downlink frame; `frame.payload` is valid only during the call.
  virtual void on_link_frame(const mist_core::Frame &frame) {}
};

class MistLink : public Component, public mist_core::UplinkSink {
 public:
  static constexpr uint32_t kBackoffFirstMs = 1000;
  static constexpr uint32_t kBackoffMaxMs = 15000;
  // The largest outbound message: a `0x10` frame is 656 bytes; the largest JSON the device sends,
  // an `alarm_report` with 64-byte ids, is under 400.
  static constexpr size_t kOutboundMaxBytes = 1024;
  // The outbound ring, in PSRAM: about 280 `0x10` frames with their headers, so a whole 5 s
  // backlog (250 frames) fits at once.
  static constexpr size_t kOutboundRingBytes = 192 * 1024;
  static constexpr size_t kInboundSlots = 64;
  // The device listener's `ws_max_size`.
  static constexpr size_t kInboundMaxBytes = 65536;
  static constexpr uint32_t kSenderStackBytes = 4096;
  static constexpr uint32_t kClientStackBytes = 6144;
  static constexpr uint32_t kStatsIntervalMs = 10000;
  static constexpr uint32_t kTokenPool = 512;

  void set_uri(const char *uri) { this->uri_ = uri; }
  void set_token(const char *token) { this->token_ = token; }
  void set_ca_pem(const char *pem) { this->ca_pem_ = pem; }

  void setup() override;
  void loop() override;
  void dump_config() override;
  float get_setup_priority() const override { return setup_priority::AFTER_WIFI; }

  // Queue one message for the current connection. False when not connected or the queue is
  // full; the uplink then retries a frame on its next pump (`mist_core::UplinkSink`).
  bool send_text(const char *text, size_t len) override;
  bool send_binary(const uint8_t *data, size_t len) override;

  void add_listener(LinkListener *listener) { this->listeners_.push_back(listener); }
  // The mute state the next `voice_start` carries; the uplink sends `mute` itself.
  void set_muted(bool muted) { this->muted_.store(muted); }
  bool is_connected() const { return this->connected_.load(); }

  // Counters other components add to; the spike build reports them in `device_stats`.
  void add_mic_overruns(uint32_t count) { this->mic_overruns_.fetch_add(count); }
  void add_ring_overruns(uint32_t count) { this->ring_overruns_.fetch_add(count); }
  void add_playback_underruns(uint32_t count) { this->playback_underruns_.fetch_add(count); }

#ifdef USE_MIST_LINK_SPIKE_PLAYER
  // Spike build only: the speaker the SpikePlayer feeds.
  void set_spike_speaker(speaker::Speaker *output) { this->spike_speaker_ = output; }
#endif

 protected:
  enum class Kind : uint8_t { Connected, Disconnected, Text, Binary };

  // Precedes each outbound message in the ring.
  struct OutboundHeader {
    uint32_t generation;
    uint32_t binary;
  };

  // An inbound event; `data` is a PSRAM buffer the main loop frees.
  struct Event {
    Kind kind;
    uint32_t len;
    uint8_t *data;
  };

  static void on_event(void *arg, esp_event_base_t base, int32_t event_id, void *event_data);
  static void sender_main(void *arg);

  void handle_event(int32_t event_id, const esp_websocket_event_data_t *data);
  void handle_data(const esp_websocket_event_data_t *data);
  void post(Kind kind, uint8_t *data, uint32_t len);
  void drop_partial();
  bool enqueue(const uint8_t *data, size_t len, bool binary);
  void sender_loop();
  void dispatch_text(const char *text, size_t len);
  void dispatch_binary(const uint8_t *data, size_t len);
  void connected();
  void disconnected();
#ifdef USE_MIST_LINK_DEVICE_STATS
  void send_stats();
#endif

  const char *uri_{nullptr};
  const char *token_{nullptr};
  const char *ca_pem_{nullptr};
  std::string headers_;

  esp_websocket_client_handle_t client_{nullptr};
  bool started_{false};
  StaticTask sender_task_;
  RingbufHandle_t outbound_{nullptr};
  QueueHandle_t inbound_{nullptr};
  std::vector<LinkListener *> listeners_;

  // Written by the main loop; read by the sender task and by any task that sends.
  std::atomic<bool> connected_{false};
  std::atomic<uint32_t> generation_{0};
  std::atomic<bool> muted_{false};
  uint32_t req_{0};
  bool ever_connected_{false};

  // The client task's state: the message being reassembled, and the next backoff.
  uint8_t *partial_{nullptr};
  uint32_t partial_len_{0};
  bool partial_binary_{false};
  uint32_t backoff_ms_{kBackoffFirstMs};

  // The main loop's parse state, kept off its stack.
  mist_core::json::Token *tokens_{nullptr};
  mist_core::Inbound message_{};

  std::atomic<uint32_t> mic_overruns_{0};
  std::atomic<uint32_t> ring_overruns_{0};
  std::atomic<uint32_t> playback_underruns_{0};
  std::atomic<uint32_t> reconnects_{0};
  std::atomic<uint32_t> inbound_dropped_{0};
  std::atomic<uint32_t> outbound_dropped_{0};
  uint32_t last_stats_ms_{0};

#ifdef USE_MIST_LINK_SPIKE_PLAYER
  speaker::Speaker *spike_speaker_{nullptr};
  SpikePlayer *spike_player_{nullptr};
#endif
};

#ifdef USE_MIST_LINK_SPIKE_PLAYER
// Spike build only (host step H4): plays the spike listener's tone so G1 can measure the uplink
// during playback before the room build's player exists. Every `0x01` frame's 24 kHz PCM16 goes,
// in arrival order, through a 1 s PSRAM FIFO into the reply resampler, mixer and DAC; `0x02`
// lets the speaker drain and `0x03` stops it. There are no tag rules, fades or ducking; T6's
// `mist_player` is the room build's player.
//
// An underrun is counted, once per dry spell, when a stream is playing, the speaker has held
// audio since the stream started, and both the FIFO and the speaker are empty.
class SpikePlayer : public LinkListener {
 public:
  static constexpr uint32_t kSampleRate = 24000;
  static constexpr size_t kFifoBytes = kSampleRate * 2;  // 1 s of PCM16 mono

  SpikePlayer(MistLink *link, speaker::Speaker *output) : link_(link), output_(output) {}

  bool setup();
  // Runs on the main loop: moves FIFO audio into the speaker and checks for an underrun.
  void loop();
  void on_link_frame(const mist_core::Frame &frame) override;
  void on_link_disconnected() override;

 protected:
  void stop_stream();

  MistLink *link_;
  speaker::Speaker *output_;
  uint8_t *fifo_{nullptr};
  size_t head_{0};  // next byte to play
  size_t used_{0};
  bool streaming_{false};
  bool finishing_{false};  // `0x02` came; finish the speaker once the FIFO is empty
  bool primed_{false};
  bool dry_{false};
  uint32_t dropped_bytes_{0};
};
#endif

}  // namespace mist_link
}  // namespace esphome
