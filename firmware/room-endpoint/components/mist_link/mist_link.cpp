// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.

#include "mist_link.h"

#include <algorithm>
#include <cstring>

#include <esp_heap_caps.h>
#include <freertos/idf_additions.h>

#include "esphome/components/network/util.h"
#include "esphome/core/hal.h"
#include "esphome/core/helpers.h"
#include "esphome/core/log.h"

namespace esphome {
namespace mist_link {

static const char *const TAG = "mist_link";

// WebSocket opcodes (RFC 6455).
static constexpr uint8_t kOpText = 0x1;
static constexpr uint8_t kOpBinary = 0x2;

static constexpr UBaseType_t kTaskPriority = 5;
static constexpr int kClientBufferBytes = 2048;
static constexpr int kNetworkTimeoutMs = 10000;
static constexpr int kPingIntervalS = 10;
static constexpr int kPongTimeoutS = 30;
static constexpr TickType_t kSendTimeout = pdMS_TO_TICKS(1000);
// The most inbound events the main loop handles per pass.
static constexpr int kEventsPerLoop = 32;

void MistLink::setup() {
  this->headers_ = std::string("Authorization: Bearer ") + this->token_ + "\r\n";

  this->outbound_ = xRingbufferCreateWithCaps(kOutboundRingBytes, RINGBUF_TYPE_NOSPLIT, MALLOC_CAP_SPIRAM);
  this->inbound_ = xQueueCreate(kInboundSlots, sizeof(Event));
  RAMAllocator<mist_core::json::Token> token_allocator(RAMAllocator<mist_core::json::Token>::ALLOC_EXTERNAL);
  this->tokens_ = token_allocator.allocate(kTokenPool);
  if (this->outbound_ == nullptr || this->inbound_ == nullptr || this->tokens_ == nullptr) {
    ESP_LOGE(TAG, "Could not allocate the link's queues");
    this->mark_failed();
    return;
  }
  if (!this->sender_task_.create(MistLink::sender_main, "mist_link_tx", kSenderStackBytes, this, kTaskPriority, true)) {
    ESP_LOGE(TAG, "Could not start the sender task");
    this->mark_failed();
    return;
  }

  esp_websocket_client_config_t config{};
  config.uri = this->uri_;
  config.cert_pem = this->ca_pem_;
  config.headers = this->headers_.c_str();
  config.task_name = "mist_link_rx";
  config.task_prio = kTaskPriority;
  config.task_stack = kClientStackBytes;
  config.buffer_size = kClientBufferBytes;
  config.reconnect_timeout_ms = kBackoffFirstMs;
  config.network_timeout_ms = kNetworkTimeoutMs;
  config.ping_interval_sec = kPingIntervalS;
  config.pingpong_timeout_sec = kPongTimeoutS;
  // The listener closes a replaced (4001) or revoked (4003) connection cleanly; reconnect anyway,
  // on the same backoff, so a device returns once it is registered again.
  config.enable_close_reconnect = true;
  config.user_context = this;
  this->client_ = esp_websocket_client_init(&config);
  if (this->client_ == nullptr) {
    ESP_LOGE(TAG, "Could not create the WebSocket client");
    this->mark_failed();
    return;
  }
  esp_websocket_register_events(this->client_, WEBSOCKET_EVENT_ANY, MistLink::on_event, this);
}

void MistLink::loop() {
  if (!this->started_ && network::is_connected()) {
    if (esp_websocket_client_start(this->client_) == ESP_OK) {
      this->started_ = true;
    } else {
      ESP_LOGW(TAG, "Could not start the WebSocket client; retrying");
    }
  }

  Event event;
  for (int i = 0; i < kEventsPerLoop && xQueueReceive(this->inbound_, &event, 0) == pdTRUE; ++i) {
    switch (event.kind) {
      case Kind::Connected:
        this->connected();
        break;
      case Kind::Disconnected:
        this->disconnected();
        break;
      case Kind::Text:
        this->dispatch_text(reinterpret_cast<const char *>(event.data), event.len);
        break;
      case Kind::Binary:
        this->dispatch_binary(event.data, event.len);
        break;
    }
    heap_caps_free(event.data);
  }

#ifdef USE_MIST_LINK_DEVICE_STATS
  const uint32_t now = millis();
  if (this->connected_.load() && now - this->last_stats_ms_ >= kStatsIntervalMs) {
    this->last_stats_ms_ = now;
    this->send_stats();
  }
#endif
}

void MistLink::dump_config() {
  ESP_LOGCONFIG(TAG,
                "MIST link:\n"
                "  URI: %s\n"
                "  CA pinned: %s",
                this->uri_, YESNO(this->ca_pem_ != nullptr));
#ifdef USE_MIST_LINK_DEVICE_STATS
  ESP_LOGCONFIG(TAG, "  Device stats: every %u ms (spike build)", (unsigned) kStatsIntervalMs);
#endif
}

// -- Outbound ----------------------------------------------------------------------------------

bool MistLink::send_text(const char *text, size_t len) {
  return this->enqueue(reinterpret_cast<const uint8_t *>(text), len, false);
}

bool MistLink::send_binary(const uint8_t *data, size_t len) { return this->enqueue(data, len, true); }

bool MistLink::enqueue(const uint8_t *data, size_t len, bool binary) {
  if (!this->connected_.load() || len == 0 || len > kOutboundMaxBytes || this->outbound_ == nullptr) {
    return false;
  }
  void *slot = nullptr;
  if (xRingbufferSendAcquire(this->outbound_, &slot, sizeof(OutboundHeader) + len, 0) != pdTRUE) {
    this->outbound_dropped_.fetch_add(1);
    return false;
  }
  OutboundHeader header{this->generation_.load(), binary ? 1u : 0u};
  std::memcpy(slot, &header, sizeof(header));
  std::memcpy(static_cast<uint8_t *>(slot) + sizeof(header), data, len);
  xRingbufferSendComplete(this->outbound_, slot);
  return true;
}

void MistLink::sender_main(void *arg) { static_cast<MistLink *>(arg)->sender_loop(); }

void MistLink::sender_loop() {
  for (;;) {
    size_t size = 0;
    void *item = xRingbufferReceive(this->outbound_, &size, portMAX_DELAY);
    if (item == nullptr) {
      continue;
    }
    OutboundHeader header;
    std::memcpy(&header, item, sizeof(header));
    const char *payload = static_cast<const char *>(item) + sizeof(header);
    const int len = static_cast<int>(size - sizeof(header));
    // A message queued on an earlier connection is dropped, never sent on this one.
    if (header.generation == this->generation_.load() && this->connected_.load()) {
      const int sent = header.binary != 0 ? esp_websocket_client_send_bin(this->client_, payload, len, kSendTimeout)
                                          : esp_websocket_client_send_text(this->client_, payload, len, kSendTimeout);
      if (sent < 0) {
        this->outbound_dropped_.fetch_add(1);
      }
    } else {
      this->outbound_dropped_.fetch_add(1);
    }
    vRingbufferReturnItem(this->outbound_, item);
  }
}

// -- The client task -----------------------------------------------------------------------------

void MistLink::on_event(void *arg, esp_event_base_t base, int32_t event_id, void *event_data) {
  static_cast<MistLink *>(arg)->handle_event(event_id, static_cast<const esp_websocket_event_data_t *>(event_data));
}

void MistLink::handle_event(int32_t event_id, const esp_websocket_event_data_t *data) {
  switch (event_id) {
    case WEBSOCKET_EVENT_CONNECTED:
      this->backoff_ms_ = kBackoffFirstMs;
      esp_websocket_client_set_reconnect_timeout(this->client_, kBackoffFirstMs);
      this->post(Kind::Connected, nullptr, 0);
      break;
    case WEBSOCKET_EVENT_DISCONNECTED:
    case WEBSOCKET_EVENT_CLOSED:
      // The client waits this long before its next attempt; the one after waits twice as long.
      esp_websocket_client_set_reconnect_timeout(this->client_, static_cast<int>(this->backoff_ms_));
      this->backoff_ms_ = std::min(this->backoff_ms_ * 2, kBackoffMaxMs);
      this->drop_partial();
      this->post(Kind::Disconnected, nullptr, 0);
      break;
    case WEBSOCKET_EVENT_DATA:
      this->handle_data(data);
      break;
    case WEBSOCKET_EVENT_ERROR:
      ESP_LOGW(TAG, "Link error: handshake status %d, TLS error 0x%x, socket errno %d",
               data->error_handle.esp_ws_handshake_status_code, data->error_handle.esp_tls_last_esp_err,
               data->error_handle.esp_transport_sock_errno);
      break;
    default:
      break;
  }
}

void MistLink::handle_data(const esp_websocket_event_data_t *data) {
  if (data->op_code != kOpText && data->op_code != kOpBinary) {
    return;  // control frames; a fragmented message's continuations are never sent by the server
  }
  if (data->payload_offset == 0) {
    this->drop_partial();
    if (!data->fin || data->payload_len <= 0 || static_cast<size_t>(data->payload_len) > kInboundMaxBytes) {
      this->inbound_dropped_.fetch_add(1);
      return;
    }
    // One byte more than the payload, so a text message is NUL-terminated.
    this->partial_ = static_cast<uint8_t *>(heap_caps_malloc(data->payload_len + 1, MALLOC_CAP_SPIRAM));
    if (this->partial_ == nullptr) {
      this->inbound_dropped_.fetch_add(1);
      return;
    }
    this->partial_len_ = static_cast<uint32_t>(data->payload_len);
    this->partial_binary_ = data->op_code == kOpBinary;
  }
  if (this->partial_ == nullptr || data->data_len < 0 ||
      static_cast<uint32_t>(data->payload_offset) + static_cast<uint32_t>(data->data_len) > this->partial_len_) {
    return;
  }
  std::memcpy(this->partial_ + data->payload_offset, data->data_ptr, data->data_len);
  if (static_cast<uint32_t>(data->payload_offset + data->data_len) == this->partial_len_) {
    this->partial_[this->partial_len_] = 0;
    this->post(this->partial_binary_ ? Kind::Binary : Kind::Text, this->partial_, this->partial_len_);
    this->partial_ = nullptr;
    this->partial_len_ = 0;
  }
}

void MistLink::drop_partial() {
  heap_caps_free(this->partial_);
  this->partial_ = nullptr;
  this->partial_len_ = 0;
}

void MistLink::post(Kind kind, uint8_t *data, uint32_t len) {
  Event event{kind, len, data};
  // A connection change must reach the main loop, so it waits for room; a message that finds the
  // queue full is dropped and counted.
  const TickType_t wait = data == nullptr ? portMAX_DELAY : 0;
  if (xQueueSend(this->inbound_, &event, wait) != pdTRUE) {
    heap_caps_free(data);
    this->inbound_dropped_.fetch_add(1);
  }
}

// -- The main loop -----------------------------------------------------------------------------

void MistLink::connected() {
  if (this->connected_.load()) {
    return;
  }
  this->generation_.fetch_add(1);
  this->connected_.store(true);
  if (this->ever_connected_) {
    this->reconnects_.fetch_add(1);
  }
  this->ever_connected_ = true;
  ESP_LOGI(TAG, "Connected");

  char buf[256];
  const size_t len = mist_core::build_voice_start(buf, sizeof(buf), ++this->req_, this->muted_.load());
  if (len == 0 || !this->send_text(buf, len)) {
    ESP_LOGE(TAG, "Could not queue voice_start");
  }
  for (auto *listener : this->listeners_) {
    listener->on_link_connected();
  }
}

void MistLink::disconnected() {
  if (!this->connected_.load()) {
    return;
  }
  this->connected_.store(false);
  this->generation_.fetch_add(1);
  ESP_LOGW(TAG, "Disconnected; reconnecting");
  for (auto *listener : this->listeners_) {
    listener->on_link_disconnected();
  }
}

void MistLink::dispatch_text(const char *text, size_t len) {
  mist_core::json::Doc doc(this->tokens_, kTokenPool);
  const mist_core::ReadError error = mist_core::read_inbound(&doc, text, len, &this->message_);
  if (error != mist_core::ReadError::None) {
    ESP_LOGW(TAG, "Dropped an unreadable message (%u)", static_cast<unsigned>(error));
    return;
  }
  if (this->message_.type == mist_core::InboundType::Unknown) {
    return;  // a type the device does not act on
  }
  if (this->message_.type == mist_core::InboundType::Error) {
    ESP_LOGW(TAG, "Server error: %s (retriable: %s)", this->message_.error.kind,
             YESNO(this->message_.error.retriable));
  }
  for (auto *listener : this->listeners_) {
    listener->on_link_message(this->message_, doc);
  }
}

void MistLink::dispatch_binary(const uint8_t *data, size_t len) {
  mist_core::Frame frame;
  const mist_core::FrameError error = mist_core::decode_frame(data, len, mist_core::Direction::Down, &frame);
  if (error != mist_core::FrameError::None) {
    ESP_LOGW(TAG, "Dropped a downlink frame: %s", mist_core::frame_error_field(error));
    return;
  }
  for (auto *listener : this->listeners_) {
    listener->on_link_frame(frame);
  }
}

#ifdef USE_MIST_LINK_DEVICE_STATS
void MistLink::send_stats() {
  mist_core::DeviceStats stats{};
  stats.heap_free_min = static_cast<uint32_t>(heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL));
  stats.psram_free = static_cast<uint32_t>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  stats.mic_overruns = this->mic_overruns_.load();
  stats.ring_overruns = this->ring_overruns_.load();
  stats.playback_underruns = this->playback_underruns_.load();
  stats.uptime_s = millis() / 1000;
  stats.reconnects = this->reconnects_.load();
  char buf[256];
  const size_t len = mist_core::build_device_stats(buf, sizeof(buf), stats);
  if (len == 0 || !this->send_text(buf, len)) {
    ESP_LOGW(TAG, "Could not queue device_stats");
  }
}
#endif

}  // namespace mist_link
}  // namespace esphome
