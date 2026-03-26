# Backend Log Streaming -- Design Spec

**Date:** 2026-03-26
**Branch:** feat/backend-log-streaming
**Status:** Approved (rev 3)

---

## 1. Overview

Stream structured backend logs to the Flutter frontend in real-time over the existing WebSocket connection. Provide a full-featured log viewer in the Flutter app with level filtering, text search, request-based and component-based grouping, and runtime log level control.

Additionally, introduce an expandable sidebar navigation rail to the Flutter app to support the log viewer and future sections (Voice Profiles, Settings).

---

## 2. Goals

1. Real-time visibility into backend operations from within the Flutter app
2. Filterable by log level (DEBUG, INFO, WARNING, ERROR) and by component (logger name)
3. Groupable by conversation turn (request_id) or by component
4. Runtime log level adjustment per-logger without backend restart
5. Expandable sidebar navigation for Chat, Logs, Voice Profiles (stub), Settings (stub)
6. Architecture extensible for future features beyond devtools

## 3. Non-Goals

1. Log persistence to disk from the Flutter app (Docker log driver handles archival)
2. Log export to file (future enhancement)
3. Implementation of Voice Profiles or Settings screens (stubs only)
4. Backfill of historical logs on reconnect (client starts fresh)

---

## 4. Architecture

### 4.1 Backend: WebSocketLogHandler

A custom `logging.Handler` subclass that captures Python log records and broadcasts them as structured WebSocket messages.

**Location:** `backend/log_handler.py` (new file)

**Behavior:**
- Attaches to the root Python logger on server startup, alongside the existing `StreamHandler` (console output is retained)
- Captures all log records from all backend components
- Converts each record to a structured dict and queues it for broadcast
- Respects a configurable floor level per-logger via an internal gating map (default: INFO for all). This gating is in the handler only -- Python logger `.setLevel()` is never modified by runtime config changes, avoiding the cost of record creation at suppressed levels across all loggers
- Filters out noisy third-party loggers by default (see Section 4.4)

**Re-entrancy guard:** The handler's `emit()` method is called on whatever thread logged the record. If any code in the broadcast path (e.g., `broadcast_messages()` error handling) itself logs, the handler would re-enter and recurse infinitely. To prevent this:

```python
_thread_local = threading.local()

def emit(self, record):
    if getattr(_thread_local, '_in_emit', False):
        return
    _thread_local._in_emit = True
    try:
        # format, gate, queue
    finally:
        _thread_local._in_emit = False
```

The handler's own logger name (`backend.log_handler`) is also added to the default suppression list.

**Thread-safe queueing:** The `message_queue` is an `asyncio.Queue` which cannot be accessed directly from sync threads. The handler stores a reference to the running event loop (set during `server.py` startup) and uses `asyncio.run_coroutine_threadsafe(message_queue.put(record_dict), loop)` to enqueue from any thread. This matches the existing pattern in `voice_processor.py`.

**Log record schema:**
```json
{
  "type": "log",
  "timestamp": "2026-03-26T02:13:24.478000+00:00",
  "level": "INFO",
  "levelno": 20,
  "logger": "voice_models.model_manager",
  "request_id": "turn-0042",
  "message": "TTS worker: Generated 4.20s audio"
}
```

**Fields:**
- `type`: Always `"log"`. Distinguishes from other WebSocket message types.
- `timestamp`: ISO 8601 UTC. Converted via `datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()`.
- `level`: String level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
- `levelno`: Numeric level (10, 20, 30, 40, 50). Enables client-side numeric comparison.
- `logger`: Dotted logger name. Acts as the component identifier. Single-segment names (e.g., `__main__`, `server`) are used as-is.
- `request_id`: Conversation turn identifier. `null` for logs outside a request context (startup, scheduler, TTS worker thread -- see Section 4.2).
- `message`: Formatted log message string (`self.format(record)`).

### 4.2 Request ID Propagation

**Mechanism:** `contextvars.ContextVar`

**Location:** `backend/request_context.py` (new file)
- `current_request_id: ContextVar[str | None]` -- defaults to `None`
- `new_request_id() -> str` -- increments counter, sets the var, returns the ID
- `spawn_with_context(target, *args, **thread_kwargs)` -- utility to spawn a `threading.Thread` that inherits the current context

**ID format:** `turn-{monotonic_counter}` (e.g., `turn-1`, `turn-42`). Python's `%d` formatting naturally accommodates any width. Counter resets on server restart.

**Where request_id is set:** In `voice_processor._process_conversation_turn()`, before any processing begins. This is the entry point for all conversation turns.

**Thread propagation -- site-by-site plan:**

The codebase currently uses bare `threading.Thread` at several sites. Each must be updated to propagate context:

| Site | File:Line | Change |
|------|-----------|--------|
| Speech end handler | `voice_processor.py:116` | Replace `threading.Thread(target=fn, args=...)` with `spawn_with_context(fn, *args)` |
| Pending input | `voice_processor.py:274` | Same replacement |
| Complete audio | `voice_processor.py:285` | Same replacement |

`spawn_with_context` implementation:
```python
def spawn_with_context(target, *args, **thread_kwargs):
    ctx = copy_context()
    thread_kwargs.setdefault("daemon", True)
    t = threading.Thread(target=ctx.run, args=(target, *args), **thread_kwargs)
    t.start()
    return t
```

**Long-lived threads:** The TTS worker thread (`model_manager.py:113`) is a daemon started at init time and processes requests from a queue. It does NOT inherit per-request context -- its `request_id` will always be `null`. This is acceptable because TTS logs are correlated by `gen_id` in the message text, and the worker processes one request at a time. If richer correlation is needed later, the `gen_id` could be mapped to `request_id` via the request queue.

**`run_in_executor`:** The `server.py` code uses `loop.run_in_executor(None, ...)` for `load_all_models` during startup (not per-request). Per-request processing enters via `_on_speech_end` and `process_complete_audio` which spawn threads directly. No `run_in_executor` changes are needed for request_id propagation.

### 4.3 Runtime Log Level Control

Clients can send a WebSocket message to change the handler's gating level for a specific logger:

```json
{
  "type": "log_config",
  "action": "set_level",
  "logger": "voice_models.model_manager",
  "level": "DEBUG"
}
```

- `logger`: Dotted logger name. Use `"root"` to change the default gating level for all loggers without explicit overrides.
- `level`: Must be one of `DEBUG`, `INFO`, `WARNING`, `ERROR`. Any other value returns an error.
- `action`: Must be `"set_level"`. Any other value returns an error.

**Important:** This changes the handler's internal gating map only, NOT the Python logger's `.setLevel()`. This avoids the cost of creating LogRecord objects at suppressed levels across all loggers.

Changes are ephemeral (reset on restart). The server responds:

Success:
```json
{
  "type": "log_config_ack",
  "logger": "voice_models.model_manager",
  "level": "DEBUG"
}
```

Error (invalid logger, invalid level, invalid action):
```json
{
  "type": "log_config_error",
  "message": "Invalid level: 'TRACE'. Must be one of DEBUG, INFO, WARNING, ERROR."
}
```

### 4.4 Gating and Performance

**Problem:** DEBUG-level logging can generate hundreds of messages per second. Pushing all of these over WebSocket wastes bandwidth and floods the client.

**Solution:** Server-side gating at two levels.

**Level 1 -- Per-logger gating map:**
- The handler maintains a `dict[str, int]` mapping logger names to minimum level numbers.
- Default entry: `{"": logging.INFO}` (empty string = default for all loggers).
- Check: `record.levelno >= gating_map.get(record.name, gating_map[""])`.
- Third-party loggers suppressed to WARNING by default:
  `httpx`, `urllib3`, `uvicorn.access`, `uvicorn.error`, `httpcore`, `hpack`, `diffusers`, `transformers`, `sentence_transformers`, `asyncio`, `neo4j`, `torch`, `PIL`, `backend.log_handler`

**Level 2 -- Token bucket rate limiter:**
- Capacity: 100 tokens. Refill rate: 100 tokens/second.
- Each log message consumes 1 token.
- When tokens are exhausted, messages are dropped silently.
- When the bucket transitions from empty to having tokens (dropping ends), a single summary message is emitted: `{"type": "log", "level": "WARNING", "logger": "backend.log_handler", "message": "Dropped N log messages in the last Xs (rate limit)", "request_id": null}`. This summary message is exempt from the rate limit.
- The rate limit is global across all loggers.
- **Thread safety:** The bucket's state (`tokens`, `last_refill`, `dropped_count`) is guarded by a `threading.Lock` since `emit()` runs on arbitrary threads concurrently.

---

## 5. Flutter: Navigation Rail

### 5.1 Sidebar Component

**Widget:** `AppShell` (new widget, wraps the entire app)

**Structure:**
```
Row
+-- NavigationRail (collapsed: 56px, expanded: 200px)
|   +-- Toggle button (chevron) at top
|   +-- Chat icon + label
|   +-- Logs icon + label
|   +-- Voice Profiles icon + label (disabled/stub)
|   +-- Settings icon + label (disabled/stub)
+-- Expanded content area (selected destination)
```

**Behavior:**
- Default: collapsed (icon-only with tooltips on hover)
- Toggle: click chevron at top of rail, 200ms `AnimatedContainer` width transition
- State: `sidebarExpandedProvider` (`StateProvider<bool>`, default `false`)
- Selected index: `selectedDestinationProvider` (`StateProvider<int>`, default `0`)
- Stub items show a centered placeholder: "Coming soon" text
- Dark background, accent color on selected item

### 5.2 Destination Screens

| Index | Label | Icon | Screen Widget | Status |
|-------|-------|------|--------------|--------|
| 0 | Chat | `Icons.chat_bubble_outline` | `ChatScreen` (existing) | Active |
| 1 | Logs | `Icons.terminal` | `LogScreen` (new) | Active |
| 2 | Voice | `Icons.record_voice_over` | `StubScreen("Voice Profiles")` | Stub |
| 3 | Settings | `Icons.settings_outlined` | `StubScreen("Settings")` | Stub |

---

## 6. Flutter: Log Viewer

### 6.1 State Management

**Provider:** `logProvider` -- `NotifierProvider<LogNotifier, LogState>`

The `LogNotifier` subscribes to `websocketServiceProvider` (the existing singleton) via its message stream, following the same pattern as `ChatNotifier` and `VoiceNotifier`. It must be eagerly initialized (not lazy) so that logs are captured from the moment of connection, even before the user navigates to the log screen. Eager init is achieved by calling `ref.read(logProvider)` in `AppShell.build()` on first render.

**Note on fan-out:** All three notifiers (Chat, Voice, Log) receive every WebSocket message. Log messages at high volume will be delivered to ChatNotifier and VoiceNotifier as well, hitting their default/ignore case. This is acceptable. If it becomes a bottleneck, `WebSocketService` can pre-route messages by type to separate streams in a future optimization.

**LogState fields:**
- `Queue<LogEntry> entries` -- ring buffer using `dart:collection` `Queue` (O(1) add/remove from both ends), max 5,000 entries, oldest removed on overflow via `removeFirst()`
- `Set<String> activeLevels` -- which levels are visible (default: `{INFO, WARNING, ERROR}`)
- `String searchQuery` -- text filter on message content and logger name
- `LogGroupMode groupMode` -- enum: `none`, `byRequest`, `byComponent`
- `Set<String> collapsedGroups` -- which group IDs are collapsed
- `bool isPaused` -- toolbar pause button state (entries still buffer, auto-scroll disabled)
- `Map<String, String> loggerLevels` -- tracks per-logger level overrides sent to backend

**LogEntry model:**
```dart
class LogEntry {
  final DateTime timestamp;
  final String level;
  final int levelno;
  final String logger;
  final String? requestId;
  final String message;
}
```

### 6.2 Log Screen Layout

```
Column
+-- Toolbar Row (LogToolbar widget)
|   +-- Level filter chips (DEBUG | INFO | WARN | ERROR) -- toggle on/off
|   +-- Search field (filters message + logger name)
|   +-- Group mode dropdown (None | By Request | By Component)
|   +-- Pause/Resume button
|   +-- Clear button
|
+-- Expanded log list (ListView.builder)
    +-- If groupMode == none:
    |     Flat list of LogEntryTile widgets
    +-- If groupMode == byRequest:
    |     Collapsible sections keyed by request_id
    |     Header: "Turn 42 -- 15 entries (2 warnings)"
    |     Children: LogEntryTile widgets for that request_id
    +-- If groupMode == byComponent:
          Collapsible sections keyed by logger name
          Header: "voice_models.model_manager -- 23 entries"
          Children: LogEntryTile widgets for that logger
```

### 6.3 LogEntryTile Widget

A single log line. Compact, monospace font, color-coded by level.

```
[02:13:24.478] [INFO] model_manager: TTS worker: Generated 4.20s audio
```

- Timestamp: dimmed/muted color
- Level badge: color-coded (DEBUG=gray, INFO=blue, WARNING=amber, ERROR=red)
- Logger name: shortened to last segment of dotted name (e.g., `model_manager` from `voice_models.model_manager`). Single-segment names used as-is. Full name shown in tooltip.
- Message: primary text color
- Tap to expand: shows full logger name, request_id, full timestamp with date

### 6.4 Runtime Level Control

A context menu (right-click or long-press on a logger name in any log entry) offering:
- "Set [logger] to DEBUG"
- "Set [logger] to INFO"
- "Set [logger] to WARNING"
- "Reset to default"

This sends the `log_config` WebSocket message to the backend. The filter chip state updates to show that logger has been overridden (e.g., a small indicator dot).

### 6.5 Auto-scroll Behavior

- **Auto-scroll active:** When `_isAtBottom` is true (scroll position within 50px of `maxScrollExtent`) and `isPaused` is false, new entries trigger `animateTo(maxScrollExtent)`.
- **Implicit scroll pause:** When user scrolls up beyond the 50px threshold, auto-scroll stops. This does NOT set `isPaused` to true (toolbar pause is a separate concept).
- **Scroll-to-bottom FAB:** Appears when not at bottom. Clicking it scrolls to bottom and re-enables auto-scroll.
- **Toolbar Pause button:** Explicitly sets `isPaused`. When paused, entries buffer but no auto-scroll even if at bottom. Resume un-pauses and scrolls to bottom.
- **Batched updates:** State updates from incoming log messages are debounced at 100ms to prevent UI jank from rapid arrivals.

---

## 7. WebSocket Message Types (New)

### Server -> Client

| Type | Description |
|------|-------------|
| `log` | Structured log record (see section 4.1) |
| `log_config_ack` | Acknowledges a log level change |
| `log_config_error` | Error response for invalid log_config requests |

### Client -> Server

| Type | Description |
|------|-------------|
| `log_config` | Request to change log level for a logger |

---

## 8. File Inventory

### New Files

**Backend:**
| File | Purpose |
|------|---------|
| `backend/log_handler.py` | WebSocketLogHandler, log record formatting, rate limiting, re-entrancy guard |
| `backend/request_context.py` | ContextVar for request_id, `new_request_id()`, `spawn_with_context()` |

**Flutter:**
| File | Purpose |
|------|---------|
| `lib/widgets/app_shell.dart` | NavigationRail + content area wrapper |
| `lib/screens/log_screen.dart` | Log viewer screen |
| `lib/widgets/log_entry_tile.dart` | Single log entry display |
| `lib/widgets/log_toolbar.dart` | Filter chips, search, group mode, controls |
| `lib/models/log_entry.dart` | LogEntry data class |
| `lib/providers/log_provider.dart` | LogNotifier + LogState |
| `lib/providers/navigation_provider.dart` | Sidebar state (expanded, selected index) |
| `lib/screens/stub_screen.dart` | Placeholder for unimplemented destinations |

### Modified Files

**Backend:**
| File | Change |
|------|--------|
| `backend/server.py` | Attach WebSocketLogHandler on startup (pass event loop + message_queue), handle `log_config` messages in WebSocket handler, add validation |
| `backend/voice_processor.py` | Import and use `new_request_id()` at start of `_process_conversation_turn()`, replace `threading.Thread` with `spawn_with_context` at 3 sites |

**Flutter:**
| File | Change |
|------|--------|
| `lib/main.dart` | Replace `ChatScreen` with `AppShell` as root widget |
| `lib/models/websocket_message.dart` | Add `log`, `log_config`, `log_config_ack`, `log_config_error` message types |
| `lib/providers/chat_provider.dart` | Add explicit `case "log":` to ignore log messages (not fall through to default) |

---

## 9. Testing Strategy

**Backend:**
- Unit test `WebSocketLogHandler`: record formatting, level gating, rate limiting (token bucket depletion and recovery), re-entrancy guard, third-party logger suppression
- Unit test `request_context`: ID generation, `spawn_with_context` propagation
- Integration test: emit a log, connect a raw WebSocket client, assert `log` message arrives with correct schema
- Integration test: send `log_config` message, verify `log_config_ack` response and that subsequent logs respect the new level

**Flutter:**
- Unit test `LogNotifier`: ring buffer overflow (Queue behavior), level filtering, search filtering, group mode sorting, pause/resume
- Widget test `LogEntryTile`: level color coding, timestamp formatting, logger name shortening (dotted and single-segment), tap-to-expand
- Widget test `LogToolbar`: filter chip toggle, search input, group mode switching, pause/resume button state
- Widget test `AppShell`: rail collapse/expand animation, destination selection, stub screen rendering

---

## 10. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Log handler infinite recursion | Re-entrancy guard via `threading.local()` flag + handler logger in suppression list |
| Log volume overwhelms WebSocket | Server-side gating (INFO default) + token bucket rate limiter (100/sec) |
| Async queue access from sync threads | Use `asyncio.run_coroutine_threadsafe()` with stored event loop reference |
| Ring buffer performance in Dart | `dart:collection` Queue (O(1) both ends) instead of List |
| Request ID lost across thread boundaries | `spawn_with_context()` utility at all 3 thread spawn sites; TTS worker acknowledged as null |
| Third-party library log noise | Default WARNING for 14 known noisy loggers |
| UI jank from rapid log updates | ListView.builder (lazy) + 100ms debounce on state updates |
| Triple fan-out on WebSocket messages | Acceptable overhead; explicit `case "log":` in ChatNotifier to document intent |
| Root logger level change cascading | Handler gating map only, never touches Python logger.setLevel() |
| Invalid log_config requests | Validation with `log_config_error` response |
