import 'dart:async';
import 'dart:collection';

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:logger/logger.dart';
import '../models/log_entry.dart';
import '../models/websocket_message.dart';
import '../services/websocket_service.dart';
import 'websocket_provider.dart';

/// How log entries are grouped in the viewer.
enum LogGroupMode { none, byRequest, byComponent }

/// Maximum number of log entries retained in the ring buffer.
const int _maxLogEntries = 5000;

/// Debounce interval for batching incoming log entries into state updates.
const Duration _debounceInterval = Duration(milliseconds: 100);

/// Immutable state for the log viewer.
class LogState {
  final Queue<LogEntry> entries;
  final Set<String> activeLevels;
  final String searchQuery;
  final LogGroupMode groupMode;
  final Set<String> collapsedGroups;
  final bool isPaused;
  final Map<String, String> loggerLevels;

  LogState({
    Queue<LogEntry>? entries,
    this.activeLevels = const {'INFO', 'WARNING', 'ERROR'},
    this.searchQuery = '',
    this.groupMode = LogGroupMode.none,
    this.collapsedGroups = const {},
    this.isPaused = false,
    this.loggerLevels = const {},
  }) : entries = entries ?? Queue<LogEntry>();

  LogState copyWith({
    Queue<LogEntry>? entries,
    Set<String>? activeLevels,
    String? searchQuery,
    LogGroupMode? groupMode,
    Set<String>? collapsedGroups,
    bool? isPaused,
    Map<String, String>? loggerLevels,
  }) {
    return LogState(
      entries: entries ?? this.entries,
      activeLevels: activeLevels ?? this.activeLevels,
      searchQuery: searchQuery ?? this.searchQuery,
      groupMode: groupMode ?? this.groupMode,
      collapsedGroups: collapsedGroups ?? this.collapsedGroups,
      isPaused: isPaused ?? this.isPaused,
      loggerLevels: loggerLevels ?? this.loggerLevels,
    );
  }

  /// Applies active level and search query filters. Returns a new list.
  List<LogEntry> get filteredEntries {
    final query = searchQuery.toLowerCase();
    return entries.where((entry) {
      if (!activeLevels.contains(entry.level)) return false;
      if (query.isNotEmpty) {
        final matchesMessage = entry.message.toLowerCase().contains(query);
        final matchesLogger = entry.logger.toLowerCase().contains(query);
        if (!matchesMessage && !matchesLogger) return false;
      }
      return true;
    }).toList();
  }
}

/// Notifier that subscribes to the WebSocket stream and captures log entries.
///
/// Follows the same subscription pattern as [ChatNotifier] and [VoiceNotifier].
class LogNotifier extends Notifier<LogState> {
  late WebSocketService _wsService;
  final Logger _logger = Logger();

  /// Pending entries accumulated during the debounce window.
  final List<LogEntry> _pendingEntries = [];
  Timer? _debounceTimer;

  @override
  LogState build() {
    _wsService = ref.read(websocketServiceProvider);

    final wsSub = _wsService.messageStream.listen((message) {
      _handleWebSocketMessage(message);
    });

    ref.onDispose(() {
      wsSub.cancel();
      _debounceTimer?.cancel();
    });

    return LogState();
  }

  /// Route incoming WebSocket messages -- only process log type.
  void _handleWebSocketMessage(WebSocketMessage message) {
    if (message.type != WsMessageType.log) return;

    try {
      final entry = LogEntry.fromJson(message.data);
      _pendingEntries.add(entry);
      _scheduleBatchUpdate();
    } catch (e) {
      _logger.e('Failed to parse log entry: $e');
    }
  }

  /// Debounce state updates to prevent UI jank from rapid log arrivals.
  void _scheduleBatchUpdate() {
    if (_debounceTimer?.isActive ?? false) return;
    _debounceTimer = Timer(_debounceInterval, _flushPendingEntries);
  }

  /// Flush all pending entries into the ring buffer in a single state update.
  void _flushPendingEntries() {
    if (_pendingEntries.isEmpty) return;

    final buffer = Queue<LogEntry>.from(state.entries);
    for (final entry in _pendingEntries) {
      buffer.addLast(entry);
      if (buffer.length > _maxLogEntries) {
        buffer.removeFirst();
      }
    }
    _pendingEntries.clear();

    state = state.copyWith(entries: buffer);
  }

  /// Toggle a level in the active filter set.
  void toggleLevel(String level) {
    final updated = Set<String>.from(state.activeLevels);
    if (updated.contains(level)) {
      updated.remove(level);
    } else {
      updated.add(level);
    }
    state = state.copyWith(activeLevels: updated);
  }

  /// Set the search query used for filtering entries.
  void setSearchQuery(String query) {
    state = state.copyWith(searchQuery: query);
  }

  /// Set the grouping mode for the log viewer.
  void setGroupMode(LogGroupMode mode) {
    state = state.copyWith(groupMode: mode);
  }

  /// Toggle collapse state for a group header.
  void toggleGroup(String groupId) {
    final updated = Set<String>.from(state.collapsedGroups);
    if (updated.contains(groupId)) {
      updated.remove(groupId);
    } else {
      updated.add(groupId);
    }
    state = state.copyWith(collapsedGroups: updated);
  }

  /// Toggle the pause state. When paused, entries still buffer but
  /// auto-scroll is disabled.
  void togglePause() {
    state = state.copyWith(isPaused: !state.isPaused);
  }

  /// Clear all buffered entries.
  void clear() {
    _pendingEntries.clear();
    state = state.copyWith(entries: Queue<LogEntry>());
  }

  /// Send a log_config message to the backend to change the gating level
  /// for a specific logger.
  void setLoggerLevel(String logger, String level) {
    try {
      _wsService.sendMessage(
        WebSocketMessage(
          type: WsMessageType.logConfig,
          data: {'action': 'set_level', 'logger': logger, 'level': level},
        ),
      );

      final updated = Map<String, String>.from(state.loggerLevels);
      updated[logger] = level;
      state = state.copyWith(loggerLevels: updated);
    } catch (e) {
      _logger.e('Failed to send log_config: $e');
    }
  }

  /// Reset a logger to the server default by removing the local override
  /// and sending a reset config.
  void resetLoggerLevel(String logger) {
    try {
      _wsService.sendMessage(
        WebSocketMessage(
          type: WsMessageType.logConfig,
          data: {'action': 'set_level', 'logger': logger, 'level': 'INFO'},
        ),
      );

      final updated = Map<String, String>.from(state.loggerLevels);
      updated.remove(logger);
      state = state.copyWith(loggerLevels: updated);
    } catch (e) {
      _logger.e('Failed to send log_config reset: $e');
    }
  }
}

/// Log Provider
final logProvider = NotifierProvider<LogNotifier, LogState>(() {
  return LogNotifier();
});
