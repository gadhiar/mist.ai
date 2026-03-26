import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/log_entry.dart';
import '../providers/log_provider.dart';
import '../widgets/log_toolbar.dart';
import '../widgets/log_entry_tile.dart';

/// Full-featured log viewer screen with filtering, grouping, and auto-scroll.
class LogScreen extends ConsumerStatefulWidget {
  const LogScreen({super.key});

  @override
  ConsumerState<LogScreen> createState() => _LogScreenState();
}

class _LogScreenState extends ConsumerState<LogScreen> {
  final ScrollController _scrollController = ScrollController();
  bool _isAtBottom = true;

  @override
  void initState() {
    super.initState();
    _scrollController.addListener(_onScroll);
  }

  void _onScroll() {
    if (!_scrollController.hasClients) return;
    final position = _scrollController.position;
    final atBottom = position.pixels >= position.maxScrollExtent - 50;
    if (atBottom != _isAtBottom) {
      setState(() => _isAtBottom = atBottom);
    }
  }

  void _scrollToBottom() {
    if (_scrollController.hasClients) {
      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent,
        duration: const Duration(milliseconds: 200),
        curve: Curves.easeOut,
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    // Auto-scroll only when new entries arrive, not on every rebuild.
    ref.listen(logProvider, (prev, next) {
      if (_isAtBottom &&
          !next.isPaused &&
          next.entries.length > (prev?.entries.length ?? 0)) {
        WidgetsBinding.instance.addPostFrameCallback((_) => _scrollToBottom());
      }
    });

    final logState = ref.watch(logProvider);
    final filtered = logState.filteredEntries;

    return Scaffold(
      backgroundColor: const Color(0xFF121212),
      body: Column(
        children: [
          const LogToolbar(),
          Expanded(
            child: filtered.isEmpty
                ? _buildEmptyState(logState)
                : _buildLogList(logState, filtered),
          ),
        ],
      ),
      floatingActionButton: _isAtBottom
          ? null
          : FloatingActionButton.small(
              onPressed: () {
                _scrollToBottom();
                setState(() => _isAtBottom = true);
              },
              backgroundColor: const Color(0xFF2A2A2A),
              tooltip: 'Scroll to bottom',
              child: const Icon(Icons.arrow_downward, size: 18),
            ),
    );
  }

  Widget _buildEmptyState(LogState logState) {
    final hasEntries = logState.entries.isNotEmpty;
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            hasEntries ? Icons.filter_list_off : Icons.terminal,
            size: 48,
            color: Colors.grey.shade700,
          ),
          const SizedBox(height: 12),
          Text(
            hasEntries
                ? 'No entries match current filters'
                : 'No log entries yet',
            style: TextStyle(fontSize: 14, color: Colors.grey.shade600),
          ),
          if (!hasEntries) ...[
            const SizedBox(height: 4),
            Text(
              'Logs will appear when the backend sends them',
              style: TextStyle(fontSize: 12, color: Colors.grey.shade700),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildLogList(LogState logState, List<LogEntry> filtered) {
    final collapsedGroups = logState.collapsedGroups;
    final logNotifier = ref.read(logProvider.notifier);

    switch (logState.groupMode) {
      case LogGroupMode.none:
        return _buildFlatList(filtered);
      case LogGroupMode.byRequest:
        return _buildGroupedList(
          filtered,
          collapsedGroups: collapsedGroups,
          logNotifier: logNotifier,
          groupKeyFn: (e) => e.requestId ?? '(no request)',
          headerFn: _buildRequestGroupHeader,
        );
      case LogGroupMode.byComponent:
        return _buildGroupedList(
          filtered,
          collapsedGroups: collapsedGroups,
          logNotifier: logNotifier,
          groupKeyFn: (e) => e.logger,
          headerFn: _buildComponentGroupHeader,
        );
    }
  }

  Widget _buildFlatList(List<LogEntry> entries) {
    return ListView.builder(
      controller: _scrollController,
      itemCount: entries.length,
      itemBuilder: (context, index) {
        return LogEntryTile(
          key: ValueKey(
            '${entries[index].timestamp.microsecondsSinceEpoch}_$index',
          ),
          entry: entries[index],
        );
      },
    );
  }

  /// Generic grouped list builder. Groups entries by [groupKeyFn] and
  /// renders each group with a header from [headerFn].
  Widget _buildGroupedList(
    List<LogEntry> entries, {
    required Set<String> collapsedGroups,
    required LogNotifier logNotifier,
    required String Function(LogEntry) groupKeyFn,
    required Widget Function(String key, List<LogEntry> group, bool collapsed)
    headerFn,
  }) {
    // Preserve insertion order by using LinkedHashMap
    final groups = <String, List<LogEntry>>{};
    for (final entry in entries) {
      final key = groupKeyFn(entry);
      groups.putIfAbsent(key, () => []).add(entry);
    }

    final groupKeys = groups.keys.toList();

    return ListView.builder(
      controller: _scrollController,
      itemCount: groupKeys.length,
      itemBuilder: (context, index) {
        final key = groupKeys[index];
        final group = groups[key]!;
        final isCollapsed = collapsedGroups.contains(key);

        return Column(
          key: ValueKey(key),
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            InkWell(
              onTap: () => logNotifier.toggleGroup(key),
              child: headerFn(key, group, isCollapsed),
            ),
            if (!isCollapsed)
              ...group.map(
                (entry) => LogEntryTile(
                  key: ValueKey(
                    '${entry.timestamp.microsecondsSinceEpoch}_${entry.hashCode}',
                  ),
                  entry: entry,
                ),
              ),
          ],
        );
      },
    );
  }

  Widget _buildRequestGroupHeader(
    String key,
    List<LogEntry> group,
    bool collapsed,
  ) {
    final warningCount = group
        .where((e) => e.level == 'WARNING' || e.level == 'ERROR')
        .length;
    final suffix = warningCount > 0 ? ' ($warningCount issues)' : '';

    return _GroupHeader(
      icon: collapsed ? Icons.chevron_right : Icons.expand_more,
      label: '$key -- ${group.length} entries$suffix',
    );
  }

  Widget _buildComponentGroupHeader(
    String key,
    List<LogEntry> group,
    bool collapsed,
  ) {
    return _GroupHeader(
      icon: collapsed ? Icons.chevron_right : Icons.expand_more,
      label: '$key -- ${group.length} entries',
    );
  }

  @override
  void dispose() {
    _scrollController.removeListener(_onScroll);
    _scrollController.dispose();
    super.dispose();
  }
}

/// Styled group header row used by both request and component grouping.
class _GroupHeader extends StatelessWidget {
  final IconData icon;
  final String label;

  const _GroupHeader({required this.icon, required this.label});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.04),
        border: Border(
          bottom: BorderSide(color: Colors.white.withValues(alpha: 0.06)),
        ),
      ),
      child: Row(
        children: [
          Icon(icon, size: 16, color: Colors.grey.shade500),
          const SizedBox(width: 6),
          Text(
            label,
            style: TextStyle(
              fontFamily: 'Consolas',
              fontFamilyFallback: const ['Courier New', 'monospace'],
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: Colors.grey.shade400,
            ),
          ),
        ],
      ),
    );
  }
}
