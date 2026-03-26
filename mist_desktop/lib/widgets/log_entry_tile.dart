import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';
import '../models/log_entry.dart';
import '../providers/log_provider.dart';

/// Color associated with each log level.
Color _levelColor(String level) {
  switch (level) {
    case 'DEBUG':
      return Colors.grey;
    case 'INFO':
      return Colors.lightBlueAccent;
    case 'WARNING':
      return Colors.amber;
    case 'ERROR':
    case 'CRITICAL':
      return Colors.redAccent;
    default:
      return Colors.grey;
  }
}

/// A single compact log line, monospace, color-coded by level.
///
/// Tap to expand and show full metadata. Right-click to change the
/// backend log level for the originating logger.
class LogEntryTile extends ConsumerStatefulWidget {
  final LogEntry entry;

  const LogEntryTile({super.key, required this.entry});

  @override
  ConsumerState<LogEntryTile> createState() => _LogEntryTileState();
}

class _LogEntryTileState extends ConsumerState<LogEntryTile> {
  bool _expanded = false;

  static const _monoStyle = TextStyle(
    fontFamily: 'Consolas',
    fontFamilyFallback: ['Courier New', 'monospace'],
    fontSize: 12,
    height: 1.4,
  );

  static final _timeFormat = DateFormat('HH:mm:ss.SSS');
  static final _fullTimeFormat = DateFormat('yyyy-MM-dd HH:mm:ss.SSS');

  @override
  Widget build(BuildContext context) {
    final entry = widget.entry;
    final color = _levelColor(entry.level);
    final ts = _timeFormat.format(entry.timestamp);

    return GestureDetector(
      onTap: () => setState(() => _expanded = !_expanded),
      onSecondaryTapDown: (details) =>
          _showContextMenu(context, details.globalPosition, entry),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 3),
        color: _expanded
            ? Colors.white.withValues(alpha: 0.04)
            : Colors.transparent,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            // Compact log line
            _buildCompactLine(ts, entry, color),
            // Expanded metadata
            if (_expanded) _buildExpandedDetails(entry, color),
          ],
        ),
      ),
    );
  }

  Widget _buildCompactLine(String ts, LogEntry entry, Color color) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Timestamp
        Text('[$ts]', style: _monoStyle.copyWith(color: Colors.grey.shade600)),
        const SizedBox(width: 6),
        // Level badge
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 1),
          decoration: BoxDecoration(
            color: color.withValues(alpha: 0.15),
            borderRadius: BorderRadius.circular(3),
          ),
          child: Text(
            entry.level,
            style: _monoStyle.copyWith(
              color: color,
              fontWeight: FontWeight.w600,
              fontSize: 11,
            ),
          ),
        ),
        const SizedBox(width: 6),
        // Logger name
        Tooltip(
          message: entry.logger,
          child: Text(
            entry.shortLoggerName,
            style: _monoStyle.copyWith(
              color: Colors.tealAccent.withValues(alpha: 0.7),
            ),
          ),
        ),
        Text(':', style: _monoStyle.copyWith(color: Colors.grey.shade600)),
        const SizedBox(width: 6),
        // Message
        Expanded(
          child: Text(
            entry.message,
            style: _monoStyle.copyWith(
              color: Colors.white.withValues(alpha: 0.87),
            ),
            overflow: TextOverflow.ellipsis,
            maxLines: _expanded ? 20 : 1,
          ),
        ),
      ],
    );
  }

  Widget _buildExpandedDetails(LogEntry entry, Color color) {
    return Padding(
      padding: const EdgeInsets.only(left: 24, top: 4, bottom: 4),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _detailRow(
            'Logger',
            entry.logger,
            Colors.tealAccent.withValues(alpha: 0.7),
          ),
          _detailRow(
            'Timestamp',
            _fullTimeFormat.format(entry.timestamp),
            Colors.grey.shade500,
          ),
          if (entry.requestId != null)
            _detailRow('Request', entry.requestId!, color),
        ],
      ),
    );
  }

  Widget _detailRow(String label, String value, Color valueColor) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 1),
      child: Row(
        children: [
          SizedBox(
            width: 80,
            child: Text(
              label,
              style: _monoStyle.copyWith(
                color: Colors.grey.shade600,
                fontSize: 11,
              ),
            ),
          ),
          Flexible(
            child: Text(
              value,
              style: _monoStyle.copyWith(color: valueColor, fontSize: 11),
            ),
          ),
        ],
      ),
    );
  }

  void _showContextMenu(BuildContext context, Offset position, LogEntry entry) {
    final logNotifier = ref.read(logProvider.notifier);
    final loggerName = entry.logger;

    showMenu<String>(
      context: context,
      position: RelativeRect.fromLTRB(
        position.dx,
        position.dy,
        position.dx,
        position.dy,
      ),
      items: [
        PopupMenuItem(value: 'DEBUG', child: Text('Set $loggerName to DEBUG')),
        PopupMenuItem(value: 'INFO', child: Text('Set $loggerName to INFO')),
        PopupMenuItem(
          value: 'WARNING',
          child: Text('Set $loggerName to WARNING'),
        ),
        const PopupMenuDivider(),
        const PopupMenuItem(value: 'RESET', child: Text('Reset to default')),
      ],
    ).then((value) {
      if (value == null) return;
      if (value == 'RESET') {
        logNotifier.resetLoggerLevel(loggerName);
      } else {
        logNotifier.setLoggerLevel(loggerName, value);
      }
    });
  }
}
