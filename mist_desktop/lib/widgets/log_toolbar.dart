import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/log_provider.dart';

/// Level chips with their display colors.
const _levelChips = <String, Color>{
  'DEBUG': Colors.grey,
  'INFO': Colors.lightBlueAccent,
  'WARNING': Colors.amber,
  'ERROR': Colors.redAccent,
};

/// Group mode labels for the dropdown.
const _groupModeLabels = <LogGroupMode, String>{
  LogGroupMode.none: 'None',
  LogGroupMode.byRequest: 'By Request',
  LogGroupMode.byComponent: 'By Component',
};

/// Filter and control bar for the log viewer.
class LogToolbar extends ConsumerStatefulWidget {
  const LogToolbar({super.key});

  @override
  ConsumerState<LogToolbar> createState() => _LogToolbarState();
}

class _LogToolbarState extends ConsumerState<LogToolbar> {
  final TextEditingController _searchController = TextEditingController();
  Timer? _searchDebounce;

  @override
  void dispose() {
    _searchDebounce?.cancel();
    _searchController.dispose();
    super.dispose();
  }

  void _onSearchChanged(String value) {
    _searchDebounce?.cancel();
    _searchDebounce = Timer(const Duration(milliseconds: 300), () {
      ref.read(logProvider.notifier).setSearchQuery(value);
    });
  }

  @override
  Widget build(BuildContext context) {
    final logState = ref.watch(logProvider);
    final logNotifier = ref.read(logProvider.notifier);

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A1A),
        border: Border(
          bottom: BorderSide(color: Colors.white.withValues(alpha: 0.08)),
        ),
      ),
      child: Row(
        children: [
          // Level filter chips
          ..._levelChips.entries.map((entry) {
            final isActive = logState.activeLevels.contains(entry.key);
            return Padding(
              padding: const EdgeInsets.only(right: 6),
              child: FilterChip(
                label: Text(
                  entry.key,
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                    color: isActive ? Colors.black87 : entry.value,
                  ),
                ),
                selected: isActive,
                selectedColor: entry.value,
                backgroundColor: entry.value.withValues(alpha: 0.1),
                side: BorderSide(color: entry.value.withValues(alpha: 0.4)),
                showCheckmark: false,
                visualDensity: VisualDensity.compact,
                padding: const EdgeInsets.symmetric(horizontal: 4),
                materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                onSelected: (_) => logNotifier.toggleLevel(entry.key),
              ),
            );
          }),

          const SizedBox(width: 8),

          // Search field
          SizedBox(
            width: 200,
            height: 32,
            child: TextField(
              controller: _searchController,
              onChanged: _onSearchChanged,
              style: const TextStyle(fontSize: 12),
              decoration: InputDecoration(
                hintText: 'Search logs...',
                hintStyle: TextStyle(fontSize: 12, color: Colors.grey.shade600),
                prefixIcon: Icon(
                  Icons.search,
                  size: 16,
                  color: Colors.grey.shade600,
                ),
                prefixIconConstraints: const BoxConstraints(
                  minWidth: 32,
                  minHeight: 0,
                ),
                filled: true,
                fillColor: Colors.white.withValues(alpha: 0.05),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(6),
                  borderSide: BorderSide.none,
                ),
                contentPadding: const EdgeInsets.symmetric(
                  horizontal: 8,
                  vertical: 0,
                ),
              ),
            ),
          ),

          const SizedBox(width: 12),

          // Group mode dropdown
          SizedBox(
            height: 32,
            child: DropdownButtonHideUnderline(
              child: DropdownButton<LogGroupMode>(
                value: logState.groupMode,
                isDense: true,
                style: const TextStyle(fontSize: 12, color: Colors.white70),
                dropdownColor: const Color(0xFF2A2A2A),
                icon: const Icon(Icons.arrow_drop_down, size: 18),
                items: _groupModeLabels.entries
                    .map(
                      (entry) => DropdownMenuItem(
                        value: entry.key,
                        child: Text(entry.value),
                      ),
                    )
                    .toList(),
                onChanged: (mode) {
                  if (mode != null) logNotifier.setGroupMode(mode);
                },
              ),
            ),
          ),

          const Spacer(),

          // Pause / Resume button
          IconButton(
            icon: Icon(
              logState.isPaused ? Icons.play_arrow : Icons.pause,
              size: 18,
            ),
            tooltip: logState.isPaused ? 'Resume' : 'Pause',
            visualDensity: VisualDensity.compact,
            onPressed: logNotifier.togglePause,
          ),

          // Clear button
          IconButton(
            icon: const Icon(Icons.delete_sweep, size: 18),
            tooltip: 'Clear logs',
            visualDensity: VisualDensity.compact,
            onPressed: logNotifier.clear,
          ),
        ],
      ),
    );
  }
}
