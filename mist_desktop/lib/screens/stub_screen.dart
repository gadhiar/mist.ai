import 'package:flutter/material.dart';

/// Placeholder screen for unimplemented navigation destinations.
class StubScreen extends StatelessWidget {
  final String destinationName;

  const StubScreen({super.key, required this.destinationName});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Text(
        '$destinationName\nComing soon',
        textAlign: TextAlign.center,
        style: TextStyle(
          fontSize: 16,
          color: Colors.grey.shade600,
          fontWeight: FontWeight.w400,
        ),
      ),
    );
  }
}
