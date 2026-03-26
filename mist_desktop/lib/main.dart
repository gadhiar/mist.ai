import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'config/theme_config.dart';
import 'config/app_config.dart';
import 'widgets/app_shell.dart';

void main() {
  runApp(const ProviderScope(child: MistAIApp()));
}

class MistAIApp extends StatelessWidget {
  const MistAIApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: AppConfig.appName,
      theme: ThemeConfig.darkTheme,
      debugShowCheckedModeBanner: false,
      home: const AppShell(),
    );
  }
}
