/// Application Configuration
class AppConfig {
  // WebSocket Configuration
  static const String wsHost = 'localhost';
  static const int wsPort = 8001;
  static String get wsUrl => 'ws://$wsHost:$wsPort/ws';

  // Audio Configuration
  static const int audioSampleRate = 16000;
  static const int audioChannels = 1;

  // UI Configuration
  static const Duration messageTimeout = Duration(seconds: 30);
  static const int maxMessageHistory = 100;

  // Feature Flags
  static const bool enableKnowledgeGraph = true;
  static const bool enableVoiceInput = true;

  // App Metadata
  static const String appName = 'MIST.AI';
  static const String appVersion = '1.0.0';
}
