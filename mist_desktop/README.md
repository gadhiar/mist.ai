# MIST.AI Desktop - Flutter Application

Flutter desktop application for MIST.AI voice assistant with knowledge graph integration.

## Phase 1 Implementation Status ✅

### Completed Components

1. **Project Structure**
   - ✅ Organized folder structure (config, models, providers, screens, services, widgets)
   - ✅ Configuration files (app_config.dart, theme_config.dart)

2. **State Management**
   - ✅ Riverpod 3.x providers
   - ✅ WebSocket provider
   - ✅ Chat provider with full state management

3. **Services**
   - ✅ WebSocket service with connection management
   - ✅ Message handling (text, audio, status, errors)

4. **Models**
   - ✅ ChatMessage model
   - ✅ WebSocketMessage model
   - ✅ Message type enums

5. **UI Components**
   - ✅ Chat screen with message list
   - ✅ Chat message widget
   - ✅ Connection status indicator
   - ✅ Text input with send button
   - ✅ Streaming message support

## Setup Instructions

### Prerequisites

- Flutter SDK installed
- Backend server running on `localhost:8001`

### Installation

1. **Install dependencies:**
   ```bash
   cd mist_desktop
   flutter pub get
   ```

2. **Run the application:**
   ```bash
   flutter run -d windows
   # or for other platforms:
   # flutter run -d macos
   # flutter run -d linux
   ```

## Project Structure

```
lib/
├── config/
│   ├── app_config.dart          # App-wide configuration
│   └── theme_config.dart        # Theme and styling
├── models/
│   ├── message_model.dart       # Chat message model
│   └── websocket_message.dart   # WebSocket message model
├── providers/
│   ├── chat_provider.dart       # Chat state management
│   └── websocket_provider.dart  # WebSocket state
├── screens/
│   └── chat_screen.dart         # Main chat interface
├── services/
│   └── websocket_service.dart   # WebSocket communication
├── widgets/
│   ├── chat_message_widget.dart      # Message bubble
│   └── connection_status_widget.dart # Connection indicator
└── main.dart                    # App entry point
```

## Features

### Current (Phase 1)

- ✅ Real-time WebSocket connection to backend
- ✅ Send and receive text messages
- ✅ Connection status indicator
- ✅ Message history display
- ✅ Streaming AI responses
- ✅ Dark theme UI
- ✅ Auto-scroll to latest messages

### Coming Next (Phase 2)

- 🔲 Voice input/recording
- 🔲 Audio playback
- 🔲 Knowledge graph visualization
- 🔲 Entity highlighting
- 🔲 Conversation context

## Configuration

Edit `lib/config/app_config.dart` to change backend connection:

```dart
static const String wsHost = 'localhost';
static const int wsPort = 8001;
```

## Testing WebSocket Connection

1. Start the backend server:
   ```bash
   cd ..
   venv\Scripts\python.exe backend\server.py
   ```

2. Run the Flutter app:
   ```bash
   flutter run -d windows
   ```

3. Check connection status in the app bar
4. Send a test message

## Troubleshooting

### WebSocket Connection Failed

- Ensure backend is running on port 8001
- Check `backend\config.py` for port configuration
- Verify no firewall blocking localhost:8001

### Build Errors

- Run `flutter clean && flutter pub get`
- Ensure Flutter SDK is up to date: `flutter upgrade`

## Development

### Hot Reload

During development, use hot reload:
- Press `r` in terminal for hot reload
- Press `R` for hot restart

### Debug Mode

The app includes debug logging via the Logger package. Check console output for WebSocket events.

## Next Steps

See [FLUTTER_MIGRATION_PLAN.md](../FLUTTER_MIGRATION_PLAN.md) for the complete migration roadmap.
