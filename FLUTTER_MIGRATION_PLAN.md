# MIST.AI Flutter Desktop Migration Plan

**Date:** December 14, 2024
**Status:** Planning Phase
**Target Platform:** Windows Desktop (with iOS/Android ready for future)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Prerequisites & Environment Setup](#prerequisites--environment-setup)
3. [Project Architecture](#project-architecture)
4. [Required Flutter Packages](#required-flutter-packages)
5. [Project Structure](#project-structure)
6. [Implementation Phases](#implementation-phases)
7. [Knowledge Graph Visualization](#knowledge-graph-visualization)
8. [WebSocket Integration](#websocket-integration)
9. [UI Components Design](#ui-components-design)
10. [Audio Playback](#audio-playback)
11. [System Integration](#system-integration)
12. [Testing Strategy](#testing-strategy)
13. [Timeline & Milestones](#timeline--milestones)

---

## Executive Summary

### Why Flutter?

1. **Native Performance**: Compiled to machine code, minimal resource usage (~50-100MB RAM)
2. **Cross-Platform**: Same codebase for Windows, macOS, Linux, iOS, Android
3. **Knowledge Graph Viz**: Viable libraries (graphview, flutter_graph_view)
4. **Mature & Stable**: Production-ready desktop support since 2021
5. **Future-Proof**: Mobile migration requires zero frontend rewrite

### Key Goals

- Build desktop app with real-time voice conversation
- Beautiful knowledge graph visualization showing LLM's reasoning
- Seamless WebSocket connection to existing Python backend
- System tray integration for always-running experience
- Prepare foundation for future mobile app

### Success Criteria

-  Connect to Python backend via WebSocket (port 8001)
-  Display conversation with streaming LLM tokens
-  Visualize knowledge graph with interactive nodes/edges
-  Play audio from backend with gap-free experience
-  Show real-time entity extraction and retrieval
-  Run in system tray, <100MB RAM usage

---

## Prerequisites & Environment Setup

### 1. Install Flutter SDK

**Windows Installation:**

```powershell
# Download Flutter SDK
# Visit: https://docs.flutter.dev/get-started/install/windows

# Extract to C:\flutter
# Add to PATH: C:\flutter\bin

# Verify installation
flutter --version
# Expected: Flutter 3.24+ (stable channel)

# Run Flutter doctor
flutter doctor -v

# Expected output:
# [] Flutter (Channel stable, 3.24.x)
# [] Windows Version (Windows 10+)
# [] Visual Studio (2022+ with Desktop development with C++)
# [] VS Code or Android Studio
```

**Required Visual Studio Components:**
- Desktop development with C++
- Windows 10 SDK (10.0.17763.0 or later)
- C++ CMake tools for Windows

### 2. Install Dart (comes with Flutter)

```bash
dart --version
# Expected: Dart SDK version 3.4+ (stable)
```

### 3. Editor Setup

**VS Code (Recommended):**
```bash
# Install extensions
code --install-extension Dart-Code.dart-code
code --install-extension Dart-Code.flutter
```

**Or Android Studio:**
- Install Flutter plugin
- Install Dart plugin

### 4. Enable Desktop Support

```bash
# Enable Windows desktop support
flutter config --enable-windows-desktop

# Verify
flutter devices
# Should show: Windows (desktop)
```

### 5. Verify Installation

```bash
flutter doctor
# All checks should pass ()
```

---

## Project Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Flutter Desktop App (Windows/macOS/Linux)                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  UI Layer (Widgets)                                   │  │
│  │  ├─ ConversationView                                  │  │
│  │  ├─ KnowledgeGraphView (graphview)                    │  │
│  │  ├─ SettingsView                                      │  │
│  │  └─ SystemTrayIcon                                    │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  State Management (Riverpod)                          │  │
│  │  ├─ ConversationState                                 │  │
│  │  ├─ GraphState                                        │  │
│  │  ├─ ConnectionState                                   │  │
│  │  └─ AudioState                                        │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Services Layer                                       │  │
│  │  ├─ WebSocketService (connects to backend:8001)      │  │
│  │  ├─ AudioPlayerService (plays TTS audio)             │  │
│  │  ├─ GraphDataService (processes graph updates)       │  │
│  │  └─ SystemTrayService (tray icon management)         │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Models                                               │  │
│  │  ├─ Message                                           │  │
│  │  ├─ GraphNode                                         │  │
│  │  ├─ GraphEdge                                         │  │
│  │  └─ WebSocketMessage                                  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                         -> WebSocket
┌─────────────────────────────────────────────────────────────┐
│  Python Backend (Existing, No Changes)                      │
│  - FastAPI WebSocket Server (port 8001)                     │
│  - Voice Processor (VAD, STT, LLM, TTS)                     │
│  - Knowledge System (Neo4j, Entity Extraction, Retrieval)   │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Frontend (Flutter):**
- **Language**: Dart 3.4+
- **Framework**: Flutter 3.24+ (stable)
- **State Management**: Riverpod (modern, performant)
- **Graph Viz**: graphview or flutter_graph_view
- **Audio**: audioplayers or just_audio
- **WebSocket**: web_socket_channel
- **System Tray**: system_tray or tray_manager
- **UI Components**: Material Design 3

**Backend (Unchanged):**
- Python 3.11+
- FastAPI + WebSockets
- Neo4j + Knowledge Graph

---

## Required Flutter Packages

### Core Dependencies

```yaml
# pubspec.yaml
dependencies:
  flutter:
    sdk: flutter

  # State Management
  flutter_riverpod: ^2.5.0         # Modern, performant state management
  riverpod_annotation: ^2.3.0      # Code generation for providers

  # WebSocket & Network
  web_socket_channel: ^3.0.0       # WebSocket client
  http: ^1.2.0                     # HTTP requests (if needed)

  # Knowledge Graph Visualization
  graphview: ^1.2.0                # Primary graph visualization
  # OR
  flutter_graph_view: ^0.3.0       # Alternative with force-directed layout

  # Audio Playback
  just_audio: ^0.9.36              # High-quality audio playback
  # OR
  audioplayers: ^6.0.0             # Alternative audio player

  # UI Components
  flutter_animate: ^4.5.0          # Smooth animations
  animated_text_kit: ^4.2.2        # Text animations for LLM streaming
  loading_animation_widget: ^1.2.1 # Loading indicators

  # System Integration
  system_tray: ^2.0.3              # System tray icon
  window_manager: ^0.3.8           # Window controls (minimize, maximize)
  launch_at_startup: ^0.2.2        # Auto-start on boot

  # Utilities
  logger: ^2.0.2                   # Logging
  freezed_annotation: ^2.4.1       # Immutable models
  json_annotation: ^4.8.1          # JSON serialization
  intl: ^0.19.0                    # Internationalization (date/time formatting)

dev_dependencies:
  flutter_test:
    sdk: flutter

  # Code Generation
  build_runner: ^2.4.8             # Code generation runner
  riverpod_generator: ^2.4.0       # Riverpod code gen
  freezed: ^2.4.7                  # Model generation
  json_serializable: ^6.7.1        # JSON code gen

  # Linting
  flutter_lints: ^3.0.1            # Official Flutter lints
```

### Package Justifications

**Riverpod vs Provider/BLoC:**
- More performant (compile-time safety)
- Better testing support
- Cleaner syntax with code generation

**graphview vs flutter_graph_view:**
- **graphview**: More mature, better docs, hierarchical graphs
- **flutter_graph_view**: Better force-directed layouts, newer
- **Recommendation**: Start with graphview, evaluate flutter_graph_view later

**just_audio vs audioplayers:**
- **just_audio**: Better for streaming, more features
- **audioplayers**: Simpler, lighter
- **Recommendation**: just_audio (better for gap-free playback)

---

## Project Structure

```
mist_desktop/                        # Flutter app root
├── lib/
│   ├── main.dart                    # App entry point
│   │
│   ├── app/
│   │   ├── app.dart                 # Main app widget
│   │   ├── router.dart              # Navigation/routing
│   │   └── theme.dart               # App theme (dark mode)
│   │
│   ├── core/
│   │   ├── constants/
│   │   │   ├── api_constants.dart   # WebSocket URL, etc.
│   │   │   └── app_constants.dart   # App-wide constants
│   │   ├── utils/
│   │   │   ├── logger.dart          # Logging setup
│   │   │   └── extensions.dart      # Dart extensions
│   │   └── config/
│   │       └── env_config.dart      # Environment configuration
│   │
│   ├── models/
│   │   ├── message.dart             # Chat message model
│   │   ├── message.freezed.dart     # Generated
│   │   ├── message.g.dart           # Generated
│   │   ├── graph_node.dart          # Knowledge graph node
│   │   ├── graph_edge.dart          # Knowledge graph edge
│   │   └── websocket_message.dart   # WebSocket message types
│   │
│   ├── services/
│   │   ├── websocket/
│   │   │   ├── websocket_service.dart        # WebSocket client
│   │   │   └── websocket_message_handler.dart # Message parsing
│   │   ├── audio/
│   │   │   └── audio_player_service.dart     # Audio playback
│   │   ├── graph/
│   │   │   └── graph_data_service.dart       # Graph data processing
│   │   └── system/
│   │       ├── system_tray_service.dart      # System tray
│   │       └── window_service.dart           # Window management
│   │
│   ├── providers/
│   │   ├── conversation_provider.dart        # Conversation state
│   │   ├── graph_provider.dart               # Graph state
│   │   ├── connection_provider.dart          # WebSocket state
│   │   └── audio_provider.dart               # Audio state
│   │
│   ├── screens/
│   │   ├── home/
│   │   │   ├── home_screen.dart              # Main screen
│   │   │   └── widgets/
│   │   │       ├── conversation_panel.dart   # Left: conversation
│   │   │       └── graph_panel.dart          # Right: graph viz
│   │   ├── settings/
│   │   │   └── settings_screen.dart          # Settings
│   │   └── splash/
│   │       └── splash_screen.dart            # Loading screen
│   │
│   └── widgets/
│       ├── common/
│       │   ├── loading_indicator.dart        # Custom loading
│       │   └── error_widget.dart             # Error display
│       ├── conversation/
│       │   ├── message_bubble.dart           # Chat bubble
│       │   ├── typing_indicator.dart         # LLM typing animation
│       │   └── audio_visualizer.dart         # Audio waveform
│       └── graph/
│           ├── graph_viewer.dart             # Graph visualization widget
│           ├── node_widget.dart              # Custom node rendering
│           └── edge_widget.dart              # Custom edge rendering
│
├── assets/
│   ├── images/
│   │   └── logo.png                          # App logo
│   └── icons/
│       └── tray_icon.ico                     # System tray icon
│
├── windows/                                  # Windows-specific
├── macos/                                    # macOS-specific (future)
├── linux/                                    # Linux-specific (future)
│
├── test/
│   ├── unit/
│   ├── widget/
│   └── integration/
│
├── pubspec.yaml                              # Dependencies
├── analysis_options.yaml                     # Linter config
└── README.md                                 # Flutter app docs
```

---

## Implementation Phases

### Phase 1: Project Setup & Foundation (Week 1)

**Goal**: Create Flutter project, setup architecture, establish WebSocket connection

**Tasks:**

1. **Create Flutter Project**
   ```bash
   cd d:\Users\rajga\mist.ai
   flutter create --org com.mistai --project-name mist_desktop mist_desktop
   cd mist_desktop
   flutter pub add flutter_riverpod web_socket_channel logger
   ```

2. **Setup Project Structure**
   - Create folder structure (as defined above)
   - Setup routing (if multi-screen)
   - Configure app theme (dark mode)

3. **Implement WebSocket Service**
   ```dart
   // lib/services/websocket/websocket_service.dart
   class WebSocketService {
     final _channel = WebSocketChannel.connect(
       Uri.parse('ws://localhost:8001/ws'),
     );

     Stream<dynamic> get messages => _channel.stream;

     void send(Map<String, dynamic> message) {
       _channel.sink.add(jsonEncode(message));
     }
   }
   ```

4. **Create Basic UI Shell**
   - Main window layout (conversation + graph panels)
   - Connection status indicator
   - Basic styling

5. **Test WebSocket Connection**
   - Connect to existing Python backend
   - Send/receive test messages
   - Handle connection errors

**Deliverable**: Flutter app connects to backend, displays "Connected" status

---

### Phase 2: Conversation UI & Message Handling (Week 1-2)

**Goal**: Display real-time conversation with streaming LLM tokens

**Tasks:**

1. **Define Message Models**
   ```dart
   @freezed
   class Message with _$Message {
     factory Message({
       required String id,
       required String role,  // 'user' or 'assistant'
       required String content,
       DateTime? timestamp,
     }) = _Message;
   }
   ```

2. **Implement Conversation Provider**
   ```dart
   @riverpod
   class ConversationNotifier extends _$ConversationNotifier {
     @override
     List<Message> build() => [];

     void addMessage(Message message) {
       state = [...state, message];
     }

     void updateLastMessage(String token) {
       // Append token to last message (streaming)
       state = [
         ...state.take(state.length - 1),
         state.last.copyWith(
           content: state.last.content + token,
         ),
       ];
     }
   }
   ```

3. **Build Conversation UI**
   - ScrollView with message bubbles
   - User messages (right-aligned, blue)
   - Assistant messages (left-aligned, gray)
   - Streaming animation (typing indicator -> token-by-token reveal)

4. **Handle WebSocket Messages**
   ```dart
   // Listen for 'transcription', 'llm_token', 'llm_response', etc.
   _wsService.messages.listen((message) {
     final data = jsonDecode(message);
     switch (data['type']) {
       case 'transcription':
         _addUserMessage(data['text']);
       case 'llm_token':
         _appendToken(data['token']);
       case 'llm_response':
         _finalizeAssistantMessage(data['text']);
     }
   });
   ```

5. **Add Text Input (Optional)**
   - For testing without voice
   - Text field at bottom
   - Send button

**Deliverable**: Conversation UI displays messages with streaming LLM tokens

---

### Phase 3: Knowledge Graph Visualization (Week 2)

**Goal**: Beautiful interactive knowledge graph showing entities and relationships

**Tasks:**

1. **Choose Graph Library**
   - Install `graphview` or `flutter_graph_view`
   - Test with sample data

2. **Define Graph Models**
   ```dart
   @freezed
   class GraphNode with _$GraphNode {
     factory GraphNode({
       required String id,
       required String label,
       required String type,  // 'User', 'Technology', 'Project', etc.
       Map<String, dynamic>? properties,
     }) = _GraphNode;
   }

   @freezed
   class GraphEdge with _$GraphEdge {
     factory GraphEdge({
       required String id,
       required String source,
       required String target,
       required String relationship,  // 'USES', 'WORKS_ON', etc.
       Map<String, dynamic>? properties,
     }) = _GraphEdge;
   }
   ```

3. **Implement Graph Provider**
   ```dart
   @riverpod
   class GraphNotifier extends _$GraphNotifier {
     @override
     GraphData build() => GraphData(nodes: [], edges: []);

     void addNode(GraphNode node) {
       state = state.copyWith(
         nodes: [...state.nodes, node],
       );
     }

     void addEdge(GraphEdge edge) {
       state = state.copyWith(
         edges: [...state.edges, edge],
       );
     }

     void highlightPath(List<String> nodeIds) {
       // Highlight nodes in retrieval path
     }
   }
   ```

4. **Build Graph Visualization Widget**
   ```dart
   // Using graphview
   class KnowledgeGraphViewer extends ConsumerWidget {
     @override
     Widget build(BuildContext context, WidgetRef ref) {
       final graphData = ref.watch(graphProvider);

       final graph = Graph();

       // Add nodes
       for (var node in graphData.nodes) {
         graph.addNode(Node.Id(node.id));
       }

       // Add edges
       for (var edge in graphData.edges) {
         graph.addEdge(
           graph.getNodeUsingId(edge.source),
           graph.getNodeUsingId(edge.target),
         );
       }

       return InteractiveViewer(
         constrained: false,
         boundaryMargin: EdgeInsets.all(100),
         minScale: 0.01,
         maxScale: 5.0,
         child: GraphView(
           graph: graph,
           algorithm: FruchtermanReingoldAlgorithm(),
           paint: Paint()
             ..color = Colors.blue
             ..strokeWidth = 2
             ..style = PaintingStyle.stroke,
           builder: (Node node) {
             return _buildNodeWidget(node);
           },
         ),
       );
     }

     Widget _buildNodeWidget(Node node) {
       final data = graphData.nodes.firstWhere((n) => n.id == node.key);

       return Container(
         padding: EdgeInsets.all(16),
         decoration: BoxDecoration(
           color: _getNodeColor(data.type),
           borderRadius: BorderRadius.circular(8),
           border: Border.all(color: Colors.white, width: 2),
         ),
         child: Text(
           data.label,
           style: TextStyle(color: Colors.white),
         ),
       );
     }
   }
   ```

5. **Handle Graph Update Messages**
   ```dart
   // WebSocket messages: 'graph_update', 'entity_extracted', 'retrieval_path'
   case 'entity_extracted':
     _graphNotifier.addNode(GraphNode.fromJson(data['node']));
     _graphNotifier.addEdge(GraphEdge.fromJson(data['edge']));

   case 'retrieval_path':
     _graphNotifier.highlightPath(data['path']);
   ```

6. **Add Interactivity**
   - Pan/zoom (via InteractiveViewer)
   - Click node -> show details
   - Hover -> highlight connections
   - Animate new nodes appearing

**Deliverable**: Interactive knowledge graph showing real-time entity extraction and retrieval

---

### Phase 4: Audio Playback (Week 2-3)

**Goal**: Play TTS audio from backend with gap-free experience

**Tasks:**

1. **Install Audio Package**
   ```bash
   flutter pub add just_audio
   ```

2. **Implement Audio Service**
   ```dart
   class AudioPlayerService {
     final _player = AudioPlayer();
     final _audioQueue = <Uint8List>[];
     bool _isPlaying = false;

     Future<void> playAudioChunk(Uint8List audioData) async {
       _audioQueue.add(audioData);
       if (!_isPlaying) {
         await _playNext();
       }
     }

     Future<void> _playNext() async {
       if (_audioQueue.isEmpty) {
         _isPlaying = false;
         return;
       }

       _isPlaying = true;
       final chunk = _audioQueue.removeAt(0);

       // Convert bytes to audio source
       final source = BytesSource(chunk);
       await _player.setAudioSource(source);
       await _player.play();

       // Wait for completion
       await _player.processingStateStream
         .firstWhere((state) => state == ProcessingState.completed);

       // Play next chunk
       await _playNext();
     }

     void stop() {
       _player.stop();
       _audioQueue.clear();
       _isPlaying = false;
     }
   }
   ```

3. **Handle Audio WebSocket Messages**
   ```dart
   case 'audio_chunk':
     final audioBytes = base64Decode(data['audio']);
     await _audioService.playAudioChunk(audioBytes);

   case 'audio_complete':
     // All chunks received
   ```

4. **Add Audio Visualizer (Optional)**
   - Waveform visualization
   - Volume meter
   - Playing/paused indicator

**Deliverable**: Gap-free audio playback from backend TTS

---

### Phase 5: System Integration & Polish (Week 3)

**Goal**: Desktop features (system tray, window management, auto-start)

**Tasks:**

1. **System Tray Icon**
   ```dart
   class SystemTrayService {
     final SystemTray _tray = SystemTray();

     Future<void> init() async {
       await _tray.initSystemTray(
         title: "MIST.AI",
         iconPath: "assets/icons/tray_icon.ico",
       );

       final menu = Menu();
       menu.buildFrom([
         MenuItemLabel(
           label: 'Show Window',
           onClicked: (_) => _showWindow(),
         ),
         MenuItemLabel(
           label: 'Settings',
           onClicked: (_) => _openSettings(),
         ),
         MenuSeparator(),
         MenuItemLabel(
           label: 'Exit',
           onClicked: (_) => exit(0),
         ),
       ]);

       await _tray.setContextMenu(menu);
     }
   }
   ```

2. **Window Management**
   ```dart
   class WindowService {
     Future<void> init() async {
       await windowManager.ensureInitialized();

       const windowOptions = WindowOptions(
         size: Size(1400, 900),
         minimumSize: Size(1000, 700),
         center: true,
         backgroundColor: Colors.transparent,
         skipTaskbar: false,
         titleBarStyle: TitleBarStyle.hidden,
       );

       await windowManager.waitUntilReadyToShow(windowOptions);
       await windowManager.show();
       await windowManager.focus();
     }

     void minimize() {
       windowManager.minimize();
     }

     void hide() {
       windowManager.hide();
     }
   }
   ```

3. **Auto-Start on Boot (Optional)**
   ```dart
   final launchAtStartup = LaunchAtStartup();

   await launchAtStartup.setup(
     appName: "MIST Desktop",
     appPath: Platform.resolvedExecutable,
   );

   await launchAtStartup.enable();
   ```

4. **Global Keyboard Shortcuts (Optional)**
   - Hotkey to show/hide window (e.g., Ctrl+Shift+M)
   - Push-to-talk key

5. **Polish UI**
   - Smooth animations
   - Loading states
   - Error handling
   - Connection retry logic
   - Dark mode theme
   - Custom window chrome (minimize, maximize, close buttons)

**Deliverable**: Production-ready desktop app with native features

---

### Phase 6: Testing & Optimization (Week 3-4)

**Goal**: Ensure stability, performance, and user experience

**Tasks:**

1. **Unit Tests**
   - Test models (freezed classes)
   - Test services (mock WebSocket)
   - Test providers (state changes)

2. **Widget Tests**
   - Test conversation UI
   - Test graph visualization
   - Test audio controls

3. **Integration Tests**
   - Test full conversation flow
   - Test graph updates
   - Test audio playback

4. **Performance Optimization**
   - Profile memory usage (target <100MB)
   - Optimize graph rendering (1000+ nodes)
   - Reduce CPU usage when idle

5. **Error Handling**
   - WebSocket reconnection
   - Audio playback failures
   - Graph rendering errors

6. **User Testing**
   - Real conversation scenarios
   - Knowledge extraction flow
   - Graph visualization clarity

**Deliverable**: Stable, optimized desktop app ready for daily use

---

## Knowledge Graph Visualization

### Library Choice: `graphview`

**Why graphview:**
- Mature, well-documented
- Multiple layout algorithms (Fruchterman-Reingold, Sugiyama, etc.)
- Good for hierarchical and network graphs
- Interactive with InteractiveViewer

### Graph Layout Strategy

**Algorithm: Fruchterman-Reingold (Force-Directed)**

```dart
final algorithm = FruchtermanReingoldAlgorithm(
  iterations: 1000,
  attractionConstant: 0.5,
  repulsionConstant: 5000,
);
```

**Why Force-Directed:**
- Natural clustering of related entities
- Clear visualization of relationships
- Automatic layout (no manual positioning)

### Node Styling by Type

```dart
Color _getNodeColor(String type) {
  switch (type) {
    case 'User':
      return Colors.blue[700]!;
    case 'Technology':
      return Colors.green[600]!;
    case 'Project':
      return Colors.orange[600]!;
    case 'Person':
      return Colors.purple[500]!;
    case 'Topic':
      return Colors.teal[500]!;
    default:
      return Colors.grey[600]!;
  }
}

double _getNodeSize(GraphNode node) {
  // Size based on number of connections
  final connections = _graphData.edges
    .where((e) => e.source == node.id || e.target == node.id)
    .length;

  return 40 + (connections * 5).toDouble();
}
```

### Edge Styling by Relationship

```dart
Color _getEdgeColor(String relationship) {
  switch (relationship) {
    case 'USES':
      return Colors.blue[300]!;
    case 'WORKS_ON':
      return Colors.green[300]!;
    case 'EXPERT_IN':
      return Colors.amber[600]!;
    case 'LEARNING':
      return Colors.lightBlue[300]!;
    default:
      return Colors.grey[400]!;
  }
}
```

### Real-Time Updates

**Animation Strategy:**

```dart
class GraphNotifier extends StateNotifier<GraphData> {
  void addNode(GraphNode node) {
    state = state.copyWith(
      nodes: [...state.nodes, node],
    );

    // Trigger animation
    _animateNodeAppearance(node.id);
  }

  void highlightRetrievalPath(List<String> path) {
    // Highlight nodes in retrieval sequence
    for (var i = 0; i < path.length; i++) {
      Future.delayed(Duration(milliseconds: i * 300), () {
        _highlightNode(path[i]);
      });
    }
  }
}
```

### Performance Considerations

**For Large Graphs (500+ nodes):**

1. **Viewport culling**: Only render visible nodes
2. **Simplify layout**: Reduce iterations
3. **LOD (Level of Detail)**: Show fewer details when zoomed out
4. **Clustering**: Group distant nodes

---

## WebSocket Integration

### Message Protocol

**Backend -> Frontend Messages:**

```dart
// Message types from Python backend
enum WSMessageType {
  // Connection
  status,              // {"type": "status", "message": "Connected"}

  // VAD
  vadStatus,           // {"type": "vad_status", "status": "speech_started"}

  // Transcription
  transcription,       // {"type": "transcription", "text": "user message"}

  // LLM
  llmToken,           // {"type": "llm_token", "token": "word"}
  llmResponse,        // {"type": "llm_response", "text": "full response"}

  // Audio
  audioChunk,         // {"type": "audio_chunk", "audio": [...], "sample_rate": 24000}
  audioComplete,      // {"type": "audio_complete"}

  // Knowledge Graph (NEW - need to add to backend)
  graphUpdate,        // {"type": "graph_update", "nodes": [...], "edges": [...]}
  entityExtracted,    // {"type": "entity_extracted", "node": {...}, "edge": {...}}
  retrievalPath,      // {"type": "retrieval_path", "path": ["User", "Python", ...]}

  // Error
  error,              // {"type": "error", "message": "..."}
}
```

**Frontend -> Backend Messages:**

```dart
// Send text message
{
  "type": "text",
  "text": "user message"
}

// Manual interrupt
{
  "type": "interrupt"
}

// Reset VAD
{
  "type": "reset_vad"
}
```

### WebSocket Service Implementation

```dart
class WebSocketService {
  IOWebSocketChannel? _channel;
  final _connectionController = StreamController<ConnectionState>.broadcast();
  final _messageController = StreamController<WSMessage>.broadcast();

  Stream<ConnectionState> get connectionState => _connectionController.stream;
  Stream<WSMessage> get messages => _messageController.stream;

  Future<void> connect(String url) async {
    try {
      _channel = IOWebSocketChannel.connect(Uri.parse(url));
      _connectionController.add(ConnectionState.connected);

      // Listen to messages
      _channel!.stream.listen(
        _onMessage,
        onError: _onError,
        onDone: _onDone,
      );
    } catch (e) {
      _connectionController.add(ConnectionState.error);
      logger.e('WebSocket connection failed: $e');
    }
  }

  void _onMessage(dynamic message) {
    try {
      final data = jsonDecode(message);
      final wsMessage = WSMessage.fromJson(data);
      _messageController.add(wsMessage);
    } catch (e) {
      logger.e('Failed to parse WebSocket message: $e');
    }
  }

  void send(Map<String, dynamic> message) {
    if (_channel == null) return;
    _channel!.sink.add(jsonEncode(message));
  }

  void disconnect() {
    _channel?.sink.close();
    _channel = null;
    _connectionController.add(ConnectionState.disconnected);
  }

  void dispose() {
    disconnect();
    _connectionController.close();
    _messageController.close();
  }
}
```

### Reconnection Logic

```dart
class ReconnectingWebSocketService extends WebSocketService {
  Timer? _reconnectTimer;
  int _reconnectAttempts = 0;
  static const _maxReconnectDelay = Duration(seconds: 30);

  @override
  void _onDone() {
    _connectionController.add(ConnectionState.disconnected);
    _scheduleReconnect();
  }

  void _scheduleReconnect() {
    _reconnectAttempts++;
    final delay = Duration(
      seconds: math.min(_reconnectAttempts * 2, _maxReconnectDelay.inSeconds),
    );

    logger.i('Reconnecting in ${delay.inSeconds}s (attempt $_reconnectAttempts)');

    _reconnectTimer?.cancel();
    _reconnectTimer = Timer(delay, () {
      connect('ws://localhost:8001/ws');
    });
  }

  @override
  Future<void> connect(String url) async {
    _reconnectAttempts = 0;
    await super.connect(url);
  }
}
```

---

## UI Components Design

### Main Window Layout

```
┌────────────────────────────────────────────────────────────┐
│  [≡] MIST.AI                           [○ Connected] [_][□][X]
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌─────────────────────────┐  ┌────────────────────────┐  │
│  │  Conversation           │  │  Knowledge Graph       │  │
│  │  ┌─────────────────────┐│  │                        │  │
│  │  │ User: Hello         ││  │     [User]             │  │
│  │  │                     ││  │       │                │  │
│  │  │ Assistant:          ││  │       ├─[USES]─[Python]│  │
│  │  │ Hi! How can I help? ││  │       │                │  │
│  │  │                     ││  │       └─[WORKS_ON]─    │  │
│  │  │ [Typing...]         ││  │         [MIST.AI]      │  │
│  │  │                     ││  │                        │  │
│  │  └─────────────────────┘│  │  [Pan/Zoom controls]   │  │
│  │                         │  │                        │  │
│  │  ┌─────────────────────┐│  │  Stats: 15 nodes,     │  │
│  │  │ Type message...   [->]││  │         23 edges      │  │
│  │  └─────────────────────┘│  │                        │  │
│  └─────────────────────────┘  └────────────────────────┘  │
│                                                            │
│  [Status: Backend connected | Last update: 2s ago]         │
└────────────────────────────────────────────────────────────┘
```

### Theme Configuration

```dart
final darkTheme = ThemeData.dark().copyWith(
  colorScheme: ColorScheme.dark(
    primary: Colors.blue[400]!,
    secondary: Colors.teal[300]!,
    surface: Color(0xFF1E1E1E),
    background: Color(0xFF121212),
  ),
  scaffoldBackgroundColor: Color(0xFF121212),
  cardTheme: CardTheme(
    color: Color(0xFF1E1E1E),
    elevation: 2,
  ),
  textTheme: TextTheme(
    bodyLarge: TextStyle(fontSize: 16, color: Colors.white),
    bodyMedium: TextStyle(fontSize: 14, color: Colors.white70),
  ),
);
```

### Conversation Message Bubble

```dart
class MessageBubble extends StatelessWidget {
  final Message message;

  @override
  Widget build(BuildContext context) {
    final isUser = message.role == 'user';

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        constraints: BoxConstraints(maxWidth: 600),
        margin: EdgeInsets.symmetric(vertical: 8, horizontal: 16),
        padding: EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: isUser ? Colors.blue[700] : Colors.grey[800],
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              isUser ? 'You' : 'MIST',
              style: TextStyle(
                fontSize: 12,
                color: Colors.white70,
                fontWeight: FontWeight.bold,
              ),
            ),
            SizedBox(height: 4),
            Text(
              message.content,
              style: TextStyle(fontSize: 16, color: Colors.white),
            ),
            if (message.timestamp != null)
              Padding(
                padding: EdgeInsets.only(top: 4),
                child: Text(
                  _formatTime(message.timestamp!),
                  style: TextStyle(fontSize: 10, color: Colors.white54),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
```

---

## Audio Playback

### just_audio Implementation

```dart
class AudioPlayerService {
  final AudioPlayer _player = AudioPlayer();
  final _playlistController = ConcatenatingAudioSource(children: []);

  Future<void> init() async {
    await _player.setAudioSource(_playlistController);
    _player.play();
  }

  Future<void> addChunk(Uint8List audioData) async {
    // Convert bytes to audio source
    final source = BytesSource(audioData);
    await _playlistController.add(source);
  }

  void stop() {
    _player.stop();
    _playlistController.clear();
  }

  void dispose() {
    _player.dispose();
  }
}

// BytesSource implementation
class BytesSource extends StreamAudioSource {
  final Uint8List _bytes;

  BytesSource(this._bytes);

  @override
  Future<StreamAudioResponse> request([int? start, int? end]) async {
    start ??= 0;
    end ??= _bytes.length;

    return StreamAudioResponse(
      sourceLength: _bytes.length,
      contentLength: end - start,
      offset: start,
      stream: Stream.value(_bytes.sublist(start, end)),
      contentType: 'audio/wav',  // Adjust based on backend format
    );
  }
}
```

### Handling Audio WebSocket Messages

```dart
// In WebSocket message handler
case WSMessageType.audioChunk:
  final audioBase64 = message.data['audio'];
  final audioBytes = base64Decode(audioBase64);
  await ref.read(audioPlayerProvider).addChunk(audioBytes);
  break;

case WSMessageType.audioComplete:
  // All chunks received, audio will play to completion
  logger.i('Audio playback complete');
  break;
```

---

## System Integration

### System Tray Configuration

```dart
// In main.dart initialization
Future<void> _initSystemTray() async {
  final systemTray = SystemTray();

  await systemTray.initSystemTray(
    title: "MIST.AI",
    iconPath: Platform.isWindows
        ? 'assets/icons/tray_icon.ico'
        : 'assets/icons/tray_icon.png',
  );

  final menu = Menu();
  await menu.buildFrom([
    MenuItemLabel(
      label: 'Show Window',
      onClicked: (menuItem) => windowManager.show(),
    ),
    MenuItemLabel(
      label: 'Hide Window',
      onClicked: (menuItem) => windowManager.hide(),
    ),
    MenuSeparator(),
    MenuItemLabel(
      label: 'Settings',
      onClicked: (menuItem) => _openSettings(),
    ),
    MenuItemLabel(
      label: 'About',
      onClicked: (menuItem) => _showAbout(),
    ),
    MenuSeparator(),
    MenuItemLabel(
      label: 'Exit',
      onClicked: (menuItem) => exit(0),
    ),
  ]);

  await systemTray.setContextMenu(menu);

  // Handle tray icon clicks
  systemTray.registerSystemTrayEventHandler((eventName) {
    if (eventName == kSystemTrayEventClick) {
      Platform.isWindows ? windowManager.show() : systemTray.popUpContextMenu();
    }
  });
}
```

### Window Manager Setup

```dart
Future<void> _initWindowManager() async {
  await windowManager.ensureInitialized();

  const windowOptions = WindowOptions(
    size: Size(1400, 900),
    minimumSize: Size(1000, 700),
    center: true,
    backgroundColor: Colors.transparent,
    skipTaskbar: false,
    titleBarStyle: TitleBarStyle.hidden,  // Custom title bar
  );

  await windowManager.waitUntilReadyToShow(windowOptions, () async {
    await windowManager.show();
    await windowManager.focus();
  });
}
```

---

## Testing Strategy

### Unit Tests

```dart
// test/services/websocket_service_test.dart
void main() {
  group('WebSocketService', () {
    late WebSocketService service;

    setUp(() {
      service = WebSocketService();
    });

    test('connects successfully', () async {
      await service.connect('ws://localhost:8001/ws');
      expect(
        service.connectionState,
        emitsInOrder([ConnectionState.connected]),
      );
    });

    test('receives and parses messages', () async {
      // Mock WebSocket channel
      final mockChannel = MockWebSocketChannel();
      when(mockChannel.stream).thenAnswer((_) => Stream.value(
        jsonEncode({'type': 'status', 'message': 'Connected'}),
      ));

      service.setChannel(mockChannel);

      expect(
        service.messages,
        emits(isA<WSMessage>()),
      );
    });
  });
}
```

### Widget Tests

```dart
// test/widgets/message_bubble_test.dart
void main() {
  testWidgets('MessageBubble displays correctly', (tester) async {
    final message = Message(
      id: '1',
      role: 'user',
      content: 'Hello',
      timestamp: DateTime.now(),
    );

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: MessageBubble(message: message),
        ),
      ),
    );

    expect(find.text('Hello'), findsOneWidget);
    expect(find.text('You'), findsOneWidget);
  });
}
```

### Integration Tests

```dart
// integration_test/app_test.dart
void main() {
  testWidgets('Full conversation flow', (tester) async {
    await tester.pumpWidget(MistApp());

    // Wait for connection
    await tester.pumpAndSettle();
    expect(find.text('Connected'), findsOneWidget);

    // Send message
    await tester.enterText(find.byType(TextField), 'Hello MIST');
    await tester.tap(find.byIcon(Icons.send));
    await tester.pump();

    // Verify message appears
    expect(find.text('Hello MIST'), findsOneWidget);

    // Wait for response
    await tester.pumpAndSettle(Duration(seconds: 5));

    // Verify assistant response
    expect(find.textContaining('Hi'), findsOneWidget);
  });
}
```

---

## Timeline & Milestones

### Week 1: Foundation
- **Days 1-2**: Setup Flutter, create project, basic structure
- **Days 3-4**: WebSocket connection, basic UI shell
- **Days 5-7**: Conversation UI, message display

**Milestone**: App connects to backend, displays messages

### Week 2: Core Features
- **Days 1-3**: Knowledge graph visualization (graphview integration)
- **Days 4-5**: Graph updates from WebSocket, animations
- **Days 6-7**: Audio playback implementation

**Milestone**: Working conversation + graph + audio

### Week 3: Polish & Integration
- **Days 1-2**: System tray, window management
- **Days 3-4**: UI polish, animations, error handling
- **Days 5-7**: Testing, bug fixes, optimization

**Milestone**: Production-ready desktop app

### Week 4: Optional/Buffer
- Performance optimization
- Advanced features (hotkeys, settings panel)
- Documentation
- User testing feedback

**Total: 3-4 weeks to production-ready app**

---

## Backend Changes Required

### New WebSocket Messages (Add to Python backend)

```python
# backend/server.py or conversation_handler.py

# After entity extraction
await message_queue.put({
    "type": "entity_extracted",
    "node": {
        "id": entity_id,
        "label": entity_label,
        "type": entity_type,
    },
    "edge": {
        "id": edge_id,
        "source": source_id,
        "target": target_id,
        "relationship": relationship_type,
    }
})

# After knowledge retrieval
await message_queue.put({
    "type": "retrieval_path",
    "path": ["User", "Python", "FastAPI"],  # Node IDs in retrieval order
    "query": user_query,
})

# Send full graph periodically or on request
await message_queue.put({
    "type": "graph_update",
    "nodes": [
        {"id": "User", "label": "User", "type": "Person"},
        {"id": "Python", "label": "Python", "type": "Technology"},
    ],
    "edges": [
        {"id": "e1", "source": "User", "target": "Python", "relationship": "USES"},
    ]
})
```

### Optional: Graph Query Endpoint

```python
# For initial graph load
@app.get("/api/graph")
async def get_graph():
    """Return current knowledge graph"""
    graph_store = get_graph_store()

    nodes = graph_store.get_all_entities()
    edges = graph_store.get_all_relationships()

    return {
        "nodes": [
            {"id": n.id, "label": n.label, "type": n.type}
            for n in nodes
        ],
        "edges": [
            {"id": e.id, "source": e.source, "target": e.target, "relationship": e.type}
            for e in edges
        ]
    }
```

---

## Next Steps

1. **Review this plan** - Confirm scope and timeline
2. **Install Flutter SDK** - Follow prerequisites section
3. **Create Flutter project** - Start Phase 1
4. **Implement WebSocket connection** - Test with existing backend
5. **Build conversation UI** - Phase 2
6. **Add knowledge graph viz** - Phase 3
7. **Integrate audio** - Phase 4
8. **Polish & ship** - Phase 5-6

---

## Questions to Answer Before Starting

1. **Mobile timeline?** When do you realistically need iOS/Android?
2. **Graph complexity?** Max number of nodes/edges to support?
3. **Backend changes?** Can we add graph update WebSocket messages?
4. **Design preferences?** Any specific UI/UX inspirations?
5. **Deployment?** Will you package as installer or portable exe?

---

## Resources

**Flutter Documentation:**
- [Flutter Desktop Docs](https://docs.flutter.dev/platform-integration/desktop)
- [Riverpod Guide](https://riverpod.dev/)
- [graphview Package](https://pub.dev/packages/graphview)
- [just_audio Package](https://pub.dev/packages/just_audio)

**Learning Resources:**
- [Dart Language Tour](https://dart.dev/guides/language/language-tour)
- [Flutter Cookbook](https://docs.flutter.dev/cookbook)
- [Riverpod Quick Start](https://riverpod.dev/docs/getting_started)

**Community:**
- [Flutter Discord](https://discord.gg/flutter)
- [r/FlutterDev](https://reddit.com/r/FlutterDev)
- [Stack Overflow: flutter](https://stackoverflow.com/questions/tagged/flutter)

---

**Last Updated:** December 14, 2024
**Author:** Claude (via Claude Code)
**Status:** Ready for Implementation
