# Utils

Utility functions and services for the Mist.AI frontend.

## Current Architecture

**Note:** The frontend uses a **services pattern** rather than pure utility functions. Services are located in [../services/](../services/).

### Implemented Services

#### `services/websocket.ts`
**Status:** ✅ Complete

WebSocket service with automatic reconnection and message routing.

**Features:**
- Connection management with auto-reconnect (max 5 attempts)
- Message handler registration by type
- Exponential backoff for reconnections
- Type-safe message handling

**Used by:** [stores/middleware/websocketMiddleware.ts](../stores/middleware/websocketMiddleware.ts)

#### `services/audioPlayback.ts`
**Status:** ⏸️ Implemented but not connected

Web Audio API service for browser-based audio playback (prepared for future Web Audio integration).

**Features:**
- Web Audio API context management
- Audio chunk queueing
- Volume and mute controls
- Playback callbacks
- Base64 and float array audio support

**Note:** Currently audio is played via Python CLI client. This service is ready for when browser playback is implemented.

### Implemented Types

#### `types/index.ts` & `types/messages.ts`
**Status:** ✅ Complete

TypeScript type definitions for:
- `BackendMessage` - Messages from backend (text chunks, audio chunks, VAD status)
- `FrontendMessage` - Messages to backend (text, audio, commands)
- `ConversationState` - Enum for conversation state (LISTENING, PROCESSING, SPEAKING)
- `ConnectionStatus` - Enum for WebSocket connection state
- `ConversationTurn` - Individual conversation message structure

## Future Utils/Services

When additional features are added:

- `formatters.ts` - Timestamp formatting, duration display, text truncation
- `constants.ts` - App-wide constants (WebSocket URL, audio settings, theme colors)
- `audioUtils.ts` - Audio format conversion helpers (Float32Array ↔ base64)
- `storageUtils.ts` - LocalStorage helpers for saving conversation history

## Guidelines

- Use services for stateful functionality ✅
- Use pure util functions for stateless transformations
- Document complex logic ✅
- Add unit tests for critical utilities
- Use TypeScript for type safety ✅
