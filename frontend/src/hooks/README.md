# Hooks

Custom React hooks for the Mist.AI frontend.

## Current Architecture

**Note:** The frontend currently uses **Redux Toolkit** for state management instead of custom hooks. This provides centralized state with Redux DevTools support.

### Redux Hooks (from `@reduxjs/toolkit`)

#### `useAppDispatch()`
Type-safe dispatch hook for Redux actions.

**Usage:**
```typescript
const dispatch = useAppDispatch()
dispatch(connectWebSocket())
```

#### `useAppSelector()`
Type-safe selector hook for reading Redux state.

**Usage:**
```typescript
const turns = useAppSelector(state => state.conversation.turns)
const status = useAppSelector(state => state.connection.status)
```

**Available State Slices:**
- `conversation` - Conversation turns, streaming state, conversation state
- `connection` - WebSocket status, VAD status, client ID, latency
- `audio` - Audio playback queue, buffer state, volume (planned for Web Audio API)

### Future Custom Hooks

When browser-based features are added, these hooks may be implemented:

- `useAudioRecording` - Browser microphone access (MediaRecorder API)
- `useWebAudioPlayback` - Web Audio API playback control
- `useKeyboardShortcuts` - Hotkeys for mute, pause, etc.
- `useSpeechRecognition` - Browser Web Speech API (alternative to Python VAD)

## State Management Architecture

**Current:** Redux Toolkit with WebSocket middleware
- Centralized state in [stores/](../stores/)
- WebSocket integration via [middleware/websocketMiddleware.ts](../stores/middleware/websocketMiddleware.ts)
- Type-safe actions and reducers

**Why Redux over Custom Hooks:**
- WebSocket state needs to be shared across multiple components
- Redux DevTools for debugging message flow
- Time-travel debugging for conversation replay
- Middleware for WebSocket lifecycle management

## Hook Guidelines

- Follow React hooks rules- Keep hooks focused on single responsibility
- Use TypeScript for type safety- Document parameters and return values
