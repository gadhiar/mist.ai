# Redux Store Architecture

Redux Toolkit-based state management for Mist.AI frontend.

## Structure

```
stores/
├── store.ts                 # Redux store configuration
├── index.ts                 # Centralized exports
├── slices/
│   ├── conversationSlice.ts # Conversation history and streaming
│   ├── audioSlice.ts        # Audio playback and buffering
│   └── connectionSlice.ts   # WebSocket connection and VAD status
└── middleware/
    └── websocketMiddleware.ts # Syncs Redux with WebSocket messages
```

## Slices

### conversationSlice.ts

**Manages:** Conversation history, streaming LLM responses, user input tracking

**State:**
```typescript
{
  turns: ConversationTurn[];           // All conversation turns
  state: ConversationState;            // Current conversation state
  isStreaming: boolean;                // Is LLM currently streaming?
  currentStreamingText: string;        // Current streaming text
  currentStreamingTurnId: string | null;
  lastUserInput: string | null;
  lastUserInputTimestamp: number | null;
}
```

**Key Actions:**
- `addUserTranscription` - Add user speech transcription
- `startAssistantResponse` - Begin streaming LLM response
- `appendResponseChunk` - Append chunk to streaming response
- `completeAssistantResponse` - Mark response complete
- `setConversationState` - Update conversation state (idle/listening/processing/speaking)
- `clearConversation` - Clear all history
- `removeLastTurn` - Remove last turn (for interruptions)

### audioSlice.ts

**Manages:** Audio playback state, audio queue, buffering, volume

**State:**
```typescript
{
  playbackState: AudioPlaybackState;   // idle/buffering/playing/paused
  audioQueue: AudioChunk[];            // Queued audio chunks
  currentChunkIndex: number;
  isBuffering: boolean;
  bufferProgress: number;              // 0-100
  volume: number;                      // 0-1
  isMuted: boolean;
  isPlaying: boolean;
  currentAudioDuration: number;
  currentAudioPosition: number;
}
```

**Key Actions:**
- `addAudioChunk` - Add audio chunk to queue
- `clearAudioQueue` - Clear queue (for interruptions)
- `setPlaybackState` - Update playback state
- `startBuffering` / `finishBuffering` - Manage buffering
- `play` / `pause` / `stop` - Playback controls
- `setVolume` / `toggleMute` - Volume controls
- `audioComplete` - Mark audio generation complete

### connectionSlice.ts

**Manages:** WebSocket connection status, VAD status, network metrics

**State:**
```typescript
{
  status: ConnectionStatus;            // disconnected/connecting/connected/reconnecting/error
  clientId: string | null;
  connectedAt: number | null;
  disconnectedAt: number | null;
  reconnectAttempts: number;
  lastError: string | null;
  vadStatus: VADStatus;                // idle/speech_started/speech_ended/processing
  vadTimestamp: number | null;
  latency: number | null;
  messageCount: number;
}
```

**Key Actions:**
- `connecting` / `connected` / `disconnected` - Connection lifecycle
- `reconnecting` - Increment reconnection attempts
- `connectionError` - Handle connection errors
- `setVADStatus` - Update VAD status
- `updateLatency` - Track network latency
- `incrementMessageCount` - Track message throughput

## Middleware

### websocketMiddleware.ts

**Purpose:** Syncs WebSocket messages with Redux state

**How it works:**
1. Listens to all WebSocket messages via `WebSocketService`
2. Dispatches appropriate Redux actions based on message type
3. Handles connection lifecycle events
4. Manages conversation state transitions
5. Coordinates audio playback with conversation state

**Message Routing:**
- `connection` → `connected()`
- `transcription` → `addUserTranscription()`, `setConversationState(PROCESSING)`
- `llm_response_chunk` → `startAssistantResponse()`, `appendResponseChunk()`
- `llm_response_complete` → `completeAssistantResponse()`
- `audio_chunk` → `addAudioChunk()`, `startBuffering()` (if first chunk)
- `audio_complete` → `audioComplete()`, `setConversationState(IDLE)`
- `vad_status` → `setVADStatus()`, state transitions, `clearAudioQueue()` (on speech_started)
- `error` → `connectionError()`

## Usage

### Setting up the Provider

```typescript
import { Provider } from 'react-redux';
import { store } from './stores';

function App() {
  return (
    <Provider store={store}>
      {/* Your app */}
    </Provider>
  );
}
```

### Using in Components

```typescript
import { useAppSelector, useAppDispatch } from './stores';
import { clearConversation, setVolume } from './stores';

function MyComponent() {
  // Select state
  const messages = useAppSelector(state => state.conversation.turns);
  const isPlaying = useAppSelector(state => state.audio.isPlaying);
  const connectionStatus = useAppSelector(state => state.connection.status);

  // Dispatch actions
  const dispatch = useAppDispatch();

  const handleClear = () => {
    dispatch(clearConversation());
  };

  const handleVolumeChange = (volume: number) => {
    dispatch(setVolume(volume));
  };

  // ...
}
```

### Connecting to WebSocket

```typescript
import { connectWebSocket, disconnectWebSocket } from './stores';

// Connect
dispatch(connectWebSocket());

// Disconnect
dispatch(disconnectWebSocket());
```

## Redux DevTools

The store is configured with Redux DevTools support in development mode. Use the browser extension to:
- Time-travel debug
- Inspect state changes
- Replay actions
- Export/import state

## Type Safety

All state, actions, and selectors are fully typed with TypeScript. Use the provided hooks:
- `useAppSelector` - Typed `useSelector`
- `useAppDispatch` - Typed `useDispatch`

## Performance Optimization

### Selector Best Practices

```typescript
// Bad - Creates new array on every render
const messages = useAppSelector(state => state.conversation.turns.filter(t => t.role === 'user'));

// Good - Use memoized selectors (create with createSelector from reselect)
import { createSelector } from '@reduxjs/toolkit';

const selectUserMessages = createSelector(
  (state: RootState) => state.conversation.turns,
  (turns) => turns.filter(t => t.role === 'user')
);

const userMessages = useAppSelector(selectUserMessages);
```

### Avoid Unnecessary Re-renders

```typescript
// Bad - Selects entire state object
const conversation = useAppSelector(state => state.conversation);

// Good - Select only what you need
const isStreaming = useAppSelector(state => state.conversation.isStreaming);
const turns = useAppSelector(state => state.conversation.turns);
```

## Future Enhancements

- **Persistence:** Add redux-persist for conversation history
- **Undo/Redo:** Leverage Redux for undo/redo functionality
- **Offline Support:** Queue actions when disconnected
- **Analytics:** Track user interactions via middleware
- **Error Boundaries:** Dispatch errors to Redux for centralized handling
