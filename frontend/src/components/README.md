# Components

React components for the Mist.AI frontend.

## Current Structure

### Implemented Components

#### `ConversationDisplay.tsx`
**Status:** [COMPLETE]

Main conversation display with real-time streaming support.

**Features:**
- Real-time conversation display (user turns + AI responses)
- Streaming text support with cursor animation
- Auto-scroll to bottom on new messages
- State indicators (Listening, Processing, Speaking)
- Redux integration via `useAppSelector`

**Used by:** [App.tsx](../App.tsx:28)

**State Dependencies:**
- `conversation.turns` - Array of conversation turns
- `conversation.state` - Current conversation state (LISTENING, PROCESSING, SPEAKING)
- `conversation.isStreaming` - Whether AI is currently streaming
- `conversation.currentStreamingText` - Text being streamed

#### `ConnectionStatus.tsx`
**Status:** [COMPLETE]

WebSocket connection status indicator with VAD feedback.

**Features:**
- Connection state visualization (Connected, Connecting, Reconnecting, Error, Disconnected)
- Color-coded status indicators
- VAD status display (Listening, Processing indicators)
- Client ID display (first 8 chars)

**Used by:** [App.tsx](../App.tsx:22)

**State Dependencies:**
- `connection.status` - WebSocket connection status
- `connection.clientId` - Unique client identifier
- `connection.vadStatus` - Voice Activity Detection status

### Planned Components

- `AudioStatus/` - Dedicated audio playback visualization
- `RichContent/` - Display images, links, markdown from AI
- `Controls/` - Manual controls (mute, pause, settings)

## Component Guidelines

- Use TypeScript strict mode- Use Tailwind CSS for styling- Keep components small and focused- Extract reusable logic into hooks