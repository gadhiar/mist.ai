# Frontend Setup

This document describes the scaffolding that has been created for the Mist.AI frontend.

## What's Been Set Up

### 1. Project Structure

```
frontend/
├── src/
│   ├── components/     # React components (empty, ready for implementation)
│   ├── hooks/         # Custom React hooks (empty, ready for implementation)
│   ├── types/         # TypeScript type definitions
│   │   ├── messages.ts   # WebSocket message types matching backend protocol
│   │   └── index.ts      # Type exports
│   ├── services/      # Service layer
│   │   └── websocket.ts  # WebSocket service with reconnection logic
│   ├── stores/        # Redux slices and store configuration (empty, ready for implementation)
│   ├── utils/         # Utility functions (empty, ready for implementation)
│   ├── App.tsx        # Main app component (Vite default, needs update)
│   ├── main.tsx       # React entry point
│   └── index.css      # Global styles with Tailwind directives
├── tailwind.config.js # Tailwind configuration with custom theme
├── postcss.config.js  # PostCSS configuration
├── vite.config.ts     # Vite configuration
├── tsconfig.json      # TypeScript configuration
└── package.json       # Frontend dependencies

Root:
└── .gitignore        # Updated with frontend artifacts
```

### 2. Configuration Files Created

- **tailwind.config.js**: Configured with custom color system (light/dark modes), animations
- **postcss.config.js**: Basic PostCSS setup for Tailwind
- **index.css**: Tailwind directives + CSS custom properties for theming

### 3. Type Definitions Created

**src/types/messages.ts** includes:
- All WebSocket message types matching backend protocol
- `BackendMessage` and `FrontendMessage` unions
- UI state enums (`ConversationState`, `AudioPlaybackState`)
- `ConversationTurn` interface for display

### 4. Services Created

**src/services/websocket.ts** includes:
- WebSocket connection management
- Auto-reconnection with exponential backoff
- Message type-based event handlers
- Type-safe send/receive methods

## Next Steps (NOT YET IMPLEMENTED)

### Phase 1: Core Infrastructure
1. Install additional dependencies:
   ```bash
   cd frontend
   npm install @reduxjs/toolkit react-redux framer-motion
   npm install -D @types/node
   ```

2. Create Redux store and slices:
   - `store.ts` - Redux store configuration
   - `slices/conversationSlice.ts` - Conversation history management
   - `slices/audioSlice.ts` - Audio playback state
   - `slices/connectionSlice.ts` - WebSocket connection state
   - `middleware/websocketMiddleware.ts` - Sync Redux with WebSocket

3. Create custom hooks:
   - `useWebSocket.ts` - WebSocket connection hook
   - `useAudioPlayback.ts` - Web Audio API integration

### Phase 2: Core Components
1. Update `App.tsx` to main layout
2. Create `ConversationDisplay` component
3. Create `AudioStatus` component
4. Create `ConnectionStatus` component

### Phase 3: Audio Integration
1. Implement Web Audio API playback
2. Handle base64 audio decoding
3. Implement audio buffering

### Phase 4: Testing & Polish
1. Test WebSocket connection to backend
2. Test audio playback
3. Add error boundaries
4. Polish UI/UX

## Running the Frontend

Once dependencies are installed:

```bash
cd frontend
npm run dev
```

The frontend will run on [http://localhost:5173](http://localhost:5173) by default.

## Backend Connection

The frontend will need to connect to the backend WebSocket server. You'll need to:

1. Update the WebSocket URL in your connection logic (currently hardcoded in service)
2. Ensure backend is running on the expected port (default: 8001)
3. Handle CORS if backend and frontend are on different origins

## Technology Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **Redux Toolkit** (to be installed) - Robust state management with DevTools, middleware support
- **React Redux** (to be installed) - React bindings for Redux
- **Framer Motion** (to be installed) - Animation library

## Development Guidelines

See README.md files in each subdirectory for component-specific guidelines:
- `src/components/README.md`
- `src/hooks/README.md`
- `src/stores/README.md`
- `src/utils/README.md`

## Architecture Notes

- Python backend handles all audio I/O (STT, TTS, LLM)
- Frontend is display-only (receives transcriptions and audio)
- WebSocket protocol matches backend implementation
- Type system ensures message format consistency
- Modular structure allows easy feature additions
