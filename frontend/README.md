# Mist.AI Frontend

React + TypeScript + Vite + Redux Toolkit frontend for the Mist.AI voice conversation system.

**Status:** Scaffolding complete, components in development

## Quick Start

```bash
# Install dependencies
npm install
npm install @reduxjs/toolkit react-redux framer-motion

# Run development server
npm run dev
```

Frontend will run on `http://localhost:5173`

## What's Implemented

✓ React 18 + Vite + TypeScript
✓ Tailwind CSS with custom theme
✓ Redux Toolkit state management (3 slices)
✓ WebSocket service
✓ Type definitions matching backend protocol
✓ Redux Provider wired up

**Not Yet Implemented:**
- UI components (ConversationDisplay, AudioStatus, etc.)
- Web Audio API playback
- Conversation history display

## Documentation

- [SETUP.md](SETUP.md) - Installation and setup guide
- [REDUX_IMPLEMENTATION.md](REDUX_IMPLEMENTATION.md) - State management architecture
- [src/stores/README.md](src/stores/README.md) - Redux store usage guide
- [Frontend Architecture](../docs/frontend/FRONTEND_ARCHITECTURE.md) - Complete technical spec

## Architecture

```
Frontend (Port 5173)
    ↓ WebSocket
Backend (Port 8001)
```

**State Management:**
- `conversationSlice` - Conversation history, streaming LLM responses
- `audioSlice` - Audio playback state, buffering, volume
- `connectionSlice` - WebSocket connection, VAD status

**Middleware:**
- `websocketMiddleware` - Automatically syncs backend messages → Redux actions

## Technology Stack

- **Framework:** React 18
- **Build Tool:** Vite 6
- **Language:** TypeScript (strict mode)
- **State:** Redux Toolkit
- **Styling:** Tailwind CSS
- **WebSocket:** Native WebSocket API
- **Animations:** Framer Motion (to be added)

## Project Structure

```
frontend/
├── src/
│   ├── stores/            # Redux slices & middleware
│   ├── services/          # WebSocket service
│   ├── components/        # React components (empty)
│   ├── hooks/             # Custom hooks (empty)
│   ├── types/             # TypeScript types
│   └── utils/             # Utility functions (empty)
├── SETUP.md               # Setup guide
├── REDUX_IMPLEMENTATION.md # Redux details
└── README.md              # This file
```

## Development

```bash
# Install dependencies
npm install

# Run dev server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint
```

## Next Steps

1. Install remaining dependencies (@reduxjs/toolkit, react-redux, framer-motion)
2. Create ConversationDisplay component
3. Create AudioStatus component
4. Create ConnectionStatus component
5. Implement Web Audio API playback
6. Test WebSocket connection to backend

## Backend Connection

The frontend connects to the backend WebSocket server at `ws://localhost:8001`. Ensure the backend is running before starting the frontend.

**Start backend:**
```bash
cd ../backend
python server.py
```

## Contributing

See [SETUP.md](SETUP.md) for detailed setup instructions and [Frontend Architecture](../docs/frontend/FRONTEND_ARCHITECTURE.md) for technical specifications.
