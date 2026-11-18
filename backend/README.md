# Backend

The FastAPI and Uvicorn server MIST's frontend talks to over one WebSocket, `ws://<host>:8001/ws`,
plus a few HTTP routes (`/health`, `/extraction/status`, `/capabilities`, `/usage/...`). It runs in
the `mist-backend` container; see the repository's CLAUDE.md for the stack.

## Architecture

```
WebSocket /ws (backend/server.py, websocket_endpoint)
  -> SessionRegistry (realtime/registry.py): one RealtimeSession per connection
     -> RealtimeSession (realtime/session.py): binary frames to the audio ingress,
        text to control handling; drains its `control`, `turn` and `bulk` lanes
        -> AudioIngress (realtime/ingress.py): VAD, speech-to-text, end of turn, name hits
        -> TurnController (realtime/turn.py): one mailbox task owns the session's turns
           -> responder: ConversationHandler (chat/conversation_handler.py), or
              DirectResponder (realtime/degraded.py) when knowledge integration is off
           -> segmenter (realtime/segmenter.py) -> TTS engine -> playback (realtime/playback.py)
```

A turn's frames go to its own session's socket only. Process-wide events (heartbeats,
`system_status`, `health_status`, `extraction_status`, log streaming) go through `message_queue` and
the broadcaster in `server.py`; `backend/realtime` never touches that queue.

`lifespan` in `server.py` builds everything once at startup through `build_mist_runtime`
(`factories.py`): the LLM provider, the conversation handler, and the speech engines, each slot
loaded from its chain by `load_speech_engines` (`realtime/runtime.py`). Model calls run on
inference lanes (`realtime/lanes.py`), one worker thread per engine, so they never block the event
loop.

## Package map

- `server.py` - the app, `lifespan`, the WebSocket endpoint and the HTTP routes.
- `factories.py` - the composition root: every real dependency is built here and injected.
- `realtime/` - the voice path: sessions, the turn controller, ingress, playback, barge-in, wake by
  name, power levels and the speech engine adapters in `realtime/engines/`. Each module's docstring
  says what it owns; `REPOSITORY_STRUCTURE.md` lists them.
- `chat/` - the conversation handler: the pass loop, persona, tools and the stream events a turn
  yields (`chat/stream_events.py`).
- `knowledge/` - extraction, retrieval, curation and the Neo4j graph store.
- `vault/` - the markdown vault writer, sidecar index and filewatcher.
- `llm/` - the LLM providers.
- `extraction_backlog/` - the extraction dispatcher and its admin CLI.
- `capabilities/` - tools that act on the world, with proposals that wait for the user's yes.

## Protocol

The WebSocket contract is the realtime protocol, version 2.0.0, defined in
`realtime/protocol.py`: the binary audio frame codec and every control message schema. The first
frame on a connection is `session_started`, carrying the server-minted `session_id`,
`protocol_version` and `capabilities.voice`. A turn opens with `stream_start{turn_id, channel,
origin}` (`audio_tag` only for a spoken turn), streams `stream_token`, ends its stream with
`stream_complete` or `stream_cancelled`, and closes with `turn_metrics` after the turn is
committed. Frontend coordination is recorded in ADR-016 and ADR-017.

## Configuration

Every setting comes from the environment and is documented in `.env.example`. The voice settings
(`MIST_VOICE_*`) are read once at startup by `load_voice_settings` (`realtime/budget.py`); an
invalid value stops the backend with a message naming the variable. `MIST_VOICE_STT_CHAIN` and
`MIST_VOICE_TTS_CHAIN` choose the speech engines, and `off` turns a slot off. `TTS_ENABLED=false`
keeps replies unspoken. `ENABLE_KNOWLEDGE_INTEGRATION=false` answers every turn through
`DirectResponder`, which records nothing.

## Running

```bash
docker compose up -d mist-backend
```

## Testing

Tests run in the backend container (see `TESTING.md`). The voice path's unit tests are in
`tests/unit/realtime/`; they drive the session and turn controller with fake engines
(`tests/unit/realtime/turn_fakes.py`, `test_speech_fakes.py`), so they need no GPU and no model
weights.

```bash
MSYS_NO_PATHCONV=1 docker compose exec -T mist-backend python -m pytest tests/unit/realtime -q
```
