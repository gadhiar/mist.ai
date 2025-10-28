import { useEffect } from 'react'
import { useAppDispatch, useAppSelector, connectWebSocket } from './stores'
import ConversationDisplay from './components/ConversationDisplay'
import ConnectionStatus from './components/ConnectionStatus'

function App() {
  const dispatch = useAppDispatch()
  const connectionStatus = useAppSelector(state => state.connection.status)

  // Connect to WebSocket on mount
  useEffect(() => {
    dispatch(connectWebSocket())
  }, [dispatch])

  return (
    <>
    <div className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <header className="border-b border-border p-4">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <h1 className="text-2xl font-bold">Mist.AI</h1>
          <ConnectionStatus />
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto p-4">
        {connectionStatus === 'connected' ? (
          <ConversationDisplay />
        ) : (
          <div className="flex items-center justify-center h-96">
            <p className="text-muted-foreground">
              {connectionStatus === 'connecting' && 'Connecting to backend...'}
              {connectionStatus === 'disconnected' && 'Disconnected from backend'}
              {connectionStatus === 'reconnecting' && 'Reconnecting...'}
              {connectionStatus === 'error' && 'Connection error - check if backend is running'}
            </p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="fixed bottom-0 left-0 right-0 border-t border-border bg-background p-4">
        <div className="max-w-4xl mx-auto text-center text-sm text-muted-foreground">
          Use the Python CLI client (cli_client/voice_client.py) for voice conversation
        </div>
      </footer>
    </div>
  </>
  )
}

export default App
