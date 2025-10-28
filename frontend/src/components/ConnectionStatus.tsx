import { useAppSelector } from '../stores'
import { ConnectionStatus as Status } from '../stores'

export default function ConnectionStatus() {
  const status = useAppSelector(state => state.connection.status)
  const clientId = useAppSelector(state => state.connection.clientId)
  const vadStatus = useAppSelector(state => state.connection.vadStatus)

  const getStatusColor = () => {
    switch (status) {
      case Status.CONNECTED:
        return 'bg-green-500'
      case Status.CONNECTING:
      case Status.RECONNECTING:
        return 'bg-yellow-500 animate-pulse'
      case Status.ERROR:
        return 'bg-red-500'
      case Status.DISCONNECTED:
      default:
        return 'bg-gray-500'
    }
  }

  const getStatusText = () => {
    switch (status) {
      case Status.CONNECTED:
        return 'Connected'
      case Status.CONNECTING:
        return 'Connecting...'
      case Status.RECONNECTING:
        return 'Reconnecting...'
      case Status.ERROR:
        return 'Error'
      case Status.DISCONNECTED:
      default:
        return 'Disconnected'
    }
  }

  const getVADText = () => {
    if (status !== Status.CONNECTED) return null

    switch (vadStatus) {
      case 'speech_started':
        return '🎤 Listening'
      case 'processing':
        return '⚙️ Processing'
      default:
        return null
    }
  }

  return (
    <div className="flex items-center gap-3">
      {/* VAD Status */}
      {getVADText() && (
        <span className="text-sm text-muted-foreground">
          {getVADText()}
        </span>
      )}

      {/* Connection Status */}
      <div className="flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${getStatusColor()}`} />
        <span className="text-sm">
          {getStatusText()}
        </span>
      </div>

      {/* Client ID (when connected) */}
      {clientId && (
        <span className="text-xs text-muted-foreground font-mono">
          {clientId.slice(0, 8)}
        </span>
      )}
    </div>
  )
}
