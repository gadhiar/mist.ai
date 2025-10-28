import { useEffect, useRef } from 'react'
import { useAppSelector } from '../stores'
import { ConversationState } from '../types'

export default function ConversationDisplay() {
  const turns = useAppSelector(state => state.conversation.turns)
  const conversationState = useAppSelector(state => state.conversation.state)
  const isStreaming = useAppSelector(state => state.conversation.isStreaming)
  const currentStreamingText = useAppSelector(state => state.conversation.currentStreamingText)

  const scrollRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [turns, currentStreamingText])

  const getStateIndicator = () => {
    switch (conversationState) {
      case ConversationState.LISTENING:
        return (
          <div className="flex items-center gap-2 text-blue-500">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
            <span>Listening...</span>
          </div>
        )
      case ConversationState.PROCESSING:
        return (
          <div className="flex items-center gap-2 text-yellow-500">
            <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse" />
            <span>Processing...</span>
          </div>
        )
      case ConversationState.SPEAKING:
        return (
          <div className="flex items-center gap-2 text-green-500">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            <span>Speaking...</span>
          </div>
        )
      default:
        return null
    }
  }

  return (
    <div className="flex flex-col h-[calc(100vh-200px)]">
      {/* State Indicator */}
      <div className="mb-4 text-sm">
        {getStateIndicator()}
      </div>

      {/* Conversation History */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto space-y-4 p-4 bg-card rounded-lg border border-border"
      >
        {turns.length === 0 && !isStreaming && (
          <p className="text-center text-muted-foreground py-8">
            No conversation yet. Start speaking using the CLI client.
          </p>
        )}

        {turns.map((turn) => (
          <div
            key={turn.id}
            className={`flex ${turn.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] rounded-lg p-3 ${
                turn.role === 'user'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-secondary text-secondary-foreground'
              }`}
            >
              <div className="text-xs opacity-70 mb-1">
                {turn.role === 'user' ? 'You' : 'Mist.AI'}
              </div>
              <div className="whitespace-pre-wrap">{turn.text}</div>
              {!turn.isComplete && (
                <div className="text-xs opacity-70 mt-1 italic">
                  (streaming...)
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Show streaming text if assistant is currently streaming */}
        {isStreaming && currentStreamingText && (
          <div className="flex justify-start">
            <div className="max-w-[80%] rounded-lg p-3 bg-secondary text-secondary-foreground">
              <div className="text-xs opacity-70 mb-1">Mist.AI</div>
              <div className="whitespace-pre-wrap">{currentStreamingText}</div>
              <div className="inline-block w-1 h-4 bg-current animate-pulse ml-1" />
            </div>
          </div>
        )}
      </div>

      {/* Info Footer */}
      <div className="mt-4 text-xs text-muted-foreground text-center">
        Total messages: {turns.length}
      </div>
    </div>
  )
}
