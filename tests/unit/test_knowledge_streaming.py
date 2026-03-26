"""Tests for true token streaming in KnowledgeIntegration."""

from unittest.mock import MagicMock, patch


class TestKnowledgeStreamingTokens:
    """Test that generate_tokens_streaming yields individual tokens."""

    def test_yields_multiple_tokens(self):
        """Should yield individual tokens, not a single blob."""
        from backend.chat.knowledge_integration import KnowledgeIntegration

        ki = KnowledgeIntegration.__new__(KnowledgeIntegration)
        ki.enabled = True
        ki.current_session_id = "test"

        # Mock conversation handler with realistic attributes
        handler = MagicMock()
        handler.config.auto_inject_docs = False  # Skip RAG for this test
        handler.config.llm.model = "qwen2.5:7b-instruct"
        session = MagicMock()
        session.messages = []
        handler.get_or_create_session.return_value = session
        handler.event_store = None
        ki.conversation_handler = handler

        mock_chunks = [
            {"message": {"content": "Hello"}},
            {"message": {"content": " there"}},
            {"message": {"content": "."}},
        ]

        with patch("backend.chat.knowledge_integration.ollama") as mock_ollama:
            mock_ollama.chat.return_value = iter(mock_chunks)

            tokens = list(
                ki.generate_tokens_streaming(
                    "hello how are you doing today", event_loop=MagicMock()
                )
            )

        assert len(tokens) == 3
        assert tokens[0] == "Hello"
        assert tokens[1] == " there"
        assert tokens[2] == "."
        session.add_message.assert_any_call("user", "hello how are you doing today")
        session.add_message.assert_any_call("assistant", "Hello there.")

    def test_rag_context_included_when_available(self):
        """RAG results should be included in messages sent to Ollama."""
        from backend.chat.knowledge_integration import KnowledgeIntegration

        ki = KnowledgeIntegration.__new__(KnowledgeIntegration)
        ki.enabled = True
        ki.current_session_id = "test"

        handler = MagicMock()
        handler.config.auto_inject_docs = True
        handler.config.auto_inject_limit = 3
        handler.config.auto_inject_threshold = 0.4
        handler.config.llm.model = "qwen2.5:7b-instruct"
        session = MagicMock()
        session.messages = []
        handler.get_or_create_session.return_value = session
        handler.event_store = None

        retrieval_result = MagicMock()
        retrieval_result.total_facts = 2
        retrieval_result.formatted_context = "Known facts: User likes Python."
        handler.retriever.retrieve.return_value = retrieval_result
        ki.conversation_handler = handler

        mock_loop = MagicMock()

        mock_chunks = [{"message": {"content": "Sure!"}}]

        with (
            patch("backend.chat.knowledge_integration.ollama") as mock_ollama,
            patch("asyncio.run_coroutine_threadsafe") as mock_rcts,
        ):
            # Make the future return the retrieval result
            mock_future = MagicMock()
            mock_future.result.return_value = retrieval_result
            mock_rcts.return_value = mock_future

            mock_ollama.chat.return_value = iter(mock_chunks)

            tokens = list(
                ki.generate_tokens_streaming(
                    "what programming languages do I know",
                    event_loop=mock_loop,
                )
            )

        assert tokens == ["Sure!"]
        # Verify RAG context was included in Ollama call
        call_args = mock_ollama.chat.call_args
        messages = call_args.kwargs["messages"]
        context_messages = [m for m in messages if "Known facts" in m.get("content", "")]
        assert len(context_messages) == 1
