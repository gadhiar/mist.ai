"""
Test Conversation Handler with Knowledge Graph Integration

Tests the MCP-like autonomous tool use for query and extraction.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

from backend.chat.conversation_handler import ConversationHandler
from backend.knowledge.storage.graph_store import GraphStore
from backend.knowledge.config import ExtractionConfig


async def test_conversation_handler():
    """Test the conversation handler with sample interactions"""

    print("=" * 70)
    print("CONVERSATION HANDLER TEST")
    print("=" * 70)
    print()

    # Configuration
    print("1. Initializing knowledge graph connection...")
    try:
        graph_store = GraphStore(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="your_password"
        )
        print("   ✅ Connected to Neo4j")
    except Exception as e:
        print(f"   ❌ Failed to connect to Neo4j: {e}")
        print("   Make sure Neo4j is running on bolt://localhost:7687")
        return

    # Initialize config
    config = ExtractionConfig(
        model_name="qwen2.5:7b",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password"
    )

    # Initialize conversation handler
    print("2. Initializing conversation handler...")
    handler = ConversationHandler(
        config=config,
        graph_store=graph_store,
        model_name="qwen2.5:7b"
    )
    print("   ✅ Conversation handler ready")
    print()

    # Test scenarios
    scenarios = [
        {
            "name": "Learning Phase",
            "messages": [
                "Hi! I'm Raj and I love working with Python and FastAPI.",
                "I've been working on a project called MIST for about 3 months now.",
                "I'm also learning Rust for systems programming."
            ]
        },
        {
            "name": "Query Phase",
            "messages": [
                "What programming languages do I know?",
                "Tell me about my projects.",
                "What technologies am I learning?"
            ]
        },
        {
            "name": "Mixed Interaction",
            "messages": [
                "I just started learning Neo4j for graph databases.",
                "What backend technologies do I use?"
            ]
        }
    ]

    session_id = "test-session-1"

    for scenario in scenarios:
        print("=" * 70)
        print(f"SCENARIO: {scenario['name']}")
        print("=" * 70)
        print()

        for i, user_message in enumerate(scenario['messages'], 1):
            print(f"[{i}] USER: {user_message}")
            print()

            # Process message
            try:
                response = await handler.handle_message(
                    user_message=user_message,
                    session_id=session_id
                )

                print(f"    MIST: {response}")
                print()

                # Check session info
                session_info = handler.get_session_info(session_id)
                if session_info:
                    print(f"    📊 Session: {session_info['message_count']} messages")
                    print()

            except Exception as e:
                print(f"    ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                print()

        print()

    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print()
    print("Session Summary:")
    session_info = handler.get_session_info(session_id)
    if session_info:
        print(f"  Session ID: {session_info['session_id']}")
        print(f"  Total messages: {session_info['message_count']}")
        print(f"  Started: {session_info['started_at']}")
    print()


async def test_simple_interaction():
    """Simple single interaction test"""

    print("=" * 70)
    print("SIMPLE INTERACTION TEST")
    print("=" * 70)
    print()

    try:
        # Quick setup
        graph_store = GraphStore(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="your_password"
        )

        config = ExtractionConfig(
            model_name="qwen2.5:7b",
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="your_password"
        )

        handler = ConversationHandler(config, graph_store, "qwen2.5:7b")

        # Test message
        user_msg = "I use Python and FastAPI for backend development."
        print(f"USER: {user_msg}")
        print()

        response = await handler.handle_message(user_msg, "simple-test")

        print(f"MIST: {response}")
        print()
        print("✅ Test passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test conversation handler")
    parser.add_argument(
        "--mode",
        choices=["simple", "full"],
        default="simple",
        help="Test mode: simple or full scenarios"
    )

    args = parser.parse_args()

    if args.mode == "simple":
        asyncio.run(test_simple_interaction())
    else:
        asyncio.run(test_conversation_handler())
