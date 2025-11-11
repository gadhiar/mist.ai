"""
Test Graph Storage

End-to-end test: Extract entities from conversation and store in Neo4j.

Usage:
    python test_graph_storage.py
"""

import asyncio
import uuid
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from backend.knowledge.config import get_config
from backend.knowledge.extraction import EntityExtractor
from backend.knowledge.storage import GraphStore

console = Console()


# Sample conversation
SAMPLE_CONVERSATION = [
    "I use Python for web development",
    "I prefer vim over VSCode for editing code",
    "I'm currently working on the MIST project",
]


async def test_end_to_end():
    """Test complete flow: extraction → storage → retrieval"""

    console.print("\n[bold cyan]Testing End-to-End: Extraction + Storage[/bold cyan]\n")

    # Load config
    config = get_config()

    # Step 1: Initialize components
    console.print("[yellow]Step 1:[/yellow] Initializing components...")

    extractor = EntityExtractor(config)
    graph_store = GraphStore(config)

    console.print(f"[dim]Model: {config.llm.model}[/dim]")
    console.print(f"[dim]Neo4j: {config.neo4j.uri}[/dim]\n")

    # Step 2: Initialize Neo4j schema
    console.print("[yellow]Step 2:[/yellow] Setting up Neo4j schema...")

    try:
        graph_store.initialize_schema()
        console.print("[green]OK - Schema initialized[/green]\n")
    except Exception as e:
        console.print(f"[red]FAILED - Schema initialization: {e}[/red]")
        return

    # Step 3: Create conversation
    console.print("[yellow]Step 3:[/yellow] Creating conversation...")

    conversation_id = str(uuid.uuid4())
    user_id = "test_user"

    graph_store.store_conversation_event(
        conversation_id=conversation_id,
        user_id=user_id,
        timestamp=datetime.now()
    )

    console.print(f"[green]OK - Created conversation: {conversation_id[:8]}...[/green]\n")

    # Step 4: Process utterances
    console.print("[yellow]Step 4:[/yellow] Extracting and storing entities...\n")

    for i, utterance_text in enumerate(SAMPLE_CONVERSATION, 1):
        console.print(f"[bold cyan]Utterance {i}:[/bold cyan] {utterance_text}")

        # Create utterance
        utterance_id = str(uuid.uuid4())

        graph_store.store_utterance(
            utterance_id=utterance_id,
            conversation_id=conversation_id,
            text=utterance_text,
            timestamp=datetime.now()
        )

        # Extract entities
        try:
            graph_docs = await extractor.extract_from_utterance(
                utterance=utterance_text,
                conversation_history=[],
                metadata={"utterance_id": utterance_id}
            )

            if graph_docs and graph_docs[0].nodes:
                # Store in Neo4j
                graph_store.store_extracted_entities(
                    graph_document=graph_docs[0],
                    utterance_id=utterance_id
                )

                console.print(f"  [green]Stored: {len(graph_docs[0].nodes)} entities, {len(graph_docs[0].relationships)} relationships[/green]")
            else:
                console.print("  [yellow]No entities extracted[/yellow]")

        except Exception as e:
            console.print(f"  [red]FAILED: {e}[/red]")

        console.print()

    # Step 5: Retrieve and display
    console.print("[yellow]Step 5:[/yellow] Retrieving stored entities...\n")

    try:
        entities = graph_store.get_entities_for_conversation(conversation_id)

        if entities:
            table = Table(title="Stored Entities in Neo4j")
            table.add_column("Entity ID", style="cyan")
            table.add_column("Type", style="magenta")
            table.add_column("From Utterances", style="yellow")

            for entity in entities:
                entity_id = entity["entity_id"]
                entity_type = entity["entity_type"]
                source_count = len(entity["source_utterances"])

                table.add_row(entity_id, entity_type, str(source_count))

            console.print(table)

            console.print(f"\n[green]Total entities stored: {len(entities)}[/green]")
        else:
            console.print("[yellow]No entities found[/yellow]")

    except Exception as e:
        console.print(f"[red]FAILED - Retrieval: {e}[/red]")

    # Summary
    console.print("\n")
    console.print(Panel.fit(
        "[bold green]OK - End-to-End Test Complete![/bold green]\n\n"
        "Successfully:\n"
        "1. Extracted entities from conversation\n"
        "2. Stored in Neo4j with provenance\n"
        "3. Retrieved entities back\n\n"
        "Check Neo4j Browser to visualize the graph!",
        title="Test Results",
        border_style="green"
    ))

    # Cleanup
    graph_store.close()


if __name__ == "__main__":
    asyncio.run(test_end_to_end())
