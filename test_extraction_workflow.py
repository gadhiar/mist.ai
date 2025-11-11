"""
Extraction Testing Workflow

Complete workflow for testing entity extraction:
1. Wipe database (optional)
2. Run extraction and store in Neo4j
3. Export and analyze results

Usage:
    python test_extraction_workflow.py
    python test_extraction_workflow.py --no-wipe  # Skip database wipe
"""

import asyncio
import sys
import uuid
from datetime import datetime
from rich.console import Console
from rich.panel import Panel

from backend.knowledge.config import get_config
from backend.knowledge.extraction import EntityExtractor
from backend.knowledge.storage import GraphStore, Neo4jConnection

console = Console()


# Sample conversation utterances
SAMPLE_UTTERANCES = [
    "I use Python for web development",
    "I prefer vim over VSCode for editing code",
    "I'm currently working on the MIST project",
    "I want to learn more about machine learning and neural networks",
    "I'm collaborating with Sarah on the GraphRAG implementation",
    "I dislike debugging JavaScript but I'm good at React",
]


def wipe_database(config):
    """Wipe all data from Neo4j"""
    console.print("\n[bold yellow]Step 1: Wiping database...[/bold yellow]")

    connection = Neo4jConnection(config.neo4j)

    try:
        connection.connect()
        connection.execute_write("MATCH (n) DETACH DELETE n")

        # Verify
        result = connection.execute_query("MATCH (n) RETURN count(n) AS count")
        node_count = result[0]["count"] if result else 0

        if node_count == 0:
            console.print("[green]Database wiped successfully[/green]\n")
        else:
            console.print(f"[yellow]Warning: {node_count} nodes still remain[/yellow]\n")

    finally:
        connection.disconnect()


async def run_extraction(config):
    """Run extraction and store in Neo4j"""
    console.print("[bold yellow]Step 2: Running extraction...[/bold yellow]\n")

    # Initialize components
    extractor = EntityExtractor(config)
    graph_store = GraphStore(config)
    graph_store.initialize_schema()

    # Create conversation
    conversation_id = str(uuid.uuid4())
    user_id = "test_user"

    graph_store.store_conversation_event(
        conversation_id=conversation_id,
        user_id=user_id,
        timestamp=datetime.now()
    )

    console.print(f"[dim]Conversation ID: {conversation_id[:8]}...[/dim]\n")

    # Process each utterance
    for i, utterance_text in enumerate(SAMPLE_UTTERANCES, 1):
        console.print(f"[cyan]{i}.[/cyan] {utterance_text}")

        # Store utterance
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
                graph_store.store_extracted_entities(
                    graph_document=graph_docs[0],
                    utterance_id=utterance_id
                )
                console.print(f"   [green]Stored: {len(graph_docs[0].nodes)} entities, {len(graph_docs[0].relationships)} relationships[/green]")
            else:
                console.print("   [yellow]No entities extracted[/yellow]")

        except Exception as e:
            console.print(f"   [red]Error: {e}[/red]")

    console.print()
    graph_store.close()

    return conversation_id


def analyze_extraction(config):
    """Export and display extraction results"""
    console.print("[bold yellow]Step 3: Analyzing extraction...[/bold yellow]\n")

    connection = Neo4jConnection(config.neo4j)

    try:
        connection.connect()

        # Get entities
        entity_query = """
        MATCH (e:__Entity__)
        RETURN e.id AS id, e.entity_type AS type
        ORDER BY e.id
        """
        entities = connection.execute_query(entity_query)

        console.print(f"[bold]Entities extracted:[/bold] {len(entities)}")
        for entity in entities:
            console.print(f"  - {entity['id']} ({entity['type']})")

        console.print()

        # Get relationships
        rel_query = """
        MATCH (source:__Entity__)-[r]->(target:__Entity__)
        RETURN source.id AS source, type(r) AS rel, target.id AS target
        ORDER BY source.id
        """
        relationships = connection.execute_query(rel_query)

        console.print(f"[bold]Relationships extracted:[/bold] {len(relationships)}")
        for rel in relationships:
            console.print(f"  - {rel['source']} -[{rel['rel']}]-> {rel['target']}")

        console.print()

        # Get provenance
        prov_query = """
        MATCH (u:Utterance)-[:HAS_ENTITY]->(e:__Entity__)
        WITH u.text AS utterance, collect(DISTINCT e.id) AS entities
        RETURN utterance, entities
        """
        provenance = connection.execute_query(prov_query)

        console.print("[bold]Provenance:[/bold]")
        for prov in provenance:
            entities_str = ", ".join(prov["entities"])
            console.print(f"  '{prov['utterance']}'")
            console.print(f"    -> {entities_str}")

        console.print()

    finally:
        connection.disconnect()


async def main():
    """Run complete extraction testing workflow"""

    console.print(Panel.fit(
        "[bold cyan]Extraction Testing Workflow[/bold cyan]\n\n"
        "This will:\n"
        "1. Wipe the Neo4j database (optional)\n"
        "2. Run entity extraction on sample utterances\n"
        "3. Display results for analysis",
        title="Workflow",
        border_style="cyan"
    ))

    # Check if --no-wipe flag is present
    skip_wipe = "--no-wipe" in sys.argv

    config = get_config()

    # Step 1: Wipe database (unless --no-wipe)
    if not skip_wipe:
        wipe_database(config)
    else:
        console.print("\n[yellow]Skipping database wipe (--no-wipe flag)[/yellow]\n")

    # Step 2: Run extraction
    conversation_id = await run_extraction(config)

    # Step 3: Analyze results
    analyze_extraction(config)

    # Summary
    console.print(Panel.fit(
        "[bold green]Workflow complete![/bold green]\n\n"
        "Review the extraction results above.\n\n"
        "Next steps:\n"
        "- Analyze if relationships are correct\n"
        "- Check if 'User' entity is properly extracted\n"
        "- Verify relationship directions\n\n"
        "To see visual graph: Open Neo4j Browser at http://localhost:7474\n"
        "Run: MATCH (n) RETURN n",
        title="Done",
        border_style="green"
    ))


if __name__ == "__main__":
    asyncio.run(main())
