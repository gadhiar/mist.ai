"""
Test Entity Extraction with LLMGraphTransformer

Tests LangChain's LLMGraphTransformer with our Qwen model to verify
it can extract entities and relationships from conversational text.

Usage:
    python test_entity_extraction.py
"""

import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama import ChatOllama

from backend.knowledge.config import get_config

console = Console()


# Sample conversational utterances to test
SAMPLE_CONVERSATIONS = [
    "I use Python for web development",
    "I prefer vim over VSCode for editing code",
    "I'm currently working on the MIST project",
    "I want to learn more about GraphRAG and knowledge graphs",
    "My tech stack includes FastAPI, React, and PostgreSQL"
]


async def test_extraction():
    """Test entity extraction with LLMGraphTransformer"""

    console.print("\n[bold cyan]Testing Entity Extraction with LLMGraphTransformer[/bold cyan]\n")

    # Load configuration
    config = get_config()

    # Step 1: Initialize LLM via Ollama (using MODEL env var)
    console.print("[yellow]Step 1:[/yellow] Initializing LLM model...")
    console.print(f"[dim]Model: {config.llm.model}[/dim]")
    console.print(f"[dim]Base URL: {config.llm.base_url}[/dim]\n")

    try:
        llm = ChatOllama(
            model=config.llm.model,
            base_url=config.llm.base_url,
            temperature=config.llm.temperature
        )
        console.print("[green]OK - LLM initialized[/green]\n")
    except Exception as e:
        console.print(f"[red]FAILED - Failed to initialize LLM: {e}[/red]")
        console.print("[yellow]Make sure Ollama is running and qwen2.5:32b is pulled[/yellow]")
        return

    # Step 2: Initialize LLMGraphTransformer
    console.print("[yellow]Step 2:[/yellow] Initializing LLMGraphTransformer...")

    additional_instructions = """
    Extract entities and relationships from this conversational statement.
    Important guidelines:
    - Treat "I", "me", "my" as references to the User entity
    - Do NOT extract dates, numbers, or quantities as separate entities - use them as properties
    - Focus on: tools, topics, preferences, tasks, people, and concepts mentioned by the user
    - Extract relationships that show how entities connect (USES, PREFERS, WORKS_ON, KNOWS_ABOUT, etc.)
    """

    try:
        transformer = LLMGraphTransformer(
            llm=llm,
            node_properties=["description"],
            relationship_properties=["description"],
            allowed_nodes=None,  # Dynamic - extract any entities
            allowed_relationships=None,  # Dynamic - extract any relationships
            additional_instructions=additional_instructions
        )
        console.print("[green]OK - Transformer initialized[/green]\n")
    except Exception as e:
        console.print(f"[red]FAILED - Failed to initialize transformer: {e}[/red]")
        return

    # Step 3: Extract from sample conversations
    console.print("[yellow]Step 3:[/yellow] Extracting entities from sample conversations...\n")

    for i, utterance in enumerate(SAMPLE_CONVERSATIONS, 1):
        console.print(f"[bold cyan]Conversation {i}:[/bold cyan] {utterance}")

        # Create document
        doc = Document(page_content=utterance, metadata={"utterance_id": i})

        try:
            # Extract entities and relationships
            graph_docs = await transformer.aconvert_to_graph_documents([doc])

            if not graph_docs:
                console.print("[yellow]No entities extracted[/yellow]\n")
                continue

            graph_doc = graph_docs[0]

            # Display extracted nodes
            if graph_doc.nodes:
                node_table = Table(title="Extracted Entities", show_header=True)
                node_table.add_column("ID", style="cyan")
                node_table.add_column("Type", style="magenta")
                node_table.add_column("Properties", style="green")

                for node in graph_doc.nodes:
                    node_id = node.id
                    node_type = node.type if hasattr(node, 'type') else "Unknown"
                    properties = str(getattr(node, 'properties', {}))[:50]
                    node_table.add_row(node_id, node_type, properties)

                console.print(node_table)

            # Display extracted relationships
            if graph_doc.relationships:
                rel_table = Table(title="Extracted Relationships", show_header=True)
                rel_table.add_column("Source", style="cyan")
                rel_table.add_column("Relationship", style="yellow")
                rel_table.add_column("Target", style="cyan")
                rel_table.add_column("Properties", style="green")

                for rel in graph_doc.relationships:
                    source = rel.source.id
                    rel_type = rel.type
                    target = rel.target.id
                    properties = str(getattr(rel, 'properties', {}))[:50]
                    rel_table.add_row(source, rel_type, target, properties)

                console.print(rel_table)

            console.print()

        except Exception as e:
            console.print(f"[red]FAILED - Extraction failed: {e}[/red]\n")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]\n")

    # Summary
    console.print(Panel.fit(
        "[bold green]OK - Entity Extraction Test Complete![/bold green]\n\n"
        "The LLMGraphTransformer successfully extracted entities and relationships.\n"
        "Next: Store these in Neo4j!",
        title="Test Results",
        border_style="green"
    ))


if __name__ == "__main__":
    asyncio.run(test_extraction())
