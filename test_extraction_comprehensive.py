"""
Comprehensive Extraction Testing

Tests edge cases and complex scenarios to identify extraction weaknesses.

Usage:
    python test_extraction_comprehensive.py
"""

import asyncio
import uuid
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from backend.knowledge.config import get_config
from backend.knowledge.extraction import EntityExtractor
from backend.knowledge.storage import GraphStore, Neo4jConnection

console = Console()


# Comprehensive test cases covering edge cases
TEST_CASES = [
    # Basic cases (baseline)
    {
        "category": "Basic Actions",
        "utterances": [
            "I use Python for web development",
            "I'm learning Rust for systems programming",
            "I work with Docker and Kubernetes daily",
        ]
    },

    # Complex relationships
    {
        "category": "Multi-entity Relationships",
        "utterances": [
            "I use FastAPI with PostgreSQL for building REST APIs",
            "I'm migrating from MongoDB to Neo4j for better graph queries",
            "Sarah taught me React, and now I'm teaching it to John",
        ]
    },

    # Temporal and context
    {
        "category": "Temporal & Context",
        "utterances": [
            "I used to use Java but now I prefer Python",
            "I've been working on MIST for 3 months and it's going well",
            "I plan to learn TypeScript next month",
        ]
    },

    # Preferences and comparisons
    {
        "category": "Preferences & Comparisons",
        "utterances": [
            "I prefer functional programming over object-oriented programming",
            "I think GraphQL is better than REST for complex queries",
            "I love Python but hate its packaging system",
        ]
    },

    # Skills and proficiency
    {
        "category": "Skills & Proficiency",
        "utterances": [
            "I'm an expert in Python but a beginner in Rust",
            "I can write SQL queries but I struggle with query optimization",
            "I'm proficient in React and Vue but not Angular",
        ]
    },

    # Problems and goals
    {
        "category": "Problems & Goals",
        "utterances": [
            "I'm having trouble understanding async/await in JavaScript",
            "My goal is to master GraphRAG by end of year",
            "I need to improve my understanding of vector databases",
        ]
    },

    # Tools and workflows
    {
        "category": "Tools & Workflows",
        "utterances": [
            "I use VS Code for Python and IntelliJ for Java",
            "I debug with print statements instead of using a debugger",
            "My workflow involves Git, Docker, and CI/CD pipelines",
        ]
    },

    # Negations and edge cases
    {
        "category": "Negations & Edge Cases",
        "utterances": [
            "I don't use TypeScript even though my team does",
            "I've never worked with Kubernetes in production",
            "I avoid using global variables in my code",
        ]
    },
]


def wipe_database(config):
    """Wipe Neo4j database"""
    connection = Neo4jConnection(config.neo4j)
    try:
        connection.connect()
        connection.execute_write("MATCH (n) DETACH DELETE n")
    finally:
        connection.disconnect()


async def test_category(category_name, utterances, extractor, graph_store, conversation_id):
    """Test a category of utterances"""

    console.print(f"\n[bold cyan]Category: {category_name}[/bold cyan]")
    console.print("=" * 60)

    results = []

    for i, utterance_text in enumerate(utterances, 1):
        console.print(f"\n[yellow]{i}.[/yellow] {utterance_text}")

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
                metadata={"utterance_id": utterance_id, "category": category_name}
            )

            if graph_docs and graph_docs[0].nodes:
                entities = [f"{n.id} ({n.type})" for n in graph_docs[0].nodes]
                relationships = [f"{r.source.id} -[{r.type}]-> {r.target.id}"
                               for r in graph_docs[0].relationships]

                console.print(f"   [green]Entities:[/green] {', '.join(entities)}")
                console.print(f"   [green]Relationships:[/green]")
                for rel in relationships:
                    console.print(f"      - {rel}")

                graph_store.store_extracted_entities(
                    graph_document=graph_docs[0],
                    utterance_id=utterance_id
                )

                results.append({
                    "utterance": utterance_text,
                    "entities": len(graph_docs[0].nodes),
                    "relationships": len(graph_docs[0].relationships),
                    "success": True
                })
            else:
                console.print("   [yellow]No entities extracted[/yellow]")
                results.append({
                    "utterance": utterance_text,
                    "entities": 0,
                    "relationships": 0,
                    "success": False
                })

        except Exception as e:
            console.print(f"   [red]Error: {e}[/red]")
            results.append({
                "utterance": utterance_text,
                "error": str(e),
                "success": False
            })

    return results


async def main():
    """Run comprehensive extraction tests"""

    console.print(Panel.fit(
        "[bold cyan]Comprehensive Extraction Testing[/bold cyan]\n\n"
        "Testing edge cases to identify extraction weaknesses",
        title="Test Suite",
        border_style="cyan"
    ))

    config = get_config()

    # Wipe database
    console.print("\n[yellow]Wiping database...[/yellow]")
    wipe_database(config)

    # Initialize
    extractor = EntityExtractor(config)
    graph_store = GraphStore(config)
    graph_store.initialize_schema()

    conversation_id = str(uuid.uuid4())
    graph_store.store_conversation_event(
        conversation_id=conversation_id,
        user_id="test_user",
        timestamp=datetime.now()
    )

    # Run all test categories
    all_results = {}

    for test_case in TEST_CASES:
        category = test_case["category"]
        utterances = test_case["utterances"]

        results = await test_category(
            category, utterances, extractor, graph_store, conversation_id
        )
        all_results[category] = results

    # Summary statistics
    console.print("\n\n[bold cyan]Summary Statistics[/bold cyan]")
    console.print("=" * 60)

    summary_table = Table(title="Extraction Results by Category")
    summary_table.add_column("Category", style="cyan")
    summary_table.add_column("Utterances", style="yellow")
    summary_table.add_column("Avg Entities", style="green")
    summary_table.add_column("Avg Relationships", style="magenta")
    summary_table.add_column("Success Rate", style="blue")

    for category, results in all_results.items():
        total = len(results)
        successful = sum(1 for r in results if r.get("success", False))
        avg_entities = sum(r.get("entities", 0) for r in results) / total if total > 0 else 0
        avg_rels = sum(r.get("relationships", 0) for r in results) / total if total > 0 else 0
        success_rate = f"{(successful/total)*100:.0f}%"

        summary_table.add_row(
            category,
            str(total),
            f"{avg_entities:.1f}",
            f"{avg_rels:.1f}",
            success_rate
        )

    console.print(summary_table)

    # Get final graph stats
    connection = Neo4jConnection(config.neo4j)
    try:
        connection.connect()

        entity_count = connection.execute_query("MATCH (e:__Entity__) RETURN count(e) AS count")[0]["count"]
        rel_count = connection.execute_query("MATCH ()-[r]->(:__Entity__) WHERE type(r) <> 'HAS_ENTITY' AND type(r) <> 'PART_OF' RETURN count(r) AS count")[0]["count"]

        console.print(f"\n[bold]Total entities in graph:[/bold] {entity_count}")
        console.print(f"[bold]Total relationships in graph:[/bold] {rel_count}")

    finally:
        connection.disconnect()

    graph_store.close()

    console.print(Panel.fit(
        "[bold green]Comprehensive testing complete![/bold green]\n\n"
        "Review the results above to identify:\n"
        "- Missing relationships\n"
        "- Incorrect entity types\n"
        "- Temporal information loss\n"
        "- Negation handling issues\n"
        "- Multi-entity relationship problems",
        title="Analysis Needed",
        border_style="green"
    ))


if __name__ == "__main__":
    asyncio.run(main())
