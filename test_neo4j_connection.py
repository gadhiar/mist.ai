"""
Test Neo4j Connection

Simple script to verify Neo4j Desktop is running and accessible.

Usage:
    python test_neo4j_connection.py
"""

import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from backend.knowledge.config import get_config
from backend.knowledge.storage import Neo4jConnection

console = Console()


def test_connection():
    """Test Neo4j connection and display results"""

    console.print("\n[bold cyan]Testing Neo4j Connection...[/bold cyan]\n")

    # Load config
    config = get_config()

    console.print(f"[yellow]URI:[/yellow] {config.neo4j.uri}")
    console.print(f"[yellow]Username:[/yellow] {config.neo4j.username}")
    console.print(f"[yellow]Database:[/yellow] {config.neo4j.database}\n")

    # Try to connect
    connection = Neo4jConnection(config.neo4j)

    try:
        # Attempt connection
        connection.connect()

        # Run health check
        health = connection.health_check()

        # Display results
        if health["status"] == "healthy":
            console.print(
                Panel.fit(
                    f"[bold green] SUCCESS![/bold green]\n\n{health['message']}",
                    title="Neo4j Connection",
                    border_style="green",
                )
            )

            # Test a simple query
            console.print("\n[bold cyan]Running test query...[/bold cyan]")
            result = connection.execute_query("MATCH (n) RETURN count(n) AS node_count")

            table = Table(title="Database Stats")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            for record in result:
                table.add_row("Total Nodes", str(record["node_count"]))

            console.print(table)

            console.print("\n[bold green] Neo4j is ready to use![/bold green]\n")
            return True

        else:
            console.print(
                Panel.fit(
                    f"[bold red] FAILED[/bold red]\n\n{health['message']}\n\n{health.get('error', '')}",
                    title="Neo4j Connection",
                    border_style="red",
                )
            )
            return False

    except Exception as e:
        console.print(
            Panel.fit(
                f"[bold red] CONNECTION FAILED[/bold red]\n\n{str(e)}",
                title="Neo4j Connection Error",
                border_style="red",
            )
        )

        console.print("\n[yellow]Troubleshooting:[/yellow]")
        console.print("1. Make sure Neo4j Desktop is running")
        console.print("2. Check that a database is started")
        console.print("3. Verify credentials in .env file or backend/knowledge/config.py")
        console.print("4. Confirm URI is 'bolt://localhost:7687'\n")

        return False

    finally:
        connection.disconnect()


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
