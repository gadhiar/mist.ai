"""Wipe Neo4j Database.

Deletes all nodes and relationships from Neo4j database.
Use this to start fresh for testing.

Usage:
    python scripts/wipe_neo4j.py
"""

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

# Running this script by path puts scripts/ on sys.path[0], NOT the repo
# root -- so `import backend` would fail. Add the repo root first (mirrors
# scripts/mist_admin.py:81-83).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.knowledge.config import get_config  # noqa: E402
from backend.knowledge.storage import Neo4jConnection  # noqa: E402

console = Console()


def wipe_database():
    """Wipe all data from Neo4j database."""
    console.print("\n[bold red]WARNING: This will DELETE ALL DATA from Neo4j![/bold red]\n")

    # Ask for confirmation
    confirm = input("Type 'yes' to confirm: ")

    if confirm.lower() != "yes":
        console.print("[yellow]Aborted - No data was deleted[/yellow]\n")
        return

    config = get_config()
    connection = Neo4jConnection(config.neo4j)

    try:
        connection.connect()

        console.print("\n[yellow]Deleting all nodes and relationships...[/yellow]")

        # Delete everything
        delete_query = """
        MATCH (n)
        DETACH DELETE n
        """

        connection.execute_write(delete_query)

        # Verify deletion
        count_query = "MATCH (n) RETURN count(n) AS count"
        result = connection.execute_query(count_query)
        node_count = result[0]["count"] if result else 0

        if node_count == 0:
            console.print(
                Panel.fit(
                    "[bold green]Database wiped successfully![/bold green]\n\n"
                    "All nodes and relationships have been deleted.\n"
                    "Ready for fresh extraction test.",
                    title="Success",
                    border_style="green",
                )
            )
        else:
            console.print(f"[yellow]Warning: {node_count} nodes still remain[/yellow]")

    except Exception as e:
        console.print(
            Panel.fit(
                f"[bold red]Error wiping database[/bold red]\n\n{str(e)}",
                title="Error",
                border_style="red",
            )
        )

    finally:
        connection.disconnect()


if __name__ == "__main__":
    wipe_database()
