"""
Export Graph Structure

Exports the current graph structure from Neo4j for analysis.

Usage:
    python export_graph.py
"""

from rich.console import Console
from rich.table import Table

from backend.knowledge.config import get_config
from backend.knowledge.storage import Neo4jConnection

console = Console()


def export_graph():
    """Export graph structure from Neo4j"""

    config = get_config()
    connection = Neo4jConnection(config.neo4j)

    try:
        connection.connect()

        console.print("\n[bold cyan]Exporting Graph Structure[/bold cyan]\n")

        # Get all entities
        console.print("[yellow]Entities:[/yellow]\n")

        entity_query = """
        MATCH (e:__Entity__)
        RETURN e.id AS id,
               e.entity_type AS type,
               properties(e) AS props
        ORDER BY e.id
        """

        entities = connection.execute_query(entity_query)

        entity_table = Table(title="Entities", show_lines=True)
        entity_table.add_column("ID", style="cyan")
        entity_table.add_column("Type", style="magenta")
        entity_table.add_column("Properties", style="green")

        for entity in entities:
            entity_id = entity["id"]
            entity_type = entity["type"]
            props = {
                k: v
                for k, v in entity["props"].items()
                if k
                not in [
                    "id",
                    "entity_type",
                    "ontology_version",
                    "created_at",
                    "created_from_utterance",
                ]
            }
            props_str = str(props) if props else "-"
            entity_table.add_row(entity_id, entity_type, props_str)

        console.print(entity_table)
        console.print()

        # Get all relationships
        console.print("[yellow]Relationships:[/yellow]\n")

        rel_query = """
        MATCH (source:__Entity__)-[r]->(target:__Entity__)
        RETURN source.id AS source,
               type(r) AS relationship,
               target.id AS target,
               properties(r) AS props
        ORDER BY source.id, target.id
        """

        relationships = connection.execute_query(rel_query)

        rel_table = Table(title="Relationships", show_lines=True)
        rel_table.add_column("Source", style="cyan")
        rel_table.add_column("Relationship", style="yellow")
        rel_table.add_column("Target", style="cyan")
        rel_table.add_column("Properties", style="green")

        for rel in relationships:
            source = rel["source"]
            rel_type = rel["relationship"]
            target = rel["target"]
            props = {
                k: v
                for k, v in rel["props"].items()
                if k not in ["ontology_version", "created_at", "created_from_utterance"]
            }
            props_str = str(props) if props else "-"
            rel_table.add_row(source, rel_type, target, props_str)

        console.print(rel_table)
        console.print()

        # Get utterance provenance
        console.print("[yellow]Provenance (Utterances):[/yellow]\n")

        provenance_query = """
        MATCH (u:Utterance)-[:HAS_ENTITY]->(e:__Entity__)
        WITH u.text AS utterance, u.timestamp AS timestamp, collect(DISTINCT e.id) AS entities
        RETURN utterance, entities
        ORDER BY timestamp
        """

        provenance = connection.execute_query(provenance_query)

        prov_table = Table(title="Entity Provenance", show_lines=True)
        prov_table.add_column("Utterance", style="yellow", width=50)
        prov_table.add_column("Extracted Entities", style="cyan")

        for prov in provenance:
            utterance = prov["utterance"]
            entities_list = ", ".join(prov["entities"])
            prov_table.add_row(utterance, entities_list)

        console.print(prov_table)
        console.print()

        console.print("[green]Graph export complete![/green]\n")

    finally:
        connection.disconnect()


if __name__ == "__main__":
    export_graph()
