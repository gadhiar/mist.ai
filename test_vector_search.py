"""Test vector search directly"""
import asyncio
from backend.knowledge.config import KnowledgeConfig
from backend.knowledge.storage.graph_store import GraphStore

async def test():
    config = KnowledgeConfig.from_env()
    graph_store = GraphStore(config)

    # Test query
    query = "MIST capabilities"

    print(f"Testing vector search for: '{query}'")
    print(f"Threshold: 0.4")
    print("-" * 60)

    # Search with NO threshold to see what scores we're actually getting
    query_embedding = graph_store.embedding_generator.generate_embedding(query)

    test_query = """
    CALL db.index.vector.queryNodes('chunk_embeddings', 10, $query_embedding)
    YIELD node, score
    RETURN
        node.chunk_id AS chunk_id,
        node.text AS text,
        score AS similarity
    ORDER BY score DESC
    LIMIT 10
    """

    params = {"query_embedding": query_embedding}

    results = graph_store.connection.execute_query(test_query, params)

    print(f"\nTop 10 results (no threshold):")
    for i, r in enumerate(results, 1):
        print(f"{i}. Score: {r['similarity']:.4f}")
        print(f"   Text: {r['text'][:100]}...")
        print()

    if not results:
        print("NO RESULTS FOUND!")
        print("\nChecking if chunks exist...")
        check_query = "MATCH (c:DocumentChunk) RETURN count(c) as count, collect(c.chunk_id)[0..3] as sample_ids"
        check_results = graph_store.connection.execute_query(check_query, {})
        print(f"Total chunks: {check_results[0]['count']}")
        print(f"Sample IDs: {check_results[0]['sample_ids']}")

if __name__ == "__main__":
    asyncio.run(test())
