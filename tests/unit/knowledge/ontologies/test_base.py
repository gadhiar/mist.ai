"""Unit tests for backend.knowledge.ontologies.base."""


def test_node_type_definition_supports_parent_type():
    from backend.knowledge.ontologies.base import KnowledgeDomain, NodeTypeDefinition

    leaf = NodeTypeDefinition(
        type_name="Concept",
        description="x",
        knowledge_domain=KnowledgeDomain.EXTERNAL,
        parent_type="Abstraction",
    )
    assert leaf.parent_type == "Abstraction"


def test_node_type_definition_parent_type_defaults_none():
    from backend.knowledge.ontologies.base import KnowledgeDomain, NodeTypeDefinition

    nt = NodeTypeDefinition(
        type_name="User", description="x", knowledge_domain=KnowledgeDomain.EXTERNAL
    )
    assert nt.parent_type is None
