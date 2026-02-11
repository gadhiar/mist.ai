"""
Knowledge Graph - Core data structure for transparent reasoning

This implements a learning, reasoning knowledge graph with:
- Provenance tracking (where knowledge came from)
- Confidence scoring (how certain we are)
- Usage tracking (reinforcement learning)
- Inference rules (pattern-based reasoning)
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum

import networkx as nx


class NodeType(Enum):
    """Types of nodes in the knowledge graph"""

    CONCEPT = "concept"  # Abstract concept (e.g., "Programming Language")
    ENTITY = "entity"  # Concrete entity (e.g., "Python")
    FACT = "fact"  # Factual statement
    RULE = "rule"  # Inference rule
    PATTERN = "pattern"  # Learned pattern


class EdgeType(Enum):
    """Types of relationships between nodes"""

    IS_A = "is_a"  # Taxonomy (Python is_a Programming Language)
    HAS_PROPERTY = "has_property"  # Attribute (Python has_property "interpreted")
    USED_FOR = "used_for"  # Purpose (Python used_for "web development")
    PART_OF = "part_of"  # Composition
    CAUSES = "causes"  # Causality
    RELATED_TO = "related_to"  # General association
    IMPLIES = "implies"  # Logical implication (for inference rules)


class Source(Enum):
    """Where knowledge came from (provenance)"""

    USER = "user"  # Taught by user
    CLOUD = "cloud"  # From cloud AI (Claude/GPT)
    INFERENCE = "inference"  # Inferred from existing knowledge
    WEB = "web"  # From web search
    OBSERVATION = "observation"  # From observing patterns


@dataclass
class Node:
    """
    A node in the knowledge graph

    Nodes represent concepts, entities, facts, or rules.
    They track their own confidence, usage, and learning history.
    """

    id: str
    label: str
    node_type: NodeType
    properties: dict = field(default_factory=dict)
    learned_from: Source = Source.USER
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0  # 0.0 to 1.0
    usage_count: int = 0  # How often used in reasoning
    last_used: datetime | None = None

    def __post_init__(self):
        """Convert string enums to enum objects if needed"""
        if isinstance(self.node_type, str):
            self.node_type = NodeType(self.node_type)
        if isinstance(self.learned_from, str):
            self.learned_from = Source(self.learned_from)

    def use(self):
        """Mark this node as used (reinforcement)"""
        self.usage_count += 1
        self.last_used = datetime.now()

    def update_confidence(self, delta: float):
        """Adjust confidence based on feedback"""
        self.confidence = max(0.0, min(1.0, self.confidence + delta))

    def to_dict(self) -> dict:
        """Serialize to dictionary"""
        data = asdict(self)
        data["node_type"] = self.node_type.value
        data["learned_from"] = self.learned_from.value
        data["timestamp"] = self.timestamp.isoformat()
        if self.last_used:
            data["last_used"] = self.last_used.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Node":
        """Deserialize from dictionary"""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        if data.get("last_used"):
            data["last_used"] = datetime.fromisoformat(data["last_used"])
        return cls(**data)


@dataclass
class Edge:
    """
    An edge (relationship) in the knowledge graph

    Edges connect nodes and represent typed relationships.
    They also track confidence and learning provenance.
    """

    from_id: str
    to_id: str
    edge_type: EdgeType
    properties: dict = field(default_factory=dict)
    learned_from: Source = Source.USER
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    reasoning_path: str | None = None  # How was this inferred
    usage_count: int = 0
    success_count: int = 0  # How often this edge led to correct reasoning

    def __post_init__(self):
        """Convert string enums to enum objects if needed"""
        if isinstance(self.edge_type, str):
            self.edge_type = EdgeType(self.edge_type)
        if isinstance(self.learned_from, str):
            self.learned_from = Source(self.learned_from)

    def use(self, successful: bool = True):
        """Mark edge as used in reasoning"""
        self.usage_count += 1
        if successful:
            self.success_count += 1

    def update_confidence(self):
        """Update confidence based on success rate"""
        if self.usage_count > 0:
            success_rate = self.success_count / self.usage_count
            # Weighted average with prior confidence
            self.confidence = 0.7 * self.confidence + 0.3 * success_rate

    def to_dict(self) -> dict:
        """Serialize to dictionary"""
        data = asdict(self)
        data["edge_type"] = self.edge_type.value
        data["learned_from"] = self.learned_from.value
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Edge":
        """Deserialize from dictionary"""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class KnowledgeGraph:
    """
    The core knowledge graph with reasoning capabilities

    This is NOT a simple database - it's a learning, reasoning system:
    - Tracks provenance and confidence
    - Learns from usage patterns
    - Supports inference and pattern matching
    - Self-modifies based on feedback
    """

    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.graph = nx.DiGraph()  # NetworkX for efficient graph operations

        # Indexes for fast lookup
        self._label_index: dict[str, set[str]] = {}  # label -> node_ids
        self._type_index: dict[NodeType, set[str]] = {}  # type -> node_ids

    def add_node(self, node: Node) -> None:
        """Add a node to the graph"""
        # Store node
        self.nodes[node.id] = node
        self.graph.add_node(node.id, **node.to_dict())

        # Update indexes
        if node.label not in self._label_index:
            self._label_index[node.label] = set()
        self._label_index[node.label].add(node.id)

        if node.node_type not in self._type_index:
            self._type_index[node.node_type] = set()
        self._type_index[node.node_type].add(node.id)

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph"""
        # Validate nodes exist
        if edge.from_id not in self.nodes:
            raise ValueError(f"Node {edge.from_id} not found")
        if edge.to_id not in self.nodes:
            raise ValueError(f"Node {edge.to_id} not found")

        # Store edge
        self.edges.append(edge)
        self.graph.add_edge(edge.from_id, edge.to_id, **edge.to_dict())

    def get_node(self, node_id: str) -> Node | None:
        """Get node by ID"""
        return self.nodes.get(node_id)

    def find_nodes_by_label(self, label: str) -> list[Node]:
        """Find all nodes with matching label"""
        node_ids = self._label_index.get(label, set())
        return [self.nodes[nid] for nid in node_ids]

    def find_nodes_by_type(self, node_type: NodeType) -> list[Node]:
        """Find all nodes of a given type"""
        node_ids = self._type_index.get(node_type, set())
        return [self.nodes[nid] for nid in node_ids]

    def find_paths(self, start_id: str, end_id: str, max_depth: int = 5) -> list[list[Edge]]:
        """
        Find all paths between two nodes

        Returns list of paths, where each path is a list of edges
        """
        try:
            # Find all simple paths (no cycles)
            paths = nx.all_simple_paths(
                self.graph, source=start_id, target=end_id, cutoff=max_depth
            )

            result = []
            for path_nodes in paths:
                # Convert node path to edge path
                path_edges = []
                for i in range(len(path_nodes) - 1):
                    from_id = path_nodes[i]
                    to_id = path_nodes[i + 1]
                    # Find edge between these nodes
                    edge = self._find_edge(from_id, to_id)
                    if edge:
                        path_edges.append(edge)
                if path_edges:
                    result.append(path_edges)

            return result
        except nx.NodeNotFound:
            return []

    def _find_edge(self, from_id: str, to_id: str) -> Edge | None:
        """Find edge between two nodes"""
        for edge in self.edges:
            if edge.from_id == from_id and edge.to_id == to_id:
                return edge
        return None

    def get_neighborhood(self, node_id: str, radius: int = 2) -> dict:
        """
        Get all nodes within radius hops

        Returns: {node_id: Node} dictionary
        """
        if node_id not in self.nodes:
            return {}

        # BFS to find neighbors within radius
        neighbors = {}
        visited = {node_id}
        current_level = {node_id}

        for _ in range(radius):
            next_level = set()
            for nid in current_level:
                # Get successors and predecessors
                for neighbor in self.graph.successors(nid):
                    if neighbor not in visited:
                        neighbors[neighbor] = self.nodes[neighbor]
                        next_level.add(neighbor)
                        visited.add(neighbor)

                for neighbor in self.graph.predecessors(nid):
                    if neighbor not in visited:
                        neighbors[neighbor] = self.nodes[neighbor]
                        next_level.add(neighbor)
                        visited.add(neighbor)

            current_level = next_level
            if not current_level:
                break

        # Include center node
        neighbors[node_id] = self.nodes[node_id]
        return neighbors

    def prune_low_confidence(self, threshold: float = 0.3):
        """Remove nodes and edges below confidence threshold"""
        # Remove low confidence nodes
        to_remove = [nid for nid, node in self.nodes.items() if node.confidence < threshold]

        for nid in to_remove:
            self.remove_node(nid)

        # Remove low confidence edges
        self.edges = [e for e in self.edges if e.confidence >= threshold]

        # Rebuild graph
        self._rebuild_graph()

    def remove_node(self, node_id: str):
        """Remove a node and all connected edges"""
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]

        # Remove from indexes
        if node.label in self._label_index:
            self._label_index[node.label].discard(node_id)
        if node.node_type in self._type_index:
            self._type_index[node.node_type].discard(node_id)

        # Remove edges connected to this node
        self.edges = [e for e in self.edges if e.from_id != node_id and e.to_id != node_id]

        # Remove from nodes and graph
        del self.nodes[node_id]
        if self.graph.has_node(node_id):
            self.graph.remove_node(node_id)

    def _rebuild_graph(self):
        """Rebuild NetworkX graph from nodes and edges"""
        self.graph = nx.DiGraph()
        for node_id, node in self.nodes.items():
            self.graph.add_node(node_id, **node.to_dict())
        for edge in self.edges:
            self.graph.add_edge(edge.from_id, edge.to_id, **edge.to_dict())

    def get_stats(self) -> dict:
        """Get graph statistics"""
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "node_types": {nt.value: len(self._type_index.get(nt, set())) for nt in NodeType},
            "avg_confidence": (
                sum(n.confidence for n in self.nodes.values()) / len(self.nodes)
                if self.nodes
                else 0
            ),
            "sources": {
                s.value: sum(1 for n in self.nodes.values() if n.learned_from == s) for s in Source
            },
        }

    def save(self, filepath: str):
        """Save graph to JSON file"""
        data = {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "metadata": {"saved_at": datetime.now().isoformat(), "stats": self.get_stats()},
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, filepath: str):
        """Load graph from JSON file"""
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        # Clear current graph
        self.nodes = {}
        self.edges = []
        self.graph = nx.DiGraph()
        self._label_index = {}
        self._type_index = {}

        # Load nodes
        for node_data in data["nodes"]:
            node = Node.from_dict(node_data)
            self.add_node(node)

        # Load edges
        for edge_data in data["edges"]:
            edge = Edge.from_dict(edge_data)
            self.add_edge(edge)
