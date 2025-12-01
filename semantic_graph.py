"""
Semantic Graph for Knowledge Representation and Reasoning

A comprehensive graph-based knowledge system that supports:
- Multiple node types (facts, concepts, tools, plans, tasks, emotions, etc.)
- Rich relationship types with properties
- Temporal tracking and versioning
- Confidence and source attribution
- Query and traversal capabilities
- Reasoning support (inference, contradiction detection, etc.)
"""

import json
import re
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import hashlib


class CompactEmbeddingEncoder(json.JSONEncoder):
    """Custom JSON encoder that keeps embedding arrays on a single line."""
    def encode(self, obj):
        result = super().encode(obj)
        # Pattern to find "embeddings": followed by a multi-line array
        # This collapses the array to a single line
        def compact_embedding(match):
            # Get the array content and remove newlines/extra spaces
            array_str = match.group(1)
            # Remove newlines and collapse multiple spaces
            compact = re.sub(r'\s+', ' ', array_str).strip()
            return f'"embeddings": [{compact}]'
        
        result = re.sub(
            r'"embeddings":\s*\[([\s\S]*?)\]',
            compact_embedding,
            result
        )
        return result


# ============================================================================
# ENUMS - Node and Relationship Types
# ============================================================================

class NodeType(Enum):
    """Types of nodes in the semantic graph."""
    # Core Knowledge Types
    FACT = "fact"                    # Verified or stated facts
    CONCEPT = "concept"              # Abstract concepts/ideas
    ENTITY = "entity"                # Named entities (people, places, things)
    
    # Action & Planning
    TASK = "task"                    # Tasks to be done
    PLAN = "plan"                    # Plans (collections of tasks)
    GOAL = "goal"                    # Goals/objectives
    ACTION = "action"                # Actions that can be taken
    TOOL = "tool"                    # Tools/capabilities available
    
    # Reasoning & Logic
    CONCLUSION = "conclusion"        # Derived conclusions
    HYPOTHESIS = "hypothesis"        # Unverified hypotheses
    ARGUMENT = "argument"            # Arguments for/against something
    EVIDENCE = "evidence"            # Supporting evidence
    RULE = "rule"                    # Logical rules/constraints
    
    # Mental State
    EMOTION = "emotion"              # Emotional states
    BELIEF = "belief"                # Beliefs held
    PREFERENCE = "preference"        # Preferences
    MEMORY = "memory"                # Episodic memories
    OBSERVATION = "observation"      # Direct observations
    
    # Meta
    QUESTION = "question"            # Open questions
    UNCERTAINTY = "uncertainty"      # Areas of uncertainty
    CONTEXT = "context"              # Contextual information
    EVENT = "event"                  # Events in time
    
    # Domain-Specific
    SKILL = "skill"                  # Skills/abilities
    RESOURCE = "resource"            # Resources available
    CONSTRAINT = "constraint"        # Constraints/limitations
    PROCEDURE = "procedure"          # Step-by-step procedures


class RelationType(Enum):
    """Types of relationships between nodes."""
    # Logical Relations
    CAUSES = "causes"                # A causes B
    CAUSED_BY = "caused_by"          # A is caused by B
    IMPLIES = "implies"              # A implies B
    CONTRADICTS = "contradicts"      # A contradicts B
    SUPPORTS = "supports"            # A supports B
    REFUTES = "refutes"              # A refutes B
    
    # Hierarchical Relations
    IS_A = "is_a"                    # A is a type of B
    PART_OF = "part_of"              # A is part of B
    HAS_PART = "has_part"            # A has B as a part
    INSTANCE_OF = "instance_of"      # A is an instance of B
    SUBCLASS_OF = "subclass_of"      # A is a subclass of B
    
    # Semantic Relations
    RELATED_TO = "related_to"        # General relation
    SIMILAR_TO = "similar_to"        # A is similar to B
    OPPOSITE_OF = "opposite_of"      # A is opposite of B
    SYNONYM_OF = "synonym_of"        # A is synonym of B
    DEFINED_BY = "defined_by"        # A is defined by B
    
    # Temporal Relations
    BEFORE = "before"                # A happens before B
    AFTER = "after"                  # A happens after B
    DURING = "during"                # A happens during B
    FOLLOWS = "follows"              # A follows B in sequence
    PRECEDES = "precedes"            # A precedes B in sequence
    
    # Task/Action Relations
    REQUIRES = "requires"            # A requires B
    ENABLES = "enables"              # A enables B
    BLOCKS = "blocks"                # A blocks B
    ACHIEVES = "achieves"            # A achieves B (goal)
    DEPENDS_ON = "depends_on"        # A depends on B
    USES = "uses"                    # A uses B (tool/resource)
    PRODUCES = "produces"            # A produces B
    
    # Attribution Relations
    CREATED_BY = "created_by"        # A was created by B
    DERIVED_FROM = "derived_from"    # A was derived from B
    BASED_ON = "based_on"            # A is based on B
    EVIDENCED_BY = "evidenced_by"    # A is evidenced by B
    
    # Ownership/Association
    BELONGS_TO = "belongs_to"        # A belongs to B
    ASSOCIATED_WITH = "associated_with"  # A is associated with B
    ABOUT = "about"                  # A is about B (topic)
    
    # Emotional/Evaluative
    TRIGGERS = "triggers"            # A triggers B (emotion)
    EVOKES = "evokes"                # A evokes B
    PREFERS = "prefers"              # A prefers B
    VALUES = "values"                # A values B


class ConfidenceLevel(Enum):
    """Confidence levels for knowledge."""
    CERTAIN = 1.0
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4
    UNCERTAIN = 0.2
    UNKNOWN = 0.0


class NodeStatus(Enum):
    """Status of a node."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    PENDING = "pending"
    INVALIDATED = "invalidated"
    SUPERSEDED = "superseded"


# ============================================================================
# DATA CLASSES - Graph Elements
# ============================================================================

@dataclass
class NodeMetadata:
    """Metadata associated with a node."""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    accessed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0
    version: int = 1
    source: str = "system"
    tags: list[str] = field(default_factory=list)
    
    def touch(self):
        """Update access time and count."""
        self.accessed_at = datetime.now().isoformat()
        self.access_count += 1
    
    def update(self):
        """Mark as updated."""
        self.updated_at = datetime.now().isoformat()
        self.version += 1


@dataclass
class Node:
    """A node in the semantic graph."""
    id: str
    type: NodeType
    label: str
    content: str
    properties: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.8
    status: NodeStatus = NodeStatus.ACTIVE
    metadata: NodeMetadata = field(default_factory=NodeMetadata)
    embeddings: Optional[list[float]] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "label": self.label,
            "content": self.content,
            "properties": self.properties,
            "confidence": self.confidence,
            "status": self.status.value,
            "metadata": asdict(self.metadata),
            "embeddings": self.embeddings
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Node":
        """Create from dictionary."""
        metadata = NodeMetadata(**data.get("metadata", {}))
        return cls(
            id=data["id"],
            type=NodeType(data["type"]),
            label=data["label"],
            content=data["content"],
            properties=data.get("properties", {}),
            confidence=data.get("confidence", 0.8),
            status=NodeStatus(data.get("status", "active")),
            metadata=metadata,
            embeddings=data.get("embeddings")
        )


@dataclass
class Edge:
    """An edge (relationship) in the semantic graph."""
    id: str
    source_id: str
    target_id: str
    relation: RelationType
    properties: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    confidence: float = 0.8
    bidirectional: bool = False
    metadata: NodeMetadata = field(default_factory=NodeMetadata)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation.value,
            "properties": self.properties,
            "weight": self.weight,
            "confidence": self.confidence,
            "bidirectional": self.bidirectional,
            "metadata": asdict(self.metadata)
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Edge":
        """Create from dictionary."""
        metadata = NodeMetadata(**data.get("metadata", {}))
        return cls(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation=RelationType(data["relation"]),
            properties=data.get("properties", {}),
            weight=data.get("weight", 1.0),
            confidence=data.get("confidence", 0.8),
            bidirectional=data.get("bidirectional", False),
            metadata=metadata
        )


@dataclass
class QueryResult:
    """Result of a graph query."""
    nodes: list[Node]
    edges: list[Edge]
    paths: list[list[str]] = field(default_factory=list)
    score: float = 0.0
    explanation: str = ""


# ============================================================================
# SEMANTIC GRAPH - Main Class
# ============================================================================

class SemanticGraph:
    """
    A semantic graph for knowledge representation and reasoning.
    
    Supports:
    - Adding, updating, removing nodes and edges
    - Rich querying and traversal
    - Inference and reasoning
    - Persistence and serialization
    - Conflict detection
    """
    
    def __init__(self, name: str = "knowledge_graph"):
        """Initialize the semantic graph."""
        self.name = name
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Edge] = {}
        
        # Indexes for fast lookup
        self._nodes_by_type: dict[NodeType, set[str]] = defaultdict(set)
        self._nodes_by_label: dict[str, set[str]] = defaultdict(set)
        self._edges_by_source: dict[str, set[str]] = defaultdict(set)
        self._edges_by_target: dict[str, set[str]] = defaultdict(set)
        self._edges_by_relation: dict[RelationType, set[str]] = defaultdict(set)
        
        # Reasoning rules
        self._inference_rules: list[Callable] = []
        
        # Graph metadata
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "version": 1
        }
    
    # ========================================================================
    # Node Operations
    # ========================================================================
    
    def add_node(
        self,
        node_type: NodeType,
        label: str,
        content: str,
        properties: Optional[dict[str, Any]] = None,
        confidence: float = 0.8,
        source: str = "system",
        tags: Optional[list[str]] = None,
        node_id: Optional[str] = None
    ) -> Node:
        """
        Add a new node to the graph.
        
        Args:
            node_type: Type of the node
            label: Short label for the node
            content: Full content/description
            properties: Additional properties
            confidence: Confidence level (0-1)
            source: Source of this knowledge
            tags: Tags for categorization
            node_id: Optional specific ID
            
        Returns:
            The created node
        """
        node_id = node_id or self._generate_id(f"{node_type.value}:{label}")
        
        # Check for existing node with same ID
        if node_id in self.nodes:
            updated = self.update_node(node_id, content=content, properties=properties)
            if updated:
                return updated
        
        metadata = NodeMetadata(
            source=source,
            tags=tags or []
        )
        
        node = Node(
            id=node_id,
            type=node_type,
            label=label,
            content=content,
            properties=properties or {},
            confidence=confidence,
            metadata=metadata
        )
        
        self.nodes[node_id] = node
        self._index_node(node)
        self._update_metadata()
        
        return node
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        node = self.nodes.get(node_id)
        if node:
            node.metadata.touch()
        return node
    
    def update_node(
        self,
        node_id: str,
        label: Optional[str] = None,
        content: Optional[str] = None,
        properties: Optional[dict[str, Any]] = None,
        confidence: Optional[float] = None,
        status: Optional[NodeStatus] = None
    ) -> Optional[Node]:
        """Update an existing node."""
        node = self.nodes.get(node_id)
        if not node:
            return None
        
        # Unindex before update
        self._unindex_node(node)
        
        if label is not None:
            node.label = label
        if content is not None:
            node.content = content
        if properties is not None:
            node.properties.update(properties)
        if confidence is not None:
            node.confidence = confidence
        if status is not None:
            node.status = status
        
        node.metadata.update()
        
        # Reindex after update
        self._index_node(node)
        self._update_metadata()
        
        return node
    
    def remove_node(self, node_id: str, cascade: bool = True) -> bool:
        """
        Remove a node from the graph.
        
        Args:
            node_id: ID of the node to remove
            cascade: If True, also remove connected edges
            
        Returns:
            True if removed, False if not found
        """
        node = self.nodes.get(node_id)
        if not node:
            return False
        
        if cascade:
            # Remove all connected edges
            edge_ids = list(self._edges_by_source.get(node_id, set()))
            edge_ids.extend(self._edges_by_target.get(node_id, set()))
            for edge_id in edge_ids:
                self.remove_edge(edge_id)
        
        self._unindex_node(node)
        del self.nodes[node_id]
        self._update_metadata()
        
        return True
    
    # ========================================================================
    # Edge Operations
    # ========================================================================
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: RelationType,
        properties: Optional[dict[str, Any]] = None,
        weight: float = 1.0,
        confidence: float = 0.8,
        bidirectional: bool = False,
        edge_id: Optional[str] = None
    ) -> Optional[Edge]:
        """
        Add a new edge (relationship) to the graph.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            relation: Type of relationship
            properties: Additional properties
            weight: Edge weight
            confidence: Confidence level
            bidirectional: If True, relation goes both ways
            edge_id: Optional specific ID
            
        Returns:
            The created edge, or None if nodes don't exist
        """
        # Verify nodes exist
        if source_id not in self.nodes or target_id not in self.nodes:
            return None
        
        edge_id = edge_id or self._generate_id(
            f"{source_id}:{relation.value}:{target_id}"
        )
        
        edge = Edge(
            id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            properties=properties or {},
            weight=weight,
            confidence=confidence,
            bidirectional=bidirectional
        )
        
        self.edges[edge_id] = edge
        self._index_edge(edge)
        self._update_metadata()
        
        return edge
    
    def get_edge(self, edge_id: str) -> Optional[Edge]:
        """Get an edge by ID."""
        return self.edges.get(edge_id)
    
    def get_edges_between(
        self,
        source_id: str,
        target_id: str,
        relation: Optional[RelationType] = None
    ) -> list[Edge]:
        """Get all edges between two nodes."""
        edges = []
        for edge_id in self._edges_by_source.get(source_id, set()):
            edge = self.edges[edge_id]
            if edge.target_id == target_id:
                if relation is None or edge.relation == relation:
                    edges.append(edge)
        return edges
    
    def remove_edge(self, edge_id: str) -> bool:
        """Remove an edge from the graph."""
        edge = self.edges.get(edge_id)
        if not edge:
            return False
        
        self._unindex_edge(edge)
        del self.edges[edge_id]
        self._update_metadata()
        
        return True
    
    # ========================================================================
    # Query Operations
    # ========================================================================
    
    def get_nodes_by_type(self, node_type: NodeType, active_only: bool = True) -> list[Node]:
        """Get all nodes of a specific type."""
        node_ids = self._nodes_by_type.get(node_type, set())
        nodes = [self.nodes[nid] for nid in node_ids if nid in self.nodes]
        if active_only:
            nodes = [n for n in nodes if n.status == NodeStatus.ACTIVE]
        return nodes
    
    def get_nodes_by_label(self, label: str, fuzzy: bool = False, active_only: bool = True) -> list[Node]:
        """Get nodes by label."""
        if fuzzy:
            results = []
            label_lower = label.lower()
            for node in self.nodes.values():
                if label_lower in node.label.lower():
                    if not active_only or node.status == NodeStatus.ACTIVE:
                        results.append(node)
            return results
        
        node_ids = self._nodes_by_label.get(label.lower(), set())
        nodes = [self.nodes[nid] for nid in node_ids if nid in self.nodes]
        if active_only:
            nodes = [n for n in nodes if n.status == NodeStatus.ACTIVE]
        return nodes
    
    def get_neighbors(
        self,
        node_id: str,
        relation: Optional[RelationType] = None,
        direction: str = "both",
        active_only: bool = True
    ) -> list[Node]:
        """
        Get neighboring nodes.
        
        Args:
            node_id: ID of the node
            relation: Filter by relation type
            direction: "outgoing", "incoming", or "both"
            active_only: Only return active nodes (default True)
            
        Returns:
            List of neighboring nodes
        """
        neighbors = []
        
        if direction in ("outgoing", "both"):
            for edge_id in self._edges_by_source.get(node_id, set()):
                edge = self.edges[edge_id]
                if relation is None or edge.relation == relation:
                    if edge.target_id in self.nodes:
                        node = self.nodes[edge.target_id]
                        if not active_only or node.status == NodeStatus.ACTIVE:
                            neighbors.append(node)
        
        if direction in ("incoming", "both"):
            for edge_id in self._edges_by_target.get(node_id, set()):
                edge = self.edges[edge_id]
                if relation is None or edge.relation == relation:
                    if edge.source_id in self.nodes:
                        node = self.nodes[edge.source_id]
                        if not active_only or node.status == NodeStatus.ACTIVE:
                            neighbors.append(node)
        
        return neighbors
    
    def get_outgoing_edges(
        self,
        node_id: str,
        relation: Optional[RelationType] = None,
        active_only: bool = True
    ) -> list[Edge]:
        """Get all outgoing edges from a node.
        
        Args:
            node_id: The source node ID
            relation: Optional filter for relation type
            active_only: Only return edges to active target nodes (default True)
        """
        edges = []
        for edge_id in self._edges_by_source.get(node_id, set()):
            edge = self.edges[edge_id]
            if relation is None or edge.relation == relation:
                # Skip edges to inactive target nodes
                if active_only:
                    target_node = self.nodes.get(edge.target_id)
                    if target_node and target_node.status != NodeStatus.ACTIVE:
                        continue
                edges.append(edge)
        return edges
    
    def get_incoming_edges(
        self,
        node_id: str,
        relation: Optional[RelationType] = None,
        active_only: bool = True
    ) -> list[Edge]:
        """Get all incoming edges to a node.
        
        Args:
            node_id: The target node ID
            relation: Optional filter for relation type
            active_only: Only return edges from active source nodes (default True)
        """
        edges = []
        for edge_id in self._edges_by_target.get(node_id, set()):
            edge = self.edges[edge_id]
            if relation is None or edge.relation == relation:
                # Skip edges from inactive source nodes
                if active_only:
                    source_node = self.nodes.get(edge.source_id)
                    if source_node and source_node.status != NodeStatus.ACTIVE:
                        continue
                edges.append(edge)
        return edges
    
    def search(
        self,
        query: str,
        node_types: Optional[list[NodeType]] = None,
        min_confidence: float = 0.0,
        limit: int = 10
    ) -> list[Node]:
        """
        Search for nodes matching a query.
        
        Args:
            query: Search query
            node_types: Filter by node types
            min_confidence: Minimum confidence level
            limit: Maximum results
            
        Returns:
            List of matching nodes
        """
        results = []
        query_lower = query.lower()
        
        for node in self.nodes.values():
            # Type filter
            if node_types and node.type not in node_types:
                continue
            
            # Confidence filter
            if node.confidence < min_confidence:
                continue
            
            # Status filter
            if node.status != NodeStatus.ACTIVE:
                continue
            
            # Text match
            score = 0.0
            if query_lower in node.label.lower():
                score += 2.0
            if query_lower in node.content.lower():
                score += 1.0
            for tag in node.metadata.tags:
                if query_lower in tag.lower():
                    score += 0.5
            
            if score > 0:
                results.append((score * node.confidence, node))
        
        # Sort by score
        results.sort(key=lambda x: x[0], reverse=True)
        
        return [node for _, node in results[:limit]]
    
    def query(
        self,
        start_type: Optional[NodeType] = None,
        relation_path: Optional[list[RelationType]] = None,
        end_type: Optional[NodeType] = None,
        filters: Optional[dict[str, Any]] = None,
        active_only: bool = True
    ) -> QueryResult:
        """
        Execute a structured query on the graph.
        
        Args:
            start_type: Type of starting nodes
            relation_path: Path of relations to follow
            end_type: Type of ending nodes
            filters: Property filters
            active_only: Only include active nodes (default True)
            
        Returns:
            QueryResult with matching nodes, edges, and paths
        """
        result_nodes = set()
        result_edges = set()
        paths = []
        
        # Get starting nodes
        if start_type:
            start_nodes = self.get_nodes_by_type(start_type, active_only=active_only)
        else:
            start_nodes = list(self.nodes.values())
            if active_only:
                start_nodes = [n for n in start_nodes if n.status == NodeStatus.ACTIVE]
        
        # Apply filters
        if filters:
            start_nodes = [
                n for n in start_nodes
                if all(n.properties.get(k) == v for k, v in filters.items())
            ]
        
        # Follow relation path
        if relation_path:
            for start_node in start_nodes:
                self._traverse_path(
                    start_node.id,
                    relation_path,
                    0,
                    [start_node.id],
                    result_nodes,
                    result_edges,
                    paths,
                    end_type,
                    active_only
                )
        else:
            result_nodes = set(n.id for n in start_nodes)
        
        return QueryResult(
            nodes=[self.nodes[nid] for nid in result_nodes if nid in self.nodes],
            edges=[self.edges[eid] for eid in result_edges if eid in self.edges],
            paths=paths
        )
    
    def _traverse_path(
        self,
        current_id: str,
        relations: list[RelationType],
        depth: int,
        current_path: list[str],
        result_nodes: set,
        result_edges: set,
        paths: list,
        end_type: Optional[NodeType] = None,
        active_only: bool = True
    ):
        """Recursively traverse a relation path."""
        if depth >= len(relations):
            # Check end type filter
            node = self.nodes.get(current_id)
            if node and (end_type is None or node.type == end_type):
                # Check if node is active
                if not active_only or node.status == NodeStatus.ACTIVE:
                    result_nodes.add(current_id)
                    paths.append(current_path.copy())
            return
        
        relation = relations[depth]
        for edge_id in self._edges_by_source.get(current_id, set()):
            edge = self.edges[edge_id]
            if edge.relation == relation:
                # Skip edges to inactive nodes
                target_node = self.nodes.get(edge.target_id)
                if active_only and target_node and target_node.status != NodeStatus.ACTIVE:
                    continue
                
                result_edges.add(edge_id)
                new_path = current_path + [edge.target_id]
                self._traverse_path(
                    edge.target_id,
                    relations,
                    depth + 1,
                    new_path,
                    result_nodes,
                    result_edges,
                    paths,
                    end_type,
                    active_only
                )
    
    # ========================================================================
    # Traversal Operations
    # ========================================================================
    
    def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5,
        relations: Optional[list[RelationType]] = None,
        active_only: bool = True
    ) -> Optional[list[str]]:
        """
        Find a path between two nodes using BFS.
        
        Args:
            start_id: Starting node ID
            end_id: Ending node ID
            max_depth: Maximum path length
            relations: Optional filter for relation types
            active_only: Only traverse through active nodes (default True)
            
        Returns:
            List of node IDs in the path, or None if no path found
        """
        if start_id not in self.nodes or end_id not in self.nodes:
            return None
        
        # Check if start/end nodes are active
        start_node = self.nodes[start_id]
        end_node = self.nodes[end_id]
        if active_only:
            if start_node.status != NodeStatus.ACTIVE or end_node.status != NodeStatus.ACTIVE:
                return None
        
        if start_id == end_id:
            return [start_id]
        
        visited = {start_id}
        queue = [(start_id, [start_id])]
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
            
            for edge_id in self._edges_by_source.get(current, set()):
                edge = self.edges[edge_id]
                
                if relations and edge.relation not in relations:
                    continue
                
                next_id = edge.target_id
                
                # Skip inactive nodes
                if active_only:
                    next_node = self.nodes.get(next_id)
                    if next_node and next_node.status != NodeStatus.ACTIVE:
                        continue
                
                if next_id == end_id:
                    return path + [next_id]
                
                if next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, path + [next_id]))
        
        return None
    
    def find_all_paths(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5,
        active_only: bool = True
    ) -> list[list[str]]:
        """Find all paths between two nodes."""
        if start_id not in self.nodes or end_id not in self.nodes:
            return []
        
        # Check if start/end nodes are active
        start_node = self.nodes[start_id]
        end_node = self.nodes[end_id]
        if active_only:
            if start_node.status != NodeStatus.ACTIVE or end_node.status != NodeStatus.ACTIVE:
                return []
        
        all_paths = []
        
        def dfs(current: str, path: list[str], visited: set):
            if len(path) > max_depth:
                return
            
            if current == end_id:
                all_paths.append(path.copy())
                return
            
            for edge_id in self._edges_by_source.get(current, set()):
                edge = self.edges[edge_id]
                next_id = edge.target_id
                
                # Skip inactive nodes
                if active_only:
                    next_node = self.nodes.get(next_id)
                    if next_node and next_node.status != NodeStatus.ACTIVE:
                        continue
                
                if next_id not in visited:
                    visited.add(next_id)
                    path.append(next_id)
                    dfs(next_id, path, visited)
                    path.pop()
                    visited.remove(next_id)
        
        dfs(start_id, [start_id], {start_id})
        return all_paths
    
    def get_subgraph(
        self,
        center_id: str,
        depth: int = 2,
        active_only: bool = True
    ) -> "SemanticGraph":
        """
        Extract a subgraph centered on a node.
        
        Args:
            center_id: Center node ID
            depth: How many hops to include
            active_only: Only include active nodes (default True)
            
        Returns:
            A new SemanticGraph containing the subgraph
        """
        subgraph = SemanticGraph(name=f"{self.name}_subgraph")
        
        if center_id not in self.nodes:
            return subgraph
        
        # Check if center node is active
        center_node = self.nodes[center_id]
        if active_only and center_node.status != NodeStatus.ACTIVE:
            return subgraph
        
        visited = set()
        queue = [(center_id, 0)]
        
        while queue:
            node_id, current_depth = queue.pop(0)
            
            if node_id in visited:
                continue
            
            node = self.nodes[node_id]
            
            # Skip inactive nodes
            if active_only and node.status != NodeStatus.ACTIVE:
                continue
            
            visited.add(node_id)
            
            # Add node to subgraph
            subgraph.nodes[node_id] = Node.from_dict(node.to_dict())
            subgraph._index_node(subgraph.nodes[node_id])
            
            if current_depth < depth:
                # Add neighbors
                for edge_id in self._edges_by_source.get(node_id, set()):
                    edge = self.edges[edge_id]
                    if edge.target_id not in visited:
                        queue.append((edge.target_id, current_depth + 1))
                
                for edge_id in self._edges_by_target.get(node_id, set()):
                    edge = self.edges[edge_id]
                    if edge.source_id not in visited:
                        queue.append((edge.source_id, current_depth + 1))
        
        # Add edges between included nodes
        for edge in self.edges.values():
            if edge.source_id in visited and edge.target_id in visited:
                subgraph.edges[edge.id] = Edge.from_dict(edge.to_dict())
                subgraph._index_edge(subgraph.edges[edge.id])
        
        return subgraph
    
    # ========================================================================
    # Reasoning Operations
    # ========================================================================
    
    def find_contradictions(self, active_only: bool = True) -> list[tuple[Node, Node, str]]:
        """
        Find contradictory nodes in the graph.
        
        Args:
            active_only: Only consider active nodes (default True)
        
        Returns:
            List of (node1, node2, explanation) tuples
        """
        contradictions = []
        
        # Check explicit contradiction edges
        for edge_id in self._edges_by_relation.get(RelationType.CONTRADICTS, set()):
            edge = self.edges[edge_id]
            source = self.nodes.get(edge.source_id)
            target = self.nodes.get(edge.target_id)
            if source and target:
                # Skip if either node is inactive
                if active_only and (source.status != NodeStatus.ACTIVE or target.status != NodeStatus.ACTIVE):
                    continue
                contradictions.append((
                    source,
                    target,
                    f"Explicit contradiction: {source.label} contradicts {target.label}"
                ))
        
        # Check refutation edges
        for edge_id in self._edges_by_relation.get(RelationType.REFUTES, set()):
            edge = self.edges[edge_id]
            source = self.nodes.get(edge.source_id)
            target = self.nodes.get(edge.target_id)
            if source and target:
                # Skip if either node is inactive
                if active_only and (source.status != NodeStatus.ACTIVE or target.status != NodeStatus.ACTIVE):
                    continue
                contradictions.append((
                    source,
                    target,
                    f"Refutation: {source.label} refutes {target.label}"
                ))
        
        return contradictions
    
    def infer_relations(self, active_only: bool = True) -> list[Edge]:
        """
        Infer new relations based on existing patterns.
        
        Args:
            active_only: Only consider active nodes for inference (default True)
        
        Returns:
            List of inferred edges (not yet added to graph)
        """
        inferred = []
        
        def is_node_active(node_id: str) -> bool:
            """Helper to check if a node is active."""
            node = self.nodes.get(node_id)
            return node and (not active_only or node.status == NodeStatus.ACTIVE)
        
        # Transitivity for IS_A relations
        # If A is_a B and B is_a C, then A is_a C
        for edge1_id in self._edges_by_relation.get(RelationType.IS_A, set()):
            edge1 = self.edges[edge1_id]
            # Skip if source or target nodes are inactive
            if not is_node_active(edge1.source_id) or not is_node_active(edge1.target_id):
                continue
            for edge2_id in self._edges_by_source.get(edge1.target_id, set()):
                edge2 = self.edges[edge2_id]
                if edge2.relation == RelationType.IS_A:
                    # Skip if target node is inactive
                    if not is_node_active(edge2.target_id):
                        continue
                    # Check if this edge already exists
                    existing = self.get_edges_between(
                        edge1.source_id,
                        edge2.target_id,
                        RelationType.IS_A
                    )
                    if not existing:
                        inferred.append(Edge(
                            id=self._generate_id(f"inferred:{edge1.source_id}:is_a:{edge2.target_id}"),
                            source_id=edge1.source_id,
                            target_id=edge2.target_id,
                            relation=RelationType.IS_A,
                            properties={"inferred": True, "reason": "transitivity"},
                            confidence=min(edge1.confidence, edge2.confidence) * 0.9
                        ))
        
        # If A causes B and B causes C, A indirectly causes C
        for edge1_id in self._edges_by_relation.get(RelationType.CAUSES, set()):
            edge1 = self.edges[edge1_id]
            # Skip if source or target nodes are inactive
            if not is_node_active(edge1.source_id) or not is_node_active(edge1.target_id):
                continue
            for edge2_id in self._edges_by_source.get(edge1.target_id, set()):
                edge2 = self.edges[edge2_id]
                if edge2.relation == RelationType.CAUSES:
                    # Skip if target node is inactive
                    if not is_node_active(edge2.target_id):
                        continue
                    existing = self.get_edges_between(
                        edge1.source_id,
                        edge2.target_id,
                        RelationType.CAUSES
                    )
                    if not existing:
                        inferred.append(Edge(
                            id=self._generate_id(f"inferred:{edge1.source_id}:causes:{edge2.target_id}"),
                            source_id=edge1.source_id,
                            target_id=edge2.target_id,
                            relation=RelationType.CAUSES,
                            properties={"inferred": True, "reason": "causal_chain"},
                            confidence=min(edge1.confidence, edge2.confidence) * 0.8
                        ))
        
        # Run custom inference rules
        for rule in self._inference_rules:
            inferred.extend(rule(self))
        
        return inferred
    
    def apply_inferences(self) -> int:
        """Apply all inferred relations to the graph."""
        inferred = self.infer_relations()
        count = 0
        for edge in inferred:
            self.edges[edge.id] = edge
            self._index_edge(edge)
            count += 1
        if count > 0:
            self._update_metadata()
        return count
    
    def add_inference_rule(self, rule: Callable[["SemanticGraph"], list[Edge]]):
        """Add a custom inference rule."""
        self._inference_rules.append(rule)
    
    def get_support_chain(self, node_id: str, active_only: bool = True) -> list[tuple[Node, Edge]]:
        """
        Get the chain of evidence supporting a node.
        
        Args:
            node_id: The node ID to get support chain for
            active_only: Only include active supporting nodes (default True)
        
        Returns:
            List of (supporting_node, edge) tuples
        """
        chain = []
        visited = set()
        
        def traverse(current_id: str):
            if current_id in visited:
                return
            visited.add(current_id)
            
            for edge_id in self._edges_by_target.get(current_id, set()):
                edge = self.edges[edge_id]
                if edge.relation in (
                    RelationType.SUPPORTS,
                    RelationType.EVIDENCED_BY,
                    RelationType.BASED_ON,
                    RelationType.DERIVED_FROM
                ):
                    source = self.nodes.get(edge.source_id)
                    if source:
                        # Skip inactive nodes
                        if active_only and source.status != NodeStatus.ACTIVE:
                            continue
                        chain.append((source, edge))
                        traverse(edge.source_id)
        
        traverse(node_id)
        return chain
    
    def calculate_confidence(self, node_id: str, active_only: bool = True) -> float:
        """
        Calculate aggregate confidence for a node based on supporting evidence.
        
        Args:
            node_id: The node ID to calculate confidence for
            active_only: Only consider active nodes (default True)
        
        Returns:
            Calculated confidence score
        """
        node = self.nodes.get(node_id)
        if not node:
            return 0.0
        
        base_confidence = node.confidence
        support_chain = self.get_support_chain(node_id, active_only=active_only)
        
        if not support_chain:
            return base_confidence
        
        # Aggregate support
        support_score = sum(
            n.confidence * e.confidence * e.weight
            for n, e in support_chain
        ) / len(support_chain)
        
        # Check for contradictions (only from active nodes)
        contradiction_penalty = 0.0
        for edge_id in self._edges_by_target.get(node_id, set()):
            edge = self.edges[edge_id]
            if edge.relation in (RelationType.CONTRADICTS, RelationType.REFUTES):
                source = self.nodes.get(edge.source_id)
                if source:
                    # Skip inactive source nodes
                    if active_only and source.status != NodeStatus.ACTIVE:
                        continue
                    contradiction_penalty += source.confidence * 0.5
        
        final_confidence = min(1.0, max(0.0,
            (base_confidence + support_score) / 2 - contradiction_penalty
        ))
        
        return final_confidence
    
    # ========================================================================
    # Convenience Methods for Common Operations
    # ========================================================================
    
    def add_fact(
        self,
        label: str,
        content: str,
        confidence: float = 0.8,
        source: str = "observation",
        **properties
    ) -> Node:
        """Add a fact to the graph."""
        return self.add_node(
            NodeType.FACT, label, content,
            properties=properties, confidence=confidence, source=source
        )
    
    def add_concept(
        self,
        label: str,
        definition: str,
        **properties
    ) -> Node:
        """Add a concept to the graph."""
        return self.add_node(
            NodeType.CONCEPT, label, definition,
            properties=properties
        )
    
    def add_task(
        self,
        label: str,
        description: str,
        priority: int = 5,
        status: str = "pending",
        **properties
    ) -> Node:
        """Add a task to the graph."""
        props = {"priority": priority, "task_status": status, **properties}
        return self.add_node(
            NodeType.TASK, label, description,
            properties=props
        )
    
    def add_plan(
        self,
        label: str,
        description: str,
        tasks: Optional[list[str]] = None,
        **properties
    ) -> Node:
        """Add a plan to the graph."""
        plan = self.add_node(
            NodeType.PLAN, label, description,
            properties=properties
        )
        
        # Link tasks to plan
        if tasks:
            for task_id in tasks:
                self.add_edge(plan.id, task_id, RelationType.HAS_PART)
        
        return plan
    
    def add_tool(
        self,
        name: str,
        description: str,
        capabilities: Optional[list[str]] = None,
        **properties
    ) -> Node:
        """Add a tool to the graph."""
        props = {"capabilities": capabilities or [], **properties}
        return self.add_node(
            NodeType.TOOL, name, description,
            properties=props
        )
    
    def add_conclusion(
        self,
        label: str,
        content: str,
        evidence_ids: Optional[list[str]] = None,
        confidence: float = 0.7,
        **properties
    ) -> Node:
        """Add a conclusion with supporting evidence."""
        conclusion = self.add_node(
            NodeType.CONCLUSION, label, content,
            properties=properties, confidence=confidence
        )
        
        # Link evidence
        if evidence_ids:
            for evidence_id in evidence_ids:
                self.add_edge(
                    conclusion.id, evidence_id,
                    RelationType.EVIDENCED_BY
                )
        
        return conclusion
    
    def add_emotion(
        self,
        emotion: str,
        intensity: float = 0.5,
        trigger: Optional[str] = None,
        trigger_id: Optional[str] = None,
        **properties
    ) -> Node:
        """Add an emotional state."""
        props = {"intensity": intensity, "trigger": trigger, **properties}
        emotion_node = self.add_node(
            NodeType.EMOTION, emotion, f"Emotional state: {emotion}",
            properties=props
        )
        
        if trigger_id:
            self.add_edge(trigger_id, emotion_node.id, RelationType.TRIGGERS)
        
        return emotion_node
    
    def link(
        self,
        source: str | Node,
        relation: RelationType,
        target: str | Node,
        **properties
    ) -> Optional[Edge]:
        """Convenience method to link two nodes."""
        source_id = source.id if isinstance(source, Node) else source
        target_id = target.id if isinstance(target, Node) else target
        return self.add_edge(source_id, target_id, relation, properties=properties)
    
    # ========================================================================
    # Persistence
    # ========================================================================
    
    def save(self, filepath: str):
        """Save the graph to a JSON file."""
        data = {
            "name": self.name,
            "metadata": self.metadata,
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(CompactEmbeddingEncoder(indent=2, ensure_ascii=False).encode(data))
    
    @classmethod
    def load(cls, filepath: str) -> "SemanticGraph":
        """Load a graph from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        graph = cls(name=data.get("name", "loaded_graph"))
        graph.metadata = data.get("metadata", graph.metadata)
        
        # Load nodes
        for node_data in data.get("nodes", []):
            node = Node.from_dict(node_data)
            graph.nodes[node.id] = node
            graph._index_node(node)
        
        # Load edges
        for edge_data in data.get("edges", []):
            edge = Edge.from_dict(edge_data)
            graph.edges[edge.id] = edge
            graph._index_edge(edge)
        
        return graph
    
    def to_dict(self) -> dict:
        """Convert graph to dictionary."""
        return {
            "name": self.name,
            "metadata": self.metadata,
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()]
        }
    
    # ========================================================================
    # Statistics
    # ========================================================================
    
    def stats(self, active_only: bool = False) -> dict:
        """Get graph statistics.
        
        Args:
            active_only: If True, only count active nodes (default False for backward compatibility)
        """
        type_counts = defaultdict(int)
        status_counts = defaultdict(int)
        active_nodes = []
        
        for node in self.nodes.values():
            status_counts[node.status.value] += 1
            if not active_only or node.status == NodeStatus.ACTIVE:
                type_counts[node.type.value] += 1
                active_nodes.append(node)
        
        relation_counts = defaultdict(int)
        for edge in self.edges.values():
            relation_counts[edge.relation.value] += 1
        
        nodes_for_avg = active_nodes if active_only else list(self.nodes.values())
        
        result = {
            "total_nodes": len(self.nodes),
            "active_nodes": status_counts.get("active", 0),
            "total_edges": len(self.edges),
            "nodes_by_type": dict(type_counts),
            "nodes_by_status": dict(status_counts),
            "edges_by_relation": dict(relation_counts),
            "avg_confidence": sum(n.confidence for n in nodes_for_avg) / max(1, len(nodes_for_avg))
        }
        
        return result
    
    def __repr__(self) -> str:
        return f"SemanticGraph(name={self.name}, nodes={len(self.nodes)}, edges={len(self.edges)})"
    
    # ========================================================================
    # Private Helper Methods
    # ========================================================================
    
    def _generate_id(self, seed: str) -> str:
        """Generate a unique ID."""
        timestamp = datetime.now().isoformat()
        unique = f"{seed}:{timestamp}:{uuid.uuid4().hex[:8]}"
        return hashlib.md5(unique.encode()).hexdigest()[:16]
    
    def _index_node(self, node: Node):
        """Add node to indexes."""
        self._nodes_by_type[node.type].add(node.id)
        self._nodes_by_label[node.label.lower()].add(node.id)
    
    def _unindex_node(self, node: Node):
        """Remove node from indexes."""
        self._nodes_by_type[node.type].discard(node.id)
        self._nodes_by_label[node.label.lower()].discard(node.id)
    
    def _index_edge(self, edge: Edge):
        """Add edge to indexes."""
        self._edges_by_source[edge.source_id].add(edge.id)
        self._edges_by_target[edge.target_id].add(edge.id)
        self._edges_by_relation[edge.relation].add(edge.id)
    
    def _unindex_edge(self, edge: Edge):
        """Remove edge from indexes."""
        self._edges_by_source[edge.source_id].discard(edge.id)
        self._edges_by_target[edge.target_id].discard(edge.id)
        self._edges_by_relation[edge.relation].discard(edge.id)
    
    def _update_metadata(self):
        """Update graph metadata."""
        self.metadata["updated_at"] = datetime.now().isoformat()
        self.metadata["version"] += 1


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Create a new semantic graph
    graph = SemanticGraph("brain_knowledge")
    
    # Add some concepts
    python = graph.add_concept("Python", "A high-level programming language")
    programming = graph.add_concept("Programming", "The process of writing computer code")
    ai = graph.add_concept("AI", "Artificial Intelligence - machines that mimic cognitive functions")
    
    # Add relationships
    graph.link(python, RelationType.IS_A, programming)
    graph.link(ai, RelationType.USES, programming)
    
    # Add a fact
    fact1 = graph.add_fact(
        "Python is popular",
        "Python is one of the most popular programming languages in 2024",
        source="observation"
    )
    graph.link(fact1, RelationType.ABOUT, python)
    
    # Add a task
    task1 = graph.add_task(
        "Learn Python",
        "Complete a Python tutorial",
        priority=8
    )
    graph.link(task1, RelationType.RELATED_TO, python)
    
    # Add an emotion
    curiosity = graph.add_emotion("curiosity", intensity=0.8, trigger="learning new things")
    graph.link(curiosity, RelationType.ASSOCIATED_WITH, ai)
    
    # Add a conclusion with evidence
    conclusion = graph.add_conclusion(
        "Python is good for AI",
        "Python is well-suited for AI development due to its simplicity and library ecosystem",
        evidence_ids=[fact1.id]
    )
    
    # Query the graph
    print("=== Graph Stats ===")
    print(graph.stats())
    
    print("\n=== All Concepts ===")
    for concept in graph.get_nodes_by_type(NodeType.CONCEPT):
        print(f"  - {concept.label}: {concept.content}")
    
    print("\n=== Search for 'Python' ===")
    results = graph.search("Python")
    for node in results:
        print(f"  - [{node.type.value}] {node.label}")
    
    print("\n=== Neighbors of Python ===")
    for neighbor in graph.get_neighbors(python.id):
        print(f"  - {neighbor.label} ({neighbor.type.value})")
    
    # Save the graph
    graph.save("knowledge_graph.json")
    print("\n=== Graph saved to knowledge_graph.json ===")
