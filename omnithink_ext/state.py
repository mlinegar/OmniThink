from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set

@dataclass
class Evidence:
    id: int
    url: str
    title: str
    snippet: str
    span: Tuple[int, int]            # character offsets within cached page text
    passage_hash: str

@dataclass
class Node:
    id: str
    parent: Optional[str]
    title: str
    query: str
    materialized: bool = True        # allows concept-only nodes
    evidence_ids: List[int] = field(default_factory=list)
    claims: List[str] = field(default_factory=list)  # distilled from pool
    node_type: str = "concept"       # tau(v)
    depth: int = 0

@dataclass
class Edge:
    parent: str
    child: str
    kind: str = "expands"            # rho(p->c)

@dataclass
class InfoTree:
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    evidences: Dict[int, Evidence] = field(default_factory=dict)

    def add_child(self, parent: str, child: Node, kind: str = "expands"):
        self.nodes[child.id] = child
        self.edges.append(Edge(parent=parent, child=child.id, kind=kind))

    def attach_evidence(self, node_id: str, ev: Evidence):
        self.evidences[ev.id] = ev
        self.nodes[node_id].evidence_ids.append(ev.id)
