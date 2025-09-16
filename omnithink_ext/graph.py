from collections import defaultdict
from typing import Dict, List, Tuple, Set

class GraphExporter:
    """
    Builds G=(𝒞, Ē ∪ E_P) by contracting T along κ and adding pool-derived supports/contradicts edges.
    See docs/MATH.md (Knowledge graph export).
    """
    def __init__(self, canonicalizer, pool):
        self.kappa = canonicalizer
        self.pool = pool

    def export(self, T):
        C_nodes: Dict[str, Dict] = {}
        E_edges: List[Tuple[str,str,str]] = []

        # nodes
        for v_id, v in T.nodes.items():
            c_id = self.kappa.find(v_id)
            b = C_nodes.setdefault(c_id, dict(claims=[], members=[]))
            b["claims"].extend(v.claims)
            b["members"].append(v_id)

        # edges from tree
        for e in T.edges:
            cp, cc = self.kappa.find(e.parent), self.kappa.find(e.child)
            if cp != cc:
                E_edges.append((cp, cc, e.kind))

        # pool-derived supports/contradicts
        E_extra = self._edges_from_pool()

        return C_nodes, E_edges + E_extra

    def _edges_from_pool(self):
        # toy: connect classes that co-occur in an insight with status
        out = []
        for ins in self.pool.insights:
            status = getattr(ins, "status", "fact")  # hypothesis, fact, contradiction
            nodes = [self.kappa.find(nid) for nid in ins.support_node_ids]
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    kind = "supports" if status != "contradiction" else "contradicts"
                    out.append((nodes[i], nodes[j], kind))
        return out
