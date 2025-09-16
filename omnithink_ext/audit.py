from math import log
from typing import Dict, List, Tuple

def kl(p: Dict[str,float], q: Dict[str,float]) -> float:
    eps = 1e-9
    return sum(pi * (log(pi+eps) - log(q.get(k,eps))) for k, pi in p.items())

def route_consistency(G_nodes, labels, kappa, kl_thresh: float) -> List[Tuple[str,str,float]]:
    """
    Returns canonical pairs (c,c') with KL divergence > threshold, indicating route-dependent labeling drift.
    """
    pairs = []
    C = list(G_nodes.keys())
    for i in range(len(C)):
        for j in range(i+1, len(C)):
            c, c2 = C[i], C[j]
            d = kl(labels[c], labels[c2])
            if d > kl_thresh:
                pairs.append((c, c2, d))
    return pairs

def monotonicity(G_edges, labels, eta: float, delta: float) -> List[Tuple[str,str,str]]:
    """
    Checks heritable labels along is_a/part_of: if L(p)(y) >= eta then L(c)(y) >= eta-delta.
    """
    violations = []
    for (p,c,rho) in G_edges:
        if rho in ("is_a","part_of"):
            for y, pv in labels[p].items():
                if pv >= eta and labels[c].get(y,0.0) < eta - delta:
                    violations.append((p,c,y))
    return violations
