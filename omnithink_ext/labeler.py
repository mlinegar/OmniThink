import math
import re
from typing import Dict, List, Tuple

class Labeler:
    """
    L: 𝒞 -> Δ(𝒴). Scores S_y(c) = α Σ_x w(x,y) + β Σ_(c->u) ρ_w(ρ) Σ_x w(x,y) + γ Σ_s u(s,y).
    Emits distribution and a minimal justification trace via greedy cover. See docs/MATH.md (Labeler).
    """
    def __init__(self, schema, lexicon, rel_weights, alpha, beta, gamma):
        self.labels = schema
        self.lex = lexicon
        self.rw = rel_weights
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def score_node(self, G_nodes, G_edges, c_id, sentence_index, cite_map):
        scores = {y: 0.0 for y in self.labels}
        # local claims
        for y in self.labels:
            scores[y] += self.alpha * sum(self._w(x, y) for x in G_nodes[c_id]["claims"])
        # neighbors
        nbrs = [(v, rho) for (u,v,rho) in G_edges if u == c_id]
        for (u_id, rho) in nbrs:
            for y in self.labels:
                bump = sum(self._w(x, y) for x in G_nodes[u_id]["claims"])
                scores[y] += self.beta * self.rw.get(rho, 0.0) * bump
        # citing sentences
        citing = sentence_index.get(c_id, [])
        for y in self.labels:
            scores[y] += self.gamma * sum(self._u(sentence_index[s], y, cite_map) for s in citing)
        return self._softmax(scores)

    def _w(self, claim: str, y: str) -> float:
        toks = set(re.findall(r"\w+", claim.lower()))
        return 1.0 if any(k.lower() in toks for k in self.lex.get(y, [])) else 0.0

    def _u(self, sent: str, y: str, cite_map) -> float:
        # light bump if label lexeme appears in a sentence that cites the node
        toks = set(re.findall(r"\w+", sent.lower()))
        return 0.5 if any(k.lower() in toks for k in self.lex.get(y, [])) else 0.0

    def _softmax(self, scores: Dict[str, float]) -> Dict[str, float]:
        import math
        vals = list(scores.values())
        m = max(vals) if vals else 0.0
        exps = {k: math.exp(v - m) for k, v in scores.items()}
        Z = sum(exps.values()) or 1.0
        return {k: v / Z for k, v in exps.items()}
