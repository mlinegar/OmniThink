import math
from typing import Dict, Set, Iterable
from collections import defaultdict

class Canonicalizer:
    """
    Maintains κ: node_id -> canonical_id using a union-find.
    Similarity sim(u,v) = λ cos(e(title_u), e(title_v)) + (1-λ) Jaccard(ngrams(claims_u), ngrams(claims_v)).
    See docs/MATH.md (Route equivalence).
    """
    def __init__(self, embedder, lambda_=0.5, theta=0.75, ngram_n=2):
        self.parent: Dict[str, str] = {}
        self.embedder = embedder
        self.lambda_ = lambda_
        self.theta = theta
        self.ngram_n = ngram_n

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def update_with_new_node(self, T, new_id: str):
        t_new, c_new = T.nodes[new_id].title, T.nodes[new_id].claims
        e_new = self.embedder(t_new)
        for old_id in T.nodes.keys():
            if old_id == new_id: 
                continue
            score = self.sim(T, e_new, c_new, old_id)
            if score >= self.theta:
                self.union(new_id, old_id)

    def sim(self, T, e_new, c_new, old_id: str) -> float:
        from omnithink_ext.utils import cosine, jaccard_ngrams
        e_old = self.embedder(T.nodes[old_id].title)
        cos = cosine(e_new, e_old)
        jac = jaccard_ngrams(c_new, T.nodes[old_id].claims, n=self.ngram_n)
        return self.lambda_ * cos + (1 - self.lambda_) * jac
