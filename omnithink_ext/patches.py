"""
Non-invasive shims around the repo’s existing DSPy pipeline.

Hook 1: after expansion, for each attached child Sj call canon.update_with_new_node(...) and compat.check(...).
Hook 2: after reflection, parse the insights to node-level claims C(v).
Hook 3: after writing/polish, parse inline citations [1][2]... into ledger φ and resolve to evidence ids.

These hooks keep Figure 3 and Algorithm 2 intact; they add typed state required by our audits and labeler.  [oai_citation:23‡OmniThink.pdf](file-service://file-5ZNd7ZBEcYkbc9RdjrkKwE)
"""
