## Minimal integration plan

We splice into OmniThink at three points and never change the prompts or loop. First, immediately after each `Add Sj to T_{m+1}` in Algorithm 2 we assign \(\tau\) and \(\rho=\textsf{expands}\) to the new edge, run incremental canonicalization over the new node against existing nodes to maintain \(\kappa\), and record the mapping. Second, immediately after `Update Conceptual Pool P_{m+1} ← Merge(Im+1,Pm)` we attach distilled claims in \(I_{m+1}\) to the leafs that yielded them so each node maintains \(C(v)\). Third, after the writer emits sections with inline citations (Listings 4–5), we parse bracketed numerals into the ledger \(\phi\), resolve evidence IDs to URLs and passage spans, and write `ledger.jsonl`. These steps are bookkeeping only; the control flow remains Figure 3 and Algorithm 2.  [oai_citation:11‡OmniThink.pdf](file-service://file-5ZNd7ZBEcYkbc9RdjrkKwE)

## Pseudocode matching the paper’s lines

```python
# Pseudocode only. Real code lives in omnithink_ext/*.py

def run_expand_reflect(topic, K):
    T, P = init_tree_and_pool(topic)                  # §3.1 init.  [oai_citation:12‡OmniThink.pdf](file-service://file-5ZNd7ZBEcYkbc9RdjrkKwE)
    for m in range(K):
        Lm = leaves(T)
        for Ni in Lm:
            if needs_expansion(P, Ni):               # Algorithm 2, line 10.  [oai_citation:13‡OmniThink.pdf](file-service://file-5ZNd7ZBEcYkbc9RdjrkKwE)
                subs = gen_subnodes(P, Ni)           # Listing 1 (ExtendConcept).  [oai_citation:14‡OmniThink.pdf](file-service://file-5ZNd7ZBEcYkbc9RdjrkKwE)
                for Sj in subs:
                    retrieve_and_attach(T, Ni, Sj)   # Algorithm 2, lines 14–15.  [oai_citation:15‡OmniThink.pdf](file-service://file-5ZNd7ZBEcYkbc9RdjrkKwE)
                    assign_types(T, parent=Ni, child=Sj, rho="expands")
                    canon_update(T, node=Sj, lambda_=0.5, theta=0.75)     # defines κ
                    assert edge_compat_ok(T, Ni, Sj)                      # uses configs/compat_schema.yaml
        Im1 = reflect_new_leaves(T)                    # Listing 2 (GenConcept).  [oai_citation:16‡OmniThink.pdf](file-service://file-5ZNd7ZBEcYkbc9RdjrkKwE)
        P = merge_into_pool(P, Im1)                    # Equation (2).  [oai_citation:17‡OmniThink.pdf](file-service://file-5ZNd7ZBEcYkbc9RdjrkKwE)
        attach_claims_from_insights(T, Im1)            # populates C(v) for leaves that yielded insights
        if sufficient_information(T, P):               # Algorithm 2, line 22.  [oai_citation:18‡OmniThink.pdf](file-service://file-5ZNd7ZBEcYkbc9RdjrkKwE)
            break
    return T, P

def run_write_and_polish(T, P, outline):
    sections = []
    for sec in outline.sections:
        info = retrieve_topK_from_tree(T, sec)         # §3.3, SBERT similarity.  [oai_citation:19‡OmniThink.pdf](file-service://file-5ZNd7ZBEcYkbc9RdjrkKwE)
        s_out = write_section(info, sec)               # Listing 4.  [oai_citation:20‡OmniThink.pdf](file-service://file-5ZNd7ZBEcYkbc9RdjrkKwE)
        ledger.record(s_out)                           # parses [1][2]... into φ
        sections.append(s_out)
    draft = concat(sections)
    final = polish(draft)                              # Listing 5. Update ledger numbering.  [oai_citation:21‡OmniThink.pdf](file-service://file-5ZNd7ZBEcYkbc9RdjrkKwE)
    ledger.synchronize(final)
    return final

Concept-only branches (optional)

If you wish to grow the “tree independent of documents,” allow retrieve_and_attach to create Node(materialized=False) with a taxonomy child produced by Listing 1, and defer retrieval until a later pass. Reflection still consumes the pool to prioritize which unmaterialized leaves to materialize next. This keeps Combine/Merge unchanged while separating concept expansion from evidence acquisition.  

---

```yaml
# configs/compat_schema.yaml

node_types:
  - topic
  - concept
  - entity
  - claim_cluster
edge_kinds:
  - expands
  - is_a
  - part_of
  - addresses_question
  - perspective_of

# Admissible (tau(parent), rho, tau(child)) triples.
compatibility:
  - [topic,   expands, concept]
  - [concept, expands, concept]
  - [concept, is_a,    concept]
  - [concept, part_of, concept]
  - [concept, addresses_question, concept]
  - [concept, perspective_of, concept]
  - [concept, expands, entity]
  - [entity,  is_a,    entity]
