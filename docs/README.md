## What this fork adds (in one paragraph)

We keep OmniThink’s three stages—Information Acquisition via expand/reflect, Concept-guided Outline Structuring, and Article Composition with per-section retrieval and inline citations—exactly as specified in Figure 3, Equations (1)–(2), Algorithm 2, and Listings 1–5. We add a minimal typed state and three hooks: after subnodes are added during expansion we canonicalize nodes (route equivalence) and record typed edges; after reflection we attach distilled claims to the leafs that yielded them; after writing we parse bracketed citations into a ledger that maps sentences to evidence. Post‑hoc, we: (i) contract the tree into a knowledge graph along canonical classes; (ii) audit route‑consistency and edge‑compatibility; (iii) label each canonical node for a downstream schema (e.g., party manifesto categories or stances) and produce a minimal justification trace with cited passages. No prompts or control flow are changed, and you can still run the paper’s stages verbatim.  [oai_citation:2‡OmniThink.pdf](file-service://file-5ZNd7ZBEcYkbc9RdjrkKwE)

## Where to look in the paper and repo

The expand/reflect alternation and the Combine/Merge algebra appear in §3.1 with Equations (1)–(2); the controller loop is Algorithm 2 (Appendix L). The outline and polish prompts are in §3.2 and Listing 3; the section writer and final editor with inline citations are §3.3 and Listings 4–5. We reuse these modules and splice only at the specific lines indicated in `docs/THINK_CHANGES.md`.  [oai_citation:3‡OmniThink.pdf](file-service://file-5ZNd7ZBEcYkbc9RdjrkKwE)

## Deliverables from a run

A normal OmniThink article plus: `run_artifacts/state.json` (typed tree and pool), `run_artifacts/ledger.jsonl` (sentence→evidence map), `run_artifacts/graph.graphml` (contracted knowledge graph), `run_artifacts/labels.json` (per‑node label distributions), and `run_artifacts/justifications.jsonl` (minimal explanation traces). KD and information‑diversity scores are computed post‑hoc from the draft using the paper’s definitions for comparability.  [oai_citation:4‡OmniThink.pdf](file-service://file-5ZNd7ZBEcYkbc9RdjrkKwE)
