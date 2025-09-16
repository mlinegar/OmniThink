# Issue: Stabilize `omnithink_ext` integration

## Context
The OmniThink extension landed with promising scaffolding (typed state dataclasses, a ledger draft, and a labeler stub), but none of the plumbing is wired into the core pipeline yet. Attempting to exercise the new code with a manifesto-sized document immediately surfaces hard failures that block end-to-end use and prevent the math described in `docs/MATH.md` from being realized in practice.

### Current blockers
- **Canonicalization crashes on import.** `canon.py` references utility functions that were never committed, so any call into the canonicalizer raises `ModuleNotFoundError` before similarity scores are computed.
- **Exporter expects nonexistent fields.** The graph exporter assumes insight objects expose `support_node_ids` and `status`, but those attributes are absent from every pool entry we produce today, so runtime errors are guaranteed once the exporter is invoked.
- **Labeler API bugs and missing features.** `Labeler.score_node` reuses sentence identifiers as keys into `sentence_index`, which raises `KeyError` unless callers duplicate their inputs. The implementation also ignores stance dimensions, so we cannot emit the “final output modes” described in the requirements or surface minimal justification traces.
- **Audit math not enforced.** Route-consistency checks compare Kullback–Leibler divergence across all canonical nodes instead of restricting comparisons to ancestor-related paths, violating the guarantees written in `docs/MATH.md`.
- **Pipeline hooks are stubs.** `patches.py` never writes the promised run artifacts, ledgers, or compatibility checks, so there is no state on disk for auditors or downstream tools to inspect.
- **Human-review workflow missing.** There is no way to randomly sample nodes, sentences, or label decisions, and no configuration to change sampling granularity through the sentence-transformer settings.
- **Ledger lacks provenance.** Without resolved evidence objects and node back-pointers, auditors cannot trace how individual labels were produced.

These gaps collectively mean we cannot take a manifesto, run it through the fork, and deliver the audited, sampleable label outputs we promised stakeholders.

## Desired outcome
Produce an end-to-end workflow that ingests long-form documents (e.g., party manifestos), builds canonical nodes, generates labels with justification traces, writes the advertised run artifacts, and enables randomized human review over documents, sentences, and label decisions.

## TODO / Plan of record
### Canonicalization and data plumbing
- [ ] Restore the missing similarity utilities (cosine + Jaccard) so `canon.py` can execute.
- [ ] Add unit tests that merge/deduplicate nodes via canonicalization to prevent regressions.
- [ ] Align the graph exporter with the actual pool schema (or extend the schema) so `_edges_from_pool` no longer references nonexistent fields.

### Labeling & justification
- [ ] Implement justification-trace generation (e.g., greedy set cover) so we can emit minimal supporting evidence for each label.
- [ ] Fix the `Labeler` sentence-index API, respect multi-dimensional schemas (issue × stance), and surface stance-specific “final output modes.”
- [ ] Document a runnable example that labels a manifesto and reports the justification trace for at least one node.

### Auditing & artifacts
- [ ] Update the audit routines to respect the ancestor constraint for KL comparisons and cover the behavior with regression tests.
- [ ] Wire `patches.py` into the OmniThink pipeline so that state snapshots, ledgers, compatibility checks, and promised `run_artifacts/*.json` files are actually written.
- [ ] Enrich the ledger with provenance pointers (resolved evidence, node references) to support manual audits.

### Human review & sampling
- [ ] Implement random sampling utilities across nodes, sentences, and label decisions, with configuration to adjust granularity (e.g., via sentence-transformer thresholds or chunk sizes).
- [ ] Expose a CLI or scripting hook that writes sampled items to disk so reviewers can inspect them without touching internal state.

## Definition of done
- All TODO items above are implemented and verified by automated tests where applicable.
- CI passes across the repository.
- Documentation ships with a step-by-step tutorial that demonstrates successful labeling, auditing, and sampling on a realistic input (e.g., a manifesto).
