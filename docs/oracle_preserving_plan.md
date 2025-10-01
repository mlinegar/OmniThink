# Oracle-Preserving Hierarchical Summarization Integration Plan

This document enumerates the minimal code and configuration adjustments required to integrate oracle-preserving hierarchical summarization with human-guided sampling into OmniThink, while keeping the existing architecture (information tree, conceptual pool, and iterative expansion/reflection loops) intact. Each change below is justified as necessary to achieve the desired behavior or instrumentation.

## 1. Architectural Assessment

* **Tree construction and exploration** are handled by `MindMap`, `MindPoint`, and the supporting DSPy predictor modules in `src/tools/mindmap.py`. The generator currently performs autonomous branching via `MindPoint.extend()` within `MindMap.build_map()` and `MindMap.recursive_extend()`, without any oracle gating or external sampling hooks.【F:src/tools/mindmap.py†L1-L135】
* **Information storage** (conceptual pool) is implicit in each `MindPoint`'s `concept`/`info` fields; cross-node merging logic is absent, so adding merge compatibility checks will require new instrumentation around child expansion and aggregation phases.【F:src/tools/mindmap.py†L96-L129】
* **Iteration loops** (expansion/reflection) are orchestrated primarily through repeated calls to `MindPoint.extend()` and the DSPy signatures `GenConcept` and `ExtendConcept`, ensuring that any new oracle must integrate cleanly with DSPy contexts to preserve training compatibility.【F:src/tools/mindmap.py†L19-L93】

## 2. Oracle Check Integration (Leaf Sufficiency, Merge Compatibility, Stability)

### 2.1 Oracle Abstraction Layer
* **Add** a dedicated module `src/tools/oracles.py` exposing lightweight callable classes (e.g., `LeafSufficiencyOracle`, `MergeCompatibilityOracle`, `StabilityOracle`). Creating a new module, rather than embedding logic directly in `mindmap.py`, keeps the current tree code unchanged unless the oracle functionality is explicitly enabled and ensures future extensibility without touching the DSPy predictors.
* Each oracle class will accept the minimal context (node concepts, supporting snippets, and optional parent/child metadata) and return structured verdicts with scores/justifications. This encapsulation is necessary so that training (including DPO data capture) can reuse the same logic without duplicating code paths.

### 2.2 Minimal Hooks in Tree Expansion
* **Inject optional oracle callbacks** into `MindPoint` via dependency injection. Extend the constructor signature with an optional `oracles` mapping, defaulting to `None`, and store it on the instance. This is the only structural change required in `mindmap.py` to access oracles during `extend()`; when not supplied, the method will behave exactly as today.
* **Modify `MindPoint.extend()`** to: (1) call the leaf sufficiency oracle before generating new categories, exiting early when the oracle declares a node complete; (2) evaluate merge compatibility after generating `new_concept` and before committing `new_node`, potentially merging results or vetoing the addition; and (3) emit stability signals that can be logged for DPO. Each hook must be guarded behind presence checks to avoid altering default behavior when oracles are unused.
* **Add a lightweight event emission method** (e.g., `_emit_oracle_event`) inside `MindPoint` for logging decisions. This will record oracle inputs/outputs but defer persistence to the new instrumentation module (Section 4), thereby avoiding intrusive changes throughout the codebase.

### 2.3 Oracle Configuration
* **Extend existing configuration files** (`config.yaml` and relevant templates in `configs/`) with optional oracle settings (`enabled`, thresholds, LM for oracle prompts). Using configuration flags ensures oracle behavior can be toggled without touching code, and preserves backward compatibility for existing users.

## 3. Human-Guided Sampling During Tree Construction

### 3.1 Sampling Interface Layer
* **Create** a narrow interface `HumanSampler` in a new module `src/tools/sampling.py` that exposes blocking and non-blocking selection methods (e.g., CLI prompt, callback hook). Implementations can defer to the console, HTTP APIs, or scripted decisions, but the default will be a no-op auto-accept to avoid impacting current automation flows.

### 3.2 Integration with MindMap
* **Augment `MindMap` initialization** to accept an optional `sampler` object. Within `build_map()`, before enqueuing child nodes, call the sampler to approve/reorder candidate expansions. This modification is required to allow human-guided sampling to influence the tree without rewriting the expansion loop. All sampler invocations will be conditional on sampler availability to avoid altering default behavior.
* **Emit sampling events** using the same `_emit_oracle_event` helper so the DPO pipeline can reuse the logged interactions.

## 4. Instrumentation for Auditability and DPO Training

### 4.1 Centralized Decision Logger
* **Introduce** `src/utils/decision_logging.py` defining a thin `DecisionLogger` with pluggable sinks (JSONL, in-memory buffer). `MindPoint` will depend only on an abstract `record(event)` method to remain decoupled from storage specifics.
* **Modify configuration** to allow enabling logging and selecting the sink path. Defaults remain disabled to prevent overhead when unused.

### 4.2 DPO-Compatible Data Formatting
* **Extend** the logging module to export oracle/sampling events into DPO-ready preference pairs (positive/negative decisions) while preserving raw transcripts for auditing. Provide a CLI utility (e.g., `omnithink_ext/export_oracle_dataset.py`) that converts recorded sessions into DPO datasets without touching the core inference path.
* **Document** the exact schema and dataset creation steps within `docs/oracle_preserving_plan.md` and a supplementary README section, ensuring practitioners can train without additional code modifications.

## 5. Interface Adjustments (Minimal and Optional)

* **CLI Enhancements**: Extend `app.py` or the relevant entry script to accept flags (`--enable-oracles`, `--sampler=human`) and pass instantiated helpers into `MindMap`. Guard these additions behind default parameters so existing scripts run unchanged when flags are absent.
* **Web/UI Considerations**: If the Streamlit or Gradio demos consume `MindMap`, expose a toggle for oracle and sampling features. Implement this via optional controls that default to off, guaranteeing the UI behaves as before unless the new features are deliberately enabled.

## 6. Testing and Validation Strategy

* **Unit Tests**: Add targeted tests under `tests/` covering oracle gating (mock oracles returning block/accept), sampler interaction (ensuring human veto prevents node creation), and logging outputs. Tests must use dependency injection to avoid hitting real LMs.
* **Integration Tests**: Create a regression test that runs a shallow tree build with oracles disabled to confirm no behavior changes, plus a second test exercising all new components to ensure decision traces are recorded.

## 7. Migration & Backward Compatibility

* All new parameters must be optional with sensible defaults. When oracles and samplers are omitted, `MindMap` must execute the existing flow to preserve current benchmarks and demos.
* Document the new configuration options and APIs in `docs/README.md` or a dedicated section, emphasizing that these features are additive and disabled by default.

## 8. Open Questions / Follow-ups

* Determine whether oracle prompts should reuse existing LMs or require dedicated models. The plan assumes configuration will specify the LM, allowing teams to reuse deployed endpoints.
* Validate that the logging volume is manageable; if not, introduce configurable sampling or compression in the logger without altering the inference pipeline.

By encapsulating new functionality in optional modules and injecting dependencies only where needed, this plan ensures OmniThink’s core expansion/reflection pipeline remains untouched unless oracle-preserving summarization is explicitly activated.
