Run the stock OmniThink stages as before. After `Article Composition` completes, call:

```python
from omnithink_ext import state, canon, compat, ledger, graph, labeler, audit
# 1) Load run_artifacts/state.json produced by our shims.
# 2) Export knowledge graph.
# 3) Apply manifesto labeler with configs/label_schemas/manifesto.example.yaml.
# 4) Write graph.graphml, labels.json, and justifications.jsonl.
# 5) Compute audits: route consistency, monotonicity, missing-citation ratio (MCR), KD, and info diversity.
```

This preserves the original behavior for users who only want the article, while enabling downstream labeling and audits without touching prompts or the loop. Definitions and equations referenced here match §3–§4 of the paper.  
