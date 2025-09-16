## Objects and operators we keep

At step \(m\) there is an information tree \(T_m=(V_m,E_m)\) and a conceptual pool \(P_m\). OmniThink’s expansion generates subnodes \(\mathrm{SUB}(N_i)\) for each leaf \(N_i\in L_m\) and updates the tree by
\[
T_{m+1}=\mathrm{Combine}\!\bigl(T_m,\mathrm{SUB}(N_0),\dots,\mathrm{SUB}(N_n)\bigr),
\]
and reflection distills insights \(I_{m+1}\) from new leaves and updates the pool by
\[
P_{m+1}=\mathrm{Merge}\!\left(I_{m+1},P_m\right).
\]
We do not alter this algebra; it is Equation (1) and Equation (2) in §3.1. The expand/reflect controller is Algorithm 2.  [oai_citation:5‡OmniThink.pdf](file-service://file-5ZNd7ZBEcYkbc9RdjrkKwE)

## Typed state

Each node \(v\in V\) carries a title \(t(v)\), a query string \(q(v)\), an evidence multiset \(EVID(v)\) (URL, title, snippet, passage-span, content-hash), and a set of distilled claims \(C(v)\). We assign a coarse semantic type \(\tau(v)\in\Sigma\) (topic, concept, entity, claim\_cluster) and annotate each directed edge \((p\!\to\!c)\) with a relation \(\rho(p\!\to\!c)\in\mathcal{R}\). Defaults are \(\tau(\text{root})=\textsf{topic}\) and \(\rho=\textsf{expands}\) for new children produced by expansion. (§3.1.1 describes subtopics; our \(\tau,\rho\) are the minimal formalization.)  [oai_citation:6‡OmniThink.pdf](file-service://file-5ZNd7ZBEcYkbc9RdjrkKwE)

## Route equivalence

Let \(\mathbf{e}(\cdot)\) be the embedding used by the repo’s retriever (§3.3 uses Sentence-BERT by default). Define
\[
\mathrm{sim}(u,v)=\lambda\,\cos\!\bigl(\mathbf{e}(t(u)),\mathbf{e}(t(v))\bigr)+(1-\lambda)\,J\bigl(\text{ngrams}(C(u)),\text{ngrams}(C(v))\bigr),
\]
with \(0<\lambda<1\), Jaccard \(J\) over n-grams of claims. Write \(u\sim v\) iff \(\mathrm{sim}(u,v)\ge\theta\). The canonicalization \(\kappa:V\to\mathcal{C}\) maps each node to its \(\sim\)-class under transitive closure; we call members of \(\mathcal{C}\) “canonicals.” Contracting \(T\) along \(\kappa\) yields a multigraph on \(\mathcal{C}\) that collapses conceptually identical content reached via different routes—precisely the redundancy class Fig. 2 critiques.  [oai_citation:7‡OmniThink.pdf](file-service://file-5ZNd7ZBEcYkbc9RdjrkKwE)

## Edge compatibility

Choose a fixed vocabulary \(\mathcal{R}=\{\textsf{expands},\textsf{is\_a},\textsf{part\_of},\textsf{addresses\_question},\textsf{perspective\_of}\}\) and a type set \(\Sigma=\{\textsf{topic},\textsf{concept},\textsf{entity},\textsf{claim\_cluster}\}\). A compatibility relation \(\mathcal{A}\subseteq \Sigma\times\mathcal{R}\times\Sigma\) defines admissible triples; the invariant is
\[
\forall (p\!\to\!c)\in E:\bigl(\tau(p),\rho(p\!\to\!c),\tau(c)\bigr)\in\mathcal{A}.
\]
This expresses that a “refinement” edge truly refines a topic/subtopic rather than jumbling outline levels (§3.2’s goal). The concrete schema we ship is in `configs/compat_schema.yaml`.  [oai_citation:8‡OmniThink.pdf](file-service://file-5ZNd7ZBEcYkbc9RdjrkKwE)

## Provenance ledger

Let \(A_D=\{S_1,\dots,S_n\}\) be the concatenated sections after drafting (§3.3 / Listings 4–5). Parse bracketed numerals into \(\phi(s)\subseteq\{1,\dots,M\}\), the evidence IDs cited by sentence \(s\). The missing-citation ratio (audit-only) is
\[
\mathrm{MCR}=\frac{\lvert\{s\in A_D:\phi(s)=\varnothing\}\rvert}{\lvert A_D\rvert}.
\]
We store \(\phi\) together with URL and passage hashes resolved from the information tree. OmniThink’s prompts already enforce inline citations; we only persist them.  [oai_citation:9‡OmniThink.pdf](file-service://file-5ZNd7ZBEcYkbc9RdjrkKwE)

## Knowledge graph export

Define \(G=(\mathcal{C},\tilde{E}\cup E_P)\) by contracting \(T\) along \(\kappa\). Here \(\tilde{E}\) contains edges between canonicals with carried-over \(\rho\); \(E_P\) adds symmetric “supports/contradicts” links derived from co-occurrence and status of insights in the reflected pool (§3.1.2). Each \(c\in\mathcal{C}\) carries \(C(c)=\biguplus_{v\in\kappa^{-1}(c)}C(v)\) with back-pointers to evidence.

## Labeler

Fix a task label set \(\mathcal{Y}\) (e.g., manifesto issue buckets or stances). For canonical \(c\) define
\[
S_y(c)=\alpha\sum_{x\in C(c)}w(x,y)+
\beta\sum_{(c\to u)\in\tilde{E}}\rho_w(\rho)\sum_{x\in C(u)}w(x,y)+
\gamma\sum_{s:\,c\leadsto s}u(s,y),
\]
with lexicon-based \(w\), relation weights \(\rho_w\), and contextual bumps \(u\) from citing sentences. Output \(L(c)(y)=\exp S_y(c)\big/\sum_{y'}\exp S_{y'}(c)\). A justification is the minimal subgraph \(H\subseteq G\) plus a minimal set of sentences \(\mathcal{S}\) that preserves \(\arg\max_y S_y(c)\); compute by greedy set cover over claims and sentences keyed by \(\phi\).

## Post-hoc invariants

Route-consistent labeling: \(D_{\mathrm{KL}}\!\bigl(L(c)\parallel L(c')\bigr)\le\varepsilon\) whenever \(\kappa^{-1}(c)\cup\kappa^{-1}(c')\) share an ancestor; label monotonicity: for \(\rho\in\{\textsf{is\_a},\textsf{part\_of}\}\), if \(L(p)(y)\ge\eta\) then \(L(c)(y)\ge\eta-\delta\). These are computed after a run.

## Metrics retained for comparability

Information diversity is the cosine-spread across retrieved pages; knowledge density is
\[
KD=\frac{\sum_{i=1}^{N}\mathbf{1}\{k_i \text{ unique}\}}{L},
\]
as in §4.2 (Equation 3). We compute both from the final draft to match the paper’s reports.  [oai_citation:10‡OmniThink.pdf](file-service://file-5ZNd7ZBEcYkbc9RdjrkKwE)
