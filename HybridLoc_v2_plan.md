# HybridLoc v2 — A Research-Backed Plan for a Novel File / Function / Line Localization Component

> **Pipeline stage:** Localization — the front of an agentic SWE pipeline that takes a GitHub issue and outputs a ranked list of `(file, function, line-range)` edit targets for a downstream patch generator.
>
> **Hardware envelope:**
> - College server: 3× RTX 2080 Ti (≈11 GB each, no NVLink assumed)
> - NVIDIA NIM API: `deepseek-ai/deepseek-v4-flash` (284B MoE, 1M-token context, "Think" reasoning modes available)
>
> **Primary benchmarks:** SWE-bench Lite (300), SWE-bench Verified (500), LocBench (LocAgent's split), SWE-bench-live (recency-controlled).

---

## 0. TL;DR — What we are building and why it is novel

We are building **HybridLoc v2**, a 5-stage localization component that produces ranked `(file, function, line)` triples for an issue. It composes ideas from 8 verified 2025–2026 papers, but its *combination* has never been published:

1. **Pre-filter** with skeleton + embedding + LLM ranking (Agentless, Meta-RAG)
2. **Build a multi-granularity heterogeneous graph** with structural edges *plus* intra-procedural data-flow edges (LocAgent + ARISE)
3. **Warm-start the graph** with concept clusters (RepoLens) *and* repository commit memory (RepoMem)
4. **Traverse the graph** with a priority queue, action decomposition, distance-aware pruning, and abductive causal reasoning (OrcaLoca + GraphLocator)
5. **Rerank** with a fine-tuned bi-encoder, explanation-guided re-scoring, and consistency voting (SweRank + RGFL + LocAgent)

**The novelty is in the joins, not the pieces:**
- **(N1)** Concept-cluster + commit-memory warm-start as *seeds* for an OrcaLoca-style priority queue. RepoLens uses concepts only for file filtering; RepoMem uses commits only as a flat memory bank. We use them as *graph entry points*.
- **(N2)** GraphLocator's abductive causal chains are *generated during traversal* (Stage 4) and then *reused as ranking signals* in RGFL-style explanation re-embedding (Stage 5). No prior work pipes traversal-time abductive chains into retrieve-and-rerank.
- **(N3)** Statement-level data-flow edges (ARISE, 2026) live *inside* the traversal graph, so the same priority queue that picks the next function can drill to *the buggy line* without a separate stage. LocAgent stops at function granularity; ARISE adds DU edges but uses a flat agent.
- **(N4)** A two-headed LLM strategy that uses **DeepSeek-V4-Flash "Non-think" mode** for cheap, high-volume calls (file ranking, skeleton scoring, explanation drafts) and **"Think-High" mode** for the small number of abductive-reasoning and final-rerank calls. This is the right way to spend a 1M-context budget.

---

## 1. Problem statement (recap with sharper framing)

Given:
- An issue description `I` (natural-language bug report, possibly with stack trace, repro steps, expected vs. observed)
- A repository snapshot `R` at the issue's base commit (often 1k–10k files, 50k–500k functions)

Produce:
- A ranked list `[(file_i, function_i, line_range_i, score_i, explanation_i)]_{i=1..k}` such that the ground-truth edit locations are recovered at high `Recall@k` for small `k` (k=1, 3, 5).

**Why localization is the hard step (validated by 2026 SOTA):** even Claude / GPT-5 systems on SWE-bench Verified miss the right file more often than they hit it on the harder splits; localization is the dominant bottleneck for end-to-end issue resolution ([SWE-bench overview, 2026](https://www.swebench.com/SWE-bench/)).

**Two failure modes localization must handle (from GraphLocator):**
- *Symptom-to-cause mismatch:* the issue describes the crash site, but the fix is in a helper several hops away.
- *One-to-many mismatch:* the fix touches 2–5 entities across 1–3 files.

---

## 2. End-to-end architecture

```
                                  ┌───────────────────────────────────────┐
                                  │   OFFLINE (per-repo, run once)        │
                                  │  • Skeleton index                     │
                                  │  • Heterogeneous graph (LocAgent edges│
                                  │     + ARISE intra-proc DU edges)      │
                                  │  • Concept clusters (RepoLens)        │
                                  │  • Commit/issue memory (RepoMem)      │
                                  │  • Function embeddings (SweRankEmbed) │
                                  └──────────────────┬────────────────────┘
                                                     │
   Issue I ──► Stage 1: Pre-Filter (skeleton + embed + LLM) ──► Top-20 files
                                                     │
                                                     ▼
              Stage 2: Subgraph Assembly  (structural + data-flow + concept + memory seeds)
                                                     │
                                                     ▼
              Stage 3: Symptom Extraction & Seed Set  (regex + LLM-extracted exceptions, traces)
                                                     │
                                                     ▼
              Stage 4: Priority-Guided Traversal with Abductive Causal Reasoning
                       (OrcaLoca priority queue + GraphLocator CIG expansion)
                                                     │
                                                     ▼
              Stage 5: Retrieve-and-Rerank with Explanation Re-embedding + Consistency Vote
                       (SweRank bi-encoder + RGFL re-embedding + 3-run majority vote)
                                                     │
                                                     ▼
                          Final ranked (file, function, line-range) list
```

The offline block is the key engineering investment — it pays for itself across thousands of issues per repo.

---

## 3. Stage 1 — Lightweight Pre-Filter

**Goal:** drop ≥95% of the repo before any expensive operation. Recall@20 (file-level) must be ≥85% or the rest of the pipeline cannot win.

**Inputs:** `I`, repo snapshot.
**Outputs:** `F_top20` — 20 candidate files.

### 3.1 Skeleton index (offline, once per repo)
For every file produce a compact text representation containing only:
- Path
- Imports
- Class names + docstring first line
- Function signatures + docstring first line
- (Optional) NL summary for the whole file (1 sentence, generated once with DeepSeek-V4-Flash Non-think mode and cached)

Implementation: **Tree-sitter** (multi-language, fast on CPU). Cache to disk as `skeleton.jsonl`.

### 3.2 Three retrievers, then union
Run these in parallel, then take the union of top-K from each (K=15–20):

1. **Sparse:** BM25 over skeletons (rank-bm25). Cheap, surprisingly strong baseline.
2. **Dense:** SweRankEmbed-small (137M, our fine-tuned variant from Stage 5; first iteration uses pretrained `nomic-ai/CodeRankEmbed`). Embed `I` once, score against precomputed file-level mean-pooled function embeddings.
3. **LLM rank:** Feed the directory tree + skeleton (compressed) to DeepSeek-V4-Flash *Non-think* with a prompt: *"Return JSON list of the 10 most-likely-to-be-edited files, with one-sentence reason each."* 1M-context lets us pass surprisingly large skeletons in one call for repos up to ~3–5k files. For larger repos, hierarchical: rank directories first, then files within top-3 directories.

Union → de-duplicate → cap at 20 files. Why union and not intersection? RepoLens / Agentless ablations show recall is more important than precision at this stage — Stage 2+ tolerate noise, but a missed file is unrecoverable.

### 3.3 Validation gate
On a held-out 20-issue dev split per language, require **Recall@20 ≥ 0.85**. If lower, increase to top-30 and accept the cost.

**Backing papers:** Agentless (Xia et al., 2024) for skeleton ranking; Meta-RAG (Aug 2025) for ~80% codebase compaction before retrieval; SweRank for the dense model choice.

---

## 4. Stage 2 — Multi-Granularity Heterogeneous Graph

**Goal:** a queryable graph that captures *both* structural relationships (LocAgent, ACL 2025) *and* intra-procedural data flow (ARISE, 2026).

### 4.1 Node types
| Level | Node | Source |
|---|---|---|
| Repo | Directory | filesystem |
| Repo | File | filesystem |
| Code | Class | Tree-sitter |
| Code | Function / Method | Tree-sitter |
| Code | Statement | Tree-sitter (only for files in `F_top20`) |
| Issue | Symptom | Stage 3 |
| Knowledge | Concept cluster | RepoLens |
| Knowledge | Commit / past-issue | RepoMem |

Statement nodes are expensive — only materialize them lazily for files surviving Stage 1. The full graph holds Directory→File→Class→Function for the whole repo (cheap, one-time), and statement-level nodes are added on demand.

### 4.2 Edge types
**Structural (LocAgent — proven 92.7% file-level acc with Qwen2.5-Coder-32B):**
- `CONTAIN` (Dir→File, File→Class, Class→Method, File→Function)
- `IMPORT` (File→File)
- `INVOKE` (Function→Function, resolved with a static call-graph builder; Python: Jedi/Pyright; Java: javalang; multi-language fallback: identifier-match within the same file/imports)
- `INHERIT` (Class→Class)

**Data-flow (ARISE, 2026 — adds 17.0 / 15.0 pts on Function/Line Recall@1 over SWE-agent):**
- `DEF_USE` (Statement→Statement, intra-procedural; built from Tree-sitter + simple SSA inside each function)

**Knowledge (the novelty):**
- `CONCEPT_OF` (Function→ConceptCluster) — RepoLens labels
- `EVOLVED_BY` (Function→Commit) — last N=10 commits touching this function
- `CO_EVOLVED` (Function↔Function) — co-edited in ≥3 historical commits (mined from RepoMem's commit history). This edge is what lets the traversal find the "second site" of a multi-file fix.

### 4.3 Build steps
1. Parse `R` with Tree-sitter; extract structural nodes/edges.
2. Run a static call-graph pass (Jedi for Python; for other languages use the LSP-server route or LocAgent's reference implementation).
3. Mine the last ~7,000 commits (RepoMem's number) for `EVOLVED_BY` and `CO_EVOLVED` edges. Cap at top-200 most-edited files for the semantic memory layer.
4. Run RepoLens offline concept extraction:
   - One-line LLM summary per function (DeepSeek-V4-Flash Non-think; cached). For 50k functions ≈ 50k cheap calls — batch to NIM with concurrency limit.
   - Cluster summary embeddings with K-means (k≈ √(N_func), capped 200–500).
   - Label each cluster with the LLM (one call per cluster).
5. Persist to disk in two formats: NetworkX pickle (development), and a SQLite + adjacency-list pair (production, fast partial-load).

**Cost note:** the offline build is dominated by the per-function LLM summaries. At ~50k calls × ~$0/call (NIM allowance) and ~150ms/call with concurrency=32, that's ≈ 4 hours. Run once per repo per ~quarter.

---

## 5. Stage 3 — Symptom Extraction & Seed Set

**Goal:** turn the issue text into a small set of *graph entry points* (symptom nodes) plus a few *anchor candidates*.

### 5.1 Symptom extraction (cheap)
DeepSeek-V4-Flash Non-think with a strict JSON-output prompt:
```
{
  "exception_types": [...],          # e.g., ["KeyError", "TypeError"]
  "error_messages":  [...],          # quoted strings from the issue
  "stack_frames":    [{file, func, line}, ...],  # if a traceback is included
  "behaviors":       [...],          # e.g., ["serialize migration", "render template"]
  "api_calls_named": [...]           # e.g., ["MigrationWriter.serialize"]
}
```

### 5.2 Symptom node insertion
Each extracted item becomes a `Symptom` node in the issue subgraph and is connected to:
- Functions whose *name or docstring* contains a literal match
- Functions whose embedding has cosine ≥ 0.55 with the symptom string (use SweRankEmbed)
- Stack-frame entries → direct `Symptom→Function` edges with weight 1.0

### 5.3 Concept + memory seeds
- Score the issue text against concept-cluster centroids (RepoLens) → top-3 clusters → all member functions become *concept seeds*.
- Retrieve top-10 similar past issues / commits from RepoMem (commit-message + diff embeddings) → files/functions touched in those commits become *memory seeds*.

### 5.4 Combined seed set `S`
Union of (stack-frame functions) ∪ (literal-match symptoms) ∪ (top concept seeds) ∪ (top memory seeds). Cap at 30 seeds. Each seed gets a prior score from its provenance:
- stack frame: 1.0
- literal match: 0.8
- top-3 concept: 0.5
- memory: 0.4 (decayed by recency)

This `S` is what gets pushed into the priority queue at the start of Stage 4.

---

## 6. Stage 4 — Priority-Guided Traversal with Abductive Causal Reasoning

**Goal:** explore the graph from `S`, scoring each visited entity, and produce ~10–15 ranked candidate functions with causal-chain explanations.

### 6.1 The priority queue (OrcaLoca, ICML 2025)

The queue holds *actions*, not nodes. Action types:
- `READ_FUNCTION(f)` — fetch full body and add to context
- `EXPAND_CALL(f)` — push all `INVOKE`-neighbors of `f`
- `EXPAND_INHERIT(c)` — push parents/children in inheritance
- `EXPAND_CO_EVOLVED(f)` — push functions historically co-edited with `f`
- `DRILL_STATEMENTS(f)` — materialize statement nodes + DU edges inside `f`
- `EXPAND_DU(stmt)` — follow def-use edges from a suspect statement

Priority for action `a` operating on entity `e`:
```
priority(a) = w_sem  · cosine(emb(I_combined), emb(e))
            + w_chain · current_causal_chain_score(e)        # from §6.3
            + w_seed · seed_prior(e)                         # from §5.4
            + w_recent · recency_bonus(e)                    # decays from last-visited
            - w_depth · max(0, hops_from_seeds(e) - 3)
            - w_visited · already_visited(e)
```
Default weights: `w_sem=1.0, w_chain=0.8, w_seed=0.6, w_recent=0.2, w_depth=0.4, w_visited=2.0`. These are tunable on a 30-issue dev split via random search.

`I_combined` is the issue text concatenated with the **explanation generated so far** (RGFL-style); this is what makes the queue update its "what am I looking for" representation as evidence accumulates.

### 6.2 Action decomposition (OrcaLoca)
When the next action targets a class/file, do not slurp the body. Instead:
1. Read its skeleton (already cached from Stage 1).
2. Score every method in the skeleton against `I_combined`.
3. Push only the top-3 methods as `READ_FUNCTION` actions.

This is what makes the 1M-context window of DeepSeek-V4-Flash *useful* rather than abused — we never blast the whole context window with junk.

### 6.3 Distance-aware context pruning (OrcaLoca)
Maintain a working LLM context with these rules, refreshed each iteration:
- distance 1 from any active symptom: full body
- distance 2: signature + docstring
- distance 3+: name only
- evicted nodes: tracked in a "seen" set, not re-fetched unless re-prioritized

### 6.4 Abductive causal-chain reasoning (GraphLocator, Dec 2025)
Every ~3 expansions, call DeepSeek-V4-Flash in **Think-High** mode with the structured prompt:

```
You are diagnosing a bug.
Issue: {I}
Symptoms: {extracted_symptoms}
Current Causal Graph (Mermaid): {CIG_mermaid}
Newly visited functions (with code): {top-3 by distance-1 budget}

Task:
For each new function, construct a causal chain explaining HOW it could
produce the observed symptoms. If no plausible chain exists, say "no link".
Output JSON:
{
  "chains": [{"function": "...", "chain": ["step1", "step2", ...], "score": 0..10}],
  "next_to_explore": ["function_name", ...]
}
```

Two outputs are consumed:
- `chain` becomes the function's `current_causal_chain_score` (used in `priority(a)`)
- `next_to_explore` is pushed onto the queue with a +0.5 boost

The `current_causal_chain_score` decays for entities whose chain has not been refreshed in the last K iterations — this prevents stale high scores.

### 6.5 Drilling to statements (ARISE, 2026)
When a function's chain score crosses a threshold (e.g., ≥7/10) AND its embedding similarity to the symptom is high, fire `DRILL_STATEMENTS(f)`:
1. Materialize statement nodes for `f` and intra-procedural DU edges.
2. For each stack-frame line that points into `f`, insert a "stack pointer" symptom directly on that statement.
3. Push `EXPAND_DU(stmt)` actions following def-use chains from the most-suspect statements.

This is what lets the same pipeline produce **line-range** outputs without a second model. ARISE proved this gives +15 pts Line Recall@1 vs. SWE-agent.

### 6.6 Termination
Stop when any of:
- top-3 candidates' scores are stable across 3 consecutive iterations
- iteration budget (default 30) is hit
- LLM call budget (default 12 Think-High calls per issue) is hit

Output: a list of ~10–15 candidate functions, each with `(score, distance, causal_chain, suspect_statements)`.

---

## 7. Stage 5 — Retrieve-and-Rerank with Explanation Re-embedding & Consistency Vote

**Goal:** convert Stage 4's candidates into a final ranked list with calibrated scores.

### 7.1 Bi-encoder retrieval (SweRank, May 2025)
- Use the SweRankEmbed bi-encoder (137M, fine-tuned on SweLoc — see §8) to score every Stage-4 candidate against `I`.
- Also re-score the top-50 from Stage-4's "extended candidate pool" (entities at distance ≤2 with `priority` above a low threshold) — this is a safety net for entities the priority queue *almost* visited.
- Output: top-20 by retrieval score.

### 7.2 Explanation generation (RGFL, Jan 2026)
For each of those 20 candidates:
- Most candidates already have a causal chain from Stage 4 — *reuse it*. (This is N2: traversal-time chains become rerank-time signals.)
- For candidates only added in §7.1, generate a fresh 2-sentence explanation with DeepSeek-V4-Flash Non-think.

### 7.3 Two-stage rerank
**Embedding rerank.** Concatenate `I + chain` for each candidate, embed with SweRankEmbed, score against the candidate function's embedding. RGFL showed file-level Hit@1 jumps from 71.4 → 85% with this re-embedding.

**Listwise LLM rerank.** Pass the top-10 (by embedding-rerank score) to DeepSeek-V4-Flash Think-High in a *single* listwise prompt: *"Return a re-ordering of these 10 candidates with a confidence ∈ [0,1]. For each one, mark whether its causal chain is sufficient to explain the observed symptom."*

### 7.4 Consistency voting (LocAgent)
Repeat Stage 4 + 7.1–7.3 **three times** with:
- Run A: temperature 0.0
- Run B: temperature 0.3, seed S2
- Run C: rephrase issue once with the LLM (generate a paraphrase + run with temperature 0.2)

Aggregate:
- Final score(`f`) = mean rerank score across runs × (k/3) where k = #runs that include `f` in top-10
- Mark candidates appearing in 3/3 runs as "high confidence"; 2/3 as "medium"; 1/3 as "low"; ≤0 dropped

### 7.5 Final output schema
```json
[
  {
    "file": "django/db/migrations/serializer.py",
    "function": "TypeSerializer.serialize",
    "line_range": [142, 168],
    "score": 0.91,
    "confidence": "high",
    "causal_chain": ["...", "..."],
    "evidence_runs": 3
  },
  ...
]
```

---

## 8. Training the bi-encoder (the only GPU-heavy part)

### 8.1 Data
- **Primary:** SweLoc (released by SweRank, Salesforce AI Research). Pairs of (issue, positive function, hard-negative functions from the same repo).
- **Augment:** RepoMem-style mining from the target benchmark repos' commit history — every fix-commit gives one (linked-issue, edited-function) positive pair plus same-file negatives. This explicitly aligns the embedding space with the *recency* the production system will see.

### 8.2 Model
- Backbone: **`nomic-ai/CodeRankEmbed`** (137M, 8192-context, the model SweRank itself initializes from). Better than CodeBERT for code retrieval per Nomic's CoRNStack benchmarks.
- Architecture: shared-encoder bi-encoder, mean-pooling.

### 8.3 Setup on 3× RTX 2080 Ti
- Loss: MultipleNegativesRankingLoss (in-batch negatives) + curriculum hard-negative mining after epoch 1
- Batch: 16/GPU × 3 GPUs × grad-accum 4 = effective 192
- Precision: bf16 not supported on 2080 Ti → use fp16 with gradient scaling
- LR: 2e-5, linear warmup, 5 epochs
- Time: ~6–8 h
- Save checkpoint that maximizes MRR on a SWE-bench Verified dev split (N=50)

### 8.4 Optional: train a small reranker too
SweRank+'s Dec-2025 follow-up shows a small (1.5–7B) listwise reranker beats prompting a closed model. If we want to remove the LLM listwise rerank in §7.3 and run fully self-hosted, we can fine-tune `Qwen2.5-Coder-1.5B` on SweLoc-Rerank in ≈ 12 h on the 2080 Tis. **Defer this to v2.1**; v2.0 ships with the LLM rerank.

---

## 9. Why DeepSeek-V4-Flash specifically (and how to use its modes)

DeepSeek-V4-Flash on NIM is a 284B MoE with **1M-token context** and three reasoning modes (Non-think / Think-High / Think-Max). For localization that is almost the perfect tool:

| Stage | Where it is called | Mode | Why |
|---|---|---|---|
| 1 | Per-issue file ranking | Non-think | Cheap, structured, 1 call |
| 2 | Per-function summary (offline, 50k+ calls) | Non-think | Pure summarization |
| 3 | Symptom extraction | Non-think | Strict-JSON IE |
| 4 | Skeleton method scoring | Non-think | Light scoring |
| 4 | Abductive causal chain (~12/issue) | **Think-High** | Real multi-step reasoning |
| 5 | Listwise rerank (top-10) | **Think-High** | Subtle ordering |
| 5 | Final commentary | Non-think | Simple |

Operational notes from NIM docs:
- Pass `chat_template_kwargs: {enable_thinking: true, thinking: true}` to actually get reasoning tokens streamed when in Think modes ([NIM forum](https://forums.developer.nvidia.com/t/deepseek-v4-pro-v4-flash-on-nvidia-nim-streaming-tool-calls-do-not-continue-in-claude-code-anthropic-compatible-agent-workflow/368085)).
- Cache hash-keyed completions for the offline summary pass — 50k summaries that change rarely should never be re-paid for.
- Use the **OpenAI-compatible** endpoint shape; wrap it with a thin retry/jitter client; cap concurrency at ~16 to stay polite.

A 1M context lets us pass the entire repo skeleton in one call for repos up to ≈ 4–5k files, which removes the multi-pass file ranking that Agentless and LocAgent both have to do for medium repos.

---

## 10. Implementation phasing (12 weeks)

| Week | Deliverable | Acceptance criterion |
|---|---|---|
| 1 | Tree-sitter parser + skeleton index + sparse/dense file retriever + LLM file ranker | Recall@20 ≥ 0.85 on 30 SWE-bench Lite issues |
| 2 | Heterogeneous graph (LocAgent edges) in NetworkX with disk persistence | Graph builds for django/sympy/sklearn under 5 min each |
| 3 | RepoLens concept extraction + RepoMem commit mining (offline batches) | Concept clusters present, queries return seeded entities for 90% of dev issues |
| 4 | Symptom extractor + seed set assembly | Stack-frame items mapped to graph nodes for 100% of issues with tracebacks |
| 5–6 | Priority queue + action decomposition + distance pruning | Pipeline beats Agentless file-level Acc@5 on 30 dev issues |
| 7 | Abductive causal reasoning prompt + integration into the queue | Function-level Recall@5 ≥ 0.55 |
| 8 | ARISE-style statement nodes + DU edges + drill action | Line-level Recall@5 measurable |
| 9 | SweRankEmbed fine-tuning on SweLoc + repo-augmented data | MRR on dev ≥ pretrained baseline + 5 pts |
| 10 | Explanation re-embedding + listwise LLM rerank | Function-level Recall@1 ≥ 0.45 |
| 11 | 3-run consistency voting + confidence calibration | False-positive rate (low-confidence wrongs) < 25% |
| 12 | Full eval on SWE-bench Verified (500) + ablation table | Beat LocAgent+OrcaLoca published numbers on Acc@5 |

---

## 11. Evaluation & ablations

**Benchmarks**
- SWE-bench Lite (300) — for fast iteration
- SWE-bench Verified (500) — for headline numbers
- LocBench — for breadth
- SWE-bench-live (recent) — for contamination-free evaluation (the *SWE-Bench Illusion* paper, [arxiv 2506.12286](https://arxiv.org/html/2506.12286v3), shows held-out splits matter)

**Metrics**
- File-level: Acc@1, Acc@3, Acc@5
- Function-level: Recall@1, @3, @5, MRR
- Line-level: Recall@5 with ±10 line tolerance (ARISE-style)
- Cost: tokens/issue, $/issue, wall-clock/issue
- Calibration: Expected Calibration Error of confidence labels

**Ablation matrix** (each row removes one component; we want every removal to *hurt*, otherwise we drop the component):

| Variant | Expected Δ Function Recall@5 |
|---|---|
| Full HybridLoc v2 | baseline |
| − ARISE statement edges | -3 to -5 (line metric) |
| − GraphLocator abductive reasoning | -4 to -7 |
| − OrcaLoca priority queue (use BFS) | -8 to -12 |
| − RepoLens concept seeds | -2 to -3 |
| − RepoMem commit seeds | -2 to -4 (much higher on SWE-bench-live) |
| − RGFL explanation re-embedding | -3 to -5 |
| − consistency voting | -2 (precision hit) |
| − fine-tuned bi-encoder (use pretrained) | -4 to -6 |

**Baselines to reproduce or quote**
- Agentless (Xia 2024)
- LocAgent (ACL 2025) — reproduce with their Qwen2.5-Coder-32B fine-tune, or quote
- OrcaLoca (ICML 2025) — quote
- SweRank (May 2025) — reproduce
- GraphLocator (Dec 2025) — quote (no public code yet)
- RepoMem-augmented LocAgent (Oct 2025) — quote

---

## 12. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Static call-graph extraction is brittle for non-Python languages | High | Start Python-only on SWE-bench, add JS/Java in v2.1 using LSP |
| 1M-context calls become latency bottleneck | Medium | Aggressive Stage-1 compaction; fall back to hierarchical ranking when skeleton > 200k tokens |
| 2080 Ti can't fit 7B reranker even sharded | Medium | Stick to 137M bi-encoder + LLM rerank; defer 7B reranker to a server with newer GPUs |
| RepoMem commits leak future information | Medium | Always filter commits with `commit_time < issue_base_commit_time`; verify with a unit test on the eval pipeline |
| Concept clustering produces noisy clusters for small repos | Low | Floor: skip RepoLens for repos with < 500 functions; the priority queue still works |
| NIM rate limits during 50k-call offline pass | Medium | Use exponential backoff; persist progress to disk; resume cleanly |
| SWE-bench contamination inflates apparent gains | High (across the field) | Always also report SWE-bench-live numbers; lead with those |

---

## 13. Repository layout (proposed)

```
file_localization/
  configs/
    base.yaml
    deepseek_v4_flash.yaml
  hybridloc/
    __init__.py
    parsing/          # Tree-sitter wrappers
    graph/
      build.py        # LocAgent edges
      dataflow.py     # ARISE DU edges
      memory.py       # RepoMem commit mining
      concepts.py     # RepoLens clustering
    retrieval/
      sparse.py       # BM25
      dense.py        # SweRankEmbed wrapper
      train.py        # bi-encoder fine-tuning
    pipeline/
      stage1_prefilter.py
      stage2_subgraph.py
      stage3_symptoms.py
      stage4_traversal.py
      stage5_rerank.py
      orchestrate.py
    llm/
      nim_client.py   # DeepSeek-V4-Flash via NIM
      prompts/
        file_rank.txt
        symptom_extract.txt
        causal_chain.txt
        listwise_rerank.txt
    eval/
      swe_bench_lite.py
      swe_bench_verified.py
      locbench.py
      ablate.py
  data/
    skeletons/        # cached per-repo
    graphs/           # cached per-repo
    embeddings/       # cached per-repo
    swelog/           # cached SWELoc dataset
  scripts/
    build_repo_index.py
    run_localization.py
    train_retriever.py
  tests/
```

---

## 14. References (verified during planning)

| Paper | Year | Link | What we use |
|---|---|---|---|
| Agentless (Xia et al.) | 2024 | arXiv 2407.01489 | Skeleton-based pre-filter |
| LocAgent (Chen et al.) | ACL 2025 | [arXiv 2503.09089](https://arxiv.org/abs/2503.09089) · [GitHub](https://github.com/gersteinlab/LocAgent) | Heterogeneous graph (4 edges), consistency voting |
| OrcaLoca (Yu et al.) | ICML 2025 | [arXiv 2502.00350](https://arxiv.org/abs/2502.00350) · [GitHub](https://github.com/fishmingyu/OrcaLoca) | Priority queue, action decomposition, distance pruning |
| SweRank (Reddy et al.) | May 2025 | [arXiv 2505.07849](https://arxiv.org/abs/2505.07849) · [GitHub](https://github.com/SalesforceAIResearch/SweRank) | Bi-encoder + reranker, SweLoc dataset |
| RepoLens | Sep 2025 | [arXiv 2509.21427](https://arxiv.org/abs/2509.21427) | Conceptual concern clustering |
| RepoMem (Wang & Xu) | Oct 2025 | [arXiv 2510.01003](https://arxiv.org/abs/2510.01003) | Commit-history memory for localization |
| GraphLocator (Liu et al.) | Dec 2025 | [arXiv 2512.22469](https://arxiv.org/abs/2512.22469) | Causal Issue Graph + abductive reasoning |
| RGFL (Sepidband et al.) | Jan 2026 | [arXiv 2601.18044](https://arxiv.org/abs/2601.18044) | Bug-specific explanation re-embedding |
| ARISE | 2026 | [arXiv 2605.03117](https://arxiv.org/abs/2605.03117) | Statement-level DU edges, line-level localization |
| SweRank+ (multi-turn) | Dec 2025 | [arXiv 2512.20482](https://arxiv.org/html/2512.20482) | Optional small reranker for v2.1 |
| Meta-RAG (Tawosia et al.) | Aug 2025 | arXiv 2508.* | Codebase compaction for retrieval |
| SWE-bench Illusion | 2026 | [arXiv 2506.12286](https://arxiv.org/html/2506.12286v3) | Contamination warning → use SWE-bench-live |
| CodeRankEmbed (Nomic / CoRNStack) | 2024 | [HF model card](https://huggingface.co/nomic-ai/CodeRankEmbed) | Bi-encoder backbone |
| DeepSeek-V4-Flash on NIM | 2026 | [NIM model card](https://docs.api.nvidia.com/nim/reference/deepseek-ai-deepseek-v4-flash) · [Build with V4 blog](https://developer.nvidia.com/blog/build-with-deepseek-v4-using-nvidia-blackwell-and-gpu-accelerated-endpoints/) | LLM backbone for all reasoning calls |

---

## 15. What ships in v2.0 vs. v2.1 vs. v3

**v2.0 (Weeks 1–12, the plan above)**
- Python-only, SWE-bench Verified target
- All 5 stages with 137M bi-encoder + DeepSeek-V4-Flash
- 3-run consistency

**v2.1 (Weeks 13–18)**
- Add a small (1.5B) self-hosted reranker (replaces LLM listwise call)
- Add JS / Java parsing
- Replace heuristic call-graph with LSP-driven one

**v3 (research extension)**
- Reinforcement-learn the priority-queue weights (`w_*`) directly from end-to-end resolution success — turns the queue from a heuristic into a learned policy. This is an open research direction that no published localization paper has done; it would be a clean follow-up paper.

---

*The combination is the contribution. Each ingredient is published; the recipe is not. The plan is sized to your hardware, your API, and a single graduate-student-quarter of focused work.*
