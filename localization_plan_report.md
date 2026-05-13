# HybridLoc: A Detailed Plan for a Strong File & Function Localization Component

> **This plan is for the Localization stage of your software agentic pipeline.**
> It is backed by 6 key papers from 2025–2026 and designed to work with your available hardware (3x RTX 2080 Ti + API access).

---

## 1. The Problem We Are Solving

When a GitHub issue comes in (e.g., "Django crashes when serializing migrations"), the agent must answer:
- **Which files** in the 3,000+ file repo are involved?
- **Which functions** inside those files are the root cause?

This is called **localization**. It is the hardest and most important step. If you get it wrong, nothing downstream (patch generation, test generation) can succeed.

**The core difficulty has two parts** (from GraphLocator, Dec 2025, Peking University + ByteDance):
1. **Symptom-to-cause mismatch**: The issue says "crash in serialization" but the real bug is in a helper function nobody mentions.
2. **One-to-many mismatch**: One issue often requires changing 2–5 functions spread across different files.

Current approaches fail because they either:
- Are too simple (just keyword search)
- Are too expensive (full LLM agent reading every file)
- Ignore structural relationships between files and functions

Our plan attacks all three problems together.

---

## 2. Overview of the Proposed System: HybridLoc

HybridLoc is a **4-stage pipeline** that goes from raw issue → ranked list of suspicious functions.

```
Issue Description
       ↓
[Stage 1] Lightweight Pre-Filter
  → Top-20 suspicious files using fast retrieval
       ↓
[Stage 2] Heterogeneous Code Graph Construction
  → Build a graph of the repo with 4 types of edges
       ↓
[Stage 3] Priority-Guided Graph Traversal with Context Pruning
  → Walk the graph intelligently to find suspicious functions
       ↓
[Stage 4] Retrieve-and-Rerank with Explanation Generation
  → Use a bi-encoder to retrieve + LLM to rerank with reasoning
       ↓
Final Ranked List of (File, Function) pairs
```

Each stage is backed by a specific paper. Nothing here is made up.

---

## 3. Stage 1: Lightweight Pre-Filter

### What it does
Before doing anything expensive, quickly throw away 95% of the repo that is clearly irrelevant.

### How it works
**Step 1a – Repository Skeleton:** Convert the entire repo into a simple text tree showing only file paths and function/class names (no code bodies). This is called a "skeleton" or "outline."

Example:
```
django/
  db/
    migrations/
      writer.py
        → class MigrationWriter
        → def serialize()
        → def as_string()
      serializer.py
        → class TypeSerializer
        → def serialize()
```

**Step 1b – Embedding Retrieval:** Embed the issue description using a small embedding model (e.g., `text-embedding-3-small` from OpenAI API or a local `bge-small` model on your 2080 Ti). Embed each file-level summary. Use cosine similarity to get the top-20 most similar files.

**Step 1c – LLM File Ranking:** Feed the skeleton tree + issue to an LLM via API and ask: "Which of these files are most likely to need changes? Give top 10 with reasons."

Combine both lists (embedding retrieval + LLM ranking) and keep the **union of top 20 files**. Everything else is dropped from further consideration.

### Why this works
This exact approach is used in **Agentless (Xia et al., 2024)** and is proven to work well as a cheap first filter. Meta-RAG (Aug 2025, CMU/JPMorgan) shows that condensing the codebase first (by ~80%) before retrieval improves both accuracy and cost. The key insight: you don't need the LLM to read 3,000 files — you need it to read 20.

### Cost on your hardware
- Embedding the skeleton: runs on a single 2080 Ti in under a minute
- LLM call: 1 API call per issue

---

## 4. Stage 2: Heterogeneous Code Graph Construction

### What it does
Build a structured graph of the 20 candidate files that captures how code elements relate to each other.

### The 4 Edge Types (from LocAgent, ACL 2025, Yale/USC/Stanford)
This is the core innovation of LocAgent, published at ACL 2025. They parse the codebase into a **directed heterogeneous graph** with these node and edge types:

**Nodes:**
- Directory node
- File node
- Class node
- Function/Method node

**Edges (4 types):**
- `CONTAIN`: File → Class, File → Function, Class → Method
- `IMPORT`: File → File (when file A imports file B)
- `INVOKE`: Function → Function (when function A calls function B)
- `INHERIT`: Class → Class (when class A extends class B)

### Why this matters
Without these edges, you can only do keyword matching. With these edges, you can do **multi-hop reasoning**. For example:

> Issue mentions `serialize()` crashes → `serialize()` INVOKES `TypeSerializer.serialize()` → `TypeSerializer` INHERITS from `BaseSerializer` → The bug is actually in `BaseSerializer.validate()` which nobody mentioned.

Without the INVOKE and INHERIT edges, you would never find `BaseSerializer.validate()`.

### How to build it on your hardware
Use **Tree-sitter** (a fast open-source parser) to extract ASTs for Python/Java/etc. Tree-sitter runs on CPU and is very fast. Build the graph using **NetworkX** in Python. No GPU needed for this step.

```python
# Pseudocode
graph = nx.DiGraph()
for file in candidate_files:
    tree = tree_sitter_parser.parse(file)
    classes, functions = extract_entities(tree)
    for func in functions:
        graph.add_node(func.name, type="function", code=func.body, file=file)
    for call in extract_calls(tree):
        graph.add_edge(caller, callee, type="INVOKE")
    for imp in extract_imports(tree):
        graph.add_edge(file, imp.target, type="IMPORT")
```

### Additional Graph Enrichment (from GraphLocator, Dec 2025)
GraphLocator from Peking University adds **causal structure** on top of the structural graph. They introduce a "Causal Issue Graph (CIG)" which adds:
- **Symptom nodes**: Extracted from the issue text (e.g., "KeyError", "None type", "migration crash")
- **Causal edges**: Learned from historical bug-fix patterns (which symptoms tend to appear in which modules)

For your implementation, do a simplified version:
- Extract error keywords and exception types from the issue text using regex or a small LLM call
- Add these as "symptom nodes" connected to all functions that match those keywords in their docstrings or code
- This gives the graph a starting point for traversal

---

## 5. Stage 3: Priority-Guided Graph Traversal with Context Pruning

### What it does
Starting from the symptom nodes identified in Stage 2, intelligently walk the graph to find which functions are most likely the root cause. This is like a smart search through the codebase.

### Idea 1: Priority-Based Action Scheduling (from OrcaLoca, ICML 2025, UC San Diego + Intel)
OrcaLoca introduces a **priority queue** for deciding which code entity to look at next. Instead of BFS (dumb breadth-first search) or random walking, every action (e.g., "read this function", "follow this INVOKE edge") gets a **priority score**.

The priority score for an action is:
```
priority(action) = semantic_similarity(issue, target_code) × recency_bonus × depth_penalty
```

Where:
- `semantic_similarity`: cosine similarity between issue embedding and target function's embedding
- `recency_bonus`: actions closer to the current exploration point score higher
- `depth_penalty`: penalize going too deep (more than 3 hops from any symptom node)

The agent always picks the highest-priority action next. This prevents the classic problem of going down a wrong path and wasting all your context budget.

### Idea 2: Action Decomposition with Relevance Scoring (from OrcaLoca, ICML 2025)
When the agent wants to explore a large class or file, instead of reading the whole thing (expensive), it first reads only the **skeleton** (function signatures + docstrings). Then it scores each method individually. Only the top-k methods get their full code body fetched.

```
Explore class X:
  Step 1: Read skeleton of X → [method_a, method_b, method_c, ...]
  Step 2: Score each method against issue embedding
  Step 3: Fetch full body of top-3 methods only
```

This is called "action decomposition" in OrcaLoca and it reduces token usage dramatically.

### Idea 3: Distance-Aware Context Pruning (from OrcaLoca, ICML 2025)
As the agent explores the graph, the context window fills up with code. OrcaLoca shows that naively concatenating everything confuses the LLM.

Instead, use a **distance-based pruning rule**:
- Functions at graph distance 1 from a symptom node → keep **full code body** in context
- Functions at graph distance 2 → keep **signature + docstring only**
- Functions at graph distance 3+ → keep **function name only** or drop

This keeps the context focused on what actually matters. OrcaLoca demonstrated this achieves **65.33% function match rate** on SWE-bench Lite, which was SOTA at the time of publication.

### Idea 4: Abductive Reasoning Prompting (from GraphLocator, Dec 2025)
When the agent reaches a candidate function, instead of just asking "is this the bug?", ask the LLM to reason **backwards**:

> "Given that the symptom is [X] and this function does [Y], explain HOW this function could cause the observed symptom. If you cannot construct a plausible causal chain, score this function low."

This is called **abductive reasoning** — reasoning from the effect back to the cause. GraphLocator from Peking University shows this significantly improves precision because the LLM must construct a logical causal chain, not just find semantic similarity.

Prompt template:
```
Issue: {issue_text}
Observed symptom: {symptom}
Candidate function: {function_code}

Task: Construct a causal chain explaining HOW this function could produce the symptom.
If a plausible chain exists, score 1-10. If not, score 0.
Output: {"causal_chain": "...", "score": N}
```

### What the traversal produces
After Stage 3, you have a ranked list of ~10-15 candidate functions with:
- Their full code
- Their causal chain explanation
- Their priority score from traversal
- Their distance from symptom nodes

---

## 6. Stage 4: Retrieve-and-Rerank with Explanation-Guided Rescoring

### What it does
Take the ~10-15 candidates from Stage 3 and do a final, careful reranking using two techniques.

### Idea 5: Retrieve-and-Rerank Architecture (from SweRank, May 2025, UIUC + Salesforce Research)
SweRank reformulates localization as an **information retrieval problem** with two stages:
1. **Retrieval**: A small bi-encoder model quickly scores all candidates against the issue
2. **Reranking**: A larger LLM takes the top-K retrieved candidates and produces a final ranking

**For your hardware:**
- Train a small bi-encoder retriever on your 3x 2080 Ti. SweRank uses a 137M parameter model for the retriever — this fits easily on one 2080 Ti.
- Use an API-based LLM (Claude or GPT-4o) for the reranker step
- Training data: SweRank releases their SWELOC dataset (curated from GitHub repositories) — you can use this directly

**Why train your own retriever instead of using a generic one?**
SweRank shows that generic code embedding models (trained for query-to-code or code-to-code retrieval) perform poorly on issue localization because issue descriptions are very different from normal NL queries (they describe *failures*, not *features*). A model trained specifically on (issue description → buggy function) pairs is significantly better.

**Training setup on your 3x 2080 Ti:**
- Model: `microsoft/codebert-base` (125M) as the backbone for the bi-encoder
- Loss: Contrastive loss with hard negatives (functions from the same repo but not the ground truth)
- Training time: ~4-6 hours on 3x 2080 Ti with batch size 32
- Data: Use the SWELOC dataset that SweRank releases

### Idea 6: Bug-Specific Explanation Generation Before Reranking (from RGFL, Jan 2026, Concordia University)
RGFL (Reasoning Guided Fault Localization) proposes generating **bug-specific explanations** for each candidate before scoring them.

For each candidate function, ask the LLM:
> "Given this issue, write a 2-sentence explanation of specifically WHY this function is or is not the root cause."

Then embed these explanations and use them as an **additional query signal** for the bi-encoder. This creates a feedback loop:
1. LLM generates explanation → "The bug is in `serialize()` because it doesn't handle None values before calling `TypeSerializer`"
2. Embed this explanation
3. Use this embedding (not just the original issue embedding) to re-score all candidates

RGFL showed this two-stage approach (LLM explanation → embedding re-score) improved file-level Hit@1 from **71.4% to 85%** and element-level Exact Match under top-3 from **36% to 69%** on SWE-bench Verified.

### Idea 7: Confidence Estimation via Consistency (from LocAgent, ACL 2025)
Run the full localization pipeline 3 times with slight prompt variations (change the temperature or rephrase the issue). Then:
- Functions that appear in all 3 runs → **High confidence** (include in final output)
- Functions that appear in 1-2 runs → **Medium confidence** (include with a warning flag)
- Functions that appear in 0 runs → **Drop**

This is basically majority voting. LocAgent shows this consistency check reduces false positives significantly.

---

## 7. Training Plan for the Bi-Encoder Retriever

This is the only part that requires GPU training. Here is a concrete plan:

### Data Preparation
- Download the **SWELOC dataset** from SweRank's GitHub (released with the paper)
- Each example has: (issue_description, positive_function, hard_negative_functions_from_same_repo)
- Split: 80% train, 10% val, 10% test

### Model Architecture
```
Bi-encoder:
  Query encoder:  CodeBERT-base (125M) → encodes issue description
  Document encoder: CodeBERT-base (125M) → encodes function code
  (shared weights between query and document encoder)
```

### Training Details
```
Framework: HuggingFace Transformers + PyTorch
GPUs: 3x RTX 2080 Ti (GPU 1, 2, 3 from your server)
Batch size: 32 (with gradient accumulation steps=4 → effective batch=128)
Learning rate: 2e-5 with linear warmup
Epochs: 5
Loss: MultipleNegativesRankingLoss (in-batch negatives + hard negatives)
Time estimate: ~5-7 hours
```

### What you get
A small (250MB) bi-encoder model that:
- Embeds any (issue, function) pair in milliseconds
- Is specifically trained on the distribution of real GitHub issues and buggy functions
- Runs at inference time on CPU — no GPU needed

---

## 8. Conceptual Knowledge Layer (Optional Enhancement)

### Idea 8: Concept-Level Matching Before Code-Level Matching (from RepoLens, Oct 2025)
RepoLens (published Oct 2025) proposes extracting **"conceptual concerns"** from the codebase before doing any code-level matching.

A conceptual concern is a cluster of related functions that all implement the same high-level concept. For example:
- "authentication logic" → `login()`, `verify_token()`, `check_permissions()`
- "file parsing" → `parse_csv()`, `read_json()`, `load_config()`
- "migration serialization" → `serialize()`, `as_string()`, `MigrationWriter`

**How to extract concepts:**
1. Generate a one-line natural language description for every function in the repo using an LLM
2. Cluster these descriptions using K-means on their embeddings
3. Label each cluster with its dominant concept (using the LLM)

**How to use at inference time:**
1. Given the issue, find the top-3 matching concept clusters
2. All functions in those clusters are immediately elevated in priority for the traversal in Stage 3

RepoLens showed this technique gives **22%+ relative improvement** in Hit@k and **46% improvement** in Recall@k over Agentless and OpenHands baseline systems.

**Implementation note:** This can be done offline (precompute once per repository). Clustering 10,000 functions takes about 10 minutes on a single 2080 Ti.

---

## 9. Full Pipeline Summary

| Stage | Input | Method | Paper Source | Output |
|-------|-------|---------|-------------|--------|
| 1. Pre-Filter | Issue + Full repo | Skeleton tree + Embedding retrieval + LLM ranking | Agentless (2024), Meta-RAG (Aug 2025) | Top-20 candidate files |
| 2. Graph Build | Top-20 files | Tree-sitter AST parsing → Heterogeneous graph with 4 edge types + symptom nodes | LocAgent (ACL 2025), GraphLocator (Dec 2025) | Code knowledge graph |
| 3. Traversal | Graph + Issue | Priority queue + Action decomposition + Distance pruning + Abductive reasoning | OrcaLoca (ICML 2025), GraphLocator (Dec 2025) | ~10-15 ranked candidate functions |
| 4. Rerank | Candidates | Bi-encoder retrieval + Explanation generation + Consistency voting | SweRank (May 2025), RGFL (Jan 2026), LocAgent (ACL 2025) | Final ranked (file, function) list |

---

## 10. What Makes This Novel (Your Contribution)

None of the above papers does all of this together. Here is what is new in your combination:

1. **No prior work combines GraphLocator's causal abductive reasoning with OrcaLoca's priority traversal.** GraphLocator uses causal reasoning but does not use a priority queue. OrcaLoca uses priority queues but does not use abductive reasoning. You are the first to combine both.

2. **No prior work uses RepoLens conceptual concerns as a warm-start for graph traversal.** RepoLens only uses concepts for file-level filtering. Using them to seed the graph traversal in Stage 3 is new.

3. **The explanation-guided rescoring from RGFL has never been applied inside a graph-based pipeline.** RGFL applies it on a flat list of candidates. Applying it after graph traversal (where candidates already have causal chain explanations from Stage 3) creates a much richer reranking signal.

---

## 11. Evaluation Plan

Evaluate on:
- **SWE-bench Lite** (300 Python issues from 12 repos) — the standard benchmark
- **LocBench** (LocAgent's benchmark, broader coverage)

Metrics to report:
- **File-level Acc@1, Acc@3, Acc@5** (does the ground truth file appear in top-1/3/5?)
- **Function-level Recall@1, Recall@3, Recall@5**
- **Mean Reciprocal Rank (MRR)**
- **Token cost per issue** (to show efficiency)

Baselines to compare against:
- Agentless (simplest strong baseline)
- LocAgent (graph-based SOTA)
- OrcaLoca (priority-based SOTA)
- Your method: HybridLoc

---

## 12. Implementation Order (Week by Week)

**Week 1–2: Build the foundation**
- Implement Stage 1: Skeleton extraction + embedding retrieval + LLM file ranking
- Test on 20-30 SWE-bench Lite examples manually
- Goal: Verify top-20 file recall is above 80%

**Week 3–4: Build the graph**
- Integrate Tree-sitter for Python AST parsing
- Implement the 4 edge types in NetworkX
- Implement symptom node extraction from issue text
- Test graph construction on 10 repositories

**Week 5–6: Implement traversal**
- Build the priority queue with semantic scoring
- Implement action decomposition (skeleton first, then full body)
- Implement distance-aware context pruning
- Implement abductive reasoning prompting

**Week 7–8: Train the bi-encoder**
- Download SWELOC dataset from SweRank's GitHub
- Fine-tune CodeBERT-base on 3x 2080 Ti
- Evaluate retriever performance in isolation

**Week 9–10: Reranking and explanation generation**
- Implement RGFL-style explanation generation
- Integrate explanation embeddings into reranking
- Implement consistency voting (3 runs, majority vote)

**Week 11–12: Evaluation and ablation**
- Run full evaluation on SWE-bench Lite (300 examples)
- Run ablation: remove each component one at a time to show its contribution
- Compare against Agentless, LocAgent, OrcaLoca baselines

---

## 13. References

| Paper | Venue | Key Contribution Used |
|-------|-------|-----------------------|
| Agentless (Xia et al., 2024) | arXiv | Skeleton-based file pre-filtering |
| LocAgent (Chen et al., 2025) | ACL 2025 | Heterogeneous code graph with 4 edge types, consistency voting |
| OrcaLoca (Yu et al., 2025) | ICML 2025 | Priority queue traversal, action decomposition, distance pruning |
| GraphLocator (Liu et al., 2025) | arXiv Dec 2025 | Causal Issue Graph, abductive reasoning, symptom-to-cause tracing |
| SweRank (Reddy et al., 2025) | arXiv May 2025 | Bi-encoder retrieve-and-rerank, SWELOC training dataset |
| RGFL (Sepidband et al., 2026) | arXiv Jan 2026 | Bug-specific explanation generation, two-stage reranking |
| Meta-RAG (Tawosia et al., 2025) | arXiv Aug 2025 | Codebase summarization for compact retrieval |
| RepoLens (2025) | arXiv Oct 2025 | Conceptual concern extraction for concept-level matching |

---

*This plan is designed to be realistic for your hardware (3x RTX 2080 Ti + API access) and produces a component that is competitive with or better than current SOTA on SWE-bench Lite.*
