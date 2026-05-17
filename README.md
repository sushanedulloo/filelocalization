# HybridLoc v2

Graph-guided **file & function localization** for SWE-bench. Given a GitHub issue and a repo, it returns the ranked list of `(file, function, suspect_lines)` that most likely need editing.

It composes ideas from 8 verified 2025–2026 papers (LocAgent, OrcaLoca, SweRank, RepoLens, RepoMem, GraphLocator, RGFL, ARISE) into a 5-stage pipeline:

1. **Pre-filter** — BM25 + dense + LLM rank the repo's files (RRF merge with test-file penalty)
2. **Graph build** — heterogeneous code graph: files, functions, classes, statements, commit memory, concept clusters
3. **Symptom extraction** — pull stack frames, error messages, named APIs from the issue
4. **Priority-guided traversal** — abductive causal-chain reasoning over the graph
5. **Rerank + consistency vote** — bi-encoder rescore + listwise LLM rerank, 3 runs, majority vote

This component **does not** generate patches. It produces the localization target that a downstream patch generator consumes.

---

## For the patch-generation teammate

You don't need to understand the pipeline internals. The component exposes a single Python API.

### Setup

```bash
git clone https://github.com/sushanedulloo/filelocalization.git
cd filelocalization
conda env create -f environment.yml      # or environment.linux-cuda.yml on a GPU box
conda activate hybridloc
pip install einops
cp .env.example .env
# fill in NIM_API_KEY in .env
```

### Optional but recommended — restore the cached graphs

Concept extraction takes hours per repo. We've already done it for all 12 SWE-bench Verified repos and cached the graphs. Pull them from HuggingFace Hub:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    "sushanedulloo/hybridloc-cache",   # ← the dataset I'll publish
    repo_type="dataset",
    local_dir="data/",
)
```

After this, `data/graphs/` will have one `.pkl` per repo. Stage 2 (the slow part) is skipped automatically.

### Using the API

```python
from pathlib import Path
from hybridloc.pipeline.orchestrate import HybridLocPipeline

# Initialize once
pipeline = HybridLocPipeline(config_path=Path("configs/swe_bench_verified.yaml"))

# For each (issue, repo) pair you want to localize:
repo_path  = Path("/path/to/cloned/repo/at/base_commit")
issue_text = "your issue text here..."

bundle = pipeline.build_index(
    repo_root=repo_path,
    base_commit_sha="abc123...",       # SWE-bench instance's base_commit
    cache_path=Path("data/graphs/<repo>__<commit_prefix>.pkl"),
)

result = pipeline.localize(
    issue=issue_text,
    bundle=bundle,
    repo_root=repo_path,
    instance_id="optional_id",
)

# What you get back
for item in result.ranked[:5]:
    print(item.file_path)          # "django/core/validators.py"
    print(item.function_key)       # "django/core/validators.py::URLValidator.__call__"
    print(item.qualname)           # "URLValidator.__call__"
    print(item.suspect_lines)      # (101, 142)  ← tuple[int, int] or None
    print(item.confidence)         # "high" | "medium" | "low"
    print(item.score)              # float, higher = more confident
    print(item.causal_chain)       # list[str], LLM's reasoning steps
```

`result.ranked` is a list of `VotedItem` objects (from `hybridloc.pipeline.stage5_rerank`). Top items are what to edit.

### Recommended integration

```python
top5 = result.ranked[:5]

# For each candidate, read the actual file content for the suspect range:
for item in top5:
    file_full = (repo_path / item.file_path).read_text()
    start, end = item.suspect_lines or (1, file_full.count("\n"))
    # Pass file_full, suspect range, and issue_text to your code generator

    patch = your_code_generator(
        issue=issue_text,
        file_path=item.file_path,
        file_content=file_full,
        suspect_lines=(start, end),
        causal_chain=item.causal_chain,
    )
    # apply patch, run tests, etc.
```

### Running the full SWE-bench Verified evaluation

```bash
# All 500 instances (slow — needs cached graphs)
python scripts/run_swebench_verified.py

# Just one repo
python scripts/run_swebench_verified.py --repos django/django --limit 5

# Only multi-file edits
python scripts/run_swebench_verified.py --multi-file --limit 10
```

Outputs:
- `results/swebench_verified.csv` — file/function accuracy metrics
- `results/swebench_verified.md` — human-readable comparison vs paper baselines
- `logs/instances/<instance_id>.log` — per-instance pipeline detail

---

## Architecture overview

```
       ┌──────────────────────────────────────────────────────────────┐
       │ Stage 1: Pre-filter (BM25 + Dense + LLM → RRF merge → top-20)│
       └──────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
       ┌──────────────────────────────────────────────────────────────┐
       │ Stage 2: Graph build (cached per repo+commit)                │
       │   - Skeleton parsing (tree-sitter)                            │
       │   - INVOKE / IMPORT / INHERIT edges                           │
       │   - RepoMem commit-history edges                              │
       │   - Concept clusters (LLM summaries + K-means)                │
       └──────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
       ┌──────────────────────────────────────────────────────────────┐
       │ Stage 3: Symptom extraction (stack frames, error msgs, APIs) │
       └──────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
       ┌──────────────────────────────────────────────────────────────┐
       │ Stage 4: Priority-guided traversal × 3 runs                  │
       │   - Seeds from concepts + literals + Stage 1 files            │
       │   - Per-file seed budget (max 8 per file)                     │
       │   - Abductive causal-chain LLM calls (Think-High)             │
       │   - Statement-level drilling for high-confidence functions    │
       └──────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
       ┌──────────────────────────────────────────────────────────────┐
       │ Stage 5: Rerank + consistency vote                           │
       │   - Extended candidate pool (RGFL safety net)                 │
       │   - Bi-encoder rescore                                        │
       │   - Listwise LLM rerank (Think-High)                          │
       │   - Majority vote across 3 runs                               │
       └──────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                        Ranked (file, function, lines) list
```

---

## Configuration

`configs/swe_bench_verified.yaml` is the main entry. Inherits from `base.yaml` and `deepseek_v4_flash.yaml`. Defaults work for SWE-bench; override via env vars if needed.

### Env vars (`.env`)

| Variable | Purpose | Required? |
|---|---|---|
| `NIM_API_KEY` | NVIDIA NIM API key for main LLM | ✅ |
| `NIM_BASE_URL` | LLM endpoint (default = NIM cloud) | optional |
| `NIM_MODEL` | Main pipeline model (`qwen/qwen3-next-80b-a3b-thinking` recommended) | optional |
| `NIM_CONCEPT_BASE_URL` | Different endpoint for concept extraction (e.g. local Ollama) | optional |
| `NIM_CONCEPT_API_KEY` | API key for concept endpoint | optional |
| `NIM_CONCEPT_MODEL` | Concept-extraction model (small/fast) | optional |
| `HYBRIDLOC_SKIP_CONCEPTS` | Set to `1` to skip concept extraction (faster, ~3-5% accuracy drop) | optional |
| `HYBRIDLOC_EMBED_DEVICE` | Force a specific GPU for CodeRankEmbed (e.g. `cuda:1`) | optional |
| `HYBRIDLOC_PROVIDER` | `nim` (default), `groq`, or `claude` | optional |

---

## Layout

```
hybridloc/         # the localization component
  parsing/         # tree-sitter wrappers, skeleton extraction
  graph/           # heterogeneous code graph (nodes, build, concepts, memory, seeds)
  retrieval/       # BM25 (sparse) + CodeRankEmbed (dense)
  pipeline/        # 5-stage orchestrator
    stage1_prefilter.py    # BM25 + Dense + LLM → RRF merge
    stage3_symptoms.py     # JSON extraction from issue text
    stage4_traversal.py    # priority-queue traversal + abductive reasoning
    stage5_rerank.py       # bi-encoder + listwise LLM rerank + consistency vote
    orchestrate.py         # top-level HybridLocPipeline class
  llm/             # NIM/Groq/Ollama client + prompts
  log.py           # logger writing to file + console
swebench_eval/     # SWE-bench Verified plug-in
  load_dataset.py  # fetch SWE-bench Verified, clone repos at base_commit
  gold_extractor.py # parse patches into (file, function, line) ground truth
  metrics.py       # file_acc@k, func_recall@k, MRR, line_recall
  runner.py        # iterate dataset, run pipeline, score
  report.py        # write CSV + markdown
scripts/           # CLI entry points
  run_swebench_verified.py  # main evaluation entry
  run_overnight.sh          # process all 12 repos overnight on a GPU server
  test_nim_queue.py         # quick latency probe for NIM API
configs/           # YAML configs (base, deepseek/qwen, swe-bench)
data/              # caches (gitignored)
  graphs/          # cached graphs per (repo, commit)
  nim_cache/       # cached LLM responses
  repos/           # cloned repos at base_commit
```

---

## Performance notes

| Repo | Concept extraction (one-time per repo) | Per-instance run (after caching) |
|---|---|---|
| flask, requests, seaborn | ~5-10 min | ~10-15 min |
| pytest, pylint, xarray, sphinx | ~20-40 min | ~15-20 min |
| astropy, matplotlib | ~1-2 hrs | ~20-30 min |
| scikit-learn, sympy | ~2-4 hrs | ~25-40 min |
| django | ~4-6 hrs | ~30-60 min |

All concept extraction is **cached on disk** — after the first run per repo, every future instance of that repo skips Stage 2 entirely.

For NIM cloud free tier, Stage 1's LLM call can take anywhere from 10 seconds to 30+ minutes depending on queue congestion. Stages 4 and 5 use multiple Think-High calls each.

---

## Citation / references

Research design and ablation rationale are in [`HybridLoc_v2_plan.md`](./HybridLoc_v2_plan.md).

Key references:
- Agentless (Xia et al., 2024) — BM25 + LLM file pre-filter
- LocAgent (2025) — heterogeneous code graph + LLM ranker
- SweRank (May 2025) — bi-encoder fine-tuning on SweLoc
- RepoLens (2025) — concept clusters from LLM summaries
- RepoMem (2025) — commit-history seeds, time-aware
- OrcaLoca (Yu et al., 2025) — priority-queue traversal with action decomposition
- GraphLocator (2025) — abductive causal-chain reasoning
- RGFL (2025) — explanation re-embedding for retrieval reranking
- ARISE (2025) — statement-level localization with intra-procedural dataflow

---

## License

MIT
