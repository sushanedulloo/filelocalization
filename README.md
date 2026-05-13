# HybridLoc v2

Graph-guided code localization component for SWE-bench. Composes ideas from
LocAgent, OrcaLoca, SweRank, RepoLens, RepoMem, GraphLocator, RGFL, and ARISE
(8 verified 2025–2026 papers) into a single pipeline.

- **Research design:** [`HybridLoc_v2_plan.md`](./HybridLoc_v2_plan.md)
- **Implementation plan:** `~/.claude/plans/purring-exploring-hollerith.md`

## Quick start

```bash
# 1. create env
#    macOS / dev:    environment.yml          (CPU + MPS, no CUDA)
#    Linux + GPU:    environment.linux-cuda.yml  (CUDA 11.8 for 2080 Ti / Turing)
conda env create -f environment.yml
conda activate hybridloc

# 2. configure secrets
cp .env.example .env
# then edit .env and set NIM_API_KEY

# 3. smoke-test the NIM client
python -m hybridloc.llm.nim_client --selftest
```

## Layout

```
hybridloc/         # the localization component
  parsing/         # Tree-sitter wrappers
  graph/           # heterogeneous code graph (LocAgent + ARISE + RepoLens + RepoMem)
  retrieval/       # BM25 + bi-encoder
  pipeline/        # 5-stage orchestrator
  llm/             # NIM client + prompts
  eval/            # local micro-evals
swebench_eval/     # SWE-bench Verified plug-in (Stage 6)
scripts/           # CLI entry points
tests/             # unit + integration
configs/           # base + model + benchmark configs
data/              # caches (gitignored)
```

## Status

Bootstrap (Week 0) in progress.
