"""End-to-end pipeline orchestrator: run Stage 1 -> Stage 5 for one issue.

Returns the final ranked list of (file, function, line-range) candidates plus
diagnostic metadata for eval.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import yaml

from ..log import info, warn
from ..graph.build import build_structural_graph
from ..graph.concepts import (
    add_concept_nodes,
    cluster_summaries,
    label_concepts,
    summarize_functions,
)
from ..graph.memory import add_memory_edges, collect_commits
from ..graph.nodes import NodeType
from ..graph.persist import load_graph, save_graph
from ..graph.seeds import attach_symptom_nodes, build_seed_set
from ..llm.nim_client import NIMClient
from ..parsing.skeleton import Skeleton
from ..retrieval.dense import DenseRetriever
from .causal import AbductiveReasoner
from .stage1_prefilter import PreFilter, Stage1Result
from .stage3_symptoms import SymptomExtractor
from .stage4_traversal import Traverser, TraversalConfig
from .stage5_rerank import RankedItem, Reranker, VotedItem, consistency_vote


@dataclass
class PipelineResult:
    instance_id: str
    ranked: list[VotedItem]
    stage1: Stage1Result
    termination: str = ""
    think_high_calls: int = 0
    raw_runs: list[list[RankedItem]] = field(default_factory=list)


@dataclass
class IndexBundle:
    """Per-(repo, base_commit) cached artifacts."""

    skeletons: list[Skeleton]
    graph: nx.MultiDiGraph
    issue_emb_cache: dict[str, "any"] = field(default_factory=dict)


class HybridLocPipeline:
    def __init__(
        self,
        *,
        config_path: Path,
        nim: NIMClient | None = None,
        dense: DenseRetriever | None = None,
    ):
        self.cfg = _load_config(config_path)
        self.nim = nim or NIMClient()
        # lightweight model for bulk concept extraction (1k-50k calls per repo)
        self.nim_concept = _make_concept_client(self.nim)
        self.dense = dense or DenseRetriever(
            model_name=self.cfg["retrieval"]["bi_encoder_model"],
            finetuned_path=self.cfg["retrieval"].get("bi_encoder_finetuned_path"),
        )
        self.prefilter = PreFilter(nim=self.nim, dense=self.dense, top_k=self.cfg["budgets"]["stage1_top_files"])
        self.symptom_extractor = SymptomExtractor(self.nim)
        self.reasoner = AbductiveReasoner(self.nim)
        self.reranker = Reranker(
            self.nim, self.dense,
            listwise_top_k=self.cfg["budgets"]["stage5_listwise_top_k"],
            retrieval_top_k=self.cfg["budgets"]["stage5_top_k_rerank"],
        )

    # ---------- offline ----------

    def build_index(
        self,
        *,
        repo_root: Path,
        base_commit_sha: str | None = None,
        base_commit_date=None,
        cache_path: Path | None = None,
    ) -> IndexBundle:
        # 1) Local cache hit
        if cache_path and cache_path.exists():
            graph = load_graph(cache_path)
            from ..parsing.skeleton import load_skeletons

            sk_path = cache_path.with_suffix(".skeletons.jsonl")
            skels = load_skeletons(sk_path) if sk_path.exists() else []
            return IndexBundle(skeletons=skels, graph=graph)

        # 2) Auto-download from HF Hub if HF_REPO_ID is set
        if cache_path is not None:
            from ..utils.hub_cache import try_download_graph
            if try_download_graph(cache_path) and cache_path.exists():
                graph = load_graph(cache_path)
                from ..parsing.skeleton import load_skeletons
                sk_path = cache_path.with_suffix(".skeletons.jsonl")
                skels = load_skeletons(sk_path) if sk_path.exists() else []
                return IndexBundle(skeletons=skels, graph=graph)

        from ..parsing.skeleton import build_repo_skeleton, save_skeletons

        from tqdm import tqdm
        import time

        stages = tqdm(total=5, desc="build_index stages", unit="stage", dynamic_ncols=True)

        stages.set_description("Stage 2a: parsing files")
        info(f"[Stage 2] Building skeleton index for {repo_root.name} ...")
        skels = build_repo_skeleton(repo_root)
        info(f"[Stage 2] {len(skels)} files parsed. Building structural graph ...")
        graph = build_structural_graph(skels, repo_root)
        info(f"[Stage 2] Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        stages.update(1)

        stages.set_description("Stage 2b: mining commits")
        info(f"[Stage 2] Mining commit history ...")
        t0 = time.perf_counter()
        commits = collect_commits(
            repo_root,
            base_commit_sha=base_commit_sha,
            base_commit_date=base_commit_date,
            max_commits=self.cfg["graph"]["repomem_commit_window"],
        )
        add_memory_edges(
            graph,
            commits,
            max_files=self.cfg["graph"]["repomem_max_files"],
            co_evolved_min_count=self.cfg["graph"]["co_evolved_min_count"],
        )
        info(f"[Stage 2] Commit mining done in {time.perf_counter()-t0:.1f}s — {len(commits)} commits")
        stages.update(1)

        import os
        skip_concepts = os.environ.get("HYBRIDLOC_SKIP_CONCEPTS", "0") == "1"
        concepts: list = []
        summaries: dict = {}

        if skip_concepts:
            info("[Stage 2] Concept extraction SKIPPED (HYBRIDLOC_SKIP_CONCEPTS=1)")
            stages.update(3)
        else:
            stages.set_description("Stage 2c: concept extraction (LLM)")
            skels_for_concepts = skels
            n_funcs = sum(len(s.functions) + sum(len(c.methods) for c in s.classes) for s in skels_for_concepts)
            info(f"[Stage 2] Running concept extraction on {len(skels_for_concepts)} files ({n_funcs} functions) ...")
            loop = asyncio.new_event_loop()
            try:
                summaries = loop.run_until_complete(summarize_functions(skels_for_concepts, self.nim_concept))
                stages.update(1)
                stages.set_description("Stage 2d: clustering")
                info(f"[Stage 2] {len(summaries)} function summaries generated. Clustering ...")
                concepts = cluster_summaries(
                    summaries,
                    dense=self.dense,
                    min_k=self.cfg["graph"]["concept_kmeans_min_k"],
                    max_k=self.cfg["graph"]["concept_kmeans_max_k"],
                )
                stages.update(1)
                stages.set_description("Stage 2e: labeling clusters")
                loop.run_until_complete(label_concepts(concepts, summaries, self.nim_concept))
            finally:
                loop.close()
            stages.update(1)

        stages.set_description("build_index complete")
        stages.close()
        info(f"[Stage 2] {len(concepts)} concept clusters built. Saving index ...")
        add_concept_nodes(graph, concepts, summaries)

        if cache_path is not None:
            save_graph(graph, cache_path)
            save_skeletons(skels, cache_path.with_suffix(".skeletons.jsonl"))

        return IndexBundle(skeletons=skels, graph=graph)

    # ---------- per issue ----------

    def localize(
        self,
        *,
        issue: str,
        bundle: IndexBundle,
        repo_root: Path,
        instance_id: str = "",
        consistency_runs: int | None = None,
    ) -> PipelineResult:
        from tqdm import tqdm
        n_runs = consistency_runs or self.cfg["budgets"]["consistency_runs"]
        stage_bar = tqdm(total=4 + n_runs, desc=f"localize {instance_id}", unit="stage", dynamic_ncols=True)

        stage_bar.set_description(f"[{instance_id}] Stage 1: pre-filter")
        info(f"[Stage 1] Pre-filtering files for instance '{instance_id}' ...")
        s1 = self.prefilter.run(issue, bundle.skeletons)
        info(f"[Stage 1] {len(s1.candidate_files)} candidate files selected from {s1.skeleton_count} total")
        stage_bar.update(1)

        stage_bar.set_description(f"[{instance_id}] Stage 3: symptoms")
        info(f"[Stage 3] Extracting symptoms from issue ...")
        symptoms = self.symptom_extractor.extract(issue)
        info(f"[Stage 3] Found: {len(symptoms.exception_types)} exceptions, {len(symptoms.stack_frames)} stack frames, {len(symptoms.api_calls_named)} named APIs")
        attach_symptom_nodes(bundle.graph, symptoms)
        stage_bar.update(1)

        info(f"[Stage 4+5] Starting {n_runs}-run consistency loop ...")
        runs: list[list[RankedItem]] = []
        last_termination = ""
        last_think = 0

        for run_ix in range(n_runs):
            stage_bar.set_description(f"[{instance_id}] Stage 4+5: run {run_ix+1}/{n_runs}")
            info(f"[Stage 4] Run {run_ix+1}/{n_runs} — traversal starting ...")
            issue_used, temperature = _run_variant(issue, run_ix, self.nim)
            issue_emb = self.dense.encode([issue_used])[0]

            seeds = build_seed_set(
                bundle.graph, symptoms,
                issue_embedding=issue_emb,
                stage1_candidate_files=s1.candidate_files,
                dense=self.dense,
            )
            traverser = Traverser(
                bundle.graph,
                repo_root=repo_root,
                reasoner=self.reasoner,
                dense=self.dense,
                config=TraversalConfig(
                    max_iterations=self.cfg["budgets"]["stage4_max_iterations"],
                    max_think_high_calls=self.cfg["budgets"]["stage4_max_think_high_calls"],
                    queue_cap=self.cfg["budgets"]["stage4_action_queue_cap"],
                ),
            )
            cands = traverser.run(issue=issue_used, symptoms=symptoms, seeds=seeds)
            info(f"[Stage 4] Run {run_ix+1} done — termination={traverser.termination_reason}, think_calls={traverser.think_calls}, candidates={len(cands)}")
            info(f"[Stage 5] Run {run_ix+1} — reranking {len(cands)} candidates ...")
            ranked = self.reranker.rerank(
                graph=bundle.graph,
                issue=issue_used,
                candidates=cands,
                stage1_files=s1.candidate_files,
            )
            info(f"[Stage 5] Run {run_ix+1} — top candidate: {ranked[0].function_key if ranked else 'none'}")
            runs.append(ranked)
            last_termination = traverser.termination_reason
            last_think += traverser.think_calls
            stage_bar.update(1)

        stage_bar.set_description(f"[{instance_id}] Stage 5: voting")
        voted = consistency_vote(runs, top_k=self.cfg["budgets"]["stage5_top_k_rerank"])
        stage_bar.update(1)
        stage_bar.set_description(f"[{instance_id}] done — top: {voted[0].function_key if voted else 'none'}")
        stage_bar.close()
        info(f"[Done] instance='{instance_id}'  top_result={voted[0].function_key if voted else 'none'}  confidence={voted[0].confidence if voted else 'n/a'}")
        info(f"[Done] Final ranked predictions:")
        for i, v in enumerate(voted[:10]):
            info(f"[Done]   #{i+1} {v.function_key} score={v.score:.3f} confidence={v.confidence} runs={v.runs_appearing}/{len(runs)} lines={v.suspect_lines}")
        return PipelineResult(
            instance_id=instance_id,
            ranked=voted,
            stage1=s1,
            termination=last_termination,
            think_high_calls=last_think,
            raw_runs=runs,
        )


# ---------------- helpers ----------------


def _make_concept_client(main_nim: NIMClient) -> NIMClient:
    """Return a NIMClient for concept extraction (Stage 2).

    Three optional env vars override the main client just for concept calls:
      NIM_CONCEPT_MODEL      — different model (e.g. llama3.1:8b)
      NIM_CONCEPT_BASE_URL   — different endpoint (e.g. http://localhost:11434/v1)
      NIM_CONCEPT_API_KEY    — different API key (e.g. "ollama")

    Each falls back to the main client's value if not set. This lets you run
    concept extraction on a fast local Ollama server (or Groq/OpenRouter)
    while keeping the main pipeline on NIM cloud with thinking-mode Qwen.
    """
    import os
    concept_model = os.environ.get("NIM_CONCEPT_MODEL", "").strip()
    concept_base_url = os.environ.get("NIM_CONCEPT_BASE_URL", "").strip()
    concept_api_key = os.environ.get("NIM_CONCEPT_API_KEY", "").strip()

    # If no overrides at all, reuse the main client (current behaviour)
    if not concept_model and not concept_base_url and not concept_api_key:
        return main_nim
    # If only model override and it matches main, also reuse
    if (concept_model == main_nim.config.model
            and not concept_base_url
            and not concept_api_key):
        return main_nim

    from ..llm.nim_client import NIMConfig, _auto_rpm
    api_keys = (
        [k.strip() for k in concept_api_key.split(",") if k.strip()]
        if concept_api_key
        else main_nim.config.api_keys
    )
    effective_base_url = concept_base_url or main_nim.config.base_url
    cfg = NIMConfig(
        api_keys=api_keys,
        base_url=effective_base_url,
        model=concept_model or main_nim.config.model,
        cache_dir=main_nim.config.cache_dir,
        max_concurrency=main_nim.config.max_concurrency,
        request_timeout=main_nim.config.request_timeout,
        rpm_limit=_auto_rpm(effective_base_url),  # CRITICAL: pick rpm per endpoint
    )
    from ..log import info
    info(
        f"[Stage 2] Using concept client: model={cfg.model}  "
        f"base_url={cfg.base_url}"
    )
    return NIMClient(cfg)


def _load_config(path: Path) -> dict:
    base = yaml.safe_load(path.read_text())
    inherits = base.pop("inherits", [])
    merged: dict = {}
    for inh in inherits:
        p = path.parent / inh
        merged = _deep_merge(merged, yaml.safe_load(p.read_text()))
    return _deep_merge(merged, base)


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _run_variant(issue: str, run_ix: int, nim: NIMClient) -> tuple[str, float]:
    """Runs A=T0, B=T0.3 (different seed via paraphrase no-op), C=paraphrase + T0.2."""
    if run_ix == 0:
        return issue, 0.0
    if run_ix == 1:
        return issue + "\n\n[seed=2]", 0.3
    # run C: paraphrase
    try:
        resp = nim.complete(
            "Paraphrase the following issue in 1-3 sentences without changing the meaning:\n\n"
            + issue,
            temperature=0.2,
        )
        return resp.text.strip() or issue, 0.2
    except Exception:
        return issue, 0.2
