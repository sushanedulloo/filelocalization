"""Iterate the SWE-bench Verified split, run HybridLoc, collect predictions + gold."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm
from hybridloc.log import info, warn

from hybridloc.pipeline.orchestrate import HybridLocPipeline, PipelineResult

from .gold_extractor import GoldLocation, extract_gold_for_instance
from .load_dataset import base_commit_date, clone_at, load_verified
from .metrics import Gold, Prediction


def to_prediction(pr: PipelineResult, *, top_k: int = 50) -> Prediction:
    files: list[str] = []
    seen_files: set[str] = set()
    funcs: list[str] = []
    seen_funcs: set[str] = set()
    lines: list[tuple[str, int, int]] = []
    confs: list[float] = []
    for v in pr.ranked[:top_k]:
        if v.file_path not in seen_files:
            seen_files.add(v.file_path)
            files.append(v.file_path)
        if v.function_key not in seen_funcs:
            seen_funcs.add(v.function_key)
            funcs.append(v.function_key)
            confs.append(v.score)
        if v.suspect_lines:
            s, e = v.suspect_lines
            lines.append((v.file_path, s, e))
    return Prediction(
        instance_id=pr.instance_id,
        files=files,
        functions=funcs,
        lines=lines,
        confidences=confs,
    )


def to_gold(instance_id: str, gl: GoldLocation) -> Gold:
    return Gold(
        instance_id=instance_id,
        files=gl.files,
        functions=gl.functions,
        line_ranges=gl.line_ranges,
    )


class HybridLocRunner:
    def __init__(self, *, config_path: Path, repos_root: Path, cache_root: Path):
        self.pipeline = HybridLocPipeline(config_path=config_path)
        self.repos_root = repos_root
        self.cache_root = cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        *,
        limit: int | None = None,
        only_repos: set[str] | None = None,
    ) -> tuple[list[Prediction], list[Gold]]:
        ds = load_verified()
        preds: list[Prediction] = []
        golds: list[Gold] = []

        for inst in tqdm(ds, total=len(ds) if limit is None else min(limit, len(ds))):
            if limit is not None and len(preds) >= limit:
                break
            if only_repos and inst["repo"] not in only_repos:
                continue

            instance_id = inst["instance_id"]
            info(f"━━━ [{len(preds)+1}] {instance_id} ━━━")
            try:
                info(f"Cloning {inst['repo']} @ {inst['base_commit'][:12]} ...")
                repo_path = clone_at(
                    inst["repo"], inst["base_commit"], repos_root=self.repos_root
                )
                bcd = base_commit_date(repo_path, inst["base_commit"])
                cache_key = (
                    f"{inst['repo'].replace('/', '__')}__{inst['base_commit'][:12]}.pkl"
                )
                bundle = self.pipeline.build_index(
                    repo_root=repo_path,
                    base_commit_sha=inst["base_commit"],
                    base_commit_date=bcd,
                    cache_path=self.cache_root / cache_key,
                )
                pr = self.pipeline.localize(
                    issue=inst["problem_statement"],
                    bundle=bundle,
                    repo_root=repo_path,
                    instance_id=instance_id,
                )
                pred = to_prediction(pr)
                preds.append(pred)

                gl = extract_gold_for_instance(
                    patch_text=inst["patch"],
                    repo_path=repo_path,
                    base_commit=inst["base_commit"],
                )
                gold = to_gold(instance_id, gl)
                info(f"[gold] files={gl.files}  functions={gl.functions}")
                info(f"[pred] top-5 funcs={pred.functions[:5]}")
                golds.append(gold)
            except Exception as e:
                warn(f"FAILED {instance_id}: {type(e).__name__}: {e}")
                preds.append(
                    Prediction(
                        instance_id=instance_id,
                        files=[],
                        functions=[],
                        lines=[],
                        confidences=[],
                    )
                )
                golds.append(Gold(instance_id=instance_id, files=set(), functions=set(), line_ranges=[]))
                print(f"[runner] {instance_id}: {type(e).__name__}: {e}")

        return preds, golds
