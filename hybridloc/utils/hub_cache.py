"""Auto-download cached graphs from HuggingFace Hub on demand.

Activated by setting HF_REPO_ID in .env. When the pipeline asks for a
cached graph at data/graphs/<repo>__<commit>.pkl and it's not present
locally, this module tries to fetch it from the Hub. Falls through
silently if the file isn't on the Hub either.
"""

from __future__ import annotations

import os
from pathlib import Path


def try_download_graph(cache_path: Path) -> bool:
    """Try to fetch `cache_path` (and its .skeletons.jsonl sibling) from HF Hub.

    Returns True if both files were downloaded successfully, False otherwise.
    Silent on every failure mode (no HF_REPO_ID, network error, file missing).
    """
    repo_id = os.environ.get("HF_REPO_ID", "").strip()
    if not repo_id:
        return False

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False

    from ..log import info, warn
    filename = cache_path.name
    skel_filename = cache_path.with_suffix(".skeletons.jsonl").name

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    info(f"[hub-cache] looking up {filename} on {repo_id} ...")

    try:
        # Graph file
        local_pkl = hf_hub_download(
            repo_id=repo_id,
            filename=f"graphs/{filename}",
            repo_type="dataset",
            local_dir=str(cache_path.parent),
        )
        # Move it into the expected location if HF placed it in graphs/
        if Path(local_pkl).resolve() != cache_path.resolve():
            import shutil
            shutil.copy2(local_pkl, cache_path)

        # Skeletons file (optional — older caches may not have it)
        try:
            local_skel = hf_hub_download(
                repo_id=repo_id,
                filename=f"graphs/{skel_filename}",
                repo_type="dataset",
                local_dir=str(cache_path.parent),
            )
            if Path(local_skel).resolve() != cache_path.with_suffix(".skeletons.jsonl").resolve():
                import shutil
                shutil.copy2(local_skel, cache_path.with_suffix(".skeletons.jsonl"))
        except Exception:
            pass  # skeletons file is optional

        info(f"[hub-cache] downloaded {filename} from {repo_id} ✓")
        return True
    except Exception as e:
        warn(f"[hub-cache] not on Hub or download failed: {type(e).__name__}: {e}")
        return False
