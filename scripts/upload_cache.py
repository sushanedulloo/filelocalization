"""Upload cached graphs (and optionally NIM cache) to HuggingFace Hub.

After an overnight run on the server, run this to publish the cached graphs
so anyone (your teammate, future Kaggle sessions) can skip Stage 2 entirely.

Usage:
    huggingface-cli login          # one-time
    python scripts/upload_cache.py # uploads data/graphs/ by default

Optional env vars:
    HF_REPO_ID         repo id, e.g. "yourname/hybridloc-cache"
                       (default falls back to $HF_REPO_ID, otherwise asks)
    INCLUDE_NIM_CACHE  set to "1" to also upload data/nim_cache/ (large!)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub not installed. Run: pip install huggingface_hub")

    repo_id = os.environ.get("HF_REPO_ID")
    if not repo_id:
        repo_id = input("Enter HF repo id (e.g. yourname/hybridloc-cache): ").strip()
        if not repo_id:
            sys.exit("No repo id provided")

    graphs_dir = Path("data/graphs")
    if not graphs_dir.exists() or not any(graphs_dir.iterdir()):
        sys.exit(f"No graphs found at {graphs_dir}. Run the pipeline first.")

    api = HfApi()
    # idempotent: only creates if it doesn't exist
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

    print(f"Uploading {graphs_dir} → {repo_id}/graphs ...")
    api.upload_folder(
        folder_path=str(graphs_dir),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo="graphs",
        commit_message=f"Add cached graphs ({len(list(graphs_dir.glob('*.pkl')))} repos)",
    )
    print("  graphs uploaded ✓")

    if os.environ.get("INCLUDE_NIM_CACHE") == "1":
        nim_dir = Path("data/nim_cache")
        if nim_dir.exists():
            print(f"Uploading {nim_dir} → {repo_id}/nim_cache ...")
            api.upload_folder(
                folder_path=str(nim_dir),
                repo_id=repo_id,
                repo_type="dataset",
                path_in_repo="nim_cache",
                commit_message="Add NIM LLM-call cache",
            )
            print("  nim_cache uploaded ✓")

    print(f"\nDone. View at: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
