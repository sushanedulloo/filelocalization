#!/usr/bin/env bash
#
# Full graph extraction for all SWE-bench Verified instances.
# Designed for multi-night runs in tmux. Resumable.
#
# Usage (in tmux):
#   ./scripts/run_full_graphs.sh
#   # detach: Ctrl+B then D
#   # reattach: tmux attach -t <session>
#
# If interrupted, just re-run — it skips already-cached graphs.
#
set -euo pipefail

# Reuse the GPU selection + Ollama startup logic from run_overnight.sh
SCRIPT_DIR="$(dirname "$0")"
cd "${SCRIPT_DIR}/.."

OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1:8b}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
MIN_FREE_MB="${MIN_FREE_MB:-6000}"

OVERNIGHT_LOG_DIR="logs/full_graphs"
mkdir -p "${OVERNIGHT_LOG_DIR}"

log()  { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }
fail() { log "ERROR: $*"; exit 1; }

# ---------- ensure Ollama is up on a real GPU ----------
ensure_ollama_simple() {
  if curl -sSf "http://localhost:${OLLAMA_PORT}/api/tags" >/dev/null 2>&1; then
    log "Ollama already running"
    return
  fi
  # Pick the GPU with most free memory
  local best_idx=0 best_free=0
  while IFS=',' read -r idx free; do
    idx="${idx// /}"; free="${free// /}"
    [ -z "$idx" ] && continue
    if [ "$free" -gt "$best_free" ]; then
      best_free="$free"; best_idx="$idx"
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)

  log "Starting Ollama on GPU ${best_idx} (${best_free} MiB free)"
  CUDA_VISIBLE_DEVICES="${best_idx}" OLLAMA_NUM_PARALLEL=1 \
    nohup ollama serve > /tmp/ollama.log 2>&1 &
  sleep 8
  curl -sSf "http://localhost:${OLLAMA_PORT}/api/tags" >/dev/null \
    || fail "Ollama failed to start. Check /tmp/ollama.log"
  ollama pull "${OLLAMA_MODEL}" >/dev/null
  log "Ollama ready on GPU ${best_idx}"
}

ensure_ollama_simple

LOG_FILE="${OVERNIGHT_LOG_DIR}/build_all_$(date +%Y%m%d_%H%M%S).log"

log "================================================================"
log "  Building all SWE-bench Verified graphs"
log "  Started: $(date)"
log "  Log:     ${LOG_FILE}"
log "  Resumable: yes (skips cached graphs)"
log "================================================================"

python scripts/build_all_graphs.py 2>&1 | tee "${LOG_FILE}"

log "================================================================"
log "  DONE: $(date)"
log "================================================================"
log ""
log "Cached graphs by repo:"
for r in pallets__flask psf__requests mwaskom__seaborn pytest-dev__pytest \
         pylint-dev__pylint pydata__xarray sphinx-doc__sphinx \
         astropy__astropy matplotlib__matplotlib scikit-learn__scikit-learn \
         sympy__sympy django__django; do
  count=$(ls data/graphs/${r}__*.pkl 2>/dev/null | wc -l)
  printf "  %-30s  %d\n" "$r" "$count"
done
