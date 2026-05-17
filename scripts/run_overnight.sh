#!/usr/bin/env bash
#
# Overnight script: run HybridLoc on 1 instance of each SWE-bench Verified repo,
# with concept extraction enabled via local Ollama.
#
# Usage:
#   ./scripts/run_overnight.sh
#
# Prereqs:
#   - .env file with NIM_API_KEY (NIM cloud) set
#   - Ollama installed (script will start the server if not running)
#   - Python venv activated
#
# Expected runtime: ~20-25 hours for all 12 repos. tmux strongly recommended.
#
set -euo pipefail

# ---------------- config ----------------

# Order: small repos first so quick wins happen early; biggest last
REPOS=(
  "pallets/flask"
  "psf/requests"
  "mwaskom/seaborn"
  "pytest-dev/pytest"
  "pylint-dev/pylint"
  "pydata/xarray"
  "sphinx-doc/sphinx"
  "astropy/astropy"
  "matplotlib/matplotlib"
  "scikit-learn/scikit-learn"
  "sympy/sympy"
  "django/django"
)

OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1:8b}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
# OLLAMA_GPU is auto-picked below (most free memory) unless CUDA_VISIBLE_DEVICES is preset.
MIN_FREE_MB="${MIN_FREE_MB:-6000}"        # require at least 6 GB free

OVERNIGHT_LOG_DIR="logs/overnight"
RESULTS_DIR="results/overnight"

# ---------------- helpers ----------------

log()  { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }
fail() { log "ERROR: $*"; exit 1; }

pick_gpu() {
  # Pick the two GPUs with the most free memory:
  #   OLLAMA_GPU       — for Ollama serving llama-3.1-8b (~5-8 GB)
  #   EMBED_GPU        — for CodeRankEmbed dense retriever (~600 MB)
  # If only one GPU has enough free memory, both use the same one.
  # Honors CUDA_VISIBLE_DEVICES if user set it (then both use that GPU).
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    log "Using preset CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    OLLAMA_GPU="${CUDA_VISIBLE_DEVICES}"
    EMBED_GPU="${CUDA_VISIBLE_DEVICES}"
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi not found; defaulting to GPU 0"
    OLLAMA_GPU=0
    EMBED_GPU=0
    return
  fi

  # Collect (index, free_mib) pairs into parallel arrays
  local -a idxs=() frees=()
  while IFS=',' read -r idx free; do
    idx="${idx// /}"
    free="${free// /}"
    [ -z "$idx" ] && continue
    idxs+=("$idx")
    frees+=("$free")
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)

  # Sort by free memory descending (simple insertion sort, list is small)
  local n=${#idxs[@]}
  local i j
  for ((i=1; i<n; i++)); do
    local key_idx=${idxs[i]}
    local key_free=${frees[i]}
    j=$((i-1))
    while [ $j -ge 0 ] && [ "${frees[j]}" -lt "$key_free" ]; do
      idxs[$((j+1))]=${idxs[j]}
      frees[$((j+1))]=${frees[j]}
      j=$((j-1))
    done
    idxs[$((j+1))]=$key_idx
    frees[$((j+1))]=$key_free
  done

  OLLAMA_GPU="${idxs[0]}"
  local ollama_free="${frees[0]}"
  if [ "$ollama_free" -lt "$MIN_FREE_MB" ]; then
    log "Warning: best GPU (${OLLAMA_GPU}) has only ${ollama_free} MiB free; need ${MIN_FREE_MB} MiB"
    log "Continuing anyway — Ollama may fall back to CPU or fail"
  fi
  log "Ollama → GPU ${OLLAMA_GPU} (${ollama_free} MiB free)"

  if [ "$n" -gt 1 ] && [ "${frees[1]}" -ge 2000 ]; then
    EMBED_GPU="${idxs[1]}"
    log "Dense retriever → GPU ${EMBED_GPU} (${frees[1]} MiB free)"
  else
    EMBED_GPU="$OLLAMA_GPU"
    log "Dense retriever → GPU ${EMBED_GPU} (shared with Ollama)"
  fi
}

verify_env() {
  [ -f .env ] || fail ".env file not found. Run from repo root."
  grep -q "^NIM_API_KEY=" .env || fail "NIM_API_KEY missing from .env"

  # Add Ollama settings if not already in .env
  if ! grep -q "^NIM_CONCEPT_BASE_URL=" .env; then
    log "Appending Ollama concept-extraction settings to .env"
    cat >> .env <<EOF

# Auto-added by run_overnight.sh
NIM_CONCEPT_BASE_URL=http://localhost:${OLLAMA_PORT}/v1
NIM_CONCEPT_API_KEY=ollama
NIM_CONCEPT_MODEL=${OLLAMA_MODEL}
EOF
  fi

  # Disable the skip flag so concepts actually run
  if grep -q "^HYBRIDLOC_SKIP_CONCEPTS=1" .env; then
    log "Disabling HYBRIDLOC_SKIP_CONCEPTS in .env"
    sed -i.bak 's/^HYBRIDLOC_SKIP_CONCEPTS=1/# HYBRIDLOC_SKIP_CONCEPTS=1/' .env
  fi
}

ensure_ollama() {
  if curl -sSf "http://localhost:${OLLAMA_PORT}/api/tags" >/dev/null 2>&1; then
    log "Ollama is already running on port ${OLLAMA_PORT}"
  else
    log "Starting Ollama server on GPU ${OLLAMA_GPU} ..."
    CUDA_VISIBLE_DEVICES="${OLLAMA_GPU}" \
      nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 8
    curl -sSf "http://localhost:${OLLAMA_PORT}/api/tags" >/dev/null \
      || fail "Ollama failed to start. Check /tmp/ollama.log"
    log "Ollama started"
  fi

  log "Ensuring model ${OLLAMA_MODEL} is available ..."
  ollama pull "${OLLAMA_MODEL}" >/dev/null
  log "Model ready"
}

run_repo() {
  local repo="$1"
  local safe
  safe=$(echo "$repo" | tr '/' '__')
  local log_file="${OVERNIGHT_LOG_DIR}/${safe}.log"
  local csv_file="${RESULTS_DIR}/${safe}.csv"
  local md_file="${RESULTS_DIR}/${safe}.md"

  log "=== START $repo ==="
  log "  log:     ${log_file}"
  log "  results: ${md_file}"

  if python scripts/run_swebench_verified.py \
       --limit 1 \
       --repos "${repo}" \
       --out "${csv_file}" \
       --out-md "${md_file}" \
       > "${log_file}" 2>&1; then
    log "=== DONE  $repo ==="
    # Print top-line metrics from the markdown
    if [ -f "${md_file}" ]; then
      head -3 "${md_file}" | tail -2 | sed 's/^/  /'
    fi
  else
    log "=== FAILED $repo (continuing with next) ==="
  fi
}

# ---------------- main ----------------

cd "$(dirname "$0")/.."   # cd to repo root

mkdir -p "${OVERNIGHT_LOG_DIR}" "${RESULTS_DIR}" data/graphs data/nim_cache

verify_env
pick_gpu
ensure_ollama

# Tell the Python pipeline which GPU to use for the dense retriever
export HYBRIDLOC_EMBED_DEVICE="cuda:${EMBED_GPU}"

log ""
log "================================================================"
log "  Overnight HybridLoc run — ${#REPOS[@]} repos"
log "  Started: $(date)"
log "  Logs:    ${OVERNIGHT_LOG_DIR}/"
log "  Results: ${RESULTS_DIR}/"
log "================================================================"
log ""

START_TIME=$(date +%s)

for repo in "${REPOS[@]}"; do
  run_repo "$repo"
done

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

log ""
log "================================================================"
log "  All ${#REPOS[@]} repos processed"
log "  Total time: ${ELAPSED} minutes"
log "  Finished: $(date)"
log "================================================================"
log ""
log "Cached graphs:"
ls -1 data/graphs/ | sed 's/^/  /'
log ""
log "Result summaries:"
for md in "${RESULTS_DIR}"/*.md; do
  [ -f "$md" ] || continue
  echo "  --- $(basename "$md" .md) ---"
  head -3 "$md" | tail -2 | sed 's/^/    /'
done
