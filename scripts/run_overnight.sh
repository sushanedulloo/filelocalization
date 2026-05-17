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

SORTED_GPU_LIST=()       # GPUs sorted by free memory descending (populated by rank_gpus)

rank_gpus() {
  # Populate SORTED_GPU_LIST with GPU indices sorted by free memory descending.
  # Also sets OLLAMA_GPU (top candidate) and EMBED_GPU (second-best).
  # Honors CUDA_VISIBLE_DEVICES if user set it (only that GPU is considered).
  SORTED_GPU_LIST=()
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    log "Using preset CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    SORTED_GPU_LIST=("${CUDA_VISIBLE_DEVICES}")
    OLLAMA_GPU="${CUDA_VISIBLE_DEVICES}"
    EMBED_GPU="${CUDA_VISIBLE_DEVICES}"
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi not found; defaulting to GPU 0"
    SORTED_GPU_LIST=(0)
    OLLAMA_GPU=0
    EMBED_GPU=0
    return
  fi

  local -a idxs=() frees=()
  while IFS=',' read -r idx free; do
    idx="${idx// /}"
    free="${free// /}"
    [ -z "$idx" ] && continue
    idxs+=("$idx")
    frees+=("$free")
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)

  # insertion sort by free memory desc
  local n=${#idxs[@]} i j
  for ((i=1; i<n; i++)); do
    local key_idx=${idxs[i]} key_free=${frees[i]}
    j=$((i-1))
    while [ $j -ge 0 ] && [ "${frees[j]}" -lt "$key_free" ]; do
      idxs[$((j+1))]=${idxs[j]}; frees[$((j+1))]=${frees[j]}; j=$((j-1))
    done
    idxs[$((j+1))]=$key_idx; frees[$((j+1))]=$key_free
  done

  SORTED_GPU_LIST=("${idxs[@]}")
  log "GPU ranking by free memory:"
  for ((i=0; i<n; i++)); do
    log "  GPU ${idxs[i]}: ${frees[i]} MiB free"
  done

  OLLAMA_GPU="${idxs[0]}"
  if [ "$n" -gt 1 ] && [ "${frees[1]}" -ge 2000 ]; then
    EMBED_GPU="${idxs[1]}"
  else
    EMBED_GPU="$OLLAMA_GPU"
  fi
  log "Dense retriever → GPU ${EMBED_GPU}"
}

verify_full_gpu_offload() {
  # Send a probe request to force model loading, then check the ollama log
  # for partial offload. Returns 0 if all model layers are on GPU, 1 otherwise.
  local probe_timeout=120
  log "  Probing Ollama with a small request to trigger model load..."
  curl -sS -m "$probe_timeout" "http://localhost:${OLLAMA_PORT}/api/generate" \
       -H "Content-Type: application/json" \
       -d "{\"model\":\"${OLLAMA_MODEL}\",\"prompt\":\"ok\",\"stream\":false,\"options\":{\"num_predict\":3}}" \
       >/dev/null 2>&1 || true
  sleep 2

  local last_offload
  last_offload=$(grep 'msg="offload to' /tmp/ollama.log 2>/dev/null | tail -1)
  if [ -z "$last_offload" ]; then
    log "  WARN: no 'offload to' entry in ollama log yet; cannot verify. Assuming GPU."
    return 0
  fi

  # Parse layers.model=N and layers.offload=M
  local total offloaded
  total=$(echo "$last_offload" | grep -oE 'layers\.model=[0-9]+' | cut -d= -f2)
  offloaded=$(echo "$last_offload" | grep -oE 'layers\.offload=[0-9]+' | cut -d= -f2)

  if [ -z "$total" ] || [ -z "$offloaded" ]; then
    log "  WARN: could not parse offload counts. Assuming GPU."
    return 0
  fi

  if [ "$offloaded" -lt "$total" ]; then
    log "  ✗ Partial offload: ${offloaded}/${total} layers on GPU (rest on CPU)"
    return 1
  fi
  log "  ✓ Full offload: all ${total} layers on GPU"
  return 0
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
  # Try each candidate GPU in order until one gives full GPU offload.
  # If any existing Ollama process is running, kill it first to start clean —
  # we cannot trust that it loaded the model fully on GPU.
  if curl -sSf "http://localhost:${OLLAMA_PORT}/api/tags" >/dev/null 2>&1; then
    log "Existing Ollama detected — killing it to ensure clean GPU placement"
    pkill -9 -f "ollama serve" 2>/dev/null || true
    sleep 5
  fi

  # Make sure we have the model pulled before trying placements
  # (pull a separate ollama serve instance briefly if needed)
  log "Ensuring model ${OLLAMA_MODEL} is available ..."
  CUDA_VISIBLE_DEVICES="${SORTED_GPU_LIST[0]}" \
    nohup ollama serve > /tmp/ollama_pull.log 2>&1 &
  sleep 6
  ollama pull "${OLLAMA_MODEL}" >/dev/null
  pkill -9 -f "ollama serve" 2>/dev/null || true
  sleep 3
  log "Model ready"

  # Try each GPU until one gives full offload
  for gpu in "${SORTED_GPU_LIST[@]}"; do
    log "Trying Ollama on GPU ${gpu} ..."
    CUDA_VISIBLE_DEVICES="${gpu}" OLLAMA_NUM_PARALLEL=1 \
      nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 8

    if ! curl -sSf "http://localhost:${OLLAMA_PORT}/api/tags" >/dev/null; then
      log "  Ollama failed to start on GPU ${gpu}; trying next"
      pkill -9 -f "ollama serve" 2>/dev/null || true
      sleep 3
      continue
    fi

    if verify_full_gpu_offload; then
      OLLAMA_GPU="$gpu"
      log "Ollama running on GPU ${OLLAMA_GPU} with full offload ✓"
      return
    fi

    log "  Killing partial-offload Ollama and trying next GPU"
    pkill -9 -f "ollama serve" 2>/dev/null || true
    sleep 5
  done

  fail "No GPU could host ${OLLAMA_MODEL} with full offload. Check /tmp/ollama.log and free up GPU memory."
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
rank_gpus
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
