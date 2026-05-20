#!/usr/bin/env bash
#
# Fine-tune CodeRankEmbed on SweLoc.
# Picks a GPU that's NOT being used by Ollama (which is running graph
# extraction in parallel).
#
# Usage (in tmux):
#   ./scripts/run_finetune.sh
#   # detach: Ctrl+B then D
#   # reattach: tmux attach -t <session>
#
# Expected runtime: 6-10 hours for 5 epochs on a 11 GB GPU.
#
set -euo pipefail

cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"           # conservative for 11 GB cards
OLLAMA_PORT="${OLLAMA_PORT:-11434}"

LOG_DIR="logs/finetune"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }
fail() { log "ERROR: $*"; exit 1; }

# ---------- find a GPU not occupied by Ollama ----------
pick_gpu_avoiding_ollama() {
  # Find Ollama's GPU (if running)
  local ollama_gpu=-1
  if curl -sSf "http://localhost:${OLLAMA_PORT}/api/tags" >/dev/null 2>&1; then
    # Look at nvidia-smi for ollama_llama_server process
    ollama_gpu=$(nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader 2>/dev/null \
                 | grep -i "ollama_llama" | head -1 | awk -F',' '{print $1}' | tr -d ' ' || echo "-1")
    # Convert UUID to index
    if [ "$ollama_gpu" != "-1" ] && [ -n "$ollama_gpu" ]; then
      ollama_gpu=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits \
                   | grep "$ollama_gpu" | awk -F',' '{print $1}' | tr -d ' ')
    fi
  fi
  log "Ollama is on GPU ${ollama_gpu} (-1 = not running)"

  # Pick the GPU with most free memory that's NOT Ollama's
  local best_idx=-1 best_free=0
  while IFS=',' read -r idx free; do
    idx="${idx// /}"; free="${free// /}"
    [ -z "$idx" ] && continue
    if [ "$idx" = "$ollama_gpu" ]; then continue; fi
    if [ "$free" -gt "$best_free" ]; then
      best_free="$free"; best_idx="$idx"
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)

  if [ "$best_idx" -lt 0 ]; then
    fail "No free GPU found (excluding Ollama's GPU)."
  fi
  if [ "$best_free" -lt 7000 ]; then
    log "WARNING: best non-Ollama GPU has only ${best_free} MiB free; training may OOM."
  fi
  log "Training GPU: ${best_idx} (${best_free} MiB free)"
  export CUDA_VISIBLE_DEVICES="$best_idx"
}

pick_gpu_avoiding_ollama

log "================================================================"
log "  Fine-tuning CodeRankEmbed on SweLoc"
log "  Epochs:     ${EPOCHS}"
log "  Batch size: ${BATCH_SIZE}"
log "  Log:        ${LOG_FILE}"
log "  Output:     data/embeddings/sweloc_finetuned/"
log "================================================================"

python scripts/train_retriever.py \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    2>&1 | tee "${LOG_FILE}"

log "================================================================"
log "  DONE — fine-tuned model at data/embeddings/sweloc_finetuned/"
log "  The pipeline will auto-use this model on the next run."
log "================================================================"
