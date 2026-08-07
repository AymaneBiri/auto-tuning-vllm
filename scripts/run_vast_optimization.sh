#!/usr/bin/env bash
#
# Launch the Gemma 4 31B FP8 tuning study on a rented vast.ai 8xH200 instance,
# then stop the instance so it stops burning GPU credits.
#
# Usage (from the repo root):
#   export HF_TOKEN=hf_xxx
#   nohup ./scripts/run_vast_optimization.sh > run.out 2>&1 &
#
# Run it detached (nohup/tmux/screen). If you run it in a bare SSH session and
# the connection drops, the script dies and the instance keeps billing.
#
# Flags:
#   --config PATH       study config (default: examples/gemma4_31b_optimization.yaml)
#   --no-stop           leave the instance running when the study finishes
#   --skip-download     assume the model is already in the HF cache
#   --skip-smoke-test   skip the pre-flight vLLM sanity check
#   --grace SECONDS     cancellation window before stopping (default: 120)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="examples/gemma4_31b_optimization.yaml"
VENV="${REPO_DIR}/venv"
AUTO_STOP=1
DO_DOWNLOAD=1
DO_SMOKE=1
GRACE=120
SMOKE_PORT=8765
EXPECTED_GPUS=8
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO_DIR}/run_logs"
RUN_LOG="${LOG_DIR}/study_${RUN_TS}.log"
SMOKE_LOG="${LOG_DIR}/smoke_${RUN_TS}.log"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)          CONFIG="$2"; shift 2 ;;
    --no-stop)         AUTO_STOP=0; shift ;;
    --skip-download)   DO_DOWNLOAD=0; shift ;;
    --skip-smoke-test) DO_SMOKE=0; shift ;;
    --grace)           GRACE="$2"; shift 2 ;;
    *) echo "Unknown flag: $1" >&2; exit 2 ;;
  esac
done

cd "$REPO_DIR"
mkdir -p "$LOG_DIR"

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[fail] %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Stop-on-exit. Registered before any long-running work so that a crash, an
# OOM, or a Ctrl-C still parks the instance instead of billing overnight.
# ---------------------------------------------------------------------------
STOP_ARMED=0

finish() {
  local rc=$?
  set +e

  if [[ $rc -eq 0 ]]; then
    log "Study finished cleanly."
  elif [[ $AUTO_STOP -eq 1 ]]; then
    warn "Script exited with code ${rc}. Stopping the instance anyway."
  else
    warn "Script exited with code ${rc}."
  fi

  echo
  echo "  study db : ${REPO_DIR}/optuna_studies/${STUDY_NAME:-<study>}/study.db"
  echo "  run log  : ${RUN_LOG}"
  echo "  trial logs: $(grep -E '^\s*file_path:' "$CONFIG" 2>/dev/null | head -1 | sed 's/.*file_path:\s*//' | tr -d '"' || echo '(see config)')"
  echo
  echo "  Download results BEFORE destroying the instance. A stopped instance"
  echo "  keeps its disk (and disk billing); a destroyed one loses everything."
  echo

  if [[ $AUTO_STOP -eq 0 ]]; then
    log "--no-stop set; leaving the instance running."
    exit $rc
  fi

  if [[ $STOP_ARMED -eq 0 ]]; then
    warn "No vast.ai stop credentials were found; cannot stop automatically."
    warn "Stop it yourself: vastai stop instance <ID>"
    exit $rc
  fi

  log "Stopping vast.ai instance ${CONTAINER_ID} in ${GRACE}s..."
  if [[ -t 0 ]]; then
    if read -r -t "$GRACE" -p "  Press ENTER to CANCEL the shutdown: "; then
      log "Shutdown cancelled. Instance left running."
      exit $rc
    fi
  else
    sleep "$GRACE"
  fi

  vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" \
    || warn "Stop command failed. Stop it manually: vastai stop instance ${CONTAINER_ID}"

  exit $rc
}
trap finish EXIT

# ---------------------------------------------------------------------------
# Pre-flight. Everything that could make the run pointless is checked here,
# before we spend an hour of 8xH200 rental finding out.
# ---------------------------------------------------------------------------
log "Pre-flight checks"

[[ -f "$CONFIG" ]] || die "Config not found: ${CONFIG}"

MODEL="$(grep -E '^\s{2}model:' "$CONFIG" | head -1 | sed 's/.*model:\s*//' | tr -d '"'"'"' ' | sed 's/#.*//')"
STUDY_NAME="$(grep -A2 '^study:' "$CONFIG" | grep -E '^\s+name:' | head -1 | sed 's/.*name:\s*//' | tr -d '"'"'"' ' | sed 's/#.*//')"
[[ -n "$MODEL" ]] || die "Could not parse the model out of ${CONFIG}"
echo "  config : ${CONFIG}"
echo "  study  : ${STUDY_NAME}"
echo "  model  : ${MODEL}"

# --- vast.ai stop credentials, checked up front ---------------------------
if [[ $AUTO_STOP -eq 1 ]]; then
  # Injected by vast.ai into the container. In an SSH session they are
  # sometimes missing from the shell env even though they exist on the box.
  if [[ -z "${CONTAINER_ID:-}" || -z "${CONTAINER_API_KEY:-}" ]] && [[ -f /etc/environment ]]; then
    set +u; . /etc/environment; set -u
  fi
  if [[ -z "${CONTAINER_ID:-}" || -z "${CONTAINER_API_KEY:-}" ]]; then
    warn "CONTAINER_ID / CONTAINER_API_KEY not set - auto-stop is DISABLED."
    warn "Check 'env | grep CONTAINER', or re-run with --no-stop to silence this."
  else
    command -v vastai >/dev/null 2>&1 || pip install --quiet vastai
    STOP_ARMED=1
    echo "  auto-stop: armed for instance ${CONTAINER_ID}"
  fi
fi

# --- GPUs -----------------------------------------------------------------
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found - no GPU driver?"
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "  GPUs   : ${GPU_COUNT}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/           /'
if [[ "$GPU_COUNT" -lt "$EXPECTED_GPUS" ]]; then
  warn "Config sets max_concurrent_trials: 8 but only ${GPU_COUNT} GPU(s) are visible."
  warn "Trials will queue instead of running 8-wide. Lower max_concurrent_trials to match."
fi

# --- HF token (Gemma is a gated repo) -------------------------------------
[[ -n "${HF_TOKEN:-}" ]] || die "HF_TOKEN is not set. Gemma is gated: accept the license on the model page, then 'export HF_TOKEN=hf_xxx'."

# --- disk ------------------------------------------------------------------
AVAIL_GB="$(df -BG --output=avail "$REPO_DIR" | tail -1 | tr -dc '0-9')"
echo "  disk   : ${AVAIL_GB} GB free"
[[ "$AVAIL_GB" -ge 80 ]] || die "Only ${AVAIL_GB} GB free. Need ~35 GB for weights plus ~20 GB for the venv (torch + vLLM)."

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
log "Setting up the Python environment"
if [[ ! -d "$VENV" ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 "$VENV"
  else
    python3 -m venv "$VENV"
  fi
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

if ! command -v auto-tune-vllm >/dev/null 2>&1; then
  log "Installing auto-tune-vllm (pulls vLLM + torch; this takes a while)"
  if command -v uv >/dev/null 2>&1; then
    uv pip install -e .
  else
    pip install --upgrade pip && pip install -e .
  fi
fi

python -c 'import vllm, guidellm; print(f"  vLLM     : {vllm.__version__}\n  guidellm : {guidellm.__version__}")'
echo "  NOTE: the config's VLLM_FLASH_ATTN_VERSION=4 arm needs a build that ships"
echo "        FA4 for Hopper. On an older vLLM it silently falls back to FA3."

# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------
if [[ $DO_DOWNLOAD -eq 1 ]]; then
  log "Downloading ${MODEL}"
  # 'hf' is the current CLI; 'huggingface-cli' is the pre-2025 name.
  if command -v hf >/dev/null 2>&1; then
    hf download "$MODEL"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$MODEL"
  else
    python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL}')"
  fi
  log "Download complete"
else
  log "Skipping download (--skip-download)"
fi

# ---------------------------------------------------------------------------
# Smoke test: one short vLLM start to confirm the two things that would
# silently invalidate the study - actual KV capacity, and whether FA4 is real.
# ---------------------------------------------------------------------------
if [[ $DO_SMOKE -eq 1 ]]; then
  log "Smoke test: starting vLLM once on GPU 0 (~3-5 min)"

  CUDA_VISIBLE_DEVICES=0 VLLM_FLASH_ATTN_VERSION=4 \
    vllm serve "$MODEL" \
      --tensor-parallel-size 1 \
      --max-model-len 8192 \
      --gpu-memory-utilization 0.92 \
      --no-enforce-eager \
      --limit-mm-per-prompt image=0,video=0 \
      --kv-cache-dtype fp8 \
      --port "$SMOKE_PORT" > "$SMOKE_LOG" 2>&1 &
  SMOKE_PID=$!

  for _ in $(seq 1 120); do
    grep -qiE "Application startup complete|GPU KV cache size" "$SMOKE_LOG" && break
    kill -0 "$SMOKE_PID" 2>/dev/null || break
    sleep 10
  done

  kill "$SMOKE_PID" 2>/dev/null || true
  wait "$SMOKE_PID" 2>/dev/null || true

  echo
  echo "  --- KV capacity -------------------------------------------------"
  grep -iE "GPU KV cache size|Maximum concurrency" "$SMOKE_LOG" | sed 's/^/  /' \
    || warn "No KV cache line found - check ${SMOKE_LOG}"
  echo "  Config assumes ~140 concurrent seqs at fp8 KV against rate: 160."
  echo "  If the real ceiling is ABOVE 160, memory stops binding and you should"
  echo "  raise rate (and the top max_num_seqs option) to keep fp8 meaningful."
  echo
  echo "  --- attention backend / FA version -------------------------------"
  grep -iE "flash.?attn|flashinfer|attention backend|FA[234]" "$SMOKE_LOG" | head -10 | sed 's/^/  /' \
    || warn "No attention backend line found - check ${SMOKE_LOG}"
  echo "  If this does not say FA4, drop VLLM_FLASH_ATTN_VERSION from the config"
  echo "  (halves the grid to 36) rather than measuring a fallback against itself."
  echo
  echo "  --- cudagraph mode ------------------------------------------------"
  grep -iE "cudagraph|capturing" "$SMOKE_LOG" | head -5 | sed 's/^/  /' || true
  echo
  echo "  full smoke log: ${SMOKE_LOG}"

  if ! grep -qi "GPU KV cache size" "$SMOKE_LOG"; then
    die "vLLM never reported a KV cache size - it likely failed to start. See ${SMOKE_LOG}"
  fi
else
  log "Skipping smoke test (--skip-smoke-test)"
fi

# ---------------------------------------------------------------------------
# The study
# ---------------------------------------------------------------------------
log "Starting optimization: ${STUDY_NAME}"
echo "  Expect ~72 trials / 8 GPUs at ~8 min each => roughly 1.2 h."
echo "  Follow along: tail -f ${RUN_LOG}"
echo "  Trial detail: auto-tune-vllm logs --study-name ${STUDY_NAME} --log-path /tmp/auto-tune-vllm-logs"
echo

# max_concurrent_trials comes from the config (8). Ray head is started here
# because a fresh vast.ai box has no cluster running.
auto-tune-vllm optimize \
  --config "$CONFIG" \
  --venv-path "$VENV" \
  --start-ray-head \
  --verbose 2>&1 | tee "$RUN_LOG"

# The trap handles the summary and the instance stop.
