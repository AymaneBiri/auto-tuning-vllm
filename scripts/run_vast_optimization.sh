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
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO_DIR}/run_logs"
RUN_LOG="${LOG_DIR}/study_${RUN_TS}.log"
SMOKE_LOG="${LOG_DIR}/smoke_${RUN_TS}.log"
CONTAINER_API_KEY="d70a14830aaad424ba819caa6eadb19df7227f95988975c399d5d43de1ca738e"

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

# Free GB on the filesystem holding $1, walking up to the nearest existing
# parent so it works on a cache dir that has not been created yet.
free_gb() {
  local target="$1"
  while [[ ! -d "$target" && "$target" != "/" ]]; do target="$(dirname "$target")"; done
  df -BG --output=avail "$target" | tail -1 | tr -dc '0-9'
}

fs_of() {
  local target="$1"
  while [[ ! -d "$target" && "$target" != "/" ]]; do target="$(dirname "$target")"; done
  df --output=source "$target" | tail -1
}

# SIGTERM, then SIGKILL if it is still alive. A bare 'kill' + 'wait' can block
# forever: vLLM runs EngineCore in its own process and the parent's graceful
# shutdown can wedge on an engine that is mid-cudagraph-capture, which would
# hang this script with the instance still billing.
stop_vllm() {
  local pid="$1" waited=0
  kill -TERM "$pid" 2>/dev/null || true
  while [[ $waited -lt 30 ]] && kill -0 "$pid" 2>/dev/null; do
    sleep 2
    waited=$(( waited + 2 ))
  done
  if kill -0 "$pid" 2>/dev/null; then
    warn "vLLM ignored SIGTERM after ${waited}s - sending SIGKILL."
    kill -KILL "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Stop-on-exit. Registered before any long-running work so that a crash, an
# OOM, or a Ctrl-C still parks the instance instead of billing overnight.
# ---------------------------------------------------------------------------
STOP_ARMED=0
VASTAI_BIN=""

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
    warn "Auto-stop was never armed (missing credentials or no vastai CLI)."
    warn "THIS INSTANCE IS STILL BILLING. Stop it yourself:"
    warn "  vastai stop instance ${CONTAINER_ID:-<ID>}"
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

  # Resolved at arm time: activating the venv reorders PATH, and pip may have
  # dropped the CLI somewhere that is not on it.
  "$VASTAI_BIN" stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" \
    || warn "Stop command failed. Stop it manually: vastai stop instance ${CONTAINER_ID}"

  exit $rc
}
trap finish EXIT

# Without these, a signal-terminated bash reports $? == 0 to the EXIT trap and
# the summary claims the study "finished cleanly" after a Ctrl-C.
trap 'warn "Received SIGINT."; exit 130' INT
trap 'warn "Received SIGTERM."; exit 143' TERM

# ---------------------------------------------------------------------------
# Pre-flight. Everything that could make the run pointless is checked here,
# before we spend an hour of 8xH200 rental finding out.
# ---------------------------------------------------------------------------
log "Pre-flight checks"

[[ -f "$CONFIG" ]] || die "Config not found: ${CONFIG}"

MODEL="$(grep -E '^\s{2}model:' "$CONFIG" | head -1 | sed 's/.*model:\s*//' | tr -d '"'"'"' ' | sed 's/#.*//')"
STUDY_NAME="$(grep -A2 '^study:' "$CONFIG" | grep -E '^\s+name:' | head -1 | sed 's/.*name:\s*//' | tr -d '"'"'"' ' | sed 's/#.*//')"
MAX_CONC="$(grep -E '^\s+max_concurrent_trials:' "$CONFIG" | head -1 | tr -dc '0-9')"
MAX_CONC="${MAX_CONC:-1}"
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
    # Never fatal. A failed install here used to abort the whole pre-flight with
    # STOP_ARMED still 0 - i.e. the one failure mode that leaves the instance
    # billing forever. Warn and carry on instead.
    if ! command -v vastai >/dev/null 2>&1; then
      pip install --quiet vastai 2>/dev/null \
        || pip install --quiet --break-system-packages vastai 2>/dev/null \
        || true
      hash -r
    fi
    VASTAI_BIN="$(command -v vastai || true)"
    if [[ -n "$VASTAI_BIN" ]]; then
      STOP_ARMED=1
      echo "  auto-stop: armed for instance ${CONTAINER_ID} (${VASTAI_BIN})"
    else
      warn "vastai CLI unavailable - auto-stop is DISABLED."
      warn "Stop it yourself when this ends: vastai stop instance ${CONTAINER_ID}"
    fi
  fi
fi

# --- GPUs -----------------------------------------------------------------
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found - no GPU driver?"
command -v curl >/dev/null 2>&1 || die "curl not found - the smoke test needs it to poll the server."
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "  GPUs   : ${GPU_COUNT}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/           /'
if [[ "$GPU_COUNT" -lt "$MAX_CONC" ]]; then
  warn "Config sets max_concurrent_trials: ${MAX_CONC} but only ${GPU_COUNT} GPU(s) are visible."
  warn "Trials will queue instead of running ${MAX_CONC}-wide. Lower max_concurrent_trials to match."
fi

# --- HF token (Gemma is a gated repo) -------------------------------------
[[ -n "${HF_TOKEN:-}" ]] || die "HF_TOKEN is not set. Gemma is gated: accept the license on the model page, then 'export HF_TOKEN=hf_xxx'."
export HF_TOKEN

# Trial vLLM servers are spawned with os.environ.copy() from the Ray worker,
# which inherits from the Ray head this script starts. Export HF_HOME here so
# the cache location actually reaches them - otherwise every trial looks in the
# default cache, misses, and re-pulls ~35 GB.
if [[ -n "${HF_HOME:-}" ]]; then
  export HF_HOME
  echo "  HF_HOME: ${HF_HOME}"
else
  echo "  HF_HOME: (unset -> ~/.cache/huggingface)"
fi

# --- disk ------------------------------------------------------------------
# The venv and the weights can land on different mounts (a mounted cache volume
# is common), so check whichever filesystem each one actually goes to.
CACHE_DIR="${HF_HOME:-${HOME:-/root}/.cache/huggingface}"
REPO_GB="$(free_gb "$REPO_DIR")"
CACHE_GB="$(free_gb "$CACHE_DIR")"
REPO_FS="$(fs_of "$REPO_DIR")"
CACHE_FS="$(fs_of "$CACHE_DIR")"
if [[ "$REPO_FS" == "$CACHE_FS" ]]; then
  echo "  disk   : ${REPO_GB} GB free on ${REPO_FS} (venv + weights)"
  [[ "$REPO_GB" -ge 80 ]] || die "Only ${REPO_GB} GB free. Need ~35 GB for weights plus ~20 GB for the venv (torch + vLLM)."
else
  echo "  disk   : ${REPO_GB} GB free on ${REPO_FS} (venv)"
  echo "           ${CACHE_GB} GB free on ${CACHE_FS} (HF cache)"
  [[ "$REPO_GB" -ge 30 ]]  || die "Only ${REPO_GB} GB free on ${REPO_FS}; the venv (torch + vLLM) needs ~20 GB."
  [[ "$CACHE_GB" -ge 45 ]] || die "Only ${CACHE_GB} GB free on ${CACHE_FS}; the weights need ~35 GB."
fi

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
log "Setting up the Python environment"
if [[ ! -d "$VENV" ]]; then
  if command -v uv >/dev/null 2>&1; then
    # --seed puts pip inside the venv. Without it 'pip' after activation silently
    # falls through PATH to the system pip, so 'pip list' shows the wrong
    # environment and 'pip install' fixes packages nobody is running.
    uv venv --python 3.12 --seed "$VENV"
  else
    python3 -m venv "$VENV"
  fi
fi
# set +u: some activate scripts reference unset vars (PS1) and would abort here.
set +u
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
set -u

# Check for the package inside the venv, not just a binary on PATH - a
# system-wide auto-tune-vllm would otherwise mask an empty venv, and the venv is
# what Ray workers are pointed at.
if ! python -c 'import auto_tune_vllm' >/dev/null 2>&1; then
  log "Installing auto-tune-vllm (pulls vLLM + torch; this takes a while)"
  if command -v uv >/dev/null 2>&1; then
    # --python is explicit on purpose: never rely on activation alone to decide
    # which interpreter gets the packages.
    uv pip install --python "${VENV}/bin/python" -e .
  else
    pip install --upgrade pip && pip install -e .
  fi
fi

# Informational only - never fatal. Read versions from distribution metadata
# rather than __version__ attributes, which not every package defines
# (guidellm does not), and which would otherwise require importing vLLM.
python - <<'PY' || warn "Could not read package versions (non-fatal)."
from importlib.metadata import version, PackageNotFoundError
for pkg in ("vllm", "guidellm", "optuna", "ray"):
    try:
        print(f"  {pkg:<9}: {version(pkg)}")
    except PackageNotFoundError:
        print(f"  {pkg:<9}: NOT INSTALLED")
PY

# Everything below runs through the venv's interpreter, so it reports on the
# environment the trials actually use.
python - <<'PY' || die "Not running inside ${VENV} - the environment setup is broken."
import sys
raise SystemExit(0 if sys.prefix != sys.base_prefix else 1)
PY

# flashinfer ships as several separate distributions that must be version-locked.
# When they skew, the mismatch raises at *import* time inside EngineCore, which
# surfaces ~10 minutes in as a stack trace after the weights have already loaded.
python - "$VENV" <<'PY' || die "flashinfer version skew (see above). Fix it before spending GPU time."
import sys
from importlib.metadata import distributions

venv = sys.argv[1]
found = {}
for dist in distributions():
    name = (dist.metadata["Name"] or "").lower()
    if name.startswith("flashinfer"):
        found[name] = dist.version

if not found:
    print("  flashinfer: not installed")
    raise SystemExit(0)
if len(set(found.values())) == 1:
    print(f"  flashinfer: {next(iter(found.values()))} ({len(found)} packages, consistent)")
    raise SystemExit(0)

print("  flashinfer VERSION SKEW - the engine will fail on import:")
for name in sorted(found):
    print(f"    {name:<24} {found[name]}")
target = found.get("flashinfer-python") or max(found.values())
for name in sorted(found):
    if found[name] != target:
        print(f'  fix: uv pip install --python {venv}/bin/python "{name}=={target}"')
raise SystemExit(1)
PY

# The CLI is typer-based, and a typer/click mismatch makes every 'raise
# typer.Exit' escape as an uncaught traceback with a non-zero status - including
# the successful paths. Catch it here rather than mid-study.
auto-tune-vllm --help >/dev/null 2>&1 \
  || die "The auto-tune-vllm CLI errors out on --help (typer/click mismatch is the usual cause). Every trial goes through it."

echo "  NOTE: the config's VLLM_FLASH_ATTN_VERSION=4 arm needs a build that ships"
echo "        FA4 for Hopper. On an older vLLM it silently falls back to FA3."

# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------
if [[ $DO_DOWNLOAD -eq 1 ]]; then
  log "Downloading ${MODEL}"
  # Use the Python API, not the 'hf' CLI. The CLI leaks an uncaught
  # click.exceptions.Exit(0) traceback and returns non-zero even when the
  # download succeeded, which set -e turns into a spurious abort. The API
  # returns the snapshot path and raises only on genuine failures.
  MODEL_PATH="$(
    python - "$MODEL" <<'PY' | tail -1
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1]))
PY
  )" || die "Model download failed for ${MODEL}"

  # Verify the weights are really there rather than trusting an exit code.
  [[ -d "$MODEL_PATH" ]] || die "Snapshot path does not exist: ${MODEL_PATH}"
  SHARDS="$(find "$MODEL_PATH" -name '*.safetensors' | wc -l)"
  [[ "$SHARDS" -gt 0 ]] || die "No .safetensors weights under ${MODEL_PATH}"
  echo "  path  : ${MODEL_PATH}"
  echo "  files : ${SHARDS} safetensors shards, $(du -shL "$MODEL_PATH" 2>/dev/null | cut -f1) on disk"
  log "Download complete"
else
  log "Skipping download (--skip-download)"
fi

# ---------------------------------------------------------------------------
# Smoke test: one short vLLM start to confirm the two things that would
# silently invalidate the study - actual KV capacity, and whether FA4 is real.
# ---------------------------------------------------------------------------
if [[ $DO_SMOKE -eq 1 ]]; then
  # Match the timeout the trials get, rather than hardcoding a stricter one that
  # fails pre-flight on a cold start the study itself would have tolerated.
  SMOKE_TIMEOUT="$(grep -E '^\s+VLLM_STARTUP_TIMEOUT:' "$CONFIG" | head -1 | tr -dc '0-9')"
  SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-1800}"
  log "Smoke test: starting vLLM once on GPU 0 (timeout ${SMOKE_TIMEOUT}s)"

  # PYTHONUNBUFFERED: vLLM logs to stdout, which is block-buffered once
  # redirected to a file.
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 VLLM_FLASH_ATTN_VERSION=4 \
    vllm serve "$MODEL" \
      --tensor-parallel-size 1 \
      --max-model-len 8192 \
      --gpu-memory-utilization 0.92 \
      --no-enforce-eager \
      --kv-cache-dtype fp8 \
      --port "$SMOKE_PORT" > "$SMOKE_LOG" 2>&1 &
  SMOKE_PID=$!

  # Readiness is the HTTP endpoint answering, NOT a string in the log. Log lines
  # get renamed between vLLM releases and the engine core writes from its own
  # process, so grepping for a magic phrase can spin forever against a server
  # that came up fine. The log is still parsed below, but only for reporting.
  SMOKE_STATE="timeout"
  SMOKE_DEADLINE=$(( SECONDS + SMOKE_TIMEOUT ))
  while [[ $SECONDS -lt $SMOKE_DEADLINE ]]; do
    if curl -fsS -m 5 "http://127.0.0.1:${SMOKE_PORT}/v1/models" >/dev/null 2>&1; then
      SMOKE_STATE="ready"
      break
    fi
    if ! kill -0 "$SMOKE_PID" 2>/dev/null; then
      SMOKE_STATE="died"
      break
    fi
    sleep 5
  done

  stop_vllm "$SMOKE_PID"

  # The study grabs all 8 GPUs immediately after this. A straggler still holding
  # 92% of GPU 0 would take out the first trial with an OOM that looks like a
  # bad parameter combination.
  GPU0_MB=0
  for _ in $(seq 1 24); do
    GPU0_MB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -dc '0-9')"
    [[ "${GPU0_MB:-0}" -lt 2000 ]] && break
    sleep 5
  done
  if [[ "${GPU0_MB:-0}" -ge 2000 ]]; then
    warn "GPU 0 still holds ${GPU0_MB} MiB two minutes after the smoke test."
    warn "Check 'nvidia-smi' for orphaned vLLM processes before trusting trial 1."
  fi

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
  echo

  # Gate on whether the server actually served. A missing log line means the
  # grep patterns above have gone stale, which is worth a warning but is not a
  # reason to abort a healthy run.
  case "$SMOKE_STATE" in
    ready)
      log "Smoke test passed: vLLM answered on port ${SMOKE_PORT}."
      ;;
    died)
      die "vLLM exited during the smoke test. See ${SMOKE_LOG}"
      ;;
    timeout)
      die "vLLM did not answer on port ${SMOKE_PORT} within ${SMOKE_TIMEOUT}s. See ${SMOKE_LOG}"
      ;;
  esac
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

# A previous aborted run can leave a Ray head behind; starting a second one
# fails on a port conflict. Attach to it instead if it is already up.
RAY_FLAG="--start-ray-head"
if ray status >/dev/null 2>&1; then
  warn "Existing Ray cluster detected - attaching instead of starting a new head."
  warn "If it is stale, cancel and run: ray stop --force"
  RAY_FLAG="--no-start-ray-head"
fi

# max_concurrent_trials comes from the config (8).
auto-tune-vllm optimize \
  --config "$CONFIG" \
  --venv-path "$VENV" \
  "$RAY_FLAG" \
  --verbose 2>&1 | tee "$RUN_LOG"

# The trap handles the summary and the instance stop.
