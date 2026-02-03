#!/usr/bin/env bash
set -euo pipefail

LOG="/var/log/zero-startup.log"
exec > >(tee -a "$LOG") 2>&1

# read run config from instance metadata
META="http://metadata.google.internal/computeMetadata/v1/instance/attributes"

get_meta() {
  curl -fs -H "Metadata-Flavor: Google" "$META/$1" || true
}

REPO_URL="$(get_meta repo_url)"
REPO_REF="$(get_meta repo_ref)"
RUN_CMD="$(get_meta run_cmd)"
AUTO_SHUTDOWN_MINUTES="$(get_meta auto_shutdown_minutes)"
GCS_BUCKET="$(get_meta gcs_bucket)"

if [[ -z "$AUTO_SHUTDOWN_MINUTES" ]]; then
  AUTO_SHUTDOWN_MINUTES=240
fi

# safety net so we don't burn money if something hangs
shutdown -h "+$AUTO_SHUTDOWN_MINUTES" || true

if [[ -z "$REPO_URL" ]]; then
  echo "repo_url is empty, nothing to run"
  shutdown -h now || true
  exit 0
fi

RUN_DIR="/opt/zero"
mkdir -p "$RUN_DIR"

# best-effort deps; dlvms already have most of this
apt-get update -y
apt-get install -y git python3-venv

if [[ -d "$RUN_DIR/.git" ]]; then
  git -C "$RUN_DIR" fetch --all
  git -C "$RUN_DIR" checkout "$REPO_REF"
else
  git clone "$REPO_URL" "$RUN_DIR"
  git -C "$RUN_DIR" checkout "$REPO_REF"
fi

python3 -m venv "$RUN_DIR/.venv"
source "$RUN_DIR/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$RUN_DIR/requirements.txt"

cd "$RUN_DIR"

if [[ -z "$RUN_CMD" ]]; then
  # default to a stage 0 run across all visible gpus
  GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
  if [[ "$GPU_COUNT" -lt 1 ]]; then
    GPU_COUNT=1
  fi
  RUN_CMD="torchrun --nproc_per_node=$GPU_COUNT src/train.py --stage 0"
fi

echo "running: $RUN_CMD"
bash -lc "$RUN_CMD"

if [[ -n "$GCS_BUCKET" ]] && command -v gsutil >/dev/null 2>&1; then
  RUN_ID=$(date +"%Y%m%d-%H%M%S")
  gsutil -m cp "$LOG" "$GCS_BUCKET/zero-runs/$RUN_ID-startup.log" || true
fi

shutdown -h now || true
