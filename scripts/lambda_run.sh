#!/usr/bin/env bash
set -euo pipefail

# this script uses the lambda cloud cli (lai) to run one command and then stop the vm

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT_DIR/.lambda-venv"

if [[ -z "${LAMBDA_CLOUD_API_TOKEN:-}" ]]; then
  echo "missing LAMBDA_CLOUD_API_TOKEN env var"
  exit 1
fi

if [[ ! -x "$VENV/bin/lai" ]]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install lambda-ai-cloud-api-client
fi

LAI="$VENV/bin/lai"

SSH_KEY_NAME="${SSH_KEY_NAME:-}"
if [[ -z "$SSH_KEY_NAME" ]]; then
  echo "missing SSH_KEY_NAME env var (this is the key name in the lambda console)"
  exit 1
fi

MIN_GPUS="${MIN_GPUS:-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-}"
RUN_CMD="${RUN_CMD:-torchrun --nproc_per_node=$MIN_GPUS src/train.py --stage 3}"

ARGS=("--ssh-key" "$SSH_KEY_NAME" "--rm")

if [[ -n "$INSTANCE_TYPE" ]]; then
  ARGS+=("--instance-type" "$INSTANCE_TYPE")
else
  ARGS+=("--cheapest" "--available" "--min-gpus" "$MIN_GPUS")
fi

# mount the repo and run the command inside it
$LAI run "${ARGS[@]}" -v "$ROOT_DIR:/home/ubuntu/cs244c-zero" -- bash -lc "cd /home/ubuntu/cs244c-zero && $RUN_CMD"
