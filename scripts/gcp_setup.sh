#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID=""
ZONE=""

usage() {
  cat <<'USAGE'
Usage: ./scripts/gcp_setup.sh --project <PROJECT_ID> [--zone <ZONE>]

This script:
- reinstalls the Google Cloud CLI from the official tarball
- uses your existing python3 (no reinstall)
- adds gcloud to your PATH (zsh)
- sets the active project
- enables the Compute API

Optional:
- --zone <ZONE> lets us sanity check GPU availability
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project|-p)
      PROJECT_ID="$2"
      shift 2
      ;;
    --zone|-z)
      ZONE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$PROJECT_ID" ]]; then
  echo "Missing --project <PROJECT_ID>"
  usage
  exit 1
fi

SDK_DIR="$HOME/google-cloud-sdk"
TMP_TAR="/tmp/google-cloud-cli.tgz"

# nuke any bad overrides
unset CLOUDSDK_PYTHON || true

OS="$(uname -s)"
ARCH="$(uname -m)"

if [[ "$OS" == "Darwin" ]]; then
  if [[ "$ARCH" == "arm64" ]]; then
    PKG="google-cloud-cli-darwin-arm.tar.gz"
  elif [[ "$ARCH" == "x86_64" ]]; then
    PKG="google-cloud-cli-darwin-x86_64.tar.gz"
  else
    echo "Unsupported mac arch: $ARCH"
    exit 1
  fi
elif [[ "$OS" == "Linux" ]]; then
  if [[ "$ARCH" == "x86_64" ]]; then
    PKG="google-cloud-cli-linux-x86_64.tar.gz"
  elif [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
    PKG="google-cloud-cli-linux-arm.tar.gz"
  else
    echo "Unsupported linux arch: $ARCH"
    exit 1
  fi
else
  echo "Unsupported OS: $OS"
  exit 1
fi

echo "Installing Google Cloud CLI from $PKG"

if [[ -d "$SDK_DIR" ]]; then
  backup="$SDK_DIR.bak.$(date +%Y%m%d-%H%M%S)"
  echo "Existing SDK found; moving to $backup"
  mv "$SDK_DIR" "$backup"
fi

curl -L -o "$TMP_TAR" "https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/$PKG"
tar -xf "$TMP_TAR" -C "$HOME"

"$SDK_DIR/install.sh" --quiet

if [[ ! -d "$SDK_DIR" ]]; then
  echo "SDK not found at $SDK_DIR after install"
  exit 1
fi

# make sure gcloud is on path for this shell
export PATH="$SDK_DIR/bin:$PATH"

PY3="$(command -v python3 || true)"
if [[ -z "$PY3" ]]; then
  echo "python3 not found in PATH. Please install a supported Python (3.9-3.14) and re-run."
  exit 1
fi
export CLOUDSDK_PYTHON="$PY3"

# add to zshrc so future shells just work
if [[ -f "$HOME/.zshrc" ]]; then
  # remove any old broken sdk sourcing lines (path.zsh.inc can blow up with set -u)
  if [[ "$OS" == "Darwin" ]]; then
    sed -i '' '/google-cloud-sdk\\/path.zsh.inc/d' "$HOME/.zshrc" || true
    sed -i '' '/google-cloud-sdk\\/completion.zsh.inc/d' "$HOME/.zshrc" || true
    sed -i '' '/CLOUDSDK_PYTHON/d' "$HOME/.zshrc" || true
  else
    sed -i '/google-cloud-sdk\\/path.zsh.inc/d' "$HOME/.zshrc" || true
    sed -i '/google-cloud-sdk\\/completion.zsh.inc/d' "$HOME/.zshrc" || true
    sed -i '/CLOUDSDK_PYTHON/d' "$HOME/.zshrc" || true
  fi

  if ! grep -q "google-cloud-sdk/bin" "$HOME/.zshrc"; then
    echo 'export PATH="$HOME/google-cloud-sdk/bin:$PATH"' >> "$HOME/.zshrc"
  fi
  if ! grep -q "CLOUDSDK_PYTHON" "$HOME/.zshrc"; then
    echo "export CLOUDSDK_PYTHON=\"$PY3\"" >> "$HOME/.zshrc"
  fi
fi

# sanity check
if ! gcloud --version; then
  echo "gcloud still not working. Try running:"
  echo "  $SDK_DIR/bin/gcloud --version"
  exit 1
fi

# set project + enable compute
# (if you see an environment tag warning, it's a policy thing; see note below)
gcloud config set project "$PROJECT_ID"

gcloud services enable compute.googleapis.com

if [[ -n "$ZONE" ]]; then
  echo "Checking GPUs in $ZONE (this can be empty if your quota isn't set yet)"
  gcloud compute accelerator-types list --filter="zone:( $ZONE )" | head -n 20 || true
fi

cat <<'NOTE'

If you see an error about an 'environment' tag, your org requires it.
You might need an admin to bind the tag. The commands look like:

  gcloud projects get-ancestors <PROJECT_ID>
  gcloud resource-manager tags keys list --parent=organizations/ORG_ID --filter=shortName=environment
  gcloud resource-manager tags values list --parent=tagKeys/TAG_KEY_ID
  gcloud resource-manager tags bindings create \
    --tag-value=tagValues/TAG_VALUE_ID \
    --parent=//cloudresourcemanager.googleapis.com/projects/<PROJECT_NUMBER>

Ping me if you want me to walk you through that.
NOTE

echo "All set. gcloud is installed and project is configured."
