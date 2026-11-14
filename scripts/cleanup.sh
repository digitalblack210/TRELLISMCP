#!/usr/bin/env bash
# Frees the disk space a finished TRELLIS install no longer needs. Safe to rerun.
# It removes download caches and build sources only: nothing needed to make a model,
# and nothing scripts/install.sh would not fetch again. The weights are left alone.
#
#   bash scripts/cleanup.sh           caches and build sources
#   bash scripts/cleanup.sh --jobs    also delete the folders of jobs that are not running
set -eo pipefail
T=${TRELLIS_HOME:-$HOME/trellis}
JOBS=${TRELLIS_MCP_JOBS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/jobs}
jobs_too=
for arg in "$@"; do
  case $arg in
    --jobs) jobs_too=1 ;;
    *) echo "Usage: bash scripts/cleanup.sh [--jobs]"; exit 1 ;;
  esac
done
if [ ! -x "$T/env/bin/python" ]; then
  echo "TRELLIS is not installed at $T. Set TRELLIS_HOME to where you installed it."
  exit 1
fi
stage() { echo; echo "=== STAGE: $* ==="; }
free_kb() { df -k --output=avail "$1" | tail -1 | tr -d ' '; }
size_kb() { du -sk "$1" 2>/dev/null | cut -f1; }
gb() { awk -v kb="$1" 'BEGIN { printf "%.1f GB", (kb > 0 ? kb : 0) / 1048576 }'; }
before=$(free_kb "$T")

stage pip cache
# The wheels pip downloaded: PyTorch alone is several gigabytes. They are already installed.
"$T/env/bin/python" -m pip cache purge || true

stage micromamba cache
# The CUDA toolkit archives. The environment itself is untouched.
if [ -x "$T/bin/micromamba" ]; then
  MAMBA_ROOT_PREFIX=$T/mamba "$T/bin/micromamba" clean --all --yes || true
fi

stage build sources
# nvdiffrast and mip-splatting were built from these and installed into the environment.
rm -rf "$T/ext" "$T/TRELLIS.partial"
echo "removed $T/ext"

jobs_kb=0
if [ -n "$jobs_too" ]; then
  stage jobs
  if [ -d "$JOBS" ]; then
    for folder in "$JOBS"/*/; do
      [ -f "$folder/job.json" ] || continue
      if grep -q '"status": "running"' "$folder/job.json"; then
        echo "kept $(basename "$folder"): still running"
        continue
      fi
      jobs_kb=$((jobs_kb + $(size_kb "$folder")))
      rm -rf "$folder"
    done
    echo "removed finished jobs from $JOBS"
  else
    echo "no job folder at $JOBS"
  fi
fi

after=$(free_kb "$T")
echo
echo "=== CLEANUP DONE === freed about $(gb $((after - before))) from the TRELLIS install"
[ -z "$jobs_too" ] || echo "and $(gb "$jobs_kb") of jobs"
if grep -qi microsoft /proc/version 2>/dev/null; then
  echo
  echo "WSL does not hand freed space back to Windows by itself. To get it back, run in PowerShell:"
  echo "  wsl --shutdown"
  echo "  wsl --manage ${WSL_DISTRO_NAME:-<your distribution>} --set-sparse true"
fi
