#!/usr/bin/env bash
# TRELLIS: one picture in, a textured GLB and a sheet of views out.
# Usage: bash generate.sh /path/picture.png /path/out_dir [seed] [steps] [simplify] [texture] [more pictures...]
# Writes NAME_sSEED.glb, NAME_sSEED_views.png and NAME_sSEED_result.json into out_dir, with a log beside them.
# The server reads the "=== stage" lines in that log to report progress, so keep their form.
set -eo pipefail
T=${TRELLIS_HOME:-$HOME/trellis}
export PATH=$T/env/bin:$PATH
export LD_LIBRARY_PATH=$T/env/lib:${LD_LIBRARY_PATH:-}
export CUDA_HOME=$T/env
here=$(dirname "$(readlink -f "$0")")
mkdir -p "$2"
in=$(readlink -f "$1"); out=$(readlink -f "$2")
seed=${3:-1}; steps=${4:-0}; simplify=${5:-0.95}; texture=${6:-1024}
shift $(( $# < 6 ? $# : 6 ))
more=()
for picture in "$@"; do more+=("$(readlink -f "$picture")"); done
name=$(basename "${in%.*}")_s$seed
{
  [ -d "$T/TRELLIS" ] || { echo "TRELLIS is not installed at $T. Run scripts/install.sh first."; exit 1; }
  python "$here/generate.py" "$in" "$out" "$seed" --home "$T" --steps "$steps" --simplify "$simplify" \
    --texture "$texture" --more "${more[@]}"
  [ -f "$out/$name.glb" ] || { echo "TRELLIS finished without writing $name.glb."; exit 1; }
  echo "=== done $(date +%T)"
} > "$out/${name}_trellis.log" 2>&1
