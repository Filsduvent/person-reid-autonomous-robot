#!/usr/bin/env bash
# USER-operated only. Qualification prepares the pinned private YAML/receipt files.
set -euo pipefail
case "${1:-}" in
  1|2|3|4) mgn_fold="$1" ;;
  *) echo 'Usage: bash docs/mgn_lodo_phase21/official_training_commands.sh <1|2|3|4>' >&2; exit 2 ;;
esac
cd /home/addirakoze/nobackup/Projects/person-reid-autonomous-robot
export CUDA_VISIBLE_DEVICES=GPU-202f74b9-f9fe-8896-5cd7-78332b0e49e2
export LODO_AUTHORIZED_GPU_UUID="$CUDA_VISIBLE_DEVICES"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mgn_launch=".local/mgn_lodo_launch/LODO-${mgn_fold}/official-20261109"
test -f "${mgn_launch}/config.yaml"
test -f "${mgn_launch}/execution.json"
export MGN_LODO_ADMISSION_RECEIPT="${PWD}/${mgn_launch}/execution.json"
/home/addirakoze/environments/Reid/bin/python scripts/train.py --config "${mgn_launch}/config.yaml" 2>&1 | tee -a "${mgn_launch}/train.log"
