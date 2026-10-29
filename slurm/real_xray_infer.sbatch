#!/bin/bash
#SBATCH -J ppcar_xray
#SBATCH -o /export/home2/reny0012/result/runlog/ppcar_xray_%j.log
#SBATCH -e /export/home2/reny0012/result/runlog/ppcar_xray_%j.err
#SBATCH -p RTXA6Kq
#SBATCH -w node11
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 8

# GPU job: PPCAR-Net reconstruction of real angiograms exported by angioseg-nnunet step 08 (one job, all cases).
# Environment: the existing venv /export/home2/reny0012/vir_env/vesseltree (PyTorch, this package, VGGT).
# Submit (from the repo root):
#   test, 5 cases:  sbatch --export=ALL,ART=lca,METHOD=xravos_A_v2,LIMIT=5 slurm/real_xray_infer.sbatch
#   all cases:      sbatch --export=ALL,ART=lca,METHOD=xravos_A_v2 slurm/real_xray_infer.sbatch
#   options:        OUTPUT_MODE=full|summary (default full), CAMERA=acquisition|training (default acquisition),
#                   NUM_VIEWS=2, CASES="Study00605_RCA Study00087_RCA"
#   full: GIFs + per-view overlays + summary.png per case;  summary: only summary.png per case (fast)
# Input:   $RECON_INPUTS/<METHOD>/<ART>/*.npz          (step 08 output)
# Output:  $RECON_OUTPUTS/<METHOD>_<camera>[_first<LIMIT>]/<ART>/<case>/..., summary.csv

set -euo pipefail

module load python3
source /export/home2/reny0012/vir_env/vesseltree/bin/activate
module load cuda12.5/toolkit/12.5.1

REPO_ROOT="${REPO_ROOT:-/export/home2/reny0012/code/PPCAR-Net}"
cd "$REPO_ROOT"
# VGGT-1B (facebook/VGGT-1B) is loaded from the venv user's usual HuggingFace cache. If the compute node has no
# internet and the weights are already cached, submit with HF_HUB_OFFLINE=1 to skip the online check.

ART="${ART:?set ART=lca or ART=rca}"
METHOD="${METHOD:?set METHOD=<step-08 output folder>, e.g. xravos_A_v2}"
CAMERA="${CAMERA:-acquisition}"
RECON_INPUTS="${RECON_INPUTS:-/dataset/reny0012/angioseg/recon_inputs}"
RECON_OUTPUTS="${RECON_OUTPUTS:-/dataset/reny0012/angioseg/recon_outputs}"
OUTPUT_MODE="${OUTPUT_MODE:-full}"
TAG="${METHOD}_${CAMERA}"
[ "$OUTPUT_MODE" = "summary" ] && TAG="${TAG}_summary"
EXTRA=(--camera "$CAMERA" --num_views "${NUM_VIEWS:-2}" --output_mode "$OUTPUT_MODE")
[ -n "${LIMIT:-}" ] && { EXTRA+=(--limit "$LIMIT"); TAG="${TAG}_first${LIMIT}"; }
[ -n "${CASES:-}" ] && { EXTRA+=(--case_ids $CASES); TAG="${TAG}_selected"; }

echo "[ppcar] host=$(hostname) job=${SLURM_JOB_ID:-none} python=$(command -v python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "[ppcar] $RECON_INPUTS/$METHOD/$ART -> $RECON_OUTPUTS/$TAG/$ART  (camera $CAMERA)"

python -m vessel_code.real_xray.infer --config "configs/real_xray/$ART.json" \
  --input_dir "$RECON_INPUTS/$METHOD/$ART" --output_dir "$RECON_OUTPUTS/$TAG/$ART" "${EXTRA[@]}"
