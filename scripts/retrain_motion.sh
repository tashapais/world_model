#!/bin/bash
# Phase 3: retrain ONLY the dynamics model with self-supervised cardinal motion
# labels (no latent-action model). Video tokenizer is reused (pinned explicitly).
set -e
cd /home/tasha/world_model
export PYTHONPATH=.
export WANDB_MODE=online

VT=results/2026_06_05_00_58_45/video_tokenizer/checkpoints/video_tokenizer_step_47500
export NG_RUN_ROOT_DIR="results/$(date +%Y_%m_%d_%H_%M_%S)_motion"
mkdir -p "$NG_RUN_ROOT_DIR"
echo "RUN_ROOT=$NG_RUN_ROOT_DIR"
echo "Using video tokenizer: $VT"

python scripts/train_dynamics.py \
    --config configs/dynamics_motion.yaml \
    --training_config configs/training_patch4_hq.yaml \
    "video_tokenizer_path=$VT" "use_motion_labels=true"

DYN_STEP=$(ls -d "$NG_RUN_ROOT_DIR"/dynamics/checkpoints/dynamics_step_* | sed 's/.*_step_//' | sort -n | tail -1)
echo "===== DONE ====="
echo "dynamics: $NG_RUN_ROOT_DIR/dynamics/checkpoints/dynamics_step_${DYN_STEP}"
echo "RUN_ROOT=$NG_RUN_ROOT_DIR"
