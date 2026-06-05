#!/bin/bash
# Retrain the action path for controllable play:
#   - latent actions: 8 codes, bigger encoder, 15k steps (was 4 codes / tiny / 10k)
#   - dynamics: Genie-style target-frame-only masking so it must use the action
# The video tokenizer is REUSED (recon is already great) — pinned explicitly to
# the good run so find_latest_checkpoint can't grab a stray aborted run.
set -e
cd /home/tasha/world_model
export PYTHONPATH=.
export WANDB_MODE=online   # user is logged in; never go offline

VT=results/2026_06_05_00_58_45/video_tokenizer/checkpoints/video_tokenizer_step_47500
TRAIN_CFG=configs/training_patch4_hq.yaml

# one shared run root for both stages
export NG_RUN_ROOT_DIR="results/$(date +%Y_%m_%d_%H_%M_%S)_actions"
mkdir -p "$NG_RUN_ROOT_DIR"
echo "RUN_ROOT=$NG_RUN_ROOT_DIR"
echo "Using video tokenizer: $VT"

echo "===== Stage 1/2: Latent Actions (8 codes, 15k steps) ====="
python scripts/train_latent_actions.py \
    --config configs/latent_actions.yaml \
    --training_config "$TRAIN_CFG"

LA=$(ls -d "$NG_RUN_ROOT_DIR"/latent_actions/checkpoints/latent_actions_step_* \
     | sort -t_ -k4 -n | tail -1)
echo "Latest latent-actions checkpoint: $LA"

echo "===== Stage 2/2: Dynamics (target-frame-only masking, 30k steps) ====="
python scripts/train_dynamics.py \
    --config configs/dynamics_30k.yaml \
    --training_config "$TRAIN_CFG" \
    "video_tokenizer_path=$VT" \
    "latent_actions_path=$LA"

DYN=$(ls -d "$NG_RUN_ROOT_DIR"/dynamics/checkpoints/dynamics_step_* \
      | sort -t_ -k4 -n | tail -1)
echo "===== DONE ====="
echo "video_tokenizer: $VT"
echo "latent_actions:  $LA"
echo "dynamics:        $DYN"
echo "RUN_ROOT=$NG_RUN_ROOT_DIR"
