"""Calibrate the unsupervised latent action codes to real screen directions.

The latent action codebook is learned without labels, so code 0..N-1 have no
inherent meaning. This script *measures* what each code does: from many seed
clips it applies a single code repeatedly for a few dynamics steps and computes
the average optical-flow vector that code induces. It then assigns the four arrow
directions (up/down/left/right) to the codes whose motion best matches each
direction and writes the mapping to configs/action_calibration.json, which
play.py loads to bind the arrow keys.

It also prints the motion *magnitude* per code. If those magnitudes are tiny /
all similar, the dynamics model is mostly ignoring the action conditioning and no
key remapping will make the sprite steer — that's a model signal, not a bug here.

Run (on the box that has the checkpoints):
    python -m scripts.calibrate_actions --config configs/inference_zelda.yaml \
        use_latest_checkpoints=true num_seeds=64 steps_per_action=4
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import random
from itertools import permutations
from typing import Optional

import cv2
import numpy as np
import torch

from datasets.data_utils import load_data_and_data_loaders
from utils.config import InferenceConfig, load_config
from utils.utils import find_latest_checkpoint
from utils.inference_utils import load_models, build_action_latent_from_index

CALIBRATION_PATH = os.path.join("configs", "action_calibration.json")

# image-coordinate unit vectors (y grows downward)
DIRECTION_VECTORS = {
    "up":    np.array([0.0, -1.0]),
    "down":  np.array([0.0,  1.0]),
    "left":  np.array([-1.0, 0.0]),
    "right": np.array([1.0,  0.0]),
}


def resolve_checkpoints(args: InferenceConfig) -> None:
    base_dir = os.getcwd()

    def missing(path: Optional[str]) -> bool:
        return (path is None) or (not os.path.exists(path))

    if args.use_latest_checkpoints or missing(args.video_tokenizer_path):
        args.video_tokenizer_path = find_latest_checkpoint(base_dir, "video_tokenizer")
    if args.use_latest_checkpoints or missing(args.latent_actions_path):
        args.latent_actions_path = find_latest_checkpoint(base_dir, "latent_actions")
    if args.use_latest_checkpoints or missing(args.dynamics_path):
        args.dynamics_path = find_latest_checkpoint(base_dir, "dynamics")
    for label, path in [("video_tokenizer", args.video_tokenizer_path),
                        ("latent_actions", args.latent_actions_path),
                        ("dynamics", args.dynamics_path)]:
        if missing(path):
            raise FileNotFoundError(f"{label} checkpoint not found ({path}).")
        print(f"Using {label} checkpoint: {path}")


def to_gray(frame_chw: torch.Tensor) -> np.ndarray:
    """[-1,1] CHW tensor -> float32 grayscale HxW for optical flow."""
    img = ((frame_chw.detach().float().cpu() + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy()
    if img.shape[2] == 1:
        gray = img[:, :, 0]
    else:
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return gray


def global_shift(prev_gray: np.ndarray, next_gray: np.ndarray) -> np.ndarray:
    """Estimate global translation (dx, dy) via phase correlation.

    Phase correlation recovers a coherent whole-frame shift, which is the right
    signal for "did the scene scroll up/down/left/right". (Averaging dense optical
    flow over the frame cancels out and reads ~0, which is why we don't use it.)
    """
    (sx, sy), _resp = cv2.phaseCorrelate(prev_gray, next_gray)
    return np.array([sx, sy])  # +x = right, +y = down


@torch.no_grad()
def dynamics_step(models, context_frames, action_index, inferred_actions, args, device):
    video_tokenizer, latent_action_model, dynamics_model = models
    video_indices = video_tokenizer.tokenize(context_frames)
    video_latents = video_tokenizer.quantizer.get_latents_from_indices(video_indices)
    _, action_latent = build_action_latent_from_index(
        action_index, inferred_actions, context_frames, latent_action_model,
        args.context_window, args.prediction_horizon, device)

    def idx_to_latents(idx):
        return video_tokenizer.quantizer.get_latents_from_indices(idx, dim=-1)

    next_video_latents = dynamics_model.forward_inference(
        context_latents=video_latents,
        prediction_horizon=args.prediction_horizon,
        num_steps=8,
        index_to_latents_fn=idx_to_latents,
        conditioning=action_latent,
        temperature=args.temperature,
        schedule=getattr(args, "maskgit_schedule", "exp"))
    next_frames = video_tokenizer.detokenize(next_video_latents)
    return next_frames[:, -args.prediction_horizon:]


def _pop_arg(name, default):
    """Pull `name=value` out of sys.argv so it isn't fed to the structured config."""
    import sys
    val = default
    keep = []
    for tok in sys.argv:
        if tok.startswith(f"{name}="):
            val = tok.split("=", 1)[1]
        else:
            keep.append(tok)
    sys.argv = keep
    return val


def main():
    # these are calibrator-only knobs, not InferenceConfig fields, so strip them first
    num_seeds = int(_pop_arg("num_seeds", 64))
    steps_per_action = int(_pop_arg("steps_per_action", 4))

    args: InferenceConfig = load_config(
        InferenceConfig,
        default_config_path=os.path.join(os.getcwd(), "configs", "inference_zelda.yaml"))

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = str(device)
    args.prediction_horizon = 1  # turn-based, one frame per action (matches play.py)
    print(f"Device: {device}  seeds: {num_seeds}  steps/action: {steps_per_action}")

    resolve_checkpoints(args)
    models = load_models(args.video_tokenizer_path, args.latent_actions_path,
                         args.dynamics_path, device, use_actions=True)
    latent_action_model = models[1]
    n_actions = latent_action_model.quantizer.codebook_size
    print(f"Latent action codebook size: {n_actions}")

    overrides = {"preload_ratio": args.preload_ratio} if getattr(args, "preload_ratio", None) else {}
    _, _, data_loader, _, _ = load_data_and_data_loaders(
        dataset=args.dataset, batch_size=1, num_frames=args.context_window, **overrides)
    n_clips = len(data_loader.dataset)

    # accumulate net flow per code across seeds
    flow_sum = np.zeros((n_actions, 2), dtype=np.float64)
    mag_sum = np.zeros(n_actions, dtype=np.float64)
    count = 0

    for s in range(num_seeds):
        idx = random.randint(0, n_clips - 1)
        seed = data_loader.dataset[idx][0].unsqueeze(0).to(device)[:, :args.context_window]
        for code in range(n_actions):
            generated = seed.clone()
            inferred_actions = []
            net = np.zeros(2, dtype=np.float64)
            for _ in range(steps_per_action):
                ctx = generated[:, -args.context_window:]
                prev_gray = to_gray(ctx[0, -1])
                nxt = dynamics_step(models, ctx, code, inferred_actions, args, device)
                generated = torch.cat([generated, nxt], dim=1)
                net += global_shift(prev_gray, to_gray(nxt[0, -1]))
            flow_sum[code] += net
            mag_sum[code] += np.linalg.norm(net)
        count += 1
        if (s + 1) % 8 == 0:
            print(f"  processed {s + 1}/{num_seeds} seeds")

    mean_vec = flow_sum / count          # (n_actions, 2) average net (dx, dy)
    mean_mag = mag_sum / count           # average path length per code

    print("\nPer-code average motion (image coords, +x=right, +y=down):")
    for code in range(n_actions):
        dx, dy = mean_vec[code]
        print(f"  code {code}: dx={dx:+.4f} dy={dy:+.4f}  |motion|={mean_mag[code]:.4f}")

    # assign the 4 cardinal directions to codes maximizing total projection score.
    dirs = list(DIRECTION_VECTORS.keys())
    score = np.zeros((len(dirs), n_actions))
    for di, d in enumerate(dirs):
        for code in range(n_actions):
            score[di, code] = float(np.dot(mean_vec[code], DIRECTION_VECTORS[d]))

    best_perm, best_total = None, -np.inf
    for perm in permutations(range(n_actions), len(dirs)):
        total = sum(score[di, perm[di]] for di in range(len(dirs)))
        if total > best_total:
            best_total, best_perm = total, perm
    mapping = {dirs[di]: int(best_perm[di]) for di in range(len(dirs))}

    print("\nCalibrated arrow -> code mapping:")
    weak_dirs = []
    for d in dirs:
        sc = score[dirs.index(d), mapping[d]]
        flag = "  <-- model has no real motion this way" if sc < 0.1 else ""
        if sc < 0.1:
            weak_dirs.append(d)
        print(f"  {d:>5} -> code {mapping[d]}  (shift {sc:+.3f} px){flag}")

    if weak_dirs:
        print(f"\n[WARN] The codes don't span all four directions — {', '.join(weak_dirs)} "
              f"got assigned\n       to codes with little/no motion that way. The "
              f"unsupervised latent\n       actions only learned a narrow motion repertoire, so "
              f"those arrows will\n       feel dead. Fixing this needs retraining (more action "
              f"codes / longer\n       latent-action training), not just remapping.")

    out = {
        "n_actions": int(n_actions),
        "mapping": mapping,
        "mean_motion": {str(c): mean_vec[c].tolist() for c in range(n_actions)},
        "mean_magnitude": {str(c): float(mean_mag[c]) for c in range(n_actions)},
        "num_seeds": num_seeds,
        "steps_per_action": steps_per_action,
    }
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {CALIBRATION_PATH}")


if __name__ == "__main__":
    main()
