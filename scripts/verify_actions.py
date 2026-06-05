"""Visual proof that the calibrated arrow actions steer differently.

From a single seed clip, generate one autoregressive rollout per direction
(holding that direction's calibrated action code the whole time), then lay them
out as rows (up/down/left/right) over time so you can see each arrow produce a
distinct rollout. Saves a grid PNG and one GIF per direction.

Run:
    python -m scripts.verify_actions --config configs/inference_zelda.yaml \
        video_tokenizer_path=... latent_actions_path=... dynamics_path=... \
        seed_index=774 steps=8
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

from datasets.data_utils import load_data_and_data_loaders
from utils.config import InferenceConfig, load_config
from utils.utils import find_latest_checkpoint
from utils.inference_utils import load_models, build_action_latent_from_index

CALIBRATION_PATH = os.path.join("configs", "action_calibration.json")
DEFAULT_MAP = {"up": 0, "down": 1, "left": 2, "right": 3}


def _pop_arg(name, default):
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


@torch.no_grad()
def rollout(models, seed, code, steps, args, device):
    vt, la, dyn = models
    generated = seed.clone()
    hist = []
    for _ in range(steps):
        ctx = generated[:, -args.context_window:]
        vi = vt.tokenize(ctx)
        vl = vt.quantizer.get_latents_from_indices(vi)
        _, al = build_action_latent_from_index(code, hist, ctx, la, args.context_window, 1, device)
        def i2l(idx): return vt.quantizer.get_latents_from_indices(idx, dim=-1)
        nl = dyn.forward_inference(context_latents=vl, prediction_horizon=1, num_steps=8,
                                   index_to_latents_fn=i2l, conditioning=al,
                                   temperature=args.temperature, schedule=getattr(args, "maskgit_schedule", "exp"))
        generated = torch.cat([generated, vt.detokenize(nl)[:, -1:]], dim=1)
    return generated[:, -steps:]  # [1, steps, C, H, W]


def to_img(f):
    return ((f.detach().float().cpu() + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    steps = int(_pop_arg("steps", 8))
    args: InferenceConfig = load_config(InferenceConfig, default_config_path="configs/inference_zelda.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = str(device); args.prediction_horizon = 1

    for label in ("video_tokenizer", "latent_actions", "dynamics"):
        p = getattr(args, f"{label}_path")
        if p is None or not os.path.exists(p):
            setattr(args, f"{label}_path", find_latest_checkpoint(os.getcwd(), label))
    models = load_models(args.video_tokenizer_path, args.latent_actions_path, args.dynamics_path, device, use_actions=True)

    mapping = DEFAULT_MAP
    if os.path.exists(CALIBRATION_PATH):
        mapping = {**DEFAULT_MAP, **json.load(open(CALIBRATION_PATH)).get("mapping", {})}
    print("direction -> code:", mapping)

    _, _, dl, _, _ = load_data_and_data_loaders(dataset=args.dataset, batch_size=1,
                                                num_frames=args.context_window,
                                                preload_ratio=getattr(args, "preload_ratio", None) or 0.1)
    idx = args.seed_index if args.seed_index is not None else 0
    seed = dl.dataset[idx][0].unsqueeze(0).to(device)[:, :args.context_window]
    print(f"seed clip {idx}")

    dirs = ["up", "down", "left", "right"]
    rolls = {d: rollout(models, seed, mapping[d], steps, args, device) for d in dirs}

    # grid: rows = directions, col 0 = seed last frame, then generated frames
    os.makedirs("inference_results/zelda_actions", exist_ok=True)
    ncol = steps + 1
    fig, axes = plt.subplots(4, ncol, figsize=(2.2 * ncol, 2.2 * 4))
    for r, d in enumerate(dirs):
        axes[r, 0].imshow(to_img(seed[0, -1])); axes[r, 0].set_ylabel(f"{d}\n(code {mapping[d]})",
                                                                      fontsize=12, rotation=0, labelpad=40, va="center")
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        if r == 0:
            axes[r, 0].set_title("seed", fontsize=11)
        for c in range(steps):
            ax = axes[r, c + 1]
            ax.imshow(to_img(rolls[d][0, c])); ax.axis("off")
            if r == 0:
                ax.set_title(f"+{c+1}", fontsize=11)
    plt.suptitle(f"Holding each arrow from clip {idx} — each row should drift differently",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    grid_path = f"inference_results/zelda_actions/steer_grid_clip{idx}.png"
    plt.savefig(grid_path, dpi=95, bbox_inches="tight"); plt.close()
    print("saved", grid_path)

    # one GIF per direction (seed last frame + rollout)
    for d in dirs:
        frames = [seed[0, -1]] + [rolls[d][0, c] for c in range(steps)]
        pil = [Image.fromarray((to_img(f) * 255).astype(np.uint8)) for f in frames]
        gif = f"inference_results/zelda_actions/steer_{d}_clip{idx}.gif"
        pil[0].save(gif, save_all=True, append_images=pil[1:], duration=200, loop=0)
        print("saved", gif)


if __name__ == "__main__":
    main()
