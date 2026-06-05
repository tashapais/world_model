"""Phase 4: verify the motion-label dynamics actually steers correctly.

For each direction we condition the dynamics on that direction's cardinal vector
and measure the ACTUAL scroll of the generated frame (FFT phase correlation),
then check the sign matches the command. Reports per-direction correct-% over many
seeds (PASS if >80%) and saves a steering grid + per-direction GIFs from a seed clip.

Inference conditioning mirrors training: the observed motion of the visible
context transitions, then the chosen direction vector for the predicted frame.

Run:
    python -m scripts.verify_motion video_tokenizer_path=... dynamics_path=... \
        seed_index=774 steps=6 n=300
"""
import os, sys
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

from datasets.data_utils import load_data_and_data_loaders
from utils.config import InferenceConfig, load_config
from utils.utils import load_videotokenizer_from_checkpoint, load_dynamics_from_checkpoint
from utils.motion_labels import motion_to_cardinal, estimate_shift, frames_to_gray, DIRECTION_VECTORS

DIRS = ["up", "down", "left", "right"]


def _pop(name, default):
    val = default
    keep = []
    for tok in sys.argv:
        if tok.startswith(f"{name}="):
            val = tok.split("=", 1)[1]
        else:
            keep.append(tok)
    sys.argv = keep
    return val


def vec(direction, device, dtype):
    return torch.tensor(DIRECTION_VECTORS[direction], device=device, dtype=dtype).view(1, 1, 2)


@torch.no_grad()
def gen_step(vt, dyn, context, direction, args, device):
    """One dynamics step conditioned on `direction`; returns the new frame [1,1,C,H,W]."""
    vi = vt.tokenize(context)
    vl = vt.quantizer.get_latents_from_indices(vi)
    cw = context.shape[1]
    ctx_motion = motion_to_cardinal(context).to(vl.dtype)          # [1, cw-1, 2] real observed motion
    cond = torch.cat([ctx_motion, vec(direction, device, vl.dtype)], dim=1)  # [1, cw, 2]
    def i2l(idx): return vt.quantizer.get_latents_from_indices(idx, dim=-1)
    nl = dyn.forward_inference(context_latents=vl, prediction_horizon=1, num_steps=8,
                               index_to_latents_fn=i2l, conditioning=cond,
                               temperature=args.temperature, schedule=getattr(args, "maskgit_schedule", "exp"))
    return vt.detokenize(nl)[:, -1:]


def main():
    steps = int(_pop("steps", 6))
    n = int(_pop("n", 300))
    args: InferenceConfig = load_config(InferenceConfig, default_config_path="configs/inference_zelda.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.prediction_horizon = 1
    vt, _ = load_videotokenizer_from_checkpoint(args.video_tokenizer_path, device); vt.eval()
    dyn, _ = load_dynamics_from_checkpoint(args.dynamics_path, device); dyn.eval()
    print(f"tokenizer: {args.video_tokenizer_path}\ndynamics:  {args.dynamics_path}")

    _, _, dl, _, _ = load_data_and_data_loaders(dataset=args.dataset, batch_size=1,
                                                num_frames=args.context_window,
                                                preload_ratio=getattr(args, "preload_ratio", None) or 0.1)
    Nclips = len(dl.dataset)

    # ---- quantitative: single-step correct-direction rate over n random seeds ----
    print(f"\n=== VERIFICATION over {n} seeds (single step per direction) ===")
    import random; random.seed(0)
    seed_idxs = random.sample(range(Nclips), min(n, Nclips))
    overall_ok = True
    for d in DIRS:
        edx, edy = DIRECTION_VECTORS[d]
        correct = 0; ddx_sum = 0.0; ddy_sum = 0.0
        for idx in seed_idxs:
            ctx = dl.dataset[idx][0].unsqueeze(0).to(device)[:, :args.context_window]
            nxt = gen_step(vt, dyn, ctx, d, args, device)
            g0 = frames_to_gray(ctx[:, -1]); g1 = frames_to_gray(nxt[:, -1])
            sx, sy = estimate_shift(g0, g1)
            sx, sy = sx.item(), sy.item()
            ddx_sum += sx; ddy_sum += sy
            primary = sx if edx != 0 else sy
            exp = edx if edx != 0 else edy
            if np.sign(primary) == np.sign(exp):
                correct += 1
        acc = correct / len(seed_idxs)
        ok = acc > 0.8; overall_ok &= ok
        print(f"  {d:>5} (cmd dx={edx:+.0f} dy={edy:+.0f}): mean generated shift "
              f"dx={ddx_sum/len(seed_idxs):+.2f} dy={ddy_sum/len(seed_idxs):+.2f} px | "
              f"correct {acc*100:5.1f}%  {'PASS' if ok else 'FAIL'}")
    print(f"\nOVERALL: {'PASS - directions are correct' if overall_ok else 'FAIL'}")

    # ---- visual: hold each direction from the seed clip ----
    idx = args.seed_index if args.seed_index is not None else seed_idxs[0]
    seed = dl.dataset[idx][0].unsqueeze(0).to(device)[:, :args.context_window]
    os.makedirs("inference_results/zelda_motion", exist_ok=True)
    ncol = steps + 1
    fig, axes = plt.subplots(4, ncol, figsize=(2.2 * ncol, 9))
    def img(f): return ((f.detach().float().cpu() + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy()
    for r, d in enumerate(DIRS):
        gen = seed.clone()
        for _ in range(steps):
            gen = torch.cat([gen, gen_step(vt, dyn, gen[:, -args.context_window:], d, args, device)], dim=1)
        axes[r, 0].imshow(img(seed[0, -1])); axes[r, 0].set_ylabel(d, fontsize=13, rotation=0, labelpad=30, va="center")
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        if r == 0: axes[r, 0].set_title("seed")
        for c in range(steps):
            axes[r, c + 1].imshow(img(gen[0, args.context_window + c])); axes[r, c + 1].axis("off")
            if r == 0: axes[r, c + 1].set_title(f"+{c+1}")
        frames = [seed[0, -1]] + [gen[0, args.context_window + c] for c in range(steps)]
        pil = [Image.fromarray((img(f) * 255).astype(np.uint8)) for f in frames]
        pil[0].save(f"inference_results/zelda_motion/steer_{d}_clip{idx}.gif",
                    save_all=True, append_images=pil[1:], duration=250, loop=0)
    plt.suptitle(f"Motion-labeled dynamics: hold each direction from clip {idx}", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(f"inference_results/zelda_motion/steer_grid_clip{idx}.png", dpi=95, bbox_inches="tight")
    plt.close()
    print(f"saved inference_results/zelda_motion/steer_grid_clip{idx}.png + gifs")


if __name__ == "__main__":
    main()
