"""Phase 1: characterize motion in the Zelda data.

Question this answers: is directional control even learnable from this dataset?
For many consecutive frame pairs we measure (a) how much changes (mean abs diff)
and (b) the dominant global translation via phase correlation. We then bucket
each transition into up/down/left/right/still and report the breakdown, plus the
shift-magnitude distribution. If most transitions are static/text or the motion
has no coherent direction, no model can learn control; if there's a healthy
spread of clean directional scroll, self-supervised motion labels will work.

CPU-only (safe to run alongside training).

Run:
    python -m scripts.analyze_zelda_motion num_clips=400
"""
import os, sys
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets.data_utils import load_data_and_data_loaders


def _pop(name, default):
    for tok in sys.argv:
        if tok.startswith(f"{name}="):
            return tok.split("=", 1)[1]
    return default


def gray(f):
    img = ((f.float() + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy()
    return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)


def main():
    num_clips = int(_pop("num_clips", 400))
    num_frames = int(_pop("num_frames", 8))
    still_px = float(_pop("still_px", 0.75))   # |shift| below this = "still"
    _, _, dl, _, _ = load_data_and_data_loaders(dataset="ZELDA", batch_size=1,
                                                num_frames=num_frames, preload_ratio=0.1)
    N = len(dl.dataset)
    num_clips = min(num_clips, N)
    win = cv2.createHanningWindow((dl.dataset[0][0].shape[-1], dl.dataset[0][0].shape[-2]), cv2.CV_32F)

    shifts, mags, diffs = [], [], []
    import random; random.seed(0)
    for idx in random.sample(range(N), num_clips):
        clip = dl.dataset[idx][0]  # [T,C,H,W]
        for t in range(clip.shape[0] - 1):
            g0, g1 = gray(clip[t]), gray(clip[t + 1])
            (sx, sy), resp = cv2.phaseCorrelate(g0, g1, win)
            shifts.append((sx, sy)); mags.append((sx**2 + sy**2) ** 0.5)
            diffs.append(float(np.abs(g1 - g0).mean()))
    shifts = np.array(shifts); mags = np.array(mags); diffs = np.array(diffs)

    # bucket each transition
    def bucket(sx, sy, m):
        if m < still_px:
            return "still"
        return ("left" if sx < 0 else "right") if abs(sx) >= abs(sy) else ("up" if sy < 0 else "down")
    buckets = [bucket(sx, sy, m) for (sx, sy), m in zip(shifts, mags)]
    order = ["up", "down", "left", "right", "still"]
    counts = {k: buckets.count(k) for k in order}
    tot = len(buckets)

    print(f"\n=== Phase 1: motion analysis over {tot} transitions ({num_clips} clips) ===")
    print(f"shift magnitude (px): mean {mags.mean():.2f}  median {np.median(mags):.2f}  "
          f"p90 {np.percentile(mags,90):.2f}  max {mags.max():.2f}")
    print(f"mean abs pixel diff (0-255): mean {diffs.mean():.2f}  median {np.median(diffs):.2f}")
    print(f"\ntransition direction breakdown (still = |shift| < {still_px}px):")
    for k in order:
        print(f"  {k:>5}: {counts[k]:5d}  ({100*counts[k]/tot:5.1f}%)")
    moving = tot - counts["still"]
    print(f"\nmoving transitions: {moving} ({100*moving/tot:.1f}%)")
    if moving:
        dir_counts = {k: counts[k] for k in ["up", "down", "left", "right"]}
        dmin, dmax = min(dir_counts.values()), max(dir_counts.values())
        print(f"directional balance among movers: min {dmin} max {dmax} "
              f"(ratio {dmax/max(dmin,1):.1f}x) -> {'balanced' if dmax/max(dmin,1) < 4 else 'IMBALANCED'}")

    # figure: shift scatter + magnitude hist + direction bars
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].scatter(shifts[:, 0], shifts[:, 1], s=4, alpha=0.3)
    ax[0].axhline(0, color="k", lw=0.5); ax[0].axvline(0, color="k", lw=0.5)
    ax[0].set_title("per-transition shift (px)\n+x=right +y=down"); ax[0].set_xlabel("dx"); ax[0].set_ylabel("dy")
    lim = np.percentile(mags, 98) + 1
    ax[0].set_xlim(-lim, lim); ax[0].set_ylim(-lim, lim)
    ax[1].hist(mags, bins=50); ax[1].axvline(still_px, color="r", ls="--", label=f"still<{still_px}")
    ax[1].set_title("shift magnitude distribution"); ax[1].set_xlabel("|shift| px"); ax[1].legend()
    ax[2].bar(order, [counts[k] for k in order],
              color=["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#999999"])
    ax[2].set_title("direction breakdown")
    plt.tight_layout()
    os.makedirs("inference_results/analysis", exist_ok=True)
    out = "inference_results/analysis/zelda_motion.png"
    plt.savefig(out, dpi=100, bbox_inches="tight"); plt.close()
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
