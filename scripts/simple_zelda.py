"""Synthesis: the proven simple pixel model + real Zelda frames + motion labels.

The token/MaskGIT dynamics failed to give clean directional control on Zelda
(extrapolates / collapses). The synthetic world proved that a deterministic
pixel CNN with single-frame context and LABELED actions gives 100% correct
control. This applies that exact recipe to real Zelda frames, using the
self-supervised cardinal motion labels (utils.motion_labels) as the action.

- context = 1 frame (so the action is the only directional signal; no extrapolation)
- 4 actions (up/down/left/right); trained only on MOVING transitions
- encoder/decoder CNN, action via FiLM at the bottleneck, MSE loss (blurry but moves)

verify(): feed each direction, measure the predicted frame's actual scroll, check
the sign matches. Reports per-direction correct-% + PASS/FAIL, saves a grid + gifs
+ a checkpoint for play_zelda_simple.py.

Run:
    python -m scripts.simple_zelda steps=4000 res=64
"""
import os, sys, random
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from datasets.data_utils import load_data_and_data_loaders
from utils.motion_labels import motion_to_cardinal, estimate_shift, frames_to_gray, DIRECTION_VECTORS

DIRS = ["up", "down", "left", "right"]
VEC2IDX = {(0.0, -1.0): 0, (0.0, 1.0): 1, (-1.0, 0.0): 2, (1.0, 0.0): 3}
CKPT = "results/simple_zelda/model.pt"


def _pop(name, default):
    for tok in sys.argv:
        if tok.startswith(f"{name}="):
            return tok.split("=", 1)[1]
    return default


class ZeldaActionCNN(nn.Module):
    """1 frame + action -> next frame. Encoder/decoder, FiLM action at bottleneck."""
    def __init__(self, n_actions=4, c=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(),     # H->H/2
            nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),    # ->H/4
            nn.Conv2d(64, c, 4, 2, 1), nn.GELU(),     # ->H/8
        )
        self.act_emb = nn.Embedding(n_actions, c)
        self.film = nn.Linear(c, 2 * c)
        self.mid = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1), nn.GELU(),
                                 nn.Conv2d(c, c, 3, 1, 1), nn.GELU())
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(c, 64, 4, 2, 1), nn.GELU(),   # ->H/4
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),  # ->H/2
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid() # ->H
        )

    def forward(self, frame01, action):
        h = self.enc(frame01)
        g, b = self.film(self.act_emb(action)).chunk(2, dim=-1)
        h = h * (1 + g[:, :, None, None]) + b[:, :, None, None]
        h = self.mid(h)
        return self.dec(h)


def moving_pairs(clip, device, still_thresh=0.75):
    """clip [B,T,C,H,W] in [-1,1] -> (f0_01, f1_01, action_idx) for MOVING transitions."""
    B, T, C, Hh, Ww = clip.shape
    cond = motion_to_cardinal(clip, still_thresh)  # [B,T-1,2]
    f0 = clip[:, :-1].reshape(-1, C, Hh, Ww)
    f1 = clip[:, 1:].reshape(-1, C, Hh, Ww)
    v = cond.reshape(-1, 2)
    moving = (v.abs().sum(-1) > 0)
    f0, f1, v = f0[moving], f1[moving], v[moving]
    idx = torch.tensor([VEC2IDX[(float(a), float(b))] for a, b in v], device=device, dtype=torch.long)
    to01 = lambda x: ((x + 1) / 2).clamp(0, 1)
    return to01(f0), to01(f1), idx


def loader(res, bs):
    _, _, dl, _, _ = load_data_and_data_loaders(dataset="ZELDA", batch_size=bs, num_frames=8,
                                                preload_ratio=1.0, resolution=(res, res))
    return dl


def train(model, dl, device, steps, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train(); it = iter(dl)
    for i in range(steps):
        try:
            clip, _ = next(it)
        except StopIteration:
            it = iter(dl); clip, _ = next(it)
        f0, f1, a = moving_pairs(clip.to(device), device)
        if f0.shape[0] < 4:
            continue
        pred = model(f0, a)
        loss = F.mse_loss(pred, f1)
        opt.zero_grad(); loss.backward(); opt.step()
        if i % 500 == 0 or i == steps - 1:
            print(f"  step {i:5d}  mse {loss.item():.5f}  (batch movers {f0.shape[0]})")


@torch.no_grad()
def verify(model, dl, device, n=400):
    model.eval()
    print("\n=== VERIFICATION: does each key scroll Zelda the right way? ===")
    # gather a pool of frames
    frames = []
    it = iter(dl)
    while len(frames) < n:
        clip, _ = next(it)
        for b in range(clip.shape[0]):
            frames.append(clip[b, 0])
    frames = torch.stack(frames[:n]).to(device)
    f01 = ((frames + 1) / 2).clamp(0, 1)
    overall = True
    for d in DIRS:
        edx, edy = DIRECTION_VECTORS[d]
        a = torch.full((n,), DIRS.index(d), device=device, dtype=torch.long)
        pred = model(f01, a)
        sx, sy = estimate_shift(frames_to_gray(f01 * 2 - 1), frames_to_gray(pred * 2 - 1))
        primary = sx if edx != 0 else sy
        exp = edx if edx != 0 else edy
        acc = (torch.sign(primary) == np.sign(exp)).float().mean().item()
        ok = acc > 0.8; overall &= ok
        print(f"  {d:>5} (cmd dx={edx:+.0f} dy={edy:+.0f}): mean shift dx={sx.mean():+.2f} dy={sy.mean():+.2f} px"
              f" | correct {acc*100:5.1f}%  {'PASS' if ok else 'FAIL'}")
    print(f"\nOVERALL: {'PASS - Zelda scrolls the right way per key' if overall else 'FAIL'}")
    return overall


@torch.no_grad()
def save_visual(model, dl, device, steps=6, res=64):
    model.eval()
    os.makedirs("results/simple_zelda", exist_ok=True)
    clip, _ = next(iter(dl))
    seed = ((clip[0, 0:1].to(device) + 1) / 2).clamp(0, 1)  # [1,3,H,W]
    img = lambda x: x[0].detach().cpu().permute(1, 2, 0).numpy()
    fig, axes = plt.subplots(4, steps + 1, figsize=(2.2 * (steps + 1), 9))
    for r, d in enumerate(DIRS):
        cur = seed.clone()
        axes[r, 0].imshow(img(cur)); axes[r, 0].set_ylabel(d, rotation=0, labelpad=28, va="center", fontsize=13)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        if r == 0: axes[r, 0].set_title("seed")
        gif = [img(cur)]
        for c in range(steps):
            cur = model(cur, torch.full((1,), DIRS.index(d), device=device, dtype=torch.long)).clamp(0, 1)
            axes[r, c + 1].imshow(img(cur)); axes[r, c + 1].axis("off")
            if r == 0: axes[r, c + 1].set_title(f"+{c+1}")
            gif.append(img(cur))
        pil = [Image.fromarray((g * 255).astype(np.uint8)) for g in gif]
        pil[0].save(f"results/simple_zelda/steer_{d}.gif", save_all=True, append_images=pil[1:], duration=250, loop=0)
    plt.suptitle("Zelda pixel model: hold each key (deterministic)", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig("results/simple_zelda/steer_grid.png", dpi=95, bbox_inches="tight"); plt.close()
    print("saved results/simple_zelda/steer_grid.png + gifs")


def main():
    steps = int(_pop("steps", 4000)); res = int(_pop("res", 64)); bs = int(_pop("bs", 32))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  steps {steps}  res {res}")
    dl = loader(res, bs)
    model = ZeldaActionCNN().to(device)
    print(f"ZeldaActionCNN params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    train(model, dl, device, steps)
    ok = verify(model, dl, device)
    save_visual(model, dl, device, res=res)
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": {"res": res, "DIRS": DIRS}}, CKPT)
    print(f"saved {CKPT}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
