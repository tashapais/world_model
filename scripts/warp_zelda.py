"""Warp-based Zelda controller: sharp, directionally-correct motion.

The MSE pixel model (scripts/simple_zelda.py) got the direction right on average
but blurred (it regenerates every pixel). A scrolling game's motion is really a
translation, so here the model predicts a flow field conditioned on the action and
WARPS the current frame (grid_sample) instead of repainting it. Warping preserves
content sharpness and makes the scroll explicit, which is exactly the right
inductive bias for cardinal movement.

Trained on MOVING transitions with the self-supervised cardinal labels; verified
by measuring the warped frame's actual scroll direction.

Run:
    python -m scripts.warp_zelda steps=4000 res=64
"""
import os, sys
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from utils.motion_labels import estimate_shift, frames_to_gray, DIRECTION_VECTORS
from scripts.simple_zelda import DIRS, moving_pairs, loader

CKPT = "results/warp_zelda/model.pt"


def _pop(name, default):
    for tok in sys.argv:
        if tok.startswith(f"{name}="):
            return tok.split("=", 1)[1]
    return default


class WarpZelda(nn.Module):
    """frame + action -> dense flow -> grid_sample warp (+ small residual fill)."""
    def __init__(self, n_actions=4, c=96, flow_scale=0.3):
        super().__init__()
        self.flow_scale = flow_scale
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(),     # H/2
            nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),    # H/4
            nn.Conv2d(64, c, 4, 2, 1), nn.GELU(),     # H/8
        )
        self.act_emb = nn.Embedding(n_actions, c)
        self.film = nn.Linear(c, 2 * c)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(c, 64, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.GELU(),
        )
        self.flow_head = nn.Conv2d(16, 2, 3, 1, 1)
        nn.init.zeros_(self.flow_head.weight); nn.init.zeros_(self.flow_head.bias)  # start as identity
        self.resid_head = nn.Conv2d(16, 3, 3, 1, 1)
        nn.init.zeros_(self.resid_head.weight); nn.init.zeros_(self.resid_head.bias)

    def forward(self, frame01, action):
        B, _, H, W = frame01.shape
        h = self.enc(frame01)
        g, b = self.film(self.act_emb(action)).chunk(2, dim=-1)
        h = h * (1 + g[:, :, None, None]) + b[:, :, None, None]
        d = self.dec(h)
        flow = torch.tanh(self.flow_head(d)) * self.flow_scale     # [B,2,H,W] in grid units
        theta = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=frame01.dtype, device=frame01.device)
        base = F.affine_grid(theta.unsqueeze(0).repeat(B, 1, 1), (B, 3, H, W), align_corners=False)
        grid = base + flow.permute(0, 2, 3, 1)
        warped = F.grid_sample(frame01, grid, align_corners=False, padding_mode="border")
        resid = torch.tanh(self.resid_head(d)) * 0.1
        return (warped + resid).clamp(0, 1)


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
            print(f"  step {i:5d}  mse {loss.item():.5f}  (movers {f0.shape[0]})")


@torch.no_grad()
def verify(model, dl, device, n=400):
    model.eval()
    print("\n=== VERIFICATION: warp model directional control ===")
    frames = []; it = iter(dl)
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
    print(f"\nOVERALL: {'PASS - warp scrolls the right way per key' if overall else 'FAIL'}")
    return overall


@torch.no_grad()
def save_visual(model, dl, device, steps=8):
    model.eval()
    os.makedirs("results/warp_zelda", exist_ok=True)
    clip, _ = next(iter(dl))
    seed = ((clip[0, 0:1].to(device) + 1) / 2).clamp(0, 1)
    img = lambda x: x[0].detach().cpu().permute(1, 2, 0).numpy()
    fig, axes = plt.subplots(4, steps + 1, figsize=(2.2 * (steps + 1), 9))
    for r, d in enumerate(DIRS):
        cur = seed.clone(); gif = [img(cur)]
        axes[r, 0].imshow(img(cur)); axes[r, 0].set_ylabel(d, rotation=0, labelpad=28, va="center", fontsize=13)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        if r == 0: axes[r, 0].set_title("seed")
        for c in range(steps):
            cur = model(cur, torch.full((1,), DIRS.index(d), device=device, dtype=torch.long))
            axes[r, c + 1].imshow(img(cur)); axes[r, c + 1].axis("off")
            if r == 0: axes[r, c + 1].set_title(f"+{c+1}")
            gif.append(img(cur))
        pil = [Image.fromarray((g * 255).astype(np.uint8)) for g in gif]
        pil[0].save(f"results/warp_zelda/steer_{d}.gif", save_all=True, append_images=pil[1:], duration=250, loop=0)
    plt.suptitle("Warp-based Zelda: hold each key (deterministic)", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig("results/warp_zelda/steer_grid.png", dpi=95, bbox_inches="tight"); plt.close()
    print("saved results/warp_zelda/steer_grid.png + gifs")


def main():
    steps = int(_pop("steps", 4000)); res = int(_pop("res", 64)); bs = int(_pop("bs", 32))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  steps {steps}  res {res}")
    dl = loader(res, bs)
    model = WarpZelda().to(device)
    print(f"WarpZelda params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    train(model, dl, device, steps)
    ok = verify(model, dl, device)
    save_visual(model, dl, device)
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": {"res": res, "DIRS": DIRS}}, CKPT)
    print(f"saved {CKPT}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
