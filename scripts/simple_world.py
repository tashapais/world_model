"""Minimal, verifiable action-conditioned world model.

The point: prove end-to-end that "press a key -> sprite moves the correct way",
in the simplest possible setting. Unlike the Zelda model, actions here are
LABELED (0=up 1=down 2=left 3=right), so correct directional control is learnable
and, crucially, *verifiable*.

World: a red sprite on a green background (64x64). An action moves the sprite a
fixed number of pixels. Model: a tiny CNN that maps (frame, action) -> next frame
(action injected via FiLM at the bottleneck). Training data is generated on the
fly, so there's effectively infinite data and nothing to overfit.

verify() tracks the sprite's centroid (via redness) in the model's predicted next
frame and checks the displacement sign matches the commanded direction, over many
random starts. It prints per-direction accuracy and an overall PASS/FAIL, and
saves a visual grid. Also saves results/simple_world/model.pt for play_simple.py.

Run:
    python -m scripts.simple_world                # train + verify
    python -m scripts.simple_world steps=4000
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

H = W = 64
SPRITE = 10          # sprite side length (px)
STEP = 6             # pixels moved per action
N_ACTIONS = 4
BG = torch.tensor([0.13, 0.55, 0.13])   # green
FG = torch.tensor([0.85, 0.10, 0.10])   # red
# action -> (dx, dy) in image coords (+x right, +y down)
ACTION_DELTA = {0: (0, -STEP), 1: (0, STEP), 2: (-STEP, 0), 3: (STEP, 0)}
ACTION_NAME = {0: "up", 1: "down", 2: "left", 3: "right"}
CKPT = "results/simple_world/model.pt"


def _pop(name, default):
    val = default
    for tok in sys.argv:
        if tok.startswith(f"{name}="):
            val = tok.split("=", 1)[1]
    return val


def render(xs, ys, device):
    """Render a batch of frames given sprite top-left positions. -> [B,3,H,W]."""
    B = xs.shape[0]
    img = BG.to(device).view(1, 3, 1, 1).expand(B, 3, H, W).clone()
    yy = torch.arange(H, device=device).view(1, 1, H, 1)
    xx = torch.arange(W, device=device).view(1, 1, 1, W)
    y0 = ys.view(B, 1, 1, 1); x0 = xs.view(B, 1, 1, 1)
    mask = (yy >= y0) & (yy < y0 + SPRITE) & (xx >= x0) & (xx < x0 + SPRITE)  # [B,1,H,W]
    img = torch.where(mask, FG.to(device).view(1, 3, 1, 1), img)
    return img


def sample_batch(B, device):
    """Random start pos (kept off the edges so any single action stays in-bounds),
    random action, returns (frame, action, next_frame)."""
    lo, hi = STEP, H - SPRITE - STEP
    xs = torch.randint(lo, hi + 1, (B,), device=device)
    ys = torch.randint(lo, hi + 1, (B,), device=device)
    a = torch.randint(0, N_ACTIONS, (B,), device=device)
    dx = torch.zeros(B, device=device, dtype=torch.long)
    dy = torch.zeros(B, device=device, dtype=torch.long)
    for act, (ddx, ddy) in ACTION_DELTA.items():
        dx = torch.where(a == act, torch.full_like(dx, ddx), dx)
        dy = torch.where(a == act, torch.full_like(dy, ddy), dy)
    nxs = (xs + dx).clamp(0, W - SPRITE)
    nys = (ys + dy).clamp(0, H - SPRITE)
    return render(xs, ys, device), a, render(nxs, nys, device), (xs, ys)


class TinyActionDynamics(nn.Module):
    def __init__(self, n_actions=N_ACTIONS, c=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(),     # 64->32
            nn.Conv2d(32, c, 3, 2, 1), nn.GELU(),     # 32->16
        )
        self.act_emb = nn.Embedding(n_actions, c)
        self.film = nn.Linear(c, 2 * c)               # -> gamma, beta
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(c, 32, 4, 2, 1), nn.GELU(),   # 16->32
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid() # 32->64
        )

    def forward(self, frame, action):
        h = self.enc(frame)                           # [B,c,16,16]
        gamma, beta = self.film(self.act_emb(action)).chunk(2, dim=-1)
        h = h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return self.dec(h)


def sprite_centroid(frames):
    """Weighted centroid of 'redness' per frame. frames [B,3,H,W] -> (cx, cy) [B]."""
    red = (frames[:, 0] - 0.5 * (frames[:, 1] + frames[:, 2])).clamp(min=0)  # [B,H,W]
    red = red + 1e-6
    yy = torch.arange(H, device=frames.device).view(1, H, 1)
    xx = torch.arange(W, device=frames.device).view(1, 1, W)
    tot = red.sum(dim=(1, 2))
    cx = (red * xx).sum(dim=(1, 2)) / tot
    cy = (red * yy).sum(dim=(1, 2)) / tot
    return cx, cy


def train(model, device, steps=3000, bs=64, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for i in range(steps):
        frame, a, nxt, _ = sample_batch(bs, device)
        pred = model(frame, a)
        loss = F.mse_loss(pred, nxt)
        opt.zero_grad(); loss.backward(); opt.step()
        if i % 500 == 0 or i == steps - 1:
            print(f"  step {i:5d}  mse {loss.item():.5f}")


@torch.no_grad()
def verify(model, device, n=400):
    """For each action, measure predicted sprite displacement vs the start frame."""
    model.eval()
    print("\n=== VERIFICATION: does each key move the sprite the right way? ===")
    overall_ok = True
    for act in range(N_ACTIONS):
        lo, hi = STEP, H - SPRITE - STEP
        xs = torch.randint(lo, hi + 1, (n,), device=device)
        ys = torch.randint(lo, hi + 1, (n,), device=device)
        start = render(xs, ys, device)
        a = torch.full((n,), act, device=device, dtype=torch.long)
        pred = model(start, a)
        cx0, cy0 = sprite_centroid(start)
        cx1, cy1 = sprite_centroid(pred)
        ddx = (cx1 - cx0); ddy = (cy1 - cy0)
        exp_dx, exp_dy = ACTION_DELTA[act]
        # correct if the dominant predicted movement axis/sign matches the command
        if exp_dx != 0:
            correct = (torch.sign(ddx) == np.sign(exp_dx))
            primary = ddx
        else:
            correct = (torch.sign(ddy) == np.sign(exp_dy))
            primary = ddy
        acc = correct.float().mean().item()
        ok = acc > 0.9
        overall_ok &= ok
        print(f"  {ACTION_NAME[act]:>5} (cmd dx={exp_dx:+d} dy={exp_dy:+d}): "
              f"predicted mean dx={ddx.mean():+.2f} dy={ddy.mean():+.2f} px | "
              f"correct-direction {acc*100:5.1f}%  {'PASS' if ok else 'FAIL'}")
    print(f"\nOVERALL: {'PASS - every key moves the sprite the right way' if overall_ok else 'FAIL'}")
    return overall_ok


@torch.no_grad()
def save_visual(model, device, path="results/simple_world/verify_grid.png"):
    model.eval()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # one fixed start, show start + predicted next for each action
    xs = torch.tensor([W // 2 - SPRITE // 2], device=device)
    ys = torch.tensor([H // 2 - SPRITE // 2], device=device)
    start = render(xs, ys, device)
    fig, axes = plt.subplots(1, N_ACTIONS + 1, figsize=(3 * (N_ACTIONS + 1), 3.4))
    axes[0].imshow(start[0].permute(1, 2, 0).cpu().numpy()); axes[0].set_title("start"); axes[0].axis("off")
    for act in range(N_ACTIONS):
        pred = model(start, torch.full((1,), act, device=device, dtype=torch.long))
        axes[act + 1].imshow(pred[0].permute(1, 2, 0).cpu().clamp(0, 1).numpy())
        axes[act + 1].set_title(f"{ACTION_NAME[act]} (a={act})"); axes[act + 1].axis("off")
    plt.suptitle("Synthetic world: predicted next frame per key (sprite should shift correctly)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=100, bbox_inches="tight"); plt.close()
    print(f"saved {path}")


def main():
    steps = int(_pop("steps", 3000))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  training steps: {steps}")
    model = TinyActionDynamics().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TinyActionDynamics params: {n_params/1e3:.1f}K")
    train(model, device, steps=steps)
    ok = verify(model, device)
    save_visual(model, device)
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "meta": {"H": H, "W": W, "SPRITE": SPRITE, "STEP": STEP,
                         "N_ACTIONS": N_ACTIONS, "ACTION_NAME": ACTION_NAME}}, CKPT)
    print(f"saved {CKPT}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
