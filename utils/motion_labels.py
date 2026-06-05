"""Self-supervised directional action labels from inter-frame motion.

The Zelda data has no action labels, but Phase 1 showed consecutive frames scroll
in clean cardinal directions. This module measures that scroll (FFT phase
correlation, batched on GPU) and quantizes each transition into a cardinal unit
vector used as the dynamics conditioning:

    right=(1,0)  left=(-1,0)  down=(0,1)  up=(0,-1)  still=(0,0)   (+x=right, +y=down)

Training: conditioning[t] = motion(frame_t -> frame_{t+1}), the real scroll that
produced the next frame, so the model is forced to associate each direction vector
with the matching visual motion. Inference: feed the desired direction's vector.
This is the same labeled-action setup proven to give 100% control on the synthetic
world, applied to Zelda via self-supervision.
"""
import torch

# direction name -> conditioning unit vector (dx, dy); +x right, +y down
DIRECTION_VECTORS = {
    "up": (0.0, -1.0),
    "down": (0.0, 1.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "still": (0.0, 0.0),
}
CONDITIONING_DIM = 2


def frames_to_gray(frames):
    """[...,3,H,W] in [-1,1] -> [...,H,W] luminance in [0,1]."""
    f = (frames + 1) / 2
    r, g, b = f[..., 0, :, :], f[..., 1, :, :], f[..., 2, :, :]
    return 0.299 * r + 0.587 * g + 0.114 * b


def estimate_shift(a, b):
    """Batched FFT phase correlation. a,b: [N,H,W] -> (dx, dy) [N] signed px.

    Convention (verified in __main__): if the content of `b` is the content of
    `a` translated by (+dx right, +dy down), this returns that (dx, dy).
    """
    N, H, W = a.shape
    Fa = torch.fft.rfft2(a)
    Fb = torch.fft.rfft2(b)
    R = Fa * torch.conj(Fb)
    R = R / (R.abs() + 1e-8)
    r = torch.fft.irfft2(R, s=(H, W))            # correlation surface [N,H,W]
    idx = r.reshape(N, -1).argmax(dim=1)
    py = (idx // W).float()
    px = (idx % W).float()
    px = torch.where(px > W / 2, px - W, px)      # wrap to signed
    py = torch.where(py > H / 2, py - H, py)
    # the correlation peak gives a->b registration with the opposite sign of the
    # content's translation, so negate to get "content moved by (+dx right,+dy down)"
    return -px, -py


@torch.no_grad()
def motion_to_cardinal(frames, still_thresh=0.75):
    """frames [B,T,C,H,W] -> conditioning [B,T-1,2] of cardinal unit vectors.

    For each consecutive pair, estimate the scroll and snap it to the dominant
    cardinal direction (or still if below threshold).
    """
    B, T, C, H, W = frames.shape
    gray = frames_to_gray(frames)                # [B,T,H,W]
    a = gray[:, :-1].reshape(B * (T - 1), H, W)
    b = gray[:, 1:].reshape(B * (T - 1), H, W)
    dx, dy = estimate_shift(a, b)                # [B*(T-1)]
    mag = torch.sqrt(dx * dx + dy * dy)
    horiz = dx.abs() >= dy.abs()
    out = torch.zeros(B * (T - 1), 2, device=frames.device, dtype=frames.dtype)
    moving = mag >= still_thresh
    # horizontal movers -> (+/-1, 0)
    hm = moving & horiz
    out[hm, 0] = torch.sign(dx[hm])
    # vertical movers -> (0, +/-1)
    vm = moving & (~horiz)
    out[vm, 1] = torch.sign(dy[vm])
    return out.reshape(B, T - 1, 2)


if __name__ == "__main__":
    import sys, random
    # 1) sign/convention self-test on a known shift
    torch.manual_seed(0)
    H = W = 64
    base = torch.rand(1, H, W)
    # shift content right by 5 and down by 3
    shifted = torch.roll(base, shifts=(3, 5), dims=(1, 2))  # (dy=+3 down, dx=+5 right)
    dx, dy = estimate_shift(base, shifted)
    print(f"convention self-test (expect dx=+5, dy=+3): dx={dx.item():+.1f} dy={dy.item():+.1f}")

    # 2) distribution on real data — should match the cv2 Phase-1 breakdown
    n_clips = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    from datasets.data_utils import load_data_and_data_loaders
    _, _, dl, _, _ = load_data_and_data_loaders(dataset="ZELDA", batch_size=1, num_frames=8, preload_ratio=0.1)
    N = len(dl.dataset)
    random.seed(0)
    counts = {"up": 0, "down": 0, "left": 0, "right": 0, "still": 0}
    vec2name = {v: k for k, v in DIRECTION_VECTORS.items()}
    for idx in random.sample(range(N), min(n_clips, N)):
        clip = dl.dataset[idx][0].unsqueeze(0)
        cond = motion_to_cardinal(clip)[0]  # [T-1,2]
        for v in cond:
            counts[vec2name[(float(v[0]), float(v[1]))]] += 1
    tot = sum(counts.values())
    print(f"\nlabel distribution over {tot} transitions:")
    for k in ["up", "down", "left", "right", "still"]:
        print(f"  {k:>5}: {counts[k]:5d} ({100*counts[k]/tot:5.1f}%)")
