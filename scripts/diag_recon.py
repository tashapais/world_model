"""Tokenizer recon-ceiling diagnostic: encode->quantize->decode REAL frames
(no dynamics) through the committed baseline checkpoint, save GT vs recon.

Streams a few frames from HF (no full 1.7GB download)."""
import numpy as np, torch, fsspec, h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models.video_tokenizer import VideoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---- stream a few clips of real frames (range reads, not full file) ----
f = fsspec.open("hf://datasets/AlmondGod/tinyworlds/zelda_frames.h5").open()
h5 = h5py.File(f, "r")
ds = h5["frames"]            # [N, H, W, 3] uint8
print("dataset shape:", ds.shape, ds.dtype)
# 4 clips of 4 frames each, strided to mimic motion (fps subsample ~stride 4)
starts = [1000, 8000, 20000, 40000]
stride = 4
clips = []
for s in starts:
    idx = [s + k * stride for k in range(4)]
    clip = ds[idx[0]:idx[-1] + 1:stride].astype(np.float32) / 255.0  # [4,H,W,3]
    clips.append(torch.from_numpy(clip).permute(0, 3, 1, 2))         # [4,3,H,W]
x = torch.stack(clips).to(DEV)   # [B=4, T=4, 3, H, W]
print("input clip tensor:", tuple(x.shape))

# ---- baseline architecture (from checkpoint inspection) ----
model = VideoTokenizer(frame_size=(128, 128), patch_size=8, embed_dim=32,
                       num_heads=8, hidden_dim=128, num_blocks=4,
                       latent_dim=5, num_bins=4).to(DEV)
sd = torch.load("checkpoints/video_tokenizer/model_state_dict.pt",
                map_location=DEV, weights_only=False)
missing, unexpected = model.load_state_dict(sd, strict=False)
print("missing:", missing, "unexpected:", unexpected)
model.eval()

with torch.no_grad():
    loss, x_hat, _ = model(x)
    idx = model.tokenize(x)
    usage = torch.unique(idx).numel() / model.codebook_size
print(f"recon smooth_l1 loss: {loss.item():.5f}")
print(f"codebook usage: {usage*100:.2f}%  ({torch.unique(idx).numel()}/{model.codebook_size} codes)")

# ---- save GT (top) vs recon (bottom), one column per (clip, last frame) ----
xc = x.detach().cpu().clamp(0, 1)
xr = x_hat.detach().cpu().clamp(0, 1)
B, T = xc.shape[:2]
cols = B * T
fig, ax = plt.subplots(2, cols, figsize=(cols * 1.6, 3.6))
for b in range(B):
    for t in range(T):
        c = b * T + t
        ax[0, c].imshow(xc[b, t].permute(1, 2, 0).numpy()); ax[0, c].axis("off")
        ax[1, c].imshow(xr[b, t].permute(1, 2, 0).numpy()); ax[1, c].axis("off")
        if t == 0:
            ax[0, c].set_title(f"clip{b}", fontsize=8)
ax[0, 0].set_ylabel("GT", fontsize=10)
ax[1, 0].set_ylabel("recon", fontsize=10)
plt.suptitle(f"Tokenizer recon ceiling (baseline patch8/1024-codes)  loss={loss.item():.4f}  usage={usage*100:.1f}%", fontsize=10)
plt.tight_layout()
out = "inference_results/diag_recon_ceiling.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print("saved:", out)
