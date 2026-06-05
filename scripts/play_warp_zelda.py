"""Play the warp-based Zelda controller with the arrow keys.

Loads the warp model + bundled seed frames (results/warp_zelda/) and lets you
scroll the world with the arrows. Direction is verified-correct (96-99.8%):
    up=0 down=1 left=2 right=3
Each keypress warps the current frame; the rollout is autoregressive. Long holds
in one direction streak at the revealed edge (warp artifact) - press 'r' to reseed.

Run (locally, needs a display):
    python -m scripts.play_warp_zelda
    r = next seed frame,  q/ESC = quit
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import cv2
import numpy as np
import torch
from scripts.warp_zelda import WarpZelda, CKPT, DIRS

SEEDS = "results/warp_zelda/seeds.pt"
DISPLAY = 512
HUD_H = 48
WINDOW = "warp zelda (arrows move | r reseed | q quit)"

ARROW_TO_ACTION = {
    65362: 0, 63232: 0, 2490368: 0,   # up
    65364: 1, 63233: 1, 2621440: 1,   # down
    65361: 2, 63234: 2, 2424832: 2,   # left
    65363: 3, 63235: 3, 2555904: 3,   # right
}
KEY_TO_ACTION = {ord("w"): 0, ord("s"): 1, ord("a"): 2, ord("d"): 3}


def to_bgr(frame01):
    img = frame01[0].detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def render(frame01, last):
    big = cv2.resize(to_bgr(frame01), (DISPLAY, DISPLAY), interpolation=cv2.INTER_NEAREST)
    hud = np.full((HUD_H, DISPLAY, 3), 24, dtype=np.uint8)
    cv2.putText(hud, f"last: {last}", (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 80), 1, cv2.LINE_AA)
    cv2.putText(hud, "arrows = move   r = reseed   q = quit", (8, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([hud, big])


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WarpZelda().to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)["state_dict"]); model.eval()
    seeds = torch.load(SEEDS, map_location=device, weights_only=False)  # [N,3,H,W] in [0,1]
    print(f"Loaded {CKPT} on {device}; {seeds.shape[0]} seed frames")

    si = 0
    frame = seeds[si:si + 1].to(device)
    last = "-"
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    print("Window open. Click it, then use the arrow keys.")
    while True:
        cv2.imshow(WINDOW, render(frame, last))
        key = cv2.waitKeyEx(0); ak = key & 0xFF
        if key in ARROW_TO_ACTION:
            act = ARROW_TO_ACTION[key]
        elif ak in (ord("q"), 27):
            break
        elif ak == ord("r"):
            si = (si + 1) % seeds.shape[0]; frame = seeds[si:si + 1].to(device); last = "-"; continue
        elif ak in KEY_TO_ACTION:
            act = KEY_TO_ACTION[ak]
        else:
            continue
        frame = model(frame, torch.full((1,), act, device=device, dtype=torch.long))
        last = DIRS[act]
    cv2.destroyAllWindows()
    print("Bye.")


if __name__ == "__main__":
    main()
