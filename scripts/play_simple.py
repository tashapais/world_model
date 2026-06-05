"""Play the minimal synthetic world model with the arrow keys.

Loads the tiny action-conditioned model trained by scripts/simple_world.py and
lets you drive it: each arrow press runs one model step and shows the predicted
next frame. Because the actions are LABELED, the binding is fixed and correct -
no calibration needed:
    up -> 0   down -> 1   left -> 2   right -> 3

The rollout is autoregressive (the model's own predicted frame is fed back in),
so this is genuinely the world model generating the motion, not a scripted sprite.

Run (locally, needs a display):
    python -m scripts.play_simple
    r = reset to center,  q/ESC = quit
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import cv2
import numpy as np
import torch

from scripts.simple_world import TinyActionDynamics, render, H, W, SPRITE, ACTION_NAME, CKPT

DISPLAY = 512
HUD_H = 48
WINDOW = "simple world (arrow keys to move | r reset | q quit)"

ARROW_TO_ACTION = {
    65362: 0, 63232: 0, 2490368: 0,   # up
    65364: 1, 63233: 1, 2621440: 1,   # down
    65361: 2, 63234: 2, 2424832: 2,   # left
    65363: 3, 63235: 3, 2555904: 3,   # right
}
KEY_TO_ACTION = {ord("w"): 0, ord("s"): 1, ord("a"): 2, ord("d"): 3}


def to_bgr(frame):
    img = frame[0].detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def render_hud(frame, last):
    bgr = to_bgr(frame)
    big = cv2.resize(bgr, (DISPLAY, DISPLAY), interpolation=cv2.INTER_NEAREST)
    hud = np.full((HUD_H, DISPLAY, 3), 24, dtype=np.uint8)
    cv2.putText(hud, f"last action: {last}", (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 80), 1, cv2.LINE_AA)
    cv2.putText(hud, "arrows = move   r = reset   q = quit", (8, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([hud, big])


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(CKPT):
        raise FileNotFoundError(f"{CKPT} not found - run `python -m scripts.simple_world` first.")
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model = TinyActionDynamics().to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    print(f"Loaded {CKPT} on {device}")

    def fresh():
        xs = torch.tensor([W // 2 - SPRITE // 2], device=device)
        ys = torch.tensor([H // 2 - SPRITE // 2], device=device)
        return render(xs, ys, device)

    frame = fresh(); last = "-"
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    print("Window open. Click it, then use the arrow keys.")
    while True:
        cv2.imshow(WINDOW, render_hud(frame, last))
        key = cv2.waitKeyEx(0); ak = key & 0xFF
        if key in ARROW_TO_ACTION:
            act = ARROW_TO_ACTION[key]
        elif ak in (ord("q"), 27):
            break
        elif ak == ord("r"):
            frame = fresh(); last = "-"; continue
        elif ak in KEY_TO_ACTION:
            act = KEY_TO_ACTION[ak]
        else:
            continue
        a = torch.full((1,), act, device=device, dtype=torch.long)
        frame = model(frame, a).clamp(0, 1)   # autoregressive: feed prediction back
        last = ACTION_NAME[act]
    cv2.destroyAllWindows()
    print("Bye.")


if __name__ == "__main__":
    main()
