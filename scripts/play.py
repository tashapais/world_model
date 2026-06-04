"""Real-time, keyboard-playable world model.

Seeds the model with a short ground-truth clip, then lets you drive it one frame
at a time by pressing keys. Each key press picks a latent *action code* and runs a
single dynamics step conditioned on it, decoding and displaying the next frame.

The action codes are learned unsupervised (FSQ codebook), so the keys map to
arbitrary codes rather than predefined directions — you discover what each code
does by playing. Use the number keys / w-a-s-d to probe each code, then steer.

Controls (shown in the on-screen HUD):
    w a s d        -> action codes 0 1 2 3
    0 1 2 .. N-1   -> action code by number
    space          -> repeat the last action (advance)
    r              -> reset to a fresh random ground-truth clip
    q / ESC        -> quit

Run:
    python -m scripts.play                       # uses configs/inference_zelda.yaml
    python -m scripts.play --config configs/inference.yaml temperature=0.8
"""
import os
# allow CPU fallback for any op not yet implemented on Apple MPS
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import random
from typing import Optional

import cv2
import numpy as np
import torch

from datasets.data_utils import load_data_and_data_loaders
from utils.config import InferenceConfig, load_config
from utils.utils import find_latest_checkpoint
from utils.inference_utils import load_models, build_action_latent_from_index

# MaskGIT decoding steps per frame. Lower = snappier, higher = better quality.
MASKGIT_STEPS = 8
# upscale the (small) generated frame to roughly this many pixels on the long side
DISPLAY_LONG_SIDE = 512
HUD_H = 54  # height of the on-screen text bar
WINDOW = "world model (press w/a/s/d, 0-9 to act | r reset | q quit)"

# physical key -> action code index
KEY_TO_ACTION = {ord("w"): 0, ord("a"): 1, ord("s"): 2, ord("d"): 3}


def pick_device(requested: str) -> torch.device:
    """Honor the requested device if available, else fall back sensibly."""
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_checkpoints(args: InferenceConfig) -> None:
    """Fill in any missing model paths from the latest available run (mirrors run_inference)."""
    base_dir = os.getcwd()

    def missing(path: Optional[str]) -> bool:
        return (path is None) or (not os.path.exists(path))

    if args.use_latest_checkpoints or missing(args.video_tokenizer_path):
        args.video_tokenizer_path = find_latest_checkpoint(base_dir, "video_tokenizer")
    if args.use_latest_checkpoints or missing(args.latent_actions_path):
        args.latent_actions_path = find_latest_checkpoint(base_dir, "latent_actions")
    if args.use_latest_checkpoints or missing(args.dynamics_path):
        args.dynamics_path = find_latest_checkpoint(base_dir, "dynamics")

    for label, path in [("video_tokenizer", args.video_tokenizer_path),
                        ("latent_actions", args.latent_actions_path),
                        ("dynamics", args.dynamics_path)]:
        if missing(path):
            raise FileNotFoundError(
                f"{label} checkpoint not found ({path}). Set it in your config or enable use_latest_checkpoints.")
        print(f"Using {label} checkpoint: {path}")


def random_context_clip(data_loader, context_window, device):
    """Grab the first `context_window` frames of a random clip as seed context."""
    idx = random.randint(0, len(data_loader.dataset) - 1)
    frames = data_loader.dataset[idx][0]  # [T, C, H, W]
    frames = frames.unsqueeze(0).to(device)  # [1, T, C, H, W]
    return frames[:, :context_window]


def frame_to_bgr(frame_chw: torch.Tensor) -> np.ndarray:
    """[-1,1] CHW tensor -> uint8 HWC BGR for cv2."""
    img = (frame_chw.detach().float().cpu() + 1) / 2  # -> [0,1]
    img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()  # HWC
    img = (img * 255).astype(np.uint8)
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def render(frame_chw, frame_no, last_action, n_actions) -> np.ndarray:
    """Upscale the frame and stack a HUD bar above it."""
    bgr = frame_to_bgr(frame_chw)
    h, w = bgr.shape[:2]
    scale = max(1, DISPLAY_LONG_SIDE // max(h, w))
    big = cv2.resize(bgr, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    hud = np.full((HUD_H, big.shape[1], 3), 24, dtype=np.uint8)
    act = "-" if last_action is None else str(last_action)
    cv2.putText(hud, f"frame {frame_no}   action {act}/{n_actions - 1}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 80), 1, cv2.LINE_AA)
    cv2.putText(hud, "w/a/s/d & 0-9 = act   space = repeat   r = reset   q = quit",
                (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([hud, big])


@torch.no_grad()
def step(video_tokenizer, latent_action_model, dynamics_model, context_frames,
         action_index, inferred_actions, args, device):
    """Run one dynamics step conditioned on `action_index`; return the new frame."""
    video_indices = video_tokenizer.tokenize(context_frames)
    video_latents = video_tokenizer.quantizer.get_latents_from_indices(video_indices)

    _, action_latent = build_action_latent_from_index(
        action_index, inferred_actions, context_frames, latent_action_model,
        args.context_window, args.prediction_horizon, device)

    def idx_to_latents(idx):
        return video_tokenizer.quantizer.get_latents_from_indices(idx, dim=-1)

    next_video_latents = dynamics_model.forward_inference(
        context_latents=video_latents,
        prediction_horizon=args.prediction_horizon,
        num_steps=MASKGIT_STEPS,
        index_to_latents_fn=idx_to_latents,
        conditioning=action_latent,
        temperature=args.temperature,
        schedule=getattr(args, "maskgit_schedule", "exp"),
    )
    next_frames = video_tokenizer.detokenize(next_video_latents)  # [1, T, C, H, W]
    return next_frames[:, -args.prediction_horizon:]


def main():
    args: InferenceConfig = load_config(
        InferenceConfig,
        default_config_path=os.path.join(os.getcwd(), "configs", "inference_zelda.yaml"))

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = pick_device(str(args.device))
    args.device = str(device)
    print(f"Device: {device}")

    resolve_checkpoints(args)
    video_tokenizer, latent_action_model, dynamics_model = load_models(
        args.video_tokenizer_path, args.latent_actions_path, args.dynamics_path,
        device, use_actions=True)
    n_actions = latent_action_model.quantizer.codebook_size
    print(f"Latent action codebook size: {n_actions}")

    overrides = {"preload_ratio": args.preload_ratio} if getattr(args, "preload_ratio", None) else {}
    _, _, data_loader, _, _ = load_data_and_data_loaders(
        dataset=args.dataset, batch_size=1, num_frames=args.context_window, **overrides)

    # prediction_horizon=1 keeps play turn-based and the action conditioning simple
    args.prediction_horizon = 1

    def new_episode():
        ctx = random_context_clip(data_loader, args.context_window, device)
        return ctx, ctx.clone(), []  # context window, full rollout, action history

    context_frames, generated, inferred_actions = new_episode()
    last_action = None
    frame_no = args.context_window

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    print("\nWindow open. Click it, then press keys. (q or ESC to quit)\n")

    while True:
        cv2.imshow(WINDOW, render(generated[0, -1], frame_no, last_action, n_actions))
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):  # q or ESC
            break
        if key == ord("r"):
            context_frames, generated, inferred_actions = new_episode()
            last_action, frame_no = None, args.context_window
            continue

        if key == ord(" "):
            action = last_action if last_action is not None else 0
        elif key in KEY_TO_ACTION:
            action = KEY_TO_ACTION[key]
        elif ord("0") <= key <= ord("9"):
            action = key - ord("0")
        else:
            continue  # ignore unmapped keys

        if action >= n_actions:
            print(f"action {action} >= codebook size {n_actions}; ignoring")
            continue

        context_frames = generated[:, -args.context_window:]
        next_frame = step(video_tokenizer, latent_action_model, dynamics_model,
                          context_frames, action, inferred_actions, args, device)
        generated = torch.cat([generated, next_frame], dim=1)
        last_action, frame_no = action, frame_no + 1

    cv2.destroyAllWindows()
    print("Bye.")


if __name__ == "__main__":
    main()
