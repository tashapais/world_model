"""Build (or extend) the Zelda frame dataset straight from gameplay video.

The original tinyworlds Zelda data was not bespoke -- it was captured from the
fan-made *Ocarina of Time 2D* demake and downsampled to 128x128. See:
  - datasets/data_utils.load_zelda  ->  '/data/Zelda oot2d 1 Cut.mp4'
  - OoT 2D archive: https://archive.org/details/zeldaoot2d
So "getting more data" just means running more OoT 2D footage through the same
preprocessing (cv2: BGR->RGB, INTER_AREA resize) and writing it to an h5 with a
uint8 `frames` dataset of shape [N, H, W, 3] -- the exact format the loader and
the HuggingFace `zelda_frames.h5` already use.

Examples
--------
Local longplay file(s), every 2nd frame, 128x128, written to a new h5:
    python scripts/ingest_zelda.py --videos "footage/*.mp4" --step 2 \
        --out data/zelda_frames_extended.h5

Download longplays first (needs `pip install yt-dlp`), then ingest:
    python scripts/ingest_zelda.py --urls https://youtu.be/XXXX https://youtu.be/YYYY \
        --out data/zelda_frames_extended.h5

Combine the existing 72k-frame set with new footage into one file:
    python scripts/ingest_zelda.py --videos "footage/*.mp4" \
        --include-existing data/zelda_frames.h5 \
        --out data/zelda_frames_extended.h5

To actually train on the result, either overwrite data/zelda_frames.h5 (the path
load_zelda reads) or point datasets/data_utils.load_zelda at --out.

Frames are streamed to disk in chunks, so multi-hour longplays don't need to fit
in RAM.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from typing import List

import cv2
import h5py
import numpy as np
from tqdm import tqdm

CHUNK = 1000  # frames buffered before each h5 write


def expand_videos(patterns: List[str]) -> List[str]:
    """Expand globs/paths into a sorted, de-duplicated list of existing files."""
    files: List[str] = []
    for pat in patterns:
        matched = glob.glob(pat)
        files.extend(matched if matched else ([pat] if os.path.exists(pat) else []))
    missing = [p for p in patterns if not glob.glob(p) and not os.path.exists(p)]
    for m in missing:
        print(f"[warn] no file matched: {m}")
    return sorted(dict.fromkeys(files))


def download_urls(urls: List[str], out_dir: str) -> List[str]:
    """Download each URL to out_dir via yt-dlp; return the resulting file paths."""
    if shutil.which("yt-dlp") is None:
        sys.exit("yt-dlp not found. Install it with: pip install yt-dlp")
    paths: List[str] = []
    for url in urls:
        print(f"Downloading {url}")
        # merge to mp4; %(id)s keeps names unique and predictable
        template = os.path.join(out_dir, "%(id)s.%(ext)s")
        subprocess.run(
            ["yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio/best", "--merge-output-format", "mp4",
             "-o", template, url],
            check=True,
        )
    paths = sorted(glob.glob(os.path.join(out_dir, "*.mp4")))
    if not paths:
        sys.exit("yt-dlp produced no .mp4 files.")
    return paths


def open_h5_frames(out_path: str, size: int, append: bool):
    """Open the output h5 and return (file, resizable `frames` dataset)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    if append and os.path.exists(out_path):
        f = h5py.File(out_path, "a")
        dset = f["frames"]
        if tuple(dset.shape[1:]) != (size, size, 3):
            f.close()
            sys.exit(f"--append size mismatch: existing {dset.shape[1:]} vs requested {(size, size, 3)}")
        return f, dset
    if os.path.exists(out_path) and not append:
        print(f"[warn] overwriting existing {out_path}")
    f = h5py.File(out_path, "w")
    dset = f.create_dataset(
        "frames", shape=(0, size, size, 3), maxshape=(None, size, size, 3),
        dtype="uint8", chunks=(CHUNK, size, size, 3), compression="lzf",
    )
    return f, dset


def append_frames(dset, frames: np.ndarray) -> None:
    """Grow the dataset and write a block of frames at the end."""
    if len(frames) == 0:
        return
    n = dset.shape[0]
    dset.resize(n + len(frames), axis=0)
    dset[n:] = frames


def copy_existing(dset, existing_path: str, size: int) -> int:
    """Append every frame from an existing h5 (resizing if needed)."""
    with h5py.File(existing_path, "r") as src:
        src_frames = src["frames"]
        total = len(src_frames)
        for i in tqdm(range(0, total, CHUNK), desc=f"copying {os.path.basename(existing_path)}"):
            block = src_frames[i:i + CHUNK][:]
            if tuple(block.shape[1:]) != (size, size, 3):
                block = np.stack([cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA) for fr in block])
            append_frames(dset, block)
    return total


def ingest_video(dset, video_path: str, size: int, step: int) -> int:
    """Stream a video into the h5: read sequentially, keep every `step`-th frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[warn] could not open {video_path}")
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    buf: List[np.ndarray] = []
    kept = 0
    idx = 0
    with tqdm(total=(total // step if total else None), desc=os.path.basename(video_path)) as bar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
                buf.append(frame)
                kept += 1
                bar.update(1)
                if len(buf) >= CHUNK:
                    append_frames(dset, np.asarray(buf, dtype=np.uint8))
                    buf.clear()
            idx += 1
    if buf:
        append_frames(dset, np.asarray(buf, dtype=np.uint8))
    cap.release()
    return kept


def main():
    p = argparse.ArgumentParser(description="Extract Zelda (OoT 2D) gameplay frames into an h5 dataset.")
    p.add_argument("--videos", nargs="*", default=[], help="local video files or globs")
    p.add_argument("--urls", nargs="*", default=[], help="video URLs to download via yt-dlp")
    p.add_argument("--out", default="data/zelda_frames_extended.h5", help="output h5 path")
    p.add_argument("--size", type=int, default=128, help="square output resolution")
    p.add_argument("--step", type=int, default=1, help="keep every Nth raw frame (subsample/dedup)")
    p.add_argument("--append", action="store_true", help="append to --out instead of overwriting")
    p.add_argument("--include-existing", default=None, help="prepend frames from an existing h5 (e.g. data/zelda_frames.h5)")
    args = p.parse_args()

    if not args.videos and not args.urls and not args.include_existing:
        p.error("nothing to do: pass --videos, --urls, and/or --include-existing")
    if args.step < 1:
        p.error("--step must be >= 1")

    tmp_dir = None
    videos = expand_videos(args.videos)
    if args.urls:
        tmp_dir = tempfile.mkdtemp(prefix="zelda_ingest_")
        videos += download_urls(args.urls, tmp_dir)

    f, dset = open_h5_frames(args.out, args.size, args.append)
    try:
        total_kept = 0
        if args.include_existing:
            total_kept += copy_existing(dset, args.include_existing, args.size)
        for v in videos:
            total_kept += ingest_video(dset, v, args.size, args.step)
        final = dset.shape[0]
    finally:
        f.close()
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\nWrote {final} frames ({args.size}x{args.size}) to {args.out}")
    print("Point datasets/data_utils.load_zelda at this file (or overwrite data/zelda_frames.h5) to train on it.")


if __name__ == "__main__":
    main()
