#!/usr/bin/env python
# coding=utf-8
"""
Prepare the Genex-DB-World-Exploration dataset straight from folders of videos and
save extracted frames back into the same root folder as:

    ./Genex-DB/
        anime_video0/0.png … 49.png
        anime_video1/…
        realistic_video0/…
        low_texture_video0/…
        …

Pass scene types in lowercase (e.g. low_texture); the script matches actual
folder names case-insensitively.
"""

import os, cv2, argparse, concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Tuple

# ────────────────────────────── CLI ──────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Extract video frames for GenEx-Explorer")
    p.add_argument("--dataset_path", default="./Genex-DB",
                   help="Root directory holding scene sub-folders with videos")
    p.add_argument("--output_dir", default=None,
                   help="Where to write frames (default: same as --dataset_path)")
    p.add_argument("--scene_types", nargs="+",
                   default=["realistic", "low_texture", "anime", "real_world"],
                   help="Scene types to process (lowercase)")
    p.add_argument("--frames_per_video", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--image_size", type=int, nargs=2, default=[1024, 576],
                   help="width height")
    return p.parse_args()

# ───────────────────── helper: map names case-insensitively ───────────────────
def _norm(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())

def find_scene_dir(root: Path, scene_type: str) -> Path:
    target = _norm(scene_type)
    for entry in root.iterdir():
        if entry.is_dir() and _norm(entry.name) == target:
            return entry
    raise FileNotFoundError(f"No directory matching '{scene_type}' under {root}")

def list_videos(scene_dir: Path) -> List[Path]:
    exts = {".mp4", ".mov", ".mkv", ".avi"}
    return sorted([p for p in scene_dir.iterdir() if p.suffix.lower() in exts])

# ───────────────────────────── core logic ────────────────────────────────────
def extract_frames(video_path: Path, out_dir: Path,
                   num_frames: int, image_size: Tuple[int, int]) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] cannot open {video_path}")
        return False

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        print(f"[WARN] 0 frames in {video_path}")
        return False

    idxs = list(range(total)) if total <= num_frames else \
           [int(i * total / num_frames) for i in range(num_frames)]

    written = 0
    for k, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.resize(frame, image_size)
        cv2.imwrite(str(out_dir / f"{k}.png"), frame)
        written += 1

    cap.release()
    return written == len(idxs)

def _process_one(i: int, video_path: Path, prefix: str,
                 out_root: Path, frames: int, img_sz: Tuple[int, int]) -> Dict[str, Any]:
    out_dir = out_root / f"{prefix}_video{i}"
    if out_dir.exists() and len(list(out_dir.glob("*.png"))) >= frames:
        return {"idx": i, "skipped": True, "success": True}
    ok = extract_frames(video_path, out_dir, frames, img_sz)
    return {"idx": i, "skipped": False, "success": ok}

def process_scene(scene_type: str, args: argparse.Namespace, out_root: Path) -> None:
    scene_dir = find_scene_dir(Path(args.dataset_path), scene_type)
    videos = list_videos(scene_dir)
    if not videos:
        print(f"[INFO] No videos for '{scene_type}' — skipping.")
        return

    prefix = scene_type  # already lower-case
    print(f"[INFO] {scene_type}: {len(videos)} videos → {out_root}")

    results = []
    with concurrent.futures.ProcessPoolExecutor(args.num_workers) as ex:
        futs = [ex.submit(_process_one, i, vp, prefix, out_root,
                          args.frames_per_video, tuple(args.image_size))
                for i, vp in enumerate(videos)]
        for f in concurrent.futures.as_completed(futs):
            results.append(f.result())

    succ = sum(r["success"] for r in results)
    skip = sum(r["skipped"] for r in results)
    print(f"[DONE] {scene_type}: {succ}/{len(videos)} ok  |  {skip} skipped")

# ────────────────────────────── main ─────────────────────────────────────────
def main() -> None:
    args = parse_args()
    out_root = Path(args.output_dir) if args.output_dir else Path(args.dataset_path)
    for st in args.scene_types:
        try:
            process_scene(st, args, out_root)
        except Exception as e:
            print(f"[ERROR] {st}: {e}")
    print("All done!")

if __name__ == "__main__":
    main()
