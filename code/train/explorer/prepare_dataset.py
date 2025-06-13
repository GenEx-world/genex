#!/usr/bin/env python
# coding=utf-8
"""
Prepare the Genex-DB-World-Exploration dataset *directly from folders of videos*.
Folder names may be capitalized or use hyphens/underscores; pass their lowercase
form (e.g. --scene_types low_texture) and the script will match them.
"""

import os, cv2, argparse, concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Tuple

# --------------------------------------------------------------------------- CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Prepare Genex-DB-World-Exploration videos")
    p.add_argument("--dataset_path", required=True,
                   help="Root directory that contains per-scene sub-folders")
    p.add_argument("--output_dir", default="./processed_dataset",
                   help="Where extracted frames are written")
    p.add_argument("--scene_types", nargs="+",
                   default=["realistic", "low_texture", "anime", "real_world"],
                   help="Scene types")
    p.add_argument("--frames_per_video", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--image_size", type=int, nargs=2, default=[1024, 576],
                   help="width height")
    return p.parse_args()

# ---------------------------------------------------------------------- helpers
def _normalize(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())

def find_scene_dir(root: str, scene_type: str) -> Path:
    target = _normalize(scene_type)
    for entry in os.listdir(root):
        p = Path(root) / entry
        if p.is_dir() and _normalize(entry) == target:
            return p
    raise FileNotFoundError(f"No directory matching '{scene_type}' found under {root}")

def list_videos(scene_dir: Path) -> List[Path]:
    exts = {".mp4", ".mov", ".mkv", ".avi"}
    return sorted([p for p in scene_dir.iterdir() if p.suffix.lower() in exts])

# ------------------------------------------------------------------ core logic
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

def _process_one(i: int, video_path: Path, scene_type: str,
                 out_base: Path, frames: int, img_size: Tuple[int, int]) -> Dict[str, Any]:
    out_dir = out_base / scene_type / f"video_{i:05d}"
    if out_dir.exists() and len(list(out_dir.glob("*.png"))) >= frames:
        return {"idx": i, "skipped": True, "success": True}
    ok = extract_frames(video_path, out_dir, frames, img_size)
    return {"idx": i, "skipped": False, "success": ok}

def process_scene(scene_type: str, args: argparse.Namespace) -> None:
    scene_dir = find_scene_dir(args.dataset_path, scene_type)
    videos = list_videos(scene_dir)
    if not videos:
        print(f"[INFO] No videos found for '{scene_type}' — skipping.")
        return

    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] {scene_type}: {len(videos)} videos → {out_base/scene_type}")

    results = []
    with concurrent.futures.ProcessPoolExecutor(args.num_workers) as ex:
        futs = [ex.submit(_process_one, i, vp, scene_type, out_base,
                          args.frames_per_video, tuple(args.image_size))
                for i, vp in enumerate(videos)]
        for f in concurrent.futures.as_completed(futs):
            results.append(f.result())

    succ = sum(r["success"] for r in results)
    skip = sum(r["skipped"] for r in results)
    print(f"[DONE] {scene_type}: {succ}/{len(videos)} ok  |  {skip} skipped")

# --------------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    for st in args.scene_types:
        try:
            process_scene(st, args)
        except Exception as e:
            print(f"[ERROR] {st}: {e}")
    print("All done!")

if __name__ == "__main__":
    main()
