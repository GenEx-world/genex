#!/usr/bin/env python
# coding=utf-8
"""
Script to prepare the Genex-DB-World-Exploration dataset for GenEx-Explorer training.
This script extracts frames from videos and organizes them in the required folder structure.
"""

import os
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import concurrent.futures
from typing import List, Dict, Any, Union, Optional


def parse_args():
    """Parse command-line arguments for the dataset preparation script."""
    parser = argparse.ArgumentParser(
        description="Prepare Genex-DB-World-Exploration dataset for GenEx-Explorer training"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset files (CSV or parquet)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./processed_dataset",
        help="Output directory for the processed dataset",
    )
    parser.add_argument(
        "--scene_types",
        nargs="+",
        default=["realistic", "low_texture", "anime", "real_world"],
        help="Scene types to process (default: all)",
    )
    parser.add_argument(
        "--frames_per_video",
        type=int,
        default=50,
        help="Number of frames to extract from each video (default: 50)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for parallel processing",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[1024, 576],
        help="Output image size as width height (default: 1024 576)",
    )
    return parser.parse_args()


def load_dataset(dataset_path: str, scene_type: str) -> pd.DataFrame:
    """
    Load the dataset for a specific scene type.
    
    Args:
        dataset_path: Path to the dataset directory
        scene_type: Type of scene to load ('realistic', 'low_texture', etc.)
        
    Returns:
        DataFrame containing the dataset
    """
    file_path = os.path.join(dataset_path, f"{scene_type}.parquet")
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
    
    file_path = os.path.join(dataset_path, f"{scene_type}.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    
    raise FileNotFoundError(f"Dataset file for '{scene_type}' not found at {dataset_path}")


def extract_frames(
    video_data: bytes,
    output_path: str,
    num_frames: int = 50,
    image_size: tuple = (1024, 576)
) -> bool:
    """
    Extract frames from a video byte stream and save them to disk.
    
    Args:
        video_data: Binary video data
        output_path: Path to save the extracted frames
        num_frames: Number of frames to extract
        image_size: Size of output frames (width, height)
        
    Returns:
        True if successful, False otherwise
    """
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Convert video bytes to numpy array
    video_array = np.frombuffer(video_data, dtype=np.uint8)
    
    # Open video from memory
    cap = cv2.VideoCapture()
    if not cap.open(cv2.CAP_OPENCV_MJPEG, video_array):
        # Try another approach if the first one fails
        with open(f"{output_path}/temp_video.mp4", "wb") as f:
            f.write(video_data)
        cap = cv2.VideoCapture(f"{output_path}/temp_video.mp4")
        if not cap.isOpened():
            print(f"Failed to open video for {os.path.basename(output_path)}")
            return False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print(f"No frames found in video for {os.path.basename(output_path)}")
        return False
    
    # Calculate frame indices to extract (evenly spaced)
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    # Extract and save frames
    frame_count = 0
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Resize frame
        frame = cv2.resize(frame, image_size)
        
        # Save frame
        output_file = os.path.join(output_path, f"frame_{frame_count+1:04d}.jpg")
        cv2.imwrite(output_file, frame)
        frame_count += 1
    
    # Clean up
    cap.release()
    if os.path.exists(f"{output_path}/temp_video.mp4"):
        os.remove(f"{output_path}/temp_video.mp4")
    
    # Check if we got enough frames
    return frame_count == len(frame_indices)


def process_row(
    row_idx: int,
    row: Dict[str, Any],
    scene_type: str,
    output_dir: str,
    frames_per_video: int,
    image_size: tuple
) -> Dict[str, Any]:
    """
    Process a single row from the dataset.
    
    Args:
        row_idx: Index of the row
        row: Row data containing video
        scene_type: Type of scene
        output_dir: Base output directory
        frames_per_video: Number of frames to extract
        image_size: Size of output frames (width, height)
        
    Returns:
        Dictionary with processing results
    """
    # Create output directory for this video
    video_dir = os.path.join(output_dir, scene_type, f"video_{row_idx:05d}")
    
    # Skip if already processed
    if os.path.exists(video_dir) and len(os.listdir(video_dir)) >= frames_per_video:
        return {"row_idx": row_idx, "success": True, "skipped": True}
    
    # Extract frames from video
    try:
        video_data = row['video']
        success = extract_frames(
            video_data, 
            video_dir, 
            frames_per_video,
            image_size
        )
        return {"row_idx": row_idx, "success": success, "skipped": False}
    except Exception as e:
        print(f"Error processing row {row_idx}: {str(e)}")
        return {"row_idx": row_idx, "success": False, "skipped": False, "error": str(e)}


def process_dataset(
    dataset: pd.DataFrame,
    scene_type: str,
    output_dir: str,
    frames_per_video: int = 50,
    num_workers: int = 4,
    image_size: tuple = (1024, 576)
) -> None:
    """
    Process the entire dataset for a specific scene type.
    
    Args:
        dataset: DataFrame containing the dataset
        scene_type: Type of scene being processed
        output_dir: Base output directory
        frames_per_video: Number of frames to extract per video
        num_workers: Number of worker processes
        image_size: Size of output frames (width, height)
    """
    # Create scene output directory
    scene_dir = os.path.join(output_dir, scene_type)
    os.makedirs(scene_dir, exist_ok=True)
    
    print(f"Processing {len(dataset)} videos for scene type '{scene_type}'...")
    
    # Process videos in parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for idx, row in dataset.iterrows():
            future = executor.submit(
                process_row,
                idx,
                row,
                scene_type,
                output_dir,
                frames_per_video,
                image_size
            )
            futures.append(future)
        
        # Track progress
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            results.append(future.result())
    
    # Report results
    successful = sum(1 for r in results if r["success"])
    skipped = sum(1 for r in results if r.get("skipped", False))
    print(f"Processed {len(results)} videos: {successful} successful, {skipped} skipped, {len(results) - successful} failed")


def main():
    """Main function to run the dataset preparation script."""
    args = parse_args()
    
    # Process each scene type
    for scene_type in args.scene_types:
        try:
            print(f"Loading dataset for scene type '{scene_type}'...")
            dataset = load_dataset(args.dataset_path, scene_type)
            
            process_dataset(
                dataset,
                scene_type,
                args.output_dir,
                args.frames_per_video,
                args.num_workers,
                tuple(args.image_size)
            )
            
        except Exception as e:
            print(f"Error processing scene type '{scene_type}': {str(e)}")
    
    print("Dataset preparation complete!")


if __name__ == "__main__":
    main()