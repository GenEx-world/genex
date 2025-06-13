import os
import argparse
import torch
import cv2
import numpy as np
import json

from calculate_fvd import calculate_fvd
from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim
from calculate_lpips import calculate_lpips

def load_video_to_tensor(video_path, target_size=64):
    """
    Loads a video using OpenCV, resizes frames to (target_size x target_size),
    and normalizes pixel values to [0,1], returning a tensor [T, C, H, W].
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize to (target_size, target_size)
        frame = cv2.resize(frame, (target_size, target_size))
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to float32 and normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        # If the video is empty or cannot be read properly, handle gracefully
        return None
    
    # frames shape: [T, H, W, 3]
    frames = np.stack(frames, axis=0)
    
    # Convert to PyTorch tensor: [T, 3, H, W]
    frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)
    return frames_tensor

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate all navigated/original videos into a single batch and compute metrics once."
    )
    parser.add_argument(
        "--input_folder", 
        type=str, 
        required=True, 
        help="Path to the main folder containing subfolders with navigated.mp4/original.mp4."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for computations (e.g., 'cuda' or 'cpu')."
    )
    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Get all subfolders
    subfolders = [
        d for d in os.listdir(args.input_folder) 
        if os.path.isdir(os.path.join(args.input_folder, d))
    ]

    # Lists to store video tensors
    navigated_list = []
    original_list = []

    for subfolder in subfolders:
        folder_path = os.path.join(args.input_folder, subfolder)
        
        navigated_path = os.path.join(folder_path, "navigated.mp4")
        original_path  = os.path.join(folder_path, "original.mp4")
        
        if not (os.path.exists(navigated_path) and os.path.exists(original_path)):
            # Skip if missing either file
            continue
        
        nav_tensor = load_video_to_tensor(navigated_path, target_size=64)
        org_tensor = load_video_to_tensor(original_path, target_size=64)

        if nav_tensor is None or org_tensor is None:
            # Skip if any video could not be loaded
            continue

        # Each shape is [T, 3, 64, 64]
        navigated_list.append(nav_tensor)
        original_list.append(org_tensor)

    if len(navigated_list) == 0:
        print("No valid videos found. Exiting.")
        return

    # Stack them along the 0th dimension to form a batch
    # final shape: [N, T, 3, 64, 64] where N is the number of subfolders
    videos1 = torch.stack(navigated_list, dim=0).to(device)
    videos2 = torch.stack(original_list, dim=0).to(device)

    # Compute metrics once for the entire batch
    result = {}
    result['fvd'] = calculate_fvd(videos1, videos2, device, method='styleganv')
    result['ssim'] = calculate_ssim(videos1, videos2)
    result['psnr'] = calculate_psnr(videos1, videos2)
    result['lpips'] = calculate_lpips(videos1, videos2, device)

    # Print JSON
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()
