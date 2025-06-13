# Video Metrics Evaluation

## Overview
This toolkit calculates various video quality metrics between pairs of videos. It measures the similarity and consistency between original videos and their processed/generated counterparts.

Supported metrics:
- FVD (Fréchet Video Distance)
- LPIPS (Learned Perceptual Image Patch Similarity)
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)

The code is adapted from https://github.com/JunyaoHu/common_metrics_on_video_quality

## Input Data Structure
Prepare your videos with the following directory structure:

```
main_folder/
├── subfolder1/
│   ├── navigated.mp4
│   └── original.mp4
├── subfolder2/
│   ├── navigated.mp4
│   └── original.mp4
...
```

## Running the Evaluation

To calculate all metrics in one command:

```
python calculate_scores.py --input_folder <PATH_TO_MAIN_FOLDER> --device <DEVICE>
```

Example:
```
python calculate_scores.py --input_folder ./results/test_run --device cuda
```

## Individual Metrics

Each metric can also be calculated independently:

### FVD
Measures the distance between feature distributions of real and generated videos:
```python
from calculate_fvd import calculate_fvd
result = calculate_fvd(videos1, videos2, device, method='styleganv')
```

### LPIPS
Computes perceptual similarity between corresponding frames:
```python
from calculate_lpips import calculate_lpips
result = calculate_lpips(videos1, videos2, device)
```

### PSNR
Measures pixel-level fidelity:
```python
from calculate_psnr import calculate_psnr
result = calculate_psnr(videos1, videos2)
```

### SSIM
Evaluates structural similarity:
```python
from calculate_ssim import calculate_ssim
result = calculate_ssim(videos1, videos2)
```

## Requirements

```
torch>=1.10.0
numpy>=1.20.0
opencv-python>=4.5.0
tqdm>=4.50.0
lpips>=0.1.4
```

Install dependencies:
```
pip install -r requirements.txt
```