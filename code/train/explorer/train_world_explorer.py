#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to fine-tune Stable Video Diffusion."""
import argparse
import random
import logging
import math
import os
import re
import cv2
import shutil
from pathlib import Path
from urllib.parse import urlparse

import accelerate
import numpy as np
import PIL
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import RandomSampler
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange

import diffusers
from diffusers import StableVideoDiffusionPipeline
from diffusers.models.lora import LoRALinearLayer
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler, UNetSpatioTemporalConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available

from torch.utils.data import Dataset

from diffusers.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionOutput
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import UNet2DConditionLoadersMixin
from diffusers.utils import BaseOutput
from diffusers.models.attention_processor import CROSS_ATTENTION_PROCESSORS, AttentionProcessor, AttnProcessor
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_3d_blocks import UNetMidBlockSpatioTemporal, get_down_block, get_up_block

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Will error if the minimal version of diffusers is not installed
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")

os.environ["WANDB_PROJECT"] = "Panorama"


def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from a lognormal distribution.
    
    Args:
        shape: The shape of the output tensor
        loc: Mean of the underlying normal distribution
        scale: Standard deviation of the underlying normal distribution
        device: The device to create the tensor on
        dtype: The data type of the tensor
        
    Returns:
        Tensor with samples from a log-normal distribution
    """
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()


class DummyDataset(Dataset):
    def __init__(self, base_folder: str, num_samples=100000, width=1024, height=576, sample_frames=25):
        """Dataset for loading video frames for training.
        
        Args:
            base_folder: Path to the folder containing video frames
            num_samples: Number of samples in the dataset
            width: Width of the frames
            height: Height of the frames
            sample_frames: Number of frames to sample for each video
        """
        self.num_samples = num_samples
        self.base_folder = base_folder
        self.folders = os.listdir(self.base_folder)
        self.channels = 3
        self.width = width
        self.height = height
        self.sample_frames = sample_frames

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Get a sample of video frames.
        
        Args:
            idx: Index of the sample to return
            
        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (sample_frames, channels, height, width)
        """
        # while True:
        chosen_folder = random.choice(self.folders)
        folder_path = os.path.join(self.base_folder, chosen_folder)
        frames = os.listdir(folder_path)
            
            # if len(frames) == 50:
            #     break
            # print('Invalid Folder.')

        # Custom sorting key function to extract numeric part
        def numerical_sort(value):
            numbers = re.findall(r'\d+', value)
            return int(numbers[0]) if numbers else 0

        # Sort the frames by numeric order
        frames.sort(key=numerical_sort)

        # Ensure the selected folder has at least `sample_frames` frames
        if len(frames) < self.sample_frames:
            raise ValueError(
                f"The selected folder '{chosen_folder}' contains fewer than `{self.sample_frames}` frames.")

        # Randomly select a start index for frame sequence
        start_idx = random.randint(0, len(frames) - self.sample_frames)
        selected_frames = frames[start_idx:start_idx + self.sample_frames]

        # Initialize a tensor to store the pixel values
        pixel_values = torch.empty((self.sample_frames, self.channels, self.height, self.width))

        # Load and process each frame
        for i, frame_name in enumerate(selected_frames):
            frame_path = os.path.join(folder_path, frame_name)
            with Image.open(frame_path) as img:
                img = img.convert("RGB") 
                # Resize the image and convert it to a tensor
                img_resized = img.resize((self.width, self.height))
                img_tensor = torch.from_numpy(np.array(img_resized)).float()

                # Normalize the image by scaling pixel values to [-1, 1]
                img_normalized = img_tensor / 127.5 - 1

                # Rearrange channels for RGB images
                img_normalized = img_normalized.permute(2, 0, 1)
                pixel_values[i] = img_normalized
                
        return {'pixel_values': pixel_values}


# Image resizing utilities
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    """Resize images with anti-aliasing.
    
    Args:
        input: Input tensor of shape [B, C, H, W]
        size: Target size (H, W)
        interpolation: Interpolation mode
        align_corners: Whether to align corners
        
    Returns:
        Resized tensor
    """
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # Determine sigma (from skimage)
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Determine kernel size (using 2 sigmas)
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(
        input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple for convolution."""
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # For even kernels we need to do asymmetric padding
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    """Apply 2D filtering operation."""
    # Prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(
        device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # Kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # Convolve the tensor with the kernel
    output = torch.nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    """Create a Gaussian kernel."""
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device,
         dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    """Apply 2D Gaussian blur."""
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out


def export_to_video(video_frames, output_video_path, fps):
    """Export frames to a video file.
    
    Args:
        video_frames: List of frames as numpy arrays
        output_video_path: Path to save the output video
        fps: Frames per second
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)


def export_to_gif(frames, output_gif_path, fps):
    """Export frames to a GIF file.
    
    Args:
        frames: List of frames as numpy arrays or PIL Image objects
        output_gif_path: Path to save the output GIF
        fps: Frames per second
    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=500,
                       loop=0)


def tensor_to_vae_latent(t, vae):
    """Convert tensor to VAE latent representation.
    
    Args:
        t: Input tensor of shape [B, F, C, H, W]
        vae: VAE model
        
    Returns:
        Latent representation
    """
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor

    return latents


def reconstruct_from_latent(latents, vae):
    """Reconstruct images from latent representation.
    
    Args:
        latents: Latent representation
        vae: VAE model
        
    Returns:
        Reconstructed images
    """
    latents = latents / vae.config.scaling_factor
    batch_size, video_length, _, _, _ = latents.shape

    latents = rearrange(latents, "b f c h w -> (b f) c h w")
    reconstructed_images = vae.decode(latents, num_frames=video_length).sample

    reconstructed_images = rearrange(reconstructed_images, "(b f) c h w -> b f c h w", f=video_length)

    return reconstructed_images


def rotate_panorama_batch(batch_tensor, rotation_degrees):
    """Rotate a batch of panoramic images.
    
    Args:
        batch_tensor: Tensor of shape [B, F, C, H, W]
        rotation_degrees: Rotation angle in degrees
        
    Returns:
        Rotated tensor
    """
    batch_size, num_images, channels, height, width = batch_tensor.shape
    
    # Combine batch_size and num_images dimensions
    combined_batch = batch_size * num_images
    reshaped_tensor = batch_tensor.view(combined_batch, channels, height, width)

    rotation_radians = torch.deg2rad(torch.tensor(rotation_degrees, dtype=torch.float32))

    # Calculate pixel coordinates
    x = torch.linspace(0, width - 1, width)
    y = torch.linspace(0, height - 1, height)
    xv, yv = torch.meshgrid(x, y, indexing='xy')

    # Calculate longitude and latitude
    longitude = (xv / width) * 2 * torch.pi
    latitude = (yv / height) * torch.pi - (torch.pi / 2)

    # Rotate the longitude
    rotated_longitude = (longitude + rotation_radians) % (2 * torch.pi)

    # Convert back to image coordinates
    uf = rotated_longitude / (2 * torch.pi) * width
    vf = (latitude + (torch.pi / 2)) / torch.pi * height

    # Ensure indices are within bounds
    ui = torch.clamp(uf, 0, width - 1).long()
    vi = torch.clamp(vf, 0, height - 1).long()

    # Apply rotation to each image in the batch
    rotated_images = []
    for i in range(combined_batch):
        panorama_tensor = reshaped_tensor[i]
        rotated_tensor = panorama_tensor[:, vi, ui]  # Apply rotation per channel
        rotated_images.append(rotated_tensor)

    # Stack the rotated images back into a batch
    rotated_batch = torch.stack(rotated_images).view(batch_size, num_images, channels, height, width)

    return rotated_batch


def print_cuda_memory_usage(cuda_index=0, show=False):
    """Print memory usage of a specified CUDA device.
    
    Args:
        cuda_index: The index of the CUDA device to check
        show: Whether to print the results
        
    Returns:
        Tuple of (allocated_memory_MB, reserved_memory_MB)
    """
    if torch.cuda.is_available():
        current_memory_allocated = torch.cuda.memory_allocated(cuda_index)
        current_memory_reserved = torch.cuda.memory_reserved(cuda_index)
        
        # Convert to MB
        current_memory_allocated_MB = current_memory_allocated / 1024 ** 2
        current_memory_reserved_MB = current_memory_reserved / 1024 ** 2

        if show:
            # Print memory usage
            print('-' * 50)
            print(f"CUDA Device {cuda_index}:")
            print(f"  Allocated Memory: {current_memory_allocated_MB:.2f} MB")
            print(f"  Reserved Memory: {current_memory_reserved_MB:.2f} MB")
            print('-' * 50, flush=True)
        return current_memory_allocated_MB, current_memory_reserved_MB
    else:
        print("CUDA is not available.")
        return None, None


def print_tensor_vram_usage(tensor):
    """Print the VRAM usage of a PyTorch tensor in gigabytes.
    
    Args:
        tensor: The input tensor to measure VRAM usage for
    """
    if not tensor.is_cuda:
        print("The tensor is not on a CUDA device.")
        return
    
    # Calculate the VRAM usage in bytes
    vram_usage_bytes = tensor.element_size() * tensor.nelement()

    # Convert to gigabytes (GB)
    vram_usage_gb = vram_usage_bytes / (1024 ** 3)
    
    print(f"VRAM usage of the tensor: {vram_usage_gb:.6f} GB")


def parse_args():
    """Parse command-line arguments for the training script."""
    parser = argparse.ArgumentParser(
        description="Script to train Stable Video Diffusion."
    )
    parser.add_argument(
        "--base_folder",
        required=True,
        type=str,
        help="Path to the folder containing video frames"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=25,
        help="Number of frames to use in each video",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Width of the frames",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=576,
        help="Height of the frames",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]',
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.1,
        help="Conditioning dropout probability for training.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Whether or not to allow TF32 on Ampere GPUs for faster training.",
    )
    parser.add_argument(
        "--use_ema", 
        action="store_true", 
        help="Whether to use EMA model."
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained non-ema model identifier.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of subprocesses to use for data loading.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", 
        type=float, 
        default=1e-2, 
        help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", 
        default=1.0, 
        type=float, 
        help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard log directory.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision training.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help='The integration to report the results and logs to. Supported platforms are "tensorboard" (default), "wandb" and "comet_ml".',
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint of the training state every X updates.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=2,
        help="Max number of checkpoints to store.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help='Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or "latest" to automatically select the last available checkpoint.',
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers for memory efficient attention.",
    )
    parser.add_argument(
        "--pretrain_unet",
        type=str,
        default=None,
        help="Use pre-trained weights for unet block",
    )
    parser.add_argument(
        "--scl_loss",
        action="store_true",
        help="Whether or not to use Spherical Contrastive Loss.",
    )
    parser.add_argument(
        "--num_scl_samples",
        type=int,
        default=1,
        help="How many spherical rotations for SCL to use.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def download_image(url):
    """Download an image from a URL or load from a local path.
    
    Args:
        url: URL or path to the image
        
    Returns:
        PIL Image object
    """
    original_image = (
        lambda image_url_or_path: load_image(image_url_or_path)
        if urlparse(image_url_or_path).scheme
        else PIL.Image.open(image_url_or_path).convert("RGB")
    )(url)
    return original_image


def verify_and_save_images(tensor, output_dir):
    """Save tensor images to disk for debugging.
    
    Args:
        tensor: Tensor of shape [B, F, C, H, W]
        output_dir: Directory to save images to
    """
    batch_size, num_frames, _, _, _ = tensor.shape
    os.makedirs(output_dir, exist_ok=True)
    for b in range(batch_size):
        for i in range(num_frames):
            frame_np = tensor[b, i].permute(1, 2, 0).detach().cpu().numpy()
            frame_np = np.clip(frame_np, 0, 1) * 255
            frame_np = frame_np.astype(np.uint8)
            image = Image.fromarray(frame_np)
            image.save(f"{output_dir}/batch_{b:02d}_frame_{i:02d}.png")


def main():
    """Main training function."""
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message="Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please use `--variant=non_ema` instead."
        )
        
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
        
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    print(f"Number of visible GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} name: {torch.cuda.get_device_name(i)}")

    generator = torch.Generator(
        device=accelerator.device).manual_seed(args.seed)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Set the training seed if specified
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load models
    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision
    )

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision
    )

    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant="fp16")

    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.pretrained_model_name_or_path if args.pretrain_unet is None else args.pretrain_unet,
        subfolder="unet",
        low_cpu_mem_usage=True,
        variant="fp16"
    )

    # Freeze vae and image_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    unet.accelerator = accelerator

    # Configure precision for mixed precision training
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move models to device and cast to weight_dtype
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    print(f'Accelerator Device: {accelerator.device}', '*' * 100)
    vae.to(accelerator.device, dtype=weight_dtype)

    # Create EMA for the unet
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(
        ), model_cls=UNetSpatioTemporalConditionModel, model_config=unet.config)

    # Enable xformers if requested
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")

    # Configure custom saving & loading hooks for accelerator
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(
                    input_dir, "unet_ema"), UNetSpatioTemporalConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                model = models.pop()
                load_model = UNetSpatioTemporalConditionModel.from_pretrained(
                    input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Scale learning rate if requested
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps *
            args.per_gpu_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam: `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # Configure trainable parameters
    parameters_list = []

    # Customize the parameters that need to be trained
    for name, param in unet.named_parameters():
        if 'temporal_transformer_block' in name:
            parameters_list.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    optimizer = optimizer_cls(
        parameters_list,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Log which parameters are frozen and which are being trained
    if accelerator.is_main_process:
        with open('params_freeze.txt', 'w') as rec_txt1, open('params_train.txt', 'w') as rec_txt2:
            for name, para in unet.named_parameters():
                if para.requires_grad is False:
                    rec_txt1.write(f'{name}\n')
                else:
                    rec_txt2.write(f'{name}\n')

    # Create data loader
    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes
    train_dataset = DummyDataset(args.base_folder, width=args.width, height=args.height, sample_frames=args.num_frames)
    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
    )

    # Configure scheduler and training steps
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with accelerator
    unet, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        unet, optimizer, lr_scheduler, train_dataloader
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)
        
    # Get module from DataParallel or DistributedDataParallel wrapper
    if isinstance(unet, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        unet = unet.module

    # Recalculate training steps after dataloader preparation
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # Initialize trackers
    if accelerator.is_main_process:
        accelerator.init_trackers("SVDXtend", config=vars(args))

    # Compute total batch size
    total_batch_size = args.per_gpu_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    # Log training information
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    global_step = 0
    first_epoch = 0

    def encode_image(pixel_values):
        """Encode an image into embeddings using the CLIP image encoder.
        
        Args:
            pixel_values: Pixel values tensor [-1, 1]
            
        Returns:
            Image embeddings
        """
        pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))
        # Unnormalize after resizing
        pixel_values = (pixel_values + 1.0) / 2.0

        # Normalize for CLIP input
        pixel_values = feature_extractor(
            images=pixel_values,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        pixel_values = pixel_values.to(
            device=accelerator.device, dtype=weight_dtype)
        image_embeddings = image_encoder(pixel_values).image_embeds
        return image_embeddings

    def _get_add_time_ids(fps, motion_bucket_id, noise_aug_strength, dtype, batch_size):
        """Get additional time embeddings for the UNet.
        
        Args:
            fps: Frames per second
            motion_bucket_id: Motion bucket ID
            noise_aug_strength: Noise augmentation strength
            dtype: Data type
            batch_size: Batch size
            
        Returns:
            Additional time embeddings tensor
        """
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        return add_time_ids

    # Potentially load weights from a previous checkpoint
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Initialize progress bar
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # Training loop
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            torch.cuda.empty_cache()
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            
            with accelerator.accumulate(unet):
                # Convert images to latent space
                pixel_values = batch["pixel_values"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True
                )
                conditional_pixel_values = pixel_values[:, 0:1, :, :, :]
                latents = tensor_to_vae_latent(pixel_values, vae)

                # Sample noise to add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Add noise to conditional pixels for robustness
                cond_sigmas = rand_log_normal(shape=[bsz,], loc=-3.0, scale=0.5).to(latents)
                noise_aug_strength = cond_sigmas[0]
                cond_sigmas = cond_sigmas[:, None, None, None, None]
                conditional_pixel_values = \
                    torch.randn_like(conditional_pixel_values) * cond_sigmas + conditional_pixel_values
                conditional_latents = tensor_to_vae_latent(conditional_pixel_values, vae)[:, 0, :, :, :]
                conditional_latents = conditional_latents / vae.config.scaling_factor

                # Sample a random timestep for each image (EDM formulation)
                sigmas = rand_log_normal(shape=[bsz,], loc=0.7, scale=1.6).to(latents.device)
                sigmas = sigmas[:, None, None, None, None]
                noisy_latents = latents + noise * sigmas
                timesteps = torch.Tensor(
                    [0.25 * sigma.log() for sigma in sigmas]).to(accelerator.device)

                inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)

                # Get the text embedding for conditioning
                encoder_hidden_states = encode_image(
                    pixel_values[:, 0, :, :, :].float())

                # Prepare additional time IDs
                added_time_ids = _get_add_time_ids(
                    7,  # Fixed fps
                    127,  # Fixed motion_bucket_id
                    noise_aug_strength,
                    encoder_hidden_states.dtype,
                    bsz,
                )
                added_time_ids = added_time_ids.to(latents.device)

                # Apply conditioning dropout for classifier-free guidance during inference
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(
                        bsz, device=latents.device, generator=generator)
                    # Sample masks for the edit prompts
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final text conditioning
                    null_conditioning = torch.zeros_like(encoder_hidden_states)
                    encoder_hidden_states = torch.where(
                        prompt_mask, null_conditioning.unsqueeze(1), encoder_hidden_states.unsqueeze(1))
                    # Sample masks for the original images
                    image_mask_dtype = conditional_latents.dtype
                    image_mask = 1 - (
                        (random_p >= args.conditioning_dropout_prob).to(
                            image_mask_dtype)
                        * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    # Final image conditioning
                    conditional_latents = image_mask * conditional_latents

                # Concatenate the conditional_latents with the noisy_latents
                conditional_latents = conditional_latents.unsqueeze(
                    1).repeat(1, noisy_latents.shape[1], 1, 1, 1)
                inp_noisy_latents = torch.cat(
                    [inp_noisy_latents, conditional_latents], dim=2)

                # Target is the original latents
                target = latents

                # Get model prediction
                model_pred = unet(
                    inp_noisy_latents, timesteps, encoder_hidden_states, added_time_ids=added_time_ids, distributed=True).sample

                # Compute loss weights for EDM formulation
                c_out = -sigmas / ((sigmas**2 + 1)**0.5)
                c_skip = 1 / (sigmas**2 + 1)
                denoised_latents = model_pred * c_out + c_skip * noisy_latents
                weighing = (1 + sigmas ** 2) * (sigmas**-2.0)

                # Compute MSE loss
                video_loss = torch.mean(
                    (weighing.float() * (denoised_latents.float() -
                     target.float()) ** 2).reshape(target.shape[0], -1),
                    dim=1,
                )
                video_loss = video_loss.mean()

                # Compute SCL loss if enabled
                if args.scl_loss:
                    batch_size, video_length, _, _, _ = latents.shape
                    spherical_loss = []
                    num_frames = args.num_frames
                    
                    for _ in range(int(args.num_scl_samples)):  # Random Camera Sample
                        generate_frame_latents = denoised_latents
                        reconstruct_frame = reconstruct_from_latent(generate_frame_latents.half(), vae)
                        target_frame = pixel_values

                        # Apply random rotation
                        rotation_degrees = random.uniform(0, 180)
                        reconstruct_frame = rotate_panorama_batch(reconstruct_frame, rotation_degrees)
                        target_frame = rotate_panorama_batch(target_frame, rotation_degrees)

                        # Convert rotated frames to latents
                        reconstruct_latents = tensor_to_vae_latent(reconstruct_frame, vae)
                        target_latents = tensor_to_vae_latent(target_frame, vae)

                        # Compute spherical MSE loss
                        single_spherical_loss = torch.mean(
                            (weighing.float() * (reconstruct_latents.float() -
                            target_latents.float()) ** 2).reshape(target.shape[0], -1),
                            dim=1,
                        )
                        single_spherical_loss = single_spherical_loss.mean()
                        spherical_loss.append(single_spherical_loss)

                    # Combine spherical loss samples
                    if len(spherical_loss) == 1:
                        spherical_loss = spherical_loss[0]
                    else:
                        spherical_loss = torch.stack(spherical_loss)
                        spherical_loss = spherical_loss.mean()

                    # Weight the losses
                    beta = min((step * 2 / args.max_train_steps), 1) * 0.95 + 0.05
                    if spherical_loss > video_loss * 10:
                        beta = 0
                    alpha = 1 - beta
                    loss = alpha * video_loss + beta * spherical_loss
                else:
                    loss = video_loss

                # Gather the losses across all processes for logging
                avg_loss = accelerator.gather(
                    loss.repeat(args.per_gpu_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update after optimization step
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, 'lr': lr_scheduler.get_lr()[0]}, step=global_step)
                train_loss = 0.0

                # Save checkpoints and run validation
                if accelerator.is_main_process:
                    # Save checkpoint
                    if global_step % args.checkpointing_steps == 0:
                        # Check for checkpoints limit before saving
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1]))

                            # Remove old checkpoints if over the limit
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(
                                    checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        
                    # Run validation
                    if (global_step % args.validation_steps == 0) or (global_step == 1):
                        logger.info(
                            f"Running validation... \nGenerating {args.num_validation_images} videos."
                        )
                        # Create pipeline
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                            
                        # Create the pipeline with unwrapped models
                        pipeline = StableVideoDiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet),
                            image_encoder=accelerator.unwrap_model(
                                image_encoder),
                            vae=accelerator.unwrap_model(vae),
                            revision=args.revision,
                            torch_dtype=weight_dtype,
                        )
                        pipeline = pipeline.to(accelerator.device)
                        pipeline.set_progress_bar_config(disable=True)

                        # Run inference
                        val_save_dir = os.path.join(
                            args.output_dir, "validation_images")

                        if not os.path.exists(val_save_dir):
                            os.makedirs(val_save_dir)

                        with torch.autocast(
                            str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"
                        ):
                            for val_img_idx in range(args.num_validation_images):
                                num_frames = args.num_frames
                                video_frames = pipeline(
                                    load_image(f'demo{val_img_idx}.jpg').resize((args.width, args.height)),
                                    height=args.height,
                                    width=args.width,
                                    num_frames=num_frames,
                                    decode_chunk_size=8,
                                    motion_bucket_id=127,
                                    fps=7,
                                    noise_aug_strength=0.02,
                                ).frames[0]

                                out_file = os.path.join(
                                    val_save_dir,
                                    f"step_{global_step}_val_img_{val_img_idx}.mp4",
                                )

                                for i in range(num_frames):
                                    img = video_frames[i]
                                    video_frames[i] = np.array(img)
                                export_to_gif(video_frames, out_file, 7)

                        if args.use_ema:
                            # Switch back to the original UNet parameters
                            ema_unet.restore(unet.parameters())

                        del pipeline
                        torch.cuda.empty_cache()

            # Update progress bar
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            # Check if we've reached max_train_steps
            if global_step >= args.max_train_steps:
                break

    # Save the final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            image_encoder=accelerator.unwrap_model(image_encoder),
            vae=accelerator.unwrap_model(vae),
            unet=unet,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )
            
    accelerator.end_training()


if __name__ == "__main__":
    main()