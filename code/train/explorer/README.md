# GenEx-Explorer 🌐🏃🏻

**GenEx-Explorer** is a training pipeline built on top of [Stable Video Diffusion (SVD)](https://stability.ai/stable-video) for panoramic world exploration. It creates a fine-tuned model that can generate videos that explore forward paths into panoramic images, enabling virtual navigation through static images.

This implementation focuses on fine-tuning the temporal transformer blocks of SVD to create smooth forward-moving video sequences from panoramic input images。

The training code is adapted from [SVD_Xtend](https://github.com/pixeli99/SVD_Xtend).

## 📂 Dataset Structure

The training code expects your dataset to be organized in the following structure:

```
base_folder/
├── video_sequence_1/
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   ├── ...
│   └── frame_0050.jpg
├── video_sequence_2/
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   ├── ...
│   └── frame_0050.jpg
...
```

Each subfolder should contain exactly 50 frames representing a sequence of forward motion through a panoramic environment. The training script will sample continuous sequences of frames (default: 25 frames) from these folders during training.

## 📥 Creating a Sample Dataset from Genex-DB-World-Exploration

The Genex-DB-World-Exploration dataset includes 4 different scenes: `realistic` (649 rows), `low_texture` (8.41k rows), `anime` (955 rows), and `real_world` (2.49k rows). Each row contains a `['video']` field with an MP4 video that needs to be converted into the frame-based format required by GenEx-Explorer.

The following script demonstrates how to process this dataset and extract the frames in the required format:

### Sample Usage

To start, download the GenEx-DB from huggingface

```bash
git lfs install
git clone https://huggingface.co/datasets/genex-world/Genex-DB-World-Exploration
```

You can use the following script to prepare the Genex-DB-World-Exploration dataset for GenEx-Explorer training. This script extracts frames from the MP4 videos in the dataset and organizes them in the required folder structure:

```bash
python prepare_dataset.py \
    --dataset_path /path/to/genex_db_dataset \
    --output_dir ./Genex-DB \
    --scene_types realistic low_texture anime real_world \
    --frames_per_video 50 \
    --num_workers 8 \
    --image_size 1024 576
```

### How It Works

The script:

1. Loads each dataset (realistic, low_texture, anime, real_world)
2. For each video in the dataset:
   - Creates a folder for the video (e.g., `realistic/video_00001/`)
   - Extracts 50 frames evenly distributed throughout the video
   - Resizes frames to the specified dimensions (default: 1024×576)
   - Saves frames as JPEG images with proper sequential naming (frame_0001.jpg, frame_0002.jpg, etc.)

### Example Integration with Training

After processing the dataset, you can train your model by pointing to the processed dataset folder:

```bash
accelerate launch train_world_explorer \
    --base_folder=./Genex-DB \
    --pretrained_model_name_or_path=stabilityai/stable-video-diffusion-img2vid-xt-1-1 \
    --num_frames=25 \
    --width=1024 \
    --height=576 \
    --output_dir=./models/genex_explorer_realistic
```

You can process and train on each scene type separately to create specialized models, or combine the processed data for a more generalized model.

## 🚀 Training Your Own Model

### Command Structure

```bash
accelerate launch --config_file /path/to/accelerate_config.yaml train_world_explorer \
    --base_folder=/path/to/dataset \
    --pretrained_model_name_or_path=/path/to/svd/model \
    --num_frames=25 \
    --width=1024 \
    --height=576 \
    --output_dir=/path/to/output \
    --per_gpu_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000 \
    --checkpointing_steps=500 \
    --checkpoints_total_limit=1 \
    --learning_rate=5e-5 \
    --lr_warmup_steps=0 \
    --seed=123 \
    --mixed_precision="fp16" \
    --validation_steps=100 \
    --report_to=wandb \
    --lr_scheduler=constant \
    --num_validation_images=1 \
    --validation_images_folder=/path/to/validation/images
```

### Example Usage

Here's a concrete example with realistic parameters:

```bash
accelerate launch --config_file ./accelerate_config.yaml train_world_explorer.py \
    --base_folder=./data/panoramic_scenes \
    --pretrained_model_name_or_path=stabilityai/stable-video-diffusion-img2vid-xt-1-1 \
    --num_frames=25 \
    --width=1024 \
    --height=576 \
    --output_dir=./models/genex_explorer_v1 \
    --per_gpu_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=5000 \
    --checkpointing_steps=500 \
    --checkpoints_total_limit=3 \
    --learning_rate=5e-5 \
    --lr_warmup_steps=100 \
    --seed=42 \
    --mixed_precision="fp16" \
    --validation_steps=200 \
    --report_to=wandb \
    --lr_scheduler=constant \
    --num_validation_images=2 \
    --validation_images_folder=./evaluation/demo_images \
    --scl_loss \
    --num_scl_samples=3
```

The `--scl_loss` flag enables Spherical Contrastive Loss, which helps the model learn consistent panoramic representations through random rotations.

## 🔧 Requirements

```
torch>=2.0.0
diffusers>=0.24.0
transformers>=4.30.0
accelerate>=0.20.0
einops>=0.6.0
pillow>=9.4.0
opencv-python>=4.7.0
numpy>=1.24.0
tqdm>=4.65.0
wandb>=0.15.0
datasets
```

For optimal performance:
- CUDA-capable GPU with at least 16GB VRAM per GPU
- For multi-GPU training: NVIDIA NVLink or similar for efficient communication
- xformers for memory-efficient attention (enable with `--enable_xformers_memory_efficient_attention`)

## 📚 Model Architecture

GenEx-Explorer fine-tunes the temporal transformer blocks of the UNet in Stable Video Diffusion while keeping the rest of the model frozen. This approach specifically targets the temporal relationships between frames while preserving the spatial understanding capabilities of the original model.

Key components:
- **UNetSpatioTemporalConditionModel**: The core architecture from SVD
- **EDM-based training**: Using the Elucidated Diffusion Models framework for noise scheduling
- **Spherical Contrastive Loss (optional)**: Enhances panoramic understanding through rotational consistency
- **EMA (optional)**: Exponential Moving Average of model weights for stability

The training process uses classifier-free guidance through conditioning dropout to enable better control at inference time.

## 🙏 Acknowledgements

- [Stability AI](https://stability.ai/) for the Stable Video Diffusion model
- [Hugging Face](https://huggingface.co/) for the Diffusers library
- The training code by [SVD_Xtend](https://github.com/pixeli99/SVD_Xtend)

This implementation builds upon these foundational works to create a specialized model for panoramic exploration.
