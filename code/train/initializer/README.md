# GenEx-Initializer 🌐🎨

**GenEx-Initializer** is a specialized LoRA training pipeline built on top of [FLUX.1-Fill](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) for panoramic world initialization.
It creates a fine-tuned model that can generate complete 360° panoramic environments from partially masked panoramic inputs.

## 📂 Dataset Structure

Based on the code and provided folder structure images, your dataset should be structured as follows:

```
dataset/
├── instance_images/  # Training images folder
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── val/              # Validation images folder
│   ├── 0.png
│   ├── 1.png
│   └── ...
└── pano_mask.png     # Required fixed panorama mask file
```

**Important notes:**
- The training script specifically looks for PNG files in the instance_images directory
- The code will skip any file named "pano_mask.png" in your instance directory
- The fixed mask file at `dataset/pano_mask.png` is required for training
- All your panoramic images should be properly formatted for training


## 📥 Creating a Sample Dataset from GenEx-DB-Panorama-World

You can easily create a training dataset using the [GenEx-DB-Panorama-World](https://huggingface.co/datasets/genex-world/GenEx-DB-Panorama-World) dataset. 

```python
python create_sample_dataset.py
```

This script:
1. Creates the necessary folder structure
2. Downloads the GenEx-DB-Panorama-World dataset
3. Uses the existing train and validation splits
4. Saves the panoramic images to the appropriate folders
5. Copies the existing pano_mask.png file



## 🚀 Training Your Own Model

To train your own GenEx-Initializer LoRA model:

```bash
# Set up environment variables
export MODEL_NAME="black-forest-labs/FLUX.1-Fill-dev"
export INSTANCE_DIR="./dataset/instance_images"
export OUTPUT_DIR="./output_directory"

# Launch training
accelerate launch --num_processes=4 --mixed_precision=bf16 \
  train_world_initializer.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --instance_data_dir="$INSTANCE_DIR" \
  --instance_prompt="Panoramic World Initialization" \
  --output_dir="$OUTPUT_DIR" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=3e-5 \
  --max_train_steps=10000 \
  --checkpointing_steps=500 \
  --validation_prompt="Panoramic World Initialization" \
  --seed=0 \
  --mixed_precision="bf16" \
  --report_to="wandb"
```

## 🎛️ Training Parameters

The training script offers numerous customizable parameters:

- **Base Model**: Uses FLUX.1-Fill as the foundation
- **Resolution**: Configurable image resolution (default: 512×512)
- **Batch Size & Gradient Accumulation**: Scale to your hardware capabilities 
- **Learning Rate**: Adjustable with warmup steps and scheduler options
- **LoRA Rank**: Control the complexity of adaptation (default: 4)
- **Validation**: Periodic validation with configurable frequency
- **Checkpointing**: Save progress and resume training as needed
- **WandB Integration**: Track experiments with detailed metrics

## 🔧 Requirements

```
diffusers>=0.32.0
transformers
torch>=2.0.0
accelerate
huggingface_hub
numpy
pillow
wandb
tqdm
accelerate
peft
```

## 📚 Model Architecture

GenEx-Initializer uses LoRA to efficiently adapt the FLUX.1-Fill model for panoramic content. The adaptation targets:

- Attention layers (query, key, value projections)
- Feedforward networks
- Context processing components

## 🙏 Acknowledgements

This project builds upon FLUX.1-Fill by Black Forest Labs and the Hugging Face Diffusers library. Special thanks to the developers of these foundational tools.
