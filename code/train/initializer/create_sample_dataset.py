import os
import datasets
import shutil
from PIL import Image
from multiprocessing import Pool
from functools import partial

def save_image(args):
    """Helper function to save a single image"""
    i, example, folder_path = args
    panorama = example["panorama"]
    panorama.save(f"{folder_path}/{i}.png")
    return f"Saved image {i}"

def save_images_parallel(dataset, folder_path, dataset_name, num_processes=8):
    """Save images using multiprocessing"""
    # Prepare arguments for each image
    args_list = [(i, example, folder_path) for i, example in enumerate(dataset)]
    
    # Use multiprocessing Pool to save images
    with Pool(num_processes) as pool:
        results = pool.map(save_image, args_list)
    
    print(f"Completed saving {len(results)} {dataset_name} images")

# Create the required folder structure
os.makedirs("dataset/instance_images", exist_ok=True)
os.makedirs("dataset/val", exist_ok=True)

# Download the dataset
dataset = datasets.load_dataset("genex-world/GenEx-DB-Panorama-World", num_proc=8)

# Use existing train and validation splits
train_dataset = dataset["train"]
val_dataset = dataset["validation"]

print(f"Starting to save {len(train_dataset)} training images...")
# Save training images with 8 processes
save_images_parallel(train_dataset, "dataset/instance_images", "training", num_processes=8)

print(f"Starting to save {len(val_dataset)} validation images...")
# Save validation images with 8 processes
save_images_parallel(val_dataset, "dataset/val", "validation", num_processes=8)

# Copy the existing pano_mask.png from this directory to the dataset folder
source_mask = "pano_mask.png"  # Replace with actual path
if os.path.exists(source_mask):
    shutil.copy(source_mask, "dataset/pano_mask.png")
    print("Copied pano_mask.png to dataset folder")
else:
    print("Warning: pano_mask.png not found in current directory")

print("Dataset preparation complete!")
