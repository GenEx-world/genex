import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from timm import create_model
from PIL import Image
from tqdm import tqdm
import argparse

# Function to load the Inception-v4 model
def load_inception_v4_model():
    """Load the pre-trained Inception-v4 model."""
    model = create_model('inception_v4', pretrained=True, num_classes=0)  # Remove the final classification layer
    model.eval()
    print('Loaded Inception Model.')
    return model

# Function to preprocess images before passing them to the Inception-v4 model
def preprocess_image(image_path, transform):
    """Preprocess the image for Inception-v4."""
    img = Image.open(image_path).convert("RGB")  # Load image and ensure it's in RGB format
    tensor = transform(img).unsqueeze(0)  # Apply transformations and add batch dimension
    return tensor

# Function to extract latent features using Inception-v4 model
def extract_features(model, image_tensor):
    """Extract features using the Inception-v4 model."""
    with torch.no_grad():
        features = model(image_tensor)
    return features.cpu().numpy()

# Function to calculate latent MSE between two images
def calculate_latent_mse_between_images(image1_path, image2_path, model, transform):
    # Preprocess images and extract features
    image1_tensor = preprocess_image(image1_path, transform)
    image2_tensor = preprocess_image(image2_path, transform)

    latent_features1 = extract_features(model, image1_tensor)
    latent_features2 = extract_features(model, image2_tensor)

    mse = np.mean((latent_features1 - latent_features2) ** 2)
    return mse

# Function to calculate average latent MSE across all image pairs in the folder
def calculate_average_latent_mse_across_all_images(folder_path, model, transform):
    subfolders = [os.path.join(folder_path, subfolder) for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder))]

    total_latent_mse = 0
    image_pairs_count = 0
    
    for subfolder in tqdm(subfolders, desc="Processing images"):
        navigated_image_path = os.path.join(subfolder, "navigated.png")
        original_image_path = os.path.join(subfolder, "original.png")
        
        if os.path.exists(navigated_image_path) and os.path.exists(original_image_path):
            latent_mse = calculate_latent_mse_between_images(navigated_image_path, original_image_path, model, transform)
            total_latent_mse += latent_mse
            image_pairs_count += 1
    
    if image_pairs_count == 0:
        return 0
    return total_latent_mse / image_pairs_count

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Calculate average latent MSE across image pairs using Inception-v4.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing subfolders with image pairs.")
    
    args = parser.parse_args()
    
    # Load Inception-v4 model and define image transformation
    model = load_inception_v4_model()
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Inception-v4 expects 299x299 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Calculate average latent MSE
    average_latent_mse = calculate_average_latent_mse_across_all_images(args.folder_path, model, transform)
    print(f"Average Latent MSE across all image pairs: {average_latent_mse}")

if __name__ == "__main__":
    main()
