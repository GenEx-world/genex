from explorer import *
from sample_path import *
from PIL import Image
from tqdm import tqdm
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse command-line arguments for navigational consistency evaluation.")

    parser.add_argument(
        "--unet_path",
        type=str,
        required=True,
        help="Path to the UNet model."
    )

    parser.add_argument(
        "--svd_path",
        type=str,
        default="stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        help="Path to the SVD model (default: stabilityai/stable-video-diffusion-img2vid-xt)."
    )

    parser.add_argument(
        "--navigation_distances",
        type=int,
        nargs='+',
        required=True,
        help="List of navigation distances as integers."
    )

    parser.add_argument(
        "--rotation_times",
        type=int,
        nargs='+',
        required=True,
        help="List of rotation times as integers."
    )

    parser.add_argument(
        "--num_data",
        type=int,
        required=True,
        help="Number of starter images for evaluation."
    )

    parser.add_argument(
        "--save_frames",
        action='store_true',
        help="Flag to save frames (default: False)."
    )

    return parser.parse_args()

# Parse arguments
args = parse_arguments()

# Access parsed arguments
print("UNet Path:", args.unet_path)
print("SVD Path:", args.svd_path)
print("Navigation Distances:", args.navigation_distances)
print("Rotation Times:", args.rotation_times)
print("Number of Data Points:", args.num_data)
print("Save Frames:", args.save_frames)

explorer = Explorer()
explorer.get_pipeline(args.unet_path, args.svd_path, model_height=576, progress_bar=False)

save_dir = 'consistency'
data_dir = 'validation'
os.makedirs(save_dir, exist_ok = True) 

val_images = os.listdir(data_dir)
val_images = val_images[:args.num_data]

for n in args.navigation_distances:
    for k in args.rotation_times:
        setup_dir = os.path.join(save_dir, f'n_{n}_k_{k}')
        os.makedirs(setup_dir, exist_ok = True) 
        for val_image in tqdm(val_images):
            image_path = os.path.join(data_dir, val_image)
            image_name = os.path.splitext(val_image)[0]
            start_image = Image.open(image_path).convert('RGB')

            save_path = os.path.join(setup_dir, image_name)
            os.makedirs(save_path, exist_ok = True) 

            path = construct_path(n, k, max_error=1e-5)
            draw_path(path, save_path=os.path.join(save_path, f'path.png'))

            generations = explorer.navigate_path(path, start_image, num_inference_steps=50)
            navigation = [intermediate_image for movement in generations for intermediate_image in movement]

            if args.save_frames:
                frames_path = os.path.join(save_path, 'movements')
                os.makedirs(frames_path, exist_ok = True) 
                for i, frame in enumerate(navigation):
                    frame.save(os.path.join(frames_path, f'frame{i}.png'))

            navigation[0].save(os.path.join(save_path, f'original.png'))
            navigation[-1].save(os.path.join(save_path, f'navigatied.png'))

            explorer.save_video(os.path.join(save_path, f'movement.mp4'), fps=10)