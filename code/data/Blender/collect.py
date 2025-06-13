import bpy
import os
import sys
import math
from tqdm import tqdm
from load_scene import load_blend_file
from load_camera import initialize_camera, set_render, set_camera, render_video
from sample import get_valid_start_end_and_direction

# Global parameters
RENDERER_TYPE = 'CYCLES'  # Options: 'CYCLES' or 'EEVEE'
BASE_OUTPUT_PATH = "path/to/save/folder"  # Base output directory
ITERATIONS = 10000  # Number of path iterations to generate
STEP_SIZE = 12  # Distance between start and end points
BLEND_FILE_PATH = "path/to/your/BlendFile.blend"  # Path to your Blender scene

# Optional parameters for other scene loading methods
# HDRI_PATH = "path/to/hdri.exr"
# GLB_PATH = "path/to/model.glb" 
# GLTF_PATH = "path/to/model.gltf"
# FBX_PATH = "path/to/model.fbx"

def main():
    # Handle output directory suffix from command line arguments
    output_suffix = "" if len(sys.argv) <= 1 else sys.argv[1]
    output_folder = os.path.join(BASE_OUTPUT_PATH, f"CYCLES{output_suffix}")
    
    # Load scene
    load_blend_file(BLEND_FILE_PATH)
    
    # Alternative scene loading methods (commented out)
    # set_hdri_environment(HDRI_PATH)
    # load_gltf_file(GLTF_PATH)
    # load_glb_file(GLB_PATH)
    # load_fbx(FBX_PATH)
    # skyBackground()
    
    # Initialize cameras and render settings
    cameras = initialize_camera(RENDERER_TYPE)
    set_render(RENDERER_TYPE)
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    print(f'Rendering to {output_folder}')
    
    # Main iteration loop
    for i in tqdm(range(ITERATIONS)):
        # Create output path for this iteration
        iteration_output_path = os.path.join(output_folder, f'iteration{i}')
        
        # Skip if this iteration has already been processed
        if os.path.exists(iteration_output_path):
            print(f'{iteration_output_path} already exists, skipping.')
            continue
            
        # Generate a valid camera path
        start, end, direction, start_trials, direction_trials, angle = get_valid_start_end_and_direction(step_size=STEP_SIZE)
        
        if start and end and direction:
            # Log path information
            print(f"Start position: {start}, End position: {end}")
            print(f"Direction: {direction}")
            print(f"Angle: {math.degrees(angle)} degrees")
            print(f"Trials to find start location: {start_trials}")
            print(f"Trials to find direction: {direction_trials}")
            
            # Adjust camera rotation based on direction angle
            rotation = (0, 0, math.degrees(angle) - 90)
            
            # Set up camera animation
            set_camera(cameras, start, end, rotation)
            
            # Create output directory for this iteration and render
            os.makedirs(iteration_output_path, exist_ok=True)
            render_video(RENDERER_TYPE, cameras, iteration_output_path)
        else:
            print("Failed to find valid camera path within the given attempts.")

if __name__ == "__main__":
    main()