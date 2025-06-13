# GenEX Data Collection Tools

These folders contains three different methods for collecting training data for GenEX.

## Overview

All three implementations share the same core concept: they sample valid camera paths through 3D environments and capture panoramic or multi-view image sequences along these paths.

## Available Implementations

- **Blender**: Python-based automation for Blender 4.0+ with support for both Cycles and EEVEE renderers
- **Unity**: C# implementation for Unity engine with cubemap capture capabilities
- **Unreal**: Blueprint implementation for Unreal Engine with high-quality rendering options

Each folder contains its own README with specific setup instructions and configuration parameters.

## Common Features

- Random path sampling with obstacle detection
- Configurable capture resolution and quality
- Resumable processing for large dataset generation
- Consistent output format for training GenEX

Choose the implementation that best matches your existing workflow or rendering requirements.

## Perspective to panorama

Due to different renderer property, some engines require post-processing to combine perspective images into panorama. This can be done using the explorer class.

For example:
```python
from explorer import Explorer

folder_path = "path/to/input/perspectives"
save_folder = "path/to/resulting/panoramas"

explorer = Explorer()

for i in tqdm(range(len(os.listdir(folder_path))), desc=f"Processing folder {folder_name}"):
  try:
      # Load cubemap face images and convert to RGB
      front = Image.open(os.path.join(folder_path, f"{i}_front.png")).convert('RGB')
      back = Image.open(os.path.join(folder_path, f"{i}_back.png")).convert('RGB')
      top = Image.open(os.path.join(folder_path, f"{i}_top.png")).convert('RGB')
      bottom = Image.open(os.path.join(folder_path, f"{i}_bottom.png")).convert('RGB')
      left = Image.open(os.path.join(folder_path, f"{i}_left.png")).convert('RGB')
      right = Image.open(os.path.join(folder_path, f"{i}_right.png")).convert('RGB')
  
      # Create a dictionary of cube faces
      cubes = {
          'front': front,
          'back': back,
          'top': top,
          'bottom': bottom,
          'left': left,
          'right': right
      }
  
      # Convert cubemap to equirectangular panorama
      pano = explorer.cubemap_to_equirectangular(
          cubemap_faces=cubes,
          output_height=576,
          output_width=1024,
          scale_factor=1
      )
  
      # Save the panorama image
      save_path = os.path.join(save_folder, f"{i}.png")
      pano.save(save_path)
```
