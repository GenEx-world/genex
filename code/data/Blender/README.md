# GenEx Blender Data Curation Tool 📹

## Overview
The GenEx Blender Data Curation Tool is a Python-based automation system for Blender designed to generate high-quality panoramic image datasets. The tool automatically navigates virtual cameras through 3D environments, capturing images along randomly sampled linear paths. It's ideal for creating training data for computer vision and machine learning projects that require panoramic view sequences.

## Features
- Automated camera movement along randomly sampled valid paths
- Support for both panoramic (Cycles) and multi-view (EEVEE) rendering
- Intelligent path validation using raycasting for obstacle detection
- Configurable rendering settings for both Cycles and EEVEE engines
- Automatic file organization by iteration number
- Progress tracking and render time reporting
- Resumable processing (skips completed iterations)

## Requirements
- Blender 4.0+
- Python libraries: tqdm (for progress bars)
- A 3D environment/scene to capture (as .blend file or importable format)
- Sufficient disk space for rendered images

## Setup
1. Download the script files to a local directory
2. Open Blender or create a new Blender project
3. Load your 3D environment/scene
4. Configure the parameters in each script file (see Configuration section)
5. Run the script from Blender's Python console or as a command-line argument:
   ```
   blender --background --python collect.py [optional_output_suffix]
   ```

## Configuration

Parameters are spread across multiple files. Here's how to modify them:

### Main Configuration (collect.py)
```python
# Global parameters
RENDERER_TYPE = 'CYCLES'  # Options: 'CYCLES' or 'EEVEE'
BASE_OUTPUT_PATH = "path/to/save/folder"  # Base output directory
ITERATIONS = 10000  # Number of path iterations to generate
STEP_SIZE = 12  # Distance between start and end points
BLEND_FILE_PATH = "path/to/your/BlendFile.blend"  # Path to your Blender scene
```

### Path Sampling Configuration (sample.py)
```python
# Global parameters
MAX_START_ATTEMPTS = 100  # Maximum attempts to find a valid start location
MAX_DIRECTION_ATTEMPTS = 10  # Maximum attempts to find a valid direction
DEFAULT_STEP_SIZE = 20  # Default distance for path length
LOCATION_BOUNDS = {  # Space boundaries for sampling locations
    'min_x': -3.51543,
    'max_x': 3.3263,
    'min_y': -3.61569,
    'max_y': 3.40674,
    'z': 1.2  # Fixed height
}
RAY_MAX_DISTANCE = 10.0  # Maximum distance for upward ray casting
```

### Camera Configuration (load_camera.py)
```python
# Global parameters
CAMERA_COLLECTION_NAME = "cameras"
DEFAULT_FRAME_START = 1
DEFAULT_FRAME_END = 30
CAMERA_POSITIONS = ['Front', 'Left', 'Back', 'Right', 'Bottom', 'Top']
CAMERA_ROTATIONS = {
    'Front': (90, 0, 0),
    'Left': (90, 0, 90),
    'Back': (90, 0, 180),
    'Right': (90, 0, 270),
    'Bottom': (0, 0, 0),
    'Top': (180, 0, 0),
}
```

## Output Structure

### For CYCLES renderer:
```
output_folder/
├── iteration0/
│   ├── video_frame0.png
│   ├── video_frame1.png
│   └── ...
├── iteration1/
│   ├── video_frame0.png
│   └── ...
└── ...
```

### For EEVEE renderer:
```
output_folder/
├── iteration0/
│   ├── frame1/
│   │   ├── front.png
│   │   ├── back.png
│   │   ├── left.png
│   │   ├── right.png
│   │   ├── bottom.png
│   │   └── top.png
│   ├── frame2/
│   │   └── ...
│   └── ...
├── iteration1/
│   └── ...
└── ...
```

## How It Works

1. **Path Generation**:
   - Randomly samples a starting position within defined boundaries
   - Checks if the position is valid (no obstacles above)
   - Finds a valid direction by casting rays to detect obstacles
   - Calculates an end position based on the direction and step size

2. **Camera Setup**:
   - Creates cameras appropriate for the selected renderer:
     - For EEVEE: Six perspective cameras (Front, Back, Left, Right, Top, Bottom)
     - For CYCLES: One equirectangular panoramic camera
   - Sets render parameters optimized for the selected engine
   - Creates animation keyframes from start to end position

3. **Rendering Process**:
   - For each iteration:
     - Checks if the iteration has already been rendered (allows resuming)
     - Generates a new random path
     - Animates the camera(s) along the path
     - Renders frames according to the configured settings
     - Saves images to the specified output directory structure
   - Reports render time

## Script Components

| File | Description |
|------|-------------|
| `collect.py` | Main script with overall workflow and parameters |
| `sample.py` | Path sampling and validation functionality |
| `load_camera.py` | Camera creation and animation setup |
| `load_scene.py` | Scene loading and environment setup utilities |

## Tips
- For better performance with Cycles, reduce the sample count in the render settings
- Increase `MAX_START_ATTEMPTS` and `MAX_DIRECTION_ATTEMPTS` in complex scenes
- Use the optional command-line suffix to create separate output directories for different runs
- Adjust the `LOCATION_BOUNDS` in sample.py to match your specific scene dimensions