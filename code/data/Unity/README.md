# GenEx Unity Data Curation Tool 📹

## Overview
GenEx Data Curation Tool is a Unity-based application designed to automatically capture panoramic image datasets along a linear path. The tool generates high-quality panoramic image sequences by moving a virtual camera through a 3D environment and capturing images in all six directions (front, back, left, right, top, bottom) at regular intervals.

## Features
- Automated camera movement along a randomly sampled linear path
- Captures complete 360° panoramic views (6 faces) at configurable intervals
- Multiple iterations with different random paths
- Collision detection to ensure clear paths
- Configurable output resolution and directory structure
- Memory management optimizations for large dataset generation

## Requirements
- Unity (tested with Unity 2019.4+)
- A 3D environment/scene to capture

## Setup
1. Create a new Unity project or open an existing one
2. Add your 3D environment/models to the scene
3. Create an empty GameObject and attach the `CameraMover.cs` script to it
4. Ensure the GameObject has a Camera component (or assign the Main Camera as a child)
5. Configure the parameters in the Inspector

## Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| `distance` | Total distance the camera will move along the path |
| `totalFrames` | Number of positions where images will be captured |
| `outputWidth` | Width of the output images in pixels |
| `outputHeight` | Height of the output images in pixels |
| `outputDirectory` | Directory where the captured images will be saved |
| `minX, maxX, minY, maxY, minZ, maxZ` | Boundaries for random position sampling |
| `maxSamplingAttempts` | Maximum attempts to find a valid path before skipping an iteration |
| `iterations` | Total number of different paths to generate |

## Output Structure
```
outputDirectory/
├── Iteration_0/
│   ├── 0_front.png
│   ├── 0_back.png
│   ├── 0_left.png
│   ├── 0_right.png
│   ├── 0_top.png
│   ├── 0_bottom.png
│   ├── 1_front.png
│   └── ...
├── Iteration_1/
│   ├── 0_front.png
│   └── ...
└── ...
```

## How It Works
1. For each iteration:
   - Samples a random starting position and orientation within defined boundaries
   - Checks if the path is clear of obstacles
   - Moves the camera along the path in incremental steps
   - At each step, captures images in all six directions
   - Saves the images with appropriate naming conventions
   - Performs memory cleanup

2. The script handles:
   - Path validation using raycasting
   - Proper camera orientation for each face
   - Memory management with garbage collection
   - File organization by iteration and frame
