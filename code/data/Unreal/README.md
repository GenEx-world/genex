# GenEx Unreal Data Curation Tool 📹

## Overview
GenEx Data Curation Tool in Unreal Engine 5 is for automatically capture panoramic image datasets along a linear path. The tool generates high-quality six-cubes image sequences by moving a virtual camera through a 3D environment and capturing images in all six directions (front, back, left, right, top, bottom) at regular intervals.

## Features
- Automated camera movement along a randomly sampled linear path
- Captures complete 360° panoramic views (6 faces) at configurable intervals
- Multiple iterations with different random paths
- Collision detection to ensure clear paths
- Configurable output resolution and directory structure

## Requirements
- Unreal Engine 5.0 or newer
- [Victory Plugin For UE5](https://forums.unrealengine.com/t/ramas-extra-blueprint-nodes-for-ue5-no-c-required/231476)
- A 3D environment/scene to capture (e.g., the [City Sample](https://www.fab.com/listings/4898e707-7855-404b-af0e-a505ee690e68))
- It is recommended to install the project on a solid state drive of 200GB+, on machine with at least Windows 10 with support for DirectX 12, 12-core CPU at 3.4 GHz, 64 GB of System RAM, GeForce RTX-2080 / AMD Radeon 6000 or higher, at least 8 GB of VRAM.

## Installation
1. Place this folder inside the editor content directory.
2. Download and install the Victory Plugin from the URL provided above
3. Enable the Victory Plugin in your project settings
4. Restart the Unreal Editor

## Quick Start Guide
1. Open your Unreal Engine 5 level (note: the blueprint does not currently support HLOD)
2. From the collector folder, place the `CollectorPawn` into your scene or set it as the default pawn
3. Configure the parameters in the blueprint according to your requirements
4. Set up your CSV path files (see CSV Format section below)
5. Start the game in the editor to begin the data collection process

## Configuration Parameters

| Parameter | Description | Recommended Value |
|-----------|-------------|------------------|
| `SavePath` | Folder to save the generated images (include trailing `/`) | `D:/GenEx/CapturedData/` |
| `CSVPath` | Path to the CSV path folder (include trailing `/`) | `D:/GenEx/Paths/` |
| `Min/Max XYZ` | The boundaries of the exploration area | Based on your scene dimensions |

## Output Structure
```
SavePath/
├── 0/                  # First iteration
│   ├── 0_front.png     # Frame 0, front view
│   ├── 0_back.png      # Frame 0, back view
│   ├── 0_left.png      # Frame 0, left view
│   ├── 0_right.png     # Frame 0, right view
│   ├── 0_top.png       # Frame 0, top view
│   ├── 0_bottom.png    # Frame 0, bottom view
│   ├── 1_front.png     # Frame 1, front view
│   └── ...
├── 1/                  # Second iteration
│   ├── 0_front.png
│   └── ...
└── ...
```

## CSV Path Format
Each CSV file defines a different camera path. The files contain pure numerical data with no headers.

### CSV Folder Structure
```
CSVPath/
├── 0.csv    # First path
├── 1.csv    # Second path
├── 2.csv    # Third path
└── ...
```

### CSV File Format
Each CSV file contains rows of numerical data representing:
- Column 1: Frame index (sequential number)
- Column 2: X position (location_x)
- Column 3: Y position (location_y)
- Column 4: Z position (location_z)
- Column 5: Pitch rotation
- Column 6: Yaw rotation
- Column 7: Roll rotation

Example CSV content (pure numerical data, no headers):
```
0,0,0,0,0,5.325,0
1,3.788,0,43.106,0,4.5767,0
2,7.29,0,89.181,0,4.0087,0
3,10.548,0,137.802,0,3.576,0
...
```

## How It Works
1. **Initialization**:
   - The tool loads a CSV file containing predefined path coordinates
   - Sets up the output directory structure

2. **Path Following**:
   - The camera is positioned at the first coordinate in the CSV
   - For each entry in the CSV file:
     - The camera moves to the specified position and rotation
     - Captures six images (one in each cardinal direction)
     - Saves images with appropriate naming conventions

3. **Multiple Iterations**:
   - After completing one path, the tool can load another CSV file
   - This process repeats for the specified number of iterations