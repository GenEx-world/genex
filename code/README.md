# GenEx: World Generation & Exploration 🌍

GenEx is a complete pipeline for creating and exploring immersive virtual environments from single images. Through advanced machine learning techniques, GenEx transforms limited visual inputs into fully explorable 3D worlds.

![](../pics/zero_shot_generations.gif)


## Table of Contents

- [GenEx: World Generation \& Exploration 🌍](#genex-world-generation--exploration-)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Components](#components)
    - [GenEx World Initializer](#genex-world-initializer)
    - [GenEx World Explorer](#genex-world-explorer)
  - [Project Structure](#project-structure)
  - [Usage](#usage)
  - [Citation](#citation)

## Introduction

GenEx is a two-stage pipeline designed to create and navigate immersive virtual environments:

1. **World Initialization**: Transform a single perspective image into a complete 360° panoramic environment
2. **World Exploration**: Enable dynamic navigation through the generated panoramic world

Together, these components allow for the creation and exploration of fully realized virtual worlds from minimal input.

## Components

### GenEx World Initializer

The World Initializer converts a single-view image into a complete 360° panorama using vision-conditioned inpainting.

Key features:
- Input: One perspective image (any size, center-cropped to square)
- Optional text prompts to guide panorama generation
- Output: 2048 × 1024 equirectangular image

### GenEx World Explorer

The World Explorer enables dynamic navigation through panoramic environments using video generation techniques built on Stable Video Diffusion.

Key features:
- Generate smooth forward movements through scenes
- Rotate panoramic images programmatically
- Convert between different panoramic representations
- Extract normal rectilinear views from panoramas
- Save explorations as videos or GIFs

## Project Structure

The repository is organized into several key directories:

- **/inference/**: Code for running inference with pre-trained models (single GPU)
- **/data/**: Collection tools and datasets for training GenEx
- **/evaluate/**: Scripts for evaluating model performance using various metrics
- **/train/**: Scripts for training both World Initializer and Explorer components
- **/demo/**: Demo applications and examples showcasing GenEx capabilities (8 GPU server)

## Usage

For implementation examples and usage instructions, please refer to the [inference](/inference) directory.

## Citation

```bibtex
@misc{lu2025genexgeneratingexplorableworld,
      title={GenEx: Generating an Explorable World}, 
      author={Taiming Lu and Tianmin Shu and Junfei Xiao and Luoxin Ye and Jiahao Wang and Cheng Peng and Chen Wei and Daniel Khashabi and Rama Chellappa and Alan Yuille and Jieneng Chen},
      year={2025},
      eprint={2412.09624},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.09624}, 
}
```