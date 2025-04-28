# Simple Diffusion Model

This repository contains a PyTorch implementation of a simple diffusion model for image generation using the CIFAR-10 dataset. The model uses a UNet architecture as the noise prediction network in a diffusion process.

## Overview

Diffusion models have gained significant attention in the field of generative AI for their ability to produce high-quality samples. This implementation provides a simplified version of diffusion models with the following components:

- **SimpleUNet**: A simplified UNet architecture for noise prediction
- **DiffusionModel**: Handles the forward and reverse diffusion processes
- **SimpleDiffusion**: Combines the UNet and diffusion processes into a single model

## Features

- Implements the complete diffusion process with forward and reverse steps
- Uses MPS (Metal Performance Shaders) for Mac users, with fallbacks to CUDA or CPU
- Includes visualization of generated samples during training
- Simple and easy-to-understand implementation of diffusion concepts

## Requirements

- Python 3.6+
- PyTorch 1.12+
- torchvision
- numpy
- matplotlib
- tqdm
- PIL

## Installation

Clone this repository:

```bash
git clone https://github.com/bemoregt/SimpleDiffusionModel.git
cd SimpleDiffusionModel
```

Install the required packages:

```bash
pip install torch torchvision numpy matplotlib tqdm pillow
```

## Usage

Run the training script:

```bash
python diffusion_model.py
```

This will:
1. Download the CIFAR-10 dataset if not already available
2. Initialize the diffusion model
3. Train the model for 100 epochs
4. Generate sample images every 5 epochs
5. Save the final model as `diffusion_model.pth`

## Model Architecture

### UNet Model

The UNet architecture consists of:
- Down-sampling path with two blocks
- Bottleneck layer
- Up-sampling path with skip connections
- Time embedding to condition the model on the diffusion timestep

### Diffusion Process

The diffusion process follows the standard approach:
1. **Forward diffusion**: Gradually adds noise to images according to a fixed schedule
2. **Training**: The model learns to predict the noise at each step
3. **Sampling**: Starting from random noise, iteratively denoise to generate new images

## Results

Generated samples will be saved in the `samples` directory during training. Each image contains 4 samples arranged in a 2x2 grid.

## Performance Notes

- The model will automatically use MPS acceleration on compatible Mac systems
- For better performance on larger datasets, consider increasing the model capacity by adjusting `hidden_channels`
- Training time varies depending on hardware; expect several hours on CPU and less on GPU/MPS

## References

This implementation is inspired by:
- ["Denoising Diffusion Probabilistic Models" by Ho et al.](https://arxiv.org/abs/2006.11239)
- ["Understanding Diffusion Models: A Unified Perspective" by Yang and Hu](https://arxiv.org/abs/2208.11970)

## License

This project is open-source and available under the MIT License.

---

Created by [@bemoregt](https://github.com/bemoregt)
