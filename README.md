# ShenSi-CFD

Physics-Informed Neural Network for Computational Fluid Dynamics (CFD) wind field simulation over complex terrain.

## Overview

ShenSi-CFD is a deep learning framework that predicts high-resolution 3D wind fields (u, v, w, k) over complex terrain using a Swin-UNet Lite architecture with physics-informed losses. It is trained on the [FuXi-CFD dataset](https://huggingface.co/datasets/linchensen/FuXi-CFD-dataset).

### Key Features

- **6-Channel Input**: u_100m, v_100m, DEM, roughness, DEM gradient (∂z/∂x, ∂z/∂y)
- **Physics-Informed Losses**:
  - Mass conservation (∂u/∂x + ∂v/∂y + ∂w/∂z = 0)
  - Normal boundary condition (u·n = 0) via DEM gradients
  - Boundary layer similarity theory (log-law profile)
  - TKE physical constraints (k ~ slope × velocity gradient²)
- **Lightweight Architecture**: ~5-31M parameters (configurable)
- **Multi-GPU Training**: DataParallel support with AMP, EMA, and gradient clipping
- **Progressive Loss Scheduling**: 3-stage curriculum from data fidelity to full physics

## Installation

```bash
# Clone repository
git clone https://github.com/sorryhorizon/shensicfd.git
cd shensicfd

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy scipy tensorboard
```

## Dataset

The project uses the FuXi-CFD dataset:
- **Source**: [Hugging Face - linchensen/FuXi-CFD-dataset](https://huggingface.co/datasets/linchensen/FuXi-CFD-dataset)
- **Location**: Download and extract to `/mnt/sdata/jz/fuxi_cfd/dataset`
- **Format**:
  - `inputs.npz`: dem(300×300), roughness(300×300), u_100m(9×9), v_100m(9×9)
  - `outputs.npz`: u(27×300×300), v(27×300×300), w(27×300×300), k(27×300×300)
- **Vertical Levels**: 27 terrain-following levels from 5m to ~214m above ground

## Training

```bash
# Multi-GPU training (default: GPUs 2,3,4,5)
python train_v3.py
```

### Training Configuration

Edit `train_v3.py` to adjust:
- `gpu_ids`: GPU devices to use
- `batch_size_per_gpu`: Batch size per GPU
- `accum_steps`: Gradient accumulation steps
- `epochs`: Total training epochs
- `lr`: Learning rate
- `warmup_epochs`: Warmup phase epochs (uses SmoothL1Loss)

### Monitoring

```bash
# TensorBoard
tensorboard --logdir=logs/train_v3/tensorboard
```

## Model Architecture

```
Input (6, 300, 300)
  ├── Wind Encoder (2ch → base_dim)
  ├── Terrain Encoder (3ch: DEM + gradients → base_dim)
  └── Roughness Encoder (1ch → base_dim)
        └── Fusion Conv → U-Net Backbone
              ├── Encoder Stages (×4 downsampling)
              ├── Swin Transformer Bottleneck
              ├── Cross-Attention (wind ↔ terrain/roughness)
              └── Decoder Stages (×4 upsampling)
                    └── Adaptive Vertical Decoder (27 levels)
                          └── Physics Constraint Layer
```

## Project Structure

```
.
├── src/
│   ├── data/
│   │   └── fuxi_cfd_dataset.py      # Dataset loader with 6-channel input
│   ├── models/
│   │   ├── swin_unet_lite.py        # Main model architecture
│   │   ├── swin_transformer.py      # Swin Transformer blocks
│   │   └── encoder.py               # Encoders & PhysicsConstraintLayer
│   └── losses/
│       └── enhanced_physics_loss.py # Physics-informed loss function
├── train_v3.py                       # Training script
├── README.md
└── .gitignore
```

## License

MIT License
