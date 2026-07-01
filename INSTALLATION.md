# Installation

This document contains the full environment setup for GN-Bench.

## 1. Create Environment

We recommend using Miniconda/Anaconda.

```bash
conda create -n bae python=3.10 -y
conda activate bae

git clone https://github.com/TeleHuman/GN0.git
cd GN0
git submodule update --init GN-Bench-Tools
```

## 2. Install PyTorch + CUDA Toolkit

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
conda install -c "nvidia/label/cuda-12.4.0" -c nvidia -c conda-forge cuda-toolkit=12.4 -y
```

## 3. Install GN-Bench-Tools

```bash
cd GN-Bench-Tools
pip install -e .
cd ..
```

## 4. Install Main Python Dependencies + BAE

```bash
pip install -r requirements.txt
pip install -e ./bae
```

## 5. CUDA Rendering Backend

GN0 eval uses `gsplat` for both perspective RGB/depth rendering and BEV rendering.
Install the `gsplat` wheel that matches your PyTorch and CUDA versions. For the
recommended Torch 2.4 + CUDA 12.4 environment:

```bash
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu124
```

If no prebuilt wheel matches your local PyTorch/CUDA setup, build `gsplat` from
source following the official `gsplat` instructions. No additional CUDA submodule
extensions are required.
