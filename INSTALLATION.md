# Installation

This document contains the full environment setup for GN-Bench.

## 1. Create Environment

We recommend using Miniconda/Anaconda.

```bash
conda create -n bae python=3.11 -y
conda activate bae

git clone --recurse-submodules https://github.com/TeleHuman/GN0.git
cd GN0
```

## 2. Install PyTorch + CUDA Toolkit

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
conda install -c "nvidia/label/cuda-12.8.0" -c nvidia -c conda-forge cuda-toolkit=12.8 -y
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

## 5. Build Submodule Extensions

```bash
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11

pip install ./submodules/diff-gaussian-rasterization --no-build-isolation
pip install ./submodules/fused-ssim --no-build-isolation
pip install ./submodules/simple-knn --no-build-isolation
```