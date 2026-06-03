<div align="center">

<img src="assets/GN0.gif" alt="GN0 teaser" width="92%">

<p>
  <a href="https://telehuman-gn0.github.io/"><img src="assets/badges/project.svg" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2606.03682"><img src="assets/badges/paper.svg" alt="GN0 Paper - arXiv"></a>
</p>

</div>

## 🏠 Introduction

GN0 is a unified research framework for **Generation**, **Evaluation**, and **Policy Learning** in Vision-and-Language Navigation (VLN). Built upon 3D Gaussian Splatting (3DGS), GN0 bridges realistic scene construction, high-fidelity embodied simulation, and navigation policy evaluation in visually grounded indoor environments.

This repository hosts the **GN-Bench** evaluation workflow. The current release focuses on the InteriorGS setting and provides a compact, reproducible pipeline for evaluating BAE-based navigation agents.

### Highlights

- **3DGS-native navigation benchmark.** GN-Bench evaluates agents directly in high-fidelity 3D Gaussian Splatting scenes.
- **Unified GN0 ecosystem.** The repository connects GN-Matrix data, GN-Bench simulation, and GN-BAE policy evaluation.
- **InteriorGS evaluation workflow.** A cleaned entry point is provided for instruction-following evaluation on InteriorGS scenes.
- **Scalable episode splitting.** Multi-GPU and multi-process evaluation are supported through configurable chunks.
- **Lightweight metric analysis.** Evaluation logs can be summarized into TL, NE, OS, SR, and SPL with a single script.

## 🔥 News

| Time | Update |
| --- | --- |
| 2026/06 | GN-Bench InteriorGS evaluation workflow  released |

## 📋 Table of Contents

- [🏠 Introduction](#-introduction)
- [🔥 News](#-news)
- [📦 Overview](#-overview)
- [📚 Getting Started](#-getting-started)
- [🧪 Evaluation](#-evaluation)
- [🔗 Citation](#-citation)
- [👏 Acknowledgements](#-acknowledgements)

## 📦 Overview

### 🧩 GN0 Components

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td><b>GN-Matrix</b></td>
      <td><b>GN-Bench</b></td>
      <td><b>GN-BAE</b></td>
    </tr>
    <tr valign="top">
      <td>Large-scale 3DGS navigation data with dynamic human avatars.</td>
      <td>Interactive benchmark and simulator for high-fidelity VLN evaluation.</td>
      <td>Navigation foundation model for map-based and map-free policy learning.</td>
    </tr>
  </tbody>
</table>

### 🤗 Model Zoo & Downloads

- [BAE checkpoint](https://huggingface.co/TeleEmbodied/GN-BAE)
- [InteriorGS dataset](https://huggingface.co/datasets/spatialverse/InteriorGS)
- InteriorGS episodes are coming soon

## 📚 Getting Started

Please refer to [INSTALLATION.md](INSTALLATION.md) for the complete environment setup, including PyTorch, CUDA extensions, GN-Bench-Tools, and BAE installation.

After installation, prepare datasets and checkpoints with the following layout:

```text
GN0
├── data
│   ├── datasets
│   │   └── GN_Matrix
│   │       └── InteriorGS_episode
│   └── scene_datasets
│       └── InteriorGS
├── GN-Bench-Tools
├── model_zoo
│   └── bae
├── VLN_CE
├── run.py
└── eval_bae_InteriorGS.sh
```

Run the InteriorGS evaluation:

```bash
zsh eval_bae_InteriorGS.sh \
  --model-path model_zoo/bae \
  --chunks 1 \
  --procs-per-gpu 1 \
  --save-path tmp/bae_eval
```

Monitor evaluation progress:

```bash
watch -n 1 python analyze_results.py --path tmp/bae_eval
```

Terminate active evaluation workers if needed:

```bash
bash kill_bae_eval.sh
```

## 🧪 Evaluation

### 📊 Metrics

`analyze_results.py` reads JSON logs under the selected result directory and reports standard VLN metrics:

| Metric | Meaning |
| --- | --- |
| TL | Average trajectory length |
| NE ↓ | Navigation error |
| OS ↑ | Oracle success |
| SR ↑ | Success rate |
| SPL ↑ | Success weighted by path length |


## 🔗 Citation

If GN-Bench is useful for your research, please cite our paper:

```bibtex
@article{li2026gn0,
  title={GN0: Toward a Unified Paradigm for Generation, Evaluation, and Policy Learning in Visual-Language Navigation},
  author={Li, Xinhai and Zhang, Xiaotao and Huang, Yuehao and Dong, Jiankun and Wang, Tianhang and Zhou, Sunyao and Wu, Yunzi and Sun, Chengnuo and Ge, Yunfei and Weng, Qizhen and Zhang, Chi and Bai, Chenjia and Li, Xuelong},
  journal={arXiv preprint arXiv:2606.03682},
  year={2026}
}
```

## 👏 Acknowledgements

GN-Bench-Tools is adapted from Habitat-Lab and customized for 3D Gaussian Splatting-based navigation. We sincerely thank:
- The Habitat-Lab developers for their foundational simulation framework.
- The InteriorGS authors for releasing their high-quality open-source dataset.
- The broader Embodied AI and 3DGS open-source communities for continuously advancing the field and making this infrastructure a reality.