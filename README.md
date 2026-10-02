# TCA: Template-Conditioned Aggregation for Accurate Few-Shot Object Recognition

<p align="center">
  <a href="https://github.com/Nanxia666/TCA-FewShot-Recognition">
    <img src="https://img.shields.io/badge/GitHub-TCA--FewShot--Recognition-181717?logo=github" alt="GitHub">
  </a>
  <img src="https://img.shields.io/badge/ACCV-2026-blue" alt="ACCV 2026">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Backbone-SAM%20ViT--B%20%7C%20ViT--H-6C63FF" alt="SAM Backbones">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

<p align="center">
  <b>Official PyTorch implementation of TCA, accepted by ACCV 2026.</b>
</p>

<p align="center">
  Few-shot object localization and counting with a frozen SAM backbone, exemplar-conditioned local aggregation, and similarity-guided prediction.
</p>

---

## Overview

Few-shot object recognition aims to localize all instances of a target object using only one or a few annotated exemplars.

**TCA (Template-Conditioned Aggregation)** uses exemplar information not only as a matching target, but also as a condition for reorganizing image features before matching. The method contains:

- a **frozen SAM image encoder** for dense visual features;
- **Template Context Aggregation (TCA)** for exemplar-conditioned multi-scale local aggregation;
- **Similarity-Prior Calibration (SPC)** for lightweight dense prediction;
- center, offset, and scale prediction for object localization and counting.

The released code supports:

- **RPINE**
- **FSCD-147**
- **FSCD-LVIS** (Seen / Unseen)

and two frozen SAM backbones:

- **SAM ViT-B**
- **SAM ViT-H**

> This repository uses the **original Segment Anything Model (SAM / SAM v1)**, not SAM 2.

---

## Framework

<p align="center">
  <img src="assets/tca_framework.png" width="95%" alt="TCA framework">
</p>

The complete framework consists of:

1. frozen SAM feature extraction;
2. exemplar feature extraction with RoIAlign;
3. Template Context Aggregation;
4. center-node similarity matching;
5. Similarity-Prior Calibration;
6. dense center / offset / scale prediction.

---

## Highlights

- **Exemplar-conditioned aggregation.**  
  The exemplar controls the contribution of multiple local aggregation branches before similarity matching.

- **Multi-scale local context.**  
  TCA combines identity, depth-wise convolution, and cross-shaped convolution branches.

- **Channel recalibration.**  
  Exemplar information is used to adaptively select target-relevant feature channels.

- **Frozen SAM backbone.**  
  SAM is used only as a feature extractor; the image encoder is not fine-tuned.

- **Lightweight trainable head.**  
  Only the TCA/SPC-related prediction modules are optimized.

- **Detection and counting.**  
  The same predictions are used for localization metrics and instance-count evaluation.

---

## Repository Structure

```text
TCA-FewShot-Recognition/
├── assets/
│   └── tca_framework.png
│
├── scripts/
│   ├── cache/
│   │   ├── build_cache_fscd147_vitb.py
│   │   ├── build_cache_fscd147_vith.py
│   │   ├── build_cache_fscdlvis_vitb.py
│   │   ├── build_cache_fscdlvis_vith.py
│   │   ├── build_cache_rpine_vitb.py
│   │   └── build_cache_rpine_vith.py
│   │
│   ├── train/
│   │   ├── train_fscd147_vitb_64plus1.py
│   │   ├── train_fscd147_vith_64plus1.py
│   │   ├── train_fscdlvis_vitb_seen_64plus1.py
│   │   ├── train_fscdlvis_vitb_unseen_64plus1.py
│   │   ├── train_fscdlvis_vith_seen_64plus1.py
│   │   ├── train_fscdlvis_vith_unseen_64plus1.py
│   │   ├── train_rpine_vitb_64plus1.py
│   │   └── train_rpine_vith_64plus1.py
│   │
│   └── eval/
│       ├── infer64_fscd147_vitb.py
│       ├── infer64_fscd147_vith.py
│       ├── infer64_fscdlvis_vitb_seen.py
│       ├── infer64_fscdlvis_vitb_unseen.py
│       ├── infer64_fscdlvis_vith_seen.py
│       ├── infer64_fscdlvis_vith_unseen.py
│       ├── infer64_rpine_vitb.py
│       └── infer64_rpine_vith.py
│
├── model/
│   └── README.md
│
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Installation

Python 3.10 is recommended.

```bash
conda create -n tca python=3.10 -y
conda activate tca
```

Install a PyTorch build compatible with your CUDA environment, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

A typical environment contains:

```text
torch
torchvision
numpy
Pillow
tqdm
torchmetrics
pycocotools
segment-anything
```

You can check the main dependencies with:

```bash
python -c "import torch; import torchvision; import torchmetrics; import pycocotools; from segment_anything import sam_model_registry; print('Environment OK')"
```

---

## SAM Backbone Checkpoints

TCA uses the **original Segment Anything Model (SAM)** image encoder as a frozen backbone.

Two SAM variants are used in the released experiments:

| TCA variant | SAM model type | Official checkpoint |
|---|---|---|
| ViT-B | `vit_b` | `sam_vit_b_01ec64.pth` |
| ViT-H | `vit_h` | `sam_vit_h_4b8939.pth` |

**SAM ViT-L is not used in the released experiments.**

### Download ViT-B

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

or:

```bash
curl -L -o sam_vit_b_01ec64.pth \
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### Download ViT-H

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

or:

```bash
curl -L -o sam_vit_h_4b8939.pth \
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

The checkpoints can also be downloaded from the official Segment Anything repository:

https://github.com/facebookresearch/segment-anything

Place the downloaded SAM checkpoints in the repository root:

```text
TCA-FewShot-Recognition/
├── sam_vit_b_01ec64.pth
├── sam_vit_h_4b8939.pth
├── assets/
├── scripts/
├── model/
├── README.md
└── requirements.txt
```

You only need the checkpoint corresponding to the experiment you want to run:

- filenames containing `vitb` use `sam_vit_b_01ec64.pth`;
- filenames containing `vith` use `sam_vit_h_4b8939.pth`.

For example:

```text
scripts/cache/build_cache_fscd147_vitb.py
scripts/train/train_fscd147_vitb_64plus1.py
scripts/eval/infer64_fscd147_vitb.py
```

use **SAM ViT-B**, while:

```text
scripts/cache/build_cache_fscd147_vith.py
scripts/train/train_fscd147_vith_64plus1.py
scripts/eval/infer64_fscd147_vith.py
```

use **SAM ViT-H**.

---

## Why Build a SAM Feature Cache?

The SAM image encoder is frozen in TCA.

Instead of running the large SAM encoder repeatedly during every training and evaluation iteration, image features are extracted once and saved as `.pt` cache files.

```text
Dataset images
      ↓
Frozen SAM ViT-B / ViT-H
      ↓
SAM feature cache (.pt)
      ↓
TCA / SPC training
      ↓
Inference and evaluation
```

A typical cached sample contains information such as:

```python
{
    "feat": Tensor[256, 64, 64],
    "file_name": ...,
    "image_path": ...,
    "sam_model_type": "vit_b" or "vit_h",
    "sam_img_size": 1024,
    ...
}
```

After cache generation, TCA training and inference load the cached SAM features directly.

---

## Dataset Preparation

The released scripts support **RPINE**, **FSCD-147**, and **FSCD-LVIS**.

Dataset paths are configured near the beginning of the corresponding Python scripts. Before running a script, check and update the dataset-related variables to match your local environment.

### FSCD-147

A typical directory structure is:

```text
FSCD_147/
└── FSC147/
    ├── images_384_VarV2/
    └── annotations/
        └── Train_Test_Val_FSC_147.json
```

The cache-generation scripts expect the official FSCD-147/FSC-147 images and split annotations.

### FSCD-LVIS

Prepare the FSCD-LVIS dataset following its official directory and annotation structure, then update the paths at the beginning of the corresponding cache / train / evaluation scripts.

Separate scripts are provided for:

```text
Seen
Unseen
```

splits.

### RPINE

Prepare RPINE according to the dataset layout used by your local copy, then update the path configuration in the RPINE scripts.

> The released scripts intentionally keep dataset-specific configuration explicit to preserve the experimental setup.

---

## Quick Start

The recommended workflow is:

```text
1. Install the environment
2. Download the required SAM checkpoint
3. Prepare the dataset
4. Build the SAM feature cache
5. Train TCA or download a pretrained TCA checkpoint
6. Run inference / evaluation
```

### Example: FSCD-147 + SAM ViT-B

#### 1. Build SAM feature cache

```bash
python scripts/cache/build_cache_fscd147_vitb.py
```

#### 2. Train TCA

```bash
python scripts/train/train_fscd147_vitb_64plus1.py
```

#### 3. Evaluate

```bash
python scripts/eval/infer64_fscd147_vitb.py
```

For ViT-H, replace `vitb` with `vith` and use the ViT-H SAM checkpoint.

---

## Build SAM Feature Cache

### FSCD-147

```bash
python scripts/cache/build_cache_fscd147_vitb.py
python scripts/cache/build_cache_fscd147_vith.py
```

### FSCD-LVIS

```bash
python scripts/cache/build_cache_fscdlvis_vitb.py
python scripts/cache/build_cache_fscdlvis_vith.py
```

### RPINE

```bash
python scripts/cache/build_cache_rpine_vitb.py
python scripts/cache/build_cache_rpine_vith.py
```

Before running cache generation, check the configuration block near the beginning of the script, for example:

```python
CUDA_VISIBLE_DEVICES = r"0"

SAM_CHECKPOINT = r"sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"

DATA_ROOT = r"..."
CACHE_ROOT = r"..."
```

The cache root must match the path used later by the training and inference scripts.

---

## Pretrained TCA Models

Pretrained TCA checkpoints are available on Hugging Face:

https://huggingface.co/Nanxia666/TCA-FewShot-Recognition

The model directory used by the released scripts follows the structure:

```text
model/
├── fscd147_vitb/
│   └── best_model.pth
├── fscd147_vith/
│   └── best_model.pth
├── fscdlvis_seen_vitb/
│   └── best_model.pth
├── fscdlvis_unseen_vitb/
│   └── best_model.pth
├── fscdlvis_seen_vith/
│   └── best_model.pth
├── fscdlvis_unseen_vith/
│   └── best_model.pth
├── rpine_vitb/
│   └── best_model.pth
└── rpine_vith/
    └── best_model.pth
```

### SAM weights vs. TCA weights

These are two different types of checkpoint:

```text
SAM backbone checkpoint
    ├── sam_vit_b_01ec64.pth
    └── sam_vit_h_4b8939.pth
```

is used to **extract frozen image features**, while:

```text
TCA checkpoint
    └── model/.../best_model.pth
```

contains the **trained TCA/SPC parameters** used for evaluation.

---

## Training

### FSCD-147

```bash
python scripts/train/train_fscd147_vitb_64plus1.py
python scripts/train/train_fscd147_vith_64plus1.py
```

### FSCD-LVIS Seen

```bash
python scripts/train/train_fscdlvis_vitb_seen_64plus1.py
python scripts/train/train_fscdlvis_vith_seen_64plus1.py
```

### FSCD-LVIS Unseen

```bash
python scripts/train/train_fscdlvis_vitb_unseen_64plus1.py
python scripts/train/train_fscdlvis_vith_unseen_64plus1.py
```

### RPINE

```bash
python scripts/train/train_rpine_vitb_64plus1.py
python scripts/train/train_rpine_vith_64plus1.py
```

Before training, check the configuration block at the beginning of the selected script, especially:

```python
CUDA_VISIBLE_DEVICES = r"0"
sam_feature_cache_root = r"..."
save_dir = r"model/..."
batch_size = 4
lr = 2e-4
epochs = ...
```

The released scripts preserve dataset- and backbone-specific experimental settings instead of forcing every experiment into a single generic entry point.

Training typically produces:

```text
last_model.pth
best_model.pth
```

under the corresponding model directory.

---

## Inference and Evaluation

### FSCD-147

```bash
python scripts/eval/infer64_fscd147_vitb.py
python scripts/eval/infer64_fscd147_vith.py
```

### FSCD-LVIS Seen

```bash
python scripts/eval/infer64_fscdlvis_vitb_seen.py
python scripts/eval/infer64_fscdlvis_vith_seen.py
```

### FSCD-LVIS Unseen

```bash
python scripts/eval/infer64_fscdlvis_vitb_unseen.py
python scripts/eval/infer64_fscdlvis_vith_unseen.py
```

### RPINE

```bash
python scripts/eval/infer64_rpine_vitb.py
python scripts/eval/infer64_rpine_vith.py
```

Before evaluation, check:

```python
CHECKPOINT_PATH = r"model/.../best_model.pth"
OUTPUT_DIR = r"..."
```

and make sure the cache root matches the cache generated for the same dataset and SAM backbone.

The evaluation scripts report detection/localization metrics and counting errors where supported.

---

## Script Reference

| Dataset | Backbone | Cache | Train | Evaluation |
|---|---|---|---|---|
| FSCD-147 | ViT-B | `build_cache_fscd147_vitb.py` | `train_fscd147_vitb_64plus1.py` | `infer64_fscd147_vitb.py` |
| FSCD-147 | ViT-H | `build_cache_fscd147_vith.py` | `train_fscd147_vith_64plus1.py` | `infer64_fscd147_vith.py` |
| FSCD-LVIS Seen | ViT-B | `build_cache_fscdlvis_vitb.py` | `train_fscdlvis_vitb_seen_64plus1.py` | `infer64_fscdlvis_vitb_seen.py` |
| FSCD-LVIS Seen | ViT-H | `build_cache_fscdlvis_vith.py` | `train_fscdlvis_vith_seen_64plus1.py` | `infer64_fscdlvis_vith_seen.py` |
| FSCD-LVIS Unseen | ViT-B | `build_cache_fscdlvis_vitb.py` | `train_fscdlvis_vitb_unseen_64plus1.py` | `infer64_fscdlvis_vitb_unseen.py` |
| FSCD-LVIS Unseen | ViT-H | `build_cache_fscdlvis_vith.py` | `train_fscdlvis_vith_unseen_64plus1.py` | `infer64_fscdlvis_vith_unseen.py` |
| RPINE | ViT-B | `build_cache_rpine_vitb.py` | `train_rpine_vitb_64plus1.py` | `infer64_rpine_vitb.py` |
| RPINE | ViT-H | `build_cache_rpine_vith.py` | `train_rpine_vith_64plus1.py` | `infer64_rpine_vith.py` |

All paths in the table are relative to:

```text
scripts/cache/
scripts/train/
scripts/eval/
```

respectively.

---

## Evaluation Metrics

The experiments use standard instance-level localization and counting metrics, including:

- **AP**
- **AP50**
- **AP75**
- **MAE**
- **RMSE**

Depending on the benchmark, experiments are evaluated under **1-shot** and/or **3-shot** settings.

---

## Notes

- The SAM image encoder is **frozen**.
- SAM features are precomputed before TCA training.
- Do not mix ViT-B caches with ViT-H training or evaluation scripts.
- Do not mix checkpoints trained with different backbones or dataset splits.
- FSCD-LVIS Seen and Unseen use separate training / evaluation entry points.
- Paths are intentionally explicit in the scripts for reproducibility.
- Large SAM checkpoints, feature caches, datasets, and generated outputs should not be committed to GitHub.

---

## Model Weights

Pretrained checkpoints:

**Hugging Face:**  
https://huggingface.co/Nanxia666/TCA-FewShot-Recognition

Repository:

**GitHub:**  
https://github.com/Nanxia666/TCA-FewShot-Recognition

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zhan2026tca,
  title     = {TCA: Template-Conditioned Aggregation for Accurate Few-Shot Object Recognition},
  author    = {Zhan, Lingtao and Zhang, Teng and Wang, Yeliang},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year      = {2026}
}
```

Please update the BibTeX entry when the official ACCV proceedings metadata becomes available.

---

## Acknowledgements

This project builds on the original **Segment Anything Model (SAM)** for frozen visual feature extraction.

We thank the authors and maintainers of SAM and the benchmark datasets used in this work.

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.
