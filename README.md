# Anti-VEGF Therapy for Neovascular Age-Related Macular Degeneration

## Overview

- `data/` — dataset (OCT images + labels)
- `checkpoint/` — pretrained model weights
- `trainer/` — training and testing functions
- `Model/` — model architecture (Generator, Discriminator)
- `Yaml/` — configuration options
- `train.py` — main entry point for training/testing
- `evaluate_all.py` — unified evaluation script (ROC, PR, boxplot, Youden, SHAP, heatmap)

## Requirements

### Setting up the environment

```bash
# 1. Create conda environment
conda create -n anti-vegf-main python=3.9 -y
conda activate anti-vegf-main

# 2. Install PyTorch (CPU version)
pip install torch==1.12.1 torchvision==0.13.1 --index-url https://download.pytorch.org/whl/cpu

# 3. Install other dependencies
pip install -r requirements.txt
```

Or use the provided `environment.yaml`:

```bash
conda env create -f environment.yaml
conda activate anti-vegf-main
```

### Pretrained Models

Download pretrained models from [here](https://drive.google.com/drive/folders/1xeDL2PpFphCa_kmmyc2Osdoqxx8YbyU-?usp=sharing) and put them in `checkpoint/`.

### Data preparation

Download data from [here](https://drive.google.com/drive/folders/1LcpJJKrzsnMEKD5PMwC9Rk3WSLMHhM_D?usp=sharing) and put them in `data/`.

## Testing

### Quick test (original entry)

Check `Yaml/CycleGan.yaml` options and run:

```bash
python train.py
```

### Full evaluation

Run all metrics (confusion matrix, ROC, PR, boxplot, Youden, SHAP, heatmap, BCVA):

```bash
python evaluate_all.py
```

### Individual evaluation scripts

| Script | Output |
|--------|--------|
| `final_confusion_matrices.py` | Confusion matrices (3 periods) |
| `evaluate_all.py` | ROC, PR, boxplot, Youden, SHAP, heatmap |
| `generate_predicted_images.py` | Predicted/before/after OCT images |
| `compute_psnr.py` | PSNR values (3 periods) |
| `bcva_scatter.py` | BCVA scatter plot with R² and 95% CI |

### Output structure

```
output/
├── Short-term/
│   ├── ROC.png, PR_curve.png, boxplot_metrics.png
│   ├── shap.png, heatmap_democam.jpg
│   └── pred/, before/, after/
├── Mid-term/
├── Long-term/
└── Regression/
    ├── figure.png, BCVA_scatter.png
```

## Results

| Metric | Short-term | Mid-term | Long-term |
|--------|-----------|----------|-----------|
| Accuracy | 0.860 | 0.921 | 0.941 |
| SSIM | 0.648 | 0.617 | 0.571 |
| PSNR | 19.06 | 16.73 | 16.48 |

BCVA Regression: MAE = 0.0579
EOF
```