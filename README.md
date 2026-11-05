# DDPF-Net

Dual-Decoder Point-supervised Framework for crowd counting and head
localization. The model uses a Swin-Tiny backbone and produces both a
density map and a localization heatmap.

The project is designed for the ShanghaiTech Crowd Counting Dataset
(Part A or Part B).

## Features

- Density-based crowd-count estimation.
- Head localization with confidence scores and point offsets.
- Training and validation metrics including MAE, RMSE, R2, Pearson
  correlation, precision, recall, and F1.
- Streamlit image-upload interface.
- JSON output for one image or a complete image folder.
- Diagnostic plots, checkpoints, and per-image evaluation reports.

## Project layout

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit web application for interactive inference. |
| `config.py` | Dataset paths, training settings, thresholds, and output paths. |
| `dataset.py` | ShanghaiTech image and ground-truth loading. |
| `engine.py` | Modular training and validation engine used by `train.py`. |
| `evaluation.py` | Validation and localization metrics. |
| `inference.py` | Image/folder inference implementation. |
| `inference_json.py` | Image/folder inference that saves JSON results. |
| `losses.py` | DDPF-Net loss functions. |
| `models.py` | Swin-Tiny DDPF-Net architecture. |
| `train.py` | Command-line entry point for the modular training pipeline. |
| `training.py` | Standalone training and evaluation pipeline with checkpoint resume and diagnostics. |
| `utils.py` | Density, heatmap, offset, and metric utilities. |

## Requirements

Use Python 3.10 or newer and install the required packages:

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision timm albumentations opencv-python numpy scipy scikit-learn matplotlib tqdm streamlit
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision timm albumentations opencv-python numpy scipy scikit-learn matplotlib tqdm streamlit
```

Install the PyTorch build appropriate for the available CUDA version if GPU
acceleration is required. CPU inference is supported, but training is
intended for a CUDA-capable machine.

## Dataset setup

Download and extract ShanghaiTech so the directory has this shape:

```text
data/
└── part_B/
    ├── train_data/
    │   ├── images/
    │   └── ground-truth/
    └── test_data/
        ├── images/
        └── ground-truth/
```

Update `ROOT`/`LOCAL_DATA_ROOT` and `DATASET_PART` in `config.py` to match
the local dataset. The default modular pipeline uses `part_B`.

Ground-truth files must use the ShanghaiTech naming convention expected by
`dataset.py`, for example `GT_IMG_1.mat`.

## Training

### Modular pipeline

Run the modular training and evaluation entry point:

```bash
python train.py
```

Select a fold with:

```bash
python train.py --fold 0
```

The modular pipeline writes checkpoints and metrics to the directory set by
`CHECKPOINT_DIR` in `config.py` (normally `fin_try/`).

### Standalone pipeline

`training.py` is a self-contained training/evaluation script. It also
supports automatic installation of a few missing Python packages:

```bash
python training.py
```

Start without resuming from the latest checkpoint:

```bash
python training.py --fresh-start
```

Evaluate the saved best checkpoint without training:

```bash
python training.py --eval-only
```

Its output directory and dataset settings are configured near the top of
`training.py`.

## JSON inference

`inference_json.py` accepts either one image or a folder and writes
per-image JSON files plus `summary.json`:

```bash
python inference_json.py `
  --input path\to\image-or-folder `
  --ckpt path\to\model.pth `
  --output-root inference_json_results
```

Optional detection controls:

```bash
python inference_json.py `
  --input path\to\images `
  --ckpt path\to\model.pth `
  --thresh 0.30 `
  --min_dist 4
```

Each JSON result contains:

- density-based count;
- localization-based count;
- overall and quadrant density scores;
- detected head coordinates and confidence values.

The script's default checkpoint search paths are defined in
`resolve_checkpoint_path`. Pass `--ckpt` explicitly when using a checkpoint
stored in a custom directory.

## Streamlit application

Start the interactive application from the repository root:

```bash
streamlit run app.py
```

Set `CHECKPOINT_PATH` in `app.py` to a compatible checkpoint before starting
the app. The application currently defaults to
`op/best_balanced_ddpfnet_final_try.pth`.

## Outputs

Training and diagnostics may generate:

- `.pth` model checkpoints;
- epoch metric CSV files;
- density/error/scatter plots;
- validation visualizations;
- failure-case images;
- JSON inference results.

These generated artifacts are intentionally excluded by `.gitignore`.
Source code, including `training.py` and `inference_json.py`, remains
trackable.

## Reproducibility

The default random seed is `42`. Results can still vary with GPU hardware,
CUDA/cuDNN versions, worker settings, and dataset preprocessing. Record the
dataset part, configuration values, checkpoint path, and detection threshold
when reporting results.

## License

This project is released under the MIT License. See `LICENSE`.
