import os
import torch
import numpy as np

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOCAL_DATA_ROOT = r"d:\Projects\final_year\Implementation\DDPF\data"
KAGGLE_DATA_ROOT = "/kaggle/input/datasets/hosammhmdali/shanghai-tech-dataset-part-a-and-part-b/ShanghaiTech"

if os.path.exists(LOCAL_DATA_ROOT):
    ROOT = LOCAL_DATA_ROOT
elif os.path.exists(KAGGLE_DATA_ROOT):
    ROOT = KAGGLE_DATA_ROOT
else:
    ROOT = LOCAL_DATA_ROOT

DATASET_PART = "part_B"

TRAIN_IMAGE_DIR = os.path.join(ROOT, DATASET_PART, "train_data", "images")
TRAIN_GT_DIR    = os.path.join(ROOT, DATASET_PART, "train_data", "ground-truth")
TEST_IMAGE_DIR  = os.path.join(ROOT, DATASET_PART, "test_data", "images")
TEST_GT_DIR     = os.path.join(ROOT, DATASET_PART, "test_data", "ground-truth")

IMAGE_SIZE  = 512
OUTPUT_SIZE = 512

BATCH_SIZE = 1
NUM_WORKERS = 2
BASE_LR = 2e-4
WARMUP_EPOCHS = 5
EPOCHS = 300
MIN_LR = 1e-6
LR = BASE_LR
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
GRAD_ACCUM_STEPS = 8
EARLY_STOPPING_PATIENCE = 60
IMPROVEMENT_EPS = 1e-4

DENSITY_WEIGHT = 3.0
COUNT_WEIGHT = 1.0
HEATMAP_WEIGHT = 3.0
CONSISTENCY_WEIGHT = 0.0

FOCAL_ALPHA = 0.85
FOCAL_GAMMA = 2.0

DENSITY_SCALE = 100.0

CHECKPOINT_DIR = os.path.join("fin_try")
BEST_F1_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "best_f1_ddpfnet.pth")
BEST_MAE_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "best_mae_ddpfnet.pth")
BEST_BALANCED_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "best_balanced_ddpfnet.pth")
LATEST_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "latest_ddpfnet.pth")
EPOCH_METRICS_CSV = os.path.join(CHECKPOINT_DIR, "epoch_metrics.csv")
DIAGNOSTICS_DIR = os.path.join(CHECKPOINT_DIR, "diagnostics")
USE_TTA_FOR_TEST = True
TRAINING_OBJECTIVE_VERSION = 3
ANNOTATION_SCALE = 2.0

NUM_FOLDS = 5
CURRENT_FOLD = 0
CHECKPOINT_EMA_ALPHA = 0.3

MIN_SIGMA = 4.0
MAX_SIGMA = 16.0
SIGMA_SCALE = 0.3

HEATMAP_MIN_SIGMA = 2.0
HEATMAP_MAX_SIGMA = 8.0
HEATMAP_SIGMA_SCALE = 0.15

PEAK_THRESHOLD = 0.12
ACTIVE_PEAK_THRESHOLD = PEAK_THRESHOLD
MIN_DISTANCE = 4
BORDER_MASK = 0

PEAK_THRESHOLD_SWEEP = [round(x, 2) for x in np.arange(0.05, 0.61, 0.05)]
