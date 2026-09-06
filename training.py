============================================================================

DDPF-Net v2: Dual-Decoder Point-supervised Framework for Crowd Counting

Architecture: Swin-Tiny Backbone -> SwinFPN -> Dual Decoder + Cross Fusion

Outputs: Density Map (Softplus) + Localization Heatmap (Logits)

Dataset: ShanghaiTech (Part A / Part B)

============================================================================

%% ========================================================================

CELL 1: ENVIRONMENT SETUP & IMPORTS

========================================================================

import subprocess
import sys
import os

import multiprocessing as _mp

Reduces CUDA allocator fragmentation during long 512x512 training runs.  This

must be set before importing torch so CUDA picks it up during initialization.

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments")

def install_package(package):
"""Install a package only if not already importable, and only in main process."""
if _mp.current_process().name != 'MainProcess':
return   # workers must never run pip — they just import
try:
import(package)
except ImportError:
print(f"Installing {package}...")
subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

Install required packages (main process only)

install_package("timm")
install_package("albumentations")
install_package("torchvision")
install_package("scikit-learn")

import glob
import math
import random
import datetime
import warnings
from collections import defaultdict

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scipy.io import loadmat
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import KDTree, distance
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, explained_variance_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision

import timm
import albumentations as A
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
print("Libraries Loaded Successfully.")

%% ========================================================================

CELL 2: CONFIGURATION & REPRODUCIBILITY

========================================================================

SEED = 42

def set_seed(seed=42):
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

set_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Paths configuration

LOCAL_DATA_ROOT = "./data"
KAGGLE_DATA_ROOT = "/kaggle/input/datasets/hosammhmdali/shanghai-tech-dataset-part-a-and-part-b/ShanghaiTech"

if os.path.exists(LOCAL_DATA_ROOT):
ROOT = LOCAL_DATA_ROOT
elif os.path.exists(KAGGLE_DATA_ROOT):
ROOT = KAGGLE_DATA_ROOT
else:
ROOT = LOCAL_DATA_ROOT

DATASET_PART = "part_B"  # Change to "part_B" if desired

TRAIN_IMAGE_DIR = os.path.join(ROOT, DATASET_PART, "train_data", "images")
TRAIN_GT_DIR    = os.path.join(ROOT, DATASET_PART, "train_data", "ground-truth")
TEST_IMAGE_DIR  = os.path.join(ROOT, DATASET_PART, "test_data", "images")
TEST_GT_DIR     = os.path.join(ROOT, DATASET_PART, "test_data", "ground-truth")

Image / map sizes

IMAGE_SIZE  = 512
OUTPUT_SIZE = 512   # Full resolution output — eliminates head merging in dense scenes   # 1/2 resolution for prediction maps

Training hyperparameters

BATCH_SIZE = 1                   # 512x512 full-res needs batch=1 on 8GB GPU
NUM_WORKERS = 1    # Single-process loading — no workers, no repeated blocks, training starts instantly
EPOCHS = 150              # short fine-tune cycle on top of pre-trained weights
LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
GRAD_ACCUM_STEPS = 8             # effective batch = 1 × 8 = 8 samples per update
EARLY_STOPPING_PATIENCE = 40  # Stop if val metrics don't improve for this many epochs

Loss weights

DENSITY_WEIGHT = 1.0
COUNT_WEIGHT = 0.50       # Increased for lower MAE
HEATMAP_WEIGHT = 8.0
CONSISTENCY_WEIGHT = 0.0  # Controlled localization run: do not couple density and heatmap heads.

FOCAL_ALPHA = 0.85
FOCAL_GAMMA = 1.5

Density map scale factor

DENSITY_SCALE = 100.0

CHECKPOINT_DIR = "lasthope"
BEST_F1_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "best_f1_ddpfnet_last_hope.pth")
BEST_MAE_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "best_mae_ddpfnet_last_hope.pth")
BEST_BALANCED_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "best_balanced_ddpfnet_last_hope.pth")
LATEST_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "latest_ddpfnet_last_hope.pth")
USE_TTA_FOR_TEST = True

============================================================================

ROBUST MULTI-TASK MODEL SELECTION

============================================================================

REG_SCORE_WEIGHT_MAE = 0.60
REG_SCORE_WEIGHT_R2 = 0.20
REG_SCORE_WEIGHT_PEARSON = 0.20

BASE_REG_WEIGHT = 0.50
BASE_LOC_WEIGHT = 0.50

MIN_TASK_WEIGHT = 0.25
MAX_TASK_WEIGHT = 0.75

How aggressively to shift importance toward the weaker task

ADAPTIVE_STRENGTH = 0.50

Guardrails

MAX_F1_DROP = 0.02       # allow max 2% relative F1 degradation
MAX_MAE_INCREASE = 0.05  # allow max 5% relative MAE degradation

Ignore microscopic changes

IMPROVEMENT_EPS = 1e-4

Density map settings (wider Gaussians — smooth regression target)

MIN_SIGMA = 4.0
MAX_SIGMA = 16.0
SIGMA_SCALE = 0.3

Heatmap settings (tight per-head Gaussians — CenterNet-style)

Each head is independently normalized to peak=1.0 in the GT target.

HEATMAP_MIN_SIGMA = 2.0
HEATMAP_MAX_SIGMA = 8.0
HEATMAP_SIGMA_SCALE = 0.15

Evaluation threshold — 0.30 filters early-training noise (model outputs ~0.2

everywhere initially) while still catching real heads once the model converges

PEAK_THRESHOLD = 0.12   # Slightly below original 0.15 - mild recall improvement

Updated whenever a checkpoint provides a calibrated threshold.  Keeping this

separate from PEAK_THRESHOLD preserves a safe default for fresh training.

ACTIVE_PEAK_THRESHOLD = PEAK_THRESHOLD
PEAK_THRESHOLD_SWEEP = [round(x, 2) for x in np.arange(0.05, 0.61, 0.05)]
MIN_DISTANCE = 4        # Between original 5 and too-aggressive 3
BORDER_MASK = 0

print(f"Device: {DEVICE}")
print(f"Configuration loaded. Dataset: {DATASET_PART}, Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}, Output Size: {OUTPUT_SIZE}x{OUTPUT_SIZE}")
print("TRAIN_IMAGE_DIR:", TRAIN_IMAGE_DIR)
print("TRAIN_GT_DIR   :", TRAIN_GT_DIR)

%% ========================================================================

CELL 3: ADAPTIVE DENSITY / HEATMAP GENERATION

========================================================================

def compute_sigmas(points, min_sigma=2.0, max_sigma=8.0, sigma_scale=0.3):
"""
Geometry-adaptive sigma for each point using KDTree.
Smaller sigma in dense areas, larger sigma in sparse areas.
"""
if len(points) == 0:
return np.array([], dtype=np.float32)

if len(points) == 1:
    return np.array([6.0], dtype=np.float32)

tree = KDTree(points)
k = min(4, len(points))
distances, _ = tree.query(points, k=k)

if distances.ndim == 1:
    nearest_mean = np.full((len(points),), 6.0, dtype=np.float32)
else:
    nearest_mean = distances[:, 1:].mean(axis=1)

sigmas = nearest_mean * sigma_scale
sigmas = np.clip(sigmas, min_sigma, max_sigma).astype(np.float32)
return sigmas

def generate_density_map(image_shape, points, min_sigma=2.0, max_sigma=8.0, sigma_scale=0.3, scale=1.0):
"""
Fast Gaussian splatting for density map.
"""
h, w = image_shape[:2]
density = np.zeros((h, w), dtype=np.float32)

if len(points) == 0:
    return density

sigmas = compute_sigmas(points, min_sigma=min_sigma, max_sigma=max_sigma, sigma_scale=sigma_scale)

for (x, y), sigma in zip(points, sigmas):
    x_int, y_int = int(round(x)), int(round(y))

    radius = int(np.ceil(3 * sigma))
    x1, y1 = max(0, x_int - radius), max(0, y_int - radius)
    x2, y2 = min(w, x_int + radius + 1), min(h, y_int + radius + 1)

    if x1 >= w or y1 >= h or x2 <= 0 or y2 <= 0:
        continue

    X, Y = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
    g = np.exp(-((X - x_int)**2 + (Y - y_int)**2) / (2 * sigma**2))

    g_sum = g.sum()
    if g_sum > 0:
        g = g / g_sum

    density[y1:y2, x1:x2] += g

return density * scale

def generate_heatmap(image_shape, points, min_sigma=2.0, max_sigma=8.0, sigma_scale=0.3):
"""
Fast Gaussian splatting for CenterNet-style heatmap.
"""
h, w = image_shape[:2]
heatmap = np.zeros((h, w), dtype=np.float32)

if len(points) == 0:
    return heatmap

sigmas = compute_sigmas(points, min_sigma=min_sigma, max_sigma=max_sigma, sigma_scale=sigma_scale)

for (x, y), sigma in zip(points, sigmas):
    x_int, y_int = int(round(x)), int(round(y))

    radius = int(np.ceil(3 * sigma))
    x1, y1 = max(0, x_int - radius), max(0, y_int - radius)
    x2, y2 = min(w, x_int + radius + 1), min(h, y_int + radius + 1)

    if x1 >= w or y1 >= h or x2 <= 0 or y2 <= 0:
        continue

    X, Y = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
    g = np.exp(-((X - x_int)**2 + (Y - y_int)**2) / (2 * sigma**2))

    # Element-wise max preserves the 1.0 peaks
    heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], g)

return heatmap

def generate_offset_map(image_shape, points):
"""
Sub-pixel offset maps (dx, dy) and a mask map where GT exists.
"""
h, w = image_shape[:2]
offset = np.zeros((2, h, w), dtype=np.float32)
mask = np.zeros((1, h, w), dtype=np.float32)

for x, y in points:
    # The target must be stored at the same integer centre where NMS finds
    # a heatmap maximum.  The residual is therefore signed in [-0.5, 0.5].
    ix, iy = int(np.rint(x)), int(np.rint(y))
    if 0 <= ix < w and 0 <= iy < h:
        dx = x - ix
        dy = y - iy
        offset[0, iy, ix] = dx
        offset[1, iy, ix] = dy
        mask[0, iy, ix] = 1.0

return offset, mask

def generate_center_map(image_shape, points):
"""One-pixel detection centres, using the same rounded convention as offsets."""
h, w = image_shape[:2]
centers = np.zeros((h, w), dtype=np.float32)
for x, y in points:
ix, iy = int(np.rint(x)), int(np.rint(y))
if 0 <= ix < w and 0 <= iy < h:
centers[iy, ix] = 1.0
return centers

def count_from_density(density_map):
"""Count estimate from a density map."""
return float(density_map.sum())

print("Density map & heatmap generators ready.")

%% ========================================================================

CELL 4: DATASET & COLLATE FUNCTION

========================================================================

def get_image_and_gt_paths(image_dir, gt_dir):
images = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
return images

class ShanghaiTechDataset(Dataset):
def init(self, image_dir, gt_dir, transform=None, indices=None):
self.image_dir = image_dir
self.gt_dir = gt_dir
self.transform = transform

    self.images = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

    if indices is not None:
        self.images = [self.images[i] for i in indices]

def __len__(self):
    return len(self.images)

def _load_points_and_count(self, img_path):
    img_name = os.path.basename(img_path)
    gt_name = img_name.replace("processed_", "").replace(".jpg", ".mat")
    gt_path = os.path.join(self.gt_dir, "GT_" + gt_name)

    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    mat = loadmat(gt_path)

    # ShanghaiTech annotation format
    record = mat["image_info"][0, 0][0, 0]
    points = record[0].astype(np.float32)          # shape: (N, 2)
    count = float(record[1][0, 0])

    return points, count

def __getitem__(self, idx):
    img_path = self.images[idx]

    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {img_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h0, w0 = image.shape[:2]

    points, count = self._load_points_and_count(img_path)

    # Annotations are at half resolution in processed ShanghaiTech MAT files
    if len(points) > 0:
        points = points.copy()
        max_x, max_y = float(points[:, 0].max()), float(points[:, 1].max())
        if max_x < w0 * 0.55 and max_y < h0 * 0.55:
            points[:, 0] *= 2.0
            points[:, 1] *= 2.0

    # Letterbox/Pad to square
    scale = IMAGE_SIZE / max(h0, w0)
    new_w, new_h = int(w0 * scale), int(h0 * scale)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = IMAGE_SIZE - new_w, IMAGE_SIZE - new_h
    image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
    
    # Rescale points to resized image coordinates (padding is bottom/right so coords don't shift)
    if len(points) > 0:
        points[:, 0] *= scale
        points[:, 1] *= scale

    # Augmentations
    if self.transform is not None:
        if hasattr(self.transform, "keypoint_params") and self.transform.keypoint_params is not None:
            aug = self.transform(image=image, keypoints=[tuple(p) for p in points])
            image = aug["image"]
            points = np.asarray(aug["keypoints"], dtype=np.float32) if len(aug["keypoints"]) > 0 else np.zeros((0, 2), dtype=np.float32)
        else:
            aug = self.transform(image=image)
            image = aug["image"]

    # Convert points to output-map coordinates
    out_points = points.copy()
    if len(out_points) > 0:
        out_points[:, 0] *= OUTPUT_SIZE / IMAGE_SIZE
        out_points[:, 1] *= OUTPUT_SIZE / IMAGE_SIZE

    # Supervision maps (density is scaled so per-pixel values are O(1))
    density = generate_density_map(
        (OUTPUT_SIZE, OUTPUT_SIZE, 3),
        out_points,
        min_sigma=MIN_SIGMA,
        max_sigma=MAX_SIGMA,
        sigma_scale=SIGMA_SCALE,
        scale=DENSITY_SCALE
    )

    heatmap = generate_heatmap(
        (OUTPUT_SIZE, OUTPUT_SIZE, 3),
        out_points,
        min_sigma=HEATMAP_MIN_SIGMA,
        max_sigma=HEATMAP_MAX_SIGMA,
        sigma_scale=HEATMAP_SIGMA_SCALE
    )

    # Offsets
    offset, offset_mask = generate_offset_map(
        (OUTPUT_SIZE, OUTPUT_SIZE, 3),
        out_points
    )
    center = generate_center_map((OUTPUT_SIZE, OUTPUT_SIZE, 3), out_points)

    # Tensors
    image = image.astype(np.float32).transpose(2, 0, 1)
    image = torch.from_numpy(image).float()

    density = torch.from_numpy(density).unsqueeze(0).float()
    heatmap = torch.from_numpy(heatmap).unsqueeze(0).float()
    center = torch.from_numpy(center).unsqueeze(0).float()
    offset = torch.from_numpy(offset).float()
    offset_mask = torch.from_numpy(offset_mask).float()
    count = torch.tensor(count, dtype=torch.float32)

    return {
        "image": image,
        "density": density,
        "heatmap": heatmap,
        "center": center,
        "offset": offset,
        "offset_mask": offset_mask,
        "count": count,
        "points": points
    }

def crowd_collate(batch):
images = torch.stack([item["image"] for item in batch], dim=0)
densities = torch.stack([item["density"] for item in batch], dim=0)
heatmaps = torch.stack([item["heatmap"] for item in batch], dim=0)
centers = torch.stack([item["center"] for item in batch], dim=0)
offsets = torch.stack([item["offset"] for item in batch], dim=0)
offset_masks = torch.stack([item["offset_mask"] for item in batch], dim=0)
counts = torch.stack([item["count"] for item in batch], dim=0)
points = [item["points"] for item in batch]

return {
    "image": images,
    "density": densities,
    "heatmap": heatmaps,
    "center": centers,
    "offset": offsets,
    "offset_mask": offset_masks,
    "count": counts,
    "points": points
}

print("Dataset and collate function defined.")

%% ========================================================================

CELL 5: TRANSFORMS & DATALOADERS — instantiated inside run_training_and_eval()

so DataLoader workers only import class definitions and never instantiate objects.

========================================================================

def build_dataloaders():
"""Build and return (train_loader, valid_loader, test_loader, train_dataset, val_dataset, test_dataset)."""
train_transform = A.Compose([
A.HorizontalFlip(p=0.5),
A.ShiftScaleRotate(
shift_limit=0.03,
scale_limit=0.15,
rotate_limit=5,
border_mode=cv2.BORDER_REFLECT_101,
p=0.5
),
A.RandomBrightnessContrast(p=0.3),
A.HueSaturationValue(p=0.2),
A.GaussNoise(p=0.2),
A.MotionBlur(blur_limit=3, p=0.1),
A.Normalize(
mean=(0.485, 0.456, 0.406),
std=(0.229, 0.224, 0.225),
max_pixel_value=255.0
)
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))

valid_transform = A.Compose([
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=255.0
    )
])

# Train / Val split — widened to 95/5 to give ~18 extra training images
all_images = sorted(glob.glob(os.path.join(TRAIN_IMAGE_DIR, "*.jpg")))
rng = np.random.default_rng(SEED)
indices = np.arange(len(all_images))
rng.shuffle(indices)
split = int(0.95 * len(indices))
train_indices = indices[:split].tolist()
val_indices = indices[split:].tolist()

train_dataset = ShanghaiTechDataset(
    image_dir=TRAIN_IMAGE_DIR, gt_dir=TRAIN_GT_DIR,
    transform=train_transform, indices=train_indices
)
val_dataset = ShanghaiTechDataset(
    image_dir=TRAIN_IMAGE_DIR, gt_dir=TRAIN_GT_DIR,
    transform=valid_transform, indices=val_indices
)
test_dataset = ShanghaiTechDataset(
    image_dir=TEST_IMAGE_DIR, gt_dir=TEST_GT_DIR,
    transform=valid_transform
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True,
    collate_fn=crowd_collate, drop_last=True,
    persistent_workers=(NUM_WORKERS > 0)
)
valid_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    collate_fn=crowd_collate,
    persistent_workers=(NUM_WORKERS > 0)
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    collate_fn=crowd_collate,
    persistent_workers=(NUM_WORKERS > 0)
)

print(f"Train samples: {len(train_dataset)}", flush=True)
print(f"Val samples  : {len(val_dataset)}", flush=True)
print(f"Test samples : {len(test_dataset)}", flush=True)
print("DataLoaders ready.", flush=True)
return train_loader, valid_loader, test_loader, train_dataset, val_dataset, test_dataset

%% ========================================================================

CELL 6: SWIN-TINY BACKBONE

========================================================================

class SwinTinyBackbone(nn.Module):
def init(self, pretrained=True):
super().init()
self.backbone = timm.create_model(
"swin_tiny_patch4_window7_224",
pretrained=pretrained,
features_only=True,
out_indices=(0, 1, 2, 3),
img_size=IMAGE_SIZE,
)
print("Feature Channels:", self.backbone.feature_info.channels())

def forward(self, x):
    features = self.backbone(x)
    outputs = []
    for f in features:
        if f.shape[-1] in [96, 192, 384, 768]:
            f = f.permute(0, 3, 1, 2).contiguous()
        outputs.append(f)
    return outputs

print("SwinTinyBackbone class defined.")

%% ========================================================================

CELL 7: FEATURE PYRAMID NETWORK (SWIN-FPN)

========================================================================

class ConvBNAct(nn.Module):
def init(self, in_ch, out_ch):
super().init()
self.block = nn.Sequential(
nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
nn.BatchNorm2d(out_ch),
nn.ReLU(inplace=True)
)

def forward(self, x):
    return self.block(x)

def norm_layer(channels):
groups = 32
while channels % groups != 0 and groups > 1:
groups //= 2
return nn.GroupNorm(groups, channels)

class ConvGNAct(nn.Module):
def init(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
super().init()
self.block = nn.Sequential(
nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
norm_layer(out_channels),
nn.SiLU(inplace=True),
)

def forward(self, x):
    return self.block(x)

class ResidualBlock(nn.Module):
def init(self, channels):
super().init()
self.conv1 = ConvGNAct(channels, channels, kernel_size=3, padding=1)
self.conv2 = nn.Sequential(
nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
norm_layer(channels),
)
self.act = nn.SiLU(inplace=True)

def forward(self, x):
    identity = x
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.act(x + identity)
    return x

class SwinUNetDecoder(nn.Module):
def init(self, in_channels=[96, 192, 384, 768], out_channels=256):
super().init()

    self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    self.conv1 = ConvGNAct(in_channels[3] + in_channels[2], in_channels[2])

    self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    self.conv2 = ConvGNAct(in_channels[2] + in_channels[1], in_channels[1])

    self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    self.conv3 = ConvGNAct(in_channels[1] + in_channels[0], out_channels)

    # NEW: lightweight high-res feature extractor running on the input image itself
    self.hires_stem = nn.Sequential(
        ConvGNAct(3, 32),
        ConvGNAct(32, 32),
    )

    self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    self.conv4 = ConvGNAct(out_channels + 32, out_channels)

    self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    self.conv5 = ConvGNAct(out_channels + 32, out_channels)

    self.shared_conv = ResidualBlock(out_channels)

def forward(self, c1, c2, c3, c4, image):
    x = self.up1(c4)
    x = torch.cat([x, c3], dim=1)
    x = self.conv1(x)

    x = self.up2(x)
    x = torch.cat([x, c2], dim=1)
    x = self.conv2(x)

    x = self.up3(x)
    x = torch.cat([x, c1], dim=1)
    x = self.conv3(x)

    hires = self.hires_stem(image)
    hires_half = F.avg_pool2d(hires, 2)

    x = self.up4(x)
    x = torch.cat([x, hires_half], dim=1)
    x = self.conv4(x)

    x = self.up5(x)
    x = torch.cat([x, hires], dim=1)
    x = self.conv5(x)

    return self.shared_conv(x)

class DensityHead(nn.Module):
def init(self, in_channels, out_channels=1):
super().init()
self.head = nn.Sequential(
ConvGNAct(in_channels, 128),
ConvGNAct(128, 64),
nn.Conv2d(64, out_channels, kernel_size=1)
)

def forward(self, x):
    return F.softplus(self.head(x))

class LocalizationHead(nn.Module):
def init(self, in_channels, out_channels=1):
super().init()
self.head = nn.Sequential(
ConvGNAct(in_channels, 128),
ConvGNAct(128, 64),
nn.Conv2d(64, out_channels, kernel_size=1)
)

def forward(self, x):
    return self.head(x)  # logits

class OffsetHead(nn.Module):
def init(self, in_channels, out_channels=2):
super().init()
self.head = nn.Sequential(
ConvGNAct(in_channels, 128),
ConvGNAct(128, 64),
nn.Conv2d(64, out_channels, kernel_size=1)
)

def forward(self, x):
    # Rounded-centre residuals are signed and bounded to half a pixel.
    return 0.5 * torch.tanh(self.head(x))



print("SwinUNetDecoder and Task Heads defined.")

%% ========================================================================

CELL 9: FULL DDPF-NET V3 MODEL

========================================================================

class DDPFNetLastHope(nn.Module):
"""
DDPF-Net v6
Swin-Tiny Backbone + SwinUNet Decoder (512x512) + Density/Heatmap/Offset Heads
"""
def freeze_early_stages(self):
for name, param in self.encoder.named_parameters():
if any(name.startswith(p) for p in ["patch_embed", "layers.0", "layers.1"]):
param.requires_grad = False

def __init__(self, pretrained=True, shared_channels=256):
    super().__init__()
    self.encoder = SwinTinyBackbone(pretrained=pretrained)
    self.decoder = SwinUNetDecoder(in_channels=[96, 192, 384, 768], out_channels=shared_channels)
    self.density_head = DensityHead(in_channels=shared_channels)
    self.heatmap_head = LocalizationHead(in_channels=shared_channels)
    self.offset_head = OffsetHead(in_channels=shared_channels)

def forward(self, x):
    c1, c2, c3, c4 = self.encoder(x)
    shared_feat = self.decoder(c1, c2, c3, c4, x)
    density = self.density_head(shared_feat)
    heatmap_logits = self.heatmap_head(shared_feat)
    offset = self.offset_head(shared_feat)

    return {
        "density": density,
        "heatmap_logits": heatmap_logits,
        "offset": offset
    }

print("DDPFNetLastHope class defined.")

For backward compatibility with some loading scripts if needed

DDPFNetV3 = DDPFNetLastHope

%% ========================================================================

CELL 10: MULTI-TASK LOSS FUNCTIONS

========================================================================

class CrossConsistencyLoss(nn.Module):
"""
Aligns density prediction with localization heatmap.
The density map is normalized to [0,1] per image and compared with heatmap probabilities.
"""

def __init__(self):
    super().__init__()

def forward(self, density_pred, heatmap_logits):
    heatmap_prob = torch.sigmoid(heatmap_logits)
    B = density_pred.size(0)
    density_flat = density_pred.view(B, -1)

    d_min = density_flat.min(dim=1, keepdim=True)[0]
    d_max = density_flat.max(dim=1, keepdim=True)[0]

    density_norm = (density_flat - d_min) / (d_max - d_min + 1e-6)
    density_norm = density_norm.view_as(density_pred)

    return F.mse_loss(density_norm, heatmap_prob)

class WeightedDensityLoss(nn.Module):
"""
Background-aware density MSE.

Pixels near an annotated head (GT density > threshold) are foreground and
receive full gradient weight.  Pixels far from any head are background and
receive a reduced weight (bg_weight).  This explicitly teaches the network
that trees, buildings, and other textured regions must produce near-zero
density outputs — the plain MSE loss cannot do this because it treats every
background pixel as equally unimportant once its error is small.

Args:
    bg_weight  : gradient scale for background pixels (default 0.1)
    fg_thresh  : GT density value above which a pixel is "foreground".
                 With DENSITY_SCALE=100 and typical Gaussian kernels,
                 any pixel that receives any Gaussian mass will exceed 1e-3.
"""

def __init__(self, bg_weight: float = 0.15, fg_thresh: float = 1e-3):
    super().__init__()
    self.bg_weight = bg_weight
    self.fg_thresh = fg_thresh

def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    fg_mask = (target > self.fg_thresh).float()          # 1 near heads, 0 elsewhere
    bg_mask = 1.0 - fg_mask
    pixel_sq_err = (pred - target) ** 2
    weighted = fg_mask * pixel_sq_err + self.bg_weight * bg_mask * pixel_sq_err
    return weighted.mean()

def centernet_focal_loss(logits, target):
"""CenterNet focal loss for maps with exact 1.0 centre pixels."""
pred = torch.sigmoid(logits.float()).clamp(1e-4, 1.0 - 1e-4)
pos_inds = target.eq(1).float()
neg_inds = target.lt(1).float()
neg_weights = torch.pow(1.0 - target, 4)
pos_loss = torch.log(pred) * torch.pow(1.0 - pred, 2) * pos_inds
neg_loss = torch.log(1.0 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
num_pos = pos_inds.sum()
return -neg_loss.sum() if num_pos < 1 else -(pos_loss.sum() + neg_loss.sum()) / num_pos

class DDPFLoss(nn.Module):
def init(
self,
density_weight=1.0,
heatmap_weight=1.0,
count_weight=0.1,
consistency_weight=0.2,
focal_alpha=0.85,
focal_gamma=2.0,
density_bg_weight=0.2,   # FIX: was silently 0.005 — now actually suppresses bg density
heatmap_bg_weight=0.05,  # FIX: was silently 0.005 — now actually suppresses fp heatmap peaks
offset_weight=0.03,      # FIX: was silently 1.0 — offset dominated 70% of gradient for nothing
center_weight=1.0,
):
super().init()
# Background-aware density MSE — density_bg_weight controls bg pixel penalty
self.density_loss     = WeightedDensityLoss(bg_weight=density_bg_weight)
self.focal_alpha      = focal_alpha
self.focal_gamma      = focal_gamma
self.count_loss       = nn.L1Loss()
self.consistency_loss = CrossConsistencyLoss()

    self.dw  = density_weight
    self.hw  = heatmap_weight
    self.cw  = count_weight
    self.sw  = consistency_weight
    self.ow  = offset_weight
    self.bgw = heatmap_bg_weight  # heatmap false-positive suppression weight
    self.center_weight = center_weight

def forward(self, outputs, gt_density, gt_heatmap, gt_center, gt_offset, gt_offset_mask, gt_count):
    pred_density        = outputs["density"]
    pred_heatmap_logits = outputs["heatmap_logits"]

    # ── Density loss (background-aware weighted MSE) ────────────────────────
    loss_density = self.density_loss(pred_density, gt_density)

    # ── Heatmap focal loss ────────────────────────────────────────────
    # Focal loss focuses gradient on hard examples (missed heads, near-threshold
    # pixels) and down-weights the overwhelming easy-negative background pixels.
    # Higher alpha increases the contribution of positive head pixels.
    # gamma=2.0 is the standard RetinaNet/CenterNet value.
    # Gaussian context plus an explicit one-pixel detection-centre target.
    loss_heatmap = centernet_focal_loss(pred_heatmap_logits, gt_heatmap.float())
    loss_center = centernet_focal_loss(pred_heatmap_logits, gt_center.float())

    # ── Count loss (normalized L1) ───────────────────────────────────────
    pred_count_raw = pred_density.sum(dim=(1, 2, 3)) / DENSITY_SCALE
    gt_count = gt_count.float()
    mean_gt_count = gt_count.mean().clamp(min=1.0)
    loss_count = self.count_loss(pred_count_raw, gt_count) / mean_gt_count

    # ── Cross-consistency loss ──────────────────────────────────────────
    loss_consistency = self.consistency_loss(pred_density, pred_heatmap_logits)

    # ── Heatmap background suppression ──────────────────────────────────
    # Directly suppress confidence outside localization centres.
    bg_mask = (gt_center < 0.5).float()
    pred_heatmap_probs = torch.sigmoid(pred_heatmap_logits.float())
    loss_heatmap_bg = (pred_heatmap_probs.pow(2.0) * bg_mask).sum() / (bg_mask.sum() + 1e-6)

    # ── Offset loss (L1 on positive head locations) ──────────────────────
    pred_offset = outputs["offset"]
    loss_offset_raw = F.l1_loss(pred_offset, gt_offset, reduction='none')
    loss_offset = (loss_offset_raw * gt_offset_mask).sum() / (gt_offset_mask.sum() + 1e-6)

    total_loss = (
        self.dw * loss_density
        + self.hw * (loss_heatmap + self.center_weight * loss_center + self.bgw * loss_heatmap_bg)
        + self.cw * loss_count
        + self.sw * loss_consistency
        + self.ow * loss_offset
    )

    return {
        "loss"             : total_loss,
        "density_loss"     : self.dw * loss_density,
        "heatmap_loss"     : self.hw * (loss_heatmap + self.center_weight * loss_center + self.bgw * loss_heatmap_bg),
        "center_loss"      : self.hw * self.center_weight * loss_center,
        "count_loss"       : self.cw * loss_count,
        "consistency_loss" : self.sw * loss_consistency,
        "offset_loss"      : self.ow * loss_offset,
    }

print("Multi-task loss functions defined.")

%% ========================================================================

CELL 11: TRAIN & VALIDATION FUNCTIONS

========================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
model.train()
running_loss = 0.0
running_density = 0.0
running_heatmap = 0.0
running_count = 0.0
running_consistency = 0.0
running_offset = 0.0

amp_dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16

optimizer.zero_grad(set_to_none=True)
for step, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
    images = batch["image"].to(device, non_blocking=True)
    density = batch["density"].to(device, non_blocking=True)
    heatmap = batch["heatmap"].to(device, non_blocking=True)
    center = batch["center"].to(device, non_blocking=True)
    offset = batch["offset"].to(device, non_blocking=True)
    offset_mask = batch["offset_mask"].to(device, non_blocking=True)
    count = batch["count"].to(device, non_blocking=True)

    try:
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == 'cuda')):
            outputs = model(images)
            losses = criterion(outputs, density, heatmap, center, offset, offset_mask, count)
            loss = losses["loss"] / GRAD_ACCUM_STEPS   # scale loss

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            del images, density, heatmap, center, offset, offset_mask, count, outputs, losses, loss
            continue

        scaler.scale(loss).backward()
    except RuntimeError as exc:
        # Keep a rare fragmented-allocation failure from ending the entire
        # run.  The next batch starts with no retained gradients or cache.
        if device.type != "cuda" or "out of memory" not in str(exc).lower():
            raise
        print(f"  [!] CUDA OOM at training batch {step + 1}; clearing cache and skipping it.")
        optimizer.zero_grad(set_to_none=True)
        # These names are available before the forward pass; forward-pass
        # outputs are conditional because allocation may have failed early.
        del images, density, heatmap, center, offset, offset_mask, count
        if "outputs" in locals():
            del outputs
        if "losses" in locals():
            del losses
        if "loss" in locals():
            del loss
        torch.cuda.empty_cache()
        continue

    # Only update weights every GRAD_ACCUM_STEPS steps
    if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(loader):
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    running_loss += losses["loss"].item()
    running_density += losses["density_loss"].item()
    running_heatmap += losses["heatmap_loss"].item()
    running_count += losses["count_loss"].item()
    running_consistency += losses["consistency_loss"].item()
    running_offset += losses["offset_loss"].item()

    # Do not retain the previous batch while the next batch is copied to
    # CUDA.  At full resolution, that short overlap can trigger an OOM.
    del images, density, heatmap, center, offset, offset_mask, count, outputs, losses, loss

n = max(len(loader), 1)
return {
    "loss": running_loss / n,
    "density_loss": running_density / n,
    "heatmap_loss": running_heatmap / n,
    "count_loss": running_count / n,
    "consistency_loss": running_consistency / n,
    "offset_loss": running_offset / n,
}

@torch.no_grad()
def validate(model, loader, criterion, device):
model.eval()
running_loss = running_density = running_heatmap = running_count_l = running_consistency = running_offset = 0.0

all_gt_counts = []
all_pred_counts = []

all_density_mean = []
all_density_max = []
all_density_ratio = []

# Cache outputs for threshold sweep
cached_heats = []
cached_offsets = []
cached_gt_pts = []

amp_dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16

for batch in tqdm(loader, desc="Validating", leave=False):
    images = batch["image"].to(device, non_blocking=True)
    densities = batch["density"].to(device, non_blocking=True)
    counts = batch["count"].to(device, non_blocking=True)
    heatmaps = batch["heatmap"].to(device, non_blocking=True)
    centers = batch["center"].to(device, non_blocking=True)
    offsets = batch["offset"].to(device, non_blocking=True)
    offset_masks = batch["offset_mask"].to(device, non_blocking=True)

    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == 'cuda')):
            outputs = model(images)
            loss_dict = criterion(outputs, densities, heatmaps, centers, offsets, offset_masks, counts)
            loss = loss_dict["loss"]

    running_loss += loss.item()
    running_density += loss_dict["density_loss"].item()
    running_heatmap += loss_dict["heatmap_loss"].item()
    running_count_l += loss_dict["count_loss"].item()
    running_offset += loss_dict["offset_loss"].item()
    if "consistency_loss" in loss_dict:
        running_consistency += loss_dict["consistency_loss"].item()

    # Density stats
    pred_den = outputs["density"].detach()
    pred_cnt = (pred_den.sum(dim=(1, 2, 3)) / DENSITY_SCALE).cpu().numpy()
    gt_cnt = counts.cpu().numpy()

    all_pred_counts.extend(pred_cnt.tolist())
    all_gt_counts.extend(gt_cnt.tolist())

    heat_prob = torch.sigmoid(outputs["heatmap_logits"].float())
    heat_np   = heat_prob.cpu().numpy()
    offset_np = outputs["offset"].detach().float().cpu().numpy()

    for i in range(images.size(0)):
        cached_heats.append(heat_np[i, 0])
        cached_offsets.append(offset_np[i])
        
        gt_pts = batch["points"][i].copy() if len(batch["points"][i]) > 0 else np.zeros((0, 2))
        if len(gt_pts) > 0:
            gt_pts[:, 0] *= OUTPUT_SIZE / IMAGE_SIZE
            gt_pts[:, 1] *= OUTPUT_SIZE / IMAGE_SIZE
        cached_gt_pts.append(gt_pts)
        
        d_map = pred_den[i, 0].cpu().numpy()
        all_density_mean.append(float(d_map.mean()))
        all_density_max.append(float(d_map.max()))
        ratio = float(pred_cnt[i]) / max(float(gt_cnt[i]), 1.0)
        all_density_ratio.append(ratio)

n_b = max(len(loader), 1)
pred_arr = np.array(all_pred_counts)
gt_arr   = np.array(all_gt_counts)
err_arr  = pred_arr - gt_arr

count_ratio = np.sum(pred_arr) / max(np.sum(gt_arr), 1.0)

# Threshold Sweep
best_f1 = -1.0
best_metrics = {}
metrics_at_fixed_thresh = {}

for th in PEAK_THRESHOLD_SWEEP:
    g_TP = g_FP = g_FN = 0
    all_peaks = []
    for i in range(len(cached_heats)):
        pts = extract_points(cached_heats[i], offset_map=cached_offsets[i], threshold=th, min_distance=MIN_DISTANCE, border_mask=BORDER_MASK)
        all_peaks.append(len(pts))
        gt_pts = cached_gt_pts[i]
        
        gt_sigmas = np.zeros(len(gt_pts))
        if len(gt_pts) > 0:
            gt_sigmas = compute_sigmas(gt_pts, min_sigma=HEATMAP_MIN_SIGMA, max_sigma=HEATMAP_MAX_SIGMA, sigma_scale=HEATMAP_SIGMA_SCALE)
            
        tp, fp, fn = localization_metrics_adaptive(pts, gt_pts, gt_sigmas)
        g_TP += tp
        g_FP += fp
        g_FN += fn
        
    g_prec = g_TP / (g_TP + g_FP + 1e-8)
    g_rec = g_TP / (g_TP + g_FN + 1e-8)
    g_f1 = 2 * g_prec * g_rec / (g_prec + g_rec + 1e-8)
    
    if abs(th - 0.20) < 1e-5:
        metrics_at_fixed_thresh["f1_0.20"] = float(g_f1)
    elif abs(th - 0.30) < 1e-5:
        metrics_at_fixed_thresh["f1_0.30"] = float(g_f1)
    elif abs(th - 0.40) < 1e-5:
        metrics_at_fixed_thresh["f1_0.40"] = float(g_f1)
    elif abs(th - 0.50) < 1e-5:
        metrics_at_fixed_thresh["f1_0.50"] = float(g_f1)
    
    if g_f1 > best_f1:
        best_f1 = g_f1
        best_metrics = {
            "precision": float(g_prec),
            "recall": float(g_rec),
            "f1": float(g_f1),
            "global_TP": g_TP,
            "global_FP": g_FP,
            "global_FN": g_FN,
            "avg_peaks": float(np.mean(all_peaks)),
            "peak_ratio": float(np.sum(all_peaks)) / max(np.sum(gt_arr), 1.0),
            "best_thresh": th
        }

r2 = r2_score(gt_arr, pred_arr) if len(gt_arr) > 1 else 0.0
pearson, _ = pearsonr(gt_arr, pred_arr) if len(gt_arr) > 1 else (0.0, 0.0)

metrics = {
    "loss": running_loss / n_b,
    "density_loss": running_density / n_b,
    "heatmap_loss": running_heatmap / n_b,
    "count_loss": running_count_l / n_b,
    "offset_loss": running_offset / n_b,
    "MAE": float(np.mean(np.abs(err_arr))),
    "RMSE": float(np.sqrt(np.mean(err_arr ** 2))),
    "gt_mean": float(gt_arr.mean()),
    "count_ratio": count_ratio,
    "pred_mean": float(pred_arr.mean()),
    "pred_std": float(pred_arr.std()),
    "density_map_mean": float(np.mean(all_density_mean)),
    "density_map_max": float(np.mean(all_density_max)),
    "R2": float(r2),
    "Pearson": float(pearson),
}
metrics.update(best_metrics)
metrics.update(metrics_at_fixed_thresh)
if "consistency_loss" in locals() or running_consistency > 0:
    metrics["consistency_loss"] = running_consistency / n_b
    
return metrics

print("Train & validation functions defined.")

%% ========================================================================

CELL 12: LOCALIZATION UTILITIES

========================================================================

def extract_points(heatmap, offset_map=None, threshold=None, min_distance=5, border_mask=0):
"""
Extract local maxima from predicted heatmap and apply predicted sub-pixel offsets.
"""
if torch.is_tensor(heatmap):
heatmap = heatmap.detach().float().cpu().numpy()

threshold = ACTIVE_PEAK_THRESHOLD if threshold is None else threshold
heatmap = heatmap.squeeze().copy()

# Mask borders
if border_mask > 0:
    heatmap[:border_mask, :] = 0
    heatmap[-border_mask:, :] = 0
    heatmap[:, :border_mask] = 0
    heatmap[:, -border_mask:] = 0

local_max = maximum_filter(heatmap, size=min_distance)
peaks = (heatmap == local_max)
peaks &= (heatmap > threshold)

ys, xs = np.where(peaks)

points = []
for x, y in zip(xs, ys):
    if offset_map is not None:
        # apply offset with clipping for safety
        dx = np.clip(offset_map[0, y, x], -0.5, 0.5)
        dy = np.clip(offset_map[1, y, x], -0.5, 0.5)
        px, py = x + dx, y + dy
        
        if 0 <= px < heatmap.shape[1] and 0 <= py < heatmap.shape[0]:
            points.append((px, py))
    else:
        points.append((x, y))

return points

def upscale_points(points, input_size=OUTPUT_SIZE, original_size=IMAGE_SIZE):
scale = original_size / input_size
output = [(x * scale, y * scale) for x, y in points]
return output

def set_active_checkpoint_threshold(checkpoint):
"""Apply a calibrated checkpoint threshold, falling back only when absent."""
global ACTIVE_PEAK_THRESHOLD
ACTIVE_PEAK_THRESHOLD = float(checkpoint.get("best_conf_threshold", PEAK_THRESHOLD))
print(f"[Inference] Active peak threshold: {ACTIVE_PEAK_THRESHOLD:.2f}")

def predict_count(density_map):
if torch.is_tensor(density_map):
return density_map.sum().item()
return density_map.sum()

def localization_metrics_adaptive(pred_points, gt_points, gt_sigmas):
if len(pred_points) == 0 and len(gt_points) == 0:
return 0, 0, 0
if len(pred_points) == 0 or len(gt_points) == 0:
return 0, len(pred_points), len(gt_points)

pred = np.array(pred_points)
gt = np.array(gt_points)
D = cdist(pred, gt)

# Scale-adaptive radius
radii = np.maximum(4.0, 1.5 * gt_sigmas)

matched_pred = set()
matched_gt = set()

while True:
    idx = np.unravel_index(np.argmin(D), D.shape)
    d = D[idx]
    p, g = idx
    
    if d > radii[g]:
        break

    if p in matched_pred:
        D[p, :] = 1e9
        continue
    if g in matched_gt:
        D[:, g] = 1e9
        continue

    matched_pred.add(p)
    matched_gt.add(g)
    D[p, :] = 1e9
    D[:, g] = 1e9

TP = len(matched_pred)
FP = len(pred) - TP
FN = len(gt) - TP

return TP, FP, FN

print("Localization utilities defined.")

%% ========================================================================

CELL 13: EVALUATION & VISUALIZATION UTILITIES

========================================================================

def apply_offset_flip_correction(offset, mode):
"""Signed rounded-centre offsets, already spatially un-flipped."""
if mode == "hflip":
offset = offset.clone()
offset[:, 0] = -offset[:, 0]         # dx direction reverses
elif mode == "vflip":
offset = offset.clone()
offset[:, 1] = -offset[:, 1]         # dy direction reverses
return offset

@torch.no_grad()
def run_tta_forward(model, images, device, amp_dtype):
modes = ["orig", "hflip", "vflip"]
aug_imgs = {
"orig": images,
"hflip": torch.flip(images, dims=[3]),
"vflip": torch.flip(images, dims=[2]),
}
densities, heatmaps, offsets = [], [], []
for mode in modes:
with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == 'cuda')):
out = model(aug_imgs[mode])
d = out["density"].float()
h = torch.sigmoid(out["heatmap_logits"].float())
o = out["offset"].float()
if mode == "hflip":
d = torch.flip(d, dims=[3]); h = torch.flip(h, dims=[3]); o = torch.flip(o, dims=[3])
o = apply_offset_flip_correction(o, "hflip")
elif mode == "vflip":
d = torch.flip(d, dims=[2]); h = torch.flip(h, dims=[2]); o = torch.flip(o, dims=[2])
o = apply_offset_flip_correction(o, "vflip")
densities.append(d); heatmaps.append(h); offsets.append(o)

density = torch.stack(densities).mean(dim=0)
heatmap = torch.stack(heatmaps).mean(dim=0)
offset  = torch.stack(offsets).mean(dim=0).cpu().numpy()
return density, heatmap, offset

@torch.no_grad()
def evaluate_model(model, loader, device, threshold=None, min_dist=MIN_DISTANCE, use_tta=USE_TTA_FOR_TEST, verbose=True):
model.eval()
threshold = ACTIVE_PEAK_THRESHOLD if threshold is None else threshold
gt_counts = []
pred_counts = []
global_TP = global_FP = global_FN = 0   # accumulate across all images

amp_dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16

iterator = tqdm(loader, desc=f"Evaluating (TTA={use_tta})", leave=False) if verbose else loader
for batch in iterator:
    images = batch["image"].to(device, non_blocking=True)

    if use_tta:
        density, heatmap, offset_map = run_tta_forward(model, images, device, amp_dtype)
    else:
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == 'cuda')):
            out = model(images)
        density = out["density"].float()
        heatmap = torch.sigmoid(out["heatmap_logits"].float())
        offset_map = out["offset"].detach().float().cpu().numpy()

    pred = (density.sum(dim=(1, 2, 3)) / DENSITY_SCALE).cpu().numpy()
    gt = batch["count"].numpy()

    pred_counts.extend(pred.tolist())
    gt_counts.extend(gt.tolist())

    for i in range(images.size(0)):
        pred_pts = extract_points(heatmap[i, 0].cpu().numpy(), offset_map=offset_map[i], threshold=threshold, min_distance=min_dist, border_mask=BORDER_MASK)
        gt_pts = batch["points"][i].copy() if len(batch["points"][i]) > 0 else np.zeros((0, 2))

        gt_sigmas = np.zeros(len(gt_pts))
        if len(gt_pts) > 0:
            gt_pts[:, 0] *= OUTPUT_SIZE / IMAGE_SIZE
            gt_pts[:, 1] *= OUTPUT_SIZE / IMAGE_SIZE
            gt_sigmas = compute_sigmas(gt_pts, min_sigma=HEATMAP_MIN_SIGMA, max_sigma=HEATMAP_MAX_SIGMA, sigma_scale=HEATMAP_SIGMA_SCALE)

        TP, FP, FN = localization_metrics_adaptive(pred_pts, gt_pts, gt_sigmas)
        global_TP += TP
        global_FP += FP
        global_FN += FN

gt_counts = np.array(gt_counts)
pred_counts = np.array(pred_counts)

mae = np.mean(np.abs(gt_counts - pred_counts))
rmse = np.sqrt(np.mean((gt_counts - pred_counts) ** 2))
mape = np.mean(np.abs(gt_counts - pred_counts) / (gt_counts + 1e-6)) * 100
r2 = r2_score(gt_counts, pred_counts)
explained = explained_variance_score(gt_counts, pred_counts)
pearson, _ = pearsonr(gt_counts, pred_counts) if len(gt_counts) > 1 else (0.0, 0.0)

precision = global_TP / (global_TP + global_FP + 1e-8)
recall    = global_TP / (global_TP + global_FN + 1e-8)
f1        = 2 * precision * recall / (precision + recall + 1e-8)

if verbose:
    print("=" * 60)
    print("Regression Metrics")
    print("=" * 60)
    print(f"MAE                 : {mae:.3f}")
    print(f"RMSE                : {rmse:.3f}")
    print(f"MAPE                : {mape:.2f}%")
    print(f"R2 Score            : {r2:.4f}")
    print(f"Explained Variance  : {explained:.4f}")
    print(f"Pearson Correlation : {pearson:.4f}")

    print()
    print("=" * 60)
    print("Localization Metrics")
    print("=" * 60)
    print(f"Precision           : {precision:.4f}")
    print(f"Recall              : {recall:.4f}")
    print(f"F1 Score            : {f1:.4f}")

    # Save Scatter Plot
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(gt_counts, pred_counts, alpha=0.6, color='#2196F3')
    m = max(gt_counts.max(), pred_counts.max()) if len(gt_counts) > 0 else 100
    plt.plot([0, m], [0, m], 'r--', linewidth=2)
    plt.xlabel("Ground Truth Count")
    plt.ylabel("Predicted Count")
    plt.title("GT vs Predicted Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, "scatter_gt_vs_pred.png"), dpi=150)
    plt.close()

    # Save Error Histogram
    errors = pred_counts - gt_counts
    plt.figure(figsize=(6, 4))
    plt.hist(errors, bins=30, color='#FF9800', edgecolor='k', alpha=0.7)
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, "error_histogram.png"), dpi=150)
    plt.close()

return mae, f1, precision, recall

@torch.no_grad()
def visualize_prediction(model, dataset, idx=0, save_path=None):
model.eval()
sample = dataset[idx]
image = sample["image"].unsqueeze(0).to(DEVICE)
amp_dtype = torch.bfloat16 if (DEVICE.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16

with torch.amp.autocast(device_type=DEVICE.type, dtype=amp_dtype, enabled=(DEVICE.type == 'cuda')):
    output = model(image)

density = output["density"][0, 0].float().cpu().numpy()
heatmap = torch.sigmoid(output["heatmap_logits"].float())[0, 0].cpu().numpy()
offset_map = output["offset"][0].detach().float().cpu().numpy()

pred_count = density.sum() / DENSITY_SCALE  # undo DENSITY_SCALE for actual count
pred_points = extract_points(heatmap, offset_map=offset_map, threshold=ACTIVE_PEAK_THRESHOLD, min_distance=MIN_DISTANCE, border_mask=BORDER_MASK)

fig, ax = plt.subplots(1, 3, figsize=(18, 6))

img = sample["image"].permute(1, 2, 0).numpy()
img = img * 0.229 + 0.485
img = np.clip(img, 0, 1)

ax[0].imshow(img)
ax[0].set_title(f"GT:{sample['count']} Pred:{pred_count:.1f}")

ax[1].imshow(density, cmap="jet")
ax[1].set_title("Density")

ax[2].imshow(heatmap, cmap="hot")
for x, y in pred_points:
    ax[2].plot(x, y, "go", markersize=3)
ax[2].set_title(f"Localization ({len(pred_points)})")

plt.tight_layout()
if save_path:
    plt.savefig(save_path, dpi=150)
plt.close()

print("Evaluation and visualization functions ready.")

%% ========================================================================

CELL 14: CUDA CHECK & READINESS --- HARD STOP ---

========================================================================

def cuda_readiness_check(model, config_dict, train_loader, val_loader, test_loader):
separator = "=" * 70
print(separator)
print("          DDPF-Net v6 -- PRE-TRAINING READINESS CHECK")
print(separator)

cuda_available = torch.cuda.is_available()
print(f"\n[SYSTEM INFO]")
print(f"   CUDA Available      : {'[+] Yes' if cuda_available else '[-] No'}")
if cuda_available:
    print(f"   Device Name         : {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"   GPU Memory          : {total_mem:.1f} GB")
print(f"   PyTorch Version     : {torch.__version__}")
print(f"   Device Selected     : {DEVICE}")

n_train = len(train_loader.dataset)
n_val = len(val_loader.dataset)
n_test = len(test_loader.dataset)
print(f"\n[DATA SUMMARY]")
print(f"   Dataset             : ShanghaiTech {DATASET_PART}")
print(f"   Training Samples    : {n_train}")
print(f"   Validation Samples  : {n_val}")
print(f"   Test Samples        : {n_test}")
print(f"   Train Batches/Epoch : {len(train_loader)}")
print(f"   Batch Size          : {BATCH_SIZE}")
print(f"   Input Resolution    : {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"   Output Resolution   : {OUTPUT_SIZE}x{OUTPUT_SIZE}")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n[MODEL STATS]")
print(f"   Architecture        : DDPF-Net v6 (Swin-Tiny + SwinUNetDecoder 512x512 + TTA)")
print(f"   Total Parameters    : {total_params:,}")
print(f"   Trainable Parameters: {trainable_params:,}")
print(f"   Parameter Size      : {total_params * 4 / (1024 ** 2):.1f} MB (float32)")

print(f"\n[DUMMY FORWARD PASS]")
model.to(DEVICE)
dummy = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
try:
    with torch.no_grad():
        amp_dtype = torch.bfloat16 if (DEVICE.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
        with torch.amp.autocast(device_type=DEVICE.type, dtype=amp_dtype, enabled=(DEVICE.type == 'cuda')):
            out = model(dummy)
    print(f"   Input Shape         : {list(dummy.shape)}")
    print(f"   Density Output      : {list(out['density'].shape)}")
    print(f"   Heatmap Logits      : {list(out['heatmap_logits'].shape)}")
    print(f"   Forward Pass        : [+] Success")
except Exception as e:
    print(f"   Forward Pass        : [-] FAILED -- {e}")
    return False

if cuda_available:
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
    print(f"\n[GPU MEMORY] (after dummy pass)")
    print(f"   Allocated           : {allocated:.1f} MB")
    print(f"   Reserved            : {reserved:.1f} MB")

print(f"\n[TRAINING CONFIG]")
print(f"   Epochs              : {EPOCHS}")
print(f"   Learning Rate       : {LR}")
print(f"   Weight Decay        : {WEIGHT_DECAY}")
print(f"   Gradient Clipping   : {GRAD_CLIP}")
print(f"   Loss Weights        : density={DENSITY_WEIGHT}, heatmap={HEATMAP_WEIGHT}, count={COUNT_WEIGHT}, consistency={CONSISTENCY_WEIGHT}")

print(f"\n{separator}")
print(f"   [+] READY TO START TRAINING: YES")
print(f"{separator}")

del dummy, out
if cuda_available:
    torch.cuda.empty_cache()

return True

Initialization moved inside run_training_and_eval to prevent multiprocessing VRAM deadlocks

%% ========================================================================

ROBUST MULTI-TASK HELPER FUNCTIONS

========================================================================

def compute_task_scores(vs):
"""
Convert regression + localization metrics into [0,1] scores.
Higher is always better.
"""
gt_mean = max(float(vs["gt_mean"]), 1.0)
mae_score = 1.0 / (1.0 + float(vs["MAE"]) / gt_mean)
r2_score_norm = np.clip(float(vs.get("R2", 0.0)), 0.0, 1.0)
pearson_score_norm = np.clip(float(vs.get("Pearson", 0.0)), 0.0, 1.0)

regression_score = (
    REG_SCORE_WEIGHT_MAE * mae_score +
    REG_SCORE_WEIGHT_R2 * r2_score_norm +
    REG_SCORE_WEIGHT_PEARSON * pearson_score_norm
)

localization_score = np.clip(float(vs["f1"]), 0.0, 1.0)
return float(regression_score), float(localization_score)

def compute_adaptive_task_weights(regression_score, localization_score):
"""
The weaker task receives more weight.
"""
total = regression_score + localization_score + 1e-8
reg_weakness = localization_score / total
loc_weakness = regression_score / total

reg_weight = BASE_REG_WEIGHT + ADAPTIVE_STRENGTH * (reg_weakness - 0.5)
loc_weight = BASE_LOC_WEIGHT + ADAPTIVE_STRENGTH * (loc_weakness - 0.5)

reg_weight = np.clip(reg_weight, MIN_TASK_WEIGHT, MAX_TASK_WEIGHT)
loc_weight = np.clip(loc_weight, MIN_TASK_WEIGHT, MAX_TASK_WEIGHT)

total_weight = reg_weight + loc_weight
return float(reg_weight / total_weight), float(loc_weight / total_weight)

def compute_overall_score(regression_score, localization_score, regression_weight, localization_weight):
"""
Weighted geometric mean.
Both tasks must remain healthy for the score to remain high.
"""
r_score = max(regression_score, 1e-8)
l_score = max(localization_score, 1e-8)
return float((r_score ** regression_weight) * (l_score ** localization_weight))

def pareto_improved(current_mae, current_f1, best_mae, best_f1, eps=1e-4):
"""
True if current model is no worse in both metrics
and strictly better in at least one.
"""
no_worse_mae = current_mae <= best_mae + eps
no_worse_f1 = current_f1 >= best_f1 - eps
strictly_better_mae = current_mae < best_mae - eps
strictly_better_f1 = current_f1 > best_f1 + eps

return no_worse_mae and no_worse_f1 and (strictly_better_mae or strictly_better_f1)

def passes_metric_guardrails(current_mae, current_f1, best_mae, best_f1):
"""
Prevent one task from being sacrificed for the other.
"""
f1_ok = True
if best_f1 > 0:
f1_drop = (best_f1 - current_f1) / best_f1
f1_ok = f1_drop <= MAX_F1_DROP

mae_ok = True
if best_mae != float("inf"):
    mae_increase = (current_mae - best_mae) / max(best_mae, 1e-8)
    mae_ok = mae_increase <= MAX_MAE_INCREASE

return f1_ok and mae_ok

%% ========================================================================

CELL 15: TRAINING LOOP & AUTOMATIC EVALUATION

========================================================================

def run_training_and_eval(eval_only=False):
# Build datasets & loaders here (not at module level) so DataLoader workers
# only import class definitions and don't re-run expensive instantiation.
print("[INIT] Building dataloaders...", flush=True)
train_loader, valid_loader, test_loader, train_dataset, val_dataset, test_dataset = build_dataloaders()

print("\n--- Initialising model, optimiser, scheduler, and loss ---", flush=True)
model = DDPFNetLastHope(pretrained=True, shared_channels=256).to(DEVICE)

model.freeze_early_stages()

no_decay = set()
import torch.nn as nn
for mn, m in model.named_modules():
    if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        for pn, p in m.named_parameters():
            no_decay.add(f"{mn}.{pn}" if mn else pn)
            
backbone_params = []
head_params = []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    is_backbone = name.startswith("encoder")
    is_bias_or_norm = name.endswith(".bias") or name in no_decay

    group = {"params": [param], "weight_decay": 0.0 if is_bias_or_norm else WEIGHT_DECAY}
    if is_backbone:
        group["lr"] = 1e-5
        group["is_backbone"] = True
        backbone_params.append(group)
    else:
        group["lr"] = LR
        group["is_backbone"] = False
        head_params.append(group)

optimizer = AdamW(backbone_params + head_params)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler = torch.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))
criterion = DDPFLoss(
    density_weight=DENSITY_WEIGHT,
    heatmap_weight=HEATMAP_WEIGHT,
    count_weight=COUNT_WEIGHT,
    consistency_weight=CONSISTENCY_WEIGHT,
    focal_alpha=FOCAL_ALPHA,
    focal_gamma=FOCAL_GAMMA,
    density_bg_weight=0.4,   # increased for lower MAE
    heatmap_bg_weight=0.20,  # increased for higher precision
    offset_weight=0.03,      # near-zero: sub-pixel offset doesn't affect F1
)

if not eval_only:
    cuda_readiness_check(model, {}, train_loader, valid_loader, test_loader)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
best_combined_score = -float("inf")
best_mae = float("inf")
best_f1 = 0.0
best_pareto_mae = float("inf")
best_pareto_f1 = 0.0
best_conf_threshold = PEAK_THRESHOLD
epochs_no_improve = 0
history = {"train_loss": [], "val_loss": [], "mae": [], "rmse": [], "lr": []}

start_epoch = 0

# ── Warm-start: load model weights only, reset everything else ─────────
# The best_f1 checkpoint has well-trained features; we want a fresh LR
# cycle with the corrected loss so the network can actually re-shape the
# density/heatmap heads.  Loading optimizer/scheduler state would start
# at LR≈8e-6 (the floor of the previous cosine decay) — too small to
# learn anything with the new loss balance.
#
# Priority order:
#  1. lasthope/latest  → interrupt-safe resume of THIS run (full: weights + optimizer)
#  2. lasthope/best_f1 → weights-only warm-start from the best saved model
#  3. nothing → fresh start from scratch
warm_start_path = None
full_resume     = False   # True = also restore optimizer/scheduler/epoch

if os.path.exists(LATEST_SAVE_PATH):
    warm_start_path = LATEST_SAVE_PATH
    full_resume     = True
else:
    old_best = BEST_F1_SAVE_PATH
    if os.path.exists(old_best):
        warm_start_path = old_best
        full_resume     = False

if warm_start_path:
    print(f"[Warm-Start] Loading weights from {warm_start_path} (full_resume={full_resume})")
    ckpt = torch.load(warm_start_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    best_combined_score = ckpt.get("best_combined_score", -float("inf"))
    best_mae            = ckpt.get("best_mae", float("inf"))
    best_f1             = ckpt.get("best_f1", 0.0)
    best_pareto_mae     = ckpt.get("best_pareto_mae", float("inf"))
    best_pareto_f1      = ckpt.get("best_pareto_f1", 0.0)
    best_conf_threshold = ckpt.get("best_conf_threshold", PEAK_THRESHOLD)
    set_active_checkpoint_threshold(ckpt)
    if full_resume:
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch       = ckpt.get("epoch", 0)
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        history           = ckpt.get("history", history)
        print(f"[Warm-Start] Full resume from epoch {start_epoch + 1} | "
              f"MAE={best_mae:.3f} F1={best_f1:.3f} Patience={epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")
    else:
        # Weights-only: fresh LR cycle so corrected losses can reshape the heads
        print(f"[Warm-Start] Weights loaded. Epoch/optimizer/scheduler reset to fresh state.")
        print(f"[Warm-Start] Prev best from old run: MAE={best_mae:.3f} F1={best_f1:.3f}")
else:
    print("[Warm-Start] No checkpoint found — training from scratch.")

# ── eval-only: skip training entirely ─────────────────────────────────
if eval_only:
    eval_path = BEST_F1_SAVE_PATH if os.path.exists(BEST_F1_SAVE_PATH) else (LATEST_SAVE_PATH if os.path.exists(LATEST_SAVE_PATH) else None)
    if eval_path:
        print(f"[--eval-only] Loading checkpoint from {eval_path} ...")
        ckpt = torch.load(eval_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        set_active_checkpoint_threshold(ckpt)
        eval_threshold = ACTIVE_PEAK_THRESHOLD
        print(f"[--eval-only] Checkpoint loaded. Best Thresh: {eval_threshold}. Skipping training.")
    else:
        print("[--eval-only] ERROR: No checkpoint found! Train the model first.")
        return

    print("\n--- Sweeping Hyperparameters on Validation Set ---")
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    min_distances = [3, 4, 5, 6, 7]
    
    best_f1_sweep = -1
    best_thresh_sweep = 0.2
    best_min_dist_sweep = 4
    
    for t in thresholds:
        for md in min_distances:
            mae, f1, prec, rec = evaluate_model(model, valid_loader, DEVICE, threshold=t, min_dist=md, use_tta=False, verbose=False)
            print(f"Thresh: {t:.2f} | MinDist: {md} --> F1: {f1:.4f} (P: {prec:.4f}, R: {rec:.4f})")
            if f1 > best_f1_sweep:
                best_f1_sweep = f1
                best_thresh_sweep = t
                best_min_dist_sweep = md

    print(f"\nOptimal Hyperparameters: Threshold = {best_thresh_sweep}, Min Distance = {best_min_dist_sweep} (Val F1 = {best_f1_sweep:.4f})")
    print("\n--- Final Test Set Evaluation ---")
    test_results = evaluate_model(model, test_loader, DEVICE, threshold=best_thresh_sweep, min_dist=best_min_dist_sweep, use_tta=USE_TTA_FOR_TEST, verbose=True)
    visualize_prediction(model, test_dataset, idx=0, save_path=os.path.join(CHECKPOINT_DIR, "prediction_sample.png"))
    return test_results
# ──────────────────────────────────────────────────────────────────────

UNFREEZE_EPOCH = 5
for epoch in range(start_epoch, EPOCHS):
    if epoch == UNFREEZE_EPOCH:
        print(f"\n[Epoch {epoch+1}] Backbone fully unfrozen.")
        for param in model.encoder.parameters():
            param.requires_grad = True

    # Linear Warmup
    if epoch < 3:
        warmup_factor = (epoch + 1) / 3
        for param_group in optimizer.param_groups:
            if param_group.get('is_backbone', False):
                param_group['lr'] = 1e-5 * warmup_factor
            else:
                param_group['lr'] = LR * warmup_factor

    train_stats = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler)
    val_stats = validate(model, valid_loader, criterion, DEVICE)

    if epoch >= 3:
        scheduler.step()
    
    # Determine current LR to log (prefer head LR)
    current_lr = LR
    for group in optimizer.param_groups:
        if not group.get('is_backbone', False):
            current_lr = group['lr']
            break

    history["train_loss"].append(train_stats["loss"])
    history["val_loss"].append(val_stats["loss"])
    history["mae"].append(val_stats["MAE"])
    history["rmse"].append(val_stats["RMSE"])
    history["lr"].append(current_lr)

    vs = val_stats   # shorthand

    # Sample GT vs predicted counts
    batch_val = next(iter(valid_loader))
    images_val = batch_val["image"].to(DEVICE)
    with torch.no_grad():
        amp_dtype = torch.bfloat16 if (DEVICE.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
        with torch.amp.autocast(device_type=DEVICE.type, dtype=amp_dtype, enabled=(DEVICE.type == 'cuda')):
            out = model(images_val)
    n_show = min(8, images_val.size(0))
    pred_s = (out["density"].sum(dim=(1, 2, 3)) / DENSITY_SCALE).cpu().numpy()[:n_show]
    gt_s   = batch_val["count"][:n_show].numpy()
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] Epoch {epoch + 1}/{EPOCHS} | "
          f"Train Loss={train_stats['loss']:.3f} (D:{train_stats['density_loss']:.3f} H:{train_stats['heatmap_loss']:.3f} C:{train_stats['count_loss']:.3f} O:{train_stats['offset_loss']:.3f}) | "
          f"Val Loss={vs['loss']:.3f} | "
          f"MAE={vs['MAE']:.1f} | "
          f"RMSE={vs['RMSE']:.1f} | "
          f"Prec={vs['precision']:.3f} | "
          f"Rec={vs['recall']:.3f} | "
          f"F1={vs['f1']:.3f} (th={vs['best_thresh']:.2f}) | "
          f"CountRatio={vs['count_ratio']:.2f} | "
          f"PeakRatio={vs['peak_ratio']:.2f} | "
          f"LR={current_lr:.2e}")
    print(f"Sample GT   : {np.round(gt_s, 1)}")
    print(f"Sample Pred : {np.round(pred_s, 1)}")

    # The diagnostic forward pass is not needed after logging.  Releasing
    # it here prevents its full-resolution output tensors surviving into
    # the following training epoch.
    del out, images_val, batch_val, pred_s, gt_s
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # ── Checkpoint payload shared by both saves ────────────────────────
    config_dict = {
        "BATCH_SIZE": BATCH_SIZE,
        "OUTPUT_SIZE": OUTPUT_SIZE,
        "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
        "HEATMAP_WEIGHT": HEATMAP_WEIGHT,
        "MIN_SIGMA": MIN_SIGMA,
        "MAX_SIGMA": MAX_SIGMA,
        "HEATMAP_MIN_SIGMA": HEATMAP_MIN_SIGMA,
        "HEATMAP_MAX_SIGMA": HEATMAP_MAX_SIGMA,
        "MIN_DISTANCE": MIN_DISTANCE,
        "BORDER_MASK": BORDER_MASK,
    }
    
    ckpt_payload = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_combined_score": best_combined_score,
        "best_mae": best_mae,
        "best_f1": best_f1,
        "best_pareto_mae": best_pareto_mae,
        "best_pareto_f1": best_pareto_f1,
        "best_conf_threshold": best_conf_threshold,
        "epochs_no_improve": epochs_no_improve,
        "history": history,
        "config": config_dict,
    }

    # ── Always save latest (interrupt-safe resume) ─────────────────────
    torch.save(ckpt_payload, LATEST_SAVE_PATH)

    # ── ROBUST BEST-MODEL SELECTION ─
    regression_score, localization_score = compute_task_scores(vs)

    reg_weight, loc_weight = compute_adaptive_task_weights(
        regression_score,
        localization_score
    )

    overall_score = compute_overall_score(
        regression_score,
        localization_score,
        reg_weight,
        loc_weight
    )

    pareto_good = pareto_improved(
        current_mae=vs["MAE"],
        current_f1=vs["f1"],
        best_mae=best_pareto_mae,
        best_f1=best_pareto_f1
    )

    guardrails_ok = passes_metric_guardrails(
        current_mae=vs["MAE"],
        current_f1=vs["f1"],
        best_mae=best_mae,
        best_f1=best_f1
    )

    score_improved = (overall_score > best_combined_score + IMPROVEMENT_EPS)
    should_save = guardrails_ok and (score_improved or pareto_good)

    print(f"  Regression Score : {regression_score:.4f}")
    print(f"  Localization Score : {localization_score:.4f}")
    print(f"  Adaptive Weights -> Regression={reg_weight:.3f}, Localization={loc_weight:.3f}")
    print(f"  Overall Score : {overall_score:.4f}")
    print(f"  Pareto Improvement : {pareto_good}")
    print(f"  Guardrails : {guardrails_ok}")

    # Best MAE and Best F1 diagnostic checkpoints
    if vs["MAE"] < best_mae:
        ckpt_payload["best_mae"] = vs["MAE"]
        torch.save(ckpt_payload, BEST_MAE_SAVE_PATH)
    if vs["f1"] > best_f1:
        ckpt_payload["best_f1"] = vs["f1"]
        torch.save(ckpt_payload, BEST_F1_SAVE_PATH)

    if should_save:
        best_combined_score = overall_score
        best_pareto_mae = min(best_pareto_mae, vs["MAE"])
        best_pareto_f1 = max(best_pareto_f1, vs["f1"])
        
        best_mae = vs["MAE"]
        best_f1 = vs["f1"]
        best_conf_threshold = vs["best_thresh"]

        ckpt_payload.update({
            "best_combined_score": best_combined_score,
            "regression_score": regression_score,
            "localization_score": localization_score,
            "regression_weight": reg_weight,
            "localization_weight": loc_weight,
            "best_mae": best_mae,
            "best_f1": best_f1,
            "best_conf_threshold": best_conf_threshold,
            "best_pareto_mae": best_pareto_mae,
            "best_pareto_f1": best_pareto_f1,
        })

        torch.save(ckpt_payload, BEST_BALANCED_SAVE_PATH)
        torch.save(ckpt_payload, LATEST_SAVE_PATH)

        print("\n  [+] *** BEST BALANCED MODEL SAVED ***")
        print(f"      Score : {overall_score:.4f}")
        print(f"      MAE   : {vs['MAE']:.3f}")
        print(f"      F1    : {vs['f1']:.4f}")

        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"  [-] Balanced model not improved. Patience: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\n[Early Stopping] No improvement for {EARLY_STOPPING_PATIENCE} epochs. Stopping at Epoch {epoch + 1}.")
            print(f"  Saved {EPOCHS - (epoch + 1)} epochs of compute.")
            break

    # ============================================================================
    # ADAPT NEXT EPOCH'S TRAINING EMPHASIS
    # ============================================================================
    criterion.dw = DENSITY_WEIGHT * reg_weight
    criterion.cw = COUNT_WEIGHT * reg_weight
    criterion.hw = HEATMAP_WEIGHT * loc_weight
    criterion.ow = 0.03 * loc_weight

    print(f"  Next Epoch Loss Weights -> D={criterion.dw:.3f}, C={criterion.cw:.3f}, H={criterion.hw:.3f}, O={criterion.ow:.4f}")

print("\n" + "=" * 70)
print("Training Finished.")
print(f"Best Validation Score: {best_combined_score:.3f} | MAE: {best_mae:.3f} | F1: {best_f1:.3f}")
print("=" * 70)

# Final Evaluation on Test Set
if os.path.exists(BEST_F1_SAVE_PATH):
    print(f"\nLoading best checkpoint from {BEST_F1_SAVE_PATH} for final evaluation...")
    ckpt = torch.load(BEST_F1_SAVE_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    set_active_checkpoint_threshold(ckpt)

print("\n--- Sweeping Hyperparameters on Validation Set ---")
thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
min_distances = [3, 4, 5, 6, 7]

best_f1_sweep = -1
best_thresh_sweep = 0.2
best_min_dist_sweep = 4

for t in thresholds:
    for md in min_distances:
        mae, f1, prec, rec = evaluate_model(model, valid_loader, DEVICE, threshold=t, min_dist=md, use_tta=False, verbose=False)
        print(f"Thresh: {t:.2f} | MinDist: {md} --> F1: {f1:.4f} (P: {prec:.4f}, R: {rec:.4f})")
        if f1 > best_f1_sweep:
            best_f1_sweep = f1
            best_thresh_sweep = t
            best_min_dist_sweep = md

print(f"\nOptimal Hyperparameters: Threshold = {best_thresh_sweep}, Min Distance = {best_min_dist_sweep} (Val F1 = {best_f1_sweep:.4f})")
set_active_checkpoint_threshold({"best_conf_threshold": best_thresh_sweep})
print("\n--- Final Test Set Evaluation ---")
test_results = evaluate_model(model, test_loader, DEVICE, threshold=best_thresh_sweep, min_dist=best_min_dist_sweep, use_tta=USE_TTA_FOR_TEST, verbose=True)
visualize_prediction(model, test_dataset, idx=0, save_path=os.path.join(CHECKPOINT_DIR, "prediction_sample.png"))

return test_results

Execute training and evaluation when script is run

if name == "main":
import argparse
parser = argparse.ArgumentParser(description="DDPF-Net v6 Training & Evaluation")
parser.add_argument("--eval-only", action="store_true", help="Skip training and run final test evaluation on the saved best checkpoint")
args = parser.parse_args()
run_training_and_eval(eval_only=args.eval_only)
