import os
import sys
import random
import numpy as np
import torch
import multiprocessing as _mp
import subprocess
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.ndimage import maximum_filter
from sklearn.metrics import r2_score, explained_variance_score
from scipy.stats import pearsonr

import config as cfg

def install_package(package):
    if _mp.current_process().name != 'MainProcess':
        return
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_sigmas(points, min_sigma=2.0, max_sigma=8.0, sigma_scale=0.3):
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

        heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], g)

    return heatmap


def generate_offset_map(image_shape, points):
    h, w = image_shape[:2]
    offset = np.zeros((2, h, w), dtype=np.float32)
    mask = np.zeros((1, h, w), dtype=np.float32)

    for x, y in points:
        ix, iy = int(np.rint(x)), int(np.rint(y))
        if 0 <= ix < w and 0 <= iy < h:
            dx = x - ix
            dy = y - iy
            offset[0, iy, ix] = dx
            offset[1, iy, ix] = dy
            mask[0, iy, ix] = 1.0

    return offset, mask


def generate_center_map(image_shape, points):
    h, w = image_shape[:2]
    centers = np.zeros((h, w), dtype=np.float32)
    for x, y in points:
        ix, iy = int(np.rint(x)), int(np.rint(y))
        if 0 <= ix < w and 0 <= iy < h:
            centers[iy, ix] = 1.0
    return centers

def count_rounded_center_collisions(points):
    if len(points) == 0:
        return 0
    centers = [(int(np.rint(x)), int(np.rint(y))) for x, y in points]
    return len(centers) - len(set(centers))

def count_from_density(density_map):
    return float(density_map.sum())

def extract_points(heatmap, offset_map=None, threshold=None, min_distance=5, border_mask=0):
    if torch.is_tensor(heatmap):
        heatmap = heatmap.detach().float().cpu().numpy()
        
    threshold = cfg.ACTIVE_PEAK_THRESHOLD if threshold is None else threshold
    heatmap = heatmap.squeeze().copy()
    
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
            dx = np.clip(offset_map[0, y, x], -0.5, 0.5)
            dy = np.clip(offset_map[1, y, x], -0.5, 0.5)
            px, py = x + dx, y + dy
            
            if 0 <= px < heatmap.shape[1] and 0 <= py < heatmap.shape[0]:
                points.append((px, py))
        else:
            points.append((x, y))

    return points

def upscale_points(points, input_size=cfg.OUTPUT_SIZE, original_size=cfg.IMAGE_SIZE):
    scale = original_size / input_size
    output = [(x * scale, y * scale) for x, y in points]
    return output

def set_active_checkpoint_threshold(checkpoint):
    cfg.ACTIVE_PEAK_THRESHOLD = float(checkpoint.get("best_conf_threshold", cfg.PEAK_THRESHOLD))
    print(f"[Inference] Active peak threshold: {cfg.ACTIVE_PEAK_THRESHOLD:.2f}")

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

def regression_metrics_from_arrays(gt_counts, pred_counts):
    gt_counts = np.asarray(gt_counts, dtype=float)
    pred_counts = np.asarray(pred_counts, dtype=float)
    errors = pred_counts - gt_counts
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    r2 = float(r2_score(gt_counts, pred_counts)) if len(gt_counts) > 1 else 0.0
    pearson_val, _ = pearsonr(gt_counts, pred_counts) if len(gt_counts) > 1 else (0.0, 0.0)
    return {"MAE": mae, "RMSE": rmse, "R2": float(r2), "Pearson": float(pearson_val)}
