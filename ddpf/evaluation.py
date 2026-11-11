import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import config as cfg
from utils import extract_points, compute_sigmas, localization_metrics_adaptive, r2_score, explained_variance_score, pearsonr

def apply_offset_flip_correction(offset, mode):
    if mode == "hflip":
        offset = offset.clone()
        offset[:, 0] = -offset[:, 0]
    elif mode == "vflip":
        offset = offset.clone()
        offset[:, 1] = -offset[:, 1]
    return offset

@torch.no_grad()
def run_tta_forward(model, images, device, amp_dtype):
    modes = ["orig", "hflip"]
    aug_imgs = {
        "orig": images,
        "hflip": torch.flip(images, dims=[3]),
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
        densities.append(d); heatmaps.append(h); offsets.append(o)

    density = torch.stack(densities).mean(dim=0)
    heatmap = torch.stack(heatmaps).mean(dim=0)
    offset  = torch.stack(offsets).mean(dim=0).cpu().numpy()
    return density, heatmap, offset

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = running_density = running_heatmap = running_count_l = running_consistency = running_offset = 0.0

    all_gt_counts = []
    all_pred_counts = []
    all_density_mean = []
    all_density_max = []
    all_density_ratio = []

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

        pred_den = outputs["density"].detach()
        pred_cnt = (pred_den.sum(dim=(1, 2, 3)) / cfg.DENSITY_SCALE).cpu().numpy()
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
                gt_pts[:, 0] *= cfg.OUTPUT_SIZE / cfg.IMAGE_SIZE
                gt_pts[:, 1] *= cfg.OUTPUT_SIZE / cfg.IMAGE_SIZE
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

    best_f1 = -1.0
    best_metrics = {}
    metrics_at_fixed_thresh = {}
    
    for th in cfg.PEAK_THRESHOLD_SWEEP:
        g_TP = g_FP = g_FN = 0
        all_peaks = []
        for i in range(len(cached_heats)):
            pts = extract_points(cached_heats[i], offset_map=cached_offsets[i], threshold=th, min_distance=cfg.MIN_DISTANCE, border_mask=cfg.BORDER_MASK)
            all_peaks.append(len(pts))
            gt_pts = cached_gt_pts[i]
            
            gt_sigmas = np.zeros(len(gt_pts))
            if len(gt_pts) > 0:
                gt_sigmas = compute_sigmas(gt_pts, min_sigma=cfg.HEATMAP_MIN_SIGMA, max_sigma=cfg.HEATMAP_MAX_SIGMA, sigma_scale=cfg.HEATMAP_SIGMA_SCALE)
                
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
    pearson_val, _ = pearsonr(gt_arr, pred_arr) if len(gt_arr) > 1 else (0.0, 0.0)

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
        "Pearson": float(pearson_val),
        "sample_gt": gt_arr[:8] if len(gt_arr) > 0 else np.array([]),
        "sample_pred": pred_arr[:8] if len(pred_arr) > 0 else np.array([]),
    }
    metrics.update(best_metrics)
    metrics.update(metrics_at_fixed_thresh)
    if running_consistency > 0:
        metrics["consistency_loss"] = running_consistency / n_b
        
    return metrics

@torch.no_grad()
def evaluate_model(model, loader, device, threshold=None, min_dist=None, use_tta=None, verbose=True):
    model.eval()
    threshold = cfg.ACTIVE_PEAK_THRESHOLD if threshold is None else threshold
    min_dist = cfg.MIN_DISTANCE if min_dist is None else min_dist
    use_tta = cfg.USE_TTA_FOR_TEST if use_tta is None else use_tta

    gt_counts = []
    pred_counts = []
    global_TP = global_FP = global_FN = 0

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

        pred = (density.sum(dim=(1, 2, 3)) / cfg.DENSITY_SCALE).cpu().numpy()
        gt = batch["count"].numpy()

        pred_counts.extend(pred.tolist())
        gt_counts.extend(gt.tolist())

        for i in range(images.size(0)):
            pred_pts = extract_points(heatmap[i, 0].cpu().numpy(), offset_map=offset_map[i], threshold=threshold, min_distance=min_dist, border_mask=cfg.BORDER_MASK)
            gt_pts = batch["points"][i].copy() if len(batch["points"][i]) > 0 else np.zeros((0, 2))

            gt_sigmas = np.zeros(len(gt_pts))
            if len(gt_pts) > 0:
                gt_pts[:, 0] *= cfg.OUTPUT_SIZE / cfg.IMAGE_SIZE
                gt_pts[:, 1] *= cfg.OUTPUT_SIZE / cfg.IMAGE_SIZE
                gt_sigmas = compute_sigmas(gt_pts, min_sigma=cfg.HEATMAP_MIN_SIGMA, max_sigma=cfg.HEATMAP_MAX_SIGMA, sigma_scale=cfg.HEATMAP_SIGMA_SCALE)

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
    pearson_val, _ = pearsonr(gt_counts, pred_counts) if len(gt_counts) > 1 else (0.0, 0.0)

    precision = global_TP / (global_TP + global_FP + 1e-8)
    recall    = global_TP / (global_TP + global_FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    if verbose:
        print("=" * 60)
        print("Regression Metrics")
        print(f"MAE                 : {mae:.3f}")
        print(f"RMSE                : {rmse:.3f}")
        print(f"MAPE                : {mape:.2f}%")
        print(f"R2 Score            : {r2:.4f}")
        print(f"Explained Variance  : {explained:.4f}")
        print(f"Pearson Correlation : {pearson_val:.4f}")
        print("=" * 60)
        print("Localization Metrics")
        print(f"Precision           : {precision:.4f}")
        print(f"Recall              : {recall:.4f}")
        print(f"F1 Score            : {f1:.4f}")

        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
        plt.figure(figsize=(6, 6))
        plt.scatter(gt_counts, pred_counts, alpha=0.6, color='#2196F3')
        m = max(gt_counts.max(), pred_counts.max()) if len(gt_counts) > 0 else 100
        plt.plot([0, m], [0, m], 'r--', linewidth=2)
        plt.xlabel("Ground Truth Count")
        plt.ylabel("Predicted Count")
        plt.title("GT vs Predicted Count")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.CHECKPOINT_DIR, "scatter_gt_vs_pred.png"), dpi=150)
        plt.close()

        errors = pred_counts - gt_counts
        plt.figure(figsize=(6, 4))
        plt.hist(errors, bins=30, color='#FF9800', edgecolor='k', alpha=0.7)
        plt.xlabel("Prediction Error")
        plt.ylabel("Frequency")
        plt.title("Error Distribution")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.CHECKPOINT_DIR, "error_histogram.png"), dpi=150)
        plt.close()

    return mae, f1, precision, recall

@torch.no_grad()
def visualize_prediction(model, dataset, idx=0, save_path=None):
    model.eval()
    sample = dataset[idx]
    image = sample["image"].unsqueeze(0).to(cfg.DEVICE)
    amp_dtype = torch.bfloat16 if (cfg.DEVICE.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16

    with torch.amp.autocast(device_type=cfg.DEVICE.type, dtype=amp_dtype, enabled=(cfg.DEVICE.type == 'cuda')):
        output = model(image)

    density = output["density"][0, 0].float().cpu().numpy()
    heatmap = torch.sigmoid(output["heatmap_logits"].float())[0, 0].cpu().numpy()
    offset_map = output["offset"][0].detach().float().cpu().numpy()

    pred_count = density.sum() / cfg.DENSITY_SCALE
    pred_points = extract_points(heatmap, offset_map=offset_map, threshold=cfg.ACTIVE_PEAK_THRESHOLD, min_distance=cfg.MIN_DISTANCE, border_mask=cfg.BORDER_MASK)

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
