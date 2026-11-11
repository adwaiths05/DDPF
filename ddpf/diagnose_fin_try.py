"""Inference-only diagnostics for a supplied DDPF-Net checkpoint.

This script intentionally does not call the training entry point. It writes
diagnostics to the requested directory and final-test figures beside it.
"""
import csv
import copy
import json
import os
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.ndimage import maximum_filter
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

import config as cfg
import utils as ut
import dataset as ds
from models import DDPFNet

CHECKPOINT = Path("fin/best_balanced_ddpfnet_final_try.pth")
OUT_DIR = Path("fin/diagnostics_best_balanced_2")
TARGET_P = 0.75
TARGET_R = 0.70
TARGET_F1 = 0.72
THRESHOLDS = [round(x, 3) for x in np.arange(0.03, 0.301, 0.01)]
RANKED_THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
MIN_DISTANCES = [2, 3, 4, 5, 6, 7, 8, 10]
CANDIDATE_DISTANCES = [2, 3, 4]
FINAL_DISTANCES = [2.0, 2.5, 3.0]
COUNT_SCALES = [0.80, 0.85, 0.90, 0.95, 1.00]
COUNT_BIASES = [-5, 0, 3, 5]


def match_points(pred_points, gt_points, gt_sigmas):
    """Same greedy adaptive matching as the codebase, plus indices for plots."""
    if not len(pred_points) or not len(gt_points):
        return [], list(range(len(pred_points))), list(range(len(gt_points)))
    distances = cdist(np.asarray(pred_points), np.asarray(gt_points))
    radii = np.maximum(4.0, 1.5 * gt_sigmas)
    matches, used_p, used_g = [], set(), set()
    while True:
        p, g = np.unravel_index(np.argmin(distances), distances.shape)
        if distances[p, g] > radii[g]:
            break
        if p in used_p:
            distances[p, :] = 1e9
            continue
        if g in used_g:
            distances[:, g] = 1e9
            continue
        matches.append((p, g))
        used_p.add(p); used_g.add(g)
        distances[p, :] = 1e9; distances[:, g] = 1e9
    return matches, [p for p in range(len(pred_points)) if p not in used_p], [g for g in range(len(gt_points)) if g not in used_g]


def build_ranked_candidates(heatmap, offset_map=None, candidate_distance=3,
                            threshold_min=0.0, border_mask=12):
    heatmap = np.asarray(heatmap).squeeze().copy()
    if border_mask > 0:
        heatmap[:border_mask, :] = 0
        heatmap[-border_mask:, :] = 0
        heatmap[:, :border_mask] = 0
        heatmap[:, -border_mask:] = 0
    local_max = maximum_filter(heatmap, size=candidate_distance)
    ys, xs = np.where((heatmap == local_max) & (heatmap > threshold_min))
    candidates = []
    for x, y in zip(xs, ys):
        dx = dy = 0.0
        if offset_map is not None:
            dx = np.clip(offset_map[0, y, x], -0.5, 0.5)
            dy = np.clip(offset_map[1, y, x], -0.5, 0.5)
        candidates.append((float(heatmap[y, x]), x + dx, y + dy))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def ranked_points_from_candidates(candidates, threshold=0.05,
                                  final_distance=2.5, max_points=None):
    """Greedy score-ordered NMS with an exact spatial-neighbour lookup.

    The previous implementation rebuilt an array of every accepted point for
    every candidate.  Dense heatmaps therefore made ranked NMS quadratic and
    could appear to hang.  Bucketing accepted points by ``final_distance``
    limits each candidate to checking only adjacent cells while retaining the
    same greedy selection rule and output order.
    """
    if final_distance <= 0:
        raise ValueError("final_distance must be positive")

    selected = []
    cell_size = float(final_distance)
    min_distance_sq = cell_size * cell_size
    spatial_cells = {}
    for score, px, py in candidates:
        if score <= threshold:
            break

        cell_x = int(np.floor(px / cell_size))
        cell_y = int(np.floor(py / cell_size))
        is_suppressed = False
        for neighbour_y in range(cell_y - 1, cell_y + 2):
            for neighbour_x in range(cell_x - 1, cell_x + 2):
                for selected_x, selected_y in spatial_cells.get((neighbour_x, neighbour_y), ()):
                    dx = selected_x - px
                    dy = selected_y - py
                    if dx * dx + dy * dy < min_distance_sq:
                        is_suppressed = True
                        break
                if is_suppressed:
                    break
            if is_suppressed:
                break
        if is_suppressed:
            continue

        selected.append((score, px, py))
        spatial_cells.setdefault((cell_x, cell_y), []).append((px, py))
        if max_points is not None and len(selected) >= max_points:
            break
    return [(x, y) for _, x, y in selected]


def extract_points_ranked(heatmap, offset_map=None, threshold=0.05,
                          candidate_distance=3, final_distance=2.5,
                          border_mask=12, max_points=None):
    candidates = build_ranked_candidates(
        heatmap, offset_map, candidate_distance, threshold, border_mask
    )
    return ranked_points_from_candidates(candidates, threshold, final_distance, max_points)


def select_by_count_budget(points, density_count, scale, bias):
    target_k = max(1, int(round(scale * density_count + bias)))
    return list(points[:target_k])


def metrics_from_point_sets(records, point_sets, extractor="standard",
                            threshold=None, min_distance=None,
                            candidate_distance=None, final_distance=None,
                            count_scale=None, count_bias=0):
    tp = fp = fn = 0
    peak_counts = []
    per_image = []
    for rec, points in zip(records, point_sets):
        pts = points
        if count_scale is not None:
            pts = select_by_count_budget(pts, rec["pred_count"], count_scale, count_bias)
        matches, unmatched_p, unmatched_g = match_points(pts, rec["gt_points"], rec["gt_sigmas"])
        p, f, n = len(matches), len(unmatched_p), len(unmatched_g)
        tp += p
        fp += f
        fn += n
        peak_counts.append(len(pts))
        per_image.append((pts, matches, unmatched_p, unmatched_g))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"extractor": extractor, "threshold": threshold, "min_distance": min_distance,
            "candidate_distance": candidate_distance, "final_distance": final_distance,
            "count_scale": count_scale, "count_bias": count_bias,
            "precision": precision, "recall": recall, "f1": f1,
            "TP": tp, "FP": fp, "FN": fn,
            "avg_predicted_peaks": float(np.mean(peak_counts)),
            "peak_ratio": float(np.sum(peak_counts) / max(sum(len(r["gt_points"]) for r in records), 1)),
            "per_image": per_image}


def metrics(records, threshold, min_distance, extractor="standard",
            candidate_distance=None, final_distance=None,
            count_scale=None, count_bias=0):
    point_sets = []
    for rec in records:
        if extractor == "ranked":
            pts = extract_points_ranked(
                rec["heatmap"], rec["offset"], threshold,
                candidate_distance or min_distance,
                final_distance or float(min_distance), cfg.BORDER_MASK
            )
        else:
            pts = ut.extract_points(rec["heatmap"], rec["offset"], threshold, min_distance, cfg.BORDER_MASK)
        point_sets.append(pts)
    return metrics_from_point_sets(
        records, point_sets, extractor, threshold, min_distance,
        candidate_distance, final_distance, count_scale, count_bias
    )


def select_best(sweep):
    all_loc_ok = [x for x in sweep if x["precision"] >= TARGET_P and x["recall"] >= TARGET_R and x["f1"] >= TARGET_F1]
    both_pr_ok = [x for x in sweep if x["precision"] >= TARGET_P and x["recall"] >= TARGET_R]
    precision_ok = [x for x in sweep if x["precision"] >= TARGET_P]
    if all_loc_ok:
        return max(all_loc_ok, key=lambda x: (x["f1"], x["recall"]))
    if both_pr_ok:
        return max(both_pr_ok, key=lambda x: (x["f1"], x["recall"]))
    if precision_ok:
        return max(precision_ok, key=lambda x: (x["recall"], x["f1"]))
    def gap(x):
        return (
            max(0.0, TARGET_P - x["precision"])
            + max(0.0, TARGET_R - x["recall"])
            + max(0.0, TARGET_F1 - x["f1"])
        )
    return min(sweep, key=gap)


def fit_linear_calibration(records):
    raw = np.asarray([r["pred_count"] for r in records], dtype=float)
    gt = np.asarray([r["gt_count"] for r in records], dtype=float)
    slope, intercept = np.polyfit(raw, gt, 1) if len(raw) > 1 else (1.0, 0.0)
    return {"slope": float(slope), "intercept": float(intercept)}


def regression_metrics(records):
    gt = np.asarray([r["gt_count"] for r in records], dtype=float)
    pred = np.asarray([r["pred_count"] for r in records], dtype=float)
    shared = ut.regression_metrics_from_arrays(gt, pred)
    return {**shared,
            "gt_total_count": float(gt.sum()), "predicted_total_count": float(pred.sum()),
            "count_ratio": float(pred.sum() / max(gt.sum(), 1.0)), "avg_gt_points": float(gt.mean())}


@torch.no_grad()
def cache_predictions(model, loader, dataset, label):
    records, heat_values = [], []
    amp_dtype = torch.bfloat16 if (cfg.DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    for index, batch in enumerate(tqdm(loader, desc=f"Caching {label} predictions")):
        images = batch["image"].to(cfg.DEVICE, non_blocking=True)
        with torch.amp.autocast(device_type=cfg.DEVICE.type, dtype=amp_dtype, enabled=cfg.DEVICE.type == "cuda"):
            output = model(images)
        density = output["density"].float().cpu().numpy()
        heatmap = torch.sigmoid(output["heatmap_logits"].float()).cpu().numpy()
        offsets = output["offset"].float().cpu().numpy()
        for i in range(images.size(0)):
            # DataLoaders used here have shuffle=False, so this remains aligned to dataset.images.
            ds_index = index * loader.batch_size + i
            gt_points = batch["points"][i].copy() if len(batch["points"][i]) else np.zeros((0, 2), dtype=np.float32)
            gt_points *= cfg.OUTPUT_SIZE / cfg.IMAGE_SIZE
            sigmas = ut.compute_sigmas(gt_points, min_sigma=cfg.HEATMAP_MIN_SIGMA,
                                      max_sigma=cfg.HEATMAP_MAX_SIGMA, sigma_scale=cfg.HEATMAP_SIGMA_SCALE)
            records.append({"image_name": os.path.basename(dataset.images[ds_index]), "image": batch["image"][i].numpy(),
                            "gt_count": float(batch["count"][i]), "pred_count": float(density[i, 0].sum() / cfg.DENSITY_SCALE),
                            "gt_points": gt_points, "gt_sigmas": sigmas, "density": density[i, 0],
                            "heatmap": heatmap[i, 0], "offset": offsets[i]})
            heat_values.append(heatmap[i, 0].ravel())
        del images, output, density, heatmap, offsets
    return records, np.concatenate(heat_values)


def save_visualization(rec, result, out_path):
    pts, matches, unmatched_p, unmatched_g = result
    tp, fp, fn = len(matches), len(unmatched_p), len(unmatched_g)
    precision = tp / (tp + fp + 1e-8); recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    image = rec["image"].transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    axes[0].imshow(image); axes[0].scatter(rec["gt_points"][:, 0], rec["gt_points"][:, 1], s=10, c="lime", marker="x")
    axes[0].set_title(f"Original + GT ({int(rec['gt_count'])})")
    axes[1].imshow(rec["density"], cmap="jet"); axes[1].set_title(f"Density count: {rec['pred_count']:.1f}")
    axes[2].imshow(rec["heatmap"], cmap="hot", vmin=0, vmax=1); axes[2].set_title("Heatmap probability")
    axes[3].imshow(image);
    if len(rec["gt_points"]): axes[3].scatter(rec["gt_points"][:, 0], rec["gt_points"][:, 1], s=18, c="lime", marker="x", label="GT")
    if pts: axes[3].scatter(np.asarray(pts)[:, 0], np.asarray(pts)[:, 1], s=12, c="cyan", marker="o", facecolors="none", label="peaks")
    axes[3].legend(loc="upper right"); axes[3].set_title(f"Predicted peaks: {len(pts)}")
    axes[4].axis("off"); axes[4].text(0.03, 0.95, f"{rec['image_name']}\n\nGT count: {rec['gt_count']:.0f}\nDensity count: {rec['pred_count']:.2f}\nPeaks: {len(pts)}\n\nTP: {tp}  FP: {fp}  FN: {fn}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}", va="top", fontsize=12)
    for ax in axes[:4]: ax.axis("off")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def save_final_test_artifacts(records, localization_result, artifact_dir):
    """Save the three requested final-test artifacts beside the checkpoints."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    gt_counts = np.asarray([r["gt_count"] for r in records], dtype=float)
    pred_counts = np.asarray([r["pred_count"] for r in records], dtype=float)

    plt.figure(figsize=(6, 6))
    plt.scatter(gt_counts, pred_counts, alpha=0.65, color="#2196F3")
    limit = max(float(gt_counts.max()), float(pred_counts.max()), 1.0)
    plt.plot([0, limit], [0, limit], "r--", linewidth=1.5, label="perfect prediction")
    plt.xlabel("Ground-truth count"); plt.ylabel("Predicted count")
    plt.title("Final test: ground truth vs prediction"); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(artifact_dir / "scatter_gt_vs_pred.png", dpi=150); plt.close()

    errors = pred_counts - gt_counts
    plt.figure(figsize=(6, 4))
    plt.hist(errors, bins=30, color="#FF9800", edgecolor="black", alpha=0.75)
    plt.axvline(0, color="red", linestyle="--", linewidth=1.2)
    plt.xlabel("Prediction error (predicted − ground truth)"); plt.ylabel("Images")
    plt.title("Final test: prediction-error histogram"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(artifact_dir / "error_histogram.png", dpi=150); plt.close()

    # A median-error test image is more representative than arbitrarily using
    # the first image in the folder.
    sample_index = int(np.argsort(np.abs(errors))[len(errors) // 2])
    save_visualization(records[sample_index], localization_result["per_image"][sample_index],
                       artifact_dir / "prediction_sample.png")


def write_per_image(records, selected):
    with (OUT_DIR / "validation_per_image.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "gt_count", "predicted_count", "peak_count", "TP", "FP", "FN", "precision", "recall", "f1"])
        writer.writeheader()
        for rec, result in zip(records, selected["per_image"]):
            pts, matched, un_p, un_g = result; tp, fp, fn = len(matched), len(un_p), len(un_g)
            p = tp / (tp + fp + 1e-8); r = tp / (tp + fn + 1e-8); f1 = 2*p*r/(p+r+1e-8)
            writer.writerow({"image_name": rec["image_name"], "gt_count": rec["gt_count"], "predicted_count": rec["pred_count"], "peak_count": len(pts), "TP": tp, "FP": fp, "FN": fn, "precision": p, "recall": r, "f1": f1})


def mark_targets(row):
    row = dict(row)
    row["target_p_ok"] = row["precision"] >= TARGET_P
    row["target_r_ok"] = row["recall"] >= TARGET_R
    row["target_f1_ok"] = row["f1"] >= TARGET_F1
    row["all_loc_targets_ok"] = row["target_p_ok"] and row["target_r_ok"] and row["target_f1_ok"]
    return row


def main(checkpoint_path=CHECKPOINT, output_dir=OUT_DIR):
    global CHECKPOINT, OUT_DIR
    CHECKPOINT, OUT_DIR = Path(checkpoint_path), Path(output_dir)
    if not CHECKPOINT.is_file(): raise FileNotFoundError(CHECKPOINT)
    # Diagnostics may be re-run after an interrupted evaluation.  Reuse the
    # requested output directory and overwrite the deterministic report files.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "validation_visualizations").mkdir(exist_ok=True)
    (OUT_DIR / "failure_cases").mkdir(exist_ok=True)
    checkpoint = torch.load(CHECKPOINT, map_location=cfg.DEVICE, weights_only=False)
    print(f"Checkpoint: {CHECKPOINT}\nEpoch: {checkpoint.get('epoch', 'unknown')}\nBest MAE: {checkpoint.get('best_mae', 'unknown')}\nBest F1: {checkpoint.get('best_f1', 'unknown')}\nBest threshold: {checkpoint.get('best_conf_threshold', 'not stored')}\nConfig: {checkpoint.get('config', 'not stored')}")
    (train_loader, val_selection_loader, test_loader,
     _, val_selection_dataset, test_dataset) = ds.build_dataloaders()
    val_calibration_loader, val_calibration_dataset = val_selection_loader, val_selection_dataset
    del train_loader
    model = DDPFNet(pretrained=False, shared_channels=256).to(cfg.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False); model.eval()
    print("Checkpoint loaded successfully into DDPFNet.")
    records, heat_values = cache_predictions(model, val_selection_loader, val_selection_dataset, "selection validation")
    calibration_records, _ = cache_predictions(model, val_calibration_loader, val_calibration_dataset, "calibration validation")
    base = metrics(records, cfg.PEAK_THRESHOLD, cfg.MIN_DISTANCE); reg = regression_metrics(records)
    standard_configs = [(t, d) for t in THRESHOLDS for d in MIN_DISTANCES]
    standard_sweep = [
        metrics(records, t, d)
        for t, d in tqdm(standard_configs, desc="Standard NMS", unit="config")
    ]

    ranked_configs = [
        (t, cd, fd)
        for t in RANKED_THRESHOLDS
        for cd in CANDIDATE_DISTANCES
        for fd in FINAL_DISTANCES
    ]
    ranked_cache = {}
    candidate_cache = {
        candidate_distance: [
            build_ranked_candidates(
                rec["heatmap"], rec["offset"], candidate_distance,
                RANKED_THRESHOLDS[0], cfg.BORDER_MASK
            )
            for rec in records
        ]
        for candidate_distance in CANDIDATE_DISTANCES
    }
    ranked_sweep = []
    for threshold, candidate_distance, final_distance in tqdm(
        ranked_configs, desc="Ranked NMS", unit="config"
    ):
        candidate_sets = candidate_cache[candidate_distance]
        point_sets = [
            ranked_points_from_candidates(candidates, threshold, final_distance)
            for candidates in candidate_sets
        ]
        config_key = (threshold, candidate_distance, final_distance)
        ranked_cache[config_key] = point_sets
        ranked_sweep.append(metrics_from_point_sets(
            records, point_sets, extractor="ranked", threshold=threshold,
            min_distance=candidate_distance, candidate_distance=candidate_distance,
            final_distance=final_distance
        ))

    ranked_top3 = sorted(
        ranked_sweep,
        key=lambda row: (
            row["precision"] >= TARGET_P and row["recall"] >= TARGET_R and row["f1"] >= TARGET_F1,
            row["precision"] >= TARGET_P and row["recall"] >= TARGET_R,
            row["precision"] >= TARGET_P,
            row["f1"], row["recall"]
        ),
        reverse=True
    )[:3]
    count_configs = [
        (ranked_row, scale, bias)
        for ranked_row in ranked_top3
        for scale in COUNT_SCALES
        for bias in COUNT_BIASES
    ]
    count_sweep = []
    for ranked_row, scale, bias in tqdm(
        count_configs, desc="Count budget (top 3 ranked)", unit="config"
    ):
        config_key = (
            ranked_row["threshold"], ranked_row["candidate_distance"],
            ranked_row["final_distance"]
        )
        count_sweep.append(metrics_from_point_sets(
            records, ranked_cache[config_key], extractor="ranked",
            threshold=ranked_row["threshold"],
            min_distance=ranked_row["min_distance"],
            candidate_distance=ranked_row["candidate_distance"],
            final_distance=ranked_row["final_distance"],
            count_scale=scale, count_bias=bias
        ))
    sweep = standard_sweep + ranked_sweep + count_sweep
    with (OUT_DIR / "validation_threshold_nms_sweep.csv").open("w", newline="") as f:
        rows = [mark_targets({k: v for k, v in row.items() if k != "per_image"}) for row in sweep]
        fieldnames = ["extractor", "threshold", "min_distance", "candidate_distance", "final_distance", "count_scale", "count_bias", "precision", "recall", "f1", "TP", "FP", "FN", "avg_predicted_peaks", "peak_ratio", "target_p_ok", "target_r_ok", "target_f1_ok", "all_loc_targets_ok"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
    best = select_best(sweep)
    precision_ok = [x for x in sweep if x["precision"] >= TARGET_P]
    recall_ok = [x for x in sweep if x["recall"] >= TARGET_R]
    f1_ok = [x for x in sweep if x["f1"] >= TARGET_F1]
    all_loc_ok = [x for x in sweep if x["precision"] >= TARGET_P and x["recall"] >= TARGET_R and x["f1"] >= TARGET_F1]
    best_precision = max(precision_ok, key=lambda x: (x["recall"], x["f1"])) if precision_ok else None
    best_recall = max(recall_ok, key=lambda x: (x["precision"], x["f1"])) if recall_ok else None
    best_f1 = max(f1_ok, key=lambda x: x["f1"]) if f1_ok else None
    best_all_targets = max(all_loc_ok, key=lambda x: x["f1"]) if all_loc_ok else None
    best_ranked = select_best(ranked_sweep)
    best_count = select_best(count_sweep)
    selected = best
    write_per_image(records, selected)
    for i, (rec, result) in enumerate(zip(records[:20], selected["per_image"])): save_visualization(rec, result, OUT_DIR / "validation_visualizations" / f"{i:02d}_{rec['image_name']}.png")
    # Ranked failure sets; duplicate images are retained in category-specific paths intentionally.
    rankings = {"worst_mae": sorted(range(len(records)), key=lambda i: abs(records[i]["gt_count"]-records[i]["pred_count"]), reverse=True),
                "worst_f1": sorted(range(len(records)), key=lambda i: (2*len(selected['per_image'][i][1])/(2*len(selected['per_image'][i][1])+len(selected['per_image'][i][2])+len(selected['per_image'][i][3])+1e-8))),
                "highest_fp": sorted(range(len(records)), key=lambda i: len(selected['per_image'][i][2]), reverse=True),
                "highest_fn": sorted(range(len(records)), key=lambda i: len(selected['per_image'][i][3]), reverse=True),
                "low_recall": sorted(range(len(records)), key=lambda i: len(selected['per_image'][i][1]) / max(len(records[i]['gt_points']), 1)),
                "high_fp_low_tp": sorted(range(len(records)), key=lambda i: (len(selected['per_image'][i][2]), -len(selected['per_image'][i][1])), reverse=True)}
    for category, indices in rankings.items():
        folder = OUT_DIR / "failure_cases" / category; folder.mkdir(exist_ok=True)
        for rank, i in enumerate(indices[:10], 1): save_visualization(records[i], selected["per_image"][i], folder / f"{rank:02d}_{records[i]['image_name']}.png")
    hist_stats = {"min":float(heat_values.min()), "max":float(heat_values.max()), "mean":float(heat_values.mean()), "median":float(np.median(heat_values)), **{f"p{q}":float(np.percentile(heat_values,q)) for q in [50,75,90,95,99]}}
    plt.figure(figsize=(7,4)); plt.hist(heat_values, bins=100, range=(0,1)); plt.yscale("log"); plt.xlabel("Heatmap probability"); plt.ylabel("pixels (log)"); plt.tight_layout(); plt.savefig(OUT_DIR / "validation_heatmap_histogram.png", dpi=150); plt.close()
    test_records, _ = cache_predictions(model, test_loader, test_dataset, "test")
    test_loc, test_reg_raw = metrics(test_records, selected["threshold"], selected["min_distance"], selected.get("extractor", "standard"), selected.get("candidate_distance"), selected.get("final_distance"), selected.get("count_scale"), selected.get("count_bias", 0)), regression_metrics(test_records)
    calibration_loc = metrics(calibration_records, selected["threshold"], selected["min_distance"], selected.get("extractor", "standard"), selected.get("candidate_distance"), selected.get("final_distance"), selected.get("count_scale"), selected.get("count_bias", 0))
    calibration_reg = regression_metrics(calibration_records)
    calibration = fit_linear_calibration(calibration_records)
    calibrated_test_records = copy.deepcopy(test_records)
    for rec in calibrated_test_records:
        rec["pred_count"] = calibration["slope"] * rec["pred_count"] + calibration["intercept"]
    test_reg_calibrated = regression_metrics(calibrated_test_records)
    report = {"model":str(CHECKPOINT), "checkpoint_epoch":checkpoint.get("epoch"), "checkpoint_best_mae":checkpoint.get("best_mae"), "checkpoint_best_f1":checkpoint.get("best_f1"), "checkpoint_best_threshold":checkpoint.get("best_conf_threshold"), "targets":{"precision":TARGET_P,"recall":TARGET_R,"f1":TARGET_F1}, "baseline_validation":{**reg, **{k:v for k,v in base.items() if k != 'per_image'}}, "calibration_validation":{**calibration_reg, **{k:v for k,v in calibration_loc.items() if k != 'per_image'}}, "best_standard_nms":None if not standard_sweep else {k:v for k,v in select_best(standard_sweep).items() if k != 'per_image'}, "best_ranked_nms":{k:v for k,v in mark_targets(best_ranked).items() if k != 'per_image'}, "best_ranked_count_budget":{k:v for k,v in mark_targets(best_count).items() if k != 'per_image'}, "best_validation":{k:v for k,v in mark_targets(selected).items() if k != 'per_image'}, "best_validation_precision_ge_070":None if best_precision is None else {k:v for k,v in mark_targets(best_precision).items() if k != 'per_image'}, "best_validation_recall_ge_065":None if best_recall is None else {k:v for k,v in mark_targets(best_recall).items() if k != 'per_image'}, "best_validation_f1_ge_068":None if best_f1 is None else {k:v for k,v in mark_targets(best_f1).items() if k != 'per_image'}, "best_validation_all_targets":None if best_all_targets is None else {k:v for k,v in mark_targets(best_all_targets).items() if k != 'per_image'}, "heatmap_distribution":hist_stats, "calibration":calibration, "final_test_raw":{**test_reg_raw, **{k:v for k,v in test_loc.items() if k != 'per_image'}}, "final_test_calibrated":test_reg_calibrated}
    (OUT_DIR / "diagnostics_summary.json").write_text(json.dumps(report, indent=2))
    # All three figures use the final-test predictions and the validation-
    # selected threshold/NMS pair now recorded in diagnostics_summary.json.
    save_final_test_artifacts(test_records, test_loc, OUT_DIR.parent)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cached single-forward DDPF-Net diagnostics")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    main(args.checkpoint, args.output_dir)
