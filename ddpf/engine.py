import os
import gc
import csv
import datetime
import subprocess
import sys
from tqdm.auto import tqdm
import numpy as np

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import config as cfg
from dataset import build_dataloaders
from models import DDPFNet
from losses import DDPFLoss
from evaluation import validate
from utils import set_active_checkpoint_threshold

def _recover_cuda_after_oom(device):
    gc.collect()
    if device.type != "cuda":
        return
    try:
        torch.cuda.synchronize()
    except RuntimeError:
        pass
    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        pass

EPOCH_METRICS_FIELDS = [
    "timestamp", "epoch", "epochs_total",
    "train_loss", "train_density_loss", "train_heatmap_loss", "train_count_loss", "train_offset_loss",
    "val_loss", "MAE", "RMSE", "R2", "Pearson",
    "precision", "recall", "f1", "best_thresh", "count_ratio", "peak_ratio", "lr",
    "sample_gt", "sample_pred",
    "reg_gap", "loc_gap", "gap_mae", "gap_rmse", "gap_r2", "gap_pearson",
    "gap_precision", "gap_recall", "gap_f1",
    "reg_weight", "loc_weight", "weighting_mode", "balanced_score",
    "reg_target_reached", "loc_target_reached", "both_targets_reached",
    "best_model_saved", "epochs_no_improve",
    "next_density_weight", "next_count_weight", "next_heatmap_weight", "next_offset_weight",
    "ema_mae"
]

def log_epoch_metrics_csv(row, csv_path=cfg.EPOCH_METRICS_CSV):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EPOCH_METRICS_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in EPOCH_METRICS_FIELDS})

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = running_density = running_heatmap = running_count = running_consistency = running_offset = 0.0
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
                loss = losses["loss"] / cfg.GRAD_ACCUM_STEPS

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                del images, density, heatmap, center, offset, offset_mask, count, outputs, losses, loss
                continue

            scaler.scale(loss).backward()
        except RuntimeError as exc:
            if device.type != "cuda" or "out of memory" not in str(exc).lower():
                raise
            print(f"  [!] CUDA OOM at training batch {step + 1}; clearing cache and skipping it.")
            optimizer.zero_grad(set_to_none=True)
            del images, density, heatmap, center, offset, offset_mask, count
            if "outputs" in locals(): del outputs
            if "losses" in locals(): del losses
            if "loss" in locals(): del loss
            _recover_cuda_after_oom(device)
            continue

        if (step + 1) % cfg.GRAD_ACCUM_STEPS == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += losses["loss"].item()
        running_density += losses["density_loss"].item()
        running_heatmap += losses["heatmap_loss"].item()
        running_count += losses["count_loss"].item()
        running_consistency += losses["consistency_loss"].item()
        running_offset += losses["offset_loss"].item()

        del images, density, heatmap, center, offset, offset_mask, count, outputs, losses, loss
        if device.type == "cuda" and (step + 1) % 25 == 0:
            _recover_cuda_after_oom(device)

    n = max(len(loader), 1)
    return {
        "loss": running_loss / n,
        "density_loss": running_density / n,
        "heatmap_loss": running_heatmap / n,
        "count_loss": running_count / n,
        "consistency_loss": running_consistency / n,
        "offset_loss": running_offset / n,
    }

def cuda_readiness_check(model, config_dict, train_loader, val_loader, test_loader):
    separator = "=" * 70
    print(separator)
    print("          DDPF-Net -- PRE-TRAINING READINESS CHECK")
    print(separator)

    cuda_available = torch.cuda.is_available()
    print(f"\n[SYSTEM INFO]")
    print(f"   CUDA Available      : {'[+] Yes' if cuda_available else '[-] No'}")
    if cuda_available:
        print(f"   Device Name         : {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"   GPU Memory          : {total_mem:.1f} GB")
    print(f"   PyTorch Version     : {torch.__version__}")
    print(f"   Device Selected     : {cfg.DEVICE}")

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)
    print(f"\n[DATA SUMMARY]")
    print(f"   Dataset             : ShanghaiTech {cfg.DATASET_PART}")
    print(f"   Training Samples    : {n_train}")
    print(f"   Validation Samples  : {n_val}")
    print(f"   Test Samples        : {n_test}")
    print(f"   Train Batches/Epoch : {len(train_loader)}")
    print(f"   Batch Size          : {cfg.BATCH_SIZE}")
    print(f"   Input Resolution    : {cfg.IMAGE_SIZE}x{cfg.IMAGE_SIZE}")
    print(f"   Output Resolution   : {cfg.OUTPUT_SIZE}x{cfg.OUTPUT_SIZE}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[MODEL STATS]")
    print(f"   Architecture        : DDPF-Net (Swin-Tiny + SwinUNetDecoder 512x512 + TTA)")
    print(f"   Total Parameters    : {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Parameter Size      : {total_params * 4 / (1024 ** 2):.1f} MB (float32)")

    print(f"\n[DUMMY FORWARD PASS]")
    model.to(cfg.DEVICE)
    dummy = torch.randn(2, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE).to(cfg.DEVICE)
    try:
        with torch.no_grad():
            amp_dtype = torch.bfloat16 if (cfg.DEVICE.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
            with torch.amp.autocast(device_type=cfg.DEVICE.type, dtype=amp_dtype, enabled=(cfg.DEVICE.type == 'cuda')):
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
    print(f"   Epochs              : {cfg.EPOCHS}")
    print(f"   Learning Rate       : {cfg.LR}")
    print(f"   Weight Decay        : {cfg.WEIGHT_DECAY}")
    print(f"   Gradient Clipping   : {cfg.GRAD_CLIP}")
    print(f"   Loss Weights        : density={cfg.DENSITY_WEIGHT}, heatmap={cfg.HEATMAP_WEIGHT}, count={cfg.COUNT_WEIGHT}, consistency={cfg.CONSISTENCY_WEIGHT}")

    print(f"\n{separator}")
    print(f"   [+] READY TO START TRAINING: YES")
    print(f"{separator}")

    del dummy, out
    if cuda_available:
        torch.cuda.empty_cache()

    return True

def compute_balanced_score(mae, f1, mae_baseline=20.0):
    mae_score = max(0.0, 1.0 - mae / mae_baseline)
    return 0.5 * mae_score + 0.5 * f1

def apply_stage_freeze(model, optimizer=None, stage2=False):
    for module in (model.encoder, model.decoder, model.density_head, model.heatmap_head, model.offset_head):
        for param in module.parameters():
            param.requires_grad = True

    if optimizer is not None:
        for group in optimizer.param_groups:
            params = group.get("params", [])
            if not params:
                continue
            group["base_lr"] = group.get("base_lr", group["lr"])
            if any(not p.requires_grad for p in params):
                group["lr"] = 0.0
            else:
                group["lr"] = group["base_lr"]
    return

def run_training_and_eval():
    print("\n=== FINAL TRAINING CONFIGURATION ===")
    print(f"architecture = Swin-Tiny | output = {cfg.OUTPUT_SIZE}x{cfg.OUTPUT_SIZE} | consistency = {cfg.CONSISTENCY_WEIGHT}")
    print(f"focal alpha = {cfg.FOCAL_ALPHA} | focal gamma = {cfg.FOCAL_GAMMA}")
    print(f"base LR = {cfg.BASE_LR:.1e} | warmup epochs = {cfg.WARMUP_EPOCHS} | min LR = {cfg.MIN_LR:.1e}")
    print(f"epochs = {cfg.EPOCHS} | early-stopping patience = {cfg.EARLY_STOPPING_PATIENCE}")
    print(f"K-fold CV = {cfg.NUM_FOLDS} folds | current fold = {cfg.CURRENT_FOLD}")
    print(f"loss weights: density={cfg.DENSITY_WEIGHT} heatmap={cfg.HEATMAP_WEIGHT} count={cfg.COUNT_WEIGHT} offset=0.03")
    print(f"checkpoint EMA alpha = {cfg.CHECKPOINT_EMA_ALPHA}")
    print(f"TTA for final evaluation = {'ENABLED' if cfg.USE_TTA_FOR_TEST else 'DISABLED'}")
    print(f"output directory = {cfg.CHECKPOINT_DIR}")
    print(f"epoch metrics csv = {cfg.EPOCH_METRICS_CSV}\n")

    print("[INIT] Building dataloaders...", flush=True)
    train_loader, valid_loader, test_loader, train_dataset, val_dataset, test_dataset = build_dataloaders()

    print("\n--- Initialising model, optimiser, scheduler, and loss ---", flush=True)
    model = DDPFNet(pretrained=True, shared_channels=256).to(cfg.DEVICE)

    model.freeze_early_stages()

    no_decay = set()
    import torch.nn as nn
    for mn, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            for pn, p in m.named_parameters():
                no_decay.add(f"{mn}.{pn}" if mn else pn)

    early_stage_params, late_backbone_params, head_params = [], [], []
    for name, param in model.named_parameters():
        param.requires_grad = True
        is_bias_or_norm = name.endswith(".bias") or name in no_decay
        group = {"params": [param], "weight_decay": 0.0 if is_bias_or_norm else cfg.WEIGHT_DECAY}

        if any(name.startswith(f"encoder.backbone.{p}") for p in ["patch_embed", "layers.0", "layers.1"]):
            group["lr"] = 0.0
            group["is_backbone"] = True
            group["is_early_backbone"] = True
            early_stage_params.append(group)
        elif name.startswith("encoder"):
            group["lr"] = 1e-5
            group["is_backbone"] = True
            group["is_early_backbone"] = False
            late_backbone_params.append(group)
        else:
            group["lr"] = cfg.LR
            group["is_backbone"] = False
            group["is_early_backbone"] = False
            head_params.append(group)

    optimizer = AdamW(early_stage_params + late_backbone_params + head_params)
    for group in optimizer.param_groups:
        group["base_lr"] = group["lr"]
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS - cfg.WARMUP_EPOCHS, eta_min=cfg.MIN_LR)
    scaler = torch.amp.GradScaler(enabled=(cfg.DEVICE.type == 'cuda'))
    criterion = DDPFLoss(
        density_weight=cfg.DENSITY_WEIGHT,
        heatmap_weight=cfg.HEATMAP_WEIGHT,
        count_weight=cfg.COUNT_WEIGHT,
        consistency_weight=cfg.CONSISTENCY_WEIGHT,
        focal_alpha=cfg.FOCAL_ALPHA,
        focal_gamma=cfg.FOCAL_GAMMA,
        density_bg_weight=0.30,
        heatmap_bg_weight=0.0,
        offset_weight=0.03,
        center_weight=0.0,
    )

    cuda_readiness_check(model, {}, train_loader, valid_loader, test_loader)

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    best_combined_score = -float("inf")
    best_mae = float("inf")
    best_f1 = 0.0
    ema_mae = None
    best_conf_threshold = cfg.ACTIVE_PEAK_THRESHOLD
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "mae": [], "rmse": [], "lr": []}

    start_epoch = 0
    if os.path.exists(cfg.LATEST_SAVE_PATH):
        print(f"[Resume] Found latest checkpoint: {cfg.LATEST_SAVE_PATH}")
        ckpt = torch.load(cfg.LATEST_SAVE_PATH, map_location=cfg.DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch       = ckpt.get("epoch", 0)
        best_combined_score = ckpt.get("best_combined_score", best_combined_score)
        best_mae          = ckpt.get("best_mae", best_mae)
        best_f1           = ckpt.get("best_f1", best_f1)
        ema_mae           = ckpt.get("ema_mae", ema_mae)
        best_conf_threshold = ckpt.get("best_conf_threshold", best_conf_threshold)
        epochs_no_improve = ckpt.get("epochs_no_improve", epochs_no_improve)
        history           = ckpt.get("history", history)
        set_active_checkpoint_threshold(ckpt)
        print(f"[Resume] Resuming from epoch {start_epoch + 1} | "
              f"best MAE={best_mae:.3f} F1={best_f1:.3f} "
              f"EMA MAE={ema_mae if ema_mae is not None else 'N/A'} "
              f"Patience={epochs_no_improve}/{cfg.EARLY_STOPPING_PATIENCE}")
        if os.path.exists(cfg.EPOCH_METRICS_CSV):
            print(f"[Resume] Appending new epochs to {cfg.EPOCH_METRICS_CSV}")
        del ckpt
    else:
        print("[Fresh Start] Training begins at epoch 1.")

    apply_stage_freeze(model, optimizer, stage2=False)

    UNFREEZE_EPOCH = 5
    if start_epoch >= UNFREEZE_EPOCH:
        for idx, group in enumerate(optimizer.param_groups):
            if group.get("is_early_backbone", False):
                group["lr"] = 1e-5
                group["base_lr"] = 1e-5
                if idx < len(scheduler.base_lrs):
                    scheduler.base_lrs[idx] = 1e-5
        print(f"[Resume] Early backbone already unfrozen at epoch {start_epoch + 1}.")

    for epoch in range(start_epoch, cfg.EPOCHS):
        stage_name = "training"
        criterion.dw = cfg.DENSITY_WEIGHT
        criterion.cw = cfg.COUNT_WEIGHT
        criterion.hw = cfg.HEATMAP_WEIGHT
        criterion.sw = cfg.CONSISTENCY_WEIGHT
        criterion.ow = 0.03

        if epoch == UNFREEZE_EPOCH:
            print(f"\n[Epoch {epoch+1}] Backbone early stages unfrozen and learning-rate restored.")
            for idx, group in enumerate(optimizer.param_groups):
                if group.get("is_early_backbone", False):
                    group["lr"] = 1e-5
                    group["base_lr"] = 1e-5
                    if idx < len(scheduler.base_lrs):
                        scheduler.base_lrs[idx] = 1e-5

        if epoch < cfg.WARMUP_EPOCHS:
            warmup_factor = (epoch + 1) / cfg.WARMUP_EPOCHS
            for param_group in optimizer.param_groups:
                if param_group.get('is_early_backbone', False):
                    param_group['lr'] = 0.0
                elif param_group.get('is_backbone', False):
                    param_group['lr'] = 1e-5 * warmup_factor
                else:
                    param_group['lr'] = cfg.LR * warmup_factor

        train_stats = train_one_epoch(model, train_loader, optimizer, criterion, cfg.DEVICE, scaler)
        val_stats = validate(model, valid_loader, criterion, cfg.DEVICE)

        if epoch >= cfg.WARMUP_EPOCHS:
            scheduler.step()
        
        current_lr = cfg.LR
        for group in optimizer.param_groups:
            if not group.get('is_backbone', False):
                current_lr = group['lr']
                break

        history["train_loss"].append(train_stats["loss"])
        history["val_loss"].append(val_stats["loss"])
        history["mae"].append(val_stats["MAE"])
        history["rmse"].append(val_stats["RMSE"])
        history["lr"].append(current_lr)

        vs = val_stats

        gt_s   = vs.get("sample_gt", np.array([]))
        pred_s = vs.get("sample_pred", np.array([]))
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] Epoch {epoch + 1}/{cfg.EPOCHS} [{stage_name}] | "
              f"Train Loss={train_stats['loss']:.3f} | "
              f"Val Loss={vs['loss']:.3f} | "
              f"MAE={vs['MAE']:.1f} | "
              f"RMSE={vs['RMSE']:.1f} | "
              f"Prec={vs['precision']:.3f} | "
              f"Rec={vs['recall']:.3f} | "
              f"F1={vs['f1']:.3f} (th={vs['best_thresh']:.2f}) | "
              f"LR={current_lr:.2e}")
        sample_gt_str = np.array2string(np.round(gt_s, 1), separator=", ")
        sample_pred_str = np.array2string(np.round(pred_s, 1), separator=", ")

        if cfg.DEVICE.type == "cuda":
            _recover_cuda_after_oom(cfg.DEVICE)

        config_dict = {
            "BATCH_SIZE": cfg.BATCH_SIZE,
            "BASE_LR": cfg.BASE_LR,
            "WARMUP_EPOCHS": cfg.WARMUP_EPOCHS,
            "MIN_LR": cfg.MIN_LR,
            "EPOCHS": cfg.EPOCHS,
            "training_objective_version": cfg.TRAINING_OBJECTIVE_VERSION,
            "OUTPUT_SIZE": cfg.OUTPUT_SIZE,
            "EARLY_STOPPING_PATIENCE": cfg.EARLY_STOPPING_PATIENCE,
            "HEATMAP_WEIGHT": cfg.HEATMAP_WEIGHT,
            "CONSISTENCY_WEIGHT": cfg.CONSISTENCY_WEIGHT,
            "MIN_SIGMA": cfg.MIN_SIGMA,
            "MAX_SIGMA": cfg.MAX_SIGMA,
            "HEATMAP_MIN_SIGMA": cfg.HEATMAP_MIN_SIGMA,
            "HEATMAP_MAX_SIGMA": cfg.HEATMAP_MAX_SIGMA,
            "MIN_DISTANCE": cfg.MIN_DISTANCE,
            "BORDER_MASK": cfg.BORDER_MASK,
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
            "ema_mae": ema_mae,
            "best_conf_threshold": best_conf_threshold,
            "epochs_no_improve": epochs_no_improve,
            "history": history,
            "config": config_dict,
        }

        if ema_mae is None:
            ema_mae = vs["MAE"]
        else:
            ema_mae = cfg.CHECKPOINT_EMA_ALPHA * vs["MAE"] + (1 - cfg.CHECKPOINT_EMA_ALPHA) * ema_mae

        overall_score = compute_balanced_score(ema_mae, vs["f1"])
        score_improved = overall_score > best_combined_score + cfg.IMPROVEMENT_EPS
        ckpt_payload.update({
            "best_combined_score": best_combined_score,
            "best_mae": best_mae,
            "best_f1": best_f1,
            "ema_mae": ema_mae,
            "best_conf_threshold": best_conf_threshold,
        })
        
        print(f"  EMA MAE: {ema_mae:.3f} | Balanced Score: {overall_score:.4f}")

        if vs["MAE"] < best_mae:
            best_mae = vs["MAE"]
            ckpt_payload["best_mae"] = best_mae
            torch.save(ckpt_payload, cfg.BEST_MAE_SAVE_PATH)
        if vs["f1"] > best_f1:
            best_f1 = vs["f1"]
            ckpt_payload["best_f1"] = best_f1
            torch.save(ckpt_payload, cfg.BEST_F1_SAVE_PATH)

        if score_improved:
            best_combined_score = overall_score
            best_conf_threshold = vs["best_thresh"]
            epochs_no_improve = 0

            ckpt_payload.update({
                "best_combined_score": best_combined_score,
                "best_conf_threshold": best_conf_threshold,
                "ema_mae": ema_mae,
                "epochs_no_improve": epochs_no_improve,
            })

            torch.save(ckpt_payload, cfg.BEST_BALANCED_SAVE_PATH)
            torch.save(ckpt_payload, cfg.LATEST_SAVE_PATH)

            print("\n  [+] *** BEST BALANCED MODEL SAVED ***")
            print(f"      Score : {overall_score:.4f}")
            print(f"      MAE   : {vs['MAE']:.3f}")
            print(f"      F1    : {vs['f1']:.4f}")

        else:
            epochs_no_improve += 1
            print(f"  [-] Balanced model not improved. Patience: {epochs_no_improve}/{cfg.EARLY_STOPPING_PATIENCE}")

        # Save only after EMA, best-score and patience state have been updated.
        # This makes a resumed run start from the exact completed epoch.
        ckpt_payload.update({
            "best_combined_score": best_combined_score,
            "best_mae": best_mae,
            "best_f1": best_f1,
            "ema_mae": ema_mae,
            "best_conf_threshold": best_conf_threshold,
            "epochs_no_improve": epochs_no_improve,
        })
        torch.save(ckpt_payload, cfg.LATEST_SAVE_PATH)

        log_epoch_metrics_csv({
            "timestamp": ts,
            "epoch": epoch + 1,
            "epochs_total": cfg.EPOCHS,
            "train_loss": train_stats["loss"],
            "train_density_loss": train_stats["density_loss"],
            "train_heatmap_loss": train_stats["heatmap_loss"],
            "train_count_loss": train_stats["count_loss"],
            "train_offset_loss": train_stats["offset_loss"],
            "val_loss": vs["loss"],
            "MAE": vs["MAE"],
            "RMSE": vs["RMSE"],
            "R2": vs["R2"],
            "Pearson": vs["Pearson"],
            "precision": vs["precision"],
            "recall": vs["recall"],
            "f1": vs["f1"],
            "best_thresh": vs["best_thresh"],
            "count_ratio": vs["count_ratio"],
            "peak_ratio": vs["peak_ratio"],
            "lr": current_lr,
            "sample_gt": sample_gt_str,
            "sample_pred": sample_pred_str,
            "ema_mae": ema_mae,
            "balanced_score": overall_score,
            "best_model_saved": int(score_improved),
            "epochs_no_improve": epochs_no_improve,
        })

        if epochs_no_improve >= cfg.EARLY_STOPPING_PATIENCE:
            print(f"\n[Early Stopping] No improvement for {cfg.EARLY_STOPPING_PATIENCE} epochs. Stopping at Epoch {epoch + 1}.")
            break

    print("\n" + "=" * 70)
    print("Training Finished.")
    print(f"Best Validation Score: {best_combined_score:.3f} | Best MAE: {best_mae:.3f} | Best F1: {best_f1:.3f}")
    print("=" * 70)

    diagnostics_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnose_fin_try.py")
    subprocess.check_call([
        sys.executable, diagnostics_script,
        "--checkpoint", cfg.BEST_BALANCED_SAVE_PATH,
        "--output-dir", cfg.DIAGNOSTICS_DIR,
    ])
    return True
