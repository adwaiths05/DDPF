import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg

class CrossConsistencyLoss(nn.Module):
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
    def __init__(self, bg_weight: float = 0.15, fg_thresh: float = 1e-3):
        super().__init__()
        self.bg_weight = bg_weight
        self.fg_thresh = fg_thresh

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fg_mask = (target > self.fg_thresh).float()
        bg_mask = 1.0 - fg_mask
        pixel_sq_err = (pred - target) ** 2
        weighted = fg_mask * pixel_sq_err + self.bg_weight * bg_mask * pixel_sq_err
        return weighted.mean()

def centernet_focal_loss(logits, target, alpha=0.85, gamma=2.0):
    pred = torch.sigmoid(logits.float()).clamp(1e-4, 1.0 - 1e-4)
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    neg_weights = torch.pow(1.0 - target, 4)
    pos_loss = alpha * torch.log(pred) * torch.pow(1.0 - pred, gamma) * pos_inds
    neg_loss = (1.0 - alpha) * torch.log(1.0 - pred) * torch.pow(pred, gamma) * neg_weights * neg_inds
    num_pos = pos_inds.sum()
    return -neg_loss.sum() if num_pos < 1 else -(pos_loss.sum() + neg_loss.sum()) / num_pos

class DDPFLoss(nn.Module):
    def __init__(
        self, density_weight=5.0, heatmap_weight=3.0, count_weight=10.0,
        consistency_weight=0.3, focal_alpha=0.85, focal_gamma=2.0,
        density_bg_weight=0.3, heatmap_bg_weight=0.0, offset_weight=0.03, center_weight=0.0,
    ):
        super().__init__()
        self.density_loss = WeightedDensityLoss(bg_weight=density_bg_weight)
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.count_loss = nn.L1Loss()
        self.consistency_loss = CrossConsistencyLoss()

        self.dw = density_weight
        self.hw = heatmap_weight
        self.cw = count_weight
        self.sw = consistency_weight
        self.ow = offset_weight
        self.bgw = heatmap_bg_weight
        self.center_weight = center_weight

    def forward(self, outputs, gt_density, gt_heatmap, gt_center, gt_offset, gt_offset_mask, gt_count):
        pred_density = outputs["density"]
        pred_heatmap_logits = outputs["heatmap_logits"]

        loss_density = self.density_loss(pred_density, gt_density)
        loss_heatmap = centernet_focal_loss(
            pred_heatmap_logits, gt_heatmap.float(),
            alpha=self.focal_alpha, gamma=self.focal_gamma
        )

        if self.center_weight > 0.0:
            loss_center = centernet_focal_loss(
                pred_heatmap_logits, gt_center.float(),
                alpha=self.focal_alpha, gamma=self.focal_gamma
            )
        else:
            loss_center = torch.zeros(1, device=pred_density.device, dtype=torch.float32).squeeze()

        pred_count_raw = pred_density.sum(dim=(1, 2, 3)) / cfg.DENSITY_SCALE
        gt_count = gt_count.float()
        mean_count_norm = max(float(getattr(cfg, 'FIXED_DATASET_MEAN_COUNT', 100.0)), 1.0)
        loss_count = self.count_loss(pred_count_raw, gt_count) / mean_count_norm

        loss_consistency = self.consistency_loss(pred_density, pred_heatmap_logits)

        if self.bgw > 0.0:
            bg_mask = (gt_center < 0.5).float()
            pred_heatmap_probs = torch.sigmoid(pred_heatmap_logits.float())
            loss_heatmap_bg = (pred_heatmap_probs.pow(2.0) * bg_mask).sum() / (bg_mask.sum() + 1e-6)
        else:
            loss_heatmap_bg = torch.zeros(1, device=pred_density.device, dtype=torch.float32).squeeze()

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
