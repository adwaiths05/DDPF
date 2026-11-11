import os
import glob
import cv2
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A

import config as cfg
from utils import generate_density_map, generate_heatmap, generate_offset_map, generate_center_map, count_rounded_center_collisions

class ShanghaiTechDataset(Dataset):
    def __init__(self, image_dir, gt_dir, transform=None, indices=None, is_train=False):
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.is_train = is_train
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
        record = mat["image_info"][0, 0][0, 0]
        points = record[0].astype(np.float32)
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

        if len(points) > 0:
            points = points.copy()
            points *= cfg.ANNOTATION_SCALE

        if self.is_train:
            # Native-resolution training crop (512x512)
            if h0 < cfg.IMAGE_SIZE or w0 < cfg.IMAGE_SIZE:
                pad_h = max(0, cfg.IMAGE_SIZE - h0)
                pad_w = max(0, cfg.IMAGE_SIZE - w0)
                image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
                h0, w0 = image.shape[:2]

            # 70% probability: center crop around an existing head to focus on crowd regions
            if len(points) > 0 and np.random.rand() < 0.7:
                rand_pt = points[np.random.randint(0, len(points))]
                min_x = max(0, int(rand_pt[0]) - cfg.IMAGE_SIZE + 32)
                max_x = min(w0 - cfg.IMAGE_SIZE, int(rand_pt[0]) - 32)
                x1 = np.random.randint(min_x, max_x + 1) if min_x <= max_x else np.random.randint(0, max(1, w0 - cfg.IMAGE_SIZE + 1))

                min_y = max(0, int(rand_pt[1]) - cfg.IMAGE_SIZE + 32)
                max_y = min(h0 - cfg.IMAGE_SIZE, int(rand_pt[1]) - 32)
                y1 = np.random.randint(min_y, max_y + 1) if min_y <= max_y else np.random.randint(0, max(1, h0 - cfg.IMAGE_SIZE + 1))
            else:
                x1 = np.random.randint(0, max(1, w0 - cfg.IMAGE_SIZE + 1))
                y1 = np.random.randint(0, max(1, h0 - cfg.IMAGE_SIZE + 1))

            x2 = x1 + cfg.IMAGE_SIZE
            y2 = y1 + cfg.IMAGE_SIZE

            image = image[y1:y2, x1:x2]
            if len(points) > 0:
                mask = (points[:, 0] >= x1) & (points[:, 0] < x2) & (points[:, 1] >= y1) & (points[:, 1] < y2)
                points = points[mask]
                points[:, 0] -= x1
                points[:, 1] -= y1
            else:
                points = np.zeros((0, 2), dtype=np.float32)
        else:
            scale = cfg.IMAGE_SIZE / max(h0, w0)
            new_w, new_h = int(w0 * scale), int(h0 * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            pad_w, pad_h = cfg.IMAGE_SIZE - new_w, cfg.IMAGE_SIZE - new_h
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            if len(points) > 0:
                points[:, 0] *= scale
                points[:, 1] *= scale

        if self.transform is not None:
            if hasattr(self.transform, "keypoint_params") and self.transform.keypoint_params is not None:
                aug = self.transform(image=image, keypoints=[tuple(p) for p in points])
                image = aug["image"]
                points = np.asarray(aug["keypoints"], dtype=np.float32) if len(aug["keypoints"]) > 0 else np.zeros((0, 2), dtype=np.float32)
            else:
                aug = self.transform(image=image)
                image = aug["image"]

        out_points = points.copy()
        if len(out_points) > 0:
            out_points[:, 0] *= cfg.OUTPUT_SIZE / cfg.IMAGE_SIZE
            out_points[:, 1] *= cfg.OUTPUT_SIZE / cfg.IMAGE_SIZE

        density = generate_density_map(
            (cfg.OUTPUT_SIZE, cfg.OUTPUT_SIZE, 3), out_points,
            min_sigma=cfg.MIN_SIGMA, max_sigma=cfg.MAX_SIGMA, sigma_scale=cfg.SIGMA_SCALE, scale=cfg.DENSITY_SCALE
        )
        heatmap = generate_heatmap(
            (cfg.OUTPUT_SIZE, cfg.OUTPUT_SIZE, 3), out_points,
            min_sigma=cfg.HEATMAP_MIN_SIGMA, max_sigma=cfg.HEATMAP_MAX_SIGMA, sigma_scale=cfg.HEATMAP_SIGMA_SCALE
        )
        offset, offset_mask = generate_offset_map((cfg.OUTPUT_SIZE, cfg.OUTPUT_SIZE, 3), out_points)
        center = generate_center_map((cfg.OUTPUT_SIZE, cfg.OUTPUT_SIZE, 3), out_points)

        image = image.astype(np.float32).transpose(2, 0, 1)
        image = torch.from_numpy(image).float()
        density = torch.from_numpy(density).unsqueeze(0).float()
        heatmap = torch.from_numpy(heatmap).unsqueeze(0).float()
        center = torch.from_numpy(center).unsqueeze(0).float()
        offset = torch.from_numpy(offset).float()
        offset_mask = torch.from_numpy(offset_mask).float()
        count = torch.tensor(float(len(out_points)), dtype=torch.float32)

        return {
            "image": image, "density": density, "heatmap": heatmap, "center": center,
            "offset": offset, "offset_mask": offset_mask, "count": count, "points": points
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
        "image": images, "density": densities, "heatmap": heatmaps, "center": centers,
        "offset": offsets, "offset_mask": offset_masks, "count": counts, "points": points
    }

def build_dataloaders(fold=None):
    fold = fold if fold is not None else cfg.CURRENT_FOLD
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), rotate=(-5, 5), translate_percent=(-0.03, 0.03), border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.2),
        A.GaussNoise(p=0.2),
        A.MotionBlur(blur_limit=3, p=0.1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0)
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))

    valid_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0)
    ])

    all_images = sorted(glob.glob(os.path.join(cfg.TRAIN_IMAGE_DIR, "*.jpg")))
    rng = np.random.default_rng(cfg.SEED)
    counts = []
    collision_total = 0
    point_total = 0
    for img_path in all_images:
        img_name = os.path.basename(img_path)
        gt_name = "GT_" + img_name.replace("processed_", "").replace(".jpg", ".mat")
        gt_path = os.path.join(cfg.TRAIN_GT_DIR, gt_name)
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"GT file not found for split calibration: {gt_path}")
        mat = loadmat(gt_path)
        record = mat["image_info"][0, 0][0, 0]
        counts.append(float(record[1][0, 0]))
        points = record[0].astype(np.float32)
        if len(points) > 0:
            image = cv2.imread(img_path)
            h0, w0 = image.shape[:2]
            points = points * cfg.ANNOTATION_SCALE
            points *= cfg.IMAGE_SIZE / max(h0, w0)
            points *= cfg.OUTPUT_SIZE / cfg.IMAGE_SIZE
        collision_total += count_rounded_center_collisions(points)
        point_total += len(points)
    counts = np.asarray(counts, dtype=np.float32)

    if len(counts) > 1:
        bucket_edges = np.quantile(counts, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        bucket_ids = np.digitize(counts, bucket_edges[1:-1], right=False)
        fold_assignments = np.zeros(len(all_images), dtype=int)
        for b in np.unique(bucket_ids):
            idx = np.where(bucket_ids == b)[0]
            rng.shuffle(idx)
            for i, img_idx in enumerate(idx):
                fold_assignments[img_idx] = i % cfg.NUM_FOLDS
        val_indices = sorted(np.where(fold_assignments == fold)[0].tolist())
        train_indices = sorted(np.where(fold_assignments != fold)[0].tolist())
    else:
        indices = np.arange(len(all_images))
        rng.shuffle(indices)
        split = int(0.90 * len(indices))
        train_indices = indices[:split].tolist()
        val_indices = indices[split:].tolist()

    cfg.FIXED_DATASET_MEAN_COUNT = float(np.asarray([counts[i] for i in train_indices], dtype=np.float32).mean()) if train_indices else 1.0

    train_dataset = ShanghaiTechDataset(image_dir=cfg.TRAIN_IMAGE_DIR, gt_dir=cfg.TRAIN_GT_DIR, transform=train_transform, indices=train_indices, is_train=True)
    val_dataset = ShanghaiTechDataset(image_dir=cfg.TRAIN_IMAGE_DIR, gt_dir=cfg.TRAIN_GT_DIR, transform=valid_transform, indices=val_indices, is_train=False)
    test_dataset = ShanghaiTechDataset(image_dir=cfg.TEST_IMAGE_DIR, gt_dir=cfg.TEST_GT_DIR, transform=valid_transform, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True, collate_fn=crowd_collate, drop_last=True, persistent_workers=(cfg.NUM_WORKERS > 0))
    valid_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True, collate_fn=crowd_collate, persistent_workers=(cfg.NUM_WORKERS > 0))
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True, collate_fn=crowd_collate, persistent_workers=(cfg.NUM_WORKERS > 0))

    return train_loader, valid_loader, test_loader, train_dataset, val_dataset, test_dataset
