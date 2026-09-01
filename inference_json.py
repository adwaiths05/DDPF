import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import maximum_filter

from DDPF_Net_last_hope import DDPFNetLastHope, DENSITY_SCALE, BORDER_MASK, MIN_DISTANCE, IMAGE_SIZE


def resolve_checkpoint_path(ckpt: str | None = None) -> str:
    if ckpt:
        return ckpt

    candidates = [
        Path(__file__).resolve().parent / "lasthope" / "best_balanced_ddpfnet_last_hope.pth",
        Path(__file__).resolve().parent / "last_hope_checkpoint" / "best_balanced_ddpfnet_last_hope.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def sanitize_name(value: str) -> str:
    safe = ''.join(ch if ch.isalnum() or ch in ['-', '_'] else '_' for ch in value)
    return safe.strip('_') or 'results'


def load_model(checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DDPFNetLastHope(pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, device


def preprocess_image(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image could not be read: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_h, original_w = image.shape[:2]
    scale = IMAGE_SIZE / max(original_h, original_w)
    new_w, new_h = int(original_w * scale), int(original_h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = IMAGE_SIZE - new_w, IMAGE_SIZE - new_h
    padded = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    img_tensor = padded.astype(np.float32).transpose(2, 0, 1)
    img_tensor = torch.from_numpy(img_tensor).float().unsqueeze(0)
    return image, img_tensor, scale, (original_h, original_w)


def build_json_payload(image_name: str, image_arr, density_map, heatmap_logits, offset_map, scale, original_h, original_w, threshold, min_distance):
    total_area_pixels = original_w * original_h
    density_based_count = float(np.sum(density_map) / DENSITY_SCALE)
    overall_density_score = (density_based_count / total_area_pixels) * 10000 if total_area_pixels > 0 else 0.0

    h, w = density_map.shape
    mid_h, mid_w = h // 2, w // 2
    regions = {
        'top_left': density_map[0:mid_h, 0:mid_w],
        'top_right': density_map[0:mid_h, mid_w:w],
        'bottom_left': density_map[mid_h:h, 0:mid_w],
        'bottom_right': density_map[mid_h:h, mid_w:w],
    }

    regional_analysis = {}
    region_area_pixels = total_area_pixels / 4.0
    for region_name, region_data in regions.items():
        region_count = float(np.sum(region_data) / DENSITY_SCALE)
        region_score = (region_count / region_area_pixels) * 10000 if region_area_pixels > 0 else 0.0
        regional_analysis[region_name] = {
            'estimated_count': round(region_count, 2),
            'density_score': round(region_score, 3),
        }

    heatmap_probs = 1.0 / (1.0 + np.exp(-heatmap_logits))
    local_max = maximum_filter(heatmap_probs, size=min_distance)
    peaks = (heatmap_probs == local_max) & (heatmap_probs > threshold)
    ys, xs = np.where(peaks)

    detected_heads = []
    for x, y in zip(xs, ys):
        dx = np.clip(offset_map[0, y, x], 0.0, 1.0)
        dy = np.clip(offset_map[1, y, x], 0.0, 1.0)
        px, py = float(x + dx), float(y + dy)
        if px >= IMAGE_SIZE or py >= IMAGE_SIZE:
            continue

        orig_px = px / scale
        orig_py = py / scale
        confidence = float(heatmap_probs[y, x])

        detected_heads.append({
            'x': round(orig_px, 2),
            'y': round(orig_py, 2),
            'confidence': round(confidence, 3),
        })

    final_output = {
        'image_file': image_name,
        'original_resolution': f'{original_w}x{original_h}',
        'overall_metrics': {
            'density_based_count': round(density_based_count, 2),
            'localization_based_count': len(detected_heads),
            'overall_density_score': round(overall_density_score, 3),
        },
        'regional_analysis': regional_analysis,
        'detected_heads': detected_heads,
    }
    return final_output


def save_json_result(output_dir: Path, filename: str, payload: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / filename
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    return str(file_path)


def process_single_image(image_path: str, checkpoint_path: str, threshold: float, min_distance: int, output_root: str = 'inference_json_results'):
    model, device = load_model(checkpoint_path)

    image, tensor, scale, (original_h, original_w) = preprocess_image(image_path)
    tensor = tensor.to(device)

    with torch.no_grad():
        amp_dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == 'cuda')):
            outputs = model(tensor)

    density_map = outputs['density'][0, 0].cpu().float().numpy()
    heatmap_logits = outputs['heatmap_logits'][0, 0].cpu().float().numpy()
    offset_map = outputs['offset'][0].cpu().float().numpy()

    payload = build_json_payload(
        image_name=os.path.basename(image_path),
        image_arr=image,
        density_map=density_map,
        heatmap_logits=heatmap_logits,
        offset_map=offset_map,
        scale=scale,
        original_h=original_h,
        original_w=original_w,
        threshold=threshold,
        min_distance=min_distance,
    )

    parent_name = sanitize_name(str(Path(image_path).resolve().parent.name))
    target_dir = Path(output_root) / parent_name
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"result_{Path(image_path).stem}.json"
    saved_path = save_json_result(target_dir, file_name, payload)

    print(f"Saved JSON for {os.path.basename(image_path)} to {saved_path}")
    print(f"Density count: {payload['overall_metrics']['density_based_count']:.2f} | peaks: {payload['overall_metrics']['localization_based_count']}")
    return payload, saved_path


def process_folder(input_dir: str, checkpoint_path: str, threshold: float, min_distance: int, output_root: str = 'inference_json_results'):
    folder_path = Path(input_dir).resolve()
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {input_dir}")

    supported_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = sorted(
        [p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in supported_exts],
        key=lambda p: p.name.lower(),
    )

    if not image_files:
        raise FileNotFoundError(f"No supported image files were found in: {input_dir}")

    model, device = load_model(checkpoint_path)
    result_bundle = []
    folder_output_dir = Path(output_root) / sanitize_name(folder_path.name)
    folder_output_dir.mkdir(parents=True, exist_ok=True)

    for image_file in image_files:
        image, tensor, scale, (original_h, original_w) = preprocess_image(str(image_file))
        tensor = tensor.to(device)

        with torch.no_grad():
            amp_dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == 'cuda')):
                outputs = model(tensor)

        density_map = outputs['density'][0, 0].cpu().float().numpy()
        heatmap_logits = outputs['heatmap_logits'][0, 0].cpu().float().numpy()
        offset_map = outputs['offset'][0].cpu().float().numpy()

        payload = build_json_payload(
            image_name=image_file.name,
            image_arr=image,
            density_map=density_map,
            heatmap_logits=heatmap_logits,
            offset_map=offset_map,
            scale=scale,
            original_h=original_h,
            original_w=original_w,
            threshold=threshold,
            min_distance=min_distance,
        )

        file_name = f"result_{image_file.stem}.json"
        saved_path = save_json_result(folder_output_dir, file_name, payload)
        result_bundle.append({
            'image_file': image_file.name,
            'json_path': saved_path,
            'density_based_count': payload['overall_metrics']['density_based_count'],
            'localization_based_count': payload['overall_metrics']['localization_based_count'],
        })

    summary = {
        'source_folder': str(folder_path),
        'output_folder': str(folder_output_dir),
        'num_images': len(result_bundle),
        'results': result_bundle,
    }
    summary_path = save_json_result(folder_output_dir, 'summary.json', summary)
    print(f"Saved folder results to {folder_output_dir}")
    print(f"Summary JSON: {summary_path}")
    return summary, summary_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run DDPF-Net inference on an image or folder and save JSON results.')
    parser.add_argument('--input', type=str, required=True, help='Path to an image file or a folder of images.')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint. Defaults to lasthope/best_balanced_ddpfnet_last_hope.pth.')
    parser.add_argument('--thresh', type=float, default=0.30, help='Peak threshold for detection.')
    parser.add_argument('--min_dist', type=int, default=MIN_DISTANCE, help='Minimum distance between detected peaks.')
    parser.add_argument('--output-root', type=str, default='inference_json_results', help='Root folder for saved JSON outputs.')
    args = parser.parse_args()

    checkpoint_path = resolve_checkpoint_path(args.ckpt)
    input_path = Path(args.input).resolve()

    if input_path.is_dir():
        process_folder(str(input_path), checkpoint_path, args.thresh, args.min_dist, args.output_root)
    else:
        process_single_image(str(input_path), checkpoint_path, args.thresh, args.min_dist, args.output_root)
