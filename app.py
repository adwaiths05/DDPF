import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import json
from scipy.ndimage import maximum_filter

from models import DDPFNet as DDPFNetLastHope

# ========================================================================
# STREAMLIT CONFIG & UI
# ========================================================================
st.set_page_config(page_title="DDPF-Net Crowd Counter", layout="wide")

st.title("👥 DDPF-Net Crowd Counting Engine")
st.markdown("Upload an image of a crowd, and the model will estimate the number of people and locate their heads.")

# ========================================================================
# CACHED MODEL LOADING
# ========================================================================
@st.cache_resource
def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DDPFNetLastHope(pretrained=False).to(device)
    
    if not os.path.exists(checkpoint_path):
        st.error(f"Checkpoint not found at {checkpoint_path}")
        return None, device
        
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model, device

CHECKPOINT_PATH = "op/best_balanced_ddpfnet_final_try.pth"

with st.spinner("Loading AI Model (this only happens once)..."):
    model, device = load_model(CHECKPOINT_PATH)

if model is None:
    st.stop()

# ========================================================================
# INFERENCE PIPELINE
# ========================================================================
def process_image(image_pil, threshold, filename):
    image_np = np.array(image_pil)
    original_h, original_w = image_np.shape[:2]
    
    IMAGE_SIZE = 512
    scale = IMAGE_SIZE / max(original_h, original_w)
    new_w, new_h = int(original_w * scale), int(original_h * scale)
    padded = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = IMAGE_SIZE - new_w, IMAGE_SIZE - new_h
    padded = cv2.copyMakeBorder(padded, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
    
    input_tensor = (padded.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    input_tensor = torch.from_numpy(input_tensor.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
            outputs = model(input_tensor)
            
    density_map = outputs["density"][0, 0].cpu().float().numpy()
    heatmap_logits = outputs["heatmap_logits"][0, 0].cpu().float().numpy()
    offset_map = outputs["offset"][0].cpu().float().numpy()
        
    DENSITY_SCALE = 100.0
    total_area_pixels = original_w * original_h
    density_based_count = float(np.sum(density_map) / DENSITY_SCALE)
    overall_density_score = (density_based_count / total_area_pixels) * 10000

    # Regional Analysis
    h, w = density_map.shape
    mid_h, mid_w = h // 2, w // 2
    regions = {
        "top_left": density_map[0:mid_h, 0:mid_w],
        "top_right": density_map[0:mid_h, mid_w:w],
        "bottom_left": density_map[mid_h:h, 0:mid_w],
        "bottom_right": density_map[mid_h:h, mid_w:w]
    }
    
    regional_analysis = {}
    region_area_pixels = total_area_pixels / 4.0
    for region_name, region_data in regions.items():
        region_count = float(np.sum(region_data) / DENSITY_SCALE)
        region_score = (region_count / region_area_pixels) * 10000
        regional_analysis[region_name] = {
            "estimated_count": round(region_count, 2),
            "density_score": round(region_score, 3)
        }

    # Extract Heads & Confidence
    heatmap_probs = 1 / (1 + np.exp(-heatmap_logits))
    local_max = maximum_filter(heatmap_probs, size=4)
    peaks = (heatmap_probs == local_max) & (heatmap_probs > threshold)
    ys, xs = np.where(peaks)
    
    detected_heads = []
    points_only = []
    for x, y in zip(xs, ys):
        dx = np.clip(offset_map[0, y, x], 0.0, 1.0)
        dy = np.clip(offset_map[1, y, x], 0.0, 1.0)
        px, py = float(x + dx), float(y + dy)
        if px >= new_w or py >= new_h: continue
        
        orig_px = px / scale
        orig_py = py / scale
        confidence = float(heatmap_probs[y, x])
        
        points_only.append((int(orig_px), int(orig_py)))
        detected_heads.append({
            "x": round(orig_px, 2),
            "y": round(orig_py, 2),
            "confidence": round(confidence, 3)
        })

    # Exact JSON match
    json_output = {
        "image_file": filename,
        "original_resolution": f"{original_w}x{original_h}",
        "overall_metrics": {
            "density_based_count": round(density_based_count, 2),
            "localization_based_count": len(detected_heads),
            "overall_density_score": round(overall_density_score, 3)
        },
        "regional_analysis": regional_analysis,
        "detected_heads": detected_heads
    }

    return json_output, points_only, density_map, heatmap_probs

# ========================================================================
# MAIN APP FLOW
# ========================================================================
col1, col2 = st.columns([1, 3])

with col1:
    st.header("Settings")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    threshold = st.slider("Detection Sensitivity (Threshold)", min_value=0.01, max_value=0.99, value=0.12, step=0.01)

with col2:
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert("RGB")
        
        st.markdown("### Analyzing Image...")
        progress_bar = st.progress(0)
        
        json_output, points, density_map, heatmap = process_image(image_pil, threshold, uploaded_file.name)
        progress_bar.progress(100)
        
        st.metric("Density Estimated Count", f"{json_output['overall_metrics']['density_based_count']:.1f}")
        
        st.markdown("### Visual Results")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Density Map", "Localization Heatmap", "Localization Map", "Original Image", "JSON Data"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(density_map, cmap='jet')
            ax.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            st.image(buf, use_container_width=True, caption="Density Map (Broad Features)")
            
        with tab2:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(heatmap, cmap='hot')
            ax.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            st.image(buf, use_container_width=True, caption="Localization Heatmap (Sharp Point Probabilities)")
            
        with tab3:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(image_pil)
            if len(points) > 0:
                points_np = np.array(points)
                ax.scatter(points_np[:, 0], points_np[:, 1], c='lime', s=15, marker='+')
            ax.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            st.image(buf, use_container_width=True, caption="Localization Map (Image with Detected Heads)")
            
        with tab4:
            st.image(image_pil, use_container_width=True, caption="Original Input")
            
        with tab5:
            json_string = json.dumps(json_output, indent=2)
            st.code(json_string, language="json")
            
            st.download_button(
                label="Download JSON Output",
                file_name=f"{uploaded_file.name}_results.json",
                mime="application/json",
                data=json_string
            )
