import os
import cv2
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
import segmentation_models_pytorch as smp
from tqdm import tqdm

# --- 1. CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "mopr_hybrid_shape_3050.pth"
PATCH_SIZE = 512

def run_ai_scanner(image_path):
    print("[1/2] Waking up the AI Brain...")
    
    # Extract unique village name and setup paths
    village_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = "Final_Outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_mask_path = os.path.join(output_dir, f"{village_name}_AI_Mask.tif")

    # Keep your original architecture
    model = smp.Unet("resnet18", encoder_weights=None, in_channels=3, classes=3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print(f"\n[2/2] Scanning {village_name} (Low-RAM Windowed Mode)...")
    with rasterio.open(image_path) as src:
        meta = src.meta.copy()
        meta.update({"count": 1, "dtype": 'uint8', "compress": 'lzw', "nodata": None})

        with rasterio.open(output_mask_path, 'w', **meta) as dst:
            for y in tqdm(range(0, src.height, PATCH_SIZE), desc="Scanning Rows"):
                for x in range(0, src.width, PATCH_SIZE):
                    h, w = min(PATCH_SIZE, src.height - y), min(PATCH_SIZE, src.width - x)
                    window = Window(x, y, w, h)
                    
                    img_patch = src.read([1, 2, 3], window=window)
                    img_patch = np.moveaxis(img_patch, 0, -1)
                    
                    # Pad edges
                    pad_h, pad_w = PATCH_SIZE - h, PATCH_SIZE - w
                    if pad_h > 0 or pad_w > 0:
                        img_patch = np.pad(img_patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
                        
                    img_tensor = torch.from_numpy(img_patch / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
                    
                    with torch.no_grad():
                        pred = model(img_tensor)
                        pred_class = torch.argmax(pred, dim=1).squeeze().cpu().numpy().astype(np.uint8)
                    
                    # --- RESTORED PREMIUM FILTERS ---
                    # 1. Median Blur: Kills the tiny 1-pixel 'static'
                        pred_class = cv2.medianBlur(pred_class, 3)
                    
                    # 2. Morphological Opening: Uses an Elliptical Kernel to smooth building edges
                    # This removes 'burrs' and protrusions for a cleaner footprint
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                        pred_class = cv2.morphologyEx(pred_class, cv2.MORPH_OPEN, kernel)
                        
                    if pad_h > 0 or pad_w > 0:
                        pred_class = pred_class[:h, :w]
                        
                    dst.write(pred_class, 1, window=window)

    print(f"\n✅ AI Mask saved to: {output_mask_path}")
    return output_mask_path