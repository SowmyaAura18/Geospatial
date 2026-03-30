import os
import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

# IDs used for the final QGIS-ready map
ID_RCC = 1; ID_TIN = 2; ID_TILED = 3; ID_UTILITY = 4; ID_WATER = 5; ID_ROAD = 6

def apply_color_and_context(raw_img_path, ai_mask_path):
    print("[1/2] Initializing the Master Context Engine...")
    
    village_name = os.path.splitext(os.path.basename(raw_img_path))[0]
    output_dir = "Final_Outputs"
    os.makedirs(output_dir, exist_ok=True)
    final_map_path = os.path.join(output_dir, f"{village_name}_Final_Map.tif")

    # Smooth Elliptical Kernels for natural shapes
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    k_massive = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

    with rasterio.open(raw_img_path) as src_raw, rasterio.open(ai_mask_path) as src_ai:
        meta = src_ai.meta.copy()
        # Ensure metadata is clean for the final output
        meta.update({"count": 1, "dtype": 'uint8', "compress": 'lzw', "nodata": 0})
        
        with rasterio.open(final_map_path, 'w', **meta) as dst:
            for y in tqdm(range(0, src_raw.height, 512), desc="Applying Pixel Logic"):
                for x in range(0, src_raw.width, 512):
                    h, w = min(512, src_raw.height - y), min(512, src_raw.width - x)
                    window = Window(x, y, w, h)
                    
                    # Read patches
                    raw_patch = np.moveaxis(src_raw.read([1, 2, 3], window=window), 0, -1)
                    bgr_patch = cv2.cvtColor(raw_patch, cv2.COLOR_RGB2BGR)
                    
                    bgr_blurred = cv2.medianBlur(bgr_patch, 7)
                    hsv_blurred = cv2.cvtColor(bgr_blurred, cv2.COLOR_BGR2HSV)
                    
                    ai_patch = src_ai.read(1, window=window)
                    final_patch = np.zeros_like(ai_patch)
                    
                    # Background Mask
                    nodata_mask = np.all(raw_patch <= 15, axis=-1) | np.all(raw_patch >= 240, axis=-1)
                    
                    # 1. UTILITIES (AI Class 2)
                    final_patch[ai_patch == 2] = ID_UTILITY
                    
                    # GRASSLAND VETO
                    grass_mask = cv2.inRange(hsv_blurred, np.array([30, 40, 40]), np.array([85, 255, 255]))
                    ai_patch[(ai_patch == 1) & (grass_mask == 255)] = 0
                    
                    # 2. ROOFTOP MATERIAL
                    building_mask = (ai_patch == 1)
                    H, S, V = hsv_blurred[:,:,0], hsv_blurred[:,:,1], hsv_blurred[:,:,2]
                    
                    is_tiled = (H >= 0) & (H <= 20) & (S > 100) & (V > 50)
                    is_tin_blue = (H >= 90) & (H <= 130) & (S > 50) & (V > 100)
                    is_tin_white = (S < 15) & (V > 220)
                    is_tin = is_tin_blue | is_tin_white
                    
                    final_patch[building_mask & is_tiled] = ID_TILED
                    final_patch[building_mask & is_tin] = ID_TIN
                    final_patch[building_mask & (~is_tiled) & (~is_tin)] = ID_RCC
                    
                    # 3. WATERBODIES
                    mask_w1 = cv2.inRange(hsv_blurred, np.array([85, 40, 30]), np.array([135, 255, 255]))
                    mask_w2 = cv2.inRange(hsv_blurred, np.array([10, 15, 10]), np.array([50, 150, 90]))
                    water_combined = cv2.bitwise_or(mask_w1, mask_w2)
                    
                    gray_raw = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray_raw, 10, 50)
                    texture_mask = cv2.dilate(edges, k_large, iterations=2)
                    
                    water_flat = cv2.bitwise_and(water_combined, cv2.bitwise_not(texture_mask))
                    water_cleaned = cv2.morphologyEx(water_flat, cv2.MORPH_OPEN, k_med)
                    
                    water_filled = np.zeros_like(water_cleaned)
                    contours, _ = cv2.findContours(water_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area >= 1000:
                            perimeter = cv2.arcLength(cnt, True)
                            circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
                            hull = cv2.convexHull(cnt)
                            hull_area = cv2.contourArea(hull)
                            solidity = float(area) / hull_area if hull_area > 0 else 0
                            if solidity > 0.4 and circularity > 0.1:
                                cv2.drawContours(water_filled, [cnt], -1, 255, thickness=cv2.FILLED)
                                
                    final_patch[(water_filled == 255) & (final_patch == 0) & (~nodata_mask)] = ID_WATER
                    
                    # 4. ROADS
                    shadow_mask = (V < 55).astype(np.uint8) * 255
                    mask_r1 = cv2.inRange(hsv_blurred, np.array([0, 0, 60]), np.array([180, 35, 170])) 
                    mask_r2 = cv2.inRange(hsv_blurred, np.array([10, 15, 110]), np.array([25, 75, 230])) 
                    road_combined = cv2.bitwise_or(mask_r1, mask_r2)
                    
                    road_no_shadows = cv2.bitwise_and(road_combined, cv2.bitwise_not(shadow_mask))
                    road_no_buildings = cv2.bitwise_and(road_no_shadows, cv2.bitwise_not(building_mask.astype(np.uint8)*255))
                    
                    road_cleaned = cv2.morphologyEx(road_no_buildings, cv2.MORPH_OPEN, k_small)
                    road_cleaned = cv2.morphologyEx(road_cleaned, cv2.MORPH_CLOSE, k_massive)
                    
                    num_r_labels, r_labels, r_stats, _ = cv2.connectedComponentsWithStats(road_cleaned, connectivity=8)
                    for j in range(1, num_r_labels):
                        if r_stats[j, cv2.CC_STAT_AREA] >= 1000: 
                            final_patch[(r_labels == j) & (final_patch == 0) & (~nodata_mask)] = ID_ROAD
                    
                    final_patch[nodata_mask] = 0
                    dst.write(final_patch, 1, window=window)
            
            # Write Colormap for QGIS
            dst.write_colormap(1, {
                0: (0,0,0,0), 1: (140,140,140,255), 2: (0,191,255,255), 
                3: (225,87,89,255), 4: (156,39,176,255), 5: (78,121,167,255), 6: (242,203,108,255)
            })

    print(f"\n🏆 FINAL MAP SAVED: {final_map_path}")
    return final_map_path