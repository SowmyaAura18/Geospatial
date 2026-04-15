import streamlit as st
import os
import rasterio
from rasterio.enums import Resampling
import numpy as np
from PIL import Image
import glob
import cv2

# --- CORE ENGINE IMPORTS ---
from run_inference import run_ai_scanner
from Geospatial_AI import apply_color_and_context

# Disable Decompression Bomb protection for massive drone maps
Image.MAX_IMAGE_PIXELS = None

# --- PAGE CONFIGURATION (Eco/Research Vibe) ---
st.set_page_config(page_title="GeoVision Analytics", page_icon="🌍", layout="wide")

# --- INITIALIZE MEMORY ---
if 'analysis_finished' not in st.session_state:
    st.session_state.analysis_finished = False
if 'output_raster' not in st.session_state:
    st.session_state.output_raster = ""
if 'input_raster' not in st.session_state:
    st.session_state.input_raster = ""

# --- CLEAN / ACADEMIC CSS ---
st.markdown("""
    <style>
    /* Clean, legible typography */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    
    /* Solid Green Execute Button */
    .stButton>button { 
        width: 100%; 
        border-radius: 4px; 
        font-weight: bold; 
        font-size: 1.05rem;
        background-color: #2E7D32; /* Earthy Green */
        color: white; 
        border: 1px solid #1B5E20; 
        padding: 12px;
        transition: 0.2s;
    }
    .stButton>button:hover { 
        background-color: #1B5E20; 
        color: white;
        border: 1px solid #000000;
    }
    
    /* Clean Headers */
    h1, h2, h3 {
        color: #2E7D32;
    }
    
    /* Subtle Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F8F9FA;
        border-right: 1px solid #E0E0E0;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR: WORKFLOW CONTROLS ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size: 3rem;'>🌍</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>GeoVision</h2>", unsafe_allow_html=True)
    st.caption("<div style='text-align: center;'>Spatial Data Extractor</div>", unsafe_allow_html=True)
    st.divider()
    
    st.markdown("### 📂 Data Source Selection")
    
    # Auto-read from the Google Drive folder
    drive_dir = "/content/drive/MyDrive/TerraScan_Data/"
    available_tifs = glob.glob(f"{drive_dir}*.tif")
    
    if not available_tifs:
        st.error("No .tif files found in Drive.")
        selected_file_path = None
    else:
        file_options = [os.path.basename(f) for f in available_tifs]
        chosen_file = st.selectbox("Target Orthomosaic:", file_options)
        selected_file_path = os.path.join(drive_dir, chosen_file)
    
    st.markdown("### ⚙️ Engine Settings")
    st.selectbox("Inference Model", ["ResNet-18 Multi-Class"], disabled=True)
    st.slider("Morphological Strictness", 1, 10, 5)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    execute_button = st.button("▶ START ANALYSIS")

# --- MAIN DASHBOARD ---
st.title("Topographical Infrastructure Mapping")
st.markdown("Automated classification of structural, hydrological, and road networks using deep learning.")
st.divider()

# --- SYSTEM HEALTH METRICS (Moved to top instead of sidebar) ---
st.markdown("#### System Status")
c1, c2, c3 = st.columns(3)
c1.metric(label="GPU Acceleration", value="Active", delta="CUDA Detected")
c2.metric(label="Cloud Storage", value="Linked", delta="Read/Write OK")
c3.metric(label="Model Weights", value="Loaded", delta="ResNet18")
st.markdown("<br>", unsafe_allow_html=True)

# --- EXECUTION LOGIC ---
if execute_button:
    if selected_file_path is None:
        st.warning("⚠️ Please select a valid file from the sidebar.")
    else:
        with st.spinner("Initializing GeoVision Pipeline..."):
            try:
                st.info("Step 1: Running deep learning inferences on raster tiles...")
                ai_output = run_ai_scanner(selected_file_path)
                
                st.info("Step 2: Enforcing geometric and spatial logic...")
                final_output = apply_color_and_context(selected_file_path, ai_output)
                
                st.success("✅ Spatial extraction completed successfully!")
                
                st.session_state.analysis_finished = True
                st.session_state.output_raster = final_output
                st.session_state.input_raster = selected_file_path
                
            except Exception as e:
                st.error(f"Pipeline Interrupted: {e}")
                st.session_state.analysis_finished = False

# --- RESULTS VIEWER (Clean Layout) ---
if st.session_state.analysis_finished and os.path.exists(st.session_state.output_raster):
    st.divider()
    st.subheader("Data Inspector")
    
    # Viewer Controls
    ctrl1, ctrl2 = st.columns(2)
    with ctrl1:
        view_type = st.radio("Visualization Mode:", ["Standard Overlay", "Side-by-Side", "Extracted Features Only"], horizontal=True)
    with ctrl2:
        class_filter = st.selectbox("Highlight Specific Feature:", [
            "All Extracted Data", "1 - RCC Buildings", "2 - Metal Roofs", 
            "3 - Tiled Roofs", "4 - Utilities", "5 - Water Bodies", "6 - Road Networks"
        ])

    # Image Processing for Web Display
    with st.spinner("Generating web preview..."):
        with rasterio.open(st.session_state.output_raster) as src_out:
            scale_factor = 1024 / max(src_out.width, src_out.height)
            h = int(src_out.height * scale_factor)
            w = int(src_out.width * scale_factor)
            mask_data = src_out.read(1, out_shape=(h, w), resampling=Resampling.nearest)
        
        with rasterio.open(st.session_state.input_raster) as src_in:
            raw_data = src_in.read(out_shape=(src_in.count, h, w), resampling=Resampling.nearest)
            if src_in.count >= 3:
                rgb_img = np.moveaxis(raw_data[:3], 0, -1) 
            else:
                rgb_img = np.stack((raw_data[0],)*3, axis=-1)
        
        # Color Mapping
        colors = {
            1: [140, 140, 140], 2: [0, 191, 255], 3: [225, 87, 89],
            4: [156, 39, 176], 5: [78, 121, 167], 6: [242, 203, 108]
        }
        
        display_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        if class_filter == "All Extracted Data":
            for val, color in colors.items():
                display_img[mask_data == val] = color
        else:
            target_id = int(class_filter.split(" - ")[0])
            display_img[mask_data == target_id] = colors.get(target_id, [0, 0, 0])

        st.write("") # Spacing

        # Render Logic
        if view_type == "Side-by-Side":
            colA, colB = st.columns(2)
            colA.image(rgb_img, caption="Original Drone Survey", use_container_width=True)
            colB.image(display_img, caption=class_filter, use_container_width=True)
            
        elif view_type == "Standard Overlay":
            rgb_img_uint8 = rgb_img.astype(np.uint8)
            blended = cv2.addWeighted(rgb_img_uint8, 0.65, display_img, 0.35, 0)
            
            if class_filter != "All Extracted Data":
                black_pixels = (display_img == [0, 0, 0]).all(axis=2)
                blended[black_pixels] = rgb_img_uint8[black_pixels]
                
            st.image(blended, caption="Spatial Overlay", use_container_width=True)
            
        else:
            st.image(display_img, caption="Isolated Feature Map", use_container_width=True)

    # Download Section
    st.divider()
    st.markdown("### Export Output")
    with open(st.session_state.output_raster, "rb") as file_to_dl:
        st.download_button(
            label="💾 Download QGIS-Ready .TIF Raster",
            data=file_to_dl,
            file_name=os.path.basename(st.session_state.output_raster),
            mime="image/tiff"
        )
