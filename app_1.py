import streamlit as st
import os
import rasterio
from rasterio.enums import Resampling
import numpy as np
from PIL import Image

# Cloud-Safe Imports
from run_inference import run_ai_scanner
from Geospatial_AI import apply_color_and_context

# Disable Decompression Bomb protection
Image.MAX_IMAGE_PIXELS = None

# --- PAGE CONFIGURATION (Clean/Light Vibe) ---
st.set_page_config(page_title="TerraScan Analytics", page_icon="🗺️", layout="wide")

# --- MEMORY STATE ---
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'output_map' not in st.session_state:
    st.session_state.output_map = ""

# --- MINIMALIST CSS ---
st.markdown("""
    <style>
    .stButton>button { 
        width: 100%; 
        border-radius: 4px; 
        font-weight: 600; 
        background-color: #2E7D32; /* Forest Green */
        color: white; 
        border: none; 
        padding: 10px;
    }
    .stButton>button:hover { background-color: #1B5E20; color: white; }
    h1 { color: #2E7D32; font-family: 'Helvetica Neue', sans-serif; }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/182px-Python-logo-notext.svg.png", width=50)
    st.title("TerraScan Hub")
    st.markdown("Deep Learning Topography Extractor")
    st.divider()
    
    st.markdown("### 1. Data Input")
    uploaded_file = st.file_uploader("Upload Orthomosaic (.tif)", type=['tif', 'tiff'])
    
    st.markdown("### 2. Processing Parameters")
    st.selectbox("Inference Model", ["ResNet-18 (U-Net Backbone)", "VGG-16 (Legacy)"], disabled=True)
    st.slider("Spectral Confidence Threshold", 0.0, 1.0, 0.85)
    
    st.markdown("<br>", unsafe_allow_html=True)
    process_btn = st.button("Start Topographic Extraction")

# --- MAIN DASHBOARD ---
st.markdown("<h1>🗺️ TerraScan Infrastructure Analytics</h1>", unsafe_allow_html=True)
st.markdown("Automated parcel and feature extraction using semantic segmentation and morphological post-processing.")
st.divider()

if process_btn:
    if uploaded_file is None:
        st.warning("Please upload a .tif file in the sidebar to begin.")
    else:
        # File Handling
        upload_dir = "Input_Uploads"
        os.makedirs(upload_dir, exist_ok=True)
        tif_path = os.path.join(upload_dir, uploaded_file.name)
        
        with open(tif_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.status("Running TerraScan Extraction Pipeline...", expanded=True) as status:
            try:
                st.write("Phase 1: Generating Semantic Probabilities...")
                ai_mask_path = run_ai_scanner(tif_path)
                
                st.write("Phase 2: Applying Morphological Constraints...")
                final_map_path = apply_color_and_context(tif_path, ai_mask_path)
                
                status.update(label="Extraction Complete", state="complete", expanded=False)
                
                # Update State
                st.session_state.analysis_done = True
                st.session_state.output_map = final_map_path
                
            except Exception as e:
                status.update(label="Pipeline Error", state="error")
                st.error(f"Error encountered: {e}")

# --- RESULTS VIEWER ---
if st.session_state.analysis_done and os.path.exists(st.session_state.output_map):
    st.success("✅ Topographic extraction finished successfully.")
    
    col_a, col_b = st.columns([3, 1])
    
    with col_b:
        st.markdown("### Export Data")
        with open(st.session_state.output_map, "rb") as f:
            st.download_button(
                label="Download Raster (.tif)",
                data=f,
                file_name=os.path.basename(st.session_state.output_map),
                mime="image/tiff"
            )
        
        st.markdown("### Feature Legend")
        st.markdown("🟢 **1 - RCC Structures**")
        st.markdown("🔵 **2 - Tin Roofing**")
        st.markdown("🔴 **3 - Tiled Roofing**")
        st.markdown("🟣 **4 - Utility Infrastructure**")
        st.markdown("💧 **5 - Hydrology / Water**")
        st.markdown("🟡 **6 - Road Networks**")

    with col_a:
        st.markdown("### Spatial Viewer")
        filter_view = st.radio("Display Mode:", ["Composite Map", "Isolated Feature Map"], horizontal=True)
        
        with st.spinner("Loading raster preview..."):
            with rasterio.open(st.session_state.output_map) as src:
                scale = 1024 / max(src.width, src.height)
                data = src.read(1, out_shape=(int(src.height*scale), int(src.width*scale)), resampling=Resampling.nearest)
            
            color_map = {
                1: [140, 140, 140], 2: [0, 191, 255], 3: [225, 87, 89],
                4: [156, 39, 176], 5: [78, 121, 167], 6: [242, 203, 108]
            }
            viz_img = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)

            if filter_view == "Composite Map":
                display_label = "All Features" # Fallback label
                for val, col in color_map.items():
                    viz_img[data == val] = col
            else:
                target_class = st.selectbox("Select Feature ID:", [1, 2, 3, 4, 5, 6])
                display_label = f"Feature ID: {target_class}" # Specific label
                viz_img[data == target_class] = color_map.get(target_class, [0,0,0])

            # Use the new 'display_label' variable here
            st.image(viz_img, caption=f"Previewing: {display_label}", use_column_width=True)