# 🌍 TerraScan Hub: Enterprise Geospatial Intelligence

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ResNet18-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)](https://streamlit.io/)

TerraScan Hub is a high-resolution semantic segmentation pipeline engineered for massive-scale rural and agricultural drone telemetry. It extracts structural, hydrological, and utility features from 10GB+ orthomosaics without triggering Out-Of-Memory (OOM) crashes, outputting QGIS-ready multi-class rasters.

---

## 🚀 The Core Problem & Solution

**The Problem:** Traditional computer vision models struggle with rural geospatial data. They suffer from spectral confusion (hallucinating massive farm fields as dirt roads) and fail catastrophically when attempting to load massive, highly-compressed `.tif` files into standard cloud RAM limits. 

**The TerraScan Approach:** This pipeline bypasses standard hardware limitations using a two-pronged methodology:
1. **Low-RAM Windowed Inference:** The system never loads the full image into memory. Instead, it slides a 512x512 moving window (`rasterio.windows.Window`) across the dataset, allowing infinite-scale processing on free-tier edge nodes (like Google Colab T4 GPUs).
2. **Geometric Subtractive Math:** TerraScan implements custom OpenCV Top-Hat morphology to mathematically delete massive volumetric anomalies (farms) while preserving high-aspect-ratio linear networks (roads) that share the same spectral signature.

---

## 🧠 System Architecture

* **Deep Learning Backbone:** PyTorch ResNet18 U-Net trained to detect primary infrastructure footprints.
* **Spectral & Contextual Engine:** A post-processing layer that heals 1-pixel grid seams dynamically and enforces spatial logic (e.g., preventing buildings from spawning inside lakes).
* **Interactive X-Ray Inspector:** A native Streamlit dashboard that blends original RGB telemetry with AI extraction masks in real-time (`cv2.addWeighted`).
* **Serverless GPU Tunneling:** Bypasses standard HTTP payload limits via asynchronous Ngrok tunneling, allowing the dashboard to run directly off a cloud GPU instance.

---

## 📊 Classification Output Matrix

Outputs map strictly to a 1-channel QGIS-compliant `.tif` raster with embedded hexadecimal colormaps.

| ID | Feature Class | Target Hex |
| :---: | :--- | :--- |
| `1` | RCC Concrete Structures | `#8C8C8C` (Grey) |
| `2` | Tin / Metal Roofing | `#00BFFF` (Cyan) |
| `3` | Tiled / Sloped Roofing | `#E15759` (Red) |
| `4` | Utility Infrastructure | `#9C27B0` (Purple) |
| `5` | Hydrology / Water Bodies | `#4E79A7` (Blue) |
| `6` | Road Networks (Concrete & Dirt) | `#F2CB6C` (Yellow) |

---

## ⚡ Zero-Config Cloud Deployment

This pipeline runs entirely in the cloud. No local setup required.

1. **Open the Notebook:** Open `Geospatial_AI(1).ipynb` in Google Colab.
2. **Enable GPU:** Go to the top menu, click `Runtime` > `Change runtime type`, and select **T4 GPU**.
3. **Initialize Storage:** Run the very first code cell. It will connect to your Google Drive and automatically create a `TerraScan_Data` folder for you.
4. **Upload Map:** Drop your `.tif` drone map directly into that newly created `TerraScan_Data` folder in your Google Drive.
5. **Launch:** Click `Runtime` > `Run all`. The engine will build the environment and generate a secure Ngrok link at the bottom. Click the link to open the app!

---

## 💻 Local Developer Build

Requires an active CUDA-enabled environment and Python 3.8+.

```bash
# 1. Clone the repository
git clone [https://github.com/ExoduZz07/Geospatial-intelligence.git](https://github.com/ExoduZz07/Geospatial-intelligence.git)
cd Geospatial-intelligence

# 2. Resolve dependencies
pip install -r requirements.txt
pip install segmentation-models-pytorch pyngrok gdown

# 3. Fetch model weights
gdown 1INNelyEwutO9QAMD_XgNgWpCALZM5fXy -O mopr_hybrid_shape_3050.pth

# 4. Initialize local server
streamlit run app.py --server.maxUploadSize 10000
```
