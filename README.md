# 🌍 Automated Mapping Engine (GeoAI)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ExoduZz07/Geospatial-intelligence/blob/main/Geospatial_.ipynb)

## Overview
Automated AI pipeline for extracting building footprints, road networks, and waterbodies from massive drone orthomosaics. Built for the SVAMITVA Geospatial Hackathon.

This system uses a ResNet18 U-Net backbone combined with a custom **Texture Veto & Spectral Engine** to eliminate false positives. It includes a Streamlit dashboard for real-time, GIS-ready class filtering and can securely handle 2GB+ files.

---

## 🚨 CRITICAL: AI MODEL WEIGHTS REQUIRED
Due to GitHub's file size limits, the trained PyTorch model weights are hosted on Google Drive. **You must have this file for the engine to run.**

* **Download Model:** [mopr_hybrid_shape_3050.pth](https://drive.google.com/file/d/1qYywmgl9I_G2Qm8Q580r2tbtweZcWAd7/view?usp=sharing)
* **File Name:** `mopr_hybrid_shape_3050.pth`

---

## 🚀 How to Run (1-Click Cloud Demo)
The absolute easiest way to test this pipeline is via Google Colab. No local installation or manual downloading is required.

1. Click the **Open in Colab** badge at the top of this README.
2. Go to the top menu and select `Runtime` > `Run all`.
3. The notebook will automatically download the AI weights from Google Drive, install all dependencies, and generate a secure Ngrok tunnel.
4. Scroll to the bottom of the notebook and click the **STABLE ACCESS LINK** to open the Streamlit dashboard.

---

## 💻 Local Installation (For Developers)
If you prefer to run the engine locally on a Windows/Linux machine with a dedicated GPU:
```bash
git clone [https://github.com/ExoduZz07/Geospatial-intelligence.git](https://github.com/ExoduZz07/Geospatial-intelligence.git)
cd Geospatial-intelligence
