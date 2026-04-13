---
title: Steel Scratch Detector
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: streamlit_app.py
pinned: false
---

# Steel Scratch Detector ⚡

An industrial-grade tool for detecting and segmenting surface scratches on steel.

## Overview
This application uses a high-precision **ResUNet++ with Attention mechanism** to analyze steel surface images and identify critical defects. It provides a real-time detection overlay and confidence metrics for each defect class.

## Key Features
- **Instant Analysis**: Drag and drop any steel surface image for immediate detection.
- **Deep Learning Core**: Powered by PyTorch and ResUNet++ segmentation models.
- **Interactive Reports**: Detailed confidence scores for each classified scratch.

## How to use locally
1. Install requirements: `pip install -r requirements.txt`
2. Run the app: `streamlit run streamlit_app.py`
