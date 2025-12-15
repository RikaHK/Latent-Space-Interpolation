# Latent Space Interpolation: SD v1.5 vs SDXL

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

A comparative analysis of **Linear Interpolation (Lerp)** and **Spherical Linear Interpolation (Slerp)** in the latent space of Text-to-Image Diffusion Models. This project explores the geometric structure of the latent manifold and quantifies the perceptual smoothness of transitions using LPIPS.

## Key Features
*   **Dual Model Support:** Full pipeline for Stable Diffusion v1.5 and SDXL (with CPU offloading).
*   **Hybrid Interpolation:** Simultaneous interpolation of Gaussian Noise ($z_T$) and CLIP Text Embeddings.
*   **Geometric Visualization:** PCA projection of latent trajectories proving the hyperspherical nature of the space.
*   **Quantitative Metrics:** Automated LPIPS analysis for measuring transition stability.
*   **Ablation Study:** Tools to isolate Semantic Drift (Prompt-only) vs. Structural Drift (Seed-only).

##  Project Structure
├── analysis_graphs/      # LPIPS and PCA Graphs
├── output_videos/        # Raw Generated Transitions
├── comparison_grids/     # 2x2 Side-by-Side Comparisons
├── main.py              # Core logic (Exported from Notebook)
├── Dockerfile           # Containerization setup
└── requirements.txt     # Dependency list

## Installation

### Option A: Local Python
bash
pip install -r requirements.txt

python main.py```

### Option B: Docker
docker build -t latent-space-explorer .
docker run --gpus all -v $(pwd)/outputs:/app/output_videos latent-explorer
