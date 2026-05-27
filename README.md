# HouseTS: Spatiotemporal Housing Image Clustering

End-to-end big data ML pipeline that clusters U.S. housing neighborhoods by combining **satellite/street-view images** (MobileNetV2 embeddings) with **tabular socioeconomic data** (Redfin HouseTS), all orchestrated through **Apache Spark**.

---

## Project Structure

```
project/                        ← root folder (rename as needed)
├── G7_ATLAS-KM.ipynb           ← main pipeline notebook
├── README.md
├── requirements.txt
│
├── HouseTS.csv                 ← tabular dataset (set CSV_PATH)
│
└── Housets_Final/              ← image dataset root (set IMAGE_ROOT)
    ├── Atlanta/
    │   ├── 30301/
    │   │   ├── image1.png
    │   │   └── ...
    │   └── ...
    ├── Boston/
    └── ...                     ← one folder per city
```

> **`project/`** is the suggested root folder name. You can rename it, but make sure `CSV_PATH` and `IMAGE_ROOT` inside the notebook still point to the correct locations.

---

## Quick Start

### 1. Clone / copy the project folder

```bash
cd project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your data paths

Open `G7_ATLAS-KM.ipynb` and edit the **two lines** in **Stage 1 — Cell 1.0 (Configuration)**:

```python
# EDIT THESE TWO LINES to match your local paths
CSV_PATH   = "/absolute/path/to/HouseTS.csv"
IMAGE_ROOT = "/absolute/path/to/Housets_Final"
```

- `CSV_PATH` — full path to the `HouseTS.csv` file.
- `IMAGE_ROOT` — full path to the folder that contains one sub-folder per city (e.g. `Atlanta/`, `Boston/`), each of which contains zip-code sub-folders with `.png` images.

Everything else in the notebook adapts automatically once these two paths are correct.

### 4. Launch the notebook

```bash
jupyter notebook G7_ATLAS-KM.ipynb
# or
jupyter lab G7_ATLAS-KM.ipynb
```

Run all cells top-to-bottom (**Kernel → Restart & Run All** is recommended for a clean run).

---

## Pipeline Overview

| Stage | Description |
|-------|-------------|
| **0 — Environment Setup** | Imports, SparkSession initialization |
| **1 — Data Preprocessing** | Load CSV, impute missing values, derive temporal & economic features |
| **2 — EDA** | City-price bar charts, temporal trends, correlation heatmap |
| **3 — Feature Engineering** | Build image manifest, extract MobileNetV2 (1 280-dim) embeddings via Spark Pandas UDF |
| **3.4 — Tabular Join** | Map city abbreviations to image folder names, compute city-level socioeconomic statistics |
| **3.5 — Multimodal Fusion** | Two-branch architecture → fused 1 288-dim vector → 2D *Geography × Wealth* latent space via PCA |
| **4 — K-Means Clustering** | Top-K evaluation (Silhouette, WCSSE, balance, interpretability), final model, t-SNE visualization |
| **5 — Results** | Evaluation table, geographic treemap, radar chart, sample images per cluster |

---

## Outputs Generated

All output files are written to the working directory (wherever the notebook runs):

| File | Description |
|------|-------------|
| `eda_city_prices.png` | Top-20 cities by median price + amenity scatter |
| `eda_temporal_trends.png` | Quarterly housing market trends |
| `eda_correlation.png` | Feature correlation heatmap |
| `latent_space_2d.png` | 2D fused latent space (pre-clustering) |
| `topk_evaluation.png` | Silhouette / WCSSE / balance scores across K values |
| `topk_latent_scatter.png` | Latent-space scatter for top-3 K candidates |
| `cluster_visualization.png` | Final cluster scatter + t-SNE |
| `cluster_sample_images.png` | Representative images per cluster |
| `city_cluster_heatmap.png` | City × cluster proportion heatmap |
| `cluster_radar.html` | Interactive radar chart of cluster profiles |
| `geo_treemap.html` | Interactive city → cluster treemap |

---

## Configuration Parameters

These can be adjusted in **Stage 1.0** without touching any other cell:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | `64` | Images per inference batch |
| `IMG_SIZE` | `128` | Resize target for MobileNetV2 input |
| `N_PCA_COMPONENTS` | `64` | PCA components (reserved; main pipeline uses full 1 280-dim) |
| `SAMPLE_FRACTION` | `1.0` | Fraction of images to use (set < 1.0 for fast testing) |
| `RANDOM_SEED` | `42` | Global random seed |
| `MIN_IMAGES` | `100` | Minimum images required per city |

---

## Hardware Notes

- **CPU-only** is supported; GPU (CUDA) is detected automatically by PyTorch and used if available.
- The Spark session is configured for **local mode** (`local[*]`) with 10 GB driver/executor memory. Reduce these values in Stage 0 if you are on a machine with less RAM.
- Feature extraction across ~35 K images typically takes **5–15 minutes** on a modern CPU.

---

## City Abbreviation Mapping

The CSV uses short city codes (e.g. `ATL`, `NY`, `SF`). The notebook maps these to the image folder names via the `CSV_ABBREV_TO_FOLDER` dictionary in Stage 3.4. If your image folders use different names, update that dictionary accordingly.

---

## Requirements

See `requirements.txt`. Python **3.9 – 3.11** is recommended for full PySpark + PyTorch compatibility.
