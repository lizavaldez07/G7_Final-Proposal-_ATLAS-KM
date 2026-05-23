# ATLAS-KM: HouseTS End-to-End Big Data ML Pipeline

**Spatiotemporal Housing Image Clustering — Apache Spark + MobileNetV2 + K-Means**

ATLAS-KM is a scalable, multimodal machine learning pipeline that fuses **housing images** with **tabular socioeconomic data** to cluster U.S. neighborhoods by their combined geographic appearance and economic profile. It is designed for distributed processing of large-scale datasets using Apache Spark.

---

## Overview

The pipeline constructs a *Geography × Wealth* latent space by combining:

- **Visual features** — MobileNetV2 embeddings (1,280-dim) extracted from housing images, compressed via PCA
- **Tabular socioeconomic features** — city-level aggregates of sale prices, income, amenities, unemployment, and more

These two branches are fused into a unified 2D latent space, which is then clustered using K-Means to reveal spatially and economically coherent neighborhood segments across 30 U.S. cities.

---

## Pipeline Stages

| Stage | Description |
|-------|-------------|
| **Stage 0** | Environment setup — imports, Spark session initialization |
| **Stage 1** | Data preprocessing — CSV ingestion, missing value imputation, temporal and derived feature engineering |
| **Stage 2** | Exploratory Data Analysis — city price comparisons, temporal trends, correlation heatmap |
| **Stage 3** | Feature engineering — image catalog, distributed MobileNetV2 extraction, PCA dimensionality reduction, tabular join |
| **Stage 3.5** | Multimodal feature fusion — two-branch PCA architecture producing a Geography × Wealth 2D latent space |
| **Stage 4** | K-Means clustering — Top-K composite evaluation, final model, visualization, cluster profiling, sample images |
| **Stage 5** | Results & evaluation — elbow/silhouette metrics table, geographic treemap |

---

## Project Structure

```
project/
├── G7_-_Final_Proposal_-_ATLAS-KM.ipynb   # Main pipeline notebook
├── ATLAS-KM_requirements.txt              # Python dependencies
├── ATLAS-KM_README.md                     # This file
├── HouseTS.csv                            # Tabular dataset (user-provided)
└── Housets_Final/
    └── <CityName>/
        └── <zipcode>/
            └── *.png                      # Housing images
```

**Expected image folder structure:**
```
IMAGE_ROOT/<CityName>/<zipcode>/*.png
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r ATLAS-KM_requirements.txt
```

### 2. Install Java (required for PySpark)

```bash
# macOS
brew install openjdk@11

# Ubuntu/Debian
sudo apt-get install openjdk-11-jdk
```

### 3. Configure paths

Open the notebook and update the two lines in **Stage 1.0 Configuration**:

```python
CSV_PATH   = "/path/to/HouseTS.csv"
IMAGE_ROOT = "/path/to/Housets_Final"
```

Everything else in the pipeline adapts automatically to your paths.

### 4. Launch the notebook

```bash
jupyter notebook G7_-_Final_Proposal_-_ATLAS-KM.ipynb
```

Run all cells from top to bottom. The pipeline will validate your paths and abort early with a clear message if anything is misconfigured.

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 64 | Image batch size for feature extraction |
| `IMG_SIZE` | 128 | Image resize resolution (pixels) |
| `N_PCA_COMPONENTS` | 64 | PCA dimensions for image features (Stage 3.3) |
| `SAMPLE_FRACTION` | 1.0 | Fraction of images to process — set to `0.1` for fast dev/testing |
| `RANDOM_SEED` | 42 | Global random seed for reproducibility |
| `MIN_IMAGES` | 100 | Minimum valid images required before pipeline proceeds |
| `OPTIMAL_K` | 2 | Number of clusters for the final K-Means model |

---

## Data Requirements

### HouseTS.csv

The CSV must include the following columns:

**Market Metrics:** `median_sale_price`, `median_list_price`, `median_ppsf`, `median_list_ppsf`, `homes_sold`, `pending_sales`, `new_listings`, `inventory`, `median_dom`, `avg_sale_to_list`, `sold_above_list`, `off_market_in_two_weeks`

**Demographics:** `Total Population`, `Median Age`, `Per Capita Income`, `Total Families Below Poverty`, `Total Housing Units`

**Economic:** `Median Rent`, `Median Home Value`, `Total Labor Force`, `Unemployed Population`

**Education & Commute:** `Total School Age Population`, `Total School Enrollment`, `Median Commute Time`

**Amenities:** `bank`, `bus`, `hospital`, `mall`, `park`, `restaurant`, `school`, `station`, `supermarket`

**Geographic Keys:** `city` (abbreviation), `zipcode`, `city_full`, `date`

### Images

PNG files organized as `IMAGE_ROOT/<CityName>/<zipcode>/*.png`. City folder names must use full names (e.g., `New_York`, `Los_Angeles`). The notebook maps CSV city abbreviations to folder names via the `CSV_ABBREV_TO_FOLDER` dictionary in Stage 3.4.

---

## Feature Engineering

The pipeline engineers the following derived features from the raw CSV:

| Derived Feature | Formula |
|----------------|---------|
| `affordability_ratio` | `median_sale_price / Per Capita Income` |
| `price_to_rent_ratio` | `median_sale_price / (Median Rent × 12)` |
| `unemployment_rate` | `Unemployed Population / Total Labor Force × 100` |
| `school_enrollment_rate` | `Total School Enrollment / Total School Age Population × 100` |
| `amenity_score` | Sum of all 9 individual amenity columns |
| `year_parsed`, `month`, `quarter`, `day_of_week` | Parsed from `date` column |

---

## Multimodal Fusion Architecture

```
Housing Images                        HouseTS.csv (Tabular)
      |                                       |
MobileNetV2 (1,280-dim)           8 city-level aggregates
      |  PCA #1                               |  PCA #2b
   64-dim image features               8-dim Wealth axis
      |  PCA #2a                              |
   16-dim Geography axis                      |
      |_________________ Concat _____________|
                         |
                   24-dim fused vector
                         |  PCA #3
                   2D latent space
                  (geo_axis, wealth_axis)
                         |
                      K-Means
```

---

## Clustering Evaluation

The optimal K is selected using a **composite score** across three metrics:

| Metric | Weight | Description |
|--------|--------|-------------|
| Silhouette Score | 40% | Cluster cohesion and separation |
| CV Balance | 30% | Coefficient of variation of cluster sizes (lower = more balanced) |
| Interpretability Score | 30% | Mean pairwise inter-centroid distance in tabular space |

K values from 2 to 12 are evaluated; the top-3 are ranked and visualized.

---

## Supported Cities (30 total)

Atlanta, Austin, Baltimore, Boston, Charlotte, Chicago, Cincinnati, Dallas, DC, Denver, Detroit, Houston, Las Vegas, Los Angeles, Miami, Minneapolis, New York, Orlando, Philadelphia, Phoenix, Pittsburgh, Portland, Riverside, Sacramento, San Antonio, San Diego, San Francisco, Seattle, St. Louis, Tampa

To add more cities, extend `CSV_ABBREV_TO_FOLDER` in Stage 3.4.

---

## Outputs

| File | Description |
|------|-------------|
| `eda_city_prices.png` | Top-20 cities by median sale price + amenities scatter |
| `eda_temporal_trends.png` | Quarterly housing market trends (4-panel) |
| `eda_correlation.png` | Feature correlation heatmap |
| `pca_explained_variance.png` | PCA variance explained (per-component and cumulative) |
| `latent_space_2d.png` | Geography × Wealth latent space coloured by city and income |
| `topk_evaluation.png` | 4-panel Top-K clustering metrics comparison |
| `topk_latent_scatter.png` | Latent space scatter for top-3 K values |
| `cluster_visualization.png` | Final clusters in latent space + t-SNE projection |
| `cluster_radar.html` | Interactive radar chart of cluster socioeconomic profiles |
| `city_cluster_heatmap.png` | City × Cluster proportion heatmap (top 20 cities) |
| `cluster_sample_images.png` | Sample housing images per cluster |
| `geo_treemap.html` | Interactive geographic treemap coloured by avg sale price |

---

## Spark Configuration

Runs in local mode. Default memory settings:

| Config | Value |
|--------|-------|
| Driver memory | 10 GB |
| Executor memory | 10 GB |
| Max result size | 4 GB |
| Shuffle partitions | 8 |
| Arrow optimization | Enabled |
| Adaptive query execution | Enabled |

Adjust in Stage 0 if running on a different machine. At minimum, 16 GB total RAM is recommended.

---

## System Requirements

- Python 3.9+
- Java 8 or 11 (required for PySpark)
- 16 GB RAM recommended
- GPU optional — MobileNetV2 extraction falls back to CPU automatically
