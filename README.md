# Plant Foliage Leaf Segmentation & Phenotyping Pipeline

An industry-standard, high-throughput computer vision pipeline designed for plant leaf segmentation and phenotyping. Automatically extracts **7 core geometric phenotypic traits** from plant canopy imagery across both green and anthocyanin-rich (red/purple) foliage classes.

---

## 📐 Core 7 Phenotypic Traits Extracted

1. **PLA (Projected Leaf Area)** ($\text{mm}^2$): Total, Green, and Red canopy surface areas.
2. **Canopy Width** ($\text{mm}$): Bounding extent canopy width.
3. **Canopy Height** ($\text{mm}$): Bounding extent canopy height.
4. **Convex Hull Area** ($\text{mm}^2$): Overall bounding convex envelope.
5. **Solidity**: Canopy compactness filling ratio ($\text{PLA} / \text{Convex Hull Area}$).
6. **PCA Orientation Angle** (Degrees): Principal axis orientation of leaf canopy ($0^\circ \dots 180^\circ$).
7. **PCA Aspect Ratio**: Principal axis elongation ratio $\sqrt{\lambda_1 / \lambda_2}$.

---

## 📁 Repository Directory Structure

```text
leaf-segmentation/
├── src/                          # Modular Core Building Blocks
│   ├── calibration.py            # Blue reference marker detection & mm² scale calibration
│   ├── features.py               # CIELAB & HSV feature extraction (Chroma, a*, b*, S, ExR, ExG)
│   ├── gating.py                 # Lightness, border margin, and low-chroma noise floor gates
│   ├── thresholding.py           # Multi-gate fusion & Otsu thresholding computation
│   ├── postprocessing.py         # Morphological cleanup, geometry filtering & leak-proof submask cleaning
│   ├── metrics.py                # 7 Core Phenotypic Measurements computation
│   └── visualization.py          # 4-Panel diagnostic breakdown, summary table & master grid export
│
├── pipelines/                    # Specialized Pipeline Engines
│   └── adaptive_b0.py            # Adaptive B0 Dual-Color Segmentation Pipeline
│
├── leaf-image/                   # Input Datasets (RedA, RedC, Green, etc.)
├── result/                       # Output Results (e.g. 051_adaptive_b0_13_08_2026)
├── run_segmentation.py           # Main Industry-Standard CLI Entry Point
└── README.md                     # Project Documentation
```

---

## 🚀 Industry-Standard CLI Usage

The primary execution entry point is **[run_segmentation.py](file:///e:/Smolfish/leaf-segmentation/run_segmentation.py)**.

### 1. Process Entire Dataset Directory
```bash
python run_segmentation.py --input leaf-image/
```

### 2. Process a Single Image File
```bash
python run_segmentation.py --input leaf-image/RedA/IMG_8004.JPG
```

### 3. Filter Specific Cohort(s)
```bash
python run_segmentation.py --input leaf-image/ --cohort RedA,RedC
```

### 4. Custom Output Directory & Pipeline Selection
```bash
python run_segmentation.py -i leaf-image/ -o result/ -p adaptive_b0
```

---

## ⚙️ Command-Line Arguments Reference

| Short | Long Flag | Description | Default |
| :---: | :--- | :--- | :--- |
| `-i` | `--input` | Path to image file, directory, or glob pattern | `leaf-image` |
| `-o` | `--output-dir` | Base directory for output runs | `result` |
| `-c` | `--cohort` | Filter cohort(s) (e.g. `RedA` or `RedA,RedC`) | `None` (All) |
| `-p` | `--pipeline` | Segmentation engine (`adaptive_b0`) | `adaptive_b0` |
| | `--min-area` | Override minimum contour area threshold in pixels | Dynamic |
| | `--l-min` | Minimum Lightness gate threshold ($L^*$) | `15` |
| | `--l-max` | Maximum Lightness gate threshold ($L^*$) | `235` |
| | `--no-grid` | Skip generating master stitched grid image | `False` |

---

## 📂 Output Directory Naming

Output results are automatically organized into numbered run folders inside `result/`:

```text
result/{run_id:03d}_{pipeline_name}_{date}/
```

**Examples:**
- `result/050_adaptive_b0_13_08_2026/`
- `result/051_adaptive_b0_13_08_2026/`

---

## 📊 Artifacts Generated Per Run

Each execution creates the following diagnostic outputs inside the run directory:

1. **`<sample_name>_result.png`**: High-resolution 4-panel diagnostic panel:
   - Panel 1: `Original RGB Image`
   - Panel 2: `Total Foliage Binary Mask`
   - Panel 3: `Green Foliage Class Overlay`
   - Panel 4: `Red Foliage Class Overlay`
2. **`SUMMARY_METRICS_TABLE.png`**: Formatted PNG image table summarizing all 7 phenotypic metrics per sample.
3. **`ALL_SAMPLES_MASTER_GRID.png`**: Single stitched grid image combining diagnostic panels for all processed samples.
