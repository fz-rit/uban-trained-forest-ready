# Label Remapping Scripts

This directory contains scripts for remapping labels from three different datasets (Semantic3D, ForestSemantic, and DigiForests) to a unified label set, with visualization capabilities.

## Scripts

### 1. `remap_labels.py` - Main Label Remapping Script

Remaps labels from dataset-specific format to a unified label set.

#### Label Mappings

| Dataset | Original → Unified Mapping |
|---------|----------------------------|
| **Semantic3D** | `{0: 0, 1: 1, 2: 1, 3: [2, 3], 4: 4, 5: 5, 6: 5, 7: 5, 8: 5}` |
| **ForestSemantic** | `{1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: [0,4], 7: 0}` |
| **DigiForests** | `{0: 0, 1: 1, 2: 4, 3: 2, 4: 3, 5: 5}` |

**Note on Semantic3D Label 3 (Conditional Mapping with TrunkDetector):**
- Label 3 uses **geometric feature-based** conditional mapping via TrunkDetector:
  - **Method**: PCA (Principal Component Analysis) on local neighborhoods
  - **Features**: Linearity (λ₁-λ₂)/λ₁ and Verticality |v₁·(0,0,1)|
  - **Classification Criteria**:
    - Trunk (class 2): linearity > 0.8 AND verticality > 0.9 AND height ∈ [-1.5m, 5.0m]
    - Canopy (class 3): all other points
  - **Post-processing**: Connected component filtering (min 50 points per cluster)
  - **Parameters**: radius=0.4m, search_method='radius'

**Note on ForestSemantic Label 6 (Conditional Mapping):**
- Label 6 uses a **height and position-based** conditional mapping:
  - Points with `height < 5m` **AND** within `±12m XY range` from the point cloud center → `4` (Understory)
  - All other label 6 points → `0` (Unlabeled)

#### Usage

```bash
# Basic usage - remap labels
python remap_labels.py --dataset <dataset_name> --input_dir <path> --output_dir <path>

# With dry-run to preview changes
python remap_labels.py --dataset semantic3d --input_dir /data/input --dry-run

# With plotting to generate comparison visualizations
python remap_labels.py --dataset digiforests --input_dir /data/input --plot

# e.g.
# Process Semantic3D with dry-run
python scripts/remap_labels.py --dataset semantic3d --input_dir /home/fzhcis/data/semantic3d_full/Semantic3D/ --plot --output_dir /home/fzhcis/data/semantic3d_full/Semantic3D/semantic3d_remapped_labels/ --dry-run


# Process ForestSemantic with plots
python scripts/remap_labels.py --dataset forestsemantic --input_dir /home/fzhcis/data/ForestSemantic/ --plot --output_dir /home/fzhcis/data/ForestSemantic/forestsemantic_remapped_labels/

# Process DigiForests with plots
python scripts/remap_labels.py --dataset digiforests --input_dir /home/fzhcis/data/DigiForests/aggregate_outputs/semantics_cleaned/ --plot --output_dir /home/fzhcis/data/DigiForests/aggregate_outputs/digiforests_remapped_labels/

```

#### Arguments

- `--dataset`: Dataset to process (choices: `semantic3d`, `forestsemantic`, `digiforests`)
- `--input_dir`: Input directory containing label files (required)
- `--output_dir`: Output directory for remapped labels (optional, defaults to `<input_dir>/../<dataset>_remapped_labels`)
- `--dry-run`: Preview changes without writing files (optional)
- `--plot`: Generate before/after comparison plots (optional)

#### Input File Formats

- **Semantic3D**: `.labels` files (text files with one label per line)
- **ForestSemantic**: `.las` files (LAS point cloud format with classification field)
- **DigiForests**: `.ply` files (PLY point cloud format with semantics field)

#### Output

- **Label Files**: `.labels` files (text format, one label per line)
- **Plots** (if `--plot` is used):
  - `label_comparison_sidebyside.png` - Side-by-side histogram comparison
  - `label_comparison_stacked.png` - Vertically stacked histogram comparison

### 2. `plot_label_comparison.py` - Plotting Module

Provides visualization functions for comparing label distributions before and after remapping.

#### Functions

- `plot_before_after_histogram()` - Side-by-side comparison
- `plot_stacked_comparison()` - Vertically stacked comparison
- `get_label_names_for_dataset()` - Get label name mappings for a dataset

This module is automatically imported by `remap_labels.py` when `--plot` is used.

## Dependencies

Required Python packages:

```bash
pip install numpy pandas matplotlib tqdm laspy plyfile
```

- `numpy` - Array operations
- `pandas` - CSV/data handling
- `matplotlib` - Plotting
- `tqdm` - Progress bars
- `laspy` - LAS file reading (ForestSemantic)
- `plyfile` - PLY file reading (DigiForests)



## Unified Label Set

After remapping, all datasets use this unified label set:

| ID | Label Name |
|----|------------|
| 0 | Unlabeled |
| 1 | Ground |
| 2 | Trunk |
| 3 | Canopy |
| 4 | Understory |
| 5 | Misc |

## Visualization

The plotting feature generates two types of visualizations:

1. **Side-by-side comparison**: Shows original and remapped label distributions side by side for easy comparison
2. **Stacked comparison**: Shows distributions vertically stacked with consistent y-axis scales

Both plots include:
- Bar counts with exact numbers
- Percentage ratios
- Color-coded bars for visual distinction
- Grid lines for easier reading


## Sanity-check remapped labels ✅
Use `sanity_check_labels.py` to verify remapped `.labels` files and export subsamples for visual inspection in CloudCompare. The script can either randomly select a point cloud file or use a specific file you provide.

### Install (additional dependencies)
```bash
pip install laspy plyfile
```

### Usage
```bash
# Semantic3D example (7% subsample, random file selection)
python scripts/sanity_check_labels.py \
  --dataset semantic3d \
  --point-cloud-dir /home/fzhcis/data/semantic3d_full/Semantic3D/train \
  --labels-dir /home/fzhcis/data/semantic3d_full/Semantic3D/semantic3d_remapped_labels/


# Semantic3D with specific file
python scripts/sanity_check_labels.py   --dataset semantic3d   --point-cloud-dir /home/fzhcis/mylab/data/semantic3d/remap   --labels-dir /home/fzhcis/mylab/data/semantic3d/semantic3d_remapped_labels   --point-cloud-file sg27_station5_intensity_rgb.txt

python scripts/sanity_check_labels.py   --dataset semantic3d   --point-cloud-dir /home/fzhcis/data/semantic3d_full/Semantic3D/train   --labels-dir /home/fzhcis/data/semantic3d_full/Semantic3D/semantic3d_remapped_labels/train   --point-cloud-file sg27_station5_intensity_rgb.txt

# ForestSemantic example (8% subsample)
python scripts/sanity_check_labels.py \
  --dataset forestsemantic \
  --point-cloud-dir /home/fzhcis/data/ForestSemantic/ \
  --labels-dir /home/fzhcis/data/ForestSemantic/forestsemantic_remapped_labels/ \
  --sample-fraction 0.08

# DigiForests example (5% subsample with fixed seed for reproducibility)
python scripts/sanity_check_labels.py \
  --dataset digiforests \
  --point-cloud-dir /home/fzhcis/data/DigiForests/aggregate_outputs/semantics_cleaned/ \
  --labels-dir /home/fzhcis/data/DigiForests/aggregate_outputs/digiforests_remapped_labels/ \
  --sample-fraction 0.05 \
  --seed 42

# DigiForests with specific file
python scripts/sanity_check_labels.py \
  --dataset digiforests \
  --point-cloud-dir /home/fzhcis/data/DigiForests/aggregate_outputs/semantics_cleaned/ \
  --labels-dir /home/fzhcis/data/DigiForests/aggregate_outputs/digiforests_remapped_labels/ \
  --point-cloud-file plot_01.ply
```

### Arguments
- `--dataset`: Dataset type (`semantic3d`, `forestsemantic`, or `digiforests`)
- `--point-cloud-dir`: Directory containing point cloud files
- `--labels-dir`: Directory containing remapped `.labels` files
- `--point-cloud-file`: (Optional) Specific point cloud filename to check. If not provided, randomly selects a file.
- `--sample-fraction`: (Optional) Fraction to subsample (default: 0.07)
- `--seed`: (Optional) Random seed for reproducibility of random selection and subsampling

### What it does
1. **Selects a file**: Either uses specified `--point-cloud-file` or randomly picks one from `--point-cloud-dir` (use `--seed` for reproducibility)
2. **Auto-infers label path**: Finds matching `.labels` file in `--labels-dir` by filename (stem match)
3. **Validates consistency**: Confirms point/label counts match
4. **Reports statistics**: Prints per-label distribution and flags unknown/negative labels
5. **Subsamples**: Randomly selects specified fraction (default 7%) for quick inspection
6. **Exports LAS**: Writes subset to `<labels-dir>/sanity_check_export/<filename>_subset_7pct.las` with labels in classification field

The exported LAS can be opened in CloudCompare to visually verify label quality.