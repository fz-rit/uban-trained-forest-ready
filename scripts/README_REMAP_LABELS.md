# Label Remapping Scripts

This directory contains scripts for remapping labels from three different datasets (Semantic3D, ForestSemantic, and DigiForests) to a unified label set, with visualization capabilities.

## Scripts

### 1. `remap_labels.py` - Main Label Remapping Script

Remaps labels from dataset-specific format to a unified label set.

#### Label Mappings

| Dataset | Original → Unified Mapping |
|---------|----------------------------|
| **Semantic3D** | `{0: 0, 1: 1, 2: 1, 3: 3, 4: 4, 5: 5, 6: 5, 7: 5, 8: 5}` |
| **ForestSemantic** | `{1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 5, 7: 0}` |
| **DigiForests** | `{0: 0, 1: 1, 2: 4, 3: 2, 4: 3, 5: 5}` |

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
python scripts/remap_labels.py --dataset semantic3d --input_dir /home/fzhcis/data/semantic3d_full/Semantic3D/ --plot --output_dir /home/fzhcis/data/semantic3d_full/Semantic3D/semantic3d_remapped_labels/


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

## Examples

### Example 1: Remap Semantic3D labels with dry-run

```bash
python remap_labels.py \
    --dataset semantic3d \
    --input_dir /data/semantic3d/train \
    --output_dir /data/output/semantic3d \
    --dry-run
```

This will show what would be done without actually writing files.

### Example 2: Remap ForestSemantic labels and generate plots

```bash
python remap_labels.py \
    --dataset forestsemantic \
    --input_dir /data/forestsemantic \
    --output_dir /data/output/forestsemantic \
    --plot
```

This will:
1. Remap all `.las` files in the input directory
2. Save remapped labels as `.labels` files
3. Generate two comparison plots showing label distributions

### Example 3: Remap DigiForests with default output directory

```bash
python remap_labels.py \
    --dataset digiforests \
    --input_dir /data/digiforests/aggregate_outputs
```

Output will be saved to `/data/digiforests/digiforests_remapped_labels/`

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

## Output Structure

When processing files, the output structure mirrors the input:

```
input_dir/
├── file1.labels (or .las/.ply)
└── subdir/
    └── file2.labels

output_dir/
├── file1.labels
└── subdir/
    └── file2.labels
```

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

## Notes

- The script preserves the directory structure from input to output
- All label files are saved in plain text format (one label per line)
- Progress bars show real-time processing status
- Statistics are printed for each file and overall summary
- Plotting requires matplotlib and only works when `--plot` flag is used
