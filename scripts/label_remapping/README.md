# Label Remapping Package

A modular Python package for remapping point cloud labels across three forest datasets: Semantic3D, ForestSemantic, and DigiForests to a unified label schema.

## 📁 Package Structure

```
label_remapping/
├── __init__.py              # Package initialization and exports
├── label_io.py              # I/O functions for reading/writing labels
├── label_remapper.py        # Core remapping logic with conditional mappings
├── dataset_processor.py     # Dataset-specific processing pipelines
├── remap_labels.py          # Main CLI entry point
└── README.md               # This file
```

## 🎯 Features

- **Modular Design**: Clean separation of I/O, logic, and processing
- **Multi-Dataset Support**: Handles Semantic3D (.labels), ForestSemantic (.las), and DigiForests (.ply)
- **Conditional Mapping**: Intelligent label remapping based on geometric features
  - **Semantic3D Label 3**: PCA-based trunk detection (linearity + verticality)
  - **ForestSemantic Label 6**: Height and spatial position-based classification
- **Memory Efficient**: Chunked processing for large point clouds
- **Visualization**: Optional before/after comparison plots
- **Dry-run Mode**: Preview changes without writing files

## 📊 Label Mappings

### Semantic3D → Unified Schema
```python
{0: 0, 1: 1, 2: 1, 3: [2, 3], 4: 4, 5: 5, 6: 5, 7: 5, 8: 5}
```
- Label 3 uses **TrunkDetector**: trunk (2) or canopy (3) based on geometry

### ForestSemantic → Unified Schema
```python
{1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: [0, 4], 7: 0}
```
- Label 6 conditional: understory (4) if height < 5m AND within ±12m XY from center, else unlabeled (0)

### DigiForests → Unified Schema
```python
{0: 0, 1: 1, 2: 4, 3: 2, 4: 3, 5: 5}
```
- Simple one-to-one mapping

## 🚀 Quick Start

### Basic Usage

```bash
# Process entire Semantic3D dataset
python remap_labels.py --dataset semantic3d --input_dir /data/semantic3d/train

# Process ForestSemantic with custom output directory
python remap_labels.py --dataset forestsemantic --input_dir /data/forestsemantic --output_dir /output

# Process DigiForests with visualization plots
python remap_labels.py --dataset digiforests --input_dir /data/digiforests --plot
```

### Advanced Options

```bash
# Dry-run mode (preview without writing)
python remap_labels.py --dataset semantic3d --input_dir /data --dry-run

# Process a specific file only
python remap_labels.py --dataset semantic3d --input_dir /data --file "scene01.labels"

# Generate before/after comparison plots
python remap_labels.py --dataset forestsemantic --input_dir /data --plot
```

## 📦 Module Details

### `label_io.py`
Handles reading and writing label files across different formats.

**Functions:**
- `read_semantic3d_labels(file_path)` - Read .labels files
- `read_semantic3d_points_and_labels(label_path)` - Read .txt point cloud + .labels
- `read_forestsemantic_labels(file_path)` - Read .las classification
- `read_forestsemantic_points_and_labels(file_path)` - Read .las XYZ + labels
- `read_digiforests_labels(file_path)` - Read .ply semantics field
- `write_labels(labels, output_path)` - Write remapped labels

### `label_remapper.py`
Core remapping logic with conditional mapping support.

**Main Function:**
```python
remap_labels(labels, mapping, xyz=None, dataset=None, trunk_detector=None, chunk_size=10_000_000)
```

**Parameters:**
- `labels`: Original labels array
- `mapping`: Dictionary mapping original → new labels
- `xyz`: Point coordinates (required for conditional mappings)
- `dataset`: Dataset name for conditional logic
- `trunk_detector`: Optional pre-initialized TrunkDetector (performance optimization)
- `chunk_size`: Memory-efficient processing chunk size (default: 10M points)

**Returns:** `(remapped_labels, statistics_dict)`

### `dataset_processor.py`
High-level processing pipelines for each dataset.

**Functions:**
- `process_semantic3d(input_dir, output_dir, mapping, file_extension, dry_run, plot, specific_file)`
- `process_forestsemantic(...)` - Same parameters as above
- `process_digiforests(...)` - Same parameters as above

Each processor handles:
- File discovery and validation
- Progress tracking with tqdm
- Statistics collection
- Optional visualization
- Memory management

### `remap_labels.py`
Command-line interface and main entry point.

**Configuration:** Stores dataset mappings, file extensions, and output directory names.


Additional dependency:
- `trunk_detection.py` (local module for geometric trunk detection)

## 📝 Output

### Default Output Directories
- Semantic3D: `<input_dir_parent>/semantic3d_remapped_labels/`
- ForestSemantic: `<input_dir_parent>/forestsemantic_remapped_labels/`
- DigiForests: `<input_dir_parent>/digiforests_remapped_labels/`

### Output Files
- Remapped labels saved as `.labels` files (one per input file)
- Optional plots: `label_comparison_sidebyside.png` and `label_comparison_stacked.png`

### Statistics Printed
- Files processed count
- Total points processed
- Original vs remapped label distributions

## 🧪 Usage as Python Module

```python
from label_remapping import (
    read_semantic3d_points_and_labels,
    remap_labels,
    write_labels
)

# Read data
xyz, labels = read_semantic3d_points_and_labels('scene01.labels')

# Remap with conditional logic
mapping = {0: 0, 1: 1, 2: 1, 3: [2, 3], 4: 4, 5: 5, 6: 5, 7: 5, 8: 5}
remapped, stats = remap_labels(labels, mapping, xyz=xyz, dataset='semantic3d')

# Write output
write_labels(remapped, 'output/scene01.labels')

print(f"Processed {stats['total_points']} points")
```

## 🔄 Migration from Legacy Script

The original monolithic `remap_labels.py` (800+ lines) has been refactored into this modular package (~400 lines total, split across 4 files).

**Backward Compatibility:** The original `remap_labels.py` is still available and functional.

**To switch to the new version:**
```bash
# Old
python remap_labels.py --dataset semantic3d --input_dir /data

# New (recommended)
cd label_remapping
python remap_labels.py --dataset semantic3d --input_dir /data
```

## 🐛 Troubleshooting

**Import errors for laspy/plyfile:**
```bash
pip install laspy plyfile
```

**Memory issues with large point clouds:**
- Adjust `chunk_size` parameter in `remap_labels()` (default: 10M points)
- Process files individually using `--file` flag

**Plotting not available:**
- Ensure `plot_label_comparison.py` is in the parent `scripts/` directory
- Install matplotlib if missing: `pip install matplotlib`

## 📄 License

Part of the uban-trained-forest-ready project.

## 👥 Contributing

When modifying this package:
1. Keep I/O operations in `label_io.py`
2. Keep remapping logic in `label_remapper.py`
3. Keep dataset-specific workflows in `dataset_processor.py`
4. Update this README if adding new features
