# Preprocessing DigiForests - Summary

## Overview
`preprocessing_digiforests.py` is a data preprocessing pipeline that converts raw DigiForests point cloud data into a standardized PyTorch format suitable for 3D semantic and instance segmentation tasks.

## Input Data Structure
The script expects the following directory structure:
```
root/
├── digiforests-ground/
│   └── raw/
│       ├── train/
│       │   └── {date}/
│       │       └── expXX-yy/
│       │           ├── ground_clouds/
│       │           │   └── cloud_*.pcd
│       │           └── labels/
│       │               └── cloud_*.label
│       └── val/
│           └── (same structure)
└── digiforests-aerial/
    └── raw/
        ├── train/
        │   └── {date}/
        │       └── expXX-yy/
        │           └── aerial_clouds/
        │               └── cloud_*.pcd
        └── val/
```

## Key Processing Steps

### 1. **Point Cloud Reading** (`read_pcd_xyz_i`)
- Reads `.pcd` files using Open3D
- Extracts XYZ coordinates (N × 3)
- Extracts intensity/reflectance from colors channel (N × 1)
- Normalizes intensity to [0, 1] range

### 2. **Label Reading** (`read_label_file`)
- Reads packed `.label` files (uint32 format)
- Unpacks semantic labels (lower 16 bits)
- Unpacks instance IDs (upper 16 bits)

### 3. **Semantic Label Remapping** (`remap_semantic`)
Converts raw semantic IDs to meta-classes:
- 0: unlabeled → 255 (IGNORE_IDX)
- 1: ground → 1
- 2: shrub → 2
- 3: stem → 3
- 4: canopy → 4

### 4. **Instance Reindexing** (`reindex_instances`)
- Maps instance IDs to contiguous range [0..K]
- Preserves 0 as noise/unassigned category

### 5. **Global Centering** (`center_global`)
- Centers point cloud by subtracting mean XYZ
- Stores the original center for recovery

### 6. **Offset Computation** (`compute_offsets`)
- Calculates per-point offsets to instance centroids
- Used for instance segmentation training
- Points belonging to same instance point toward their center

### 7. **Data Serialization** (`save_pt`)
Saves preprocessed data as `.pt` files containing:
- `points`: Centered XYZ coordinates (torch.Tensor)
- `feats`: Normalized intensity features (torch.Tensor)
- `labels`: Remapped semantic labels (torch.Tensor, int64)
- `instance_ids`: Reindexed instance IDs (torch.Tensor, int32)
- `offset`: Per-point offsets to instance centers (torch.Tensor)
- `center`: Original cloud center (torch.Tensor)
- `meta`: Metadata dictionary (source, paths, split)

## Processing Modes

### Ground Data Processing
- Processes labeled training/validation data
- Includes both semantic and instance labels
- Computes instance-aware offsets

### Aerial Data Processing
- Processes unlabeled target domain data
- Sets all semantic labels to IGNORE_IDX (255)
- Sets all instance IDs to 0
- Zero offsets (no instance information)

## Output

### File Structure
```
output_root/
├── digiforests-ground/
│   ├── train/
│   │   └── {date}/expXX-yy/ground_clouds/cloud_*.pt
│   └── val/
│       └── (same structure)
├── digiforests-aerial/
│   ├── train/
│   │   └── {date}/expXX-yy/aerial_clouds/cloud_*.pt
│   └── val/
└── manifest.all.json
```

### Manifest File
JSON file listing all processed samples for easy batch loading.

## Usage
```bash
python preprocessing_digiforests.py \
    --root /path/to/raw/data \
    --out /path/to/output
```

## Dependencies
- Python 3.x
- NumPy
- PyTorch
- Open3D
- Standard library: os, json, argparse, pathlib

## Constants
- `IGNORE_IDX = 255`: Label value for ignored/unlabeled points in loss computation
- Customizable semantic mapping via `DF_SEM_TO_META` dictionary

## Use Case
This preprocessing pipeline is designed for:
- Domain adaptation tasks (ground → aerial)
- Joint semantic and instance segmentation
- 3D point cloud understanding in forestry applications
