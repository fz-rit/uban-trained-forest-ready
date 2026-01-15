# Tree Trunk Detection in LAS Point Clouds

## Overview

This module detects tree trunks in 3D point clouds (LAS format) using local geometric features derived from Principal Component Analysis (PCA). Points are classified as either **trunk (class 2)** or **canopy (class 3)** based on their local linearity and verticality.

## Method

The detection pipeline consists of five main steps:

1. **Neighborhood Search**: Find neighbors for each point using radius or kNN search
2. **Local PCA**: Compute eigenvalues and eigenvectors from the covariance matrix of each neighborhood
3. **Feature Extraction**: Calculate linearity and verticality from eigenvalues/eigenvectors
4. **Classification**: Label points as trunk or canopy based on feature thresholds
5. **Post-Processing**: Filter small clusters using connected component analysis

### Geometric Features

**Linearity**: Measures how linear the local neighborhood is
```
linearity = (λ₁ - λ₂) / λ₁
```
where λ₁ ≥ λ₂ ≥ λ₃ are the eigenvalues sorted in descending order.

**Verticality**: Measures alignment with the vertical axis
```
verticality = |v₁ · (0, 0, 1)|
```
where v₁ is the first (dominant) eigenvector.

### Classification Rules

A point is classified as **trunk (class 2)** if:
- `linearity > 0.8`
- `verticality > 0.9`
- Height above ground ∈ [-1.5, 2.0] m

Otherwise, the point is classified as **canopy (class 3)**.



## Usage

### Basic Usage

Process a LAS file with default parameters:

```bash
python trunk_detection.py input.las output.las
```

### Python API

#### Simple Usage

```python
from trunk_detection import process_las_file

# Process with default parameters
process_las_file('forest.las', 'forest_trunks.las')
```

#### Custom Parameters

```python
from trunk_detection import TrunkDetector, process_las_file

# Create detector with custom parameters
detector = TrunkDetector(
    radius=0.4,                    # Neighborhood radius (m)
    search_method='radius',        # 'radius' or 'knn'
    min_neighbors=30,              # Minimum neighbors required
    linearity_threshold=0.8,       # Min linearity for trunk
    verticality_threshold=0.9,     # Min verticality for trunk
    min_height=0.3,                # Min height above ground (m)
    max_height=15.0,               # Max height above ground (m)
    min_cluster_size=50            # Min points per trunk cluster
)

# Process file
process_las_file('forest.las', 'forest_trunks.las', detector)
```

#### Programmatic Usage

```python
import numpy as np
import laspy
from trunk_detection import TrunkDetector

# Read LAS file
las = laspy.read('forest.las')
points = np.vstack([las.x, las.y, las.z]).T

# Detect trunks
detector = TrunkDetector()
labels = detector.detect_trunks(points)

# Access results
trunk_points = points[labels == 2]
canopy_points = points[labels == 3]

print(f"Trunk points: {len(trunk_points):,}")
print(f"Canopy points: {len(canopy_points):,}")

# Update LAS file
las.classification = labels
las.write('output.las')
```

### Examples

See `example_trunk_detection.py` for detailed examples:

```bash
python example_trunk_detection.py forest.las
```

The example script includes:
- Basic usage with default parameters
- Custom parameter configurations
- Dense forest optimization
- Programmatic usage with result analysis
- Visualization with Open3D
- Batch processing multiple files

## Parameters

### TrunkDetector Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `radius` | float | 0.4 | Radius for neighborhood search (meters) |
| `k_neighbors` | int | 50 | Number of neighbors for kNN search |
| `search_method` | str | 'radius' | Search method: 'radius' or 'knn' |
| `min_neighbors` | int | 30 | Minimum neighbors required for valid computation |
| `linearity_threshold` | float | 0.8 | Minimum linearity for trunk classification (0-1) |
| `verticality_threshold` | float | 0.9 | Minimum verticality for trunk classification (0-1) |
| `min_height` | float | 0.3 | Minimum height above ground (meters) |
| `max_height` | float | 15.0 | Maximum height above ground (meters) |
| `min_cluster_size` | int | 50 | Minimum points in a trunk cluster |

### Parameter Tuning Guidelines

**For dense forests with thin trunks:**
- Decrease `radius` to 0.3 m
- Decrease `min_cluster_size` to 30
- Decrease `linearity_threshold` to 0.75

**For sparse forests with thick trunks:**
- Increase `radius` to 0.5 m
- Increase `min_cluster_size` to 100
- Increase `linearity_threshold` to 0.85

**For better recall (detect more trunks):**
- Decrease `linearity_threshold` and `verticality_threshold`
- Decrease `min_cluster_size`

**For better precision (fewer false positives):**
- Increase `linearity_threshold` and `verticality_threshold`
- Increase `min_cluster_size`

## Performance

### Computational Complexity

- **Neighborhood Search**: O(N log N) with KDTree
- **PCA Computation**: O(N × k) where k is neighborhood size
- **Post-Processing**: O(N × k) for connected components

### Typical Processing Times

For a point cloud with 1 million points (on a modern CPU):
- Radius search (0.4 m): ~30-60 seconds
- kNN search (k=50): ~20-40 seconds
- Total processing: ~1-2 minutes

### Memory Requirements

- Point cloud (N points): ~24N bytes (for XYZ coordinates)
- Eigenvalues/eigenvectors: ~48N bytes
- Total: ~150-200 MB per million points

## Algorithm Details

### Neighborhood Search

Two methods are available:

1. **Radius Search**: Find all points within a fixed radius
   - Pros: Adapts to local density, good for variable-density clouds
   - Cons: May have too few/many neighbors in sparse/dense regions
   - Implementation: Open3D KDTree

2. **kNN Search**: Find k nearest neighbors
   - Pros: Consistent neighborhood size, good for uniform analysis
   - Cons: May include distant points in sparse regions
   - Implementation: scikit-learn KDTree

### PCA and Feature Computation

For each point's neighborhood:

1. Compute centroid: `c = mean(neighbors)`
2. Center points: `P' = P - c`
3. Compute covariance: `C = (P')ᵀ P' / n`
4. Eigen-decomposition: `C = V Λ Vᵀ`
5. Extract features from eigenvalues (Λ) and eigenvectors (V)

### Post-Processing

Connected component analysis removes isolated false positives:

1. Build adjacency graph of trunk points (within 1.5× radius)
2. Find connected components using sparse graph algorithms
3. Remove components with fewer than `min_cluster_size` points

## Assumptions and Limitations

### Assumptions

1. **Input Classification**: All points are initially labeled as class 3 (canopy)
2. **Ground Level**: Estimated as 1st percentile of Z values
3. **Vertical Orientation**: Tree trunks are approximately vertical
4. **Coordinate System**: Z-axis points upward (standard LiDAR convention)

### Limitations

1. **Leaning Trees**: May miss severely leaning trunks (< 25° from vertical)
2. **Branches**: Large vertical branches may be misclassified as trunks
3. **Dense Understory**: May struggle with complex vegetation structure
4. **Point Density**: Requires sufficient point density (~50+ points per trunk)
5. **Processing Time**: May be slow for very large point clouds (> 10M points)

### Known Issues

- **Edge Effects**: Points near cloud boundaries may have incomplete neighborhoods
- **Ground Points**: If ground points are present, they may affect height estimation
- **Mixed Classes**: Currently only handles class 2 (trunk) and 3 (canopy)

## Troubleshooting

### Too Few Trunk Points Detected

- **Decrease** `linearity_threshold` and `verticality_threshold`
- **Decrease** `min_cluster_size`
- **Increase** `radius` or `k_neighbors` for larger neighborhoods

### Too Many False Positives

- **Increase** `linearity_threshold` and `verticality_threshold`
- **Increase** `min_cluster_size`
- Adjust `min_height` and `max_height` to exclude ground/canopy

### Slow Processing

- Use `search_method='knn'` instead of `'radius'`
- Decrease `k_neighbors` or `radius`
- Downsample the point cloud before processing

### Memory Issues

- Process the point cloud in tiles/chunks
- Reduce `k_neighbors` or `radius`
- Use 32-bit floats instead of 64-bit

## Citation

If you use this code in your research, please cite:

```
Tree Trunk Detection using Local Geometric Features
Based on PCA-derived linearity and verticality measures
Implementation: GitHub Copilot, 2025
```

## License

This code is provided as-is for research and educational purposes.

## Contributing

Suggestions and improvements are welcome! Key areas for contribution:

- Support for additional point cloud formats (PLY, PCD, etc.)
- GPU acceleration for large-scale processing
- Advanced filtering methods (machine learning, deep learning)
- Batch processing utilities
- Visualization tools

## Contact

For questions or issues, please open an issue on the repository.
