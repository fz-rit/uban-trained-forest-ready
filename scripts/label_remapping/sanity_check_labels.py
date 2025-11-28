"""
Sanity check script for remapped labels across Semantic3D, ForestSemantic, and DigiForests datasets.

Features:
- Select a point cloud file either by specifying filename or randomly from a directory.
- Load the point cloud and automatically infer its corresponding remapped .labels file.
- Verify that number of points equals number of labels.
- Print label distribution and detect unknown / negative labels.
- Randomly subsample 5–10% (configurable) of points + labels.
- Export subsampled set to a LAS file for visual inspection in CloudCompare.

Supported point cloud formats:
- semantic3d: .txt / .xyz (whitespace separated; first three columns are X Y Z)
- forestsemantic: .las
- digiforests: .ply

Remapped label file format: single-column text file (space or whitespace separated) with integer labels.

Unified label IDs expected: 0..5 (adjust color map if extended).

Usage examples:
    # Random selection (with seed for reproducibility)
    python sanity_check_labels.py --dataset semantic3d \
        --point-cloud-dir /data/semantic3d/ \
        --labels-dir /data/semantic3d_remapped_labels/ \
        --sample-fraction 0.08 --seed 42

    # Specific file
    python sanity_check_labels.py --dataset semantic3d \
        --point-cloud-dir /data/semantic3d/ \
        --labels-dir /data/semantic3d_remapped_labels/ \
        --point-cloud-file bildstein_station1_xyz_intensity_rgb.txt

        
    python sanity_check_labels.py --dataset semantic3d \
        --point-cloud-dir /home/fzhcis/mylab/data/semantic3d/remap \
        --labels-dir /home/fzhcis/mylab/data/semantic3d/semantic3d_remapped_labels \
        --point-cloud-file sg27_station5_intensity_rgb.txt

    python sanity_check_labels.py --dataset forestsemantic \
        --point-cloud-dir /data/forestsemantic/ \
        --labels-dir /data/forestsemantic_remapped_labels/ \
        --sample-fraction 0.05

    python sanity_check_labels.py --dataset digiforests \
        --point-cloud-dir /data/digiforests/ \
        --labels-dir /data/digiforests_remapped_labels/ \
        --point-cloud-file plot_01.ply

Export directory: <labels-dir>/sanity_check_export/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import random
import numpy as np
import laspy
import plyfile

# -------------------------------------------------------------------------------------------------
# Color map for unified labels 0..5
# -------------------------------------------------------------------------------------------------
# Distinct, reasonably perceptually separated colors; tweak as needed.
DEFAULT_COLOR_MAP = {
    0: (180, 180, 180),   # Unlabeled (light gray)
    1: (210, 180, 140),   # Ground (tan soil)
    2: (101, 67, 33),     # Trunk (dark brown)
    3: (34, 139, 34),     # Canopy (forest green)
    4: (154, 205, 50),    # Understory (yellow‑green)
    5: (70, 130, 180),    # Misc (steel blue)
}

# -------------------------------------------------------------------------------------------------
# Reader functions
# -------------------------------------------------------------------------------------------------

def read_semantic3d_points(path: Path) -> np.ndarray:
    """Read Semantic3D point cloud (.txt/.xyz). Assumes whitespace separated, XYZ in first 3 columns."""
    try:
        data = np.loadtxt(path, dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to read Semantic3D point cloud {path}: {e}")
    if data.shape[1] < 3:
        raise ValueError(f"Point cloud file {path} has <3 columns; cannot extract XYZ.")
    return data[:, :3]

def read_forestsemantic_points(path: Path) -> np.ndarray:
    """Read ForestSemantic point cloud (.las) and return XYZ as float32."""
    try:
        las = laspy.read(str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to read LAS file {path}: {e}")
    xyz = np.vstack([las.X * las.header.scale[0] + las.header.offsets[0],
                     las.Y * las.header.scale[1] + las.header.offsets[1],
                     las.Z * las.header.scale[2] + las.header.offsets[2]]).T.astype(np.float32)
    return xyz

def read_digiforests_points(path: Path) -> np.ndarray:
    """Read DigiForests point cloud (.ply) and return XYZ as float32."""
    try:
        ply = plyfile.PlyData.read(str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to read PLY file {path}: {e}")
    vertex = ply['vertex']
    xyz = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T.astype(np.float32)
    return xyz

def read_labels_file(path: Path) -> np.ndarray:
    """Read a remapped .labels file (single column) and return int32 array."""
    try:
        labels = np.loadtxt(path, dtype=np.int32)
    except Exception as e:
        raise RuntimeError(f"Failed to read labels file {path}: {e}")
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    return labels

# -------------------------------------------------------------------------------------------------
# Subsampling and export
# -------------------------------------------------------------------------------------------------

def subsample(points: np.ndarray, labels: np.ndarray, fraction: float, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subsample a fraction of points (random without replacement).

    Returns (sub_points, sub_labels, indices).
    """
    if not (0 < fraction < 1):
        raise ValueError("Subsample fraction must be between 0 and 1.")
    n = points.shape[0]
    k = max(1, int(round(n * fraction)))
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    idx = np.random.choice(n, size=k, replace=False)
    return points[idx], labels[idx], idx

def export_subset_to_las(points: np.ndarray, labels: np.ndarray, export_path: Path) -> None:
    """Export subset to a LAS file with classification field filled by labels.
    Uses LAS version 1.2 point format 3 (with RGB capability even if unused).
    """
    export_path.parent.mkdir(parents=True, exist_ok=True)
    header = laspy.LasHeader(point_format=3, version="1.2")
    # Basic scales (millimeter precision)
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([float(points[:,0].min()), float(points[:,1].min()), float(points[:,2].min())])
    las = laspy.LasData(header)
    las.x = points[:,0]
    las.y = points[:,1]
    las.z = points[:,2]
    # Clip labels to 0-255 range for LAS classification
    las.classification = np.clip(labels, 0, 255).astype(np.uint8)
    las.write(str(export_path))

# -------------------------------------------------------------------------------------------------
# Sanity checks
# -------------------------------------------------------------------------------------------------

def perform_sanity_checks(points: np.ndarray, labels: np.ndarray, color_map: dict[int, tuple[int,int,int]]) -> None:
    n_points = points.shape[0]
    n_labels = labels.shape[0]
    print(f"Total points: {n_points:,}")
    print(f"Total labels: {n_labels:,}")
    if n_points != n_labels:
        print(f"❌ Mismatch: points ({n_points}) != labels ({n_labels}).")
        sys.exit(1)
    print("✅ Counts match.")

    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Label distribution:")
    for lbl, cnt in zip(unique_labels, counts):
        print(f"  Label {lbl}: {cnt:,} ({cnt / n_points * 100:.2f}%)")
    expected = sorted(color_map.keys())
    unknown = [int(u) for u in unique_labels if u not in expected]
    if unknown:
        print(f"⚠️  Unknown labels detected (not in color map): {unknown}")
    else:
        print("✅ All labels are within expected unified set.")
    if (labels < 0).any():
        print("⚠️  Negative labels present.")

# -------------------------------------------------------------------------------------------------
# Dispatch and utilities
# -------------------------------------------------------------------------------------------------

READERS = {
    "semantic3d": read_semantic3d_points,
    "forestsemantic": read_forestsemantic_points,
    "digiforests": read_digiforests_points,
}

FILE_EXTENSIONS = {
    "semantic3d": [".txt", ".xyz"],
    "forestsemantic": [".las"],
    "digiforests": [".ply"],
}

def find_point_cloud_files(point_cloud_dir: Path, dataset: str) -> list[Path]:
    """Find all point cloud files in the directory based on dataset type."""
    extensions = FILE_EXTENSIONS.get(dataset, [])
    files = []
    for ext in extensions:
        files.extend(list(point_cloud_dir.glob(f"**/*{ext}")))
    return sorted(files)

def select_random_file(point_cloud_dir: Path, dataset: str, seed: int | None = None) -> Path:
    """Randomly select a point cloud file from the directory."""
    files = find_point_cloud_files(point_cloud_dir, dataset)
    if not files:
        raise FileNotFoundError(f"No point cloud files found in {point_cloud_dir} for dataset {dataset}")
    
    if seed is not None:
        random.seed(seed)
    
    selected = random.choice(files)
    return selected

def infer_label_path(point_cloud_path: Path, labels_dir: Path) -> Path:
    """Infer the label file path from point cloud filename.
    
    Expected structure:
    - Point cloud: <point_cloud_dir>/<filename>.ext
    - Labels:      <labels_dir>/<filename>.labels
    """
    label_filename = point_cloud_path.stem + ".labels"
    return labels_dir / label_filename

def get_export_dir(labels_dir: Path) -> Path:
    """Get export directory under labels directory."""
    export_dir = labels_dir / "sanity_check_export"
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir

# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanity-check remapped labels for a point cloud (randomly selected or specified).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, choices=list(READERS.keys()), help="Dataset type (affects point cloud reader)")
    parser.add_argument("--point-cloud-dir", required=True, type=str, help="Directory containing point cloud files")
    parser.add_argument("--labels-dir", required=True, type=str, help="Directory containing remapped .labels files")
    parser.add_argument("--point-cloud-file", type=str, default=None, help="Specific point cloud filename (e.g., 'file.txt'). If provided, uses this file instead of random selection.")
    parser.add_argument("--sample-fraction", type=float, default=0.10, help="Fraction of points to subsample (recommend 0.05–0.10)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (affects both file selection and subsampling)")
    return parser.parse_args()

def main():
    args = parse_args()
    point_cloud_dir = Path(args.point_cloud_dir)
    labels_dir = Path(args.labels_dir)

    if not point_cloud_dir.exists():
        print(f"❌ Point cloud directory not found: {point_cloud_dir}")
        sys.exit(1)
    
    if not labels_dir.exists():
        print(f"❌ Labels directory not found: {labels_dir}")
        sys.exit(1)

    reader = READERS.get(args.dataset)
    if reader is None:
        print(f"❌ Unsupported dataset type: {args.dataset}")
        sys.exit(1)

    # Select point cloud file: either specified or random
    if args.point_cloud_file:
        # Use specified file
        point_path = point_cloud_dir / args.point_cloud_file
        if not point_path.exists():
            print(f"❌ Specified point cloud file not found: {point_path}")
            sys.exit(1)
        print(f"📁 Using specified file: {point_path.name}")
        print(f"   Full path: {point_path}")
    else:
        # Randomly select a point cloud file
        print(f"Scanning for point cloud files in: {point_cloud_dir}")
        try:
            point_path = select_random_file(point_cloud_dir, args.dataset, seed=args.seed)
            print(f"📁 Randomly selected file: {point_path.name}")
            print(f"   Full path: {point_path}")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            sys.exit(1)

    # Infer label path automatically
    labels_path = infer_label_path(point_path, labels_dir)
    print(f"🏷️  Inferred label path: {labels_path}")
    
    if not labels_path.exists():
        print(f"❌ Labels file not found: {labels_path}")
        sys.exit(1)

    print("\n" + "="*80)
    print("Loading point cloud...")
    points = reader(point_path)
    print(f"Loaded {points.shape[0]:,} points.")

    print("\nLoading labels...")
    labels = read_labels_file(labels_path)
    print(f"Loaded {labels.shape[0]:,} labels.")
    print("="*80 + "\n")

    perform_sanity_checks(points, labels, DEFAULT_COLOR_MAP)

    # Subsample
    fraction = args.sample_fraction
    if fraction < 0.01:
        print("\n⚠️  Sample fraction very low; may produce tiny subset.")
    if fraction < 0.05 or fraction > 0.10:
        print("ℹ️  Recommended fraction is between 0.05 and 0.10 for a quick visual check.")

    print(f"\nSubsampling {fraction*100:.1f}% of points...")
    sub_points, sub_labels, indices = subsample(points, labels, fraction, seed=args.seed)
    print(f"Subsampled {sub_points.shape[0]:,} points (~{fraction*100:.2f}%).")

    # Export LAS (always)
    export_dir = get_export_dir(labels_dir)
    export_filename = f"{point_path.stem}_subset_{int(fraction*100)}pct.las"
    export_path = export_dir / export_filename
    
    print(f"\n📤 Exporting subsample to LAS: {export_path}")
    try:
        export_subset_to_las(sub_points, sub_labels, export_path)
        print(f"✅ Subsample LAS written: {export_path}")
    except Exception as e:
        print(f"❌ Failed to export LAS: {e}")

    print("\n✅ Done.")

if __name__ == "__main__":
    main()
