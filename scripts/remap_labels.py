'''
Remap labels in Semantic3D, ForestSemantic, and DigitalForest datasets to a unified label set.

Label Mappings:
- semantic3d_mapping = {0: 0, 1: 1, 2: 1, 3: [2, 3], 4: 4, 5: 5, 6: 5, 7: 5, 8: 5}
  * Label 3 uses conditional mapping based on geometric features (TrunkDetector):
    - Uses PCA-based trunk detection with linearity + verticality features
    - Detected as trunk (class 2) if meets geometric criteria
    - Otherwise classified as canopy (class 3)
- forest_semantic_mapping = {1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: [0,4], 7: 0}
  * Label 6 uses conditional mapping based on height and position:
    - If height < 5m AND within ±12m XY range from center → 4 (Understory)
    - Otherwise → 0 (Unlabeled)
- digiforests_mapping = {0: 0, 1: 1, 2: 4, 3: 2, 4: 3, 5: 5}

Label file formats: 
- Semantic3D - .labels files (x30)
- ForestSemantic - .las files (x3)
- DigitalForest - .ply files (x10)

Outputs:
- Semantic3D - remapped .labels files in 'semantic3d_remapped_labels' directory
- ForestSemantic - remapped .labels files in 'forestsemantic_remapped_labels' directory
- DigitalForest - remapped .labels files in 'digiforests_remapped_labels' directory

Usage:
    python remap_labels.py --dataset semantic3d --input_dir /path/to/semantic3d --output_dir /path/to/output [--dry-run]
    python remap_labels.py --dataset forestsemantic --input_dir /path/to/forestsemantic --output_dir /path/to/output [--dry-run]
    python remap_labels.py --dataset digiforests --input_dir /path/to/digiforests --output_dir /path/to/output [--dry-run]
'''

import argparse
import numpy as np
import pandas as pd
import laspy
import plyfile
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import sys
import os
import warnings
import gc

# Add trunk_detection to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trunk_detection'))
from trunk_detection import TrunkDetector

# Import plotting functions
try:
    from plot_label_comparison import (
        plot_before_after_histogram,
        plot_stacked_comparison,
        get_label_names_for_dataset
    )
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️  Warning: plot_label_comparison module not found. Plotting will be disabled.")


# ============================================================================
# Label Mapping Configurations
# ============================================================================

SEMANTIC3D_MAPPING = {0: 0, 1: 1, 2: 1, 3: [2, 3], 4: 4, 5: 5, 6: 5, 7: 5, 8: 5}
FOREST_SEMANTIC_MAPPING = {1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: [0,4], 7: 0}
DIGIFORESTS_MAPPING = {0: 0, 1: 1, 2: 4, 3: 2, 4: 3, 5: 5}

DATASET_CONFIGS = {
    'semantic3d': {
        'mapping': SEMANTIC3D_MAPPING,
        'file_extension': '.labels',
        'output_dir_name': 'semantic3d_remapped_labels'
    },
    'forestsemantic': {
        'mapping': FOREST_SEMANTIC_MAPPING,
        'file_extension': '.las',
        'output_dir_name': 'forestsemantic_remapped_labels'
    },
    'digiforests': {
        'mapping': DIGIFORESTS_MAPPING,
        'file_extension': '.ply',
        'output_dir_name': 'digiforests_remapped_labels'
    }
}


# ============================================================================
# Core Functions for Reading Labels
# ============================================================================

def read_semantic3d_labels(file_path: Path) -> np.ndarray:
    """
    Read a Semantic3D .labels file and return a numpy array of class IDs.
    
    Args:
        file_path: Path to the .labels file
        
    Returns:
        1D numpy array of integer labels
    """
    labels = pd.read_csv(file_path, header=None, sep=r'\s+', dtype=np.int32).values
    labels = labels.reshape((-1,))
    return labels


def read_semantic3d_points_and_labels(label_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read both XYZ coordinates and labels from Semantic3D files.
    
    Args:
        label_path: Path to the .labels file (corresponding .txt point cloud file must exist)
        
    Returns:
        Tuple of (xyz coordinates as Nx3 array, labels as 1D array)
    """
    # Read labels
    labels = read_semantic3d_labels(label_path)
    
    # Read point cloud (.txt file with same stem as .labels file)
    pc_path = label_path.with_suffix('.txt')
    if not pc_path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {pc_path}")
    
    # Semantic3D .txt format: x y z intensity r g b
    pc_data = pd.read_csv(pc_path, header=None, sep=r'\s+', dtype=np.float32)
    xyz = pc_data.iloc[:, :3].values  # Extract first 3 columns (x, y, z)
    
    if len(xyz) != len(labels):
        raise ValueError(f"Mismatch: {len(xyz)} points vs {len(labels)} labels")
    
    return xyz, labels


def read_forestsemantic_labels(file_path: Path) -> np.ndarray:
    """
    Read labels from a ForestSemantic .las file (classification field).
    
    Args:
        file_path: Path to the .las file
        
    Returns:
        1D numpy array of integer labels
    """
    las = laspy.read(str(file_path))
    labels = np.array(las.classification, dtype=np.int32).reshape((-1,))
    return labels


def read_forestsemantic_points_and_labels(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read both XYZ coordinates and labels from a ForestSemantic .las file.
    
    Args:
        file_path: Path to the .las file
        
    Returns:
        Tuple of (xyz coordinates as Nx3 array, labels as 1D array)
    """
    las = laspy.read(str(file_path))
    xyz = np.vstack([
        las.X * las.header.scale[0] + las.header.offsets[0],
        las.Y * las.header.scale[1] + las.header.offsets[1],
        las.Z * las.header.scale[2] + las.header.offsets[2]
    ]).T.astype(np.float32)
    labels = np.array(las.classification, dtype=np.int32).reshape((-1,))
    return xyz, labels


def read_digiforests_labels(file_path: Path) -> np.ndarray:
    """
    Read labels from a DigiForests .ply file (semantics field).
    
    Args:
        file_path: Path to the .ply file
        
    Returns:
        1D numpy array of integer labels
    """
    plydata = plyfile.PlyData.read(str(file_path))
    labels = np.array(plydata['vertex'].data['semantics'], dtype=np.int32)
    return labels


# ============================================================================
# Core Functions for Writing Labels
# ============================================================================

def write_semantic3d_labels(labels: np.ndarray, output_path: Path) -> None:
    """
    Write remapped labels to a Semantic3D .labels file.
    
    Args:
        labels: 1D numpy array of remapped labels
        output_path: Path to save the .labels file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(labels)
    df.to_csv(output_path, header=False, index=False, sep=' ')


def write_forestsemantic_labels(labels: np.ndarray, output_path: Path) -> None:
    """
    Write remapped labels to a .labels file for ForestSemantic.
    
    Args:
        labels: 1D numpy array of remapped labels
        output_path: Path to save the .labels file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(labels)
    df.to_csv(output_path, header=False, index=False, sep=' ')


def write_digiforests_labels(labels: np.ndarray, output_path: Path) -> None:
    """
    Write remapped labels to a .labels file for DigiForests.
    
    Args:
        labels: 1D numpy array of remapped labels
        output_path: Path to save the .labels file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(labels)
    df.to_csv(output_path, header=False, index=False, sep=' ')


# ============================================================================
# Remapping Functions
# ============================================================================

def remap_labels(labels: np.ndarray, mapping: Dict, xyz: np.ndarray = None, dataset: str = None, trunk_detector: Optional[TrunkDetector] = None, chunk_size: int = 10_000_000) -> Tuple[np.ndarray, Dict]:
    """
    Remap labels according to the provided mapping dictionary.
    Handles conditional mappings (list values) that require point coordinates.
    
    Args:
        labels: Original labels array
        mapping: Dictionary mapping original labels to new labels (int or list for conditional)
        xyz: Optional Nx3 array of XYZ coordinates (required for conditional mappings)
        dataset: Optional dataset name ('semantic3d', 'forestsemantic', etc.) to determine conditional logic
        trunk_detector: Optional pre-initialized TrunkDetector instance (reused across calls for performance)
        chunk_size: Maximum points to process at once for memory efficiency (default: 10M points)
        
    Returns:
        Tuple of (remapped_labels, statistics_dict)
    """
    remapped = labels.copy()
    
    # Collect statistics
    unique_original = np.unique(labels)
    original_counts = {int(label): int(np.sum(labels == label)) for label in unique_original}
    
    # Apply mapping
    for old_label, new_label in mapping.items():
        mask = labels == old_label
        
        if isinstance(new_label, list):
            # Conditional mapping - behavior depends on dataset
            if xyz is None:
                raise ValueError(f"Conditional mapping for label {old_label} requires XYZ coordinates")
            
            # Get points with this label
            label_points = xyz[mask]
            
            if dataset == 'semantic3d' and old_label == 3:
                # Semantic3D label 3 (high vegetation) conditional mapping
                # new_label = [trunk_value, canopy_value] = [2, 3]
                trunk_val, canopy_val = new_label[0], new_label[1]
                
                # Early exit if no points with this label
                if len(label_points) == 0:
                    continue
                
                # Use pre-initialized TrunkDetector or create new one
                if trunk_detector is None:
                    trunk_detector = TrunkDetector(
                        radius=0.4,
                        search_method='knn',  # knn is faster than radius for large clouds
                        k_neighbors=50,
                        linearity_threshold=0.8,
                        verticality_threshold=0.9,
                        min_height=-1.5,
                        max_height=5.0,
                        min_cluster_size=50
                    )
                
                # Process in chunks if dataset is too large (memory-efficient)
                n_label3_points = len(label_points)
                if n_label3_points > chunk_size:
                    # Chunk processing for large point clouds
                    detected_labels = np.full(n_label3_points, canopy_val, dtype=np.int32)
                    n_chunks = (n_label3_points + chunk_size - 1) // chunk_size
                    
                    for i in range(n_chunks):
                        start_idx = i * chunk_size
                        end_idx = min((i + 1) * chunk_size, n_label3_points)
                        chunk_points = label_points[start_idx:end_idx]
                        chunk_labels = np.full(len(chunk_points), canopy_val, dtype=np.int32)
                        
                        # Detect trunks in chunk
                        chunk_result = trunk_detector.detect_trunks(
                            chunk_points,
                            original_labels=chunk_labels,
                            verbose=False
                        )
                        detected_labels[start_idx:end_idx] = chunk_result
                        
                        # Free memory immediately
                        del chunk_points, chunk_labels, chunk_result
                else:
                    # Process all points at once for smaller datasets
                    initial_labels = np.full(n_label3_points, canopy_val, dtype=np.int32)
                    detected_labels = trunk_detector.detect_trunks(
                        label_points, 
                        original_labels=initial_labels,
                        verbose=False
                    )
                    del initial_labels
                
                # Assign to global remapped array
                remapped[mask] = detected_labels
                del detected_labels, label_points
                
            elif dataset == 'forestsemantic' and old_label == 6:
                # ForestSemantic label 6 conditional mapping
                # new_label = [unlabeled_value, understory_value] = [0, 4]
                unlabeled_val, understory_val = new_label[0], new_label[1]
                
                # Compute center (XY plane)
                center_x = label_points[:, 0].mean()
                center_y = label_points[:, 1].mean()
                
                # Condition 1: under 5m high
                height_condition = label_points[:, 2] < 5.0
                
                # Condition 2: within ±12m range on XY plane from center
                dx = np.abs(label_points[:, 0] - center_x)
                dy = np.abs(label_points[:, 1] - center_y)
                xy_condition = (dx <= 12.0) & (dy <= 12.0)
                
                # Combined condition: both must be true for understory
                understory_mask = height_condition & xy_condition
                
                # Create a local remapped array for this label
                local_remapped = np.full(mask.sum(), unlabeled_val, dtype=np.int32)
                local_remapped[understory_mask] = understory_val
                
                # Assign to global remapped array
                remapped[mask] = local_remapped
            else:
                raise ValueError(f"Unknown conditional mapping for dataset={dataset}, label={old_label}")
        else:
            # Simple mapping
            remapped[mask] = new_label
    
    # Statistics for remapped labels
    unique_remapped = np.unique(remapped)
    remapped_counts = {int(label): int(np.sum(remapped == label)) for label in unique_remapped}
    
    stats = {
        'original_unique': list(unique_original),
        'original_counts': original_counts,
        'remapped_unique': list(unique_remapped),
        'remapped_counts': remapped_counts,
        'total_points': len(labels)
    }
    
    return remapped, stats


# ============================================================================
# Dataset-Specific Processing Functions
# ============================================================================

def process_semantic3d(input_dir: Path, output_dir: Path, dry_run: bool = False, plot: bool = False) -> None:
    """
    Process Semantic3D dataset: remap .labels files.
    
    Args:
        input_dir: Directory containing .labels files
        output_dir: Directory to save remapped .labels files
        dry_run: If True, only print what would be done without writing files
        plot: If True, generate before/after comparison plots
    """
    config = DATASET_CONFIGS['semantic3d']
    label_files = sorted(list(input_dir.glob(f'**/*{config["file_extension"]}')))
    
    if len(label_files) == 0:
        print(f"⚠️  No .labels files found in {input_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"Processing Semantic3D Dataset")
    print(f"{'='*80}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(label_files)} .labels files")
    print(f"Dry run: {dry_run}")
    print(f"Plot: {plot}")
    print(f"Mapping: {config['mapping']}")
    print(f"Note: Label 3 uses TrunkDetector (PCA-based geometric features: linearity + verticality)")
    
    overall_stats = {'files_processed': 0, 'total_points': 0}
    
    # Create shared TrunkDetector instance for performance (reused across all files)
    print("Initializing TrunkDetector (kNN with k=50)...")
    trunk_detector = TrunkDetector(
        radius=0.4,
        search_method='knn',  # knn is faster than radius for large clouds
        k_neighbors=50,
        linearity_threshold=0.8,
        verticality_threshold=0.9,
        min_height=-1.5,
        max_height=5.0,
        min_cluster_size=50
    )
    
    # Collect all labels for aggregated plot
    all_original_labels = []
    all_remapped_labels = []
    
    for label_file in tqdm(label_files, desc="Remapping Semantic3D labels", unit="file"):
        # Read both coordinates and labels for conditional mapping
        xyz, labels = read_semantic3d_points_and_labels(label_file)
        
        # Remap labels (passing xyz and shared detector for conditional mapping of label 3)
        remapped_labels, stats = remap_labels(labels, config['mapping'], xyz=xyz, dataset='semantic3d', trunk_detector=trunk_detector, chunk_size=10_000_000)
        
        # Prepare output path
        rel_path = label_file.relative_to(input_dir)
        output_path = output_dir / rel_path
        
        # Print statistics for this file
        if dry_run:
            print(f"\n[DRY RUN] Would process: {label_file.name}")
            print(f"  Original labels: {stats['original_unique']}")
            print(f"  Remapped labels: {stats['remapped_unique']}")
            print(f"  Total points: {stats['total_points']:,}")
            print(f"  Would save to: {output_path}")
        else:
            write_semantic3d_labels(remapped_labels, output_path)
        
        overall_stats['files_processed'] += 1
        overall_stats['total_points'] += stats['total_points']
        
        # Collect for plotting BEFORE cleanup
        if plot and not dry_run:
            all_original_labels.append(labels)
            all_remapped_labels.append(remapped_labels)
        
        # Explicit memory cleanup after each file
        del xyz, labels, remapped_labels, stats
        gc.collect()
    
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Files processed: {overall_stats['files_processed']}")
    print(f"  Total points: {overall_stats['total_points']:,}")
    print(f"{'='*80}\n")
    
    # Generate plots
    if plot and not dry_run and PLOTTING_AVAILABLE and len(all_original_labels) > 0:
        print("\n📊 Generating comparison plots...")
        all_original = np.concatenate(all_original_labels)
        all_remapped = np.concatenate(all_remapped_labels)
        
        original_names, remapped_names = get_label_names_for_dataset('semantic3d')
        
        # Side-by-side plot
        plot_path_side = output_dir / "label_comparison_sidebyside.png"
        plot_before_after_histogram(
            all_original, all_remapped,
            original_names, remapped_names,
            "Semantic3D", plot_path_side
        )
        
        # Stacked plot
        plot_path_stacked = output_dir / "label_comparison_stacked.png"
        plot_stacked_comparison(
            all_original, all_remapped,
            original_names, remapped_names,
            "Semantic3D", plot_path_stacked
        )


def process_forestsemantic(input_dir: Path, output_dir: Path, dry_run: bool = False, plot: bool = False) -> None:
    """
    Process ForestSemantic dataset: remap labels from .las files.
    
    Args:
        input_dir: Directory containing .las files
        output_dir: Directory to save remapped .labels files
        dry_run: If True, only print what would be done without writing files
        plot: If True, generate before/after comparison plots
    """
    config = DATASET_CONFIGS['forestsemantic']
    las_files = sorted(list(input_dir.glob(f'**/*{config["file_extension"]}')))
    
    if len(las_files) == 0:
        print(f"⚠️  No .las files found in {input_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"Processing ForestSemantic Dataset")
    print(f"{'='*80}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(las_files)} .las files")
    print(f"Dry run: {dry_run}")
    print(f"Plot: {plot}")
    print(f"Mapping: {config['mapping']}")
    print(f"Note: Label 6 uses conditional mapping (height < 5m & within ±12m XY from center → 4, else → 0)")
    
    overall_stats = {'files_processed': 0, 'total_points': 0}
    
    # Collect all labels for aggregated plot
    all_original_labels = []
    all_remapped_labels = []
    
    for las_file in tqdm(las_files, desc="Remapping ForestSemantic labels", unit="file"):
        # Read both coordinates and labels from .las file
        xyz, labels = read_forestsemantic_points_and_labels(las_file)
        
        # Remap labels (passing xyz for conditional mapping)
        remapped_labels, stats = remap_labels(labels, config['mapping'], xyz=xyz, dataset='forestsemantic')
        
        # Prepare output path (.las -> .labels)
        rel_path = las_file.relative_to(input_dir)
        output_path = output_dir / rel_path.with_suffix('.labels')
        
        # Print statistics for this file
        if dry_run:
            print(f"\n[DRY RUN] Would process: {las_file.name}")
            print(f"  Original labels: {stats['original_unique']}")
            print(f"  Remapped labels: {stats['remapped_unique']}")
            print(f"  Total points: {stats['total_points']:,}")
            print(f"  Would save to: {output_path}")
        else:
            write_forestsemantic_labels(remapped_labels, output_path)
        
        overall_stats['files_processed'] += 1
        overall_stats['total_points'] += stats['total_points']
        
        # Collect for plotting
        if plot and not dry_run:
            all_original_labels.append(labels)
            all_remapped_labels.append(remapped_labels)
    
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Files processed: {overall_stats['files_processed']}")
    print(f"  Total points: {overall_stats['total_points']:,}")
    print(f"{'='*80}\n")
    
    # Generate plots
    if plot and not dry_run and PLOTTING_AVAILABLE and len(all_original_labels) > 0:
        print("\n📊 Generating comparison plots...")
        all_original = np.concatenate(all_original_labels)
        all_remapped = np.concatenate(all_remapped_labels)
        
        original_names, remapped_names = get_label_names_for_dataset('forestsemantic')
        
        # Side-by-side plot
        plot_path_side = output_dir / "label_comparison_sidebyside.png"
        plot_before_after_histogram(
            all_original, all_remapped,
            original_names, remapped_names,
            "ForestSemantic", plot_path_side
        )
        
        # Stacked plot
        plot_path_stacked = output_dir / "label_comparison_stacked.png"
        plot_stacked_comparison(
            all_original, all_remapped,
            original_names, remapped_names,
            "ForestSemantic", plot_path_stacked
        )


def process_digiforests(input_dir: Path, output_dir: Path, dry_run: bool = False, plot: bool = False) -> None:
    """
    Process DigiForests dataset: remap labels from .ply files.
    
    Args:
        input_dir: Directory containing .ply files
        output_dir: Directory to save remapped .labels files
        dry_run: If True, only print what would be done without writing files
        plot: If True, generate before/after comparison plots
    """
    config = DATASET_CONFIGS['digiforests']
    ply_files = sorted(list(input_dir.glob(f'**/*{config["file_extension"]}')))
    
    if len(ply_files) == 0:
        print(f"⚠️  No .ply files found in {input_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"Processing DigiForests Dataset")
    print(f"{'='*80}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(ply_files)} .ply files")
    print(f"Dry run: {dry_run}")
    print(f"Plot: {plot}")
    print(f"Mapping: {config['mapping']}")
    
    overall_stats = {'files_processed': 0, 'total_points': 0}
    
    # Collect all labels for aggregated plot
    all_original_labels = []
    all_remapped_labels = []
    
    for ply_file in tqdm(ply_files, desc="Remapping DigiForests labels", unit="file"):
        # Read labels from .ply file
        labels = read_digiforests_labels(ply_file)
        
        # Remap labels
        remapped_labels, stats = remap_labels(labels, config['mapping'], dataset='digiforests')
        
        # Prepare output path (.ply -> .labels)
        rel_path = ply_file.relative_to(input_dir)
        output_path = output_dir / rel_path.with_suffix('.labels')
        
        # Print statistics for this file
        if dry_run:
            print(f"\n[DRY RUN] Would process: {ply_file.name}")
            print(f"  Original labels: {stats['original_unique']}")
            print(f"  Remapped labels: {stats['remapped_unique']}")
            print(f"  Total points: {stats['total_points']:,}")
            print(f"  Would save to: {output_path}")
        else:
            write_digiforests_labels(remapped_labels, output_path)
        
        overall_stats['files_processed'] += 1
        overall_stats['total_points'] += stats['total_points']
        
        # Collect for plotting
        if plot and not dry_run:
            all_original_labels.append(labels)
            all_remapped_labels.append(remapped_labels)
    
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Files processed: {overall_stats['files_processed']}")
    print(f"  Total points: {overall_stats['total_points']:,}")
    print(f"{'='*80}\n")
    
    # Generate plots
    if plot and not dry_run and PLOTTING_AVAILABLE and len(all_original_labels) > 0:
        print("\n📊 Generating comparison plots...")
        all_original = np.concatenate(all_original_labels)
        all_remapped = np.concatenate(all_remapped_labels)
        
        original_names, remapped_names = get_label_names_for_dataset('digiforests')
        
        # Side-by-side plot
        plot_path_side = output_dir / "label_comparison_sidebyside.png"
        plot_before_after_histogram(
            all_original, all_remapped,
            original_names, remapped_names,
            "DigiForests", plot_path_side
        )
        
        # Stacked plot
        plot_path_stacked = output_dir / "label_comparison_stacked.png"
        plot_stacked_comparison(
            all_original, all_remapped,
            original_names, remapped_names,
            "DigiForests", plot_path_stacked
        )


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Remap labels for Semantic3D, ForestSemantic, and DigiForests datasets to a unified label set.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Process Semantic3D dataset
  python remap_labels.py --dataset semantic3d --input_dir /data/semantic3d/train --output_dir /data/output
  
  # Process ForestSemantic with dry-run
  python remap_labels.py --dataset forestsemantic --input_dir /data/forestsemantic --output_dir /data/output --dry-run
  
  # Process DigiForests with plots
  python remap_labels.py --dataset digiforests --input_dir /data/digiforests --output_dir /data/output --plot
        '''
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['semantic3d', 'forestsemantic', 'digiforests'],
        help='Dataset to process'
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input directory containing label files'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=False,
        help='Output directory for remapped labels (default: <input_dir>/<dataset>_remapped_labels)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually writing files'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate before/after comparison plots for label distributions'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"❌ Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        config = DATASET_CONFIGS[args.dataset]
        output_dir = input_dir.parent / config['output_dir_name']
    
    # Process dataset
    if args.dataset == 'semantic3d':
        process_semantic3d(input_dir, output_dir, args.dry_run, args.plot)
    elif args.dataset == 'forestsemantic':
        process_forestsemantic(input_dir, output_dir, args.dry_run, args.plot)
    elif args.dataset == 'digiforests':
        process_digiforests(input_dir, output_dir, args.dry_run, args.plot)
    
    if not args.dry_run:
        print(f"✅ Remapped labels saved to: {output_dir}")
    else:
        print(f"\n✅ Dry run completed. No files were written.")


if __name__ == "__main__":
    main()

