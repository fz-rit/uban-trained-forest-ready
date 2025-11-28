"""
Dataset-specific processing functions for Semantic3D, ForestSemantic, and DigiForests.
"""

import gc
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import laspy
from tqdm import tqdm

# Add trunk_detection to path
CURRENT_DIR = Path(__file__).resolve().parent
TRUNK_DIR = CURRENT_DIR / 'semantic3d_trunk_detection'
if str(TRUNK_DIR) not in sys.path:
    sys.path.insert(0, str(TRUNK_DIR))

# from trunk_detection import TrunkDetector
from trunk_detection_torch_projection import TorchTrunkDetector

from label_io import (
    read_semantic3d_points_and_labels,
    read_forestsemantic_points_and_labels,
    read_digiforests_labels,
    write_labels
)
from label_remapper import remap_labels
from plot_label_comparison import (
    plot_before_after_histogram,
    plot_stacked_comparison,
    get_label_names_for_dataset
)


def save_to_las(
    xyz: np.ndarray, 
    labels: np.ndarray, 
    remapped_labels: np.ndarray, 
    output_path: Path,
    features: Optional[Dict[str, np.ndarray]] = None
) -> None:
    """
    Save original labels, remapped labels, and optional extra features to LAS.
    """
    assert xyz.shape[0] == labels.shape[0] == remapped_labels.shape[0], \
        f"Mismatched point counts: \n" \
        f"n_xyz={xyz.shape[0]}, n_labels={labels.shape[0]}, n_remapped={remapped_labels.shape[0]}"
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.min(xyz, axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])

    # Standard Extra Dim for original label
    header.add_extra_dim(
        laspy.ExtraBytesParams(name="original_label", type=np.uint8, description="Original Raw Labels")
    )

    # Add Dimensions for computed features (e.g., coverage, ratio)
    if features:
        for name, data in features.items():
            # Choose type based on data
            dtype = np.float32 if np.issubdtype(data.dtype, np.floating) else np.int32
            header.add_extra_dim(
                laspy.ExtraBytesParams(name=name, type=dtype, description=f"Computed Feature: {name}")
            )

    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    
    las.classification = remapped_labels.astype(np.uint8)
    las.original_label = labels.astype(np.uint8)

    # Assign Feature Data
    if features:
        for name, data in features.items():
            setattr(las, name, data)

    las.write(output_path)


def process_semantic3d(
    input_dir: Path,
    output_dir: Path,
    mapping: Dict[int, int],
    file_extension: str,
    dry_run: bool = False,
    plot: bool = False,
    specific_file: Optional[str] = None,
    chunk_size: Optional[int] = 10_000
) -> None:
    """Process Semantic3D dataset: remap .labels files using geometric features."""
    
    label_files = _get_file_list(input_dir, file_extension, specific_file)
    if not label_files:
        print(f"⚠️  No .labels files found in {input_dir}")
        return
    
    _print_processing_header('Semantic3D', input_dir, output_dir, len(label_files), 
                            dry_run, plot, mapping)

    print("🚀 Using TorchTrunkDetector (GPU Backend: Projection Ratio Method)...")
    trunk_detector = TorchTrunkDetector(
        radius=0.2,
        grid_resolution=0.01,
        projection_ratio_threshold=0.4,
        min_height=-1.5,
        max_height=3.0,
        min_cluster_size=2000,
        min_neighbors=15,
        max_neighbors=2048,
    )

    overall_stats = {'files_processed': 0, 'total_points': 0}
    all_original_labels = []
    all_remapped_labels = []
    
    for label_file in tqdm(label_files, desc="Remapping Semantic3D", unit="file"):
        xyz, labels = read_semantic3d_points_and_labels(label_file)
        
        remapped_labels, stats = remap_labels(
            labels, mapping, xyz=xyz, dataset='semantic3d',
            trunk_detector=trunk_detector, chunk_size=chunk_size
        )
        
        # 🟢 Extract intermediate features if available
        features_to_save = {}
        if trunk_detector and hasattr(trunk_detector, 'features') and trunk_detector.features:
            features_to_save = trunk_detector.features

        output_path = output_dir / label_file.relative_to(input_dir)
        
        if dry_run:
            _print_dry_run_info(label_file.name, stats, output_path)
        else:
            write_labels(remapped_labels, output_path)
            
            # # Save LAS with features
            # save_to_las(
            #     xyz, 
            #     labels, 
            #     remapped_labels, 
            #     output_path.with_suffix('.las'),
            #     features=features_to_save
            # )
        
        overall_stats['files_processed'] += 1
        overall_stats['total_points'] += stats['total_points']
        
        if plot and not dry_run:
            all_original_labels.append(labels)
            all_remapped_labels.append(remapped_labels)
        
        del xyz, labels, remapped_labels, stats, features_to_save
        gc.collect()
    
    _print_summary(overall_stats)
    
    if plot and not dry_run and all_original_labels:
        _generate_plots(all_original_labels, all_remapped_labels, 'semantic3d', output_dir)


def process_forestsemantic(
    input_dir: Path,
    output_dir: Path,
    mapping: Dict[int, int],
    file_extension: str,
    dry_run: bool = False,
    plot: bool = False,
    specific_file: Optional[str] = None
) -> None:
    """Process ForestSemantic dataset: remap labels from .las files."""
    
    label_files = _get_file_list(input_dir, file_extension, specific_file)
    if not label_files:
        print(f"⚠️  No .las files found in {input_dir}")
        return
    
    _print_processing_header('ForestSemantic', input_dir, output_dir, len(label_files),
                            dry_run, plot, mapping)
    
    overall_stats = {'files_processed': 0, 'total_points': 0}
    all_original_labels = []
    all_remapped_labels = []
    
    for las_file in tqdm(label_files, desc="Remapping ForestSemantic", unit="file"):
        xyz, labels = read_forestsemantic_points_and_labels(las_file)
        
        remapped_labels, stats = remap_labels(
            labels, mapping, xyz=xyz, dataset='forestsemantic'
        )
        
        output_path = output_dir / las_file.relative_to(input_dir).with_suffix('.labels')
        
        if dry_run:
            _print_dry_run_info(las_file.name, stats, output_path)
        else:
            write_labels(remapped_labels, output_path)
        
        overall_stats['files_processed'] += 1
        overall_stats['total_points'] += stats['total_points']
        
        if plot and not dry_run:
            all_original_labels.append(labels)
            all_remapped_labels.append(remapped_labels)

        del xyz, labels, remapped_labels
        gc.collect()
    
    _print_summary(overall_stats)
    
    if plot and not dry_run and all_original_labels:
        _generate_plots(all_original_labels, all_remapped_labels, 'forestsemantic', output_dir)


def process_digiforests(
    input_dir: Path,
    output_dir: Path,
    mapping: Dict[int, int],
    file_extension: str,
    dry_run: bool = False,
    plot: bool = False,
    specific_file: Optional[str] = None
) -> None:
    """Process DigiForests dataset: remap labels from .ply files."""
    
    label_files = _get_file_list(input_dir, file_extension, specific_file)
    if not label_files:
        print(f"⚠️  No .ply files found in {input_dir}")
        return
    
    _print_processing_header('DigiForests', input_dir, output_dir, len(label_files),
                            dry_run, plot, mapping)
    
    overall_stats = {'files_processed': 0, 'total_points': 0}
    all_original_labels = []
    all_remapped_labels = []
    
    for ply_file in tqdm(label_files, desc="Remapping DigiForests", unit="file"):
        labels = read_digiforests_labels(ply_file)
        
        remapped_labels, stats = remap_labels(labels, mapping, dataset='digiforests')
        
        output_path = output_dir / ply_file.relative_to(input_dir).with_suffix('.labels')
        
        if dry_run:
            _print_dry_run_info(ply_file.name, stats, output_path)
        else:
            write_labels(remapped_labels, output_path)
        
        overall_stats['files_processed'] += 1
        overall_stats['total_points'] += stats['total_points']
        
        if plot and not dry_run:
            all_original_labels.append(labels)
            all_remapped_labels.append(remapped_labels)
            
        del labels, remapped_labels
        gc.collect()
    
    _print_summary(overall_stats)
    
    if plot and not dry_run and all_original_labels:
        _generate_plots(all_original_labels, all_remapped_labels, 'digiforests', output_dir)


# ============================================================================
# Helper Functions
# ============================================================================

def _get_file_list(input_dir: Path, file_extension: str, specific_file: Optional[str]) -> List[Path]:
    """Get list of files to process."""
    if specific_file:
        file_path = input_dir / specific_file
        if not file_path.exists():
            print(f"❌ Error: File not found: {file_path}")
            return []
        if not str(file_path).endswith(file_extension):
            print(f"❌ Error: File must have {file_extension} extension")
            return []
        return [file_path]
    
    return sorted(list(input_dir.glob(f'**/*{file_extension}')))


def _print_processing_header(
    dataset: str, input_dir: Path, output_dir: Path, n_files: int, 
    dry_run: bool, plot: bool, mapping: dict
) -> None:
    """Print processing header information."""
    print(f"\n{'='*80}")
    print(f"Processing {dataset} Dataset")
    print(f"{'='*80}")
    print(f"Input:    {input_dir}")
    print(f"Output:   {output_dir}")
    print(f"Files:    {n_files}")
    print(f"Dry run:  {dry_run}")
    print(f"Plotting: {plot}")
    print(f"Mapping:  {mapping}")


def _print_dry_run_info(filename: str, stats: dict, output_path: Path) -> None:
    """Print dry run information for a file."""
    print(f"\n[DRY RUN] {filename}")
    print(f"  └── Original Classes: {stats['original_unique']}")
    print(f"  └── Remapped Classes: {stats['remapped_unique']}")
    print(f"  └── Points:           {stats['total_points']:,}")
    print(f"  └── Destination:      {output_path}")


def _print_summary(overall_stats: dict) -> None:
    """Print processing summary."""
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Files processed: {overall_stats['files_processed']}")
    print(f"  Total points:    {overall_stats['total_points']:,}")
    print(f"{'='*80}\n")


def _generate_plots(
    all_original_labels: List[np.ndarray], 
    all_remapped_labels: List[np.ndarray],
    dataset: str, 
    output_dir: Path
) -> None:
    """Generate comparison plots."""
    print("\n📊 Generating comparison plots...")
    all_original = np.concatenate(all_original_labels)
    all_remapped = np.concatenate(all_remapped_labels)
    
    original_names, remapped_names = get_label_names_for_dataset(dataset)
    
    plot_before_after_histogram(
        all_original, all_remapped, original_names, remapped_names,
        dataset.capitalize(), output_dir / "label_comparison_sidebyside.png"
    )
    
    plot_stacked_comparison(
        all_original, all_remapped, original_names, remapped_names,
        dataset.capitalize(), output_dir / "label_comparison_stacked.png"
    )