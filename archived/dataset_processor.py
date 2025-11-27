"""
Dataset-specific processing functions for Semantic3D, ForestSemantic, and DigiForests.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional
import gc
import sys
import os

# Add trunk_detection to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trunk_detection'))
from trunk_detection import TrunkDetector

from label_io import (
    read_semantic3d_points_and_labels,
    read_forestsemantic_points_and_labels,
    read_digiforests_labels,
    write_labels
)
from label_remapper import remap_labels

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


def process_semantic3d(
    input_dir: Path,
    output_dir: Path,
    mapping: dict,
    file_extension: str,
    dry_run: bool = False,
    plot: bool = False,
    specific_file: Optional[str] = None
) -> None:
    """Process Semantic3D dataset: remap .labels files."""
    
    # Get file list
    label_files = _get_file_list(input_dir, file_extension, specific_file)
    if not label_files:
        print(f"⚠️  No .labels files found in {input_dir}")
        return
    
    # Print header
    _print_processing_header('Semantic3D', input_dir, output_dir, len(label_files), 
                            dry_run, plot, mapping)
    print(f"Note: Label 3 uses TrunkDetector (PCA-based geometric features: linearity + verticality)")
    
    overall_stats = {'files_processed': 0, 'total_points': 0}
    
    # Create shared TrunkDetector instance
    print("Initializing TrunkDetector (radius=0.4)...")
    trunk_detector = TrunkDetector(
        radius=0.4, search_method='radius',
        linearity_threshold=0.8, verticality_threshold=0.9,
        min_height=-1.5, max_height=5.0, min_cluster_size=50
    )
    
    # Collect labels for plotting
    all_original_labels = []
    all_remapped_labels = []
    
    for label_file in tqdm(label_files, desc="Remapping Semantic3D labels", unit="file"):
        xyz, labels = read_semantic3d_points_and_labels(label_file)
        
        remapped_labels, stats = remap_labels(
            labels, mapping, xyz=xyz, dataset='semantic3d',
            trunk_detector=trunk_detector, chunk_size=10_000_000
        )
        
        output_path = output_dir / label_file.relative_to(input_dir)
        
        if dry_run:
            _print_dry_run_info(label_file.name, stats, output_path)
        else:
            write_labels(remapped_labels, output_path)
        
        overall_stats['files_processed'] += 1
        overall_stats['total_points'] += stats['total_points']
        
        if plot and not dry_run:
            all_original_labels.append(labels)
            all_remapped_labels.append(remapped_labels)
        
        del xyz, labels, remapped_labels, stats
        gc.collect()
    
    _print_summary(overall_stats)
    
    if plot and not dry_run and PLOTTING_AVAILABLE and len(all_original_labels) > 0:
        _generate_plots(all_original_labels, all_remapped_labels, 'semantic3d', output_dir)


def process_forestsemantic(
    input_dir: Path,
    output_dir: Path,
    mapping: dict,
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
    print(f"Note: Label 6 uses conditional mapping (height < 5m & within ±12m XY from center → 4, else → 0)")
    
    overall_stats = {'files_processed': 0, 'total_points': 0}
    all_original_labels = []
    all_remapped_labels = []
    
    for las_file in tqdm(label_files, desc="Remapping ForestSemantic labels", unit="file"):
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
    
    _print_summary(overall_stats)
    
    if plot and not dry_run and PLOTTING_AVAILABLE and len(all_original_labels) > 0:
        _generate_plots(all_original_labels, all_remapped_labels, 'forestsemantic', output_dir)


def process_digiforests(
    input_dir: Path,
    output_dir: Path,
    mapping: dict,
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
    
    for ply_file in tqdm(label_files, desc="Remapping DigiForests labels", unit="file"):
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
    
    _print_summary(overall_stats)
    
    if plot and not dry_run and PLOTTING_AVAILABLE and len(all_original_labels) > 0:
        _generate_plots(all_original_labels, all_remapped_labels, 'digiforests', output_dir)


# ============================================================================
# Helper Functions
# ============================================================================

def _get_file_list(input_dir: Path, file_extension: str, specific_file: Optional[str]) -> list:
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
    else:
        return sorted(list(input_dir.glob(f'**/*{file_extension}')))


def _print_processing_header(dataset: str, input_dir: Path, output_dir: Path,
                             n_files: int, dry_run: bool, plot: bool, mapping: dict) -> None:
    """Print processing header information."""
    print(f"\n{'='*80}")
    print(f"Processing {dataset} Dataset")
    print(f"{'='*80}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {n_files} files")
    print(f"Dry run: {dry_run}")
    print(f"Plot: {plot}")
    print(f"Mapping: {mapping}")


def _print_dry_run_info(filename: str, stats: dict, output_path: Path) -> None:
    """Print dry run information for a file."""
    print(f"\n[DRY RUN] Would process: {filename}")
    print(f"  Original labels: {stats['original_unique']}")
    print(f"  Remapped labels: {stats['remapped_unique']}")
    print(f"  Total points: {stats['total_points']:,}")
    print(f"  Would save to: {output_path}")


def _print_summary(overall_stats: dict) -> None:
    """Print processing summary."""
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Files processed: {overall_stats['files_processed']}")
    print(f"  Total points: {overall_stats['total_points']:,}")
    print(f"{'='*80}\n")


def _generate_plots(all_original_labels: list, all_remapped_labels: list,
                   dataset: str, output_dir: Path) -> None:
    """Generate comparison plots."""
    print("\n📊 Generating comparison plots...")
    all_original = np.concatenate(all_original_labels)
    all_remapped = np.concatenate(all_remapped_labels)
    
    original_names, remapped_names = get_label_names_for_dataset(dataset)
    
    plot_path_side = output_dir / "label_comparison_sidebyside.png"
    plot_before_after_histogram(
        all_original, all_remapped, original_names, remapped_names,
        dataset.capitalize(), plot_path_side
    )
    
    plot_path_stacked = output_dir / "label_comparison_stacked.png"
    plot_stacked_comparison(
        all_original, all_remapped, original_names, remapped_names,
        dataset.capitalize(), plot_path_stacked
    )
