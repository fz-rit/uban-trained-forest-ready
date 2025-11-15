'''
Remap labels in Semantic3D, ForestSemantic, and DigitalForest datasets to a unified label set.

Label Mappings:
- semantic3d_mapping = {0: 0, 1: 1, 2: 1, 3: 3, 4: 4, 5: 5, 6: 5, 7: 5, 8: 5}
- forest_semantic_mapping = {1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 5, 7: 0}
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
from typing import Dict, List, Tuple
import sys

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

SEMANTIC3D_MAPPING = {0: 0, 1: 1, 2: 1, 3: 3, 4: 4, 5: 5, 6: 5, 7: 5, 8: 5}
FOREST_SEMANTIC_MAPPING = {1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 5, 7: 0}
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

def remap_labels(labels: np.ndarray, mapping: Dict[int, int]) -> Tuple[np.ndarray, Dict]:
    """
    Remap labels according to the provided mapping dictionary.
    
    Args:
        labels: Original labels array
        mapping: Dictionary mapping original labels to new labels
        
    Returns:
        Tuple of (remapped_labels, statistics_dict)
    """
    remapped = labels.copy()
    
    # Collect statistics
    unique_original = np.unique(labels)
    original_counts = {int(label): int(np.sum(labels == label)) for label in unique_original}
    
    # Apply mapping
    for old_label, new_label in mapping.items():
        remapped[labels == old_label] = new_label
    
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
    
    overall_stats = {'files_processed': 0, 'total_points': 0}
    
    # Collect all labels for aggregated plot
    all_original_labels = []
    all_remapped_labels = []
    
    for label_file in tqdm(label_files, desc="Remapping Semantic3D labels", unit="file"):
        # Read labels
        labels = read_semantic3d_labels(label_file)
        
        # Remap labels
        remapped_labels, stats = remap_labels(labels, config['mapping'])
        
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
    
    overall_stats = {'files_processed': 0, 'total_points': 0}
    
    # Collect all labels for aggregated plot
    all_original_labels = []
    all_remapped_labels = []
    
    for las_file in tqdm(las_files, desc="Remapping ForestSemantic labels", unit="file"):
        # Read labels from .las file
        labels = read_forestsemantic_labels(las_file)
        
        # Remap labels
        remapped_labels, stats = remap_labels(labels, config['mapping'])
        
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
        remapped_labels, stats = remap_labels(labels, config['mapping'])
        
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

Label Mappings:
  Semantic3D:      {0: 0, 1: 1, 2: 1, 3: 3, 4: 4, 5: 5, 6: 5, 7: 5, 8: 5}
  ForestSemantic:  {1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 5, 7: 0}
  DigiForests:     {0: 0, 1: 1, 2: 4, 3: 2, 4: 3, 5: 5}
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

