#!/usr/bin/env python3
"""
Remap labels in Semantic3D, ForestSemantic, and DigitalForest datasets to a unified label set.

Label Mappings:
- semantic3d_mapping = {0: 0, 1: 1, 2: 1, 3: [2, 3], 4: 4, 5: 5, 6: 5, 7: 5, 8: 5}
  * Label 3 uses conditional mapping based on geometric features (TrunkDetector)
- forest_semantic_mapping = {1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: [0,4], 7: 0}
  * Label 6 uses conditional mapping based on height and position
- digiforests_mapping = {0: 0, 1: 1, 2: 4, 3: 2, 4: 3, 5: 5}

Usage:
    python remap_labels.py --dataset semantic3d --input_dir /path/to/data [--output_dir /path/to/output] [--dry-run] [--plot] [--file filename]
"""

import argparse
import sys
from pathlib import Path

from dataset_processor import process_semantic3d, process_forestsemantic, process_digiforests


# ============================================================================
# Configuration
# ============================================================================

DATASET_CONFIGS = {
    'semantic3d': {
        'mapping': {0: 0, 1: 1, 2: 1, 3: [2, 3], 4: 4, 5: 5, 6: 5, 7: 5, 8: 5},
        # 'mapping': {0: 0, 1: 0, 2: 1, 3: [2, 3], 4: 4, 5: 0, 6: 0, 7: 0, 8: 5},
        'file_extension': '.labels',
        'output_dir_name': 'semantic3d_remapped_labels',
        'processor': process_semantic3d
    },
    'forestsemantic': {
        'mapping': {1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: [0, 4], 7: 0},
        'file_extension': '.las',
        'output_dir_name': 'forestsemantic_remapped_labels',
        'processor': process_forestsemantic
    },
    'digiforests': {
        'mapping': {0: 0, 1: 1, 2: 4, 3: 2, 4: 3, 5: 5},
        'file_extension': '.ply',
        'output_dir_name': 'digiforests_remapped_labels',
        'processor': process_digiforests
    }
}


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Remap labels for Semantic3D, ForestSemantic, and DigiForests datasets to a unified label set.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Process entire dataset
  python remap_labels.py --dataset semantic3d --input_dir /data/semantic3d/train
  
  # Process with custom output directory
  python remap_labels.py --dataset forestsemantic --input_dir /data/forestsemantic --output_dir /data/output
  
  # Dry run (preview without writing)
  python remap_labels.py --dataset digiforests --input_dir /data/digiforests --dry-run
  
  # Process with visualization plots
  python remap_labels.py --dataset semantic3d --input_dir /data/semantic3d --plot
  
  # Process specific file only
  python remap_labels.py --dataset semantic3d --input_dir /data/semantic3d/train --file "scene01.labels"
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
    
    parser.add_argument(
        '--file',
        type=str,
        required=False,
        help='Process only a specific file (relative path from input_dir)'
    )

    parser.add_argument(
        '--trunk-backend',
        type=str,
        default='cpu',
        choices=['cpu', 'gpu'],
        help='Backend for trunk detection in Semantic3D (cpu or gpu)'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        default=100_000,
        help='Max label-3 points to process per chunk (Semantic3D). Set 0 to disable chunking.'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"❌ Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Get dataset configuration
    config = DATASET_CONFIGS[args.dataset]
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir.parent / config['output_dir_name']
    
    processor_kwargs = dict(
        input_dir=input_dir,
        output_dir=output_dir,
        mapping=config['mapping'],
        file_extension=config['file_extension'],
        dry_run=args.dry_run,
        plot=args.plot,
        specific_file=args.file
    )

    if args.dataset == 'semantic3d':
        chunk_size = None if args.chunk_size is None or args.chunk_size <= 0 else args.chunk_size
        processor_kwargs['trunk_backend'] = args.trunk_backend
        processor_kwargs['chunk_size'] = chunk_size

    config['processor'](**processor_kwargs)
    
    # Final message
    if not args.dry_run:
        print(f"✅ Remapped labels saved to: {output_dir}")
    else:
        print(f"\n✅ Dry run completed. No files were written.")


if __name__ == "__main__":
    main()
