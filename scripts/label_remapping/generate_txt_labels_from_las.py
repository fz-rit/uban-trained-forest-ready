'''
Generate .txt and .labels files from .las files for Semantic3D dataset.
'''

import numpy as np
import laspy
from pathlib import Path
from typing import Tuple


def generate_semantic3d_txt_and_labels_from_las(las_file: Path, output_dir: Path) -> None:
    """Generate .txt and .labels files for Semantic3D from a .las file."""
    # Read the .las file
    las = laspy.read(str(las_file))
    
    # Extract XYZ coordinates
    xyz = np.vstack([
        las.X * las.header.scale[0] + las.header.offsets[0],
        las.Y * las.header.scale[1] + las.header.offsets[1],
        las.Z * las.header.scale[2] + las.header.offsets[2]
    ]).T.astype(np.float32)
    
    # Extract labels from classification field
    labels = np.array(las.classification, dtype=np.int32).reshape((-1,))
    
    if len(xyz) != len(labels):
        raise ValueError(f"Mismatch: {len(xyz)} points vs {len(labels)} labels")
    
    # Prepare output paths
    output_txt_path = output_dir / (las_file.stem + '.txt')
    output_labels_path = output_dir / (las_file.stem + '.labels')
    
    # Save .txt file (x y z)
    np.savetxt(output_txt_path, xyz, fmt='%.6f', delimiter=' ')
    
    # Save .labels file
    np.savetxt(output_labels_path, labels, fmt='%d', delimiter=' ')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Semantic3D .txt and .labels files from .las files.")
    parser.add_argument("las_file", type=Path, help="Path to the input .las file.")
    parser.add_argument("output_dir", type=Path, help="Directory to save the output .txt and .labels files.")
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_semantic3d_txt_and_labels_from_las(args.las_file, args.output_dir)