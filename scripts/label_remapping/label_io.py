"""
Input/Output functions for reading and writing labels from different dataset formats.
"""

import numpy as np
import pandas as pd
import laspy
import plyfile
from pathlib import Path
from typing import Tuple


# ============================================================================
# Reading Functions
# ============================================================================

def read_semantic3d_labels(file_path: Path) -> np.ndarray:
    """Read a Semantic3D .labels file and return a numpy array of class IDs."""
    labels = pd.read_csv(file_path, header=None, sep=r'\s+', dtype=np.int32).values
    labels = labels.reshape((-1,))
    return labels


def read_semantic3d_points_and_labels(label_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read both XYZ coordinates and labels from Semantic3D files."""
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
    """Read labels from a ForestSemantic .las file (classification field)."""
    las = laspy.read(str(file_path))
    labels = np.array(las.classification, dtype=np.int32).reshape((-1,))
    return labels


def read_forestsemantic_points_and_labels(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read both XYZ coordinates and labels from a ForestSemantic .las file."""
    las = laspy.read(str(file_path))
    xyz = np.vstack([
        las.X * las.header.scale[0] + las.header.offsets[0],
        las.Y * las.header.scale[1] + las.header.offsets[1],
        las.Z * las.header.scale[2] + las.header.offsets[2]
    ]).T.astype(np.float32)
    labels = np.array(las.classification, dtype=np.int32).reshape((-1,))
    return xyz, labels


def read_digiforests_labels(file_path: Path) -> np.ndarray:
    """Read labels from a DigiForests .ply file (semantics field)."""
    plydata = plyfile.PlyData.read(str(file_path))
    labels = np.array(plydata['vertex'].data['semantics'], dtype=np.int32)
    return labels


# ============================================================================
# Writing Functions
# ============================================================================

def write_labels(labels: np.ndarray, output_path: Path) -> None:
    """Write remapped labels to a .labels file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(labels)
    df.to_csv(output_path, header=False, index=False, sep=' ')
