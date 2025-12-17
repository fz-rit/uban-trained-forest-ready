
import pandas as pd
import numpy as np

# add repository root to sys.path
import sys
from pathlib import Path
import laspy

def read_semantic3d_labels(file_path: Path) -> np.ndarray:
    """Read a Semantic3D .labels file and return a numpy array of class IDs."""
    labels = pd.read_csv(file_path, header=None, sep=r'\s+', dtype=np.int32).values
    labels = labels.reshape((-1,))
    return labels

def read_semantic3d_points_and_labels(label_path: Path):
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

def write_labels(labels: np.ndarray, output_path: Path) -> None:
    """Write remapped labels to a .labels file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(labels)
    df.to_csv(output_path, header=False, index=False, sep=' ')


def get_class3_pts_and_labels(input_label_path,
                              output_dir):
    """
    Get points and labels with class_id 3 from the input files
    and save the results to the output files.

    Parameters:
    - input_label_path: Path to the input .label file
    - output_dir: Directory to save the filtered .pts and .label files
    """
    # Read points and labels
    pts, labels = read_semantic3d_points_and_labels(input_label_path)
    print(f"Original number of points: {pts.shape[0]}")

    # Filter points and labels with class_id 3
    mask = labels == 3
    filtered_pts = pts[mask]
    filtered_labels = labels[mask]  

    print(f"Number of class 3 points: {filtered_pts.shape[0]}")
    # Save filtered points
    output_pts_path = output_dir / (input_label_path.stem + '_class3.txt')
    output_label_path = output_dir / (input_label_path.stem + '_class3.labels')
    output_las_path = output_dir / (input_label_path.stem + '_class3.las')
    # write points in ASCII format with to_csv
    pd.DataFrame(filtered_pts).to_csv(output_pts_path, header=False, index=False,  sep=' ')
    print(f"Filtered points saved to: {output_pts_path}")
    # Save filtered labels in ASCII format
    # filtered_labels.tofile(output_label_path)
    write_labels(filtered_labels, output_label_path)
    print(f"Filtered labels saved to: {output_label_path}") 

    # Save filtered points to LAS file
    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    las.x = filtered_pts[:, 0]
    las.y = filtered_pts[:, 1]
    las.z = filtered_pts[:, 2]
    las.classification = filtered_labels
    las.write(output_las_path)
    print(f"Filtered LAS file saved to: {output_las_path}")


if __name__ == "__main__":
    input_label_path = Path("/home/fzhcis/mylab/data/semantic3d/isolated_temp/inputs/sg27_station5_intensity_rgb.labels")
    output_dir = Path("/home/fzhcis/mylab/data/semantic3d/isolated_temp/class3")
    get_class3_pts_and_labels(input_label_path, output_dir)