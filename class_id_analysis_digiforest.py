import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# from preprocessing_digiforests import read_pt_label
from pprint import pprint
import pandas as pd
from tqdm import tqdm
import plyfile
import open3d as o3d
plt.rcParams.update({
    'font.size': 8,         # base font size
    'axes.titlesize': 12,    # title size
    'axes.labelsize': 10,    # x/y label size
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12
})

CLASS_MAP = {
    0: 'Unlabeled',
    1: 'Ground',
    2: 'Shrub',
    3: 'Stem',
    4: 'Canopy',
    5: 'Miscellaneous'
}


def aggregate_semantics(semantics: np.ndarray):
    """
    Aggregate semantics: 1) map 2-1, 4-2, 5-3, 6-4, and merge 98, 99 into 5, using numpy vectorized operations.;
    Exclude all other classe values.
    """
    mapping = {
        0: 0,   # Unlabeled
        2: 1,   # Ground
        4: 2,   # Shrub
        5: 3,   # Stem
        6: 4,   # Canopy
        98: 5,  # class 98 -> Miscellaneous
        99: 5   # class 99 -> Miscellaneous
    }
    for original_class, mapped_class in mapping.items():
        semantics[semantics == original_class] = mapped_class
    # Exclude all other class values
    valid_classes = set(mapping.values())
    mask = np.isin(semantics, list(valid_classes))
    semantics = semantics[mask]
    return semantics, mask

def read_semantics_from_ply_file(ply_path: Path, export_cleaned=True):
    """
    The PLY files are aggregated ones from multiple scans (.pt files).
    DigiForest has prepared an repository for aggregating the .pt files into PLY files:
    https://github.com/fz-rit/digiforests.

    The PLY files contain instance and semantics as two scalar fields, 
    apart from its X/Y/Z coordinates.
    """
    
    # Read PLY file using Open3D
    pcd = o3d.io.read_point_cloud(str(ply_path))
    
    # Get points and semantics
    xyz = np.asarray(pcd.points)
    
    # Read original ply to get semantics attribute
    plydata = plyfile.PlyData.read(str(ply_path))
    semantics = np.array(plydata['vertex'].data['semantics'])
    
    # Aggregate semantics
    semantics_cleaned, mask = aggregate_semantics(semantics.copy())

    if export_cleaned:
        # Apply mask to filter points
        xyz_cleaned = xyz[mask]
        assert len(xyz_cleaned) == len(semantics_cleaned), "Length mismatch after cleaning semantics."
        
        # Create cleaned point cloud
        pcd_cleaned = o3d.geometry.PointCloud()
        pcd_cleaned.points = o3d.utility.Vector3dVector(xyz_cleaned)
        
        # Add semantics as colors for visualization (optional)
        # Or save as custom attribute using numpy
        write_path = ply_path.parent / "semantics_cleaned" / f"{ply_path.stem}_cleaned.ply"
        write_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create structured array for writing with plyfile (vectorized)
        vertex_data = np.empty(len(xyz_cleaned), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('semantics', 'i4')])
        vertex_data['x'] = xyz_cleaned[:, 0]
        vertex_data['y'] = xyz_cleaned[:, 1]
        vertex_data['z'] = xyz_cleaned[:, 2]
        vertex_data['semantics'] = semantics_cleaned.flatten()
        
        cleaned_plydata = plyfile.PlyData([
            plyfile.PlyElement.describe(vertex_data, 'vertex')
        ])
        cleaned_plydata.write(str(write_path))

    return semantics_cleaned


    

def grab_class_id(file_paths, export_cleaned=True):
    """
    Read all CSV files and concatenate the class_id columns into a single array.
    """
    class_id_ls = []
    for file_path in tqdm(file_paths, desc="Reading .ply files", unit="file", colour='green'):
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        
        class_id_col = read_semantics_from_ply_file(file_path, export_cleaned=export_cleaned)
       
        class_id_ls.append(class_id_col)
    class_id_all = np.concatenate(class_id_ls)
    return class_id_ls, class_id_all

def plot_class_id_hist(class_id_vec, class_map, save_path):
    """
    Plot histogram for all class_id values with a second y-axis showing class ratios.
    Handles discrete values with gaps properly (e.g., 0, 1, 2, 3, 98, 99).

    Parameters:
    - class_id_vec: np.ndarray of class ids (1D array)
    - class_map: dict mapping class id to class name (e.g., {0: 'Ground', 1: 'Tree'})
    - save_path: path to save the figure
    """
    FONTSIZE = 16

    class_ids = np.unique(class_id_vec)
    counts = np.array([(class_id_vec == cid).sum() for cid in class_ids])
    ratios = counts / counts.sum()

    # Create evenly spaced positions for discrete class IDs
    x_positions = np.arange(len(class_ids))
    
    fig, ax1 = plt.subplots(figsize=(max(10, len(class_ids) * 1.2), 6))

    # Histogram (left y-axis) - use x_positions for even spacing
    bars = ax1.bar(x_positions, counts, width=0.6, edgecolor='black', align='center', color='skyblue')
    ax1.set_ylabel('Number of Points', color='blue', fontsize=FONTSIZE)
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=FONTSIZE)
    
    # Set x-axis with proper labels
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(
        [f"{class_map.get(cid, f'Class {cid}')}\n(ID: {cid})" for cid in class_ids], 
        rotation=45, 
        ha='right',
        fontsize=FONTSIZE
    )
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xlim(-0.5, len(class_ids) - 0.5)  # Add padding on edges

    # Add counts to top of each bar
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:,}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=FONTSIZE)

    # Ratio line (right y-axis) - use x_positions for alignment
    ax2 = ax1.twinx()
    ax2.plot(x_positions, ratios, 'o-', color='darkred', label='Ratio', markersize=8, linewidth=2)
    ax2.set_ylabel('Ratio (%)', color='darkred', fontsize=FONTSIZE)
    ax2.tick_params(axis='y', labelcolor='darkred', labelsize=FONTSIZE)
    ax2.set_ylim(0, max(ratios) * 1.2)
    ax2.set_yticks(np.linspace(0, max(ratios) * 1.2, 5))
    ax2.set_yticklabels([f"{r*100:.1f}%" for r in np.linspace(0, max(ratios) * 1.2, 5)], fontsize=FONTSIZE)

    # Add ratio text next to each dot
    for x_pos, y in zip(x_positions, ratios):
        ax2.annotate(f'{y*100:.1f}%',
                     xy=(x_pos, y),
                     xytext=(5, 0),
                     textcoords='offset points',
                     ha='left', va='center', fontsize=FONTSIZE, color='darkred')

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    return counts, ratios



def main():
    root_dir = Path(f"/home/fzhcis/data/DigiForests/aggregate_outputs/")
    key = "train"
    export_cleaned = True
    glob_pattern = f"*_{key}_*.ply" if key in ["train", "val"] else f"*.ply"
    pcd_paths = sorted(list(root_dir.glob(glob_pattern)))
    class_id_ls, class_id_all = grab_class_id(pcd_paths, export_cleaned=export_cleaned)
    prefix = f"digiforest_{key}"
    save_path = root_dir / f"{prefix}_class_id_histogram.png"
    counts, ratios = plot_class_id_hist(class_id_all, CLASS_MAP, save_path=save_path)
    output_csv = root_dir / f"{prefix}_class_id_distribution.csv"
    df = pd.DataFrame({
        'Class ID': np.unique(class_id_all),
        'Class Name': [CLASS_MAP[cid] for cid in np.unique(class_id_all)],
        'Count': counts,
        'Ratio (%)': ratios * 100
    })
    df.to_csv(output_csv, index=False)
    print(f"Class ID distribution saved to {output_csv}")

if __name__ == "__main__":
    main()