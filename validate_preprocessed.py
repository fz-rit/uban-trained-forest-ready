#!/usr/bin/env python3
"""
Validation script for preprocessed point cloud data.
Reads a .pt file and visualizes it using Open3D.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import open3d as o3d

# Color map for semantic classes (BGR format for Open3D)
SEMANTIC_COLORS = {
    0: [0.5, 0.5, 0.5],    # unlabeled/noise - gray
    1: [0.6, 0.4, 0.2],    # ground - brown
    2: [0.2, 0.8, 0.3],    # shrub - light green
    3: [0.4, 0.3, 0.1],    # stem - dark brown
    4: [0.1, 0.6, 0.1],    # canopy - dark green
    255: [0.0, 0.0, 0.0],  # ignore - black
}

def load_pt_file(pt_path: Path):
    """Load preprocessed .pt file."""
    data = torch.load(pt_path)
    print(f"\n{'='*60}")
    print(f"Loaded: {pt_path.name}")
    print(f"{'='*60}")
    
    # Extract data
    points = data["points"].numpy()
    feats = data["feats"].numpy()
    labels = data["labels"].numpy()
    instance_ids = data["instance_ids"].numpy()
    offsets = data["offset"].numpy()
    center = data["center"].numpy()
    meta = data["meta"]
    
    # Print statistics
    print(f"\n📊 Point Cloud Statistics:")
    print(f"  Points:        {points.shape[0]:,}")
    print(f"  Features:      {feats.shape}")
    print(f"  Center:        [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
    
    print(f"\n🏷️  Semantic Labels:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        pct = 100 * count / len(labels)
        label_name = get_label_name(label)
        print(f"  Class {label:3d} ({label_name:10s}): {count:7,} points ({pct:5.1f}%)")
    
    print(f"\n🎯 Instance Labels:")
    unique_inst = np.unique(instance_ids)
    print(f"  Unique instances: {len(unique_inst)}")
    if len(unique_inst) <= 20:
        inst_counts = [(inst, np.sum(instance_ids == inst)) for inst in unique_inst]
        for inst, count in sorted(inst_counts, key=lambda x: x[1], reverse=True):
            if inst == 0:
                print(f"  Instance {inst:3d} (noise):     {count:7,} points")
            else:
                print(f"  Instance {inst:3d}:            {count:7,} points")
    else:
        print(f"  (Too many instances to display individually)")
    
    print(f"\n📝 Metadata:")
    for key, value in meta.items():
        print(f"  {key}: {value}")
    
    print(f"\n📐 Bounding Box:")
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    dims = maxs - mins
    print(f"  Min: [{mins[0]:7.2f}, {mins[1]:7.2f}, {mins[2]:7.2f}]")
    print(f"  Max: [{maxs[0]:7.2f}, {maxs[1]:7.2f}, {maxs[2]:7.2f}]")
    print(f"  Dim: [{dims[0]:7.2f}, {dims[1]:7.2f}, {dims[2]:7.2f}]")
    
    print(f"{'='*60}\n")
    
    return {
        "points": points,
        "feats": feats,
        "labels": labels,
        "instance_ids": instance_ids,
        "offsets": offsets,
        "center": center,
        "meta": meta
    }

def get_label_name(label_id):
    """Get human-readable label name."""
    names = {
        0: "noise",
        1: "ground",
        2: "shrub",
        3: "stem",
        4: "canopy",
        255: "ignore"
    }
    return names.get(label_id, "unknown")

def colorize_by_semantic(points, labels):
    """Color points by semantic class."""
    colors = np.zeros((len(points), 3), dtype=np.float32)
    for label_id, color in SEMANTIC_COLORS.items():
        mask = (labels == label_id)
        colors[mask] = color
    return colors

def colorize_by_instance(points, instance_ids):
    """Color points by instance ID using random colors."""
    colors = np.zeros((len(points), 3), dtype=np.float32)
    unique_inst = np.unique(instance_ids)
    
    # Generate distinct colors for each instance
    np.random.seed(42)  # for reproducibility
    for i, inst_id in enumerate(unique_inst):
        if inst_id == 0:
            colors[instance_ids == inst_id] = [0.5, 0.5, 0.5]  # gray for noise
        else:
            color = np.random.rand(3)
            colors[instance_ids == inst_id] = color
    
    return colors

def colorize_by_intensity(points, feats):
    """Color points by intensity feature."""
    intensity = feats.squeeze()
    # Map intensity to grayscale
    colors = np.stack([intensity, intensity, intensity], axis=1)
    return colors

def visualize_point_cloud(data, mode="semantic"):
    """Visualize point cloud with Open3D."""
    points = data["points"]
    labels = data["labels"]
    instance_ids = data["instance_ids"]
    feats = data["feats"]
    offsets = data["offsets"]
    
    # Create point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Colorize based on mode
    if mode == "semantic":
        colors = colorize_by_semantic(points, labels)
        title = "Semantic Segmentation"
    elif mode == "instance":
        colors = colorize_by_instance(points, instance_ids)
        title = "Instance Segmentation"
    elif mode == "intensity":
        colors = colorize_by_intensity(points, feats)
        title = "Intensity Features"
    elif mode == "offset":
        # Visualize offset magnitude as color
        offset_mag = np.linalg.norm(offsets, axis=1)
        offset_mag_norm = (offset_mag - offset_mag.min()) / (offset_mag.max() - offset_mag.min() + 1e-6)
        colors = np.stack([offset_mag_norm, offset_mag_norm, offset_mag_norm], axis=1)
        title = "Offset Magnitude"
    else:
        colors = np.ones((len(points), 3)) * 0.5
        title = "Point Cloud"
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    
    # Visualize
    print(f"\n🎨 Visualizing: {title}")
    print("Controls:")
    print("  - Mouse: Rotate/pan/zoom")
    print("  - Q/Esc: Close window")
    print("  - H: Help menu")
    o3d.visualization.draw_geometries(
        [pcd, coord_frame],
        window_name=title,
        width=1280,
        height=720,
        point_show_normal=False
    )

def main():
    parser = argparse.ArgumentParser(description="Validate and visualize preprocessed point cloud data")
    parser.add_argument("pt_file", type=Path, help="Path to .pt file")
    parser.add_argument("--mode", type=str, default="semantic",
                       choices=["semantic", "instance", "intensity", "offset"],
                       help="Visualization mode (default: semantic)")
    parser.add_argument("--no-viz", action="store_true",
                       help="Skip visualization, only print statistics")
    
    args = parser.parse_args()
    
    if not args.pt_file.exists():
        print(f"❌ Error: File not found: {args.pt_file}")
        return
    
    # Load and validate
    data = load_pt_file(args.pt_file)
    
    # Visualize
    if not args.no_viz:
        visualize_point_cloud(data, mode=args.mode)
    else:
        print("✅ Validation complete (visualization skipped)")

if __name__ == "__main__":
    main()
