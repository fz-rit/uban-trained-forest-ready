"""
Example script demonstrating trunk detection on a LAS file.

This script shows different usage patterns and parameter configurations.
"""

import numpy as np
from trunk_detection import TrunkDetector, process_las_file


def example_basic_usage(input_file: str, output_file: str):
    """
    Example 1: Basic usage with default parameters.
    """
    print("=" * 70)
    print("EXAMPLE 1: Basic Usage (Default Parameters)")
    print("=" * 70)
    
    # Simple one-line processing
    process_las_file(input_file, output_file, verbose=True)


def example_custom_parameters(input_file: str, output_file: str):
    """
    Example 2: Custom parameters for specific forest conditions.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Custom Parameters")
    print("=" * 70)
    
    # Create detector with custom parameters
    # - Use kNN instead of radius search
    # - Larger neighborhood (100 neighbors)
    # - More strict linearity requirement (0.85)
    # - Allow taller trunks (up to 20m)
    detector = TrunkDetector(
        search_method='knn',
        k_neighbors=100,
        min_neighbors=40,
        linearity_threshold=0.85,
        verticality_threshold=0.9,
        min_height=0.5,
        max_height=20.0,
        min_cluster_size=100
    )
    
    process_las_file(input_file, output_file, detector, verbose=True)


def example_dense_forest(input_file: str, output_file: str):
    """
    Example 3: Parameters optimized for dense forest with thin trunks.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Dense Forest Configuration")
    print("=" * 70)
    
    # Smaller radius for dense forests
    # Lower cluster size to detect thinner trunks
    detector = TrunkDetector(
        radius=0.3,
        search_method='radius',
        min_neighbors=25,
        linearity_threshold=0.75,
        verticality_threshold=0.85,
        min_height=0.3,
        max_height=15.0,
        min_cluster_size=30
    )
    
    process_las_file(input_file, output_file, detector, verbose=True)


def example_programmatic_usage(input_file: str):
    """
    Example 4: Programmatic usage without writing to file.
    Process points directly and access results.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Programmatic Usage")
    print("=" * 70)
    
    import laspy
    
    # Read LAS file
    las = laspy.read(input_file)
    points = np.vstack([las.x, las.y, las.z]).T
    
    print(f"Loaded {len(points):,} points")
    
    # Get original labels
    if hasattr(las, 'classification'):
        original_labels = np.array(las.classification)
    else:
        original_labels = np.full(len(points), 3, dtype=np.int32)
    
    # Create detector
    detector = TrunkDetector(radius=0.4, verbose=True)
    
    # Run detection
    labels = detector.detect_trunks(points, original_labels, verbose=True)
    
    # Analyze results
    print("\n" + "=" * 70)
    print("Analysis of Results")
    print("=" * 70)
    
    trunk_points = points[labels == 2]
    canopy_points = points[labels == 3]
    
    print(f"\nTrunk statistics:")
    print(f"  Number of points: {len(trunk_points):,}")
    if len(trunk_points) > 0:
        print(f"  Height range: {trunk_points[:, 2].min():.2f} - {trunk_points[:, 2].max():.2f} m")
        print(f"  Mean height: {trunk_points[:, 2].mean():.2f} m")
        print(f"  X range: {trunk_points[:, 0].min():.2f} - {trunk_points[:, 0].max():.2f} m")
        print(f"  Y range: {trunk_points[:, 1].min():.2f} - {trunk_points[:, 1].max():.2f} m")
    
    print(f"\nCanopy statistics:")
    print(f"  Number of points: {len(canopy_points):,}")
    if len(canopy_points) > 0:
        print(f"  Height range: {canopy_points[:, 2].min():.2f} - {canopy_points[:, 2].max():.2f} m")
        print(f"  Mean height: {canopy_points[:, 2].mean():.2f} m")
    
    return labels


def example_with_visualization(input_file: str):
    """
    Example 5: Visualize results using Open3D.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Visualization with Open3D")
    print("=" * 70)
    
    import laspy
    import open3d as o3d
    
    # Read and process
    las = laspy.read(input_file)
    points = np.vstack([las.x, las.y, las.z]).T
    
    detector = TrunkDetector(radius=0.4)
    labels = detector.detect_trunks(points, verbose=True)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Color by classification
    # Trunk = brown, Canopy = green
    colors = np.zeros((len(points), 3))
    colors[labels == 2] = [0.6, 0.3, 0.1]  # Brown for trunks
    colors[labels == 3] = [0.0, 0.8, 0.0]  # Green for canopy
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print("\nVisualizing point cloud...")
    print("  Brown = Trunk (class 2)")
    print("  Green = Canopy (class 3)")
    print("\nClose the window to continue.")
    
    o3d.visualization.draw_geometries([pcd])


def example_batch_processing():
    """
    Example 6: Batch process multiple files.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Batch Processing")
    print("=" * 70)
    
    import glob
    import os
    
    # Find all LAS files in a directory
    input_dir = "./data/input/"
    output_dir = "./data/output/"
    
    las_files = glob.glob(os.path.join(input_dir, "*.las"))
    
    if not las_files:
        print(f"No LAS files found in {input_dir}")
        return
    
    print(f"Found {len(las_files)} LAS files")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create detector once (reuse for all files)
    detector = TrunkDetector(
        radius=0.4,
        linearity_threshold=0.8,
        verticality_threshold=0.9
    )
    
    # Process each file
    for i, input_file in enumerate(las_files, 1):
        print(f"\n[{i}/{len(las_files)}] Processing: {os.path.basename(input_file)}")
        
        output_file = os.path.join(
            output_dir,
            os.path.basename(input_file).replace('.las', '_trunks.las')
        )
        
        try:
            process_las_file(input_file, output_file, detector, verbose=False)
            print(f"  ✓ Saved: {output_file}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\n✓ Batch processing complete!")


if __name__ == "__main__":
    import sys
    
    # Check if input file provided
    if len(sys.argv) < 2:
        print("Usage: python example_trunk_detection.py <input.las> [output.las]")
        print("\nExamples:")
        print("  python example_trunk_detection.py forest.las")
        print("  python example_trunk_detection.py forest.las forest_trunks.las")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Determine output file
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        output_file = input_file.replace('.las', '_trunks.las')
    
    # Run examples
    print("Tree Trunk Detection Examples")
    print("=" * 70)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print()
    
    # Example 1: Basic usage
    example_basic_usage(input_file, output_file)
    
    # Uncomment to try other examples:
    
    # Example 2: Custom parameters
    # example_custom_parameters(input_file, output_file.replace('.las', '_custom.las'))
    
    # Example 3: Dense forest
    # example_dense_forest(input_file, output_file.replace('.las', '_dense.las'))
    
    # Example 4: Programmatic usage
    # labels = example_programmatic_usage(input_file)
    
    # Example 5: Visualization (requires display)
    # example_with_visualization(input_file)
    
    # Example 6: Batch processing
    # example_batch_processing()
