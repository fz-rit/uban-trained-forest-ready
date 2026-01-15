"""
Tree Trunk Detection in 3D Point Clouds

This module detects tree trunks in LAS point clouds using local geometric features
derived from PCA (Principal Component Analysis) of point neighborhoods.

Method:
    1. Compute local neighborhoods (radius or kNN search)
    2. Perform PCA on each neighborhood to extract eigenvalues/eigenvectors
    3. Calculate geometric features (linearity, verticality)
    4. Classify points as trunk (class 2) or canopy (class 3)
    5. Post-process with connected component filtering

Author: GitHub Copilot
Date: 2025-11-19
"""

import numpy as np
import laspy
import open3d as o3d
from sklearn.neighbors import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from typing import Tuple, Optional
import warnings


class TrunkDetector:
    """
    Detects tree trunks in point clouds using local geometric features.
    
    Parameters
    ----------
    radius : float, optional
        Radius for neighborhood search in meters (default: 0.4)
    k_neighbors : int, optional
        Number of neighbors for kNN search (default: 50)
    search_method : str, optional
        'radius' or 'knn' (default: 'radius')
    min_neighbors : int, optional
        Minimum neighbors required for valid computation (default: 30)
    linearity_threshold : float, optional
        Minimum linearity for trunk classification (default: 0.8)
    verticality_threshold : float, optional
        Minimum verticality for trunk classification (default: 0.9)
    min_height : float, optional
        Minimum height above ground in meters (default: 0.3)
    max_height : float, optional
        Maximum height above ground in meters (default: 15.0)
    min_cluster_size : int, optional
        Minimum points in a trunk cluster (default: 50)
    """
    
    def __init__(
        self,
        radius: float = 0.4,
        k_neighbors: int = 50,
        search_method: str = 'radius',
        min_neighbors: int = 30,
        linearity_threshold: float = 0.8,
        verticality_threshold: float = 0.9,
        min_height: float = 0.3,
        max_height: float = 15.0,
        min_cluster_size: int = 50
    ):
        self.radius = radius
        self.k_neighbors = k_neighbors
        self.search_method = search_method
        self.min_neighbors = min_neighbors
        self.linearity_threshold = linearity_threshold
        self.verticality_threshold = verticality_threshold
        self.min_height = min_height
        self.max_height = max_height
        self.min_cluster_size = min_cluster_size
        
    def find_neighbors(
        self, 
        points: np.ndarray, 
        query_points: Optional[np.ndarray] = None
    ) -> list:
        """
        Find neighbors for each point using radius or kNN search.
        
        Parameters
        ----------
        points : np.ndarray
            Point cloud coordinates (N, 3)
        query_points : np.ndarray, optional
            Query points (if None, uses points)
            
        Returns
        -------
        list
            List of neighbor indices for each point
        """
        if query_points is None:
            query_points = points
            
        if self.search_method == 'radius':
            # Use Open3D for radius search (faster for large clouds)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            
            neighbors = []
            for i in range(len(query_points)):
                [_, idx, _] = pcd_tree.search_radius_vector_3d(
                    query_points[i], self.radius
                )
                neighbors.append(idx)
                
        elif self.search_method == 'knn':
            # Use sklearn KDTree for kNN search
            tree = KDTree(points)
            distances, indices = tree.query(
                query_points, k=self.k_neighbors + 1
            )
            # Remove self (first neighbor)
            neighbors = [idx[1:].tolist() for idx in indices]
            
        else:
            raise ValueError(
                f"Unknown search_method: {self.search_method}. "
                "Use 'radius' or 'knn'."
            )
            
        return neighbors
    
    def compute_local_pca(
        self, 
        points: np.ndarray, 
        neighbor_indices: list
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute PCA (eigenvalues and eigenvectors) for each point's neighborhood.
        
        Parameters
        ----------
        points : np.ndarray
            Point cloud coordinates (N, 3)
        neighbor_indices : list
            List of neighbor indices for each point
            
        Returns
        -------
        eigenvalues : np.ndarray
            Eigenvalues for each point (N, 3), sorted descending
        eigenvectors : np.ndarray
            First eigenvector for each point (N, 3)
        """
        n_points = len(points)
        eigenvalues = np.zeros((n_points, 3))
        eigenvectors = np.zeros((n_points, 3))
        
        for i, neighbors in enumerate(neighbor_indices):
            if len(neighbors) < self.min_neighbors:
                # Not enough neighbors - mark as invalid
                eigenvalues[i] = [0, 0, 0]
                eigenvectors[i] = [0, 0, 0]
                continue
                
            # Get neighbor coordinates
            neighbor_points = points[neighbors]
            
            # Center the points (subtract mean)
            centroid = np.mean(neighbor_points, axis=0)
            centered = neighbor_points - centroid
            
            # Compute covariance matrix
            cov_matrix = np.dot(centered.T, centered) / len(neighbors)
            
            # Compute eigenvalues and eigenvectors
            eigvals, eigvecs = np.linalg.eigh(cov_matrix)
            
            # Sort in descending order
            sort_indices = np.argsort(eigvals)[::-1]
            eigvals = eigvals[sort_indices]
            eigvecs = eigvecs[:, sort_indices]
            
            eigenvalues[i] = eigvals
            # Store first (largest) eigenvector
            eigenvectors[i] = eigvecs[:, 0]
            
        return eigenvalues, eigenvectors
    
    def compute_features(
        self, 
        eigenvalues: np.ndarray, 
        eigenvectors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute geometric features from eigenvalues and eigenvectors.
        
        Parameters
        ----------
        eigenvalues : np.ndarray
            Eigenvalues (N, 3)
        eigenvectors : np.ndarray
            First eigenvector (N, 3)
            
        Returns
        -------
        linearity : np.ndarray
            Linearity feature (N,)
        verticality : np.ndarray
            Verticality feature (N,)
        """
        # Extract individual eigenvalues
        lambda1 = eigenvalues[:, 0]
        lambda2 = eigenvalues[:, 1]
        lambda3 = eigenvalues[:, 2]
        
        # Compute linearity: (λ₁ - λ₂) / λ₁
        # Avoid division by zero
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            linearity = np.where(
                lambda1 > 1e-10,
                (lambda1 - lambda2) / lambda1,
                0.0
            )
        
        # Compute verticality: |v₁ · (0, 0, 1)|
        # Absolute dot product with vertical axis
        vertical_axis = np.array([0, 0, 1])
        verticality = np.abs(np.dot(eigenvectors, vertical_axis))
        
        return linearity, verticality
    
    def classify_points(
        self,
        points: np.ndarray,
        linearity: np.ndarray,
        verticality: np.ndarray,
        original_labels: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Classify points as trunk (2) or canopy (3) based on features.
        
        Parameters
        ----------
        points : np.ndarray
            Point cloud coordinates (N, 3)
        linearity : np.ndarray
            Linearity feature (N,)
        verticality : np.ndarray
            Verticality feature (N,)
        original_labels : np.ndarray, optional
            Original classification labels (default: all 3)
            
        Returns
        -------
        labels : np.ndarray
            Updated classification labels (N,)
        """
        n_points = len(points)
        
        # Initialize labels (default to canopy = 3)
        if original_labels is not None:
            labels = original_labels.copy()
        else:
            labels = np.full(n_points, 3, dtype=np.int32)
        
        # Extract z coordinates (height)
        z = points[:, 2]
        
        # Find minimum z (approximate ground level)
        z_min = np.percentile(z, 1)  # Use 1st percentile to avoid outliers
        height_above_ground = z - z_min
        
        # Classify as trunk if all conditions are met
        trunk_mask = (
            (linearity > self.linearity_threshold) &
            (verticality > self.verticality_threshold) &
            (height_above_ground >= self.min_height) &
            (height_above_ground <= self.max_height)
        )
        
        # Update labels
        labels[trunk_mask] = 2
        
        return labels
    
    def post_process_clusters(
        self,
        points: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        """
        Post-process trunk points using connected component analysis.
        Remove small isolated clusters.
        
        Parameters
        ----------
        points : np.ndarray
            Point cloud coordinates (N, 3)
        labels : np.ndarray
            Classification labels (N,)
            
        Returns
        -------
        labels : np.ndarray
            Filtered classification labels (N,)
        """
        labels = labels.copy()
        
        # Get trunk point indices
        trunk_indices = np.where(labels == 2)[0]
        
        if len(trunk_indices) < self.min_cluster_size:
            # Not enough trunk points - revert all to canopy
            labels[trunk_indices] = 3
            return labels
        
        # Build adjacency graph for trunk points
        trunk_points = points[trunk_indices]
        
        # Use KDTree to find nearby points (within radius)
        tree = KDTree(trunk_points)
        
        # Build sparse adjacency matrix
        # Two points are connected if within radius
        radius = self.radius * 1.5  # Slightly larger radius for connectivity
        pairs = tree.query_radius(trunk_points, r=radius)
        
        # Build edge list
        row_indices = []
        col_indices = []
        for i, neighbors in enumerate(pairs):
            for j in neighbors:
                if i != j:
                    row_indices.append(i)
                    col_indices.append(j)
        
        if len(row_indices) == 0:
            # No connections - revert all to canopy
            labels[trunk_indices] = 3
            return labels
        
        # Create sparse adjacency matrix
        n_trunk = len(trunk_indices)
        data = np.ones(len(row_indices), dtype=bool)
        adjacency = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_trunk, n_trunk)
        )
        
        # Find connected components
        n_components, component_labels = connected_components(
            adjacency, directed=False
        )
        
        # Count points in each component
        component_sizes = np.bincount(component_labels)
        
        # Keep only large components
        large_components = np.where(
            component_sizes >= self.min_cluster_size
        )[0]
        
        # Filter trunk points
        for i, comp_label in enumerate(component_labels):
            if comp_label not in large_components:
                # Small cluster - revert to canopy
                original_idx = trunk_indices[i]
                labels[original_idx] = 3
        
        return labels
    
    def detect_trunks(
        self,
        points: np.ndarray,
        original_labels: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Main pipeline: detect tree trunks in point cloud.
        
        Parameters
        ----------
        points : np.ndarray
            Point cloud coordinates (N, 3)
        original_labels : np.ndarray, optional
            Original classification labels (default: all 3)
        verbose : bool, optional
            Print progress information (default: True)
            
        Returns
        -------
        labels : np.ndarray
            Updated classification labels (N,)
            - 2: trunk
            - 3: canopy
        """
        n_points = len(points)
        
        if verbose:
            print(f"Processing {n_points:,} points...")
            print(f"Search method: {self.search_method}")
            if self.search_method == 'radius':
                print(f"Radius: {self.radius} m")
            else:
                print(f"K neighbors: {self.k_neighbors}")
        
        # Step 1: Find neighbors
        if verbose:
            print("\n[1/5] Finding neighbors...")
        neighbors = self.find_neighbors(points)
        
        # Step 2: Compute PCA
        if verbose:
            print("[2/5] Computing local PCA...")
        eigenvalues, eigenvectors = self.compute_local_pca(points, neighbors)
        
        # Step 3: Compute features
        if verbose:
            print("[3/5] Computing geometric features...")
        linearity, verticality = self.compute_features(eigenvalues, eigenvectors)
        
        # Step 4: Classify points
        if verbose:
            print("[4/5] Classifying points...")
        labels = self.classify_points(
            points, linearity, verticality, original_labels
        )
        
        # Count initial trunk points
        n_trunk_initial = np.sum(labels == 2)
        if verbose:
            print(f"  Initial trunk points: {n_trunk_initial:,} "
                  f"({100 * n_trunk_initial / n_points:.2f}%)")
        
        # Step 5: Post-process clusters
        if verbose:
            print("[5/5] Post-processing clusters...")
        labels = self.post_process_clusters(points, labels)
        
        # Count final trunk points
        n_trunk_final = np.sum(labels == 2)
        n_canopy = np.sum(labels == 3)
        
        if verbose:
            print(f"\n✓ Complete!")
            print(f"  Final trunk points: {n_trunk_final:,} "
                  f"({100 * n_trunk_final / n_points:.2f}%)")
            print(f"  Canopy points: {n_canopy:,} "
                  f"({100 * n_canopy / n_points:.2f}%)")
            print(f"  Removed small clusters: {n_trunk_initial - n_trunk_final:,} points")
        
        return labels


def process_las_file(
    input_path: str,
    output_path: str,
    detector: Optional[TrunkDetector] = None,
    verbose: bool = True
) -> None:
    """
    Process a LAS file: detect trunks and save updated point cloud.
    
    Parameters
    ----------
    input_path : str
        Path to input LAS file
    output_path : str
        Path to output LAS file
    detector : TrunkDetector, optional
        Trunk detector instance (default: use default parameters)
    verbose : bool, optional
        Print progress information (default: True)
    """
    if verbose:
        print(f"Reading LAS file: {input_path}")
    
    # Read LAS file
    las = laspy.read(input_path)
    points = np.vstack([las.x, las.y, las.z]).T
    
    # Get original labels (if available)
    if hasattr(las, 'classification'):
        original_labels = np.array(las.classification)
    else:
        original_labels = None
    
    # Create detector if not provided
    if detector is None:
        detector = TrunkDetector()
    
    # Detect trunks
    labels = detector.detect_trunks(points, original_labels, verbose=verbose)
    
    # Update classification in LAS file
    las.classification = labels
    
    # Write output
    if verbose:
        print(f"\nWriting output: {output_path}")
    las.write(output_path)
    
    if verbose:
        print("✓ Done!")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python trunk_detection.py <input.las> <output.las>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Create detector with default parameters
    detector = TrunkDetector(
        radius=0.4,
        search_method='radius',
        linearity_threshold=0.8,
        verticality_threshold=0.9,
        min_height=-1.5,
        max_height=5.0,
        min_cluster_size=50
    )
    
    # Process file
    process_las_file(input_file, output_file, detector, verbose=True)
