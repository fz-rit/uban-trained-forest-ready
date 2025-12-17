"""GPU-accelerated trunk detection using PyTorch + Projection Ratios."""

from __future__ import annotations

import numpy as np
import torch
from torch_cluster import radius_graph
from sklearn.neighbors import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from typing import Optional, Tuple, Dict
import warnings


class TorchTrunkDetector:
    """Detect tree trunks using Plane Projection Grid Ratios."""

    def __init__(
        self,
        radius: float = 0.4,
        grid_resolution: float = 0.05,
        projection_ratio_threshold: float = 0.6,
        min_height: float = -1.5,
        max_height: float = 5.0,
        min_cluster_size: int = 50,
        min_neighbors: int = 15,
        max_neighbors: int = 256,
        device: Optional[str] = None
    ) -> None:
        self.radius = radius
        self.grid_resolution = grid_resolution
        self.projection_ratio_threshold = projection_ratio_threshold
        self.min_height = min_height
        self.max_height = max_height
        self.min_cluster_size = min_cluster_size
        self.min_neighbors = min_neighbors
        self.max_neighbors = max_neighbors
        
        self.max_possible_grids = np.pi * (self.radius / self.grid_resolution) ** 2
        self.features: Dict[str, np.ndarray] = {}
        
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def detect_trunks(
        self,
        points: np.ndarray,
        original_labels: Optional[np.ndarray] = None,
        verbose: bool = True,
        chunk_size: int = 30_000
    ) -> np.ndarray:
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be shaped (N, 3)")
        
        n_points = points.shape[0]

        if verbose:
            print(f"[TorchTrunkDetector] Processing {n_points:,} points...")
            print(f"                     Radius: {self.radius}m | Grid: {self.grid_resolution}m")
            print(f"                     Sorting points spatially...")

        # 1. SPATIAL SORTING
        sort_idx = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
        inverse_sort_idx = np.empty_like(sort_idx)
        inverse_sort_idx[sort_idx] = np.arange(n_points)
        points_sorted = points[sort_idx]
        
        if original_labels is None:
            labels_sorted = np.full(n_points, 3, dtype=np.int32)
        else:
            labels_sorted = original_labels[sort_idx].copy()

        # Init features
        feature_names = ['ratio', 'nn_cnts', 'count_xy', 'count_xz', 'count_yz', 
                         'coverage_xy', 'coverage_xz', 'coverage_yz']
        self.features = {k: np.zeros(n_points, dtype=np.float32) for k in feature_names}
        self.features['nn_cnts'] = self.features['nn_cnts'].astype(np.int32)
        self.features['count_xy'] = self.features['count_xy'].astype(np.int32)
        self.features['count_xz'] = self.features['count_xz'].astype(np.int32)
        self.features['count_yz'] = self.features['count_yz'].astype(np.int32)

        # 2. CHUNKED PROCESSING
        n_chunks = (n_points + chunk_size - 1) // chunk_size
        
        from tqdm import tqdm
        iterator = range(n_chunks)
        if verbose and n_chunks > 1:
            iterator = tqdm(iterator, desc="Processing chunks", unit="chunk")

        for i in iterator:
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n_points)
            
            # Dynamic Buffer: 20% of chunk size
            buffer_size = int(chunk_size * 0.2)
            buffer_start = max(0, start_idx - buffer_size)
            buffer_end = min(n_points, end_idx + buffer_size)
            
            context_points = points_sorted[buffer_start:buffer_end]
            
            # Local Centering
            chunk_shift = context_points[0].copy()
            local_points = context_points - chunk_shift
            pts = torch.from_numpy(local_points).float().to(self.device)
            
            core_start = start_idx - buffer_start
            core_end = core_start + (end_idx - start_idx)
            core_indices = np.arange(core_start, core_end)
            core_points_abs_z = points_sorted[buffer_start:buffer_end][core_indices][:, 2]

            # A. Neighbor Search
            # edge_index is (Source, Target) -> (Neighbor, Center)
            edge_index, nn_cnts = self._query_neighbors(pts)
            
            # B. Compute Coverage
            results = self._compute_grid_coverage(pts, edge_index, nn_cnts)
            (ratios, c_xy, c_xz, c_yz, cov_xy, cov_xz, cov_yz) = results
            
            # C. Store Features
            self.features['ratio'][start_idx:end_idx] = ratios[core_indices].cpu().numpy()
            self.features['nn_cnts'][start_idx:end_idx] = nn_cnts[core_indices].cpu().numpy()
            self.features['count_xy'][start_idx:end_idx] = c_xy[core_indices].cpu().numpy()
            self.features['count_xz'][start_idx:end_idx] = c_xz[core_indices].cpu().numpy()
            self.features['count_yz'][start_idx:end_idx] = c_yz[core_indices].cpu().numpy()
            self.features['coverage_xy'][start_idx:end_idx] = cov_xy[core_indices].cpu().numpy()
            self.features['coverage_xz'][start_idx:end_idx] = cov_xz[core_indices].cpu().numpy()
            self.features['coverage_yz'][start_idx:end_idx] = cov_yz[core_indices].cpu().numpy()
            
            # D. Classify
            core_ratios = ratios[core_indices]
            chunk_labels = self._classify_points(
                core_points_abs_z, 
                core_ratios, 
                labels_sorted[start_idx:end_idx]
            )
            labels_sorted[start_idx:end_idx] = chunk_labels
            
            del pts, edge_index, nn_cnts, results
            torch.cuda.empty_cache()
        
        # 3. UNSORT
        labels_sorted = self._post_process_clusters(points_sorted, labels_sorted)
        labels_final = labels_sorted[inverse_sort_idx]
        for key in self.features:
            self.features[key] = self.features[key][inverse_sort_idx]

        return labels_final

    def _query_neighbors(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_index = radius_graph(
            points,
            r=self.radius,
            loop=False,
            max_num_neighbors=self.max_neighbors,
        )
        if edge_index.numel() == 0:
            counts = torch.zeros(points.shape[0], dtype=torch.long, device=points.device)
        else:
            # FIX: Count index 1 (Targets/Centers), not index 0 (Sources/Neighbors)
            counts = torch.bincount(edge_index[1], minlength=points.shape[0])
        return edge_index, counts

    def _compute_grid_coverage(
        self,
        points: torch.Tensor,
        edge_index: torch.Tensor,
        nn_cnts: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        n_points = points.shape[0]
        
        if edge_index.numel() == 0:
            high = torch.ones(n_points, device=points.device) * 10.0
            zero = torch.zeros(n_points, device=points.device)
            return high, zero, zero, zero, zero, zero, zero

        # Unpack: source(neighbor) -> target(center)
        neighbor_idx, center_idx = edge_index
        
        # Vector: Neighbor to Center
        # diff = points[center] - points[neighbor]
        diff = points[neighbor_idx] - points[center_idx]
        
        grid_coords = torch.floor(diff / self.grid_resolution).long()
        gx, gy, gz = grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]
        
        def count_unique_per_center(u, v):
            # Hash grid coord (u,v)
            u_off = u + 100
            v_off = v + 100
            local_hash = (u_off * 1000) + v_off
            
            # Combine with CENTER index (index 1) to group results by Center
            # FIX: Used center_idx instead of neighbor_idx
            global_hash = (center_idx * 1_000_000) + local_hash
            
            unique_hashes = torch.unique(global_hash)
            active_centers = unique_hashes // 1_000_000
            
            # Count how many unique grids each Center has
            return torch.bincount(active_centers, minlength=n_points).float()

        count_xy = count_unique_per_center(gx, gy)
        count_xz = count_unique_per_center(gx, gz)
        count_yz = count_unique_per_center(gy, gz)
        
        denom_area = float(self.max_possible_grids)
        cov_xy = count_xy / denom_area
        cov_xz = count_xz / denom_area
        cov_yz = count_yz / denom_area
        
        avg_vert_coverage = (cov_xz + cov_yz) / 2.0
        avg_vert_coverage = torch.clamp(avg_vert_coverage, min=1e-6)
        
        ratio = cov_xy / avg_vert_coverage
        
        insufficient = nn_cnts < self.min_neighbors
        ratio[insufficient] = 10.0
        
        return ratio, count_xy, count_xz, count_yz, cov_xy, cov_xz, cov_yz

    def _classify_points(self, z_coords, ratios, original_labels):
        device = ratios.device
        n_points = ratios.shape[0]
        if original_labels is not None:
            labels = torch.from_numpy(original_labels).to(device=device, dtype=torch.int32)
        else:
            labels = torch.full((n_points,), 3, dtype=torch.int32, device=device)
        
        z = torch.from_numpy(z_coords).to(device)
        trunk_mask = (
            (ratios < self.projection_ratio_threshold) &
            (z >= self.min_height) &
            (z <= self.max_height)
        )
        labels[trunk_mask] = 2
        return labels.cpu().numpy()

    def _post_process_clusters(self, points: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Filters clusters by Size and Aspect Ratio (Height/Width).
        Trunks should be tall and narrow (High Aspect Ratio).
        """
        labels = labels.copy()
        trunk_indices = np.where(labels == 2)[0]
        
        # 1. Early exit if too few points
        if len(trunk_indices) < self.min_cluster_size:
            labels[trunk_indices] = 3
            return labels

        trunk_points_np = points[trunk_indices]
        
        # ---------------------------------------------------------------------
        # A. GPU-Accelerated Graph Construction
        # ---------------------------------------------------------------------
        # Shift to local coords for precision
        shift = trunk_points_np[0].copy()
        local_pts = trunk_points_np - shift
        pts_device = torch.from_numpy(local_pts).float().to(self.device)

        # Build graph with strict neighbor limit (prevents OOM)
        clustering_radius = self.radius * 1.5
        edge_index = radius_graph(
            pts_device, 
            r=clustering_radius, 
            loop=False, 
            max_num_neighbors=32 
        )

        if edge_index.numel() == 0:
            labels[trunk_indices] = 3
            return labels

        # Move to CPU
        rows = edge_index[1].cpu().numpy()
        cols = edge_index[0].cpu().numpy()
        del pts_device, edge_index
        torch.cuda.empty_cache()

        # ---------------------------------------------------------------------
        # B. Connected Components
        # ---------------------------------------------------------------------
        n_trunk = len(trunk_indices)
        data = np.ones(len(rows), dtype=bool)
        adjacency = csr_matrix((data, (rows, cols)), shape=(n_trunk, n_trunk))
        
        n_components, comp_labels = connected_components(adjacency, directed=False)
        
        # ---------------------------------------------------------------------
        # C. Geometric Filtering: Aspect Ratio > 3.0
        # ---------------------------------------------------------------------
        import pandas as pd
        
        # Create DataFrame with all coordinates
        df = pd.DataFrame({
            'comp_id': comp_labels,
            'x': trunk_points_np[:, 0],
            'y': trunk_points_np[:, 1],
            'z': trunk_points_np[:, 2]
        })
        
        # Aggregate to find Bounding Box dimensions
        stats = df.groupby('comp_id').agg({
            'x': ['min', 'max'],
            'y': ['min', 'max'],
            'z': ['min', 'max'],
            'comp_id': 'size'
        })
        
        # Flatten MultiIndex columns
        stats.columns = ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max', 'count']
        
        # Calculate Dimensions
        stats['height'] = stats['z_max'] - stats['z_min']
        stats['width_x'] = stats['x_max'] - stats['x_min']
        stats['width_y'] = stats['y_max'] - stats['y_min']
        
        # Width is the max extent in the XY plane
        stats['width'] = stats[['width_x', 'width_y']].max(axis=1)
        
        # Safety: Avoid division by zero for single-point lines
        stats['width'] = stats['width'].replace(0, 0.01)
        
        # Compute Aspect Ratio
        stats['aspect_ratio'] = stats['height'] / stats['width']
        
        # FILTER: 
        # 1. Large enough cluster
        # 2. Aspect Ratio > 3.0 (At least 3x taller than it is wide)
        # 3. (Optional) Absolute min height 0.5m to avoid accepting tiny pebbles
        aspect_ratio_threshold = 2.0
        min_absolute_height = 0.3 
        
        valid_clusters = stats[
            (stats['count'] >= self.min_cluster_size) & 
            (stats['height'] >= min_absolute_height) &
            (stats['aspect_ratio'] > aspect_ratio_threshold)
        ].index.values
        
        # Mask valid points
        keep_mask = np.isin(comp_labels, valid_clusters)
        
        # Revert labels for rejected points (leaves/bushes)
        rejected_indices = trunk_indices[~keep_mask]
        labels[rejected_indices] = 3
        
        return labels