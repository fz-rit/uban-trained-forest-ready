"""GPU-accelerated trunk detection using PyTorch + torch-cluster."""

from __future__ import annotations

import numpy as np
import torch
from torch_cluster import radius_graph
from torch_scatter import scatter_add
from sklearn.neighbors import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from typing import Optional, Tuple
import warnings


class TorchTrunkDetector:
    """Detect tree trunks with GPU-accelerated radius search + PCA."""

    def __init__(
        self,
        radius: float = 0.4,
        linearity_threshold: float = 0.8,
        verticality_threshold: float = 0.9,
        min_height: float = -1.5,
        max_height: float = 5.0,
        min_cluster_size: int = 50,
        min_neighbors: int = 30,
        max_neighbors: int = 256,
        device: Optional[str] = None
    ) -> None:
        self.radius = radius
        self.linearity_threshold = linearity_threshold
        self.verticality_threshold = verticality_threshold
        self.min_height = min_height
        self.max_height = max_height
        self.min_cluster_size = min_cluster_size
        self.min_neighbors = min_neighbors
        self.max_neighbors = max_neighbors
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type != "cuda":
            warnings.warn(
                "TorchTrunkDetector is running on CPU. For best performance use a CUDA device.",
                RuntimeWarning
            )

    def detect_trunks(
        self,
        points: np.ndarray,
        original_labels: Optional[np.ndarray] = None,
        verbose: bool = True,
        chunk_size: int = 100_000
    ) -> np.ndarray:
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be shaped (N, 3)")
        n_points = points.shape[0]

        if verbose:
            print(f"[TorchTrunkDetector] Processing {n_points:,} points on {self.device}...")
        
        # Process in chunks to avoid GPU OOM
        if n_points > chunk_size:
            if verbose:
                print(f"[TorchTrunkDetector] Using chunked processing with chunk_size={chunk_size:,}")
                print(f"[TorchTrunkDetector] Note: Using overlapping chunks with radius buffer={self.radius*2:.2f}m")
            
            if original_labels is None:
                labels = np.full(n_points, 3, dtype=np.int32)
            else:
                labels = original_labels.copy()
            
            n_chunks = (n_points + chunk_size - 1) // chunk_size
            from tqdm import tqdm
            
            for i in tqdm(range(n_chunks), desc="GPU trunk detection chunks", unit="chunk", disable=not verbose):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, n_points)
                
                # Add buffer zone around chunk (only extend boundaries, not full radius query)
                buffer_size = min(int(chunk_size * 0.1), 5000)  # 10% buffer or max 5k points
                buffer_start = max(0, start_idx - buffer_size)
                buffer_end = min(n_points, end_idx + buffer_size)
                
                # Get context points (core + buffer)
                context_points = points[buffer_start:buffer_end]
                
                # Map core indices to context indices
                core_start_in_context = start_idx - buffer_start
                core_end_in_context = core_start_in_context + (end_idx - start_idx)
                core_to_context = np.arange(core_start_in_context, core_end_in_context)
                core_points = context_points[core_to_context]
                
                # Process context points on GPU
                pts = torch.from_numpy(context_points).float().to(self.device)
                edge_index, neighbor_counts = self._query_neighbors(pts)
                eigenvalues, eigenvectors = self._compute_local_pca(pts, edge_index, neighbor_counts)
                linearity, verticality = self._compute_features(eigenvalues, eigenvectors)
                
                # Extract results only for core chunk
                core_linearity = linearity[core_to_context]
                core_verticality = verticality[core_to_context]
                
                chunk_labels = self._classify_points(
                    core_points, 
                    core_linearity, 
                    core_verticality, 
                    labels[start_idx:end_idx]
                )
                labels[start_idx:end_idx] = chunk_labels
                
                # Free GPU memory
                del pts, edge_index, neighbor_counts, eigenvalues, eigenvectors
                del linearity, verticality, core_linearity, core_verticality, chunk_labels
                torch.cuda.empty_cache()
            
            labels = self._post_process_clusters(points, labels)
            return labels
        else:
            # Process all at once for small point clouds
            pts = torch.from_numpy(points).float().to(self.device)
            edge_index, neighbor_counts = self._query_neighbors(pts)
            eigenvalues, eigenvectors = self._compute_local_pca(pts, edge_index, neighbor_counts)
            linearity, verticality = self._compute_features(eigenvalues, eigenvectors)
            labels = self._classify_points(points, linearity, verticality, original_labels)
            labels = self._post_process_clusters(points, labels)
            return labels

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
            counts = torch.bincount(edge_index[0], minlength=points.shape[0])
        return edge_index, counts

    def _compute_local_pca(
        self,
        points: torch.Tensor,
        edge_index: torch.Tensor,
        neighbor_counts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_points = points.shape[0]

        if edge_index.numel() == 0:
            zero_vals = torch.zeros((n_points, 3), device=points.device)
            zero_vecs = torch.zeros((n_points, 3, 3), device=points.device)
            return zero_vals, zero_vecs

        row, col = edge_index
        neighbor_points = points[col]

        sum_xyz = scatter_add(neighbor_points, row, dim=0, dim_size=n_points)
        counts = neighbor_counts.clamp(min=1).unsqueeze(-1).float()
        centroid = sum_xyz / counts

        diff = neighbor_points - centroid[row]
        outer = diff.unsqueeze(-1) * diff.unsqueeze(-2)
        cov = scatter_add(outer, row, dim=0, dim_size=n_points)
        cov = cov / counts.unsqueeze(-1)

        insufficient = neighbor_counts < self.min_neighbors
        
        # Add regularization for numerical stability and handle insufficient neighbors
        eye = torch.eye(3, device=cov.device, dtype=cov.dtype)
        cov = cov + 1e-6 * eye.unsqueeze(0)
        cov[insufficient] = eye

        # Compute eigendecomposition on CPU to avoid cusolver issues
        cov_cpu = cov.cpu()
        eigvals, eigvecs = torch.linalg.eigh(cov_cpu)
        eigvals = eigvals.to(cov.device)
        eigvecs = eigvecs.to(cov.device)
        
        eigvals = eigvals.flip(dims=[-1])
        eigvecs = eigvecs.flip(dims=[-1])
        
        eigvals[insufficient] = 0
        eigvecs[insufficient] = 0
        return eigvals, eigvecs

    def _compute_features(
        self,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lambda1 = eigenvalues[:, 0]
        lambda2 = eigenvalues[:, 1]
        eps = 1e-12
        linearity = torch.where(
            lambda1 > eps,
            (lambda1 - lambda2) / (lambda1 + eps),
            torch.zeros_like(lambda1)
        )

        main_dirs = eigenvectors[:, :, 0]
        vertical_axis = torch.tensor([0.0, 0.0, 1.0], device=eigenvectors.device)
        verticality = torch.abs(torch.matmul(main_dirs, vertical_axis))
        return linearity, verticality

    def _classify_points(
        self,
        points: np.ndarray,
        linearity: torch.Tensor,
        verticality: torch.Tensor,
        original_labels: Optional[np.ndarray]
    ) -> np.ndarray:
        device = linearity.device
        n_points = linearity.shape[0]
        if original_labels is not None:
            labels = torch.from_numpy(original_labels).to(device=device, dtype=torch.int32)
        else:
            labels = torch.full((n_points,), 3, dtype=torch.int32, device=device)

        z = torch.from_numpy(points[:, 2]).to(device)

        trunk_mask = (
            (linearity > self.linearity_threshold) &
            (verticality > self.verticality_threshold) &
            (z >= self.min_height) &
            (z <= self.max_height)
        )
        labels[trunk_mask] = 2
        return labels.cpu().numpy()

    def _post_process_clusters(self, points: np.ndarray, labels: np.ndarray) -> np.ndarray:
        labels = labels.copy()
        trunk_indices = np.where(labels == 2)[0]
        if len(trunk_indices) < self.min_cluster_size:
            labels[trunk_indices] = 3
            return labels

        trunk_points = points[trunk_indices]
        tree = KDTree(trunk_points)
        pairs = tree.query_radius(trunk_points, r=self.radius * 1.5)

        row_idx = []
        col_idx = []
        for i, neighbors in enumerate(pairs):
            for j in neighbors:
                if i != j:
                    row_idx.append(i)
                    col_idx.append(j)

        if not row_idx:
            labels[trunk_indices] = 3
            return labels

        n_trunk = len(trunk_indices)
        data = np.ones(len(row_idx), dtype=bool)
        adjacency = csr_matrix((data, (row_idx, col_idx)), shape=(n_trunk, n_trunk))
        n_components, component_labels = connected_components(adjacency, directed=False)
        component_sizes = np.bincount(component_labels, minlength=n_components)
        keep_components = np.where(component_sizes >= self.min_cluster_size)[0]
        for idx, comp in enumerate(component_labels):
            if comp not in keep_components:
                labels[trunk_indices[idx]] = 3
        return labels
