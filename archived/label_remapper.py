"""
Core label remapping logic with support for conditional mappings.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import sys
import os

# Add trunk_detection to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trunk_detection'))
from trunk_detection import TrunkDetector


def remap_labels(
    labels: np.ndarray, 
    mapping: Dict, 
    xyz: np.ndarray = None, 
    dataset: str = None, 
    trunk_detector: Optional[TrunkDetector] = None, 
    chunk_size: int = 10_000_000
) -> Tuple[np.ndarray, Dict]:
    """
    Remap labels according to the provided mapping dictionary.
    Handles conditional mappings (list values) that require point coordinates.
    
    Args:
        labels: Original labels array
        mapping: Dictionary mapping original labels to new labels (int or list for conditional)
        xyz: Optional Nx3 array of XYZ coordinates (required for conditional mappings)
        dataset: Optional dataset name ('semantic3d', 'forestsemantic', etc.)
        trunk_detector: Optional pre-initialized TrunkDetector instance
        chunk_size: Maximum points to process at once for memory efficiency
        
    Returns:
        Tuple of (remapped_labels, statistics_dict)
    """
    remapped = labels.copy()
    
    # Collect statistics
    unique_original = np.unique(labels)
    original_counts = {int(label): int(np.sum(labels == label)) for label in unique_original}
    
    # Apply mapping
    for old_label, new_label in mapping.items():
        mask = labels == old_label
        
        if isinstance(new_label, list):
            # Conditional mapping - behavior depends on dataset
            if xyz is None:
                raise ValueError(f"Conditional mapping for label {old_label} requires XYZ coordinates")
            
            label_points = xyz[mask]
            
            if dataset == 'semantic3d' and old_label == 3:
                remapped[mask] = _remap_semantic3d_label3(
                    label_points, new_label, trunk_detector, chunk_size
                )
                
            elif dataset == 'forestsemantic' and old_label == 6:
                remapped[mask] = _remap_forestsemantic_label6(label_points, new_label, mask.sum())
                
            else:
                raise ValueError(f"Unknown conditional mapping for dataset={dataset}, label={old_label}")
        else:
            # Simple mapping
            remapped[mask] = new_label
    
    # Statistics for remapped labels
    unique_remapped = np.unique(remapped)
    remapped_counts = {int(label): int(np.sum(remapped == label)) for label in unique_remapped}
    
    stats = {
        'original_unique': list(unique_original),
        'original_counts': original_counts,
        'remapped_unique': list(unique_remapped),
        'remapped_counts': remapped_counts,
        'total_points': len(labels)
    }
    
    return remapped, stats


def _remap_semantic3d_label3(
    label_points: np.ndarray,
    new_label: list,
    trunk_detector: Optional[TrunkDetector],
    chunk_size: int
) -> np.ndarray:
    """Remap Semantic3D label 3 (high vegetation) to trunk or canopy based on geometry."""
    trunk_val, canopy_val = new_label[0], new_label[1]
    
    if len(label_points) == 0:
        return np.array([], dtype=np.int32)
    
    # Initialize TrunkDetector if not provided
    if trunk_detector is None:
        trunk_detector = TrunkDetector(
            radius=0.4,
            search_method='radius',
            linearity_threshold=0.8,
            verticality_threshold=0.9,
            min_height=-1.5,
            max_height=5.0,
            min_cluster_size=50
        )
    
    # Process in chunks if dataset is too large
    n_points = len(label_points)
    if n_points > chunk_size:
        detected_labels = np.full(n_points, canopy_val, dtype=np.int32)
        n_chunks = (n_points + chunk_size - 1) // chunk_size
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n_points)
            chunk_points = label_points[start_idx:end_idx]
            chunk_labels = np.full(len(chunk_points), canopy_val, dtype=np.int32)
            
            chunk_result = trunk_detector.detect_trunks(
                chunk_points, original_labels=chunk_labels, verbose=False
            )
            detected_labels[start_idx:end_idx] = chunk_result
            
            del chunk_points, chunk_labels, chunk_result
    else:
        initial_labels = np.full(n_points, canopy_val, dtype=np.int32)
        detected_labels = trunk_detector.detect_trunks(
            label_points, original_labels=initial_labels, verbose=False
        )
        del initial_labels
    
    return detected_labels


def _remap_forestsemantic_label6(
    label_points: np.ndarray,
    new_label: list,
    n_points: int
) -> np.ndarray:
    """Remap ForestSemantic label 6 based on height and XY position."""
    unlabeled_val, understory_val = new_label[0], new_label[1]
    
    # Compute center (XY plane)
    center_x = label_points[:, 0].mean()
    center_y = label_points[:, 1].mean()
    
    # Condition 1: under 5m high
    height_condition = label_points[:, 2] < 5.0
    
    # Condition 2: within ±12m range on XY plane from center
    dx = np.abs(label_points[:, 0] - center_x)
    dy = np.abs(label_points[:, 1] - center_y)
    xy_condition = (dx <= 12.0) & (dy <= 12.0)
    
    # Combined condition: both must be true for understory
    understory_mask = height_condition & xy_condition
    
    local_remapped = np.full(n_points, unlabeled_val, dtype=np.int32)
    local_remapped[understory_mask] = understory_val
    
    return local_remapped
