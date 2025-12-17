'''
Plot label distribution comparisons before and after remapping.

This module provides visualization functions to compare label distributions
before and after remapping for different datasets.
'''

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple

# Configure matplotlib
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})


# ============================================================================
# Label Name Mappings for Visualization
# ============================================================================

SEMANTIC3D_LABEL_NAMES = {
    0: 'unlabeled',
    1: 'man-made terrain',
    2: 'natural terrain',
    3: 'high vegetation',
    4: 'low vegetation',
    5: 'buildings',
    6: 'hard scape',
    7: 'scanning artefacts',
    8: 'cars'
}

FOREST_SEMANTIC_LABEL_NAMES = {
    1: 'Ground',
    2: 'Trunk',
    3: 'First-order branch',
    4: 'Higher-order branch',
    5: 'Foliage',
    6: 'Miscellany',
    7: 'strange points'
}

DIGIFORESTS_LABEL_NAMES = {
    0: 'Unlabeled',
    1: 'Ground',
    2: 'Shrub',
    3: 'Stem',
    4: 'Canopy',
    5: 'Miscellaneous'
}

# Unified label names (after remapping)
UNIFIED_LABEL_NAMES = {
    0: 'Unlabeled',
    1: 'Ground',
    2: 'Trunk',
    3: 'Canopy',
    4: 'Understory',
    5: 'Misc'
}


def plot_before_after_histogram(
    original_labels: np.ndarray,
    remapped_labels: np.ndarray,
    original_label_names: Dict[int, str],
    remapped_label_names: Dict[int, str],
    dataset_name: str,
    save_path: Path
) -> None:
    """
    Plot side-by-side histograms comparing label distributions before and after remapping.
    
    Args:
        original_labels: Original label array
        remapped_labels: Remapped label array
        original_label_names: Dictionary mapping original label IDs to names
        remapped_label_names: Dictionary mapping remapped label IDs to names
        dataset_name: Name of the dataset for title
        save_path: Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # ========== Left plot: Original labels ==========
    unique_original = np.unique(original_labels)
    counts_original = np.array([np.sum(original_labels == label) for label in unique_original])
    ratios_original = counts_original / counts_original.sum() * 100
    
    x_pos_original = np.arange(len(unique_original))
    bars1 = ax1.bar(x_pos_original, counts_original, width=0.6, 
                     edgecolor='black', color='steelblue', alpha=0.8)
    
    ax1.set_ylabel('Number of Points', fontsize=12, fontweight='bold')
    ax1.set_title(f'{dataset_name} - Original Labels', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos_original)
    ax1.set_xticklabels(
        [f"{original_label_names.get(label, f'Class {label}')}\n(ID: {label})" 
         for label in unique_original],
        rotation=45, ha='right', fontsize=9
    )
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add count annotations
    for bar, count, ratio in zip(bars1, counts_original, ratios_original):
        height = bar.get_height()
        ax1.annotate(f'{count:,}\n({ratio:.1f}%)',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # ========== Right plot: Remapped labels ==========
    unique_remapped = np.unique(remapped_labels)
    counts_remapped = np.array([np.sum(remapped_labels == label) for label in unique_remapped])
    ratios_remapped = counts_remapped / counts_remapped.sum() * 100
    
    x_pos_remapped = np.arange(len(unique_remapped))
    bars2 = ax2.bar(x_pos_remapped, counts_remapped, width=0.6,
                     edgecolor='black', color='seagreen', alpha=0.8)
    
    ax2.set_ylabel('Number of Points', fontsize=12, fontweight='bold')
    ax2.set_title(f'{dataset_name} - Remapped Labels', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos_remapped)
    ax2.set_xticklabels(
        [f"{remapped_label_names.get(label, f'Class {label}')}\n(ID: {label})" 
         for label in unique_remapped],
        rotation=45, ha='right', fontsize=9
    )
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add count annotations
    for bar, count, ratio in zip(bars2, counts_remapped, ratios_remapped):
        height = bar.get_height()
        ax2.annotate(f'{count:,}\n({ratio:.1f}%)',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Make y-axes consistent for easier comparison
    max_count = max(counts_original.max(), counts_remapped.max())
    ax1.set_ylim(0, max_count * 1.15)
    ax2.set_ylim(0, max_count * 1.15)
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved comparison plot to: {save_path}")


def plot_stacked_comparison(
    original_labels: np.ndarray,
    remapped_labels: np.ndarray,
    original_label_names: Dict[int, str],
    remapped_label_names: Dict[int, str],
    dataset_name: str,
    save_path: Path
) -> None:
    """
    Plot vertically stacked histograms for before/after comparison.
    
    Args:
        original_labels: Original label array
        remapped_labels: Remapped label array
        original_label_names: Dictionary mapping original label IDs to names
        remapped_label_names: Dictionary mapping remapped label IDs to names
        dataset_name: Name of the dataset for title
        save_path: Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # ========== Top plot: Original labels ==========
    unique_original = np.unique(original_labels)
    counts_original = np.array([np.sum(original_labels == label) for label in unique_original])
    ratios_original = counts_original / counts_original.sum() * 100
    
    x_pos_original = np.arange(len(unique_original))
    bars1 = ax1.bar(x_pos_original, counts_original, width=0.6,
                     edgecolor='black', color='steelblue', alpha=0.8)
    
    ax1.set_ylabel('Number of Points', fontsize=12, fontweight='bold')
    ax1.set_title(f'{dataset_name} - Original Labels (Before Remapping)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos_original)
    ax1.set_xticklabels(
        [f"{original_label_names.get(label, f'Class {label}')}\n(ID: {label})"
         for label in unique_original],
        rotation=45, ha='right', fontsize=10
    )
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add annotations
    for bar, count, ratio in zip(bars1, counts_original, ratios_original):
        height = bar.get_height()
        ax1.annotate(f'{count:,}\n({ratio:.1f}%)',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # ========== Bottom plot: Remapped labels ==========
    unique_remapped = np.unique(remapped_labels)
    counts_remapped = np.array([np.sum(remapped_labels == label) for label in unique_remapped])
    ratios_remapped = counts_remapped / counts_remapped.sum() * 100
    
    x_pos_remapped = np.arange(len(unique_remapped))
    bars2 = ax2.bar(x_pos_remapped, counts_remapped, width=0.6,
                     edgecolor='black', color='seagreen', alpha=0.8)
    
    ax2.set_ylabel('Number of Points', fontsize=12, fontweight='bold')
    ax2.set_title(f'{dataset_name} - Remapped Labels (After Remapping)',
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos_remapped)
    ax2.set_xticklabels(
        [f"{remapped_label_names.get(label, f'Class {label}')}\n(ID: {label})"
         for label in unique_remapped],
        rotation=45, ha='right', fontsize=10
    )
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add annotations
    for bar, count, ratio in zip(bars2, counts_remapped, ratios_remapped):
        height = bar.get_height()
        ax2.annotate(f'{count:,}\n({ratio:.1f}%)',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved stacked comparison plot to: {save_path}")


def get_label_names_for_dataset(dataset: str) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Get the original and remapped label name dictionaries for a dataset.
    
    Args:
        dataset: Dataset name ('semantic3d', 'forestsemantic', or 'digiforests')
        
    Returns:
        Tuple of (original_label_names, remapped_label_names)
    """
    if dataset == 'semantic3d':
        return SEMANTIC3D_LABEL_NAMES, UNIFIED_LABEL_NAMES
    elif dataset == 'forestsemantic':
        return FOREST_SEMANTIC_LABEL_NAMES, UNIFIED_LABEL_NAMES
    elif dataset == 'digiforests':
        return DIGIFORESTS_LABEL_NAMES, UNIFIED_LABEL_NAMES
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
