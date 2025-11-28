"""
Label remapping package for Semantic3D, ForestSemantic, and DigiForests datasets.
"""

from .label_io import (
    read_semantic3d_labels,
    read_semantic3d_points_and_labels,
    read_forestsemantic_labels,
    read_forestsemantic_points_and_labels,
    read_digiforests_labels,
    write_labels
)

from .label_remapper import remap_labels
from .trunk_detection.trunk_detection_torch import TorchTrunkDetector
from .trunk_detection.trunk_detection_torch_projection import TorchTrunkDetector
from .plot_label_comparison import (
        plot_before_after_histogram,
        plot_stacked_comparison,
        get_label_names_for_dataset
    )

from .dataset_processor import (
    process_semantic3d,
    process_forestsemantic,
    process_digiforests
)

__all__ = [
    'read_semantic3d_labels',
    'read_semantic3d_points_and_labels',
    'read_forestsemantic_labels',
    'read_forestsemantic_points_and_labels',
    'read_digiforests_labels',
    'write_labels',
    'remap_labels',
    'TorchTrunkDetector',
    'process_semantic3d',
    'process_forestsemantic',
    'process_digiforests',
]
