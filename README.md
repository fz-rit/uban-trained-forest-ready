# Urban-Trained, Forest-Ready

Utilities to remap labels across datasets and visualize mappings with <abbr title="Flow diagram where link widths represent values">Sankey</abbr> diagrams. Includes notes for <abbr title="Unsupervised Domain Adaptation">UDA</abbr> and <abbr title="Pseudo-Labeling">PL</abbr> in the pipeline.

- Config: [class_mapping.json](class_mapping.json) (rules, height threshold, sample weights, display names)
- Remap labels: [remap_labels.py](remap_labels.py)
- Sankey diagram: [labelmap_sankey_diagram.py](labelmap_sankey_diagram.py)
- Histogram summary: [summarize_histograms.py](summarize_histograms.py)
- Pipeline overview: [pipeline.md](pipeline.md)

## Install
```bash
conda create -n domain_adapt_env python=3.11 -y
conda activate domain_adapt_env
pip install plotly kaleido numpy matplotlib pandas
```

## Remap labels to unified classes
```bash
python remap_labels.py \
  --mapping_json class_mapping.json \
  --dataset Semantic3D \  # <abbr title="one of: Semantic3D, ForestSemantic, DigiForests">dataset</abbr>
  --in_dir path/to/src_npz \
  --out_dir out/remapped \
  --glob "*.npz" \
  --grid 1.0 \            # <abbr title="XY grid (m) for ground estimation">grid</abbr>
  --height_th 2.0         # <abbr title="override height split (m); else uses meta.height_split.threshold_m">height_th</abbr>
```
Notes:
- Uses <abbr title="Label id ignored by loss functions">ignore_index=255</abbr>.
- Height split: vegetation with normalized height < threshold → Understory (4); else Branches & Canopy (3) (see [class_mapping.json](class_mapping.json)).

## Summarize remapped histograms
```bash
python summarize_histograms.py --dir out/remapped --glob "*.npz"
```

## Sankey diagram (weighted)
This script produces a weighted Sankey where link widths are proportional to per-class point counts.

Data source for weights:
- By default, uses <abbr title="Per-class counts to weight links">sample_weights</abbr> in [class_mapping.json](class_mapping.json).
- You can also provide your own <abbr title="JavaScript Object Notation">JSON</abbr>/<abbr title="Comma-Separated Values">CSV</abbr> file.

Run:
```bash
python labelmap_sankey_diagram.py \
  --prefix sankey_combined_all_datasets_weighted \
  # --weights path/to/weights.json  # optional override
  # --no-annotate                   # optional to hide counts in labels
```

Outputs:
- HTML: sankey_combined_all_datasets_weighted.html
- PNG: sankey_combined_all_datasets_weighted.png (4800x2700, ~300 DPI)

Editing weights:
- Edit the JSON block at `sample_weights` in [class_mapping.json](class_mapping.json). Keys are label IDs as strings.
- Example:
```json
{
  "sample_weights": {
    "Semantic3D": { "2": 125000 },
    "ForestSemantic": { "5": 95000 },
    "DigiForests": { "4": 70000 }
  }
}
```

Notes:
- Node blocks have fixed thickness; link widths reflect weights.
- Right-side merged class names are short (from `meta.class_display_names`), with counts appended unless `--no-annotate` is set.

## See also
- UDA/PL plan and code hooks: [pipeline.md](pipeline.md) (search for <abbr title="Unsupervised Domain Adaptation">UDA</abbr>, <abbr title="Pseudo-Labeling">PL</abbr>).


## **validate_preprocessed.py** 🔍
A validation script that can:
- **Load & Analyze**: Reads `.pt` files and prints comprehensive statistics
- **Visualize**: 4 different visualization modes:
  - `semantic` - Color by semantic class (ground, shrub, stem, canopy)
  - `instance` - Color by instance ID (random colors per instance)
  - `intensity` - Color by reflectance/intensity values
  - `offset` - Color by offset magnitude (for instance centroids)

### Usage Examples:
```bash
# View statistics and semantic segmentation
python validate_preprocessed.py path/to/cloud_123.pt

# View instance segmentation
python validate_preprocessed.py path/to/cloud_123.pt --mode instance

# View intensity features
python validate_preprocessed.py path/to/cloud_123.pt --mode intensity

# Only statistics, no visualization
python validate_preprocessed.py path/to/cloud_123.pt --no-viz
```

The validation script will display detailed statistics including point counts, class distributions, instance counts, bounding box dimensions, and metadata, then open an interactive Open3D window for 3D visualization! 🎨
