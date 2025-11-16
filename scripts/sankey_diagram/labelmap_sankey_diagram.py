"""
Sankey Diagram for Dataset Label Mapping
Visualizes how labels from Semantic3D, ForestSemantic, and DigiForests map to unified labels.
Supports optional weighting of link widths by class point counts (CSV/JSON) and label annotations.

Note: Some mappings are conditional (based on height/position):
- Semantic3D label 3: height <= 0.2m → 2 (Trunk), else → 3 (Canopy)
- ForestSemantic label 6: height < 5m AND within ±12m XY → 4 (Understory), else → 0 (Unlabeled)
For Sankey visualization, conditional mappings split weights across targets.
"""

import argparse
import csv
import json
import os
from typing import Dict, Optional

import plotly.graph_objects as go


# Dataset labels
semantic3d = {
    0: "Unlabeled", 1: "man-made terrain", 2: "natural terrain", 
    3: "high vegetation", 4: "low vegetation", 5: "buildings", 
    6: "hard scape", 7: "scanning artefacts", 8: "cars"
}

forest_semantic = {
    1: "Ground", 2: "Trunk", 3: "First order branch",
    4: "Higher order branch", 5: "Foliage", 6: "Miscellany", 7: "Unlabeled"
}

digiforests = {
    0: "Unlabeled", 1: "Ground", 2: "Shrub", 
    3: "Stem", 4: "Canopy", 5: "Miscellany"
}

def load_merged_labels(config_path: str = os.path.join(os.path.dirname(__file__), "class_mapping.json")) -> Dict[int, str]:
    """Load short display names for merged labels from class_mapping.json.
    Falls back to internal defaults if JSON not present.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        disp = cfg.get("meta", {}).get("class_display_names")
        if disp:
            return {int(k): v for k, v in disp.items()}
    except Exception as e:
        print(f"[warn] Could not load display names from {config_path}: {e}")
    # Fallback short names
    return {
        1: "Ground",
        2: "Trunks",
        3: "Canopy",
        4: "Understory",
        5: "Miscellany",
        0: "Unlabeled",
    }

# Load display names once
merged_labels = load_merged_labels()

"""
Mappings for Sankey. Note: when a mapping value is a list (e.g. 3 -> [2,3], 6 -> [0,4]),
we split the weight evenly across the multiple targets for visualization only,
as Sankey doesn't have access to per-point conditions.

Conditional mappings:
- Semantic3D label 3: height <= 0.2m → 2 (Trunk), else → 3 (Canopy)
- ForestSemantic label 6: height < 5m AND within ±12m XY → 4 (Understory), else → 0 (Unlabeled)
"""
semantic3d_mapping = {0: 0, 1: 1, 2: 1, 3: [2, 3], 4: 4, 5: 5, 6: 5, 7: 5, 8: 5}
forest_semantic_mapping = {1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: [0,4], 7: 0}
digiforests_mapping = {0: 0, 1: 1, 2: 4, 3: 2, 4: 3, 5: 5}


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "class_mapping.json")


def create_sankey(weights: Optional[dict] = None, annotate_labels: bool = True):
    """Create grouped Sankey diagram.
    - weights: optional dict with keys 'Semantic3D', 'ForestSemantic', 'DigiForests',
      each mapping original label id -> point count. If omitted, all links weight=1.
    - annotate_labels: when True, appends counts to labels for readability.
    """
    
    # Colors
    ds_colors = {'Semantic3D': '#4DB6AC', 'ForestSemantic': '#E57373', 'DigiForests': '#81C784'}
    mg_colors = {0: '#ADB5BD', 1: '#6A994E', 2: '#8B4513', 3: '#2D6A4F', 4: '#95D5B2', 5: '#FB8500'}
    
    node_labels, node_colors, node_x, node_y = [], [], [], []

    # Resolve weights or defaults
    w_s3d = (weights or {}).get('Semantic3D', {k: 1 for k in semantic3d.keys()})
    w_fs = (weights or {}).get('ForestSemantic', {k: 1 for k in forest_semantic.keys()})
    w_df = (weights or {}).get('DigiForests', {k: 1 for k in digiforests.keys()})

    # Pre-compute merged totals for annotation
    merged_totals: dict[int, int] = {k: 0 for k in merged_labels.keys()}

    missing_warning_cache: set[tuple[str, int]] = set()

    def resolve_targets(mapping: Dict[int, int], sid: int, dataset: str, fallback: int = 0) -> list[int]:
        """Return list of merged label targets for sid. If mapping returns a list,
        treat it as a fan-out; otherwise return single-target list. Unknown ids -> [fallback]."""
        target = mapping.get(sid)
        if target is None:
            key = (dataset, sid)
            if key not in missing_warning_cache:
                print(f"[warn] Missing mapping for {dataset} label {sid}; defaulting to merged {fallback}.")
                missing_warning_cache.add(key)
            return [fallback]
        if isinstance(target, list):
            return target
        return [target]

    for sid, w in w_s3d.items():
        targets = resolve_targets(semantic3d_mapping, sid, "Semantic3D")
        split = w / max(1, len(targets))
        for t in targets:
            merged_totals[t] = merged_totals.get(t, 0) + split
    for sid, w in w_fs.items():
        targets = resolve_targets(forest_semantic_mapping, sid, "ForestSemantic")
        split = w / max(1, len(targets))
        for t in targets:
            merged_totals[t] = merged_totals.get(t, 0) + split
    for sid, w in w_df.items():
        targets = resolve_targets(digiforests_mapping, sid, "DigiForests")
        split = w / max(1, len(targets))
        for t in targets:
            merged_totals[t] = merged_totals.get(t, 0) + split
    
    # Semantic3D nodes (rows 0-8)
    s3d_start = 0
    for i, (lid, lname) in enumerate(semantic3d.items()):
        lbl = f"S3D: {lname}"
        if annotate_labels:
            lbl += f" (n={w_s3d.get(lid,1):,})"
        node_labels.append(lbl)
        node_colors.append(ds_colors['Semantic3D'])
        node_x.append(0.01)
        node_y.append(0.05 + i * 0.045)
    
    # ForestSemantic nodes (rows 9-14)
    fs_start = len(node_labels)
    for i, (lid, lname) in enumerate(forest_semantic.items()):
        lbl = f"FS: {lname}"
        if annotate_labels:
            lbl += f" (n={w_fs.get(lid,1):,})"
        node_labels.append(lbl)
        node_colors.append(ds_colors['ForestSemantic'])
        node_x.append(0.01)
        node_y.append(0.05 + (9 + i) * 0.045)
    
    # DigiForests nodes (rows 15-18)
    df_start = len(node_labels)
    for i, (lid, lname) in enumerate(digiforests.items()):
        lbl = f"DF: {lname}"
        if annotate_labels:
            lbl += f" (n={w_df.get(lid,1):,})"
        node_labels.append(lbl)
        node_colors.append(ds_colors['DigiForests'])
        node_x.append(0.01)
        node_y.append(0.05 + (15 + i) * 0.045)
    
    # Merged labels (right side)
    merged_idx = {}
    for lid, lname in merged_labels.items():
        merged_idx[lid] = len(node_labels)
        lbl = lname
        if annotate_labels:
            total = merged_totals.get(lid, 0)
            if total:
                lbl = f"{lname} (n={total:,})"
        node_labels.append(lbl)
        node_colors.append(mg_colors[lid])
        node_x.append(0.99)
        node_y.append(None)
    
    # Links
    sources, targets, values, link_colors = [], [], [], []
    
    for i, (sid, _) in enumerate(semantic3d.items()):
        src = s3d_start + i
        w = w_s3d.get(sid, 1)
        tlist = resolve_targets(semantic3d_mapping, sid, "Semantic3D")
        split = w / max(1, len(tlist))
        for t in tlist:
            sources.append(src)
            targets.append(merged_idx[t])
            values.append(split)
            link_colors.append('rgba(77, 182, 172, 0.4)')
    
    for i, (sid, _) in enumerate(forest_semantic.items()):
        src = fs_start + i
        w = w_fs.get(sid, 1)
        tlist = resolve_targets(forest_semantic_mapping, sid, "ForestSemantic")
        split = w / max(1, len(tlist))
        for t in tlist:
            sources.append(src)
            targets.append(merged_idx[t])
            values.append(split)
            link_colors.append('rgba(229, 115, 115, 0.4)')
    
    for i, (sid, _) in enumerate(digiforests.items()):
        src = df_start + i
        w = w_df.get(sid, 1)
        tlist = resolve_targets(digiforests_mapping, sid, "DigiForests")
        split = w / max(1, len(tlist))
        for t in tlist:
            sources.append(src)
            targets.append(merged_idx[t])
            values.append(split)
            link_colors.append('rgba(129, 199, 132, 0.4)')
    
    # Create figure
    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',
        node=dict(
            pad=20, thickness=30,
            line=dict(color="white", width=2),
            label=node_labels, color=node_colors,
            x=node_x, y=node_y
        ),
        link=dict(source=sources, target=targets, value=values, color=link_colors),
        textfont=dict(color="black", size=24, family="Arial", weight='bold')
    )])
    
    fig.update_layout(
        title={
            'text': "Semantic3D • ForestSemantic • DigiForests → Unified Dataset",
            'x': 0.5, 'xanchor': 'center',
            'font': {'size': 36, 'family': 'Arial', 'color': '#2b2d42'}
        },
        font=dict(size=24, family="Arial", color="#2b2d42"),
        height=900, width=1600,
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=250, r=250, t=120, b=50)
    )
    
    return fig


def load_weights(path: str) -> dict:
    """Load weights from a JSON or CSV file.

    Accepted formats:
    - JSON: {"Semantic3D": {"0": 2000, "1": 25000, ...}, "ForestSemantic": {...}, "DigiForests": {...}}
    - CSV: dataset,label_id,count  (dataset in {Semantic3D,ForestSemantic,DigiForests})
    """
    datasets = {"Semantic3D", "ForestSemantic", "DigiForests"}
    weights: Dict[str, Dict[int, int]] = {"Semantic3D": {}, "ForestSemantic": {}, "DigiForests": {}}

    ext = os.path.splitext(path)[1].lower()
    if ext in (".json", ".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for ds, mapping in data.items():
            if ds not in datasets:
                print(f"[warn] Skipping unknown dataset in JSON: {ds}")
                continue
            for k, v in mapping.items():
                try:
                    lid = int(k)
                    weights[ds][lid] = int(v)
                except Exception:
                    print(f"[warn] Bad entry for {ds}.{k} -> {v}, skipping")
        return weights

    if ext in (".csv", ".tsv"):
        delimiter = "," if ext == ".csv" else "\t"
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                ds = row.get("dataset")
                lid = row.get("label_id")
                cnt = row.get("count")
                if not ds or ds not in datasets:
                    print(f"[warn] Skipping row with unknown dataset: {row}")
                    continue
                try:
                    lid_i = int(lid)
                    cnt_i = int(cnt)
                except Exception:
                    print(f"[warn] Skipping row with invalid numbers: {row}")
                    continue
                weights[ds][lid_i] = cnt_i
        return weights

    raise ValueError(f"Unsupported weights file type: {ext}")


def load_weights_from_config(config_path: str = DEFAULT_CONFIG_PATH) -> Optional[dict]:
    """Load sample_weights from class_mapping.json if present."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        sw = cfg.get("sample_weights")
        if not sw:
            return None
        # convert keys to ints for inner dicts
        out: Dict[str, Dict[int, int]] = {}
        for ds, mapping in sw.items():
            out[ds] = {int(k): int(v) for k, v in mapping.items()}
        return out
    except Exception as e:
        print(f"[warn] Could not read sample_weights from {config_path}: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate WEIGHTED Sankey diagram for label mappings")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to JSON/CSV weights file. If omitted, uses sample_weights in class_mapping.json if present; otherwise runs unweighted as fallback.")
    parser.add_argument("--no-annotate", action="store_true", help="Disable count annotations in labels")
    parser.add_argument("--prefix", type=str, default="labelmap_sankey",
                        help="Output filename prefix (no extension)")
    args = parser.parse_args()

    annotate = not args.no_annotate

    print("Generating WEIGHTED Sankey diagram...")
    # Load provided weights, or sample_weights from class_mapping.json, else fall back to unweighted
    weights = None
    if args.weights:
        try:
            weights = load_weights(args.weights)
        except Exception as e:
            print(f"[error] Failed to load weights from {args.weights}: {e}")
    if weights is None:
        weights = load_weights_from_config(DEFAULT_CONFIG_PATH)
    if weights is None:
        print("[warn] No weights provided and no sample_weights in class_mapping.json; using unweighted values.")
        weights = {}

    # With counts
    fig_counts = create_sankey(weights=weights, annotate_labels=True)
    fig_counts.update_layout(title_text="Semantic3D • ForestSemantic • DigiForests → Unified Dataset")
    fig_counts.write_html(f"{args.prefix}_with_counts.html")
    print(f"✓ Saved: {args.prefix}_with_counts.html")
    fig_counts.write_image(f"{args.prefix}_with_counts.png", width=1600, height=900, scale=3)
    print(f"✓ Saved: {args.prefix}_with_counts.png (4800x2700 pixels, 300 DPI)")

    # Clean (no counts)
    fig_clean = create_sankey(weights=weights, annotate_labels=False)
    fig_clean.update_layout(title_text="Semantic3D • ForestSemantic • DigiForests → Unified Dataset")
    fig_clean.write_html(f"{args.prefix}_clean.html")
    print(f"✓ Saved: {args.prefix}_clean.html")
    fig_clean.write_image(f"{args.prefix}_clean.png", width=1600, height=900, scale=3)
    print(f"✓ Saved: {args.prefix}_clean.png (4800x2700 pixels, 300 DPI)")

    print("✓ Done. Note: Node blocks have fixed thickness; link widths reflect class weights.")
