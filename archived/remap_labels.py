
import argparse, json, os, glob
import numpy as np

IGNORE = 255

def load_npz(path):
    d = np.load(path, allow_pickle=False)
    pts = d["points"]
    feats = d["feats"] if "feats" in d else None
    labels = d["labels"].astype(np.int32) if "labels" in d else None
    return pts, feats, labels

def save_npz(path, points, feats, labels_meta, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if feats is None:
        np.savez_compressed(path, points=points, labels_meta=labels_meta, **(extra or {}))
    else:
        np.savez_compressed(path, points=points, feats=feats, labels_meta=labels_meta, **(extra or {}))

def estimate_ground_z(points, cell=1.0):
    # Estimate local ground height via XY grid min pooling + simple fill.
    x, y, z = points[:,0], points[:,1], points[:,2]
    xi = np.floor((x - x.min()) / cell).astype(int)
    yi = np.floor((y - y.min()) / cell).astype(int)
    key = xi.astype(np.int64) << 32 | yi.astype(np.int64)
    order = np.argsort(key)
    key_sorted = key[order]
    z_sorted = z[order]
    unique_keys, idx_start = np.unique(key_sorted, return_index=True)
    mins = np.minimum.reduceat(z_sorted, idx_start)
    zmin_map = dict(zip(unique_keys.tolist(), mins.tolist()))
    z0 = np.array([zmin_map.get(int(k), np.min(z)) for k in key], dtype=np.float32)
    return z0

def apply_rules(labels_src, z_norm, rules, debris_policy, height_th, ignore_ids):
    out = np.full(labels_src.shape, IGNORE, dtype=np.int32)

    # ignore/unlabeled first
    if ignore_ids:
        for iid in ignore_ids:
            out[labels_src == iid] = IGNORE

    for rule in rules:
        ids = rule.get("ids", [])
        to = rule["to"]
        if not ids:
            continue
        m = np.zeros_like(labels_src, dtype=bool)
        for i in ids:
            m |= (labels_src == i)
        if to == "by_height":
            out[m & (z_norm < height_th)] = 4  # Understory
            out[m & (z_norm >= height_th)] = 3 # Branches & Canopy
        elif to == "debris_policy":
            if debris_policy == "understory":
                out[m] = 4
            elif debris_policy == "forest_floor":
                out[m] = 1
            else:
                out[m] = IGNORE
        elif to == 255:
            out[m] = IGNORE
        else:
            out[m] = int(to)

    return out

def main():
    ap = argparse.ArgumentParser(description="Remap labels to HF-aligned 5 classes.")
    ap.add_argument("--mapping_json", required=True)
    ap.add_argument("--dataset", required=True, choices=["ForestSemantic","DigiForests","Semantic3D"])
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--glob", default="*.npz")
    ap.add_argument("--grid", type=float, default=1.0, help="XY grid size (m) for ground estimation")
    ap.add_argument("--height_th", type=float, default=None, help="override height split threshold (m)")
    args = ap.parse_args()

    with open(args.mapping_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    ds_cfg = cfg["datasets"][args.dataset]
    rules = ds_cfg["rules"]
    debris_policy = ds_cfg.get("debris_policy", "understory")
    ignore_ids = ds_cfg.get("unlabeled_ids", [])
    default_th = cfg["meta"]["height_split"]["threshold_m"]
    height_th = args.height_th if args.height_th is not None else default_th

    files = sorted(glob.glob(os.path.join(args.in_dir, args.glob)))
    if not files:
        print("No files matched:", os.path.join(args.in_dir, args.glob))
        return

    total_counts = np.zeros(256, dtype=np.int64)
    meta_counts = np.zeros(256, dtype=np.int64)

    os.makedirs(args.out_dir, exist_ok=True)

    for fpath in files:
        pts, feats, labels_src = load_npz(fpath)
        if labels_src is None:
            print("Skip (no labels):", fpath)
            continue

        z0 = estimate_ground_z(pts, cell=args.grid)
        z_norm = pts[:,2] - z0

        labels_meta = apply_rules(labels_src, z_norm, rules, debris_policy, height_th, ignore_ids)

        binc_src = np.bincount(labels_src.clip(0,255), minlength=256)
        binc_meta = np.bincount(labels_meta.clip(0,255), minlength=256)
        total_counts += binc_src
        meta_counts += binc_meta

        rel = os.path.basename(fpath)
        out_f = os.path.join(args.out_dir, rel)
        save_npz(out_f, pts.astype(np.float32), feats.astype(np.float32) if feats is not None else None, labels_meta.astype(np.int32))

        src_nonzero = {i:int(c) for i,c in enumerate(binc_src) if c>0}
        meta_nonzero = {i:int(c) for i,c in enumerate(binc_meta) if c>0}
        print(f"[OK] {rel}")
        print("  src:", src_nonzero)
        print("  meta:", meta_nonzero)

    print("\n=== SUMMARY (all files) ===")
    src_nonzero = {i:int(c) for i,c in enumerate(total_counts) if c>0}
    meta_nonzero = {i:int(c) for i,c in enumerate(meta_counts) if c>0}
    print("Raw src label totals:", src_nonzero)
    print("Meta (HF 5-class) totals:", {i:int(meta_nonzero.get(i,0)) for i in [1,2,3,4,5,255]})
    print("Use ignore_index=255 in your loss.")
    print("Done.")

if __name__ == "__main__":
    main()
