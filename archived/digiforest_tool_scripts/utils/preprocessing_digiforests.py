#!/usr/bin/env python3
import os, json, argparse
from pathlib import Path
import numpy as np
import torch
import open3d as o3d   # pip/conda install open3d

IGNORE_IDX = 255  # for CE ignore_index

# ---- EDIT THIS: your reduced-class mapping (example) ----
# raw semantic id -> reduced meta id
DF_SEM_TO_META = {
    0: IGNORE_IDX,  # unlabeled
    1: 1,  # ground
    2: 2,  # shrub
    3: 3,  # stem
    4: 4,  # canopy
}

def read_pcd_xyz_i(pcd_path: Path):
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    xyz = np.asarray(pcd.points, dtype=np.float32)
    # intensity may be in pcd.colors or pcd.intensity depending on writer
    if len(pcd.colors):
        # some exports store reflectance in colors; use grayscale
        rgb = np.asarray(pcd.colors, dtype=np.float32)
        inten = rgb.mean(axis=1, keepdims=True)
    else:
        # no intensity channel available
        inten = np.zeros((xyz.shape[0], 1), dtype=np.float32)
    # normalize intensity to [0,1]
    if inten.size and inten.max() > inten.min():
        inten = (inten - inten.min()) / (inten.max() - inten.min() + 1e-6)
    return xyz, inten

def read_label_file(label_path: Path):
    """Returns semantic_raw (N,), instance_raw (N,) from packed uint32 .label"""
    arr = np.fromfile(label_path, dtype=np.uint32)
    sem = (arr & 0xFFFF).astype(np.int32)
    inst = (arr >> 16).astype(np.int32)
    return sem, inst

def remap_semantic(sem_raw: np.ndarray, lut: dict, ignore_idx=IGNORE_IDX):
    out = np.full_like(sem_raw, ignore_idx)
    for k, v in lut.items():
        out[sem_raw == int(k)] = int(v)
    return out

def reindex_instances(inst_raw: np.ndarray):
    """Map instance ids to contiguous [0..K], keep 0 as noise/unassigned."""
    inst = inst_raw.copy()
    uniq = np.unique(inst)
    mapping = {0: 0}
    next_id = 1
    for u in uniq:
        if u == 0: 
            continue
        mapping[int(u)] = next_id
        next_id += 1
    out = np.zeros_like(inst)
    for src, dst in mapping.items():
        out[inst == src] = dst
    return out

def center_global(xyz: np.ndarray):
    ctr = xyz.mean(axis=0, keepdims=True)
    return xyz - ctr, ctr.squeeze(0)

def compute_offsets(xyz_centered: np.ndarray, inst_ids: np.ndarray, ignore_id=0):
    offsets = np.zeros_like(xyz_centered, dtype=np.float32)
    for iid in np.unique(inst_ids):
        if iid == ignore_id:
            continue
        m = (inst_ids == iid)
        c = xyz_centered[m].mean(axis=0)
        offsets[m] = c - xyz_centered[m]
    return offsets

def save_pt(out_path: Path, xyz_c, inten, sem_meta, inst_re, offsets, center, meta):
    out = {
        "points": torch.from_numpy(xyz_c),
        "feats": torch.from_numpy(inten),
        "labels": torch.from_numpy(sem_meta.astype(np.int64)),
        "instance_ids": torch.from_numpy(inst_re.astype(np.int32)),
        "offset": torch.from_numpy(offsets),
        "center": torch.from_numpy(center.astype(np.float32)),
        "meta": meta,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)


def read_pt_label(file_path: Path):
    """
    Read the label from processed .pt file and return a numpy array of class ids.
    """
    data = torch.load(file_path)
    labels = data["labels"].numpy()
    labels[labels==IGNORE_IDX] = 0 # map ignore idx to 0 for analysis
    labels = labels.astype(np.int32).reshape((-1,))
    # print(f"Shape of class_id array: {labels.shape}")
    # print(f"Unique values in class_id array: {np.unique(labels)}")
    return labels

def process_ground_dir(raw_root: Path, out_root: Path, split_name: str):
    # expects .../raw/{train|val}/{date}/expXX-yy/{ground_clouds,labels}
    processed = []
    for labels_dir in raw_root.rglob("labels"):
        clouds_dir = labels_dir.parent / "ground_clouds"
        if not clouds_dir.exists(): 
            continue
        for lbl in sorted(labels_dir.glob("cloud_*.label")):
            stem = lbl.stem  # e.g., cloud_1679...
            pcd = clouds_dir / f"{stem}.pcd"
            if not pcd.exists():
                print(f"[WARN] missing pcd for {lbl}")
                continue

            xyz, inten = read_pcd_xyz_i(pcd)
            sem_raw, inst_raw = read_label_file(lbl)

            # sanity
            if len(xyz) != len(sem_raw):
                print(f"[SKIP] size mismatch {pcd} ({len(xyz)}) vs {lbl} ({len(sem_raw)})")
                continue

            sem_meta = remap_semantic(sem_raw, DF_SEM_TO_META, IGNORE_IDX)
            inst_re = reindex_instances(inst_raw)
            xyz_c, center = center_global(xyz)
            offsets = compute_offsets(xyz_c, inst_re, ignore_id=0)

            rel = pcd.relative_to(raw_root)
            out_path = out_root / split_name / rel.with_suffix(".pt")
            meta = {
                "source": "digiforests-ground",
                "raw_pcd": str(pcd),
                "raw_label": str(lbl),
                "split": split_name
            }
            save_pt(out_path, xyz_c, inten, sem_meta, inst_re, offsets, center, meta)
            processed.append(out_path)

    return processed

def process_aerial_dir(raw_root: Path, out_root: Path, split_name: str):
    # expects .../raw/{train|val}/{date}/expXX-yy/aerial_clouds
    processed = []
    for clouds_dir in raw_root.rglob("aerial_clouds"):
        for pcd in sorted(clouds_dir.glob("cloud_*.pcd")):
            xyz, inten = read_pcd_xyz_i(pcd)
            # unlabeled target
            sem_meta = np.full((xyz.shape[0],), IGNORE_IDX, dtype=np.int32)
            inst_re = np.zeros((xyz.shape[0],), dtype=np.int32)
            xyz_c, center = center_global(xyz)
            offsets = np.zeros_like(xyz_c, dtype=np.float32)

            rel = pcd.relative_to(raw_root)
            out_path = out_root / split_name / rel.with_suffix(".pt")
            meta = {
                "source": "digiforests-aerial",
                "raw_pcd": str(pcd),
                "split": split_name
            }
            save_pt(out_path, xyz_c, inten, sem_meta, inst_re, offsets, center, meta)
            processed.append(out_path)
    return processed

def write_manifest(paths, manifest_path: Path):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump({"samples": [str(p) for p in paths]}, f, indent=2)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="root_dir that contains digiforests-*")
    ap.add_argument("--out", type=Path, required=True, help="output processed root")
    args = ap.parse_args()

    all_paths = []
    for ds in sorted(args.root.glob("digiforests-*")):
        raw = ds / "raw"
        if not raw.exists(): 
            continue
        if "ground" in ds.name:
            # ground has train/val in your tree
            for split in ["train", "val"]:
                split_dir = raw / split
                if split_dir.exists():
                    all_paths += process_ground_dir(split_dir, args.out / ds.name, split)
        else:
            # aerial has train/val
            for split in ["train", "val"]:
                split_dir = raw / split
                if split_dir.exists():
                    all_paths += process_aerial_dir(split_dir, args.out / ds.name, split)

    # one global manifest (optional) + per-dataset manifests
    write_manifest(all_paths, args.out / "manifest.all.json")

if __name__ == "__main__":
    main()
