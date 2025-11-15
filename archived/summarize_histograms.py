
import argparse, glob, os, numpy as np

def main():
    ap = argparse.ArgumentParser(description="Summarize remapped label histograms over .npz files (expects labels_meta).")
    ap.add_argument("--dir", required=True, help="directory with remapped .npz files")
    ap.add_argument("--glob", default="*.npz")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, args.glob)))
    if not files:
        print("No files.")
        return

    total = np.zeros(256, dtype=np.int64)
    for fp in files:
        d = np.load(fp)
        if "labels_meta" not in d:
            print("Skip (no labels_meta):", fp)
            continue
        y = d["labels_meta"].astype(np.int32)
        total += np.bincount(y.clip(0,255), minlength=256)

    print("Class totals:")
    headers = ["1-forest_floor","2-tree_trunks","3-branches_canopy","4-understory","5-objects","255-ignore"]
    ids = [1,2,3,4,5,255]
    for h, i in zip(headers, ids):
        print(f"{h}: {int(total[i])}")
    print("Done.")

if __name__ == "__main__":
    main()
