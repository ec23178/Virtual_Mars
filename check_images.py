from PIL import Image
import numpy as np
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder1", required=True, help="Reference/original folder")
    parser.add_argument("--folder2", required=True, help="New folder to compare")
    args = parser.parse_args()

    files1 = sorted([f for f in os.listdir(args.folder1) if f.lower().endswith(".png")])
    files2 = sorted([f for f in os.listdir(args.folder2) if f.lower().endswith(".png")])

    if len(files1) != len(files2):
        print("WARNING: Different number of images!")
        print(len(files1), "vs", len(files2))

    total_mean = 0
    total_max = 0
    count = 0

    print("\nChecking images...\n")

    for f in files1:
        path1 = os.path.join(args.folder1, f)
        path2 = os.path.join(args.folder2, f)

        if not os.path.exists(path2):
            print(f"Missing in new folder: {f}")
            continue

        img1 = np.array(Image.open(path1).convert("RGB"))
        img2 = np.array(Image.open(path2).convert("RGB"))

        if img1.shape != img2.shape:
            print(f"{f}")
            print(f"  SHAPE MISMATCH: old={img1.shape}, new={img2.shape}")
            print("")
            continue

        diff = np.abs(img1.astype(int) - img2.astype(int))

        mean_diff = np.mean(diff)
        max_diff = np.max(diff)

        total_mean += mean_diff
        total_max = max(total_max, max_diff)
        count += 1

        print(f"{f}")
        print(f"  Mean diff: {mean_diff:.2f}")
        print(f"  Max diff : {max_diff}")
        print("")

    print("===== SUMMARY =====")
    if count > 0:
        print(f"Average mean diff: {total_mean / count:.2f}")
        print(f"Worst max diff  : {total_max}")
    else:
        print("No images could be compared because all had shape mismatches.")

if __name__ == "__main__":
    main()