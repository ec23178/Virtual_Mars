
# cahvor_to_nerf.py – bypass COLMAP and generate transforms.json directly
# from NASA CAHVOR camera model metadata embedded in PDS4 IMG/XML files.

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np

# PDS4 XML namespace constants
PDS_NS = {"pds": "http://pds.nasa.gov/pds4/pds/v1"}

IMG_HEADER_READ_BYTES = 200_000



# Low-level IMG header parsing
def _read_img_header(img_path: str) -> str:
    with open(img_path, "rb") as f:
        raw = f.read(IMG_HEADER_READ_BYTES)
    return raw.decode("latin1", errors="ignore")


def _parse_model_component(header_text: str, n: int):
    # Return a 3-tuple of floats for MODEL_COMPONENT_N.
    pattern = rf"MODEL_COMPONENT_{n}\s*=\s*\(([^)]*)\)"
    match = re.search(pattern, header_text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"MODEL_COMPONENT_{n} not found in IMG header.")
    parts = [p.strip() for p in match.group(1).split(",")]
    if len(parts) != 3:
        raise ValueError(f"MODEL_COMPONENT_{n}: expected 3 values, got {len(parts)}.")
    return tuple(float(p) for p in parts)


def _read_cahvor_from_img(img_path: str):
    # Return (C, A, H, V) as 3-tuples from the IMG header.
    header = _read_img_header(img_path)
    if "MODEL_TYPE" not in header or "CAHVOR" not in header:
        raise ValueError(f"CAHVOR model block not found in IMG header: {img_path}")
    C = _parse_model_component(header, 1)
    A = _parse_model_component(header, 2)
    H = _parse_model_component(header, 3)
    V = _parse_model_component(header, 4)
    return C, A, H, V


# Geometry: CAHVOR -> NeRFStudio c2w and intrinsics
def cahvor_intrinsics(C, A, H, V):
    # Return (fx, fy, cx, cy) from CAHVOR vectors.
    A = np.array(A, dtype=np.float64)
    A = A / np.linalg.norm(A)
    H = np.array(H, dtype=np.float64)
    V = np.array(V, dtype=np.float64)

    h_perp = H - np.dot(H, A) * A
    v_perp = V - np.dot(V, A) * A

    fx = float(np.linalg.norm(h_perp))
    fy = float(np.linalg.norm(v_perp))
    cx = float(np.dot(H, A))
    cy = float(np.dot(V, A))
    return fx, fy, cx, cy


def cahvor_to_c2w(C, A, H, V):
    # Convert CAHVOR vectors to a 4x4 NeRFStudio (OpenGL) c2w matrix.
    C_np = np.array(C, dtype=np.float64)
    A_np = np.array(A, dtype=np.float64)
    H_np = np.array(H, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)

    A_np = A_np / np.linalg.norm(A_np)

    h_perp = H_np - np.dot(H_np, A_np) * A_np
    r_h = h_perp / np.linalg.norm(h_perp)

    v_perp = V_np - np.dot(V_np, A_np) * A_np
    r_v = v_perp / np.linalg.norm(v_perp)

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] =  r_h
    c2w[:3, 1] = -r_v
    c2w[:3, 2] = -A_np
    c2w[:3, 3] =  C_np

    det = float(np.linalg.det(c2w[:3, :3]))
    if abs(det - 1.0) > 0.01:
        import warnings
        warnings.warn(
            f"c2w rotation det={det:.4f} (expected +1.0). "
            "CAHVOR vectors may form a left-handed coordinate system.",
            stacklevel=2,
        )

    return c2w.tolist()


# Utility
def load_keep_list(path: str):
    stems = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                stems.add(os.path.splitext(line)[0])
    if not stems:
        raise ValueError(f"keep_list.txt is empty or unreadable: {path}")
    return stems


def find_img_file(stem: str, xml_path: str, img_dir_override=None):
    # Locate the .IMG file for a given stem.
    if img_dir_override:
        candidate = os.path.join(img_dir_override, f"{stem}.IMG")
        if os.path.isfile(candidate):
            return candidate

    # Standard PDS4 layout: IMG_files/ in parent of the xml-dir
    project_root = os.path.dirname(os.path.dirname(xml_path))
    candidate = os.path.join(project_root, "IMG_files", f"{stem}.IMG")
    if os.path.isfile(candidate):
        return candidate

    return None


# Main
def main():
    parser = argparse.ArgumentParser(
        description="Generate NeRFStudio transforms.json from NASA CAHVOR metadata."
    )
    parser.add_argument("--xml-dir",   required=True,
        help="Directory containing PDS4 XML metadata files.")
    parser.add_argument("--img-dir",   default=None,
        help="Directory containing .IMG files (optional; auto-resolved if omitted).")
    parser.add_argument("--image-dir", required=True,
        help="Directory containing the demosaiced PNG images NeRFStudio will load.")
    parser.add_argument("--keep-list", default=None,
        help="Path to keep_list.txt; only stems in this list are included.")
    parser.add_argument("--output",    required=True,
        help="Path to write transforms.json.")
    args = parser.parse_args()

    xml_dir     = os.path.abspath(args.xml_dir)
    image_dir   = os.path.abspath(args.image_dir)
    output_path = os.path.abspath(args.output)
    output_dir  = os.path.dirname(output_path)

    if not os.path.isdir(xml_dir):
        print(f"ERROR: --xml-dir does not exist: {xml_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(image_dir):
        print(f"ERROR: --image-dir does not exist: {image_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    keep_stems = None
    if args.keep_list:
        keep_stems = load_keep_list(os.path.abspath(args.keep_list))
        print(f"Keep list loaded: {len(keep_stems)} stems from {args.keep_list}")

    xml_files = sorted(fn for fn in os.listdir(xml_dir) if fn.lower().endswith(".xml"))
    if not xml_files:
        print(f"ERROR: No XML files found in {xml_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(xml_files)} XML files in {xml_dir}")

    frames = []
    fx_list, fy_list, cx_list, cy_list = [], [], [], []
    skipped_keeplist = skipped_no_img = skipped_no_png = skipped_parse = 0

    # Collect all valid frames first, then majority-vote on (w, h) to automatically drop stray thumbnails with a different resolution.
    frame_candidates = []

    for xml_filename in xml_files:
        stem     = os.path.splitext(xml_filename)[0]
        xml_path = os.path.join(xml_dir, xml_filename)

        if keep_stems is not None and stem not in keep_stems:
            skipped_keeplist += 1
            continue

        img_path = find_img_file(stem, xml_path, args.img_dir)
        if img_path is None:
            print(f"  WARNING: IMG not found for {stem} -- skipping")
            skipped_no_img += 1
            continue

        png_path = os.path.join(image_dir, f"{stem}.png")
        if not os.path.isfile(png_path):
            print(f"  WARNING: PNG not found: {png_path} -- skipping")
            skipped_no_png += 1
            continue

        # Read actual PNG dimensions per-frame using PIL (NOT cached globally).
        try:
            from PIL import Image as _PIL_Image
            with _PIL_Image.open(png_path) as _im:
                png_w, png_h = _im.size   # PIL returns (width, height)
        except Exception as e:
            print(f"  WARNING: Could not read PNG size for {stem}: {e} -- skipping")
            skipped_no_png += 1
            continue

        try:
            C, A, H, V = _read_cahvor_from_img(img_path)
        except Exception as e:
            print(f"  WARNING: CAHVOR parse failed for {stem}: {e} -- skipping")
            skipped_parse += 1
            continue

        fx, fy, cx, cy = cahvor_intrinsics(C, A, H, V)
        c2w = cahvor_to_c2w(C, A, H, V)
        rel_png = os.path.relpath(png_path, output_dir).replace("\\", "/")

        frame_candidates.append({
            "file_path": rel_png,
            "transform_matrix": c2w,
            "png_w": png_w, "png_h": png_h,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "stem": stem,
        })
        print(f"  OK  {stem}  size={png_w}x{png_h}  fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}")


    # Majority-vote on (width, height) -- drop frames that don't match
    size_counts = Counter((c["png_w"], c["png_h"]) for c in frame_candidates)
    if not size_counts:
        print("\nERROR: No valid frames found.", file=sys.stderr)
        sys.exit(1)

    majority_w, majority_h = size_counts.most_common(1)[0][0]
    print(f"\nImage size majority vote: {majority_w}x{majority_h}  (distribution: {dict(size_counts)})")

    skipped_size = 0
    for cand in frame_candidates:
        if cand["png_w"] != majority_w or cand["png_h"] != majority_h:
            print(f"  SKIP (size mismatch {cand['png_w']}x{cand['png_h']}): {cand['stem']}")
            skipped_size += 1
            continue
        frames.append({
            "file_path": cand["file_path"],
            "transform_matrix": cand["transform_matrix"],
        })
        fx_list.append(cand["fx"])
        fy_list.append(cand["fy"])
        cx_list.append(cand["cx"])
        cy_list.append(cand["cy"])

    if skipped_size:
        print(f"  Dropped {skipped_size} frame(s) with non-majority image size.")

    if not frames:
        print("\nERROR: No frames generated. Check --xml-dir and --image-dir paths.", file=sys.stderr)
        sys.exit(1)

    fx_arr = np.array(fx_list)
    fy_arr = np.array(fy_list)

    print(f"\nIntrinsics summary ({len(frames)} frames):")
    print(f"  fx  mean={fx_arr.mean():.3f}  std={fx_arr.std():.3f}  min={fx_arr.min():.3f}  max={fx_arr.max():.3f}")
    print(f"  fy  mean={fy_arr.mean():.3f}  std={fy_arr.std():.3f}  min={fy_arr.min():.3f}  max={fy_arr.max():.3f}")
    print(f"  cx  mean={np.mean(cx_list):.3f}")
    print(f"  cy  mean={np.mean(cy_list):.3f}")

    if fx_arr.std() > 20 or fy_arr.std() > 20:
        print("  WARNING: High focal-length variance -- images from different cameras may be mixed.")

    # Write transforms.json
    transforms = {
        "camera_model": "OPENCV",
        "fl_x": float(fx_arr.mean()),
        "fl_y": float(fy_arr.mean()),
        "cx":   float(np.mean(cx_list)),
        "cy":   float(np.mean(cy_list)),
        "w":    majority_w,
        "h":    majority_h,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "frames": frames,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transforms, f, indent=2)

    print(f"\nWrote {len(frames)} frames -> {output_path}")
    if skipped_keeplist: print(f"  Skipped (not in keep list):   {skipped_keeplist}")
    if skipped_no_img:   print(f"  Skipped (IMG not found):      {skipped_no_img}")
    if skipped_no_png:   print(f"  Skipped (PNG not found):      {skipped_no_png}")
    if skipped_parse:    print(f"  Skipped (CAHVOR parse error): {skipped_parse}")


if __name__ == "__main__":
    main()