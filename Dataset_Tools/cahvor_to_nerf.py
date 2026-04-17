"""
cahvor_to_nerf.py – bypass COLMAP and generate transforms.json directly
from NASA CAHVOR camera model metadata embedded in PDS4 IMG/XML files.

WHY THIS SCRIPT EXISTS
----------------------
COLMAP fails on Mastcam panoramic sequences (MCAM04379, MCAM04380) because
the rover mast rotates in place with zero translational baseline.  This
produces a degenerate fundamental matrix, so COLMAP's incremental SfM cannot
triangulate any 3-D points.

The CAHVOR vectors stored in the PDS4 IMG headers already encode the exact,
GPS-precision camera pose for every frame.  We can therefore skip COLMAP
entirely and write transforms.json directly.

CAHVOR VECTOR MEANINGS (PDS4 IMG header MODEL_COMPONENT_N)
-----------------------------------------------------------
  Component 1 = C   camera centre position in world (rover-body) coordinates
  Component 2 = A   optical axis unit vector  (camera looks along +A)
  Component 3 = H   horizontal pixel vector   (encodes fx and principal cx)
  Component 4 = V   vertical pixel vector     (encodes fy and principal cy)

INTRINSICS DERIVED FROM CAHVOR
-------------------------------
  h_perp = H - (H·A)·A       (H component perpendicular to optical axis)
  v_perp = V - (V·A)·A       (V component perpendicular to optical axis)
  fx = |h_perp|
  fy = |v_perp|
  cx = H·A                   (principal point x in CAHVOR convention)
  cy = V·A                   (principal point y in CAHVOR convention)

NERFSTUDIO C2W CONVENTION (OpenGL – right / up / back)
-------------------------------------------------------
Camera looks along -Z.  Column layout of the 3×4 portion of c2w:
  col 0 =  r_h     (right  = h_perp / |h_perp|)
  col 1 = -r_v     (up     = flip of image-down axis)
  col 2 = -A       (back   = flip of optical-forward axis)
  col 3 =  C       (camera centre in world coordinates)

USAGE
-----
From the project root (FINAL_YEAR_PROJECT/):

  # Bilinear demosaic dataset
  python COLMAP_intrinsics_scripts\\cahvor_to_nerf.py ^
      --xml-dir   datasets\\maria_pass_filtered\\data ^
      --image-dir COLMAP\\colmap_bilinear\\images ^
      --keep-list COLMAP\\colmap_bilinear\\keep_list.txt ^
      --output    COLMAP\\colmap_bilinear\\transforms.json

  # Malvar demosaic dataset
  python COLMAP_intrinsics_scripts\\cahvor_to_nerf.py ^
      --xml-dir   datasets\\maria_pass_filtered\\data ^
      --image-dir COLMAP\\colmap_malvar\\images ^
      --keep-list COLMAP\\colmap_malvar\\keep_list.txt ^
      --output    COLMAP\\colmap_malvar\\transforms.json

The --img-dir flag is optional.  By default the script looks for IMG_files/
in the parent directory of --xml-dir (standard PDS4 layout):
  datasets/maria_pass_filtered/
    data/          ← --xml-dir points here
    IMG_files/     ← resolved automatically
"""

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

# ---------------------------------------------------------------------------
# PDS4 XML namespace constants
# ---------------------------------------------------------------------------

PDS_NS = {"pds": "http://pds.nasa.gov/pds4/pds/v1"}

# How many bytes to read from the start of an IMG file to find the CAHVOR
# metadata block.  200 kB covers all known Mastcam IMG headers.
IMG_HEADER_READ_BYTES = 200_000


# ---------------------------------------------------------------------------
# Low-level IMG header parsing
# (re-implements the helpers from parse_cahvor.py so this script is
# independently runnable without importing from that module)
# ---------------------------------------------------------------------------

def _read_img_header(img_path: str) -> str:
    """Read the first IMG_HEADER_READ_BYTES of an IMG file as text."""
    with open(img_path, "rb") as f:
        raw = f.read(IMG_HEADER_READ_BYTES)
    return raw.decode("latin1", errors="ignore")


def _parse_model_component(header_text: str, n: int):
    """Return a 3-tuple of floats for MODEL_COMPONENT_N.

    Example header line:
      MODEL_COMPONENT_2 = (1.513950e-01,-9.821702e-01,-1.113792e-01)

    Component mapping:
      1 = C (camera centre position)
      2 = A (optical axis)
      3 = H (horizontal pixel vector)
      4 = V (vertical pixel vector)
    """
    pattern = rf"MODEL_COMPONENT_{n}\s*=\s*\(([^)]*)\)"
    match = re.search(pattern, header_text, flags=re.DOTALL)
    if match is None:
        raise ValueError(
            f"MODEL_COMPONENT_{n} not found in IMG header."
        )
    parts = [p.strip() for p in match.group(1).split(",")]
    if len(parts) != 3:
        raise ValueError(
            f"MODEL_COMPONENT_{n}: expected 3 values, got {len(parts)}."
        )
    return tuple(float(p) for p in parts)


def _read_cahvor_from_img(img_path: str):
    """Return (C, A, H, V) as 3-tuples from the IMG header."""
    header = _read_img_header(img_path)
    if "MODEL_TYPE" not in header or "CAHVOR" not in header:
        raise ValueError(
            f"CAHVOR model block not found in IMG header: {img_path}"
        )
    C = _parse_model_component(header, 1)  # camera centre
    A = _parse_model_component(header, 2)  # optical axis
    H = _parse_model_component(header, 3)  # horizontal pixel vector
    V = _parse_model_component(header, 4)  # vertical pixel vector
    return C, A, H, V


# ---------------------------------------------------------------------------
# Image dimension extraction from PDS4 XML
# ---------------------------------------------------------------------------

def _read_dimensions_from_xml(xml_path: str):
    """Return (width, height) from Axis_Array elements in a PDS4 XML file."""
    root = ET.parse(xml_path).getroot()
    width = height = None

    for axis_array in root.findall(".//pds:Axis_Array", PDS_NS):
        name_el = axis_array.find("pds:axis_name", PDS_NS)
        elems_el = axis_array.find("pds:elements", PDS_NS)
        if name_el is None or elems_el is None:
            continue
        name = name_el.text.strip().lower()
        value = int(elems_el.text.strip())
        if name == "sample":
            width = value
        elif name == "line":
            height = value

    if width is None or height is None:
        raise ValueError(
            f"Could not read image dimensions from XML: {xml_path}  "
            f"(found width={width}, height={height})"
        )
    return width, height


# ---------------------------------------------------------------------------
# Geometry: CAHVOR → NeRFStudio c2w and intrinsics
# ---------------------------------------------------------------------------

def cahvor_intrinsics(C, A, H, V):
    """Compute pinhole intrinsics from CAHVOR vectors.

    Returns (fx, fy, cx, cy).

    Derivation:
      H = fx * r_h + cx * A   (by definition of the CAHVOR H vector)
      V = fy * r_v + cy * A   (by definition of the CAHVOR V vector)

    Taking the component perpendicular to A:
      h_perp = H - (H·A)*A  →  fx = |h_perp|
      v_perp = V - (V·A)*A  →  fy = |v_perp|

    Taking the component along A:
      H·A = cx
      V·A = cy
    """
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
    """Convert CAHVOR vectors to a 4×4 NeRFStudio (OpenGL) c2w matrix.

    NeRFStudio uses the OpenGL convention:
      +X = right
      +Y = up
      +Z = backward (camera looks along -Z)

    CAHVOR uses:
      +A = forward (optical axis)
      h_perp direction = right
      v_perp direction = down (image row increases downward)

    Mapping:
      right  (+X) =  r_h       (normalised h_perp)
      up     (+Y) = -r_v       (flip: image-down → world-up)
      back   (+Z) = -A         (flip: optical-forward → OpenGL-back)
      origin      =  C

    Returns a plain Python list-of-lists (4×4) for JSON serialisation.
    """
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
    c2w[:3, 0] =  r_h    # right
    c2w[:3, 1] = -r_v    # up   (flip image-down to world-up)
    c2w[:3, 2] = -A_np   # back (flip optical-fwd to OpenGL-back)
    c2w[:3, 3] =  C_np   # camera centre in world coordinates

    # Sanity: for physically mounted cameras (r_h x r_v = A, right-handed)
    # the resulting rotation matrix should have det = +1.  A det of -1
    # indicates the CAHVOR vectors form a left-handed system, which is
    # unusual for Mastcam but not impossible (e.g. if H or V are inverted).
    det = float(np.linalg.det(c2w[:3, :3]))
    if abs(det - 1.0) > 0.01:
        import warnings
        warnings.warn(
            f"c2w rotation det={det:.4f} (expected +1.0).  "
            "CAHVOR vectors may form a left-handed coordinate system.",
            stacklevel=2,
        )

    return c2w.tolist()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def load_keep_list(path: str):
    """Load keep_list.txt; return a set of file stems (no extension)."""
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
    """Locate the .IMG file for a given stem.

    Search order:
    1. Explicit --img-dir if provided.
    2. Standard PDS4 layout: IMG_files/ in the parent of the xml-dir.
       (xml lives in <root>/data/ → IMG in <root>/IMG_files/)
    Returns the path if found, else None.
    """
    if img_dir_override:
        candidate = os.path.join(img_dir_override, f"{stem}.IMG")
        if os.path.isfile(candidate):
            return candidate

    # Standard layout auto-resolution.
    project_root = os.path.dirname(os.path.dirname(xml_path))
    candidate = os.path.join(project_root, "IMG_files", f"{stem}.IMG")
    if os.path.isfile(candidate):
        return candidate

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate NeRFStudio transforms.json from NASA CAHVOR camera "
            "metadata, bypassing COLMAP entirely."
        )
    )
    parser.add_argument(
        "--xml-dir",
        required=True,
        help=(
            "Directory containing PDS4 XML metadata files.  "
            "Point this at datasets/maria_pass_filtered/data/ so that "
            "IMG_files/ is resolved automatically from the parent folder."
        ),
    )
    parser.add_argument(
        "--img-dir",
        default=None,
        help=(
            "Directory containing .IMG files.  Optional — by default the "
            "script finds IMG_files/ next to the parent of --xml-dir.  "
            "Supply only if your IMG files live somewhere else."
        ),
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help=(
            "Directory that holds the PNG images NeRFStudio will load "
            "(e.g. COLMAP/colmap_bilinear/images/).  "
            "Each frame's file_path is written relative to --output."
        ),
    )
    parser.add_argument(
        "--keep-list",
        default=None,
        help=(
            "Path to keep_list.txt produced by select_colmap_images.py.  "
            "Only images whose stem appears in the list are included."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write transforms.json.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------
    xml_dir = os.path.abspath(args.xml_dir)
    image_dir = os.path.abspath(args.image_dir)
    output_path = os.path.abspath(args.output)
    output_dir = os.path.dirname(output_path)

    if not os.path.isdir(xml_dir):
        print(f"ERROR: --xml-dir does not exist: {xml_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(image_dir):
        print(f"ERROR: --image-dir does not exist: {image_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load keep list
    # ------------------------------------------------------------------
    keep_stems = None
    if args.keep_list:
        keep_stems = load_keep_list(os.path.abspath(args.keep_list))
        print(f"Keep list loaded: {len(keep_stems)} stems from {args.keep_list}")

    # ------------------------------------------------------------------
    # Collect XML files
    # ------------------------------------------------------------------
    xml_files = sorted(
        fn for fn in os.listdir(xml_dir) if fn.lower().endswith(".xml")
    )
    if not xml_files:
        print(f"ERROR: No XML files found in {xml_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(xml_files)} XML files in {xml_dir}")

    # ------------------------------------------------------------------
    # Process each image
    # ------------------------------------------------------------------
    frames = []
    fx_list, fy_list, cx_list, cy_list = [], [], [], []
    dims_width = dims_height = None

    skipped_keeplist = 0
    skipped_no_img   = 0
    skipped_no_png   = 0
    skipped_parse    = 0

    for xml_filename in xml_files:
        stem = os.path.splitext(xml_filename)[0]
        xml_path = os.path.join(xml_dir, xml_filename)

        # ---- Keep-list filter ----------------------------------------
        if keep_stems is not None and stem not in keep_stems:
            skipped_keeplist += 1
            continue

        # ---- Resolve IMG file ----------------------------------------
        img_path = find_img_file(stem, xml_path, args.img_dir)
        if img_path is None:
            print(f"  WARNING: IMG not found for {stem} — skipping")
            skipped_no_img += 1
            continue

        # ---- Check PNG exists in image-dir ---------------------------
        png_filename = f"{stem}.png"
        png_path = os.path.join(image_dir, png_filename)
        if not os.path.isfile(png_path):
            print(f"  WARNING: PNG not found: {png_path} — skipping")
            skipped_no_png += 1
            continue

        # ---- Read image dimensions from XML (first time only) --------
        if dims_width is None:
            try:
                dims_width, dims_height = _read_dimensions_from_xml(xml_path)
            except Exception as e:
                print(f"  WARNING: Could not read dimensions from {xml_filename}: {e}")

        # ---- Parse CAHVOR from IMG header ----------------------------
        try:
            C, A, H, V = _read_cahvor_from_img(img_path)
        except Exception as e:
            print(f"  WARNING: CAHVOR parse failed for {stem}: {e} — skipping")
            skipped_parse += 1
            continue

        # ---- Compute intrinsics and pose ----------------------------
        fx, fy, cx, cy = cahvor_intrinsics(C, A, H, V)
        c2w = cahvor_to_c2w(C, A, H, V)

        fx_list.append(fx)
        fy_list.append(fy)
        cx_list.append(cx)
        cy_list.append(cy)

        # file_path relative to output directory (forward slashes for
        # cross-platform compatibility with NeRFStudio on Linux).
        rel_png = os.path.relpath(png_path, output_dir).replace("\\", "/")

        frames.append({
            "file_path": rel_png,
            "transform_matrix": c2w,
        })

        print(
            f"  OK  {stem}"
            f"  fx={fx:.1f}  fy={fy:.1f}"
            f"  cx={cx:.1f}  cy={cy:.1f}"
        )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    if not frames:
        print(
            "\nERROR: No frames were generated.  "
            "Check that --xml-dir, --image-dir, and --keep-list are correct.",
            file=sys.stderr,
        )
        sys.exit(1)

    fx_arr = np.array(fx_list)
    fy_arr = np.array(fy_list)

    print(f"\nIntrinsics summary ({len(frames)} frames):")
    print(f"  fx  mean={fx_arr.mean():.3f}  std={fx_arr.std():.3f}  "
          f"min={fx_arr.min():.3f}  max={fx_arr.max():.3f}")
    print(f"  fy  mean={fy_arr.mean():.3f}  std={fy_arr.std():.3f}  "
          f"min={fy_arr.min():.3f}  max={fy_arr.max():.3f}")
    print(f"  cx  mean={np.mean(cx_list):.3f}")
    print(f"  cy  mean={np.mean(cy_list):.3f}")

    if fx_arr.std() > 20 or fy_arr.std() > 20:
        print(
            "  WARNING: High focal-length variance.  "
            "Images from different cameras may have been mixed."
        )

    # ------------------------------------------------------------------
    # Shared intrinsics (mean across all frames; should be near-constant
    # for a single-camera panoramic sequence)
    # ------------------------------------------------------------------
    fl_x = float(fx_arr.mean())
    fl_y = float(fy_arr.mean())
    cx_out = float(np.mean(cx_list))
    cy_out = float(np.mean(cy_list))
    w = dims_width  if dims_width  is not None else 1344
    h = dims_height if dims_height is not None else 1200

    # ------------------------------------------------------------------
    # Write transforms.json
    # ------------------------------------------------------------------
    transforms = {
        # NeRFStudio reads camera_model to know which distortion params
        # are present.  We use OPENCV (= pinhole + k1,k2,p1,p2) and set
        # all distortion coefficients to 0 because CAHVOR uses its own
        # radial model (O and R vectors) that we are not applying here.
        "camera_model": "OPENCV",
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx_out,
        "cy": cy_out,
        "w": w,
        "h": h,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "frames": frames,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transforms, f, indent=2)

    print("\nWrote {} frames -> {}".format(len(frames), output_path))

    if skipped_keeplist:
        print("  Skipped (not in keep list):  {}".format(skipped_keeplist))
    if skipped_no_img:
        print("  Skipped (IMG not found):     {}".format(skipped_no_img))
    if skipped_no_png:
        print("  Skipped (PNG not found):     {}".format(skipped_no_png))
    if skipped_parse:
        print("  Skipped (CAHVOR parse error):{}".format(skipped_parse))


if __name__ == "__main__":
    main()
