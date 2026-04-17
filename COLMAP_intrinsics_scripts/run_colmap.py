# run_colmap.py  (UPDATED -- cross-platform: works on Windows and JupyterHub/Linux)
#
# Runs a full COLMAP sparse reconstruction pipeline using the intrinsics
# previously exported by main.py (compute_intrinsics + export_intrinsics).
#
# -----------------------------------------------------------------------
# Platform detection
# -----------------------------------------------------------------------
# On Windows  the script calls the COLMAP.bat launcher.
# On Linux    the script calls the 'colmap' binary directly (same arguments).
# You can override auto-detection with --colmap-exe.
#
# Windows default exe: C:\Users\munee\Desktop\FINAL YEAR PROJECT\colmap-x64-windows-nocuda\COLMAP.bat
# Linux  default exe: colmap   (must be on PATH, see JupyterHub setup in PIPELINE_COMMANDS.md)
#
# -----------------------------------------------------------------------
# Folder layout expected before running
# -----------------------------------------------------------------------
#   <dataset-dir>/
#     images/          <- demosaiced PNG images selected for COLMAP
#     sparse/          <- created automatically if missing
#     database.db      <- created by COLMAP feature_extractor (first run)
#
#   <intrinsics-dir>/
#     cameras.txt          <- COLMAP camera entries (one per unique size/lens group)
#     image_camera_map.txt <- maps each PNG filename to its camera ID
#
# Both folders are produced by running the pipeline in this order:
#   1. filter_xml_metadata.py   -- filters raw data
#   2. vic2png                  -- converts IMG -> PNG
#   3. demosaic_batch.py        -- demosaics PNG Bayer -> RGB PNG
#   4. select_colmap_images.py  -- thins sequences, copies to images/
#   5. main.py (intrinsics)     -- parses CAHVOR, exports cameras.txt etc.
#   6. THIS SCRIPT              -- runs feature extraction, matching, mapper
#
# -----------------------------------------------------------------------
# Usage
# -----------------------------------------------------------------------
#
# Windows (auto-detect):
#   python run_colmap.py \
#       --dataset-dir  COLMAP/colmap_bilinear \
#       --intrinsics-dir COLMAP/intrinsics_bilinear \
#       --use-gpu --reset-db
#
# JupyterHub / Linux (auto-detect, colmap must be on PATH):
#   python run_colmap.py \
#       --dataset-dir  ~/COLMAP/colmap_bilinear \
#       --intrinsics-dir ~/COLMAP/intrinsics_bilinear \
#       --use-gpu --reset-db
#
# Override executable (if colmap is not on PATH):
#   python run_colmap.py \
#       --dataset-dir  ~/COLMAP/colmap_bilinear \
#       --intrinsics-dir ~/COLMAP/intrinsics_bilinear \
#       --colmap-exe /usr/local/bin/colmap \
#       --use-gpu --reset-db
#
# Sequential matcher (faster, only valid for ordered panoramic sweeps):
#   python run_colmap.py \
#       --dataset-dir  COLMAP/colmap_bilinear \
#       --intrinsics-dir COLMAP/intrinsics_bilinear \
#       --matcher sequential --use-gpu --reset-db

import argparse
import platform
import subprocess
import sys
from pathlib import Path


# -----------------------------------------------------------------------
# Platform-aware default executable
# -----------------------------------------------------------------------
def default_colmap_exe():
    """Return the platform-appropriate default COLMAP executable path."""
    if platform.system() == "Windows":
        return (
            r"C:\Users\munee\Desktop\FINAL_YEAR_PROJECT"
            r"\colmap-x64-windows-nocuda\COLMAP.bat"
        )
    else:
        # Linux / macOS: assume 'colmap' is on the system PATH.
        # On JupyterHub install with: conda install -c conda-forge colmap
        # or: pip install pycolmap  (Python bindings, different interface)
        return "colmap"


# -----------------------------------------------------------------------
# Read cameras.txt
# -----------------------------------------------------------------------

def parse_cameras_txt(cameras_path):
    """Read cameras.txt -> {camera_id: {model, width, height, params}}."""
    cameras = {}
    with open(cameras_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"Invalid cameras.txt line: {line!r}")
            camera_id = int(parts[0])
            cameras[camera_id] = {
                "model":  parts[1],
                "width":  int(parts[2]),
                "height": int(parts[3]),
                "params": ",".join(parts[4:]),
            }
    if not cameras:
        raise ValueError("No valid camera entries found in cameras.txt")
    return cameras


# -----------------------------------------------------------------------
# Read image_camera_map.txt
# -----------------------------------------------------------------------

def parse_image_camera_map(map_path):
    """Read image_camera_map.txt -> {camera_id: [image_name, ...]}."""
    grouped      = {}
    header_done  = False
    with open(map_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not header_done:
                header_done = True
                continue
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"Invalid image_camera_map.txt line: {line!r}")
            image_name = parts[1]
            camera_id  = int(parts[3])
            grouped.setdefault(camera_id, []).append(image_name)
    if not grouped:
        raise ValueError("No valid image-to-camera mappings found")
    return grouped


# -----------------------------------------------------------------------
# Write a per-camera image list file
# -----------------------------------------------------------------------

def write_image_list(dataset_dir, camera_id, image_names):
    """Write camera_<id>_images.txt for one camera group."""
    list_path = Path(dataset_dir) / f"camera_{camera_id}_images.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for name in image_names:
            f.write(f"{name}\n")
    return str(list_path)


# -----------------------------------------------------------------------
# Subprocess runner
# -----------------------------------------------------------------------

def run_command(command):
    """Run a shell command and raise on failure."""
    # Pretty-print the command for logging.
    display = " ".join(
        f'"{p}"' if (" " in str(p) or "\\" in str(p)) else str(p)
        for p in command
    )
    print(f"\nRunning:\n  {display}\n")
    # On Windows, .bat files must be run through cmd.exe — subprocess cannot
    # execute them directly via CreateProcess. Prepend ["cmd", "/c"] so the
    # shell interprets the batch file correctly.
    actual_command = list(command)
    if platform.system() == "Windows" and str(actual_command[0]).lower().endswith(".bat"):
        actual_command = ["cmd", "/c"] + actual_command
    result = subprocess.run(actual_command, check=False)
    if result.returncode != 0:
        print(f"\n[ERROR] Command exited with code {result.returncode}")
        print("Check COLMAP output above for details.")
        sys.exit(result.returncode)


# -----------------------------------------------------------------------
# COLMAP pipeline steps
# -----------------------------------------------------------------------

def run_feature_extraction(exe, database_path, image_path,
                            image_list_path, camera_model,
                            camera_params, use_gpu):
    """Run COLMAP feature_extractor for one camera group."""
    command = [
        exe,
        "feature_extractor",
        "--database_path",             str(database_path),
        "--image_path",                str(image_path),
        "--image_list_path",           str(image_list_path),
        "--ImageReader.camera_model",  camera_model,
        "--ImageReader.camera_params", camera_params,
        "--ImageReader.single_camera", "1",
        "--FeatureExtraction.use_gpu", "1" if use_gpu else "0",
    ]
    run_command(command)


def run_matching(exe, database_path, matcher_type, use_gpu):
    """Run the chosen COLMAP matcher across all images in the database."""
    command = [
        exe,
        f"{matcher_type}_matcher",
        "--database_path",           str(database_path),
        "--FeatureMatching.use_gpu", "1" if use_gpu else "0",
    ]
    run_command(command)


def run_mapper(exe, database_path, image_path, sparse_path):
    """Run COLMAP sparse reconstruction with intrinsics held fixed."""
    command = [
        exe,
        "mapper",
        "--database_path",                     str(database_path),
        "--image_path",                        str(image_path),
        "--output_path",                       str(sparse_path),
        "--Mapper.ba_refine_focal_length",     "0",
        "--Mapper.ba_refine_principal_point",  "0",
        "--Mapper.ba_refine_extra_params",     "0",
    ]
    run_command(command)


def run_model_converter(exe, sparse_path, output_path, output_type="TXT"):
    """Convert a COLMAP binary sparse model to TXT format (for debugging/inspection)."""
    # Find the first numbered sub-model folder (e.g. sparse/0/).
    model_dirs = sorted(
        [d for d in Path(sparse_path).iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name)
    )
    if not model_dirs:
        print("  [INFO] No numbered sub-models found in sparse/ -- skipping conversion.")
        return

    model_dir = model_dirs[0]
    command = [
        exe,
        "model_converter",
        "--input_path",  str(model_dir),
        "--output_path", str(output_path),
        "--output_type", output_type,
    ]
    run_command(command)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a full COLMAP sparse pipeline using exported CAHVOR intrinsics. "
            "Works on both Windows (COLMAP.bat) and Linux/JupyterHub (colmap binary). "
            "Platform is auto-detected; override with --colmap-exe if needed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--dataset-dir",    required=True,
        help="COLMAP workspace folder, e.g. COLMAP/colmap_bilinear")
    parser.add_argument("--intrinsics-dir", required=True,
        help="Folder containing cameras.txt and image_camera_map.txt")
    parser.add_argument("--colmap-exe",     default=None,
        help=(
            "Path to COLMAP executable. "
            "Auto-detected: Windows uses COLMAP.bat, Linux uses 'colmap'. "
            "Override if COLMAP is not at the default location or not on PATH."
        ))
    parser.add_argument("--matcher",
        default="exhaustive",
        choices=["exhaustive", "sequential"],
        help=(
            "Feature matcher type. "
            "'exhaustive' checks all pairs (best for small unordered datasets). "
            "'sequential' only matches neighbouring frames (faster for panoramic sweeps "
            "where images are ordered by capture time). Default: exhaustive."
        ))
    parser.add_argument("--use-gpu", action="store_true",
        help="Enable GPU acceleration for feature extraction and matching.")
    parser.add_argument("--reset-db", action="store_true",
        help="Delete any existing database.db before running (clean restart).")
    parser.add_argument("--convert-to-txt", action="store_true",
        help=(
            "After reconstruction, convert the first sparse model from binary "
            "to TXT format inside sparse/txt/. Useful for inspection."
        ))

    args = parser.parse_args()

    # Resolve executable
    exe = args.colmap_exe if args.colmap_exe else default_colmap_exe()
    detected_platform = platform.system()
    print(f"\nPlatform : {detected_platform}")
    print(f"COLMAP   : {exe}")

    # Validate paths
    dataset_dir    = Path(args.dataset_dir).expanduser().resolve()
    intrinsics_dir = Path(args.intrinsics_dir).expanduser().resolve()

    image_path    = dataset_dir / "images"
    sparse_path   = dataset_dir / "sparse"
    database_path = dataset_dir / "database.db"
    cameras_path  = intrinsics_dir / "cameras.txt"
    image_map_path = intrinsics_dir / "image_camera_map.txt"

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")
    if not image_path.is_dir():
        raise FileNotFoundError(
            f"images/ folder not found inside dataset dir: {image_path}\n"
            "Run select_colmap_images.py first to populate it."
        )
    if not intrinsics_dir.is_dir():
        raise FileNotFoundError(f"Intrinsics folder not found: {intrinsics_dir}")
    if not cameras_path.is_file():
        raise FileNotFoundError(
            f"cameras.txt not found: {cameras_path}\n"
            "Run main.py (intrinsics pipeline) first."
        )
    if not image_map_path.is_file():
        raise FileNotFoundError(
            f"image_camera_map.txt not found: {image_map_path}\n"
            "Run main.py (intrinsics pipeline) first."
        )

    # Count images
    n_images = len(list(image_path.glob("*.png")))
    print(f"Images   : {n_images} PNGs in {image_path}")
    if n_images == 0:
        raise FileNotFoundError(
            f"No PNG files found in images/. "
            "Run select_colmap_images.py to populate the folder."
        )

    # Create sparse output folder
    sparse_path.mkdir(parents=True, exist_ok=True)

    # Optionally delete old database
    if args.reset_db and database_path.exists():
        print(f"\nDeleting existing database: {database_path}")
        database_path.unlink()

    # Read intrinsics files
    cameras        = parse_cameras_txt(cameras_path)
    grouped_images = parse_image_camera_map(image_map_path)

    print(f"\nCamera groups: {len(cameras)}")
    for cid, cam in sorted(cameras.items()):
        n = len(grouped_images.get(cid, []))
        print(f"  Camera {cid}: {cam['model']} {cam['width']}x{cam['height']} -- {n} images")

    # ----------------------------------------------------------------
    # Step 1: Feature extraction, one call per camera group
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: Feature Extraction")
    print("=" * 60)

    for camera_id, image_names in sorted(grouped_images.items()):
        if camera_id not in cameras:
            raise ValueError(
                f"Camera ID {camera_id} in image_camera_map.txt "
                f"has no entry in cameras.txt"
            )
        cam = cameras[camera_id]
        print(f"\n  Camera {camera_id} ({cam['model']} {cam['width']}x{cam['height']}) "
              f"-- {len(image_names)} images")

        list_path = write_image_list(dataset_dir, camera_id, image_names)
        run_feature_extraction(
            exe           = exe,
            database_path = database_path,
            image_path    = image_path,
            image_list_path = list_path,
            camera_model  = cam["model"],
            camera_params = cam["params"],
            use_gpu       = args.use_gpu,
        )

    # ----------------------------------------------------------------
    # Step 2: Feature matching
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"STEP 2: Feature Matching ({args.matcher})")
    print("=" * 60)
    run_matching(exe, database_path, args.matcher, args.use_gpu)

    # ----------------------------------------------------------------
    # Step 3: Sparse reconstruction (mapper)
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Sparse Reconstruction (mapper)")
    print("=" * 60)
    run_mapper(exe, database_path, image_path, sparse_path)

    # ----------------------------------------------------------------
    # Optional: convert to TXT for inspection
    # ----------------------------------------------------------------
    if args.convert_to_txt:
        txt_path = sparse_path / "txt"
        txt_path.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 60)
        print("STEP 4: Convert Model to TXT")
        print("=" * 60)
        run_model_converter(exe, sparse_path, txt_path, output_type="TXT")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COLMAP PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Dataset dir    : {dataset_dir}")
    print(f"  Intrinsics dir : {intrinsics_dir}")
    print(f"  Database       : {database_path}")
    print(f"  Sparse output  : {sparse_path}")

    # Count registered models
    model_dirs = sorted(
        [d for d in sparse_path.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name)
    ) if sparse_path.exists() else []

    if model_dirs:
        print(f"\n  Sub-models found: {len(model_dirs)}")
        for md in model_dirs:
            print(f"    {md}")
        print(
            "\n  NEXT STEP:\n"
            "    Run ns-process-data to convert to NeRFStudio format.\n"
            "    See PIPELINE_COMMANDS.md -- Section 4 (NeRFStudio)."
        )
    else:
        print(
            "\n  WARNING: No numbered sub-models found in sparse/.\n"
            "  COLMAP may have failed to register any images.\n"
            "  Check the output above for 'Registered N images' messages.\n"
            "  If N < 5, consider:\n"
            "    - Using --reset-db and rerunning\n"
            "    - Checking that the images/ folder has the correct PNGs\n"
            "    - Verifying cameras.txt params match the actual images\n"
            "    - Running with --matcher exhaustive if you used sequential"
        )

    print()


if __name__ == "__main__":
    main()