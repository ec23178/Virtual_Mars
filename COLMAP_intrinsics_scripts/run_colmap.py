import argparse
import subprocess
from pathlib import Path

# Fixed default location of your COLMAP launcher on Windows.
DEFAULT_COLMAP_BAT = (
    r"C:\Users\munee\Desktop\FINAL YEAR PROJECT\colmap-x64-windows-nocuda\COLMAP.bat"
)

# Default matcher type.
DEFAULT_MATCHER = "exhaustive"


def parse_cameras_txt(cameras_path):
    # Read cameras.txt and return:
    # camera_id -> {model, width, height, params}
    cameras = {}

    with open(cameras_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Skip comments and blank lines.
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            # Expected format:
            # CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
            if len(parts) < 5:
                raise ValueError(f"Invalid cameras.txt line: {line}")

            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = ",".join(parts[4:])

            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }

    if not cameras:
        raise ValueError("No valid camera entries found in cameras.txt")

    return cameras


def parse_image_camera_map(map_path):
    # Read image_camera_map.txt and return:
    # camera_id -> [image_name1, image_name2, ...]
    grouped = {}

    with open(map_path, "r", encoding="utf-8") as f:
        header_skipped = False

        for line in f:
            line = line.strip()

            if not line:
                continue

            # Skip the header row once.
            if not header_skipped:
                header_skipped = True
                continue

            parts = line.split()

            # Expected columns:
            # file_stem image_name camera_group camera_id width height
            if len(parts) < 6:
                raise ValueError(f"Invalid image_camera_map.txt line: {line}")

            image_name = parts[1]
            camera_id = int(parts[3])

            grouped.setdefault(camera_id, []).append(image_name)

    if not grouped:
        raise ValueError("No valid image-to-camera mappings found")

    return grouped


def write_image_list(dataset_dir, camera_id, image_names):
    # Write a temporary image-list file for one camera group.
    list_path = Path(dataset_dir) / f"camera_{camera_id}_images.txt"

    with open(list_path, "w", encoding="utf-8") as f:
        for image_name in image_names:
            f.write(f"{image_name}\n")

    return str(list_path)


def run_command(command):
    # Run one shell command and stop immediately if it fails.
    print("\nRunning:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    subprocess.run(command, check=True)


def run_feature_extraction(
    colmap_bat,
    database_path,
    image_path,
    image_list_path,
    camera_model,
    camera_params,
    use_gpu
):
    # Run COLMAP feature extraction for one camera group.
    command = [
        colmap_bat,
        "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_path,
        "--image_list_path", image_list_path,
        "--ImageReader.camera_model", camera_model,
        "--ImageReader.camera_params", camera_params,
        "--ImageReader.single_camera", "1",
        "--FeatureExtraction.use_gpu", "1" if use_gpu else "0",
    ]

    run_command(command)


def run_matching(colmap_bat, database_path, matcher_type, use_gpu):
    # Run the chosen COLMAP matcher.
    matcher_command = f"{matcher_type}_matcher"

    command = [
        colmap_bat,
        matcher_command,
        "--database_path", database_path,
        "--FeatureMatching.use_gpu", "1" if use_gpu else "0",
    ]

    run_command(command)


def run_mapper(colmap_bat, database_path, image_path, sparse_path):
    # Run COLMAP sparse reconstruction while keeping the supplied intrinsics fixed.
    command = [
        colmap_bat,
        "mapper",
        "--database_path", database_path,
        "--image_path", image_path,
        "--output_path", sparse_path,
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0",
    ]

    run_command(command)


def main():
    # Set up command-line arguments.
    parser = argparse.ArgumentParser(
        description="Run a full COLMAP sparse pipeline using exported intrinsics."
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="COLMAP workspace folder, e.g. COLMAP/colmap_bilinear"
    )
    parser.add_argument(
        "--intrinsics-dir",
        required=True,
        help="Folder containing cameras.txt and image_camera_map.txt"
    )
    parser.add_argument(
        "--colmap-bat",
        default=DEFAULT_COLMAP_BAT,
        help="Path to COLMAP.bat"
    )
    parser.add_argument(
        "--matcher",
        default=DEFAULT_MATCHER,
        choices=["exhaustive", "sequential"],
        help="Matcher type to use"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for SIFT extraction and matching if supported"
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Delete any existing database.db before running"
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    intrinsics_dir = Path(args.intrinsics_dir)
    colmap_bat = Path(args.colmap_bat)

    image_path = dataset_dir / "images"
    sparse_path = dataset_dir / "sparse"
    database_path = dataset_dir / "database.db"

    cameras_path = intrinsics_dir / "cameras.txt"
    image_map_path = intrinsics_dir / "image_camera_map.txt"

    # Check dataset folder structure.
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset_dir}")

    if not image_path.is_dir():
        raise FileNotFoundError(f"Missing images folder: {image_path}")

    if not sparse_path.is_dir():
        raise FileNotFoundError(f"Missing sparse folder: {sparse_path}")

    # Check intrinsics export files.
    if not intrinsics_dir.is_dir():
        raise FileNotFoundError(f"Intrinsics folder does not exist: {intrinsics_dir}")

    if not cameras_path.is_file():
        raise FileNotFoundError(f"Missing cameras.txt: {cameras_path}")

    if not image_map_path.is_file():
        raise FileNotFoundError(f"Missing image_camera_map.txt: {image_map_path}")

    # Check COLMAP launcher.
    if not colmap_bat.is_file():
        raise FileNotFoundError(f"COLMAP.bat not found: {colmap_bat}")

    # Optionally remove old database for a clean run.
    if args.reset_db and database_path.exists():
        print(f"Deleting existing database: {database_path}")
        database_path.unlink()

    # Read exported intrinsics files.
    cameras = parse_cameras_txt(cameras_path)
    grouped_images = parse_image_camera_map(image_map_path)

    # Run feature extraction once per camera group.
    # COLMAP will create database.db automatically if it does not exist.
    for camera_id, image_names in grouped_images.items():
        if camera_id not in cameras:
            raise ValueError(
                f"Camera ID {camera_id} exists in image_camera_map.txt but not in cameras.txt"
            )

        camera = cameras[camera_id]

        print("\n--------------------------------------------------")
        print(f"Camera ID: {camera_id}")
        print(f"Model    : {camera['model']}")
        print(f"Size     : {camera['width']} x {camera['height']}")
        print(f"Params   : {camera['params']}")
        print(f"Images   : {len(image_names)}")
        print("--------------------------------------------------")

        image_list_path = write_image_list(dataset_dir, camera_id, image_names)

        run_feature_extraction(
            colmap_bat=str(colmap_bat),
            database_path=str(database_path),
            image_path=str(image_path),
            image_list_path=image_list_path,
            camera_model=camera["model"],
            camera_params=camera["params"],
            use_gpu=args.use_gpu
        )

    # Run matching across all images in the database.
    run_matching(
        colmap_bat=str(colmap_bat),
        database_path=str(database_path),
        matcher_type=args.matcher,
        use_gpu=args.use_gpu
    )

    # Run sparse reconstruction.
    run_mapper(
        colmap_bat=str(colmap_bat),
        database_path=str(database_path),
        image_path=str(image_path),
        sparse_path=str(sparse_path)
    )

    print("\nDone.")
    print(f"Dataset folder : {dataset_dir}")
    print(f"Intrinsics dir : {intrinsics_dir}")
    print(f"Database path  : {database_path}")
    print(f"Sparse output  : {sparse_path}")


if __name__ == "__main__":
    main()