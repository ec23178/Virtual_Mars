import argparse
import os

from parse_cahvor import parse_cahvor_folder
from compute_intrinsics import compute_intrinsics_for_dataset
from export_intrinsics import export_all_intrinsics

# Default folders if no command-line arguments are provided.
DEFAULT_DATA_FOLDER   = "data"
DEFAULT_OUTPUT_FOLDER = "output"


def load_keep_list(keep_list_path):
    # Read keep_list.txt produced by select_colmap_images.py.
    #
    # Each line in the file is a PNG filename like:
    #   ML0_0990_0991_MCAM04372_..._D1.png
    #
    # We strip the .png extension to get the file stem so we can
    # match against the stems returned by parse_cahvor_folder().
    stems = set()

    with open(keep_list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Remove extension if present.
            stem = os.path.splitext(line)[0]
            stems.add(stem)

    if not stems:
        raise ValueError(f"keep_list.txt is empty or unreadable: {keep_list_path}")

    return stems


def main():
    # Set up command-line arguments so the script can be reused
    # with different input and output folders without editing code.
    parser = argparse.ArgumentParser(
        description="Parse CAHVOR XML files, compute intrinsics, and export results."
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_DATA_FOLDER,
        help="Folder containing XML metadata files."
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_FOLDER,
        help="Folder where exported intrinsics files will be saved."
    )
    parser.add_argument(
        "--keep-list",
        default=None,
        help=(
            "Optional path to keep_list.txt produced by select_colmap_images.py. "
            "When provided, only XML files whose stem appears in the keep list "
            "are processed.  This restricts the intrinsics export to exactly the "
            "images that were selected for COLMAP, ensuring cameras.txt and "
            "image_camera_map.txt contain no extra entries."
        )
    )
    args = parser.parse_args()

    # Parse XML files from the chosen folder.
    print("Parsing XML files...")
    parsed = parse_cahvor_folder(args.input_dir)
    print(f"  Found {len(parsed)} XML files.")

    # If a keep list was supplied, filter parsed items to only those images.
    if args.keep_list is not None:
        keep_stems = load_keep_list(args.keep_list)
        before = len(parsed)
        parsed = [item for item in parsed if item["file_stem"] in keep_stems]
        after  = len(parsed)

        print(
            f"  Filtered by keep list: {before} -> {after} items "
            f"({before - after} excluded)."
        )

        # Warn if the keep list references stems not found in the XML folder.
        found_stems = {item["file_stem"] for item in parsed}
        missing     = keep_stems - found_stems
        if missing:
            print(
                f"  WARNING: {len(missing)} stem(s) in keep_list.txt have no "
                f"matching XML in --input-dir and will be skipped:"
            )
            for stem in sorted(missing):
                print(f"    {stem}")

    if not parsed:
        raise ValueError(
            "No XML items remain after filtering. "
            "Check that --input-dir contains XMLs matching the keep list."
        )

    # Compute intrinsics from the parsed CAHVOR values.
    print("Computing intrinsics...")
    results = compute_intrinsics_for_dataset(parsed)

    # Export the intrinsics files into the chosen output folder.
    print("Exporting intrinsics...")
    paths = export_all_intrinsics(results, args.output_dir)

    print("Done! Output files saved.")
    for key, value in paths.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
