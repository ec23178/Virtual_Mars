import argparse

from parse_cahvor import parse_cahvor_folder
from compute_intrinsics import compute_intrinsics_for_dataset
from export_intrinsics import export_all_intrinsics

# Default folders if no command-line arguments are provided.
DEFAULT_DATA_FOLDER = "data"
DEFAULT_OUTPUT_FOLDER = "output"


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
    args = parser.parse_args()

    # Parse XML files from the chosen folder.
    print("Parsing XML files...")
    parsed = parse_cahvor_folder(args.input_dir)

    # Compute intrinsics from the parsed CAHVOR values.
    print("Computing intrinsics...")
    results = compute_intrinsics_for_dataset(parsed)

    # Export the intrinsics files into the chosen output folder.
    print("Exporting intrinsics...")
    paths = export_all_intrinsics(results, args.output_dir)

    print("Done! Output files saved.")
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()