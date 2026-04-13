# filter_xml_metadata.py
#
# Filters IMG + XML pairs for the Maria Pass dataset directly from XML metadata
# and the site/drive code embedded in each filename.
#
# NO PNG conversion is needed before running this script.
# Filtering happens purely on filename and XML content, so the full 691-pair
# dataset can be screened in seconds rather than waiting on vic2png.
#
# -----------------------------------------------------------------------
# How the site number is used as the primary geographic filter
# -----------------------------------------------------------------------
# Every PDS4 Mastcam filename follows this pattern:
#
#   ML0_<SCLK>EDR_S<SITE><DRIVE>MCAM<SEQ>D1.IMG
#
# where:
#   <SITE>  = 3-digit zero-padded rover site index   (e.g. 048 = site 48)
#   <DRIVE> = 4-digit zero-padded drive index         (e.g. 0458)
#
# All confirmed Maria Pass frames in the 2015-05-20 window sit at site 48:
#   S0480458, S0480680, S0480876  ->  site = 48
#
# Images from other sol-990 sequences at different sites, and all images
# taken on sols 991-995 after Curiosity drove away, will have a different
# site number and are automatically excluded.
#
# -----------------------------------------------------------------------
# Secondary filters (from XML metadata)
# -----------------------------------------------------------------------
#   - Image dimensions: reject tiny thumbnails (default: min 400 lines x 400 samples)
#   - Sol number: optional tight range (default: 990-991)
#   - Instrument: optionally restrict to ML0, MR0, or both (default: both)
#
# -----------------------------------------------------------------------
# Output structure
# -----------------------------------------------------------------------
# The accepted pairs are copied into an output folder that mirrors the
# existing project layout so they slot straight into the pipeline:
#
#   <output_dir>/
#     data/           <- accepted XML files
#     IMG_files/      <- accepted IMG files
#     filter_report.csv   <- full record of every file examined
#
# -----------------------------------------------------------------------
# Usage
# -----------------------------------------------------------------------
#   python filter_xml_metadata.py \
#       --xml-dir  /path/to/raw/data \
#       --img-dir  /path/to/raw/IMG_files \
#       --output-dir /path/to/maria_pass_filtered \
#       [--site 48] \
#       [--min-lines 400] [--min-samples 400] \
#       [--sol-min 990] [--sol-max 991] \
#       [--instrument both] \
#       [--dry-run]

import os
import re
import csv
import shutil
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


# -----------------------------------------------------------------------
# PDS4 XML namespace map.
# These prefixes match those used in every Mastcam PDS4 label.
# -----------------------------------------------------------------------
XML_NS = {
    "pds":         "http://pds.nasa.gov/pds4/pds/v1",
    "msn":         "http://pds.nasa.gov/pds4/msn/v1",
    "msn_surface": "http://pds.nasa.gov/pds4/msn_surface/v1",
    "img_surface": "http://pds.nasa.gov/pds4/img_surface/v1",
}

# -----------------------------------------------------------------------
# Maria Pass rover site index.
# All confirmed frames from the May 2015 Maria Pass stop carry site 48
# in their filename code (e.g. S0480458, S0480680, S0480876).
# -----------------------------------------------------------------------
SITE_MARIA_PASS = 48

# -----------------------------------------------------------------------
# Filename pattern for the S-code.
#
# Mastcam filenames contain:   ...EDR_S<SSS><DDDD>MCAM...
#   SSS  = 3-digit site index
#   DDDD = 4-digit drive index
#
# Example: ML0_485376129EDR_S0480458MCAM04372D1.IMG
#           -> site = 048 = 48,  drive = 0458 = 458
# -----------------------------------------------------------------------
S_CODE_PATTERN = re.compile(r"_S(\d{3})(\d{4})MCAM", re.IGNORECASE)

# -----------------------------------------------------------------------
# Sequence ID pattern (MCAM group in the filename).
# Used for the report; not used as a hard filter because site + sol covers it.
# -----------------------------------------------------------------------
MCAM_PATTERN = re.compile(r"(MCAM\d+)", re.IGNORECASE)


# -----------------------------------------------------------------------
# Helper: extract site and drive from a Mastcam filename stem.
#
# Returns (site_int, drive_int) or (None, None) if the pattern is absent.
# -----------------------------------------------------------------------
def parse_site_drive(filename):
    m = S_CODE_PATTERN.search(filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


# -----------------------------------------------------------------------
# Helper: extract the MCAM sequence group from a filename.
#
# Example:
#   ML0_485376129EDR_S0480458MCAM04372D1.IMG  ->  MCAM04372
# -----------------------------------------------------------------------
def parse_sequence_id(filename):
    m = MCAM_PATTERN.search(filename)
    if m:
        return m.group(1).upper()
    return "UNKNOWN"


# -----------------------------------------------------------------------
# Helper: detect camera side from filename prefix.
#
# ML0  -> LEFT
# MR0  -> RIGHT
# -----------------------------------------------------------------------
def parse_instrument(filename):
    upper = filename.upper()
    if upper.startswith("ML"):
        return "LEFT"
    if upper.startswith("MR"):
        return "RIGHT"
    return "UNKNOWN"


# -----------------------------------------------------------------------
# Helper: parse the XML label and return a metadata dictionary.
#
# Fields extracted:
#   sol          - integer sol number
#   sequence_id  - msn_surface sequence_id string (e.g. mcam04372)
#   start_time   - ISO start time string
#   lines        - image height in pixels
#   samples      - image width in pixels
#   bands        - number of colour bands (3 for Bayer EDR)
#
# Returns None if the XML cannot be parsed.
# -----------------------------------------------------------------------
def parse_xml_metadata(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as err:
        print(f"  WARNING: Could not parse XML {xml_path}: {err}")
        return None

    # Sol number.
    sol_el = root.find(".//msn:start_sol_number", XML_NS)
    sol = int(sol_el.text) if sol_el is not None and sol_el.text else None

    # Sequence ID (e.g. mcam04372).
    seq_el = root.find(".//msn_surface:sequence_id", XML_NS)
    sequence_id = seq_el.text.strip() if seq_el is not None and seq_el.text else "unknown"

    # Start time.
    time_el = root.find(".//pds:start_date_time", XML_NS)
    start_time = time_el.text.strip() if time_el is not None and time_el.text else "unknown"

    # Image dimensions from Axis_Array elements.
    # Each axis has an axis_name (Band / Line / Sample) and an elements count.
    dims = {}
    for axis in root.findall(".//pds:Axis_Array", XML_NS):
        name_el = axis.find("pds:axis_name", XML_NS)
        elem_el = axis.find("pds:elements", XML_NS)
        if name_el is not None and elem_el is not None:
            try:
                dims[name_el.text] = int(elem_el.text)
            except ValueError:
                pass

    lines   = dims.get("Line",   None)
    samples = dims.get("Sample", None)
    bands   = dims.get("Band",   None)

    return {
        "sol":         sol,
        "sequence_id": sequence_id,
        "start_time":  start_time,
        "lines":       lines,
        "samples":     samples,
        "bands":       bands,
    }


# -----------------------------------------------------------------------
# Helper: decide whether a file passes all filter criteria.
#
# Returns (True, []) if the file should be kept, or
#         (False, [list of rejection reasons]) if it should be rejected.
#
# Reasons are short strings suitable for logging and the CSV report.
# -----------------------------------------------------------------------
def apply_filters(site, drive, xml_meta, args):
    reasons = []

    # --- Site filter (primary geographic discriminator) ---
    if site is None:
        reasons.append("no_site_code_in_filename")
    elif site != args.site:
        reasons.append(f"site_{site}_not_{args.site}")

    # --- Instrument filter ---
    if args.instrument != "both":
        instrument_want = args.instrument.upper()
        if xml_meta and xml_meta.get("instrument", "").upper() != instrument_want:
            reasons.append(f"instrument_not_{args.instrument.lower()}")

    # --- Sol range filter (only applied if the user set bounds) ---
    if xml_meta is not None and xml_meta["sol"] is not None:
        sol = xml_meta["sol"]
        if args.sol_min is not None and sol < args.sol_min:
            reasons.append(f"sol_{sol}_below_min_{args.sol_min}")
        if args.sol_max is not None and sol > args.sol_max:
            reasons.append(f"sol_{sol}_above_max_{args.sol_max}")
    elif args.sol_min is not None or args.sol_max is not None:
        # XML missing sol and a sol filter was requested.
        reasons.append("sol_unreadable_in_xml")

    # --- Dimension filter (rejects thumbnails and partial reads) ---
    if xml_meta is not None:
        lines   = xml_meta.get("lines")
        samples = xml_meta.get("samples")

        if lines is None or samples is None:
            reasons.append("dimensions_missing_in_xml")
        else:
            if lines < args.min_lines:
                reasons.append(f"lines_{lines}_below_min_{args.min_lines}")
            if samples < args.min_samples:
                reasons.append(f"samples_{samples}_below_min_{args.min_samples}")
    else:
        # Could not read the XML at all.
        reasons.append("xml_unreadable")

    passed = len(reasons) == 0
    return passed, reasons


# -----------------------------------------------------------------------
# Helper: copy one XML + IMG pair into the output directory structure.
#
# Output layout:
#   output_dir/data/      <- XML files
#   output_dir/IMG_files/ <- IMG files
#
# IMG file is matched by replacing the XML extension with .IMG and
# looking in img_dir.  A warning is printed if the IMG is absent.
# -----------------------------------------------------------------------
def copy_pair(stem, xml_path, img_dir, output_data_dir, output_img_dir, dry_run):
    # Copy the XML label.
    dest_xml = output_data_dir / (stem + ".xml")
    if not dry_run:
        shutil.copy2(xml_path, dest_xml)

    # Copy the matching IMG file.
    img_path = img_dir / (stem + ".IMG")
    img_copied = False

    if img_path.exists():
        dest_img = output_img_dir / (stem + ".IMG")
        if not dry_run:
            shutil.copy2(img_path, dest_img)
        img_copied = True
    else:
        # Try case-insensitive fallback (.img lowercase).
        img_path_lower = img_dir / (stem + ".img")
        if img_path_lower.exists():
            dest_img = output_img_dir / (stem + ".img")
            if not dry_run:
                shutil.copy2(img_path_lower, dest_img)
            img_copied = True
        else:
            print(f"  WARNING: IMG not found for {stem} (looked in {img_dir})")

    return img_copied


# -----------------------------------------------------------------------
# Helper: write the full CSV report of every file examined.
# -----------------------------------------------------------------------
def write_csv_report(rows, report_path):
    fieldnames = [
        "file_stem",
        "instrument",
        "site",
        "drive",
        "sequence_id",
        "sol",
        "start_time",
        "lines",
        "samples",
        "bands",
        "status",
        "rejection_reasons",
    ]

    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Filter Mastcam IMG + XML pairs for the Maria Pass dataset using "
            "XML metadata and the site code in the filename. "
            "No PNG conversion is needed before running this script."
        )
    )

    # --- Required paths ---
    parser.add_argument(
        "--xml-dir",
        required=True,
        help="Folder containing the XML label files (e.g. datasets/raw/data).",
    )
    parser.add_argument(
        "--img-dir",
        required=True,
        help="Folder containing the matching IMG files (e.g. datasets/raw/IMG_files).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Root output folder. Accepted pairs are copied here under "
            "data/ and IMG_files/ sub-folders. Created if it does not exist."
        ),
    )

    # --- Geographic filter ---
    parser.add_argument(
        "--site",
        type=int,
        default=SITE_MARIA_PASS,
        help=(
            f"Rover site index to keep (default: {SITE_MARIA_PASS}). "
            "Maria Pass images all carry site 48 in their filename S-code."
        ),
    )

    # --- Dimension filter ---
    parser.add_argument(
        "--min-lines",
        type=int,
        default=400,
        help=(
            "Minimum number of image lines (height) to accept. "
            "Default 400 rejects thumbnails (e.g. 320-line frames) while keeping "
            "all full-frame and most subframe science images."
        ),
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=400,
        help=(
            "Minimum number of image samples (width) to accept. "
            "Default 400 mirrors the --min-lines logic."
        ),
    )

    # --- Sol filter ---
    parser.add_argument(
        "--sol-min",
        type=int,
        default=990,
        help=(
            "Minimum sol number to accept (default: 990). "
            "Curiosity arrived at Maria Pass on sol 990."
        ),
    )
    parser.add_argument(
        "--sol-max",
        type=int,
        default=991,
        help=(
            "Maximum sol number to accept (default: 991). "
            "Curiosity departed Maria Pass after sol 991."
        ),
    )

    # --- Instrument filter ---
    parser.add_argument(
        "--instrument",
        choices=["left", "right", "both"],
        default="both",
        help=(
            "Keep only left (ML0), right (MR0), or both cameras. "
            "Default: both. COLMAP and NeRFStudio benefit from stereo pairs."
        ),
    )

    # --- Dry run ---
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print what would be copied without actually writing any files. "
            "Useful for checking the filter results before committing to disk."
        ),
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Resolve and validate input paths.
    # -----------------------------------------------------------------------
    xml_dir = Path(args.xml_dir)
    img_dir = Path(args.img_dir)
    output_dir = Path(args.output_dir)

    if not xml_dir.is_dir():
        raise FileNotFoundError(f"XML directory not found: {xml_dir}")
    if not img_dir.is_dir():
        raise FileNotFoundError(f"IMG directory not found: {img_dir}")

    # -----------------------------------------------------------------------
    # Create the output folder structure (unless dry run).
    # -----------------------------------------------------------------------
    output_data_dir = output_dir / "data"
    output_img_dir  = output_dir / "IMG_files"

    if not args.dry_run:
        output_data_dir.mkdir(parents=True, exist_ok=True)
        output_img_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Collect all XML files in sorted order so processing is reproducible.
    # -----------------------------------------------------------------------
    xml_files = sorted(xml_dir.glob("*.xml"))

    if not xml_files:
        raise FileNotFoundError(f"No XML files found in: {xml_dir}")

    print(f"\nFound {len(xml_files)} XML files in: {xml_dir}")
    print(f"Filtering for site={args.site}, sol={args.sol_min}-{args.sol_max}, "
          f"min_dims={args.min_lines}x{args.min_samples}, instrument={args.instrument}")
    if args.dry_run:
        print("DRY RUN: no files will be written.\n")
    else:
        print(f"Output root: {output_dir}\n")

    # -----------------------------------------------------------------------
    # Process each XML file.
    # -----------------------------------------------------------------------
    all_rows = []
    n_accepted = 0
    n_rejected = 0

    for xml_path in xml_files:
        stem = xml_path.stem

        # Extract site/drive from filename.
        site, drive = parse_site_drive(stem)

        # Extract camera side from filename prefix.
        instrument = parse_instrument(stem)

        # Extract MCAM sequence ID from filename.
        sequence_id_fname = parse_sequence_id(stem)

        # Parse the XML label for sol, dims, and timing.
        xml_meta = parse_xml_metadata(xml_path)

        # Merge the filename-derived sequence_id into xml_meta for the filter
        # (the filter uses instrument from xml_meta if set).
        if xml_meta is not None:
            xml_meta["instrument"] = instrument

        # Apply all filters.
        passed, rejection_reasons = apply_filters(site, drive, xml_meta, args)

        # Build the report row.
        row = {
            "file_stem":          stem,
            "instrument":         instrument,
            "site":               site if site is not None else "N/A",
            "drive":              drive if drive is not None else "N/A",
            "sequence_id":        xml_meta["sequence_id"] if xml_meta else sequence_id_fname,
            "sol":                xml_meta["sol"] if xml_meta else "N/A",
            "start_time":         xml_meta["start_time"] if xml_meta else "N/A",
            "lines":              xml_meta["lines"] if xml_meta else "N/A",
            "samples":            xml_meta["samples"] if xml_meta else "N/A",
            "bands":              xml_meta["bands"] if xml_meta else "N/A",
            "status":             "ACCEPT" if passed else "REJECT",
            "rejection_reasons":  ";".join(rejection_reasons),
        }
        all_rows.append(row)

        if passed:
            n_accepted += 1

            # Copy the XML + IMG pair.
            img_copied = copy_pair(
                stem, xml_path, img_dir,
                output_data_dir, output_img_dir,
                args.dry_run
            )

            print(
                f"  ACCEPT  {stem}"
                f"  | sol={row['sol']}"
                f"  | seq={row['sequence_id']}"
                f"  | dims={row['lines']}x{row['samples']}"
                f"  | IMG={'ok' if img_copied else 'MISSING'}"
                + (" [DRY RUN]" if args.dry_run else "")
            )

        else:
            n_rejected += 1
            print(
                f"  REJECT  {stem}"
                f"  | reasons: {';'.join(rejection_reasons)}"
            )

    # -----------------------------------------------------------------------
    # Write the CSV report (skipped in dry-run mode).
    # -----------------------------------------------------------------------
    report_path = output_dir / "filter_report.csv"

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv_report(all_rows, report_path)

    # -----------------------------------------------------------------------
    # Print summary.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FILTER SUMMARY")
    print("=" * 60)
    print(f"  Total XML files examined : {len(xml_files)}")
    print(f"  Accepted (site {args.site}, sol {args.sol_min}-{args.sol_max}) : {n_accepted}")
    print(f"  Rejected                 : {n_rejected}")

    if not args.dry_run:
        print(f"\n  Accepted XML files -> {output_data_dir}")
        print(f"  Accepted IMG files -> {output_img_dir}")
        print(f"  Full report CSV    -> {report_path}")

    # -----------------------------------------------------------------------
    # Print the accepted sequence groups so you can cross-check with the
    # Atlas website before feeding the files into the pipeline.
    # -----------------------------------------------------------------------
    accepted_rows = [r for r in all_rows if r["status"] == "ACCEPT"]

    if accepted_rows:
        seq_groups = sorted({r["sequence_id"] for r in accepted_rows})
        print(f"\n  Accepted MCAM sequence groups ({len(seq_groups)}):")
        for seq in seq_groups:
            members = [r for r in accepted_rows if r["sequence_id"] == seq]
            left  = sum(1 for r in members if r["instrument"] == "LEFT")
            right = sum(1 for r in members if r["instrument"] == "RIGHT")
            print(f"    {seq.upper()}  ->  {len(members)} frames  (L={left}, R={right})")

    print()


if __name__ == "__main__":
    main()
