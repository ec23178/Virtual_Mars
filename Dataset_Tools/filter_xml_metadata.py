# filter_xml_metadata.py
#
# Filters IMG + XML pairs for Mastcam datasets directly from XML metadata and the site/drive code embedded in each filename.
#
# NO PNG conversion is needed before running this script.
# Filtering happens purely on filename and XML content.
# The output is a filtered set of XML + IMG files, ready for CAHVOR parsing and NeRF reconstruction.

import os
import re
import csv
import shutil
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


# PDS4 XML namespace map.
XML_NS = {
    "pds":         "http://pds.nasa.gov/pds4/pds/v1",
    "msn":         "http://pds.nasa.gov/pds4/msn/v1",
    "msn_surface": "http://pds.nasa.gov/pds4/msn_surface/v1",
    "img_surface": "http://pds.nasa.gov/pds4/img_surface/v1",
}

# Default Maria Pass site index.
SITE_MARIA_PASS = 48

# Mastcam filename S-code pattern.
# Captures: SSS (3-digit site) and DDDD (4-digit drive).
S_CODE_PATTERN = re.compile(r"_S(\d{3})(\d{4})MCAM", re.IGNORECASE)

# Sequence ID pattern (MCAM group).
MCAM_PATTERN = re.compile(r"(MCAM\d+)", re.IGNORECASE)


# Helpers: filename parsing
def parse_site_drive(filename):
    # Return (site_int, drive_int) from the S-code in a Mastcam filename stem.
    m = S_CODE_PATTERN.search(filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def parse_sequence_id(filename):
    # Return the MCAM sequence ID string, e.g. 'MCAM04379'.
    m = MCAM_PATTERN.search(filename)
    if m:
        return m.group(1).upper()
    return "UNKNOWN"


def parse_instrument(filename):
    # Return 'LEFT', 'RIGHT', or 'UNKNOWN' from the filename prefix (ML0/MR0).
    upper = filename.upper()
    if upper.startswith("ML"):
        return "LEFT"
    if upper.startswith("MR"):
        return "RIGHT"
    return "UNKNOWN"


# Helper: parse XML label and extract metadata
def parse_xml_metadata(xml_path):
    # Parse a PDS4 XML label and return a metadata dict (or None on failure).
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as err:
        print(f"  WARNING: Could not parse XML {xml_path}: {err}")
        return None

    sol_el = root.find(".//msn:start_sol_number", XML_NS)
    sol    = int(sol_el.text) if sol_el is not None and sol_el.text else None

    seq_el      = root.find(".//msn_surface:sequence_id", XML_NS)
    sequence_id = seq_el.text.strip() if seq_el is not None and seq_el.text else "unknown"

    time_el    = root.find(".//pds:start_date_time", XML_NS)
    start_time = time_el.text.strip() if time_el is not None and time_el.text else "unknown"

    dims = {}
    for axis in root.findall(".//pds:Axis_Array", XML_NS):
        name_el = axis.find("pds:axis_name", XML_NS)
        elem_el = axis.find("pds:elements", XML_NS)
        if name_el is not None and elem_el is not None:
            try:
                dims[name_el.text] = int(elem_el.text)
            except ValueError:
                pass

    return {
        "sol":         sol,
        "sequence_id": sequence_id,
        "start_time":  start_time,
        "lines":       dims.get("Line",   None),
        "samples":     dims.get("Sample", None),
        "bands":       dims.get("Band",   None),
    }



# Helper: parse --drive argument
def parse_drive_arg(drive_str):
    # Parse '876' -> {876} or '876,1146' -> {876, 1146}. None means 'all drives'.
    if drive_str is None:
        return None
    parts = [p.strip() for p in drive_str.split(",") if p.strip()]
    if not parts:
        return None
    try:
        return set(int(p) for p in parts)
    except ValueError:
        raise ValueError(
            f"--drive must be an integer or comma-separated integers, got: {drive_str!r}"
        )


# Helper: apply all filters
def apply_filters(site, drive, instrument, xml_meta, args):
    # Decide whether a file passes all filter criteria.
    # Returns (True, []) if accepted, or (False, [reasons]) if rejected.
    
    reasons = []

    # --- Site filter (primary geographic discriminator) ---
    if site is None:
        reasons.append("no_site_code_in_filename")
    elif site != args.site:
        reasons.append(f"site_{site}_not_{args.site}")

    # --- Drive filter ---
    # Prevents mixing images from different rover parking positions,
    # which would break COLMAP feature matching.
    if args.drives is not None:
        if drive is None:
            reasons.append("no_drive_code_in_filename")
        elif drive not in args.drives:
            drives_label = ",".join(str(d) for d in sorted(args.drives))
            reasons.append(f"drive_{drive}_not_in_[{drives_label}]")

    # --- Instrument filter ---
    if args.instrument != "both":
        want = args.instrument.upper()
        actual = instrument.upper()
        if actual != want:
            reasons.append(f"instrument_{actual}_not_{want}")

    # --- Sol range filter ---
    if xml_meta is not None and xml_meta["sol"] is not None:
        sol = xml_meta["sol"]
        if args.sol_min is not None and sol < args.sol_min:
            reasons.append(f"sol_{sol}_below_min_{args.sol_min}")
        if args.sol_max is not None and sol > args.sol_max:
            reasons.append(f"sol_{sol}_above_max_{args.sol_max}")
    elif args.sol_min is not None or args.sol_max is not None:
        reasons.append("sol_unreadable_in_xml")

    # --- Dimension filter (rejects thumbnails and narrow subframes) ---
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
        reasons.append("xml_unreadable")

    return len(reasons) == 0, reasons


# Helper: copy one XML + IMG pair into output directories
def copy_pair(stem, xml_path, img_dir, output_data_dir, output_img_dir, dry_run):
    """Copy XML and its matching IMG file into the output structure."""
    dest_xml = output_data_dir / (stem + ".xml")
    if not dry_run:
        shutil.copy2(xml_path, dest_xml)

    img_copied = False
    for ext in (".IMG", ".img"):
        img_path = img_dir / (stem + ext)
        if img_path.exists():
            if not dry_run:
                shutil.copy2(img_path, output_img_dir / (stem + ext))
            img_copied = True
            break

    if not img_copied:
        print(f"  WARNING: IMG not found for {stem} (looked in {img_dir})")

    return img_copied


# Helper: write CSV report
def write_csv_report(rows, report_path):
    fieldnames = [
        "file_stem", "instrument", "site", "drive", "sequence_id",
        "sol", "start_time", "lines", "samples", "bands",
        "status", "rejection_reasons",
    ]
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# Main
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Filter Mastcam IMG + XML pairs using XML metadata and filename codes. "
            "Use --drive to lock to one rover parking position (essential for COLMAP/NeRF). "
            "Images from different drives are at separate locations and cannot be "
            "reconstructed together."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required paths
    parser.add_argument("--xml-dir",    required=True,
        help="Folder containing XML label files.")
    parser.add_argument("--img-dir",    required=True,
        help="Folder containing matching IMG files.")
    parser.add_argument("--output-dir", required=True,
        help="Root output folder (created if absent).")

    # Geographic filters
    parser.add_argument("--site", type=int, default=SITE_MARIA_PASS,
        help=f"Rover site index to keep (default: {SITE_MARIA_PASS}).")
    parser.add_argument("--drive", default=None,
        metavar="DRIVE[,DRIVE,...]",
        help=(
            "Drive index, or comma-separated list, to keep. "
            "e.g. --drive 876  or  --drive 876,1146. "
            "CRITICAL for COLMAP: images from different drives are at "
            "different locations and will NOT reconstruct together. "
            "Drive 876 has 79 frames -- the richest single position at Maria Pass. "
            "Default: accept all drives (legacy behaviour)."
        ))

    # Dimension filter
    parser.add_argument("--min-lines",   type=int, default=400,
        help="Minimum image height in lines (default 400). Use 1000 for full-frame only.")
    parser.add_argument("--min-samples", type=int, default=400,
        help="Minimum image width in samples (default 400). Use 1000 for full-frame only.")

    # Sol filter
    parser.add_argument("--sol-min", type=int, default=990,
        help="Minimum sol number (default 990).")
    parser.add_argument("--sol-max", type=int, default=991,
        help="Maximum sol number (default 991).")

    # Instrument filter
    parser.add_argument("--instrument", choices=["left", "right", "both"], default="both",
        help=(
            "Keep only left (ML0), right (MR0), or both cameras. "
            "Default: both. "
            "For COLMAP/NeRF use 'right': the RIGHT camera at drive 876 has "
            "full-frame 1200x1344 images, while LEFT used a narrow 432x1152 subframe."
        ))

    # Dry run
    parser.add_argument("--dry-run", action="store_true",
        help="Report what would be copied without writing any files.")

    args = parser.parse_args()

    # Parse drive filter
    try:
        args.drives = parse_drive_arg(args.drive)
    except ValueError as e:
        parser.error(str(e))

    # Validate paths
    xml_dir    = Path(args.xml_dir)
    img_dir    = Path(args.img_dir)
    output_dir = Path(args.output_dir)

    if not xml_dir.is_dir():
        raise FileNotFoundError(f"XML directory not found: {xml_dir}")
    if not img_dir.is_dir():
        raise FileNotFoundError(f"IMG directory not found: {img_dir}")

    # Create output structure
    output_data_dir = output_dir / "data"
    output_img_dir  = output_dir / "IMG_files"

    if not args.dry_run:
        output_data_dir.mkdir(parents=True, exist_ok=True)
        output_img_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration header
    xml_files = sorted(xml_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in: {xml_dir}")

    drive_label = (
        ",".join(str(d) for d in sorted(args.drives))
        if args.drives else "ALL"
    )

    print(f"\nFound {len(xml_files)} XML files in: {xml_dir}")
    print(
        f"Filters: site={args.site}  drive={drive_label}  "
        f"sol={args.sol_min}-{args.sol_max}  "
        f"min={args.min_lines}x{args.min_samples}  "
        f"instrument={args.instrument}"
    )

    if args.drives is None:
        print(
            "\n  [WARNING] No --drive filter set.\n"
            "  Images from different drive positions are at separate locations\n"
            "  and will NOT reconstruct together in COLMAP.\n"
            "  For COLMAP/NeRF add: --drive 876 --instrument right\n"
        )

    if args.dry_run:
        print("DRY RUN: no files will be written.\n")
    else:
        print(f"Output: {output_dir}\n")

    # Process files
    all_rows   = []
    n_accepted = 0
    n_rejected = 0

    for xml_path in xml_files:
        stem       = xml_path.stem
        site, drive = parse_site_drive(stem)
        instrument  = parse_instrument(stem)
        seq_id_fn   = parse_sequence_id(stem)
        xml_meta    = parse_xml_metadata(xml_path)

        passed, reasons = apply_filters(site, drive, instrument, xml_meta, args)

        row = {
            "file_stem":         stem,
            "instrument":        instrument,
            "site":              site if site is not None else "N/A",
            "drive":             drive if drive is not None else "N/A",
            "sequence_id":       xml_meta["sequence_id"] if xml_meta else seq_id_fn,
            "sol":               xml_meta["sol"]        if xml_meta else "N/A",
            "start_time":        xml_meta["start_time"] if xml_meta else "N/A",
            "lines":             xml_meta["lines"]      if xml_meta else "N/A",
            "samples":           xml_meta["samples"]    if xml_meta else "N/A",
            "bands":             xml_meta["bands"]      if xml_meta else "N/A",
            "status":            "ACCEPT" if passed else "REJECT",
            "rejection_reasons": ";".join(reasons),
        }
        all_rows.append(row)

        if passed:
            n_accepted += 1
            img_ok = copy_pair(
                stem, xml_path, img_dir,
                output_data_dir, output_img_dir, args.dry_run
            )
            print(
                f"  ACCEPT  {stem}"
                f"  sol={row['sol']}  drive={row['drive']}"
                f"  {instrument}  {row['lines']}x{row['samples']}"
                f"  IMG={'ok' if img_ok else 'MISSING'}"
                + (" [DRY RUN]" if args.dry_run else "")
            )
        else:
            n_rejected += 1
            print(f"  REJECT  {stem}  {';'.join(reasons)}")

    # Write report
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv_report(all_rows, output_dir / "filter_report.csv")

    # Summary
    print("\n" + "=" * 60)
    print("FILTER SUMMARY")
    print("=" * 60)
    print(f"  Total XML examined : {len(xml_files)}")
    print(f"  Accepted           : {n_accepted}")
    print(f"  Rejected           : {n_rejected}")

    if not args.dry_run:
        print(f"\n  XML    -> {output_data_dir}")
        print(f"  IMG    -> {output_img_dir}")
        print(f"  Report -> {output_dir / 'filter_report.csv'}")

    # Per-sequence breakdown
    accepted = [r for r in all_rows if r["status"] == "ACCEPT"]
    if accepted:
        seqs = sorted({r["sequence_id"] for r in accepted})
        print(f"\n  MCAM sequences accepted ({len(seqs)}):")
        for seq in seqs:
            members = [r for r in accepted if r["sequence_id"] == seq]
            L = sum(1 for r in members if r["instrument"] == "LEFT")
            R = sum(1 for r in members if r["instrument"] == "RIGHT")
            drives = sorted({str(r["drive"]) for r in members})
            sizes  = sorted({f"{r['lines']}x{r['samples']}" for r in members})
            print(
                f"    {seq.upper():15s}  {len(members):3d} frames  "
                f"L={L} R={R}  drive={','.join(drives)}  size={','.join(sizes)}"
            )

    # COLMAP readiness
    print(f"\n  COLMAP readiness: ", end="")
    if n_accepted >= 20:
        print(f"GOOD ({n_accepted} frames)")
    elif n_accepted >= 10:
        print(f"MARGINAL ({n_accepted} frames -- reconstruction may be sparse)")
    else:
        print(f"LOW ({n_accepted} frames -- likely too few; consider relaxing filters)")

    print()


if __name__ == "__main__":
    main()