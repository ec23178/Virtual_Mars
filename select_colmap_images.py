# select_colmap_images.py
#
# Second-pass filter that selects the best demosaiced images for COLMAP input.
# Run this AFTER vic2png and demosaic_batch.py, on ONE chosen demosaicing
# folder (bilinear-images OR malvar-images -- not both mixed).
#
# -----------------------------------------------------------------------
# Why this script exists
# -----------------------------------------------------------------------
# After filter_xml_metadata.py you have ~94 pairs across 11 MCAM sequences.
# The problem is that one sequence (MCAM04379) alone has 48 frames taken
# in quick succession -- these are near-duplicates that:
#   - Slow down COLMAP feature matching significantly
#   - Can produce degenerate solutions from redundant views
#   - Add very little new scene coverage
#
# This script tackles that with two independent passes:
#
#   Pass 1 -- Quality filter
#       Rejects images that are blurry, low-texture, over/under-exposed,
#       or sky/ground dominated.  These would produce poor COLMAP tracks.
#
#   Pass 2 -- Sequence thinning
#       Within each MCAM group, if more images survive Pass 1 than
#       --max-per-sequence allows, only the top-scoring ones are kept.
#       This keeps the scene coverage while cutting near-duplicates.
#
# -----------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------
#   <output_dir>/
#     images/                 <- images selected for COLMAP (copies)
#     selection_report.csv    <- full record: score, status, reasons
#     keep_list.txt           <- just the filenames that were kept
#
# -----------------------------------------------------------------------
# Usage (PowerShell one-liners)
# -----------------------------------------------------------------------
# Bilinear demosaiced images, keep at most 6 images per MCAM sequence:
#
#   python select_colmap_images.py --input-dir datasets/maria_pass_filtered/demosaiced_images/bilinear-images --output-dir datasets/maria_pass_filtered/colmap_input_bilinear --max-per-sequence 6
#
# Malvar demosaiced images:
#
#   python select_colmap_images.py --input-dir datasets/maria_pass_filtered/demosaiced_images/malvar-images --output-dir datasets/maria_pass_filtered/colmap_input_malvar --max-per-sequence 6
#
# Dry run first (no files written):
#
#   python select_colmap_images.py --input-dir datasets/maria_pass_filtered/demosaiced_images/bilinear-images --output-dir datasets/maria_pass_filtered/colmap_input_bilinear --max-per-sequence 6 --dry-run

import os
import re
import csv
import shutil
import argparse
import numpy as np
from pathlib import Path
from PIL import Image


# -----------------------------------------------------------------------
# Sequence ID extraction.
#
# Mastcam filenames contain MCAMXXXXX e.g. MCAM04379.
# All frames sharing an MCAM ID were taken in one commanded sequence,
# usually pointing at the same scene.
# -----------------------------------------------------------------------
MCAM_PATTERN = re.compile(r"(MCAM\d+)", re.IGNORECASE)


def get_sequence_id(filename):
    m = MCAM_PATTERN.search(filename.upper())
    if m:
        return m.group(1)
    return "UNKNOWN"


# -----------------------------------------------------------------------
# Image loading.
#
# The demosaiced PNGs from demosaic_batch.py are RGB (3-channel).
# We keep the RGB array for colour stats and convert to grayscale for
# the spatial metrics (sharpness, texture) to stay consistent with the
# way filter_maria_pass_images.py works.
# -----------------------------------------------------------------------
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img, dtype=np.float32)
    gray = np.array(img.convert("L"), dtype=np.float32)
    height, width = gray.shape
    return rgb, gray, width, height


# -----------------------------------------------------------------------
# Sharpness: Laplacian variance.
#
# Higher value = more fine detail = better for COLMAP feature detection.
# Very blurry images produce near-zero variance.
# -----------------------------------------------------------------------
def compute_sharpness(gray):
    center = gray[1:-1, 1:-1]
    up     = gray[:-2, 1:-1]
    down   = gray[2:,  1:-1]
    left   = gray[1:-1, :-2]
    right  = gray[1:-1, 2:]
    lap = (up + down + left + right) - (4.0 * center)
    return float(np.var(lap))


# -----------------------------------------------------------------------
# Texture: mean absolute pixel difference in x and y.
#
# Higher value = more edges = more feature points for COLMAP to match.
# Sky and blank sand produce very low texture.
# -----------------------------------------------------------------------
def compute_texture(gray):
    dx = np.abs(gray[:, 1:] - gray[:, :-1])
    dy = np.abs(gray[1:, :] - gray[:-1, :])
    return float((np.mean(dx) + np.mean(dy)) / 2.0)


# -----------------------------------------------------------------------
# Brightness statistics.
# -----------------------------------------------------------------------
def compute_brightness(gray):
    mean_val      = float(np.mean(gray))
    std_val       = float(np.std(gray))
    dark_fraction  = float(np.mean(gray < 15))
    bright_fraction = float(np.mean(gray > 240))
    return mean_val, std_val, dark_fraction, bright_fraction


# -----------------------------------------------------------------------
# Sky dominance check.
#
# Checks the top 20% of the frame.
# Bright + flat top strip likely means sky or haze -- not useful for
# COLMAP since sky has no stable feature points.
# -----------------------------------------------------------------------
def has_sky_dominance(gray):
    h = gray.shape[0]
    top = gray[:max(1, int(h * 0.20)), :]
    top_mean    = float(np.mean(top))
    top_texture = compute_texture(top)
    if top_mean > 170 and top_texture < 8:
        return True
    if top_mean > 145 and top_texture < 6:
        return True
    return False


# -----------------------------------------------------------------------
# Ground-only check.
#
# If both halves of the image look similar and heavily textured with
# no brightness transition, the frame may be close-up ground/rock with
# little wider scene context.  These are still useful for COLMAP but
# we flag them so they can be reviewed if the reconstruction looks patchy.
# -----------------------------------------------------------------------
def has_ground_only(gray):
    h = gray.shape[0]
    top    = gray[:h // 2, :]
    bottom = gray[h // 2:, :]
    mean_diff      = abs(float(np.mean(top)) - float(np.mean(bottom)))
    top_texture    = compute_texture(top)
    bottom_texture = compute_texture(bottom)
    return mean_diff < 18 and top_texture > 10 and bottom_texture > 10


# -----------------------------------------------------------------------
# Score one image.
#
# Returns a dict with:
#   score      - composite float (higher = better for COLMAP)
#   hard_fail  - True if this image should be rejected outright
#   reasons    - list of short string tags explaining the score
#   metrics    - all raw metric values
# -----------------------------------------------------------------------
def score_image(image_path):
    rgb, gray, width, height = load_image(image_path)

    sharpness                              = compute_sharpness(gray)
    texture                                = compute_texture(gray)
    mean_val, std_val, dark_frac, bright_frac = compute_brightness(gray)
    sky_dominant                           = has_sky_dominance(gray)
    ground_only                            = has_ground_only(gray)

    reasons   = []
    score     = 0.0
    hard_fail = False

    # --- Resolution ---
    # Images smaller than 400x400 are thumbnails and should already have
    # been caught by filter_xml_metadata.py, but we guard here too.
    if width < 400 or height < 400:
        reasons.append("too_small")
        hard_fail = True

    # --- Sharpness ---
    if sharpness < 8:
        reasons.append("very_blurry")
        hard_fail = True
    elif sharpness < 25:
        reasons.append("soft")
        score += 0.5
    else:
        score += 2.0

    # --- Texture ---
    # Low texture means few SIFT/COLMAP keypoints -- poor matching.
    if texture < 4:
        reasons.append("very_low_texture")
        hard_fail = True
    elif texture < 8:
        reasons.append("low_texture")
        score += 0.5
    else:
        score += 2.0

    # --- Brightness ---
    if dark_frac > 0.40:
        reasons.append("too_dark")
        score -= 1.5
    if bright_frac > 0.35:
        reasons.append("too_bright")
        score -= 1.5
    if std_val < 20:
        reasons.append("low_contrast")
        score -= 0.5

    # --- Scene heuristics (lower score, do not hard-fail) ---
    if sky_dominant:
        reasons.append("sky_dominant")
        score -= 1.0
    if ground_only:
        reasons.append("ground_only")
        score -= 0.5

    # Bonus for full-frame size (1200x1344 is the Mastcam full frame).
    if width >= 1200 and height >= 1100:
        score += 1.5
    elif width >= 900:
        score += 0.5

    return {
        "score":         score,
        "hard_fail":     hard_fail,
        "reasons":       reasons,
        "sharpness":     sharpness,
        "texture":       texture,
        "brightness_mean": mean_val,
        "brightness_std":  std_val,
        "dark_fraction":   dark_frac,
        "bright_fraction": bright_frac,
        "sky_dominant":    sky_dominant,
        "ground_only":     ground_only,
        "width":           width,
        "height":          height,
    }


# -----------------------------------------------------------------------
# Select images for COLMAP.
#
# Two passes:
#   1. Hard quality filter -- removes outright bad images.
#   2. Per-sequence thinning -- within each MCAM group, keep only the
#      top-scoring images up to max_per_sequence.
#
# Returns a list of (filename, result_dict) tuples sorted by sequence then score.
# -----------------------------------------------------------------------
def select_images(scored_rows, max_per_sequence):
    # Group by sequence ID.
    by_sequence = {}
    for row in scored_rows:
        seq = row["sequence_id"]
        if seq not in by_sequence:
            by_sequence[seq] = []
        by_sequence[seq].append(row)

    selected_stems = set()

    for seq, rows in by_sequence.items():
        # Pass 1: drop hard failures.
        passed = [r for r in rows if not r["hard_fail"]]
        failed = [r for r in rows if r["hard_fail"]]

        # Mark failed images.
        for r in failed:
            r["final_status"] = "REJECT_QUALITY"

        if not passed:
            continue

        # Pass 2: sort surviving images by score descending.
        passed.sort(key=lambda r: r["score"], reverse=True)

        # Keep the top N.
        kept    = passed[:max_per_sequence]
        thinned = passed[max_per_sequence:]

        for r in kept:
            r["final_status"] = "KEEP"
            selected_stems.add(r["file_stem"])

        for r in thinned:
            r["final_status"] = "REJECT_THINNED"

    return selected_stems


# -----------------------------------------------------------------------
# Write CSV report.
# -----------------------------------------------------------------------
def write_report(rows, report_path):
    fieldnames = [
        "file_stem",
        "sequence_id",
        "final_status",
        "score",
        "width",
        "height",
        "sharpness",
        "texture",
        "brightness_mean",
        "brightness_std",
        "dark_fraction",
        "bright_fraction",
        "sky_dominant",
        "ground_only",
        "reasons",
    ]
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "file_stem":       row["file_stem"],
                "sequence_id":     row["sequence_id"],
                "final_status":    row.get("final_status", "UNKNOWN"),
                "score":           f"{row['score']:.3f}",
                "width":           row["width"],
                "height":          row["height"],
                "sharpness":       f"{row['sharpness']:.2f}",
                "texture":         f"{row['texture']:.2f}",
                "brightness_mean": f"{row['brightness_mean']:.1f}",
                "brightness_std":  f"{row['brightness_std']:.1f}",
                "dark_fraction":   f"{row['dark_fraction']:.3f}",
                "bright_fraction": f"{row['bright_fraction']:.3f}",
                "sky_dominant":    row["sky_dominant"],
                "ground_only":     row["ground_only"],
                "reasons":         ";".join(row["reasons"]),
            })


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Select the best demosaiced images for COLMAP from one demosaicing "
            "folder (bilinear-images OR malvar-images). Applies a quality filter "
            "then thins each MCAM sequence to --max-per-sequence images."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help=(
            "Folder of demosaiced PNG images to score. "
            "Use bilinear-images OR malvar-images, not both."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Output folder. Selected images are copied into a sub-folder "
            "called images/ inside this directory."
        ),
    )
    parser.add_argument(
        "--max-per-sequence",
        type=int,
        default=6,
        help=(
            "Maximum number of images to keep from each MCAM sequence after "
            "quality filtering. The top-scoring images are chosen. "
            "Default 6 balances scene coverage against COLMAP compute cost. "
            "Lower this (e.g. 4) for faster COLMAP runs on a weak machine."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selection without writing any files.",
    )
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    png_files = sorted(input_dir.glob("*.png"))
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in: {input_dir}")

    # -----------------------------------------------------------------------
    # Score every image.
    # -----------------------------------------------------------------------
    print(f"\nScoring {len(png_files)} images in: {input_dir}")
    print(f"Max per MCAM sequence: {args.max_per_sequence}")
    if args.dry_run:
        print("DRY RUN: no files will be written.\n")
    else:
        print()

    all_rows = []
    for png_path in png_files:
        stem        = png_path.stem
        sequence_id = get_sequence_id(stem)

        result = score_image(png_path)

        row = {
            "file_stem":   stem,
            "sequence_id": sequence_id,
            "png_path":    png_path,
            "final_status": None,
            **result,
        }
        all_rows.append(row)

    # -----------------------------------------------------------------------
    # Select images (quality filter + thinning).
    # -----------------------------------------------------------------------
    selected_stems = select_images(all_rows, args.max_per_sequence)

    # -----------------------------------------------------------------------
    # Create output folders and copy selected images.
    # -----------------------------------------------------------------------
    output_images_dir = output_dir / "images"

    if not args.dry_run:
        output_images_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Print results grouped by sequence.
    # -----------------------------------------------------------------------
    # Sort rows: sequence_id first, then score descending.
    all_rows.sort(key=lambda r: (r["sequence_id"], -r["score"]))

    current_seq = None
    for row in all_rows:
        if row["sequence_id"] != current_seq:
            current_seq = row["sequence_id"]
            seq_rows  = [r for r in all_rows if r["sequence_id"] == current_seq]
            n_kept    = sum(1 for r in seq_rows if r.get("final_status") == "KEEP")
            print(f"\n  [{current_seq}]  {len(seq_rows)} frames -> {n_kept} kept")

        status   = row.get("final_status", "UNKNOWN")
        reasons  = row["reasons"]
        tag_str  = f"  | reasons: {';'.join(reasons)}" if reasons else ""

        print(
            f"    {status:<20}  {row['file_stem']}"
            f"  score={row['score']:.2f}"
            f"  sharp={row['sharpness']:.0f}"
            f"  tex={row['texture']:.1f}"
            f"{tag_str}"
        )

        if status == "KEEP" and not args.dry_run:
            shutil.copy2(row["png_path"], output_images_dir / (row["file_stem"] + ".png"))

    # -----------------------------------------------------------------------
    # Write report and keep list.
    # -----------------------------------------------------------------------
    n_kept    = sum(1 for r in all_rows if r.get("final_status") == "KEEP")
    n_quality = sum(1 for r in all_rows if r.get("final_status") == "REJECT_QUALITY")
    n_thinned = sum(1 for r in all_rows if r.get("final_status") == "REJECT_THINNED")

    if not args.dry_run:
        report_path   = output_dir / "selection_report.csv"
        keep_list_path = output_dir / "keep_list.txt"

        write_report(all_rows, report_path)

        with open(keep_list_path, "w", encoding="utf-8") as f:
            for row in all_rows:
                if row.get("final_status") == "KEEP":
                    f.write(row["file_stem"] + ".png\n")

    # -----------------------------------------------------------------------
    # Summary.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SELECTION SUMMARY")
    print("=" * 60)
    print(f"  Total images scored    : {len(all_rows)}")
    print(f"  KEEP (for COLMAP)      : {n_kept}")
    print(f"  REJECT quality filter  : {n_quality}")
    print(f"  REJECT thinned (dupes) : {n_thinned}")

    if not args.dry_run:
        print(f"\n  Selected images  -> {output_images_dir}")
        print(f"  Selection report -> {output_dir / 'selection_report.csv'}")
        print(f"  Keep list        -> {output_dir / 'keep_list.txt'}")

    # Per-sequence summary.
    print("\n  Per-sequence breakdown:")
    seqs = sorted({r["sequence_id"] for r in all_rows})
    for seq in seqs:
        seq_rows  = [r for r in all_rows if r["sequence_id"] == seq]
        n_k = sum(1 for r in seq_rows if r.get("final_status") == "KEEP")
        n_q = sum(1 for r in seq_rows if r.get("final_status") == "REJECT_QUALITY")
        n_t = sum(1 for r in seq_rows if r.get("final_status") == "REJECT_THINNED")
        print(f"    {seq}  total={len(seq_rows)}  keep={n_k}  reject_quality={n_q}  thinned={n_t}")

    print()


if __name__ == "__main__":
    main()
