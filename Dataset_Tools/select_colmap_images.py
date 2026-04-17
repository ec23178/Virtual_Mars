# select_colmap_images.py
#
# Final COLMAP subset selector for Virtual Mars.
#
# Run this AFTER:
#   1. filter_xml_metadata.py
#   2. vic2png
#   3. raw_png_filter.py
#   4. demosaic_batch.py
#
# Run this BEFORE:
#   5. COLMAP intrinsics export
#   6. COLMAP feature extraction / matching / mapping
#
# -----------------------------------------------------------------------
# Purpose
# -----------------------------------------------------------------------
# This script selects the FINAL subset of demosaiced images to feed into
# COLMAP.
#
# It is intentionally NOT the main dataset cleanup stage.
# That job now belongs to raw_png_filter.py.
#
# This script assumes the input folder already contains a reasonably clean
# set of images, and it performs:
#
#   1. Quality scoring on the demosaiced images
#   2. Final accept/reject decision for COLMAP
#   3. Per-sequence thinning (to reduce near-duplicate frames)
#   4. Copies selected images into:
#
#        <output_dir>/
#          images/
#          selection_report.csv
#          keep_list.txt
#          sequence_summary.csv
#
# -----------------------------------------------------------------------
# Why this stage exists
# -----------------------------------------------------------------------
# After raw_png_filter.py and demosaicing, you may still have:
#   - too many near-duplicate frames in the same MCAM sequence
#   - some weaker images that are acceptable but not ideal
#
# COLMAP usually benefits from:
#   - strong texture
#   - good sharpness
#   - reasonable overlap
#   - not too many redundant frames from one exact sweep
#
# So this script keeps only the best subset for reconstruction.
#
# -----------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------
#
# Bilinear:
#   python select_colmap_images.py ^
#       --input-dir datasets\maria_pass_filtered2\demosaiced\bilinear ^
#       --output-dir COLMAP\colmap_bilinear ^
#       --max-per-sequence 6
#
# Malvar:
#   python select_colmap_images.py ^
#       --input-dir datasets\maria_pass_filtered2\demosaiced\malvar ^
#       --output-dir COLMAP\colmap_malvar ^
#       --max-per-sequence 6
#
# Dry run:
#   python select_colmap_images.py ^
#       --input-dir datasets\maria_pass_filtered2\demosaiced\bilinear ^
#       --output-dir COLMAP\colmap_bilinear ^
#       --max-per-sequence 6 ^
#       --dry-run
#
# -----------------------------------------------------------------------
# Design notes
# -----------------------------------------------------------------------
# Unlike raw_png_filter.py, this script works on fully demosaiced RGB images.
# That means the quality heuristics can reflect what COLMAP will actually see.
#
# The main difference from the earlier version is conceptual:
#   - raw_png_filter.py = early cleanup
#   - select_colmap_images.py = final COLMAP subset builder


import os
import re
import csv
import shutil
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


# -----------------------------------------------------------------------
# Sequence ID extraction
# -----------------------------------------------------------------------

MCAM_PATTERN = re.compile(r"(MCAM\d+)", re.IGNORECASE)


def get_sequence_id(filename: str) -> str:
    """Extract MCAM sequence ID from filename."""
    m = MCAM_PATTERN.search(filename.upper())
    if m:
        return m.group(1)
    return "UNKNOWN"


# -----------------------------------------------------------------------
# Image loading
# -----------------------------------------------------------------------

def load_image(image_path: Path):
    """
    Load one demosaiced PNG and return:
      - rgb array
      - gray array
      - width
      - height
    """
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img, dtype=np.float32)
    gray = np.array(img.convert("L"), dtype=np.float32)
    height, width = gray.shape
    return rgb, gray, width, height


# -----------------------------------------------------------------------
# Quality metrics
# -----------------------------------------------------------------------

def compute_sharpness(gray: np.ndarray) -> float:
    """
    Laplacian variance.
    Higher = more fine detail / less blur.
    """
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0

    center = gray[1:-1, 1:-1]
    up = gray[:-2, 1:-1]
    down = gray[2:, 1:-1]
    left = gray[1:-1, :-2]
    right = gray[1:-1, 2:]

    lap = (up + down + left + right) - (4.0 * center)
    return float(np.var(lap))


def compute_texture(gray: np.ndarray) -> float:
    """
    Mean absolute pixel difference in x and y.
    Higher = more usable scene structure.
    """
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return 0.0

    dx = np.abs(gray[:, 1:] - gray[:, :-1])
    dy = np.abs(gray[1:, :] - gray[:-1, :])

    return float((np.mean(dx) + np.mean(dy)) / 2.0)


def compute_brightness(gray: np.ndarray):
    """
    Brightness statistics.
    """
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    dark_fraction = float(np.mean(gray < 15))
    bright_fraction = float(np.mean(gray > 240))

    return mean_val, std_val, dark_fraction, bright_fraction


def has_sky_dominance(gray: np.ndarray) -> bool:
    """
    Sky-heavy frames usually give poor COLMAP support in the sky region.
    Soft penalty only.
    """
    h = gray.shape[0]
    top = gray[:max(1, int(h * 0.20)), :]
    top_mean = float(np.mean(top))
    top_texture = compute_texture(top)

    if top_mean > 170 and top_texture < 8:
        return True
    if top_mean > 145 and top_texture < 6:
        return True
    return False


def has_ground_only(gray: np.ndarray) -> bool:
    """
    Flag close-up frames dominated by similar rock/ground texture.
    Soft penalty only.
    """
    h = gray.shape[0]
    top = gray[: h // 2, :]
    bottom = gray[h // 2 :, :]

    mean_diff = abs(float(np.mean(top)) - float(np.mean(bottom)))
    top_texture = compute_texture(top)
    bottom_texture = compute_texture(bottom)

    return mean_diff < 18 and top_texture > 10 and bottom_texture > 10


# -----------------------------------------------------------------------
# Image scoring
# -----------------------------------------------------------------------

def score_image(image_path: Path) -> dict:
    """
    Score one demosaiced image for final COLMAP suitability.
    """
    try:
        rgb, gray, width, height = load_image(image_path)
    except Exception as exc:
        return {
            "score": 0.0,
            "hard_fail": True,
            "reasons": [f"image_open_failed:{type(exc).__name__}"],
            "sharpness": 0.0,
            "texture": 0.0,
            "brightness_mean": 0.0,
            "brightness_std": 0.0,
            "dark_fraction": 0.0,
            "bright_fraction": 0.0,
            "sky_dominant": False,
            "ground_only": False,
            "width": 0,
            "height": 0,
        }

    sharpness = compute_sharpness(gray)
    texture = compute_texture(gray)
    mean_val, std_val, dark_frac, bright_frac = compute_brightness(gray)
    sky_dominant = has_sky_dominance(gray)
    ground_only = has_ground_only(gray)

    reasons = []
    score = 0.0
    hard_fail = False

    # ------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------
    if width < 700 or height < 400:
        reasons.append("too_small")
        hard_fail = True

    if width >= 1200 and height >= 1100:
        score += 2.0
    elif width >= 1000 and height >= 700:
        score += 1.0

    # ------------------------------------------------------------
    # Sharpness
    # ------------------------------------------------------------
    if sharpness < 8:
        reasons.append("very_blurry")
        hard_fail = True
    elif sharpness < 25:
        reasons.append("soft")
        score += 0.5
    else:
        score += 2.0

    # ------------------------------------------------------------
    # Texture
    # ------------------------------------------------------------
    if texture < 4:
        reasons.append("very_low_texture")
        hard_fail = True
    elif texture < 8:
        reasons.append("low_texture")
        score += 0.5
    else:
        score += 2.0

    # ------------------------------------------------------------
    # Brightness / contrast
    # ------------------------------------------------------------
    if dark_frac > 0.40:
        reasons.append("too_dark")
        score -= 1.5

    if bright_frac > 0.35:
        reasons.append("too_bright")
        score -= 1.5

    if std_val < 20:
        reasons.append("low_contrast")
        score -= 0.5

    # ------------------------------------------------------------
    # Scene composition heuristics
    # ------------------------------------------------------------
    if sky_dominant:
        reasons.append("sky_dominant")
        score -= 1.0

    if ground_only:
        reasons.append("ground_only")
        score -= 0.5

    # ------------------------------------------------------------
    # Bonus for full-frame Mastcam size
    # ------------------------------------------------------------
    if width >= 1344 and height >= 1200:
        score += 1.5

    return {
        "score": score,
        "hard_fail": hard_fail,
        "reasons": reasons,
        "sharpness": sharpness,
        "texture": texture,
        "brightness_mean": mean_val,
        "brightness_std": std_val,
        "dark_fraction": dark_frac,
        "bright_fraction": bright_frac,
        "sky_dominant": sky_dominant,
        "ground_only": ground_only,
        "width": width,
        "height": height,
    }


# -----------------------------------------------------------------------
# Selection logic
# -----------------------------------------------------------------------

def select_images(scored_rows: list[dict], max_per_sequence: int) -> set[str]:
    """
    Two-pass selection:
      1. Drop hard failures
      2. Within each sequence, keep top N by score
    """
    by_sequence = {}

    for row in scored_rows:
        seq = row["sequence_id"]
        if seq not in by_sequence:
            by_sequence[seq] = []
        by_sequence[seq].append(row)

    selected_stems = set()

    for seq, rows in by_sequence.items():
        passed = [r for r in rows if not r["hard_fail"]]
        failed = [r for r in rows if r["hard_fail"]]

        for r in failed:
            r["final_status"] = "REJECT_QUALITY"

        if not passed:
            continue

        # Sort strongest first
        passed.sort(key=lambda r: (r["score"], r["sharpness"], r["texture"]), reverse=True)

        kept = passed[:max_per_sequence]
        thinned = passed[max_per_sequence:]

        for r in kept:
            r["final_status"] = "KEEP"
            selected_stems.add(r["file_stem"])

        for r in thinned:
            r["final_status"] = "REJECT_THINNED"

    return selected_stems


# -----------------------------------------------------------------------
# Reports
# -----------------------------------------------------------------------

def write_report(rows: list[dict], report_path: Path):
    """
    Full per-image selection report.
    """
    fieldnames = [
        "file_name",
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
                "file_name": row["file_name"],
                "file_stem": row["file_stem"],
                "sequence_id": row["sequence_id"],
                "final_status": row.get("final_status", "UNKNOWN"),
                "score": f"{row['score']:.3f}",
                "width": row["width"],
                "height": row["height"],
                "sharpness": f"{row['sharpness']:.2f}",
                "texture": f"{row['texture']:.2f}",
                "brightness_mean": f"{row['brightness_mean']:.1f}",
                "brightness_std": f"{row['brightness_std']:.1f}",
                "dark_fraction": f"{row['dark_fraction']:.3f}",
                "bright_fraction": f"{row['bright_fraction']:.3f}",
                "sky_dominant": row["sky_dominant"],
                "ground_only": row["ground_only"],
                "reasons": ";".join(row["reasons"]),
            })


def write_sequence_summary(rows: list[dict], summary_path: Path):
    """
    Per-sequence summary after final selection.
    """
    grouped = {}

    for row in rows:
        seq = row["sequence_id"]
        if seq not in grouped:
            grouped[seq] = {
                "sequence_id": seq,
                "count_total": 0,
                "count_keep": 0,
                "count_reject_quality": 0,
                "count_reject_thinned": 0,
                "score_sum": 0.0,
                "sample_files": [],
            }

        grouped[seq]["count_total"] += 1
        grouped[seq]["score_sum"] += row["score"]

        status = row.get("final_status", "UNKNOWN")
        if status == "KEEP":
            grouped[seq]["count_keep"] += 1
        elif status == "REJECT_QUALITY":
            grouped[seq]["count_reject_quality"] += 1
        elif status == "REJECT_THINNED":
            grouped[seq]["count_reject_thinned"] += 1

        if len(grouped[seq]["sample_files"]) < 5:
            grouped[seq]["sample_files"].append(row["file_name"])

    summary_rows = []
    for seq, item in grouped.items():
        avg_score = item["score_sum"] / max(1, item["count_total"])
        summary_rows.append({
            "sequence_id": seq,
            "count_total": item["count_total"],
            "count_keep": item["count_keep"],
            "count_reject_quality": item["count_reject_quality"],
            "count_reject_thinned": item["count_reject_thinned"],
            "avg_score": avg_score,
            "sample_files": " | ".join(item["sample_files"]),
        })

    summary_rows.sort(
        key=lambda x: (x["count_keep"], x["avg_score"], x["count_total"]),
        reverse=True,
    )

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sequence_id",
                "count_total",
                "count_keep",
                "count_reject_quality",
                "count_reject_thinned",
                "avg_score",
                "sample_files",
            ]
        )
        writer.writeheader()
        writer.writerows(summary_rows)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Select the final demosaiced images for COLMAP. "
            "Applies quality scoring and per-sequence thinning."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder containing demosaiced PNG images (bilinear OR malvar)."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "COLMAP workspace root. The script will create/use an images/ folder "
            "inside it and write reports at the root."
        )
    )
    parser.add_argument(
        "--max-per-sequence",
        type=int,
        default=6,
        help="Maximum number of images to keep per MCAM sequence (default: 6)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not copy files; only score and write reports."
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_files = sorted(input_dir.glob("*.png"))
    if not image_files:
        raise FileNotFoundError(f"No PNG files found in: {input_dir}")

    if not args.dry_run:
        images_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=========================================================")
    print("SELECT COLMAP IMAGES")
    print("=========================================================")
    print(f"Input folder       : {input_dir}")
    print(f"Output folder      : {output_dir}")
    print(f"Image count        : {len(image_files)}")
    print(f"Max per sequence   : {args.max_per_sequence}")
    print(f"Dry run            : {args.dry_run}")
    print("=========================================================\n")

    scored_rows = []

    for image_path in image_files:
        file_name = image_path.name
        file_stem = image_path.stem
        sequence_id = get_sequence_id(file_name)

        result = score_image(image_path)

        row = {
            "file_name": file_name,
            "file_stem": file_stem,
            "sequence_id": sequence_id,
            **result,
        }
        scored_rows.append(row)

        print(
            f"{file_name} | "
            f"score={result['score']:.2f} | "
            f"hard_fail={result['hard_fail']} | "
            f"reasons={';'.join(result['reasons'])}"
        )

    selected_stems = select_images(scored_rows, args.max_per_sequence)

    copied_count = 0
    for row in scored_rows:
        if row.get("final_status") == "KEEP":
            src = input_dir / row["file_name"]
            dst = images_dir / row["file_name"]

            if not args.dry_run:
                shutil.copy2(src, dst)

            copied_count += 1

    report_csv = output_dir / "selection_report.csv"
    write_report(scored_rows, report_csv)

    keep_list_path = output_dir / "keep_list.txt"
    with open(keep_list_path, "w", encoding="utf-8") as f:
        for row in scored_rows:
            if row.get("final_status") == "KEEP":
                f.write(row["file_name"] + "\n")

    summary_csv = output_dir / "sequence_summary.csv"
    write_sequence_summary(scored_rows, summary_csv)

    total_keep = sum(1 for r in scored_rows if r.get("final_status") == "KEEP")
    total_reject_quality = sum(1 for r in scored_rows if r.get("final_status") == "REJECT_QUALITY")
    total_reject_thinned = sum(1 for r in scored_rows if r.get("final_status") == "REJECT_THINNED")

    print("\n=========================================================")
    print("SELECTION COMPLETE")
    print("=========================================================")
    print(f"Total images scored       : {len(scored_rows)}")
    print(f"KEEP                      : {total_keep}")
    print(f"REJECT_QUALITY            : {total_reject_quality}")
    print(f"REJECT_THINNED            : {total_reject_thinned}")
    print(f"Copied to images/         : {copied_count}")
    print(f"Selection report          : {report_csv}")
    print(f"Keep list                 : {keep_list_path}")
    print(f"Sequence summary          : {summary_csv}")
    if args.dry_run:
        print("\nDRY RUN ONLY: no files were physically copied.")
    print("=========================================================")


if __name__ == "__main__":
    main()