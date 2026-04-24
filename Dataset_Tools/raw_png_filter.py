# raw_png_filter.py
#
# Stage 2 visual filter for Virtual Mars.
# Run this AFTER:
#   1. filter_xml_metadata.py
#   2. vic2png conversion
#
# Run this BEFORE:
#   3. demosaic_batch.py
#   4. select_colmap_images.py
#
# This script scores the raw PNG outputs from vic2png and builds a cleaner intermediate dataset before demosaicing.
#
# It does 3 things:
#   1. Scores each raw PNG using grayscale-friendly image heuristics.
#   2. Classifies each image as keep / review / reject.
#   3. Copies the selected files into a new output dataset structure
#
# It also writes:
#   - raw_png_filter_report.csv
#   - keep_list.txt
#   - review_list.txt
#   - reject_list.txt
#   - sequence_summary.csv
#

import os
import re
import csv
import shutil
import argparse
from pathlib import Path

import numpy as np
from PIL import Image



# Filename helpers
MCAM_PATTERN = re.compile(r"(MCAM\d+)", re.IGNORECASE)


def get_sequence_id(file_name: str) -> str:
    # Extract the MCAM sequence ID from a filename.
    match = MCAM_PATTERN.search(file_name.upper())
    if match:
        return match.group(1)
    return "UNKNOWN_SEQUENCE"


def get_file_stem(file_name: str) -> str:
    # Return filename without extension.
    return Path(file_name).stem



# Image loading
def load_image(image_path: Path):
    """
    Load one PNG image and return:
      - grayscale image array
      - width
      - height

    Even if the PNG somehow has RGB channels, we convert to grayscale because
    this stage should work consistently on raw vic2png output.
    """
    img = Image.open(image_path).convert("L")
    gray = np.array(img, dtype=np.float32)
    height, width = gray.shape
    return gray, width, height



# Image quality metrics
def compute_sharpness(gray: np.ndarray) -> float:
    """
    Laplacian variance sharpness score.

    Higher value usually means:
      - more detail
      - less blur
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
    # Mean absolute pixel difference in x and y.
    # Higher value usually means more edges and more structure.

    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return 0.0

    dx = np.abs(gray[:, 1:] - gray[:, :-1])
    dy = np.abs(gray[1:, :] - gray[:-1, :])

    return float((np.mean(dx) + np.mean(dy)) / 2.0)


def compute_brightness_stats(gray: np.ndarray):
    """
    Brightness statistics:
      - mean
      - std dev
      - dark fraction
      - bright fraction
    """
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    dark_fraction = float(np.mean(gray < 15))
    bright_fraction = float(np.mean(gray > 240))

    return mean_val, std_val, dark_fraction, bright_fraction


def estimate_sky_fraction(gray: np.ndarray) -> float:
    # Very rough top-strip sky heuristic.
    # Because these are raw PNGs, treat this as a soft clue only.
   
    h, _ = gray.shape
    top_h = max(1, int(h * 0.20))
    top = gray[:top_h, :]

    top_mean = float(np.mean(top))
    top_texture = compute_texture(top)

    if top_mean > 170 and top_texture < 8:
        return 1.0
    if top_mean > 145 and top_texture < 6:
        return 0.7
    return 0.0


def estimate_ground_only_flag(gray: np.ndarray) -> bool:
    # Heuristic for frames that may be mostly close-up ground/rock.
    # his should not auto-reject; it only lowers confidence.
    
    h, _ = gray.shape
    if h < 2:
        return False

    top = gray[: h // 2, :]
    bottom = gray[h // 2 :, :]

    top_mean = float(np.mean(top))
    bottom_mean = float(np.mean(bottom))
    top_texture = compute_texture(top)
    bottom_texture = compute_texture(bottom)

    mean_diff = abs(top_mean - bottom_mean)

    return mean_diff < 18 and top_texture > 10 and bottom_texture > 10


def estimate_rover_intrusion_flag(gray: np.ndarray) -> bool:
    # Very rough flag for possible rover hardware intrusion in the lower strip.
    # This is a review signal, not a hard rejection rule.
    
    h, _ = gray.shape
    bottom_h = max(1, int(h * 0.20))
    bottom = gray[h - bottom_h :, :]

    bottom_texture = compute_texture(bottom)
    bottom_dark = float(np.mean(bottom < 20))
    bottom_bright = float(np.mean(bottom > 235))

    return bottom_texture > 18 and (bottom_dark + bottom_bright) > 0.20


# Image scoring
def score_image(image_path: Path) -> dict:
    # Score one raw PNG and classify it as keep / review / reject.
    
    try:
        gray, width, height = load_image(image_path)
    except Exception as exc:
        return {
            "status": "reject",
            "score": 0.0,
            "width": 0,
            "height": 0,
            "sharpness": 0.0,
            "texture": 0.0,
            "brightness_mean": 0.0,
            "brightness_std": 0.0,
            "dark_fraction": 0.0,
            "bright_fraction": 0.0,
            "sky_fraction": 0.0,
            "ground_only_flag": False,
            "rover_intrusion_flag": False,
            "reasons": f"image_open_failed:{type(exc).__name__}",
        }

    sharpness = compute_sharpness(gray)
    texture = compute_texture(gray)
    mean_val, std_val, dark_fraction, bright_fraction = compute_brightness_stats(gray)
    sky_fraction = estimate_sky_fraction(gray)
    ground_only = estimate_ground_only_flag(gray)
    rover_intrusion = estimate_rover_intrusion_flag(gray)

    reasons = []
    score = 0.0

    # Resolution checks
    if width < 700 or height < 300:
        reasons.append("too_small")
        return {
            "status": "reject",
            "score": 0.0,
            "width": width,
            "height": height,
            "sharpness": sharpness,
            "texture": texture,
            "brightness_mean": mean_val,
            "brightness_std": std_val,
            "dark_fraction": dark_fraction,
            "bright_fraction": bright_fraction,
            "sky_fraction": sky_fraction,
            "ground_only_flag": ground_only,
            "rover_intrusion_flag": rover_intrusion,
            "reasons": ";".join(reasons),
        }

    if width >= 1200 and height >= 400:
        score += 2.0
    elif width >= 1000:
        score += 1.0


    # Sharpness checks
    if sharpness < 8:
        reasons.append("very_blurry")
        return {
            "status": "reject",
            "score": score,
            "width": width,
            "height": height,
            "sharpness": sharpness,
            "texture": texture,
            "brightness_mean": mean_val,
            "brightness_std": std_val,
            "dark_fraction": dark_fraction,
            "bright_fraction": bright_fraction,
            "sky_fraction": sky_fraction,
            "ground_only_flag": ground_only,
            "rover_intrusion_flag": rover_intrusion,
            "reasons": ";".join(reasons),
        }
    elif sharpness < 20:
        reasons.append("soft")
        score += 0.5
    else:
        score += 2.0


    # Texture checks
    if texture < 4:
        reasons.append("very_low_texture")
        return {
            "status": "reject",
            "score": score,
            "width": width,
            "height": height,
            "sharpness": sharpness,
            "texture": texture,
            "brightness_mean": mean_val,
            "brightness_std": std_val,
            "dark_fraction": dark_fraction,
            "bright_fraction": bright_fraction,
            "sky_fraction": sky_fraction,
            "ground_only_flag": ground_only,
            "rover_intrusion_flag": rover_intrusion,
            "reasons": ";".join(reasons),
        }
    elif texture < 8:
        reasons.append("low_texture")
        score += 0.5
    else:
        score += 2.0


    # Brightness sanity checks
    if dark_fraction > 0.40:
        reasons.append("too_dark")
        score -= 1.0

    if bright_fraction > 0.35:
        reasons.append("too_bright")
        score -= 1.0

    if std_val < 20:
        reasons.append("low_contrast")
        score -= 0.5

    # Scene heuristics
    if sky_fraction > 0.5:
        reasons.append("possible_sky_dominance")
        score -= 1.0

    if ground_only:
        reasons.append("possible_ground_only")
        score -= 1.0

    if rover_intrusion:
        reasons.append("possible_rover_intrusion")
        score -= 1.0

    suspicious = any(
        tag in reasons
        for tag in [
            "soft",
            "low_texture",
            "too_dark",
            "too_bright",
            "low_contrast",
            "possible_sky_dominance",
            "possible_ground_only",
            "possible_rover_intrusion",
        ]
    )

    if score >= 4.0 and not suspicious:
        status = "keep"
    elif score >= 2.0:
        status = "review"
    else:
        status = "reject"

    return {
        "status": status,
        "score": score,
        "width": width,
        "height": height,
        "sharpness": sharpness,
        "texture": texture,
        "brightness_mean": mean_val,
        "brightness_std": std_val,
        "dark_fraction": dark_fraction,
        "bright_fraction": bright_fraction,
        "sky_fraction": sky_fraction,
        "ground_only_flag": ground_only,
        "rover_intrusion_flag": rover_intrusion,
        "reasons": ";".join(reasons),
    }


# Sequence summary
def summarise_sequences(rows: list[dict]) -> list[dict]:
    # Summarise scores by MCAM sequence to help inspect whether one sequence
    # dominates the filtered set.
    
    grouped = {}

    for row in rows:
        seq = row["sequence_id"]

        if seq not in grouped:
            grouped[seq] = {
                "sequence_id": seq,
                "count_total": 0,
                "count_keep": 0,
                "count_review": 0,
                "count_reject": 0,
                "score_sum": 0.0,
                "sample_files": [],
            }

        grouped[seq]["count_total"] += 1
        grouped[seq]["score_sum"] += row["score"]

        if row["status"] == "keep":
            grouped[seq]["count_keep"] += 1
        elif row["status"] == "review":
            grouped[seq]["count_review"] += 1
        else:
            grouped[seq]["count_reject"] += 1

        if len(grouped[seq]["sample_files"]) < 5:
            grouped[seq]["sample_files"].append(row["file_name"])

    summary_rows = []

    for seq, item in grouped.items():
        avg_score = item["score_sum"] / max(1, item["count_total"])

        summary_rows.append({
            "sequence_id": seq,
            "count_total": item["count_total"],
            "count_keep": item["count_keep"],
            "count_review": item["count_review"],
            "count_reject": item["count_reject"],
            "avg_score": avg_score,
            "sample_files": " | ".join(item["sample_files"]),
        })

    summary_rows.sort(
        key=lambda x: (x["count_keep"], x["avg_score"], x["count_total"]),
        reverse=True,
    )

    return summary_rows



# File copy helpers
def find_matching_file(directory: Path, stem: str, extensions: tuple[str, ...]) -> Path | None:
    # Find a matching file stem with one of several possible extensions.
    for ext in extensions:
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def copy_if_exists(src: Path | None, dst: Path, dry_run: bool) -> bool:
    # Copy file if present.
    if src is None:
        return False
    if not dry_run:
        shutil.copy2(src, dst)
    return True


# Main
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Filter raw vic2png PNG files before demosaicing. "
            "Copies accepted PNGs and their matching XML/IMG files into a new "
            "dataset structure for the next stage."
        )
    )

    parser.add_argument(
        "--png-dir",
        required=True,
        help="Folder containing raw PNG files produced by vic2png."
    )
    parser.add_argument(
        "--xml-dir",
        required=True,
        help="Folder containing matching XML files for the same stems."
    )
    parser.add_argument(
        "--img-dir",
        required=True,
        help="Folder containing matching IMG files for the same stems."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Root output folder. The script will create: "
            "data/, IMG_files/, raw_png/ and CSV/text reports."
        )
    )
    parser.add_argument(
        "--include-review",
        action="store_true",
        help="Also copy images labelled as review. Default copies only keep."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not copy files; only score and write reports."
    )

    args = parser.parse_args()

    png_dir = Path(args.png_dir)
    xml_dir = Path(args.xml_dir)
    img_dir = Path(args.img_dir)
    output_dir = Path(args.output_dir)

    if not png_dir.is_dir():
        raise FileNotFoundError(f"PNG directory not found: {png_dir}")
    if not xml_dir.is_dir():
        raise FileNotFoundError(f"XML directory not found: {xml_dir}")
    if not img_dir.is_dir():
        raise FileNotFoundError(f"IMG directory not found: {img_dir}")

    png_files = sorted(png_dir.glob("*.png"))
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in: {png_dir}")

    output_data_dir = output_dir / "data"
    output_img_dir = output_dir / "IMG_files"
    output_png_dir = output_dir / "raw_png"

    if not args.dry_run:
        output_data_dir.mkdir(parents=True, exist_ok=True)
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_png_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=========================================================")
    print("RAW PNG FILTER")
    print("=========================================================")
    print(f"PNG input   : {png_dir}")
    print(f"XML input   : {xml_dir}")
    print(f"IMG input   : {img_dir}")
    print(f"Output root : {output_dir}")
    print(f"PNG count   : {len(png_files)}")
    print(f"Copy review : {args.include_review}")
    print(f"Dry run     : {args.dry_run}")
    print("=========================================================\n")

    all_rows = []
    copied_keep = 0
    copied_review = 0
    copied_xml = 0
    copied_img = 0

    for image_path in png_files:
        file_name = image_path.name
        stem = image_path.stem
        sequence_id = get_sequence_id(file_name)

        metrics = score_image(image_path)

        should_copy = (
            metrics["status"] == "keep" or
            (args.include_review and metrics["status"] == "review")
        )

        xml_path = find_matching_file(xml_dir, stem, (".xml", ".XML"))
        img_path = find_matching_file(img_dir, stem, (".IMG", ".img"))

        png_found = image_path.exists()
        xml_found = xml_path is not None
        img_found = img_path is not None

        copy_status = "not_selected"
        if should_copy:
            if not xml_found or not img_found:
                copy_status = "selected_but_missing_pair"
            else:
                copy_status = "selected"

        row = {
            "file_name": file_name,
            "file_stem": stem,
            "sequence_id": sequence_id,
            **metrics,
            "png_found": png_found,
            "xml_found": xml_found,
            "img_found": img_found,
            "copy_status": copy_status,
        }
        all_rows.append(row)

        print(
            f"{file_name} | "
            f"{metrics['status']} | "
            f"score={metrics['score']:.2f} | "
            f"copy={copy_status} | "
            f"reasons={metrics['reasons']}"
        )

        if should_copy and xml_found and img_found:
            png_ok = copy_if_exists(image_path, output_png_dir / file_name, args.dry_run)
            xml_ok = copy_if_exists(xml_path, output_data_dir / xml_path.name, args.dry_run)
            img_ok = copy_if_exists(img_path, output_img_dir / img_path.name, args.dry_run)

            if png_ok and metrics["status"] == "keep":
                copied_keep += 1
            elif png_ok and metrics["status"] == "review":
                copied_review += 1

            if xml_ok:
                copied_xml += 1
            if img_ok:
                copied_img += 1


    # Write main CSV report
    report_csv = output_dir / "raw_png_filter_report.csv"
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_name",
                "file_stem",
                "sequence_id",
                "status",
                "score",
                "width",
                "height",
                "sharpness",
                "texture",
                "brightness_mean",
                "brightness_std",
                "dark_fraction",
                "bright_fraction",
                "sky_fraction",
                "ground_only_flag",
                "rover_intrusion_flag",
                "png_found",
                "xml_found",
                "img_found",
                "copy_status",
                "reasons",
            ]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    # Write text lists
    keep_txt = output_dir / "keep_list.txt"
    review_txt = output_dir / "review_list.txt"
    reject_txt = output_dir / "reject_list.txt"

    with open(keep_txt, "w", encoding="utf-8") as f_keep, \
         open(review_txt, "w", encoding="utf-8") as f_review, \
         open(reject_txt, "w", encoding="utf-8") as f_reject:

        for row in all_rows:
            if row["status"] == "keep":
                f_keep.write(row["file_name"] + "\n")
            elif row["status"] == "review":
                f_review.write(row["file_name"] + "\n")
            else:
                f_reject.write(row["file_name"] + "\n")

    
    # Write sequence summary
    seq_rows = summarise_sequences(all_rows)
    seq_csv = output_dir / "sequence_summary.csv"

    with open(seq_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sequence_id",
                "count_total",
                "count_keep",
                "count_review",
                "count_reject",
                "avg_score",
                "sample_files",
            ]
        )
        writer.writeheader()
        writer.writerows(seq_rows)


    # Summary
    total_keep = sum(1 for row in all_rows if row["status"] == "keep")
    total_review = sum(1 for row in all_rows if row["status"] == "review")
    total_reject = sum(1 for row in all_rows if row["status"] == "reject")

    print("\n=========================================================")
    print("RAW PNG FILTER COMPLETE")
    print("=========================================================")
    print(f"Total PNGs scored         : {len(all_rows)}")
    print(f"KEEP                      : {total_keep}")
    print(f"REVIEW                    : {total_review}")
    print(f"REJECT                    : {total_reject}")
    print(f"Copied KEEP PNGs          : {copied_keep}")
    print(f"Copied REVIEW PNGs        : {copied_review}")
    print(f"Copied XML files          : {copied_xml}")
    print(f"Copied IMG files          : {copied_img}")
    print(f"Main report               : {report_csv}")
    print(f"Sequence summary          : {seq_csv}")
    print(f"Keep list                 : {keep_txt}")
    print(f"Review list               : {review_txt}")
    print(f"Reject list               : {reject_txt}")
    if args.dry_run:
        print("\nDRY RUN ONLY: no files were physically copied.")
    print("=========================================================")


if __name__ == "__main__":
    main()