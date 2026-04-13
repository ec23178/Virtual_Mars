import os
import re
import csv
import math
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


# ------------------------------------------------------------
# Helper: extract the MCAM sequence group from a filename.
#
# Example:
# ML0_485553931EDR_S0481146MCAM04390D1.png
# -> MCAM04390
# ------------------------------------------------------------
def get_sequence_id(file_name):
    match = re.search(r"(MCAM\d+)", file_name.upper())
    if match:
        return match.group(1)
    return "UNKNOWN_SEQUENCE"


# ------------------------------------------------------------
# Helper: read one PNG image and return:
# - rgb image
# - grayscale image
# - width
# - height
#
# We force RGB so the script behaves consistently even if some
# PNGs are grayscale and some are colour.
# ------------------------------------------------------------
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img)
    gray = np.array(img.convert("L"))
    height, width = gray.shape
    return rgb, gray, width, height


# ------------------------------------------------------------
# Helper: compute a simple Laplacian-variance sharpness score.
#
# Higher score generally means:
# - more detail
# - less blur
#
# This is a common cheap blur heuristic.
# ------------------------------------------------------------
def compute_sharpness(gray):
    gray = gray.astype(np.float32)

    # 4-neighbour Laplacian
    center = gray[1:-1, 1:-1]
    up = gray[:-2, 1:-1]
    down = gray[2:, 1:-1]
    left = gray[1:-1, :-2]
    right = gray[1:-1, 2:]

    lap = (up + down + left + right) - (4.0 * center)

    return float(np.var(lap))


# ------------------------------------------------------------
# Helper: compute a simple texture score.
#
# Higher score generally means:
# - more edges / rock structure
# - less blank sky / smooth surface
# ------------------------------------------------------------
def compute_texture(gray):
    gray = gray.astype(np.float32)

    # Horizontal and vertical pixel differences
    dx = np.abs(gray[:, 1:] - gray[:, :-1])
    dy = np.abs(gray[1:, :] - gray[:-1, :])

    # Average edge strength
    texture = (np.mean(dx) + np.mean(dy)) / 2.0
    return float(texture)


# ------------------------------------------------------------
# Helper: compute brightness statistics.
# ------------------------------------------------------------
def compute_brightness_stats(gray):
    gray = gray.astype(np.float32)

    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    # Fraction of very dark and very bright pixels
    dark_fraction = float(np.mean(gray < 15))
    bright_fraction = float(np.mean(gray > 240))

    return mean_val, std_val, dark_fraction, bright_fraction


# ------------------------------------------------------------
# Helper: estimate whether the top of the image looks like sky.
#
# This is NOT perfect.
# It only gives a rough flag.
#
# We treat the top strip as possible sky if it is:
# - relatively bright
# - relatively low texture
# ------------------------------------------------------------
def estimate_sky_fraction(gray):
    h, w = gray.shape

    # Use the top 20% as a "sky candidate" strip
    top_h = max(1, int(h * 0.20))
    top = gray[:top_h, :]

    top_mean = float(np.mean(top))
    top_texture = compute_texture(top)

    # Very rough sky-like rule
    # Bright + low texture often indicates sky/haze
    if top_mean > 170 and top_texture < 8:
        return 1.0

    if top_mean > 145 and top_texture < 6:
        return 0.7

    return 0.0


# ------------------------------------------------------------
# Helper: estimate whether the frame is "ground-only".
#
# This is also NOT perfect.
# We are only trying to catch frames that are dominated by
# near-field surface with little wider scene context.
#
# Heuristic idea:
# - compare top half texture/brightness to bottom half
# - if top looks very similar to bottom and there is no obvious
#   horizon/brightness transition, it may be mostly ground-only
# ------------------------------------------------------------
def estimate_ground_only_flag(gray):
    h, w = gray.shape

    top = gray[: h // 2, :]
    bottom = gray[h // 2 :, :]

    top_mean = float(np.mean(top))
    bottom_mean = float(np.mean(bottom))

    top_texture = compute_texture(top)
    bottom_texture = compute_texture(bottom)

    # Small brightness difference and both halves textured
    # often means "all rock / all ground" rather than a scene
    mean_diff = abs(top_mean - bottom_mean)

    if mean_diff < 18 and top_texture > 10 and bottom_texture > 10:
        return True

    return False


# ------------------------------------------------------------
# Helper: estimate whether rover hardware may be intruding.
#
# This is the weakest heuristic here.
#
# We cannot reliably detect rover parts from metadata alone,
# and even from pixels this is only a rough clue.
#
# We flag possible rover intrusion if:
# - the bottom strip is very high contrast
# - and a lot of it is either very dark or very bright
#
# That can happen with rover structure, but it can also happen
# in rocky scenes. So this is a REVIEW flag, not auto-reject.
# ------------------------------------------------------------
def estimate_rover_intrusion_flag(gray):
    h, w = gray.shape

    bottom_h = max(1, int(h * 0.20))
    bottom = gray[h - bottom_h :, :]

    bottom_texture = compute_texture(bottom)
    bottom_dark = float(np.mean(bottom < 20))
    bottom_bright = float(np.mean(bottom > 235))

    if bottom_texture > 18 and (bottom_dark + bottom_bright) > 0.20:
        return True

    return False


# ------------------------------------------------------------
# Score one image and return:
# - status: keep / review / reject
# - numeric score
# - reasons
#
# IMPORTANT:
# This is a first-pass triage, not a perfect oracle.
# ------------------------------------------------------------
def score_image(image_path):
    rgb, gray, width, height = load_image(image_path)

    sharpness = compute_sharpness(gray)
    texture = compute_texture(gray)
    mean_val, std_val, dark_fraction, bright_fraction = compute_brightness_stats(gray)
    sky_fraction = estimate_sky_fraction(gray)
    ground_only = estimate_ground_only_flag(gray)
    rover_intrusion = estimate_rover_intrusion_flag(gray)

    reasons = []
    score = 0.0

    # --------------------------------------------------------
    # Resolution checks
    #
    # Small images are less useful for COLMAP.
    # --------------------------------------------------------
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
            "sky_fraction": sky_fraction,
            "ground_only_flag": ground_only,
            "rover_intrusion_flag": rover_intrusion,
            "reasons": ";".join(reasons),
        }

    # Give points for larger frames
    if width >= 1200 and height >= 400:
        score += 2.0
    elif width >= 1000:
        score += 1.0

    # --------------------------------------------------------
    # Sharpness checks
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Texture checks
    #
    # Very low texture often means poor structure for COLMAP.
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Brightness sanity checks
    # --------------------------------------------------------
    if dark_fraction > 0.40:
        reasons.append("too_dark")
        score -= 1.0

    if bright_fraction > 0.35:
        reasons.append("too_bright")
        score -= 1.0

    if std_val < 20:
        reasons.append("low_contrast")
        score -= 0.5

    # --------------------------------------------------------
    # Scene heuristics
    #
    # These do NOT auto-reject.
    # They only push images into REVIEW if suspicious.
    # --------------------------------------------------------
    if sky_fraction > 0.5:
        reasons.append("possible_sky_dominance")
        score -= 1.0

    if ground_only:
        reasons.append("possible_ground_only")
        score -= 1.0

    if rover_intrusion:
        reasons.append("possible_rover_intrusion")
        score -= 1.0

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------
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
        "sky_fraction": sky_fraction,
        "ground_only_flag": ground_only,
        "rover_intrusion_flag": rover_intrusion,
        "reasons": ";".join(reasons),
    }


# ------------------------------------------------------------
# Build a per-sequence summary.
#
# This is important because you do NOT want to inspect 600
# random frames. You want to inspect a much smaller number of
# sequence groups.
# ------------------------------------------------------------
def summarise_sequences(rows):
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

        # Keep a few sample file names for quick inspection
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

    # Sort better-looking sequences first
    summary_rows.sort(
        key=lambda x: (x["count_keep"], x["avg_score"], x["count_total"]),
        reverse=True
    )

    return summary_rows


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Filter Maria Pass PNG candidates into keep/review/reject groups."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder containing PNG images, e.g. datasets/maria_pass/images"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Folder where CSV reports and keep/review/reject lists will be saved"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    png_files = sorted(input_dir.glob("*.png"))

    if not png_files:
        raise FileNotFoundError(f"No PNG files found in: {input_dir}")

    all_rows = []

    print("Scoring images...\n")

    for image_path in png_files:
        file_name = image_path.name
        sequence_id = get_sequence_id(file_name)

        metrics = score_image(image_path)

        row = {
            "file_name": file_name,
            "sequence_id": sequence_id,
            **metrics
        }

        all_rows.append(row)

        print(
            f"{file_name} | "
            f"{metrics['status']} | "
            f"score={metrics['score']:.2f} | "
            f"reasons={metrics['reasons']}"
        )

    # --------------------------------------------------------
    # Write per-image report
    # --------------------------------------------------------
    report_csv = output_dir / "filter_report.csv"

    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_name",
                "sequence_id",
                "status",
                "score",
                "width",
                "height",
                "sharpness",
                "texture",
                "brightness_mean",
                "brightness_std",
                "sky_fraction",
                "ground_only_flag",
                "rover_intrusion_flag",
                "reasons",
            ]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    # --------------------------------------------------------
    # Write keep/review/reject text lists
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Write sequence summary
    # --------------------------------------------------------
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

    print("\n===== DONE =====")
    print(f"Per-image report   : {report_csv}")
    print(f"Sequence summary   : {seq_csv}")
    print(f"Keep list          : {keep_txt}")
    print(f"Review list        : {review_txt}")
    print(f"Reject list        : {reject_txt}")


if __name__ == "__main__":
    main()