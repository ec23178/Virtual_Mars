import os


def get_camera_group(file_stem):
    # Group images by Mastcam lens.
    # ML0 = left Mastcam
    # MR0 = right Mastcam
    if file_stem.startswith("ML0"):
        return "ML0"
    elif file_stem.startswith("MR0"):
        return "MR0"
    else:
        raise ValueError(f"Unknown file stem: {file_stem}")


def get_image_size(file_stem):
    # Return the actual image dimensions for each Mastcam group.
    # These sizes match the verified PNG outputs in your repo.
    if file_stem.startswith("ML0"):
        return 1152, 432
    elif file_stem.startswith("MR0"):
        return 1328, 1184
    else:
        raise ValueError(f"Unknown file stem: {file_stem}")


def ensure_output_folder(output_folder):
    # Create the output folder if it does not exist.
    os.makedirs(output_folder, exist_ok=True)


def format_matrix_as_text(K):
    # Format a 3x3 intrinsic matrix in a readable way.
    return (
        f"[{K[0][0]:.6f}, {K[0][1]:.6f}, {K[0][2]:.6f}]\n"
        f"[{K[1][0]:.6f}, {K[1][1]:.6f}, {K[1][2]:.6f}]\n"
        f"[{K[2][0]:.6f}, {K[2][1]:.6f}, {K[2][2]:.6f}]"
    )


def split_results_by_camera_group(results):
    # Split all computed intrinsics into left-camera and right-camera groups.
    grouped = {
        "ML0": [],
        "MR0": []
    }

    for item in results:
        camera_group = get_camera_group(item["file_stem"])
        grouped[camera_group].append(item)

    return grouped


def compute_group_averages(group_items):
    # Average the intrinsics across one camera group.
    # This is much better than averaging left and right together.
    if not group_items:
        raise ValueError("Cannot average an empty camera group.")

    avg_fx = sum(item["fx"] for item in group_items) / len(group_items)
    avg_fy = sum(item["fy"] for item in group_items) / len(group_items)
    avg_cx = sum(item["cx"] for item in group_items) / len(group_items)
    avg_cy = sum(item["cy"] for item in group_items) / len(group_items)
    avg_f = sum(item["f"] for item in group_items) / len(group_items)

    return {
        "fx": avg_fx,
        "fy": avg_fy,
        "cx": avg_cx,
        "cy": avg_cy,
        "f": avg_f
    }


def write_intrinsics_txt(results, output_folder):
    # Write detailed per-image intrinsics for inspection and debugging.
    txt_path = os.path.join(output_folder, "intrinsics.txt")

    with open(txt_path, "w", encoding="utf-8") as f:
        for item in results:
            width, height = get_image_size(item["file_stem"])
            camera_group = get_camera_group(item["file_stem"])

            f.write(f"File: {item['file_stem']}\n")
            f.write(f"Camera group: {camera_group}\n")
            f.write(f"Image size: {width} x {height}\n")
            f.write(f"fx: {item['fx']:.6f}\n")
            f.write(f"fy: {item['fy']:.6f}\n")
            f.write(f"cx: {item['cx']:.6f}\n")
            f.write(f"cy: {item['cy']:.6f}\n")
            f.write(f"f: {item['f']:.6f}\n")
            f.write("K:\n")
            f.write(format_matrix_as_text(item["K"]))
            f.write("\n")
            f.write("-" * 50 + "\n\n")

    return txt_path


def write_group_summary_txt(grouped_results, output_folder):
    # Write one readable summary file showing the averaged intrinsics
    # for the left and right camera groups separately.
    summary_path = os.path.join(output_folder, "camera_group_summary.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        for camera_group in ["ML0", "MR0"]:
            group_items = grouped_results[camera_group]

            if not group_items:
                f.write(f"{camera_group}: no images found\n")
                f.write("-" * 50 + "\n\n")
                continue

            width, height = get_image_size(group_items[0]["file_stem"])
            avg = compute_group_averages(group_items)

            f.write(f"Camera group: {camera_group}\n")
            f.write(f"Number of images: {len(group_items)}\n")
            f.write(f"Image size: {width} x {height}\n")
            f.write(f"Average fx: {avg['fx']:.6f}\n")
            f.write(f"Average fy: {avg['fy']:.6f}\n")
            f.write(f"Average cx: {avg['cx']:.6f}\n")
            f.write(f"Average cy: {avg['cy']:.6f}\n")
            f.write(f"Average f : {avg['f']:.6f}\n")
            f.write("-" * 50 + "\n\n")

    return summary_path


def write_colmap_camera_txt(grouped_results, output_folder):
    # Write a COLMAP-compatible cameras.txt using two shared cameras:
    # one for ML0 and one for MR0.
    #
    # Format:
    # CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
    #
    # SIMPLE_RADIAL expects:
    # f cx cy k
    #
    # Here k is set to 0.0 for now.
    cameras_path = os.path.join(output_folder, "cameras.txt")

    camera_ids = {
        "ML0": 1,
        "MR0": 2
    }

    with open(cameras_path, "w", encoding="utf-8") as f:
        for camera_group in ["ML0", "MR0"]:
            group_items = grouped_results[camera_group]

            if not group_items:
                continue

            width, height = get_image_size(group_items[0]["file_stem"])
            avg = compute_group_averages(group_items)

            f.write(
                f"{camera_ids[camera_group]} "
                f"SIMPLE_RADIAL "
                f"{width} {height} "
                f"{avg['f']:.6f} {avg['cx']:.6f} {avg['cy']:.6f} 0.0\n"
            )

    return cameras_path


def write_image_camera_map(grouped_results, output_folder):
    # Write a simple helper file showing which image belongs to which camera ID.
    # This is useful when you later decide how to inject intrinsics into COLMAP.
    map_path = os.path.join(output_folder, "image_camera_map.txt")

    camera_ids = {
        "ML0": 1,
        "MR0": 2
    }

    with open(map_path, "w", encoding="utf-8") as f:
        f.write("file_stem image_name camera_group camera_id width height\n")

        for camera_group in ["ML0", "MR0"]:
            for item in grouped_results[camera_group]:
                width, height = get_image_size(item["file_stem"])
                image_name = f"{item['file_stem']}.png"

                f.write(
                    f"{item['file_stem']} "
                    f"{image_name} "
                    f"{camera_group} "
                    f"{camera_ids[camera_group]} "
                    f"{width} "
                    f"{height}\n"
                )

    return map_path


def export_all_intrinsics(results, output_folder):
    # Main export function.
    # This writes:
    # - detailed per-image intrinsics
    # - per-camera-group summary
    # - COLMAP-style cameras.txt
    # - image-to-camera assignment helper file
    ensure_output_folder(output_folder)

    grouped_results = split_results_by_camera_group(results)

    intrinsics_txt_path = write_intrinsics_txt(results, output_folder)
    summary_txt_path = write_group_summary_txt(grouped_results, output_folder)
    cameras_txt_path = write_colmap_camera_txt(grouped_results, output_folder)
    image_camera_map_path = write_image_camera_map(grouped_results, output_folder)

    return {
        "intrinsics_txt": intrinsics_txt_path,
        "camera_group_summary_txt": summary_txt_path,
        "cameras_txt": cameras_txt_path,
        "image_camera_map_txt": image_camera_map_path
    }