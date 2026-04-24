import os


def get_camera_prefix(file_stem):
    # Extract the two-character lens prefix from the filename.
    # ML0 = left Mastcam, MR0 = right Mastcam.
    # Used only for display labels and grouping key construction.
    if file_stem.startswith("ML0"):
        return "ML0"
    elif file_stem.startswith("MR0"):
        return "MR0"
    else:
        raise ValueError(f"Unknown file stem prefix: {file_stem}")


def get_camera_group_key(item):
    # Build a unique camera group key from the lens prefix and image size.
    prefix = get_camera_prefix(item["file_stem"])
    return f"{prefix}_{item['width']}x{item['height']}"


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
    # Group all computed intrinsics by unique (prefix, width, height) key.
    #
    # Returns a dict where each key is a string like "ML0_1344x1200" and each value is the list of result dicts that share that key.
    # Keys are inserted in sorted order so iteration is deterministic.
    grouped = {}

    for item in results:
        key = get_camera_group_key(item)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(item)

    # Return a new dict with keys in sorted order.
    return {k: grouped[k] for k in sorted(grouped)}


def build_camera_ids(grouped):
    # Assign a unique integer camera ID to each group key.
    # IDs are assigned in sorted order (alphabetical by key) starting at 1.
    # Sorting ensures the mapping is stable across runs.
    return {key: idx + 1 for idx, key in enumerate(sorted(grouped))}


def compute_group_averages(group_items):
    # Average the intrinsics across one camera group.
    # Averaging within a group (same lens, same image size) is valid because the CAHVOR parameters should be stable across a single sequence.
    if not group_items:
        raise ValueError("Cannot average an empty camera group.")

    avg_fx = sum(item["fx"] for item in group_items) / len(group_items)
    avg_fy = sum(item["fy"] for item in group_items) / len(group_items)
    avg_cx = sum(item["cx"] for item in group_items) / len(group_items)
    avg_cy = sum(item["cy"] for item in group_items) / len(group_items)
    avg_f  = sum(item["f"]  for item in group_items) / len(group_items)

    return {
        "fx": avg_fx,
        "fy": avg_fy,
        "cx": avg_cx,
        "cy": avg_cy,
        "f":  avg_f
    }


def write_intrinsics_txt(results, output_folder):
    # Write detailed per-image intrinsics for inspection and debugging.
    # Width and height are taken from each result dict, NOT HARDCODED.
    txt_path = os.path.join(output_folder, "intrinsics.txt")

    with open(txt_path, "w", encoding="utf-8") as f:
        for item in results:
            width  = item["width"]
            height = item["height"]
            prefix = get_camera_prefix(item["file_stem"])
            group  = get_camera_group_key(item)

            f.write(f"File: {item['file_stem']}\n")
            f.write(f"Camera group: {group}\n")
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
    # Write one readable summary file showing the averaged intrinsics for every camera group separately.
    # Groups are written in sorted key order (e.g. ML0_1152x432, ML0_1344x1200, MR0_1152x1152, ...).
    summary_path = os.path.join(output_folder, "camera_group_summary.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        for group_key in sorted(grouped_results):
            group_items = grouped_results[group_key]

            if not group_items:
                f.write(f"{group_key}: no images found\n")
                f.write("-" * 50 + "\n\n")
                continue

            # All items in this group share the same width and height.
            width  = group_items[0]["width"]
            height = group_items[0]["height"]
            avg    = compute_group_averages(group_items)

            f.write(f"Camera group: {group_key}\n")
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
    # Write a COLMAP-compatible cameras.txt.
    #
    # One entry is written per unique (prefix, width, height) group.
    # Camera IDs are assigned in sorted key order starting at 1.
    #
    # Format per line:
    # CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
    #
    # SIMPLE_RADIAL params:
    # f cx cy k
    #
    # k (radial distortion) is set to 0.0 -- COLMAP will refine it during bundle adjustment if --Mapper.ba_refine_extra_params is enabled.
    cameras_path = os.path.join(output_folder, "cameras.txt")
    camera_ids   = build_camera_ids(grouped_results)

    with open(cameras_path, "w", encoding="utf-8") as f:
        for group_key in sorted(grouped_results):
            group_items = grouped_results[group_key]

            if not group_items:
                continue

            width  = group_items[0]["width"]
            height = group_items[0]["height"]
            avg    = compute_group_averages(group_items)
            cam_id = camera_ids[group_key]

            f.write(
                f"{cam_id} "
                f"SIMPLE_RADIAL "
                f"{width} {height} "
                f"{avg['f']:.6f} {avg['cx']:.6f} {avg['cy']:.6f} 0.0\n"
            )

    return cameras_path


def write_image_camera_map(grouped_results, output_folder):
    # Write a helper file mapping each image filename to its camera ID.
    # run_colmap.py reads this to know which images belong to which camera so it can call feature_extractor once per camera group with the correct intrinsics pinned.
    map_path   = os.path.join(output_folder, "image_camera_map.txt")
    camera_ids = build_camera_ids(grouped_results)

    with open(map_path, "w", encoding="utf-8") as f:
        f.write("file_stem image_name camera_group camera_id width height\n")

        for group_key in sorted(grouped_results):
            for item in grouped_results[group_key]:
                image_name = f"{item['file_stem']}.png"
                cam_id     = camera_ids[group_key]

                f.write(
                    f"{item['file_stem']} "
                    f"{image_name} "
                    f"{group_key} "
                    f"{cam_id} "
                    f"{item['width']} "
                    f"{item['height']}\n"
                )

    return map_path


def export_all_intrinsics(results, output_folder):
    # Main export function.
    # Writes:
    # - per-image intrinsics.txt
    # - per-camera-group camera_group_summary.txt
    # - COLMAP cameras.txt  (one entry per unique prefix + size group)
    # - image_camera_map.txt (which image maps to which camera ID)
    ensure_output_folder(output_folder)

    grouped_results = split_results_by_camera_group(results)

    intrinsics_txt_path      = write_intrinsics_txt(results, output_folder)
    summary_txt_path         = write_group_summary_txt(grouped_results, output_folder)
    cameras_txt_path         = write_colmap_camera_txt(grouped_results, output_folder)
    image_camera_map_path    = write_image_camera_map(grouped_results, output_folder)

    return {
        "intrinsics_txt":           intrinsics_txt_path,
        "camera_group_summary_txt": summary_txt_path,
        "cameras_txt":              cameras_txt_path,
        "image_camera_map_txt":     image_camera_map_path
    }
