import math
import numpy as np

# Mastcam CCD pixel pitch from the report / instrument details.
# 7.4 micrometres = 0.0074 mm
PIXEL_PITCH_MM = 0.0074


def get_image_size(file_stem):
    # Return the verified image size for each Mastcam group.
    if file_stem.startswith("ML0"):
        return 1152, 432
    elif file_stem.startswith("MR0"):
        return 1328, 1184
    else:
        raise ValueError(f"Unknown file stem: {file_stem}")


def compute_intrinsics(A, H, V):
    # Compute camera intrinsics from CAHVOR A, H, V vectors.
    #
    # IMPORTANT:
    # - hc = a . h
    # - vc = a . v
    # - hs = ||a x h||
    # - vs = ||a x v||
    #
    # For COLMAP, focal length must be in PIXELS, not millimetres.
    # So the pixel-space focal values are hs and vs.
    #
    # We also compute focal length in mm for inspection/debugging,
    # but the K matrix and exported f use pixel units.

    # Convert inputs to numpy arrays.
    A = np.array(A, dtype=float)
    H = np.array(H, dtype=float)
    V = np.array(V, dtype=float)

    # Normalise the optical axis so dot/cross results are stable.
    norm_A = np.linalg.norm(A)
    if norm_A == 0:
        raise ValueError("Vector A has zero length and cannot be normalised.")
    A = A / norm_A

    # Optical centre in pixel coordinates.
    # These are the principal point coordinates.
    hc = float(np.dot(A, H))
    vc = float(np.dot(A, V))

    # Focal scale in pixel units.
    # These are the magnitudes of the cross products.
    hs = float(np.linalg.norm(np.cross(A, H)))
    vs = float(np.linalg.norm(np.cross(A, V)))

    # Average focal length in pixel units for SIMPLE_RADIAL.
    f_pixels = (hs + vs) / 2.0

    # Also compute focal lengths in millimetres for sanity checks.
    fx_mm = hs * PIXEL_PITCH_MM
    fy_mm = vs * PIXEL_PITCH_MM
    f_mm = (fx_mm + fy_mm) / 2.0

    # Build the intrinsic matrix in pixel units.
    K = np.array([
        [f_pixels, 0.0, hc],
        [0.0, f_pixels, vc],
        [0.0, 0.0, 1.0]
    ], dtype=float)

    return {
        # Pixel-space focal lengths for COLMAP-style use.
        "fx": hs,
        "fy": vs,
        "cx": hc,
        "cy": vc,
        "f": f_pixels,

        # Extra debug values for inspection.
        "fx_mm": fx_mm,
        "fy_mm": fy_mm,
        "f_mm": f_mm,

        # Matrix in pixel units.
        "K": K
    }


def compute_intrinsics_for_dataset(parsed_items):
    # Compute intrinsics for every parsed XML item.
    #
    # This also adds sanity checks against the expected image dimensions.
    # If the principal point is very far from the image centre,
    # that is a strong sign that the metadata source/formulation still
    # needs more work upstream.

    results = []

    for item in parsed_items:
        intrinsics = compute_intrinsics(item["A"], item["H"], item["V"])

        width, height = get_image_size(item["file_stem"])

        # Expected image centre in pixel coordinates.
        expected_cx = width / 2.0
        expected_cy = height / 2.0

        # Distance between computed principal point and image centre.
        cx_error = intrinsics["cx"] - expected_cx
        cy_error = intrinsics["cy"] - expected_cy

        # Flag whether the principal point is inside the image bounds.
        principal_point_in_bounds = (
            0.0 <= intrinsics["cx"] <= width and
            0.0 <= intrinsics["cy"] <= height
        )

        result = {
            "file_stem": item["file_stem"],
            "A": item["A"],
            "H": item["H"],
            "V": item["V"],
            "width": width,
            "height": height,
            "expected_cx": expected_cx,
            "expected_cy": expected_cy,
            "cx_error": cx_error,
            "cy_error": cy_error,
            "principal_point_in_bounds": principal_point_in_bounds,
            **intrinsics
        }

        results.append(result)

    return results