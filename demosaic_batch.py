import os
import glob
import argparse
import numpy as np
from PIL import Image
from colour_demosaicing import (
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_Malvar2004,
)

# Mastcam IMG headers in your dataset show:
# CFA_TYPE = BAYER_RGGB
# So use RGGB for these raw Bayer PNG images.
BAYER_PATTERN = "RGGB"


def demosaic_bilinear(img_array):
    # Make sure the input is a single-channel Bayer image.
    if img_array.ndim != 2:
        raise ValueError(
            f"Expected single-channel Bayer image, got shape {img_array.shape}"
        )

    # Convert the raw 8-bit Bayer image into float format in the 0..1 range.
    # Using the same scaling for both methods keeps the comparison fair.
    img_float = img_array.astype(np.float32) / 255.0

    # Apply bilinear demosaicing using colour-demosaicing.
    # This avoids OpenCV Bayer conversion quirks and keeps both methods
    # inside the same library for consistency.
    rgb = demosaicing_CFA_Bayer_bilinear(
        img_float,
        pattern=BAYER_PATTERN
    )

    # Clip values so they stay within the valid display range.
    rgb = np.clip(rgb, 0.0, 1.0)

    # Convert back to 8-bit so the result can be saved as PNG.
    rgb = (rgb * 255.0).astype(np.uint8)

    return rgb


def demosaic_malvar(img_array):
    # Make sure the input is a single-channel Bayer image.
    if img_array.ndim != 2:
        raise ValueError(
            f"Expected single-channel Bayer image, got shape {img_array.shape}"
        )

    # Convert the raw 8-bit Bayer image into float format in the 0..1 range.
    img_float = img_array.astype(np.float32) / 255.0

    # Apply Malvar demosaicing using the same Bayer pattern.
    rgb = demosaicing_CFA_Bayer_Malvar2004(
        img_float,
        pattern=BAYER_PATTERN
    )

    # Clip values so they stay within the valid display range.
    # Do NOT use per-image min/max normalization here,
    # because that changes each image differently and hurts reproducibility.
    rgb = np.clip(rgb, 0.0, 1.0)

    # Convert back to 8-bit so the result can be saved as PNG.
    rgb = (rgb * 255.0).astype(np.uint8)

    return rgb


def process_image(image_path, output_dir, method):
    # Get the filename without extension so the output keeps the same stem.
    stem = os.path.splitext(os.path.basename(image_path))[0]

    # Load the raw vic2png output as grayscale.
    # This should be the Bayer-pattern image, not an already-colour image.
    img = np.array(Image.open(image_path).convert("L"))

    # Run the chosen demosaicing method.
    if method == "bilinear":
        out = demosaic_bilinear(img)
    elif method == "malvar":
        out = demosaic_malvar(img)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Save to the existing output folder.
    # No resizing is done here.
    # No cropping is done here.
    # No extra colour stretching is done here.
    output_path = os.path.join(output_dir, f"{stem}.png")
    Image.fromarray(out).save(output_path)

    print(f"Saved: {output_path} | shape={out.shape}")


def main():
    # Set up command-line arguments.
    parser = argparse.ArgumentParser(
        description="Batch demosaic raw Bayer PNG images using bilinear or Malvar."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder containing raw grayscale Bayer PNG images."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Existing folder where demosaiced PNG images will be saved."
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=["bilinear", "malvar"],
        help="Demosaicing method to use."
    )
    args = parser.parse_args()

    # Make sure the input folder exists.
    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input folder does not exist: {args.input_dir}")

    # Make sure the output folder already exists.
    # We are not creating folders automatically here.
    if not os.path.isdir(args.output_dir):
        raise FileNotFoundError(
            f"Output folder does not exist: {args.output_dir}"
        )

    # Collect PNG files in sorted order so processing is reproducible.
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))

    # Stop early if no PNG files are found.
    if not files:
        raise FileNotFoundError(f"No PNG files found in: {args.input_dir}")

    # Process each image one by one.
    for image_path in files:
        process_image(image_path, args.output_dir, args.method)

    print("Demosaicing complete.")


if __name__ == "__main__":
    main()