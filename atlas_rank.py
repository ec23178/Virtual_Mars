import os
import re
import argparse
import pandas as pd


def extract_sequence_id(product_id):
    # Sequence IDs look like S0030078 inside the product name.
    if not isinstance(product_id, str):
        return ""

    match = re.search(r"(S\d{7,})", product_id)

    if match:
        return match.group(1)

    return ""


def extract_eye(product_id):
    # ML usually means Mastcam Left.
    # MR usually means Mastcam Right.
    if not isinstance(product_id, str):
        return ""

    if product_id.startswith("ML"):
        return "LEFT"

    if product_id.startswith("MR"):
        return "RIGHT"

    return ""


def main():
    # Set up command-line arguments.
    parser = argparse.ArgumentParser(
        description="Rank scraped Atlas result rows by Mastcam eye and sequence group."
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to the raw Atlas results CSV."
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to the ranked candidate CSV that will be saved."
    )
    args = parser.parse_args()

    # Make sure the input CSV exists.
    if not os.path.isfile(args.input_csv):
        raise FileNotFoundError(f"Input CSV does not exist: {args.input_csv}")

    # Make sure the output folder already exists.
    output_dir = os.path.dirname(args.output_csv)
    if output_dir and not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Output folder does not exist: {output_dir}")

    # Load the CSV.
    df = pd.read_csv(args.input_csv)

    # Clean important columns so string operations do not fail.
    for col in ["product_id", "instrument", "product_type", "timestamp"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # Extract the sequence ID and camera eye from the product name.
    df["sequence_id"] = df["product_id"].apply(extract_sequence_id)
    df["eye"] = df["product_id"].apply(extract_eye)

    # Keep only left-eye images for now.
    # This keeps the first pass simpler and more consistent.
    df = df[df["eye"] == "LEFT"]

    # If RDR images exist, prefer them over EDR for the first pass.
    rdr_df = df[df["product_type"].str.contains("RDR", case=False, na=False)]

    if len(rdr_df) > 0:
        df = rdr_df

    # Count how many images belong to each sequence.
    sequence_counts = df["sequence_id"].value_counts().to_dict()
    df["sequence_count"] = df["sequence_id"].map(sequence_counts).fillna(0).astype(int)

    # Sort so the largest sequence groups appear first.
    df = df.sort_values(
        by=["sequence_count", "sequence_id", "timestamp", "product_id"],
        ascending=[False, True, True, True]
    )

    # Save the ranked CSV.
    df.to_csv(args.output_csv, index=False)

    print("\n=========================================================")
    print(f"Saved ranked candidates to:")
    print(args.output_csv)
    print("=========================================================\n")

    # Print a quick summary of the biggest sequence groups.
    print("Top sequence groups:\n")

    summary = (
        df.groupby("sequence_id")
        .size()
        .reset_index(name="image_count")
        .sort_values("image_count", ascending=False)
    )

    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()