import os
import argparse
import pandas as pd
from playwright.sync_api import sync_playwright


# Atlas Data Access API endpoint.
DATA_ACCESS_BASE = "https://pds-imaging.jpl.nasa.gov/api/data"


def atlas_uri_to_download_url(uri):
    # Convert an Atlas URI into an Atlas Data Access URL.
    return f"{DATA_ACCESS_BASE}/{uri}"


def fetch_binary_with_browser(page, url):
    # Use the real browser engine to fetch the file.
    # This avoids the 403 problem that requests hit.
    response = page.goto(url, wait_until="networkidle", timeout=120000)

    if response is None:
        raise RuntimeError(f"No response returned for URL: {url}")

    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status} for URL: {url}")

    return response.body()


def save_bytes(output_path, data):
    # Save binary bytes to disk.
    with open(output_path, "wb") as f:
        f.write(data)


def main():
    # Set up command-line arguments.
    parser = argparse.ArgumentParser(
        description="Download a sample of IMG and XML files using a real browser engine."
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to the CSV manifest created by atlas_api_search.py"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of rows to download from the manifest."
    )
    parser.add_argument(
        "--img-dir",
        required=True,
        help="Folder where .IMG files will be saved."
    )
    parser.add_argument(
        "--xml-dir",
        required=True,
        help="Folder where .xml files will be saved."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode."
    )
    args = parser.parse_args()

    # Create output folders if needed.
    os.makedirs(args.img_dir, exist_ok=True)
    os.makedirs(args.xml_dir, exist_ok=True)

    # Load the manifest.
    df = pd.read_csv(args.input_csv)

    if df.empty:
        raise ValueError(f"No rows found in manifest: {args.input_csv}")

    # Only take a small test sample.
    df = df.head(args.limit).copy()

    print("=========================================================")
    print(f"Using manifest: {args.input_csv}")
    print(f"Rows selected: {len(df)}")
    print(f"IMG output folder: {args.img_dir}")
    print(f"XML output folder: {args.xml_dir}")
    print("=========================================================")

    with sync_playwright() as p:
        # Launch Chromium.
        browser = p.chromium.launch(headless=args.headless)

        # Create a clean browser context.
        context = browser.new_context()

        # One page is enough for sequential downloading.
        page = context.new_page()

        for i, row in df.iterrows():
            img_uri = row["uri"]
            xml_uri = row["label_uri"]

            img_url = atlas_uri_to_download_url(img_uri)
            xml_url = atlas_uri_to_download_url(xml_uri)

            img_name = os.path.basename(img_uri)
            xml_name = os.path.basename(xml_uri)

            img_output_path = os.path.join(args.img_dir, img_name)
            xml_output_path = os.path.join(args.xml_dir, xml_name)

            print("\n---------------------------------------------------------")
            print(f"Downloading sample {i + 1} / {len(df)}")
            print(f"IMG : {img_name}")
            print(f"XML : {xml_name}")

            # Download IMG through browser engine.
            try:
                img_bytes = fetch_binary_with_browser(page, img_url)
                save_bytes(img_output_path, img_bytes)
                print("IMG success: True")
            except Exception as e:
                print(f"IMG success: False")
                print(f"IMG error: {e}")

            # Download XML through browser engine.
            try:
                xml_bytes = fetch_binary_with_browser(page, xml_url)
                save_bytes(xml_output_path, xml_bytes)
                print("XML success: True")
            except Exception as e:
                print(f"XML success: False")
                print(f"XML error: {e}")

        context.close()
        browser.close()

    print("\n=========================================================")
    print("Browser-based sample download complete.")
    print("=========================================================")


if __name__ == "__main__":
    main()