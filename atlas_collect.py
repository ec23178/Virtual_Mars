import os
import csv
import time
import argparse
from urllib.parse import urlencode
from playwright.sync_api import sync_playwright

# Base Atlas search page.
ATLAS_BASE_URL = "https://pds-imaging.jpl.nasa.gov/tools/atlas/search"


def build_atlas_url(spacecraft, instrument):
    # Build a query string that preloads the Atlas filters.
    query_params = {
        "gather.common.spacecraft": spacecraft,
        "gather.common.instrument": instrument,
    }

    return f"{ATLAS_BASE_URL}?{urlencode(query_params)}"


def extract_fields_from_card_text(card_text):
    # Start with empty fields so every row has the same structure.
    row = {
        "product_id": "",
        "timestamp": "",
        "mission": "",
        "spacecraft": "",
        "instrument": "",
        "product_type": "",
        "target": "",
        "bundle": "",
        "collection": "",
        "size": "",
        "raw_text": card_text.strip(),
    }

    # Split the visible card text into clean non-empty lines.
    lines = [line.strip() for line in card_text.splitlines() if line.strip()]

    # Stop early if the card text is empty.
    if len(lines) == 0:
        return row

    # In Atlas list view, the first line is usually the product ID.
    row["product_id"] = lines[0]

    # The second line is usually the timestamp.
    if len(lines) > 1:
        row["timestamp"] = lines[1]

    # These labels appear in the visible card text.
    labels = {
        "Mission:": "mission",
        "Spacecraft:": "spacecraft",
        "Instrument:": "instrument",
        "Product Type:": "product_type",
        "Targets:": "target",
        "Bundle:": "bundle",
        "Collection:": "collection",
        "Size:": "size",
    }

    for i, line in enumerate(lines):
        for label, key in labels.items():
            if line == label and i + 1 < len(lines):
                row[key] = lines[i + 1]

    return row


def collect_visible_cards(page, spacecraft, instrument):
    # Find visible blocks that contain typical result-card labels.
    blocks = page.locator("div").filter(has_text="Mission:").all()

    rows = []

    for block in blocks:
        try:
            text = block.inner_text(timeout=2000).strip()
        except Exception:
            continue

        # Skip empty blocks.
        if not text:
            continue

        # Only keep blocks that look like actual Atlas result cards.
        if "Spacecraft:" in text and "Instrument:" in text and "Product Type:" in text:
            row = extract_fields_from_card_text(text)
            row["requested_spacecraft"] = spacecraft
            row["requested_instrument"] = instrument
            rows.append(row)

    # Remove duplicates using product_id.
    unique_rows = {}
    for row in rows:
        product_id = row["product_id"]
        if product_id:
            unique_rows[product_id] = row

    return list(unique_rows.values())


def main():
    # Set up command-line arguments.
    parser = argparse.ArgumentParser(
        description="Open NASA Atlas with preloaded filters and scrape long-scroll results into a CSV."
    )
    parser.add_argument(
        "--spacecraft",
        required=True,
        help="Spacecraft filter value, for example: curiosity"
    )
    parser.add_argument(
        "--instrument",
        required=True,
        help="Instrument filter value, for example: MAST_LEFT or MAST_RIGHT"
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to the CSV file where scraped Atlas results will be saved."
    )
    parser.add_argument(
        "--max-scrolls",
        type=int,
        default=20,
        help="Maximum number of downward scrolls to attempt."
    )
    args = parser.parse_args()

    # Make sure the output folder already exists.
    output_dir = os.path.dirname(args.output_csv)
    if output_dir and not os.path.isdir(output_dir):
        raise FileNotFoundError(
            f"Output folder does not exist: {output_dir}"
        )

    # Build the Atlas URL using the terminal arguments.
    atlas_url = build_atlas_url(args.spacecraft, args.instrument)

    # This dictionary will hold all unique rows keyed by product_id.
    all_rows = {}

    with sync_playwright() as p:
        # Open a visible Chromium browser so you can interact with Atlas.
        browser = p.chromium.launch(headless=False)
        page = browser.new_page(viewport={"width": 1600, "height": 1000})

        print("\nOpening Atlas with preloaded filters...")
        print(atlas_url)
        page.goto(atlas_url, wait_until="domcontentloaded", timeout=120000)

        print("\n=========================================================")
        print("CHECK THE BROWSER WINDOW BEFORE PRESSING ENTER")
        print("=========================================================")
        print(f"Spacecraft should already be set to: {args.spacecraft}")
        print(f"Instrument should already be set to: {args.instrument}")
        print("Now do the remaining manual steps:")
        print("1. Make sure there is NO text search set")
        print("2. Switch to LIST view")
        print("3. Sort by START_TIME")
        print("4. If you can find Product Type, choose RDR")
        print("5. If you can find a date filter, use 2015-05-20 to 2015-05-25")
        print("=========================================================\n")

        input("When that is done, press ENTER here in the terminal... ")

        # Give the page a small pause after you finish setting filters.
        time.sleep(2)

        # Keep track of whether new cards are still loading.
        no_growth_rounds = 0

        for scroll_number in range(1, args.max_scrolls + 1):
            print(f"\nCollecting visible cards before scroll {scroll_number}...")

            page.wait_for_timeout(2500)

            current_rows = collect_visible_cards(
                page,
                spacecraft=args.spacecraft,
                instrument=args.instrument
            )

            before_count = len(all_rows)

            for row in current_rows:
                product_id = row["product_id"]
                if product_id:
                    all_rows[product_id] = row

            after_count = len(all_rows)

            print(f"Unique results collected so far: {after_count}")

            # If no new rows appeared, count this as a no-growth round.
            if after_count == before_count:
                no_growth_rounds += 1
            else:
                no_growth_rounds = 0

            # Stop if the page stopped loading new result cards repeatedly.
            if no_growth_rounds >= 3:
                print("No new results appeared after several scrolls. Stopping.")
                break

            # Scroll down to trigger more lazy-loaded results.
            page.mouse.wheel(0, 5000)
            page.wait_for_timeout(2500)

        browser.close()

    final_rows = list(all_rows.values())

    # Stop if nothing was collected.
    if len(final_rows) == 0:
        print("\nNo rows were collected.")
        return

    # Save the raw manifest CSV.
    fieldnames = [
        "requested_spacecraft",
        "requested_instrument",
        "product_id",
        "timestamp",
        "mission",
        "spacecraft",
        "instrument",
        "product_type",
        "target",
        "bundle",
        "collection",
        "size",
        "raw_text",
    ]

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    print("\n=========================================================")
    print(f"Saved {len(final_rows)} unique results to:")
    print(args.output_csv)
    print("=========================================================\n")


if __name__ == "__main__":
    main()