import os
import json
import argparse
import requests
import pandas as pd


# Fixed project defaults for the Maria Pass search.
# These are the values we currently want every time unless we later discover
# we need to widen the date range or paginate further.
DEFAULT_SPACECRAFT = "curiosity"
DEFAULT_START_TIME = "2015-05-22T00:00:00Z"
DEFAULT_END_TIME = "2015-05-24T23:59:59Z"
DEFAULT_PRODUCT_TYPE = "EDR"
DEFAULT_SIZE = 500
DEFAULT_OFFSET = 0
DEFAULT_INSTRUMENT = "MAST_LEFT"

# Atlas Search API endpoint.
API_URL = "https://pds-imaging.jpl.nasa.gov/api/search/atlas/_search"


def build_query(spacecraft, instrument, start_time, end_time, size, offset, product_type):
    # Build the Atlas Elasticsearch-style query body.
    must_clauses = [
        {
            "match": {
                "gather.common.spacecraft": spacecraft
            }
        },
        {
            "match": {
                "gather.common.instrument": instrument
            }
        },
        {
            "exists": {
                "field": "gather"
            }
        },
        {
            "range": {
                "gather.time.start_time": {
                    "gte": start_time,
                    "lte": end_time
                }
            }
        }
    ]

    # Add product type filtering only if requested.
    # For your raw Bayer pipeline this should be EDR.
    if product_type is not None:
        must_clauses.append(
            {
                "match": {
                    "gather.common.product_type": product_type
                }
            }
        )

    body = {
        "query": {
            "bool": {
                "must": must_clauses
            }
        },
        # Sort from oldest to newest inside the chosen date range.
        "sort": [
            {
                "gather.time.start_time": "asc"
            }
        ],
        # Number of rows to return.
        "size": size,
        # Pagination offset.
        "from": offset,
        # Only request fields we actually need for the manifest.
        "_source": [
            "uri",
            "archive.name",
            "gather.common.spacecraft",
            "gather.common.mission",
            "gather.common.instrument",
            "gather.common.product_type",
            "gather.common.kind",
            "gather.common.target",
            "gather.time.start_time",
            "gather.pds_archive.collection_id",
            "gather.pds_archive.bundle_id",
            "gather.pds_archive.related"
        ]
    }

    return body


def send_query(body):
    # Send the POST request to Atlas Search API.
    response = requests.post(
        API_URL,
        headers={
            "Content-Type": "application/json",
            "accept": "application/json"
        },
        json=body,
        timeout=120
    )

    response.raise_for_status()
    return response.json()


def safe_get(d, keys, default=None):
    # Safely walk through a nested dictionary.
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def flatten_list_or_value(value):
    # Convert lists into a readable CSV-friendly string.
    if isinstance(value, list):
        return ",".join(str(v) for v in value)

    if value is None:
        return ""

    return str(value)


def flatten_hit(hit):
    # Convert one API hit into one flat manifest row.
    source = hit.get("_source", {})

    row = {
        "uri": source.get("uri", ""),
        "archive_name": safe_get(source, ["archive", "name"], ""),
        "spacecraft": flatten_list_or_value(
            safe_get(source, ["gather", "common", "spacecraft"], "")
        ),
        "mission": flatten_list_or_value(
            safe_get(source, ["gather", "common", "mission"], "")
        ),
        "instrument": flatten_list_or_value(
            safe_get(source, ["gather", "common", "instrument"], "")
        ),
        "product_type": flatten_list_or_value(
            safe_get(source, ["gather", "common", "product_type"], "")
        ),
        "kind": flatten_list_or_value(
            safe_get(source, ["gather", "common", "kind"], "")
        ),
        "target": flatten_list_or_value(
            safe_get(source, ["gather", "common", "target"], "")
        ),
        "start_time": safe_get(source, ["gather", "time", "start_time"], ""),
        "collection_id": safe_get(
            source,
            ["gather", "pds_archive", "collection_id"],
            ""
        ),
        "bundle_id": safe_get(
            source,
            ["gather", "pds_archive", "bundle_id"],
            ""
        ),
        "label_uri": safe_get(
            source,
            ["gather", "pds_archive", "related", "label", "uri"],
            ""
        ),
        "browse_uri": safe_get(
            source,
            ["gather", "pds_archive", "related", "browse", "uri"],
            ""
        )
    }

    return row


def ensure_parent_dir(path):
    # Create the parent folder if it does not already exist.
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    # Set up command-line arguments.
    parser = argparse.ArgumentParser(
        description="Query the PDS Atlas API and save a manifest of Mars image products."
    )
    parser.add_argument(
        "--instrument",
        default=DEFAULT_INSTRUMENT,
        help="Atlas instrument value. Default is MAST_LEFT."
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to save the raw JSON response."
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to save the flattened CSV manifest."
    )
    args = parser.parse_args()

    # Build the request body using the fixed project defaults.
    body = build_query(
        spacecraft=DEFAULT_SPACECRAFT,
        instrument=args.instrument,
        start_time=DEFAULT_START_TIME,
        end_time=DEFAULT_END_TIME,
        size=DEFAULT_SIZE,
        offset=DEFAULT_OFFSET,
        product_type=DEFAULT_PRODUCT_TYPE
    )

    print("Sending Atlas API request...")

    # Send request and parse JSON.
    response_json = send_query(body)

    # Pull out result rows.
    hits = response_json.get("hits", {}).get("hits", [])

    # Stop early if nothing was returned.
    if not hits:
        print("No results returned from Atlas API.")
        return

    # Flatten API hits into manifest rows.
    rows = [flatten_hit(hit) for hit in hits]
    df = pd.DataFrame(rows)

    # Create folders if needed.
    ensure_parent_dir(args.output_json)
    ensure_parent_dir(args.output_csv)

    # Save the raw JSON response.
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(response_json, f, indent=2)

    # Save the clean CSV manifest.
    df.to_csv(args.output_csv, index=False)

    print("=========================================================")
    print(f"Instrument used: {args.instrument}")
    print(f"Saved raw JSON to: {args.output_json}")
    print(f"Saved CSV manifest to: {args.output_csv}")
    print(f"Returned rows: {len(df)}")
    print("=========================================================")


if __name__ == "__main__":
    main()