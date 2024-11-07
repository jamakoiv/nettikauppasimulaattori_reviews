#! /usr/bin/python

import argparse
import requests
import re
import sys
import json

from concurrent.futures import ThreadPoolExecutor
from pymongo import MongoClient
from urllib.parse import quote_plus

# Constants and Configurations
SIGNATURE = "reviewText"  # Signature to look for in the JSON response
DATABASE = "verkkokauppa"
COLLECTION = "items_phones"

# Read credentials
USER = "jamakoiv"
PASS = open("passwd").read().strip()
CONN_STR = (
    f"mongodb+srv://{quote_plus(USER)}:{quote_plus(PASS)}"
    "@cosmos-mongo-testi.mongocluster.cosmos.azure.com/"
    "?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000&wtimeoutMS=0"
)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Extract and upload JSON data from URLs.")
parser.add_argument("urls", metavar="URL", nargs="+", help="URLs to process")
parser.add_argument("--save-json", action="store_true", help="Save JSON data to files")
parser.add_argument("--upload", action="store_true", help="Upload JSON data to MongoDB")
parser.add_argument("--collection", help="Collection name for upload.")
args = parser.parse_args()

if args.collection:
    COLLECTION = args.collection

# MongoDB client
client = MongoClient(CONN_STR)
db = client[DATABASE]
collection = db[COLLECTION]


def get_number_of_reviews(json_dict: dict) -> tuple[int, int, int]:
    """
    Extract number of total reviews, number of reviews per page,
    and total number of pages.

    By default the main page URL only gives one page of reviews, so we get number of pages and use that info to create extra URLs for the remaining pages.
    """

    stem = list(json_dict["products"]["reviewsByPid"].values())[0][0]

    n = int(stem["totalItems"])
    per_page = len(stem["byPage"]["1"])

    if n % per_page == 0:
        pages = int(n / per_page)
    else:
        pages = int(n / per_page) + 1

    return n, per_page, pages


def process_url(url: str):
    """
    Retrieve URL.
    """
    print(f"Processing URL: {url}")
    url = re.sub(r"#.*$", "", url)  # Strip hash params from URL

    try:
        # Fetch the page content
        response = requests.get(url)
        response.raise_for_status()

    except requests.exceptions.MissingSchema as e:
        print(e)
        return
    except requests.exceptions.ConnectionError as e:
        print(e)
        return

    return response


def extract_json_str(response: requests.Response):
    """
    Match and return the line containing SIGNATURE. This should give us single line containing the JSON we want.
    """
    json_matches = re.findall(f'{{.*"{SIGNATURE}".*}}', response.text)

    if not json_matches:
        print(f"Error: {SIGNATURE} not found in {url}")
        sys.exit(10)

    assert len(json_matches) == 1, "More than one JSON match, something borked."

    return json_matches[0]


def save_json(json_str: str, url: str):
    """
    Save the JSON to a file. Filename is automatically create from the URL.
    """
    json_filename = re.sub(r"[^\w\-_\.]", "_", url) + ".json"
    with open(json_filename, "w") as f:
        f.write(json_str)

    print(f"JSON saved to {json_filename}")


mongo_upload_data = []
url_extra_pages = []

# Process URLs from the command line
with ThreadPoolExecutor(max_workers=8) as exec:
    responses = exec.map(process_url, args.urls)

for resp, url in zip(responses, args.urls):
    json_str = extract_json_str(resp)
    json_dict = json.loads(json_str)
    mongo_upload_data.append(json_dict)

    # Check if there are more than one review page.
    n_reviews, reviews_per_page, n_pages = get_number_of_reviews(json_dict)
    for page in range(2, n_pages + 1):
        url_extra_pages.append(f"{url}reviews?page={page}")

    if args.save_json:
        save_json(json_str, url)


# Process URLs from extra pages
with ThreadPoolExecutor(max_workers=8) as exec:
    responses = exec.map(process_url, url_extra_pages)

for resp, url in zip(responses, args.urls):
    json_str = extract_json_str(resp)
    json_dict = json.loads(json_str)
    mongo_upload_data.append(json_dict)

    if args.save_json:
        save_json(json_str, url)

# Upload data if requested
if args.upload:
    # Slice into more manageable chunks for upload.
    # Technically mongodb insertMany will handle any amount of data, but
    # takes long time without any progress info.

    print("Starting upload to MongoDB.")
    N_slice = 30

    for k, data in enumerate(
        [
            mongo_upload_data[i : i + N_slice]
            for i in range(0, len(mongo_upload_data), N_slice)
        ]
    ):
        collection.insert_many(data)
        print("Uploaded records {} to {}".format(k * N_slice, k * N_slice + N_slice))
    print("Done uploading data.")

if not __name__ == "__main__":
    client.close()
