#!/usr/bin/env python3
"""
Simple script to download the project images, and page texts, for a project
at pgdp.net. The script will only download data that doesn't already exist
so if it fails (due to network connection, or reaching the rate limit) it
can be re-run over the same set of projects and it will pick up where it left
off.

Assumes your API key is available in the environment at API_KEY.

Requires requests:
    pip install requests
"""

import logging
import json
import os

import requests


class DPDownloader:
    DP_API_BASE = "https://www.pgdp.net/api/v1"

    def __init__(self, api_key, base_dir):
        self.api_key = api_key

        if not os.path.isdir(base_dir):
            raise RuntimeError(f"{base_dir} is not a valid directory")
        self.base_dir = base_dir

        # set up our session with the API key
        self.session = requests.Session()
        self.session.headers = {"X-API-KEY": self.api_key}

    def get(self, endpoint):
        url = f"{self.DP_API_BASE}/{endpoint}"
        response = self.session.get(url)
        if not response.ok:
            raise RuntimeError(response.json()["error"])
        return response.json()

    def download_project_pages(self, projectid, roundid):
        download_dir = os.path.join(self.base_dir, projectid)

        logging.info(f"Downloading project {projectid} to {download_dir}")

        os.makedirs(download_dir, exist_ok=True)

        # download the page images
        project_pages = self.get(f"/projects/{projectid}/pages")
        for page in project_pages:
            image_name = page["image"]
            filename = os.path.join(download_dir, image_name)

            # if the file already exists, skip downloading it again
            if os.path.isfile(filename):
                continue

            with open(filename, "wb") as fileobj:
                logging.debug(f"Downloading {image_name}")
                fileobj.write(self.session.get(page["image_url"]).content)

        # download the page texts if not done already
        filename = os.path.join(download_dir, "pages.json")
        if not os.path.isfile(filename):
            texts = {}
            for page in project_pages:
                logging.debug(f"Fetching page text for {image_name}")
                image_name = page["image"]
                page = self.get(
                    f"projects/{projectid}/pages/{image_name}/pagerounds/{roundid}"
                )
                texts[image_name] = page["text"]

            with open(filename, "wt") as fileobj:
                json.dump(texts, fileobj, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    projects = [
        "projectID6737b15d33ff3",
        "projectID63ac684a641d4",
        "projectID629292e7559a8",
        "projectID63ac6757567bd",
        "projectID67658de495d0c",
        "projectID66c62fca99a93",
    ]

    api_key = os.environ.get("API_KEY")
    downloader = DPDownloader(api_key, "./output")
    for projectid in projects:
        downloader.download_project_pages(projectid, "P3")
