"""
04_collect_surveillance_sources.py

Build an index of official Singapore Weekly Infectious Diseases
Bulletin links from the Communicable Diseases Agency (CDA).

This script creates a source index only.
It does not download the PDF files.

Output:
    data/interim/surveillance_source_index.csv
"""

from pathlib import Path
from urllib.parse import urljoin
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "surveillance_source_index.csv"
)

YEARS = range(2019, 2027)

BASE_URL = (
    "https://www.cda.gov.sg/resources/"
    "weekly-infectious-diseases-bulletin-{year}/"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 respiratory-multiscale-risk-model "
        "academic research"
    )
}


def collect_year(year):
    """
    Collect weekly bulletin links from one CDA annual archive.
    """

    page_url = BASE_URL.format(year=year)

    print(f"\nChecking {year}:")
    print(page_url)

    response = requests.get(
        page_url,
        headers=HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    records = []

    # Extract week number independently of the year contained
    # in the displayed link label.
    #
    # This is necessary because the official 2022 archive has
    # a Week 38 entry whose displayed label contains the wrong year.
    pattern = re.compile(
    r"week[_\s-]*(\d{1,2})",
    re.IGNORECASE,
)

    for link in soup.find_all("a", href=True):

        label = link.get_text(
            " ",
            strip=True,
        )

        href = link["href"]

        match = pattern.search(label)

        if not match:
            continue

        epi_week = int(
            match.group(1)
        )

        if epi_week < 1 or epi_week > 53:
            continue

        bulletin_url = urljoin(
            page_url,
            href,
        )

        records.append(
            {
                "year": year,
                "epi_week": epi_week,
                "label": label,
                "pdf_url": bulletin_url,
                "archive_url": page_url,
            }
        )

    # Remove duplicate year/week entries.
    unique = {}

    for record in records:

        key = (
            record["year"],
            record["epi_week"],
        )

        if key not in unique:
            unique[key] = record

    records = list(
        unique.values()
    )

    records.sort(
        key=lambda x: x["epi_week"]
    )

    print(
        f"Found {len(records)} bulletin links."
    )

    return records


def main():

    print("=" * 70)
    print(
        "SINGAPORE WEEKLY BULLETIN SOURCE COLLECTION"
    )
    print("=" * 70)

    all_records = []

    for year in YEARS:

        try:

            records = collect_year(year)

            all_records.extend(records)

        except requests.RequestException as exc:

            print(
                f"WARNING: Could not collect "
                f"{year}: {exc}"
            )

    if not all_records:

        raise RuntimeError(
            "No bulletin links were collected."
        )

    df = pd.DataFrame(
        all_records
    )

    df = (
        df
        .sort_values(
            ["year", "epi_week"]
        )
        .drop_duplicates(
            subset=[
                "year",
                "epi_week",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 70)
    print("SOURCE INDEX COMPLETE")
    print("=" * 70)

    print(
        f"\nTotal bulletin links: "
        f"{len(df)}"
    )

    print(
        "\nBulletins found by year:"
    )

    counts = (
        df
        .groupby("year")
        .size()
        .rename("bulletins")
    )

    print(counts)

    print(
        "\nWeek range by year:"
    )

    summary = (
        df
        .groupby("year")["epi_week"]
        .agg(
            ["min", "max", "count"]
        )
    )

    print(summary)

    print(
        f"\nSaved source index to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()