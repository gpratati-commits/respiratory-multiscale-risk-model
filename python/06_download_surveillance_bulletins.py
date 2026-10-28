"""
06_download_surveillance_bulletins.py

Download Singapore Weekly Infectious Diseases Bulletins listed in:

    data/interim/surveillance_source_index.csv

Features
--------
- Restart-safe: existing PDFs are skipped.
- Downloads to a temporary .part file first.
- Checks that downloaded content is actually a PDF.
- Records download status in a CSV log.
- Supports a small test run with --limit.
"""

from pathlib import Path
import argparse
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_INDEX = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "surveillance_source_index.csv"
)

DOWNLOAD_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "singapore_surveillance"
    / "weekly_bulletins"
)

DOWNLOAD_LOG = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "surveillance_download_log.csv"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 respiratory-multiscale-risk-model "
        "academic research"
    )
}


def build_session():
    """
    Create a requests session with automatic retries.
    """

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=1.0,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504,
        ],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(
        max_retries=retry
    )

    session = requests.Session()

    session.mount(
        "https://",
        adapter,
    )

    session.headers.update(
        HEADERS
    )

    return session


def is_pdf(content):
    """
    Check whether downloaded bytes begin with the PDF signature.
    """

    return content.startswith(b"%PDF")


def download_one(
    session,
    year,
    epi_week,
    url,
):
    """
    Download one bulletin.

    Returns a dictionary describing the result.
    """

    filename = (
        f"{int(year)}_week_{int(epi_week):02d}.pdf"
    )

    output_path = (
        DOWNLOAD_DIR
        / filename
    )

    temporary_path = (
        DOWNLOAD_DIR
        / f"{filename}.part"
    )

    record = {
        "year": int(year),
        "epi_week": int(epi_week),
        "filename": filename,
        "url": url,
        "status": None,
        "bytes": None,
        "message": None,
    }

    if output_path.exists():

        record["status"] = "skipped_existing"
        record["bytes"] = output_path.stat().st_size
        record["message"] = "File already exists."

        return record

    try:

        response = session.get(
            url,
            timeout=60,
            allow_redirects=True,
        )

        response.raise_for_status()

        content = response.content

        if not is_pdf(content):

            record["status"] = "failed_not_pdf"
            record["bytes"] = len(content)
            record["message"] = (
                "Downloaded content did not begin "
                "with the PDF signature."
            )

            return record

        temporary_path.write_bytes(
            content
        )

        temporary_path.replace(
            output_path
        )

        record["status"] = "downloaded"
        record["bytes"] = len(content)
        record["message"] = "Download successful."

        return record

    except requests.RequestException as exc:

        record["status"] = "failed_request"
        record["message"] = str(exc)

        return record

    finally:

        if temporary_path.exists():
            temporary_path.unlink()


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Download indexed Singapore "
            "surveillance bulletins."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of index rows "
            "to process. Useful for testing."
        ),
    )

    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help=(
            "Download only one year."
        ),
    )

    args = parser.parse_args()

    print("=" * 70)
    print(
        "SINGAPORE SURVEILLANCE BULLETIN DOWNLOAD"
    )
    print("=" * 70)

    if not SOURCE_INDEX.exists():

        raise FileNotFoundError(
            f"Source index not found:\n"
            f"{SOURCE_INDEX}"
        )

    df = pd.read_csv(
        SOURCE_INDEX
    )

    required_columns = {
        "year",
        "epi_week",
        "pdf_url",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Source index is missing columns: "
            f"{sorted(missing_columns)}"
        )

    df = (
        df
        .sort_values(
            ["year", "epi_week"]
        )
        .reset_index(drop=True)
    )

    if args.year is not None:

        df = df[
            df["year"] == args.year
        ].copy()

    if args.limit is not None:

        df = df.head(
            args.limit
        ).copy()

    if df.empty:

        raise RuntimeError(
            "No bulletin rows selected."
        )

    DOWNLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\nBulletins selected: {len(df)}"
    )

    print(
        f"Download directory:\n"
        f"{DOWNLOAD_DIR}\n"
    )

    session = build_session()

    results = []

    for i, row in df.iterrows():

        year = int(
            row["year"]
        )

        epi_week = int(
            row["epi_week"]
        )

        url = row["pdf_url"]

        print(
            f"{year} week {epi_week:02d} ... ",
            end="",
            flush=True,
        )

        result = download_one(
            session=session,
            year=year,
            epi_week=epi_week,
            url=url,
        )

        results.append(
            result
        )

        print(
            result["status"]
        )

        # Small delay to avoid sending requests
        # too rapidly to the source website.
        time.sleep(0.25)

    log_df = pd.DataFrame(
        results
    )

    if DOWNLOAD_LOG.exists():

        old_log = pd.read_csv(
            DOWNLOAD_LOG
        )

        log_df = pd.concat(
            [
                old_log,
                log_df,
            ],
            ignore_index=True,
        )

        log_df = (
            log_df
            .drop_duplicates(
                subset=[
                    "year",
                    "epi_week",
                ],
                keep="last",
            )
            .sort_values(
                [
                    "year",
                    "epi_week",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    log_df.to_csv(
        DOWNLOAD_LOG,
        index=False,
    )

    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)

    print(
        log_df["status"]
        .value_counts(
            dropna=False
        )
    )

    print(
        f"\nDownload log saved to:\n"
        f"{DOWNLOAD_LOG}"
    )


if __name__ == "__main__":
    main()