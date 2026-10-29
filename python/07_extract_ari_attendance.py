"""
07_extract_ari_attendance.py

Extract weekly average daily polyclinic attendance for
Acute Respiratory Infection (ARI) / Acute Upper Respiratory Infection
from downloaded Singapore Weekly Infectious Diseases Bulletins.

Inputs
------
data/interim/surveillance_source_index.csv

data/raw/singapore_surveillance/weekly_bulletins/

Outputs
-------
data/interim/ari_weekly_extraction.csv

data/interim/ari_extraction_log.csv

The extraction log records every bulletin so that failed or ambiguous
extractions can be inspected manually.
"""

from pathlib import Path
import re

import pandas as pd
from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_INDEX = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "surveillance_source_index.csv"
)

PDF_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "singapore_surveillance"
    / "weekly_bulletins"
)

OUTPUT_DATA = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "ari_weekly_extraction.csv"
)

OUTPUT_LOG = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "ari_extraction_log.csv"
)


# Multiple extraction patterns are used because the wording
# changed slightly across surveillance years.
# Verified manual overrides for bulletins whose PDF text
# cannot be reliably parsed automatically.
#
# Source:
# Singapore Weekly Infectious Diseases Bulletin,
# Epidemiological Week 18, 2023.
#
# The bulletin reports an average daily polyclinic
# ARI attendance of 2,797.
MANUAL_OVERRIDES = {
    (2023, 18): 2797,
}
PATTERNS = [
    (
        "table_acute_upper_respiratory_infections",
        re.compile(
            r"Acute\s+Upper\s+Respiratory\s+Infections?"
            r"\s+([0-9][0-9,]*)",
            re.IGNORECASE,
        ),
    ),
    (
        "narrative_ari_attendance",
        re.compile(
            r"average\s+daily\s+number\s+of\s+patients"
            r"\s+seeking\s+treatment\s+in\s+the\s+polyclinics"
            r"\s+for\s+ARI\s+(?:was|is)\s+([0-9][0-9,]*)",
            re.IGNORECASE,
        ),
    ),
]


def normalize_text(text):
    """
    Normalize extracted PDF text for more reliable regex matching.
    """

    text = text.replace("\u00a0", " ")

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def extract_pdf_text(pdf_path):
    """
    Extract text from the first two pages of a bulletin.

    ARI surveillance information is normally found on pages 1-2.
    """

    reader = PdfReader(
        str(pdf_path)
    )

    pages_to_read = min(
        2,
        len(reader.pages),
    )

    text_parts = []

    for page_number in range(
        pages_to_read
    ):

        page_text = (
            reader.pages[
                page_number
            ].extract_text()
            or ""
        )

        text_parts.append(
            page_text
        )

    return normalize_text(
        " ".join(text_parts)
    )


def find_ari_values(text):
    """
    Search bulletin text using all available extraction patterns.
    """

    matches = []

    for pattern_name, pattern in PATTERNS:

        for match in pattern.finditer(
            text
        ):

            value_text = (
                match.group(1)
                .replace(",", "")
            )

            try:

                value = int(
                    value_text
                )

            except ValueError:
                continue

            matches.append(
                {
                    "method": pattern_name,
                    "value": value,
                }
            )

    return matches


def process_bulletin(
    year,
    epi_week,
):
    """
    Extract ARI attendance from one bulletin.
    """

    filename = (
        f"{int(year)}_week_"
        f"{int(epi_week):02d}.pdf"
    )

    pdf_path = (
        PDF_DIR
        / filename
    )

    result = {
        "year": int(year),
        "epi_week": int(epi_week),
        "filename": filename,
        "ari_attendance": None,
        "status": None,
        "extraction_method": None,
        "candidate_values": None,
        "message": None,
    }
    override_key = (
        int(year),
        int(epi_week),
    )

    if override_key in MANUAL_OVERRIDES:

        override_value = MANUAL_OVERRIDES[
            override_key
        ]

        result["ari_attendance"] = (
            override_value
        )

        result["status"] = "success"

        result["extraction_method"] = (
            "manual_override_verified_source"
        )

        result["candidate_values"] = str(
            override_value
        )

        result["message"] = (
            "Verified manual override because "
            "PDF text extraction did not expose "
            "the ARI attendance reliably."
        )

        return result
    if not pdf_path.exists():

        result["status"] = "missing_file"
        result["message"] = (
            "PDF file does not exist."
        )

        return result

    try:

        text = extract_pdf_text(
            pdf_path
        )

    except Exception as exc:

        result["status"] = "read_error"
        result["message"] = str(exc)

        return result

    matches = find_ari_values(
        text
    )

    if not matches:

        result["status"] = (
            "pattern_not_found"
        )

        result["message"] = (
            "No ARI attendance pattern "
            "was identified."
        )

        return result

    unique_values = sorted(
        {
            match["value"]
            for match in matches
        }
    )

    methods = sorted(
        {
            match["method"]
            for match in matches
        }
    )

    result["candidate_values"] = (
        ";".join(
            str(value)
            for value in unique_values
        )
    )

    result["extraction_method"] = (
        ";".join(methods)
    )

    if len(unique_values) > 1:

        result["status"] = (
            "ambiguous_multiple_values"
        )

        result["message"] = (
            "Different ARI values were "
            "identified in the same bulletin."
        )

        return result

    value = unique_values[0]

    # Broad plausibility check.
    # This flags obvious extraction mistakes without assuming
    # a narrow expected range.
    if value < 100 or value > 20000:

        result["status"] = (
            "implausible_value"
        )

        result["ari_attendance"] = value

        result["message"] = (
            "Extracted value is outside "
            "the broad plausibility range."
        )

        return result

    result["ari_attendance"] = value

    result["status"] = "success"

    result["message"] = (
        "ARI attendance extracted successfully."
    )

    return result


def main():

    print("=" * 70)
    print(
        "WEEKLY ARI ATTENDANCE EXTRACTION"
    )
    print("=" * 70)

    if not SOURCE_INDEX.exists():

        raise FileNotFoundError(
            f"Source index not found:\n"
            f"{SOURCE_INDEX}"
        )

    if not PDF_DIR.exists():

        raise FileNotFoundError(
            f"PDF directory not found:\n"
            f"{PDF_DIR}"
        )

    source_df = pd.read_csv(
        SOURCE_INDEX
    )

    source_df = (
        source_df
        .sort_values(
            ["year", "epi_week"]
        )
        .reset_index(drop=True)
    )

    print(
        f"\nBulletins to process: "
        f"{len(source_df)}"
    )

    results = []

    for _, row in source_df.iterrows():

        year = int(
            row["year"]
        )

        epi_week = int(
            row["epi_week"]
        )

        result = process_bulletin(
            year=year,
            epi_week=epi_week,
        )

        results.append(
            result
        )

        print(
            f"{year} week "
            f"{epi_week:02d}: "
            f"{result['status']}"
        )

    log_df = pd.DataFrame(
        results
    )

    OUTPUT_LOG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_df.to_csv(
        OUTPUT_LOG,
        index=False,
    )

    successful = log_df[
        log_df["status"] == "success"
    ].copy()

    output_df = successful[
        [
            "year",
            "epi_week",
            "ari_attendance",
            "filename",
            "extraction_method",
        ]
    ].copy()

    output_df = (
        output_df
        .sort_values(
            ["year", "epi_week"]
        )
        .reset_index(drop=True)
    )

    output_df.to_csv(
        OUTPUT_DATA,
        index=False,
    )

    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)

    print(
        "\nStatus counts:"
    )

    print(
        log_df["status"]
        .value_counts(
            dropna=False
        )
    )

    print(
        f"\nSuccessful observations: "
        f"{len(output_df)}"
    )

    print(
        f"\nARI dataset saved to:\n"
        f"{OUTPUT_DATA}"
    )

    print(
        f"\nExtraction log saved to:\n"
        f"{OUTPUT_LOG}"
    )


if __name__ == "__main__":
    main()