"""
19_extract_respiratory_pathogen_signals.py

Extract weekly influenza and COVID-19 positivity signals from
Singapore Weekly Infectious Disease Bulletin PDFs.

Primary study period:
    2024 epidemiological week 9
    through
    2026 epidemiological week 41

Variables extracted:
    - Influenza positivity among community ILI samples (%)
    - COVID-19 positivity among community ARI samples (%)

Values reported as "<X%" are treated as left-censored.
For the main modelling dataset, the numerical value is set to X/2,
while a censoring flag is retained.

Example:
    <1% -> 0.5
    censored = True
"""

from pathlib import Path
import logging
import re

import pandas as pd
from pypdf import PdfReader


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PDF_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "singapore_surveillance"
    / "weekly_bulletins"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "respiratory_pathogen_signals.csv"
)

LOG_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "respiratory_pathogen_extraction_log.csv"
)


# ============================================================
# STUDY PERIOD
# ============================================================

START_YEAR = 2024
START_WEEK = 9

END_YEAR = 2026
END_WEEK = 41


# ============================================================
# REDUCE NON-FATAL PDF WARNINGS
# ============================================================

logging.getLogger("pypdf").setLevel(logging.ERROR)


# ============================================================
# REGEX PATTERNS
# ============================================================

INFLUENZA_PATTERNS = [

    re.compile(
        r"(?:overall\s+)?"
        r"positivity rate for influenza "
        r"among ILI samples"
        r".{0,250}?"
        r"(?:was|is)\s*"
        r"(<\s*)?"
        r"([0-9]+(?:\.[0-9]+)?)\s*%",
        re.IGNORECASE,
    ),

]


COVID_PATTERNS = [

    re.compile(
        r"positivity rate for COVID[- ]?19 "
        r"among ARI samples"
        r".{0,250}?"
        r"(?:was|is)\s*"
        r"(<\s*)?"
        r"([0-9]+(?:\.[0-9]+)?)\s*%",
        re.IGNORECASE,
    ),

]


# ============================================================
# HELPERS
# ============================================================

def parse_year_week(filename):
    """
    Extract year and epidemiological week from filenames such as:

        2024_week_09.pdf
        2026_week_41.pdf
    """

    match = re.fullmatch(
        r"(\d{4})_week_(\d+)\.pdf",
        filename,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None, None

    return (
        int(match.group(1)),
        int(match.group(2)),
    )


def within_study_period(year, week):
    """
    Return True only for observations between:
        2024 Week 9 and 2026 Week 41.
    """

    if year < START_YEAR:
        return False

    if year > END_YEAR:
        return False

    if (
        year == START_YEAR
        and week < START_WEEK
    ):
        return False

    if (
        year == END_YEAR
        and week > END_WEEK
    ):
        return False

    return True


def extract_pdf_text(pdf_path):
    """
    Extract and normalize text from a bulletin PDF.
    """

    reader = PdfReader(
        str(pdf_path)
    )

    pages = []

    for page in reader.pages:

        try:
            text = page.extract_text() or ""

        except Exception:
            text = ""

        pages.append(text)

    text = " ".join(pages)

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def extract_percentage(text, patterns):
    """
    Extract a percentage measure.

    Returns:
        modelling_value
        reported_threshold
        censored
        reported_text
        matched_text

    If source reports:
        <1%

    then:
        reported_threshold = 1.0
        modelling_value = 0.5
        censored = True
    """

    for pattern in patterns:

        match = pattern.search(text)

        if match is None:
            continue

        censored = (
            match.group(1)
            is not None
        )

        threshold = float(
            match.group(2)
        )

        if censored:
            modelling_value = (
                threshold / 2.0
            )

            reported_text = (
                f"<{threshold:g}%"
            )

        else:
            modelling_value = threshold

            reported_text = (
                f"{threshold:g}%"
            )

        return {
            "value":
                modelling_value,

            "reported_threshold":
                threshold,

            "censored":
                censored,

            "reported_text":
                reported_text,

            "matched_text":
                match.group(0),
        }

    return {
        "value":
            None,

        "reported_threshold":
            None,

        "censored":
            False,

        "reported_text":
            None,

        "matched_text":
            None,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("RESPIRATORY PATHOGEN SIGNAL EXTRACTION")
    print("=" * 72)

    if not PDF_DIR.exists():

        raise FileNotFoundError(
            f"PDF directory not found:\n{PDF_DIR}"
        )

    pdf_files = sorted(
        PDF_DIR.glob("*.pdf")
    )

    records = []
    log_records = []

    eligible_files = []

    for path in pdf_files:

        year, week = parse_year_week(
            path.name
        )

        if (
            year is None
            or week is None
        ):
            continue

        if within_study_period(
            year,
            week,
        ):
            eligible_files.append(
                (
                    year,
                    week,
                    path,
                )
            )

    eligible_files = sorted(
        eligible_files,
        key=lambda x: (
            x[0],
            x[1],
        ),
    )

    print(
        f"\nEligible bulletins: "
        f"{len(eligible_files)}"
    )

    for index, (
        year,
        week,
        pdf_path,
    ) in enumerate(
        eligible_files,
        start=1,
    ):

        print(
            f"[{index}/{len(eligible_files)}] "
            f"{year} Week {week:02d}"
        )

        try:

            text = extract_pdf_text(
                pdf_path
            )

            extraction_status = (
                "success"
            )

        except Exception as exc:

            text = ""

            extraction_status = (
                f"pdf_error: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        influenza = extract_percentage(
            text,
            INFLUENZA_PATTERNS,
        )

        covid = extract_percentage(
            text,
            COVID_PATTERNS,
        )

        record = {

            "year":
                year,

            "epi_week":
                week,

            "filename":
                pdf_path.name,

            "influenza_positivity":
                influenza["value"],

            "influenza_censored":
                influenza["censored"],

            "influenza_reported_threshold":
                influenza[
                    "reported_threshold"
                ],

            "influenza_reported_text":
                influenza[
                    "reported_text"
                ],

            "covid_positivity":
                covid["value"],

            "covid_censored":
                covid["censored"],

            "covid_reported_threshold":
                covid[
                    "reported_threshold"
                ],

            "covid_reported_text":
                covid[
                    "reported_text"
                ],
        }

        records.append(
            record
        )

        log_records.append(
            {
                "year":
                    year,

                "epi_week":
                    week,

                "filename":
                    pdf_path.name,

                "pdf_status":
                    extraction_status,

                "influenza_found":
                    influenza[
                        "value"
                    ]
                    is not None,

                "influenza_matched_text":
                    influenza[
                        "matched_text"
                    ],

                "covid_found":
                    covid[
                        "value"
                    ]
                    is not None,

                "covid_matched_text":
                    covid[
                        "matched_text"
                    ],
            }
        )

    # --------------------------------------------------------
    # DATAFRAMES
    # --------------------------------------------------------

    df = pd.DataFrame(
        records
    )

    log_df = pd.DataFrame(
        log_records
    )

    if df.empty:

        raise RuntimeError(
            "No observations were extracted."
        )

    df = (
        df
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

    # --------------------------------------------------------
    # QC
    # --------------------------------------------------------

    duplicate_count = int(
        df.duplicated(
            subset=[
                "year",
                "epi_week",
            ]
        ).sum()
    )

    missing_influenza = int(
        df[
            "influenza_positivity"
        ].isna().sum()
    )

    missing_covid = int(
        df[
            "covid_positivity"
        ].isna().sum()
    )

    invalid_influenza = df[
        (
            df["influenza_positivity"]
            < 0
        )
        |
        (
            df["influenza_positivity"]
            > 100
        )
    ]

    invalid_covid = df[
        (
            df["covid_positivity"]
            < 0
        )
        |
        (
            df["covid_positivity"]
            > 100
        )
    ]

    influenza_censored_count = int(
        df[
            "influenza_censored"
        ].sum()
    )

    covid_censored_count = int(
        df[
            "covid_censored"
        ].sum()
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    log_df.to_csv(
        LOG_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print(
        "\n" + "=" * 72
    )

    print(
        "EXTRACTION SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        f"Rows extracted: "
        f"{len(df)}"
    )

    print(
        f"Duplicate year/week rows: "
        f"{duplicate_count}"
    )

    print(
        f"Missing influenza values: "
        f"{missing_influenza}"
    )

    print(
        f"Missing COVID values: "
        f"{missing_covid}"
    )

    print(
        f"Influenza censored values: "
        f"{influenza_censored_count}"
    )

    print(
        f"COVID censored values: "
        f"{covid_censored_count}"
    )

    print(
        f"Invalid influenza values: "
        f"{len(invalid_influenza)}"
    )

    print(
        f"Invalid COVID values: "
        f"{len(invalid_covid)}"
    )

    # --------------------------------------------------------
    # YEARLY COUNTS
    # --------------------------------------------------------

    print(
        "\n" + "=" * 72
    )

    print(
        "OBSERVATIONS BY YEAR"
    )

    print(
        "=" * 72
    )

    yearly = (
        df
        .groupby("year")
        .agg(
            weeks=(
                "epi_week",
                "size",
            ),
            influenza_available=(
                "influenza_positivity",
                "count",
            ),
            covid_available=(
                "covid_positivity",
                "count",
            ),
        )
    )

    print(
        yearly.to_string()
    )

    # --------------------------------------------------------
    # MISSING ROWS
    # --------------------------------------------------------

    missing_rows = df[
        df[
            [
                "influenza_positivity",
                "covid_positivity",
            ]
        ]
        .isna()
        .any(
            axis=1
        )
    ]

    if not missing_rows.empty:

        print(
            "\n" + "=" * 72
        )

        print(
            "ROWS REQUIRING MANUAL REVIEW"
        )

        print(
            "=" * 72
        )

        print(
            missing_rows[
                [
                    "year",
                    "epi_week",
                    "filename",
                    "influenza_positivity",
                    "covid_positivity",
                ]
            ]
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # RECENT VALIDATION ROWS
    # --------------------------------------------------------

    print(
        "\n" + "=" * 72
    )

    print(
        "2026 WEEKS 39-41 CHECK"
    )

    print(
        "=" * 72
    )

    recent = df[
        (df["year"] == 2026)
        &
        (df["epi_week"].isin(
            [39, 40, 41]
        ))
    ]

    print(
        recent[
            [
                "year",
                "epi_week",
                "influenza_positivity",
                "influenza_censored",
                "covid_positivity",
                "covid_censored",
                "covid_reported_text",
            ]
        ]
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # OUTPUT PATHS
    # --------------------------------------------------------

    print(
        "\n" + "=" * 72
    )

    print(
        "OUTPUT FILES"
    )

    print(
        "=" * 72
    )

    print(
        f"\nPathogen dataset:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"\nExtraction log:\n"
        f"{LOG_FILE}"
    )


if __name__ == "__main__":
    main()