"""
18_audit_respiratory_pathogen_signals.py

Audit Singapore Weekly Infectious Disease Bulletin PDFs for
respiratory-pathogen surveillance information.

This script DOES NOT create the final modelling variables.
Its purpose is to identify whether influenza, COVID-19 and RSV
signals are present and to preserve representative text snippets.

Outputs
-------
data/interim/respiratory_pathogen_signal_audit.csv

outputs/tables/respiratory_pathogen_signal_summary.csv
"""

from pathlib import Path
import logging
import re

import pandas as pd
from pypdf import PdfReader


# ============================================================
# PROJECT PATHS
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
    / "interim"
    / "respiratory_pathogen_signal_audit.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "respiratory_pathogen_signal_summary.csv"
)


# ============================================================
# SUPPRESS NON-FATAL PYPDF WARNINGS
# ============================================================

logging.getLogger("pypdf").setLevel(logging.ERROR)


# ============================================================
# SEARCH TERMS
# ============================================================

KEYWORD_GROUPS = {

    "influenza": [
        r"influenza[- ]like illness",
        r"\bILI\b",
        r"positivity rate for influenza among ILI samples",
        r"overall positivity rate for influenza among ILI samples",
        r"seasonal influenza",
        r"positive for influenza A\b",
        r"positive for influenza B\b",
        r"influenza A\(H3N2\)",
        r"influenza A\(pH1N1\)",
        r"influenza B\b",
    ],

    "covid": [
        r"COVID[- ]?19",
        r"\bCOVID\b",
        r"SARS[- ]?CoV[- ]?2",
        r"positivity rate for COVID[- ]?19 among ARI samples",
    ],

    "rsv": [
        r"respiratory syncytial virus",
        r"respiratory syncytial",
        r"\bRSV\b",
    ],
}


# Terms that must not be mistaken for seasonal/human influenza.
INFLUENZA_EXCLUSIONS = [
    r"avian influenza",
    r"haemophilus influenzae",
    r"influenzae type b",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_year_week(filename):
    """
    Parse filenames such as:
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

    return int(match.group(1)), int(match.group(2))


def normalize_text(text):
    """
    Normalize PDF text before searching.
    """

    if not text:
        return ""

    # Normalize common Unicode dashes.
    text = (
        text
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2011", "-")
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def extract_pdf_text(pdf_path):
    """
    Extract text from all readable pages of a PDF.
    """

    reader = PdfReader(str(pdf_path))

    page_text = []

    for page in reader.pages:

        try:
            text = page.extract_text() or ""

        except Exception:
            text = ""

        if text:
            page_text.append(text)

    return normalize_text(
        " ".join(page_text)
    )


def clean_snippet(text):
    """
    Make snippets easier to read in CSV output.
    """

    return normalize_text(text)


def influenza_context_is_false_positive(context):
    """
    Return True if an influenza match appears to be only
    Avian Influenza or Haemophilus influenzae rather than
    human influenza surveillance.
    """

    lower_context = context.lower()

    exclusion_found = any(
        re.search(
            pattern,
            lower_context,
            flags=re.IGNORECASE,
        )
        for pattern in INFLUENZA_EXCLUSIONS
    )

    # Human-influenza indicators that override nearby exclusions.
    human_indicator_found = any(
        phrase in lower_context
        for phrase in [
            "influenza-like illness",
            "positivity rate for influenza",
            "positive for influenza a",
            "positive for influenza b",
            "influenza a(h3n2)",
            "influenza a(ph1n1)",
        ]
    )

    return (
        exclusion_found
        and not human_indicator_found
    )


def find_keyword_information(
    text,
    pathogen,
    patterns,
):
    """
    Search text for one pathogen.

    Returns
    -------
    available : bool
    mention_count : int
    snippet : str
    """

    accepted_matches = []

    for pattern in patterns:

        for match in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):

            context_start = max(
                0,
                match.start() - 150,
            )

            context_end = min(
                len(text),
                match.end() + 350,
            )

            context = text[
                context_start:
                context_end
            ]

            if pathogen == "influenza":

                if influenza_context_is_false_positive(
                    context
                ):
                    continue

            accepted_matches.append(
                match
            )

    if not accepted_matches:

        return False, 0, ""

    accepted_matches.sort(
        key=lambda x: x.start()
    )

    first_match = accepted_matches[0]

    snippet_start = max(
        0,
        first_match.start() - 250,
    )

    snippet_end = min(
        len(text),
        first_match.end() + 700,
    )

    snippet = clean_snippet(
        text[
            snippet_start:
            snippet_end
        ]
    )

    return (
        True,
        len(accepted_matches),
        snippet,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("RESPIRATORY PATHOGEN SIGNAL AUDIT")
    print("=" * 72)

    if not PDF_DIR.exists():

        raise FileNotFoundError(
            f"PDF directory not found:\n{PDF_DIR}"
        )

    pdf_files = sorted(
        PDF_DIR.glob("*.pdf")
    )

    if not pdf_files:

        raise RuntimeError(
            f"No PDF files found in:\n{PDF_DIR}"
        )

    print(
        f"\nPDF files found: {len(pdf_files)}"
    )

    records = []

    for index, pdf_path in enumerate(
        pdf_files,
        start=1,
    ):

        year, epi_week = parse_year_week(
            pdf_path.name
        )

        if (
            year is None
            or epi_week is None
        ):

            print(
                f"Skipping unexpected filename: "
                f"{pdf_path.name}"
            )

            continue

        print(
            f"[{index}/{len(pdf_files)}] "
            f"{year} Week {epi_week:02d}"
        )

        try:

            text = extract_pdf_text(
                pdf_path
            )

            extraction_status = "success"

        except Exception as exc:

            text = ""

            extraction_status = (
                f"failed: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        record = {
            "year":
                year,

            "epi_week":
                epi_week,

            "filename":
                pdf_path.name,

            "text_extraction_status":
                extraction_status,

            "text_character_count":
                len(text),
        }

        for pathogen, patterns in (
            KEYWORD_GROUPS.items()
        ):

            (
                available,
                mention_count,
                snippet,
            ) = find_keyword_information(
                text=text,
                pathogen=pathogen,
                patterns=patterns,
            )

            record[
                f"{pathogen}_available"
            ] = available

            record[
                f"{pathogen}_mention_count"
            ] = mention_count

            record[
                f"{pathogen}_snippet"
            ] = snippet

        records.append(record)

    audit = pd.DataFrame(records)

    if audit.empty:

        raise RuntimeError(
            "Audit produced no records."
        )

    audit = (
        audit
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

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # YEARLY SUMMARY
    # ========================================================

    summary = (
        audit
        .groupby("year")
        .agg(
            bulletins=(
                "epi_week",
                "size",
            ),

            influenza_available=(
                "influenza_available",
                "sum",
            ),

            covid_available=(
                "covid_available",
                "sum",
            ),

            rsv_available=(
                "rsv_available",
                "sum",
            ),
        )
        .reset_index()
    )

    SUMMARY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # ========================================================
    # REPORT RESULTS
    # ========================================================

    print(
        "\n" + "=" * 72
    )

    print(
        "PATHOGEN AVAILABILITY BY YEAR"
    )

    print(
        "=" * 72
    )

    print(
        summary.to_string(
            index=False
        )
    )

    failed = audit[
        audit[
            "text_extraction_status"
        ]
        != "success"
    ]

    print(
        "\n" + "=" * 72
    )

    print(
        "PDF TEXT EXTRACTION SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        f"Total bulletins audited: "
        f"{len(audit)}"
    )

    print(
        f"Failed text extractions: "
        f"{len(failed)}"
    )

    # ========================================================
    # SAMPLE INFLUENZA MATCHES
    # ========================================================

    influenza_examples = audit[
        audit[
            "influenza_available"
        ]
        == True
    ]

    print(
        "\n" + "=" * 72
    )

    print(
        "SAMPLE ACCEPTED INFLUENZA MATCHES"
    )

    print(
        "=" * 72
    )

    for _, row in (
        influenza_examples
        .head(5)
        .iterrows()
    ):

        print(
            f"\n{int(row['year'])} "
            f"WEEK "
            f"{int(row['epi_week']):02d}"
        )

        print(
            row[
                "influenza_snippet"
            ]
        )

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
        f"\nDetailed audit:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"\nYearly summary:\n"
        f"{SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()