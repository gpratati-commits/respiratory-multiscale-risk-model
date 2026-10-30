"""
18_audit_respiratory_pathogen_signals.py

Audit downloaded Singapore weekly infectious-disease bulletins for
potential respiratory-pathogen surveillance signals.

Pathogens audited:
    - Seasonal/human influenza
    - COVID-19 / SARS-CoV-2
    - RSV / respiratory syncytial virus

IMPORTANT
---------
This script does NOT yet extract numerical pathogen values.

Its purpose is to determine:
    1. Which pathogen signals appear in the bulletins.
    2. In which years/weeks they appear.
    3. What terminology is used.
    4. Whether reporting is consistent enough for later extraction.

Influenza false positives such as:
    - Avian Influenza
    - Haemophilus influenzae
    - Haemophilus influenzae type b

are explicitly excluded from the seasonal-influenza audit.

Outputs:
    data/interim/respiratory_pathogen_signal_audit.csv

    outputs/tables/respiratory_pathogen_signal_summary.csv
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
# REDUCE NON-FATAL PYPDF WARNING OUTPUT
# ============================================================

logging.getLogger("pypdf").setLevel(logging.ERROR)


# ============================================================
# PATHOGEN SEARCH TERMS
# ============================================================

KEYWORD_GROUPS = {

    # IMPORTANT:
    # Do NOT use generic r"\binfluenza\b".
    #
    # That would incorrectly match terms such as:
    # "Avian Influenza".
    "influenza": [
        r"influenza[- ]like illness",
        r"\bILI\b",
        r"seasonal influenza",
        r"influenza positivity",
        r"influenza positive",
        r"positive for influenza",
        r"percentage positive for influenza",
        r"percent positive for influenza",
        r"proportion positive for influenza",
        r"influenza A\b",
        r"influenza B\b",
    ],

    "covid": [
        r"covid[- ]?19",
        r"\bcovid\b",
        r"sars[- ]?cov[- ]?2",
    ],

    "rsv": [
        r"respiratory syncytial virus",
        r"respiratory syncytial",
        r"\bRSV\b",
    ],
}


# ============================================================
# INFLUENZA FALSE-POSITIVE TERMS
# ============================================================

INFLUENZA_EXCLUSIONS = [
    r"avian influenza",
    r"haemophilus influenzae",
    r"influenzae type b",
]


# ============================================================
# HELPER FUNCTION:
# EXTRACT YEAR AND WEEK FROM FILENAME
# ============================================================

def parse_year_week(filename):
    """
    Parse filenames such as:

        2024_week_40.pdf
        2019_week_01.pdf

    Returns:
        year, epi_week

    If filename does not match the expected pattern:
        None, None
    """

    match = re.match(
        r"(\d{4})_week_(\d+)\.pdf$",
        filename,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None, None

    year = int(
        match.group(1)
    )

    epi_week = int(
        match.group(2)
    )

    return year, epi_week


# ============================================================
# HELPER FUNCTION:
# EXTRACT TEXT FROM PDF
# ============================================================

def extract_pdf_text(pdf_path):
    """
    Extract text from every readable page of a PDF.

    Pages that produce no text are skipped.
    """

    reader = PdfReader(
        str(pdf_path)
    )

    page_texts = []

    for page in reader.pages:

        try:
            text = page.extract_text()

        except Exception:
            text = ""

        if text:
            page_texts.append(
                text
            )

    combined_text = "\n".join(
        page_texts
    )

    return combined_text


# ============================================================
# HELPER FUNCTION:
# CLEAN SNIPPET TEXT
# ============================================================

def clean_snippet(text):
    """
    Replace repeated whitespace/newlines so snippets are easy
    to inspect in the CSV and Terminal.
    """

    text = text.replace(
        "\n",
        " ",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# HELPER FUNCTION:
# CHECK WHETHER INFLUENZA CONTEXT IS A FALSE POSITIVE
# ============================================================

def influenza_context_is_excluded(context):
    """
    Return True when the local context appears to refer to
    Avian Influenza or Haemophilus influenzae rather than
    seasonal/human influenza surveillance.
    """

    for exclusion_pattern in INFLUENZA_EXCLUSIONS:

        if re.search(
            exclusion_pattern,
            context,
            flags=re.IGNORECASE,
        ):
            return True

    return False


# ============================================================
# HELPER FUNCTION:
# FIND VALID PATHOGEN MATCHES
# ============================================================

def find_keyword_information(
    text,
    pathogen,
    patterns,
):
    """
    Search PDF text for pathogen-specific terms.

    For influenza, false-positive local contexts such as
    "Avian Influenza" and "Haemophilus influenzae" are rejected.

    Returns:
        available
        mention_count
        representative_snippet
    """

    accepted_matches = []

    # --------------------------------------------------------
    # Search every pattern.
    # --------------------------------------------------------

    for pattern in patterns:

        matches = re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        for match in matches:

            # Small local context is used for deciding
            # whether this match is valid.
            context_start = max(
                0,
                match.start() - 100,
            )

            context_end = min(
                len(text),
                match.end() + 150,
            )

            local_context = (
                text[
                    context_start:
                    context_end
                ]
            )

            # -----------------------------------------------
            # Influenza-specific false-positive filtering.
            # -----------------------------------------------

            if pathogen == "influenza":

                if influenza_context_is_excluded(
                    local_context
                ):
                    continue

            accepted_matches.append(
                match
            )

    # --------------------------------------------------------
    # No acceptable matches.
    # --------------------------------------------------------

    if len(accepted_matches) == 0:

        return (
            False,
            0,
            "",
        )

    # Sort by location in PDF text.
    accepted_matches = sorted(
        accepted_matches,
        key=lambda match:
        match.start(),
    )

    # --------------------------------------------------------
    # Build a larger representative snippet around the first
    # acceptable match.
    # --------------------------------------------------------

    first_match = accepted_matches[0]

    snippet_start = max(
        0,
        first_match.start() - 200,
    )

    snippet_end = min(
        len(text),
        first_match.end() + 400,
    )

    snippet = text[
        snippet_start:
        snippet_end
    ]

    snippet = clean_snippet(
        snippet
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

    print("=" * 70)
    print("RESPIRATORY PATHOGEN SIGNAL AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Check PDF directory.
    # --------------------------------------------------------

    if not PDF_DIR.exists():

        raise FileNotFoundError(
            "Weekly bulletin directory "
            "was not found:\n"
            f"{PDF_DIR}"
        )

    # --------------------------------------------------------
    # Find PDFs.
    # --------------------------------------------------------

    pdf_files = sorted(
        PDF_DIR.glob(
            "*.pdf"
        )
    )

    if len(pdf_files) == 0:

        raise RuntimeError(
            "No PDF files were found in:\n"
            f"{PDF_DIR}"
        )

    print(
        f"\nPDF files found: "
        f"{len(pdf_files)}"
    )

    records = []

    # --------------------------------------------------------
    # Process each bulletin.
    # --------------------------------------------------------

    for number, pdf_path in enumerate(
        pdf_files,
        start=1,
    ):

        year, epi_week = (
            parse_year_week(
                pdf_path.name
            )
        )

        # Skip unexpected filenames.
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
            f"[{number}/{len(pdf_files)}] "
            f"{year} week {epi_week}"
        )

        # ----------------------------------------------------
        # Extract PDF text.
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Search for each pathogen.
        # ----------------------------------------------------

        for (
            pathogen,
            patterns,
        ) in KEYWORD_GROUPS.items():

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

        records.append(
            record
        )

    # --------------------------------------------------------
    # Convert results to DataFrame.
    # --------------------------------------------------------

    audit = pd.DataFrame(
        records
    )

    if audit.empty:

        raise RuntimeError(
            "The audit produced no usable records."
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

    # --------------------------------------------------------
    # Save detailed audit.
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Build yearly summary.
    # --------------------------------------------------------

    summary_records = []

    for year in sorted(
        audit["year"].unique()
    ):

        year_data = audit[
            audit["year"]
            == year
        ]

        successful_extractions = int(
            (
                year_data[
                    "text_extraction_status"
                ]
                == "success"
            ).sum()
        )

        summary_records.append(
            {
                "year":
                    int(year),

                "bulletins":
                    len(year_data),

                "successful_text_extractions":
                    successful_extractions,

                "influenza_available":
                    int(
                        year_data[
                            "influenza_available"
                        ].sum()
                    ),

                "covid_available":
                    int(
                        year_data[
                            "covid_available"
                        ].sum()
                    ),

                "rsv_available":
                    int(
                        year_data[
                            "rsv_available"
                        ].sum()
                    ),
            }
        )

    summary = pd.DataFrame(
        summary_records
    )

    # --------------------------------------------------------
    # Save summary.
    # --------------------------------------------------------

    SUMMARY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Print summary.
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "PATHOGEN AVAILABILITY BY YEAR"
    )

    print(
        "=" * 70
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Extraction-quality summary.
    # --------------------------------------------------------

    failed = audit[
        audit[
            "text_extraction_status"
        ]
        != "success"
    ]

    print(
        "\n" + "=" * 70
    )

    print(
        "PDF TEXT EXTRACTION SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Total bulletins audited: "
        f"{len(audit)}"
    )

    print(
        f"Failed text extractions: "
        f"{len(failed)}"
    )

    if not failed.empty:

        print(
            "\nFailed files:"
        )

        print(
            failed[
                [
                    "year",
                    "epi_week",
                    "filename",
                    "text_extraction_status",
                ]
            ]
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Display a few influenza examples to make false-positive
    # checking easier.
    # --------------------------------------------------------

    influenza_examples = audit[
        audit[
            "influenza_available"
        ]
        == True
    ]

    print(
        "\n" + "=" * 70
    )

    print(
        "SAMPLE ACCEPTED INFLUENZA MATCHES"
    )

    print(
        "=" * 70
    )

    if influenza_examples.empty:

        print(
            "No acceptable influenza matches "
            "were identified."
        )

    else:

        for _, row in (
            influenza_examples
            .head(5)
            .iterrows()
        ):

            print(
                f"\n{int(row['year'])} "
                f"WEEK "
                f"{int(row['epi_week'])}"
            )

            print(
                row[
                    "influenza_snippet"
                ]
            )

    # --------------------------------------------------------
    # Output locations.
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "OUTPUT FILES"
    )

    print(
        "=" * 70
    )

    print(
        f"\nDetailed audit:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"\nYearly summary:\n"
        f"{SUMMARY_FILE}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()