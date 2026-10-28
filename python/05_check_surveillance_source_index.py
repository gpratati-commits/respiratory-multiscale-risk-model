from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_INDEX = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "surveillance_source_index.csv"
)


def main():

    df = pd.read_csv(SOURCE_INDEX)

    print("=" * 60)
    print("SURVEILLANCE SOURCE INDEX CHECK")
    print("=" * 60)

    for year, group in df.groupby("year"):

        weeks = sorted(
            group["epi_week"]
            .dropna()
            .astype(int)
            .unique()
        )

        print(f"\nYear: {year}")
        print(f"Number of bulletins: {len(weeks)}")

        if weeks:
            print(
                f"Observed week range: "
                f"{min(weeks)}-{max(weeks)}"
            )

        expected = set(
            range(1, max(weeks) + 1)
        )

        observed = set(weeks)

        missing = sorted(
            expected - observed
        )

        if missing:
            print(
                f"Missing weeks within observed range: "
                f"{missing}"
            )
        else:
            print(
                "No missing weeks within observed range."
            )


if __name__ == "__main__":
    main()