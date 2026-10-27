from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_DIRECTORIES = [
    "data/raw",
    "data/interim",
    "data/processed",
    "python",
    "R",
    "notebooks",
    "outputs/figures",
    "outputs/tables",
    "outputs/predictions",
    "outputs/models",
    "microbiome",
    "manuscript",
    "docs",
]


def main():
    print("=" * 70)
    print("RESPIRATORY MULTISCALE RISK MODEL")
    print("PROJECT SETUP CHECK")
    print("=" * 70)

    print(f"\nProject root:\n{PROJECT_ROOT}\n")

    all_present = True

    for directory in EXPECTED_DIRECTORIES:
        path = PROJECT_ROOT / directory

        if path.exists():
            print(f"[OK]      {directory}")
        else:
            print(f"[MISSING] {directory}")
            all_present = False

    print("\n" + "=" * 70)

    if all_present:
        print("SUCCESS: Project directory structure is complete.")
    else:
        print("WARNING: One or more project directories are missing.")

    print("=" * 70)


if __name__ == "__main__":
    main()