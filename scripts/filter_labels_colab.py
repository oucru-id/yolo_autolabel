import argparse
import os
import shutil
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Copy label .txt files that match filenames in a CSV."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the CSV file containing a 'filename' column.",
    )
    parser.add_argument(
        "--source-labels",
        type=str,
        required=True,
        help="Path to the source folder containing .txt label files.",
    )
    parser.add_argument(
        "--dest-labels",
        type=str,
        required=True,
        help="Path to the destination folder for filtered labels (will be created).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    source_dir = Path(args.source_labels)
    dest_dir = Path(args.dest_labels)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not source_dir.exists():
        raise FileNotFoundError(f"Source labels folder not found: {source_dir}")

    df = pd.read_csv(csv_path)
    print(f"Total rows in CSV        : {len(df)}")

    unique_filenames = df["filename"].dropna().unique()
    print(f"Unique filenames in CSV  : {len(unique_filenames)}")

    label_names = {os.path.splitext(fn)[0] + ".txt" for fn in unique_filenames}
    print(f"Label files to look for  : {len(label_names)}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    not_found = []

    for lbl in sorted(label_names):
        src = source_dir / lbl
        if src.exists():
            shutil.copy2(src, dest_dir / lbl)
            copied += 1
        else:
            not_found.append(lbl)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Labels copied      : {copied}")
    print(f"Labels not found   : {len(not_found)}")
    print(f"Destination folder : {dest_dir}")
    print("=" * 60)

    if not_found:
        print(f"\n Missing labels (showing first 20):")
        for n in not_found[:20]:
            print(f"   - {n}")
        if len(not_found) > 20:
            print(f"   ... and {len(not_found) - 20} more")

    print(f"\n Done")


if __name__ == "__main__":
    main()