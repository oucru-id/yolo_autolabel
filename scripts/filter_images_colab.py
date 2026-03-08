import argparse
import shutil
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Copy image files that match filenames in a CSV."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the CSV file containing a 'filename' column.",
    )
    parser.add_argument(
        "--source-images",
        type=str,
        required=True,
        help="Path to the source folder containing image files.",
    )
    parser.add_argument(
        "--dest-images",
        type=str,
        required=True,
        help="Path to the destination folder for filtered images (will be created).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    source_dir = Path(args.source_images)
    dest_dir = Path(args.dest_images)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not source_dir.exists():
        raise FileNotFoundError(f"Source images folder not found: {source_dir}")

    df = pd.read_csv(csv_path)
    print(f"Total rows in CSV        : {len(df)}")

    unique_filenames = set(df["filename"].dropna().unique())
    print(f"Unique filenames in CSV  : {len(unique_filenames)}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    not_found = []

    for img_name in sorted(unique_filenames):
        src = source_dir / img_name
        if src.exists():
            shutil.copy2(src, dest_dir / img_name)
            copied += 1
        else:
            not_found.append(img_name)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Images copied      : {copied}")
    print(f"Images not found   : {len(not_found)}")
    print(f"Destination folder : {dest_dir}")
    print("=" * 60)

    if not_found:
        print(f"\n Missing images (showing first 20):")
        for n in not_found[:20]:
            print(f"   - {n}")
        if len(not_found) > 20:
            print(f"   ... and {len(not_found) - 20} more")

    print(f"\n Done")


if __name__ == "__main__":
    main()
