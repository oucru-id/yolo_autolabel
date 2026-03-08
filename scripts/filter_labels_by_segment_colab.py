import argparse
import shutil
from pathlib import Path


CLASSES_2023_2024 = {
    3,   # 2023_nik
    4,   # 2024_nik
    27,  # bukukia2023_page0_segment1
    28,  # bukukia2023_page0_segment2
    29,  # bukukia2023_page0_segment3
    30,  # bukukia2023_page0_segment4
    31,  # bukukia2023_page0_segment5
    32,  # bukukia2023_page0_segment6
    33,  # bukukia2023_page0_segment7
    34,  # bukukia2024_page0_segment1
    35,  # bukukia2024_page0_segment2
    36,  # bukukia2024_page0_segment3
    37,  # bukukia2024_page0_segment4
    38,  # bukukia2024_page0_segment5
}


def label_has_only_2023_2024(label_path: Path) -> bool:
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            class_id = int(line.split()[0])
            if class_id not in CLASSES_2023_2024:
                return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Copy label .txt files that contain ONLY 2023/2024 segment classes."
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

    source_dir = Path(args.source_labels)
    dest_dir = Path(args.dest_labels)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source labels folder not found: {source_dir}")

    all_labels = sorted(source_dir.glob("*.txt"))
    print(f"Total label files found  : {len(all_labels)}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    empty = 0

    for lbl in all_labels:
        if lbl.stat().st_size == 0:
            empty += 1
            continue

        if label_has_only_2023_2024(lbl):
            shutil.copy2(lbl, dest_dir / lbl.name)
            copied += 1
        else:
            skipped += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total label files  : {len(all_labels)}")
    print(f"Copied (2023/2024) : {copied}")
    print(f"Skipped (other)    : {skipped}")
    print(f"Empty (no annot.)  : {empty}")
    print(f"Destination folder : {dest_dir}")
    print("=" * 60)

    print(f"\n 2023/2024 class IDs used for filtering:")
    print(f"   {sorted(CLASSES_2023_2024)}")

    print(f"\n Done")


if __name__ == "__main__":
    main()