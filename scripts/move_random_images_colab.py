import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def main():
    parser = argparse.ArgumentParser(
        description="Randomly move (or copy) N images from one folder to another."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source folder containing images.",
    )
    parser.add_argument(
        "--dest",
        type=str,
        required=True,
        help="Destination folder (will be created if needed).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of images to move/copy (default: 1000).",
    )
    parser.add_argument(
        "--source-labels",
        type=str,
        default=None,
        help="(Optional) Source folder for matching label .txt files.",
    )
    parser.add_argument(
        "--dest-labels",
        type=str,
        default=None,
        help="(Optional) Destination folder for label files (will be created).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only — don't actually move/copy any files.",
    )
    args = parser.parse_args()

    source_dir = Path(args.source)
    dest_dir = Path(args.dest)
    action = "copy" if args.copy else "move"

    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder not found: {source_dir}")

    source_labels_dir = Path(args.source_labels) if args.source_labels else None
    dest_labels_dir = Path(args.dest_labels) if args.dest_labels else None

    if source_labels_dir and not source_labels_dir.exists():
        raise FileNotFoundError(f"Source labels folder not found: {source_labels_dir}")

    all_images = sorted(
        f for f in source_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    total_available = len(all_images)
    print(f"Total images in source   : {total_available}")

    if total_available == 0:
        print("No images found in source folder")
        return

    count = min(args.count, total_available)
    if count < args.count:
        print(f"Requested {args.count} but only {total_available} available. "
              f"Will {action} all {total_available}.")

    random.seed(args.seed)
    selected = random.sample(all_images, count)

    print(f"Images to {action}         : {count}")
    print(f"Random seed              : {args.seed}")
    if args.dry_run:
        print("MODE: Dry Run (no files will be moved/copied)")

    if not args.dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)
        if dest_labels_dir:
            dest_labels_dir.mkdir(parents=True, exist_ok=True)

    moved_images = 0
    moved_labels = 0
    missing_labels = []

    transfer_fn = shutil.copy2 if args.copy else shutil.move

    for img_path in selected:
        if not args.dry_run:
            transfer_fn(str(img_path), str(dest_dir / img_path.name))
        moved_images += 1

        if source_labels_dir and dest_labels_dir:
            label_name = img_path.stem + ".txt"
            label_src = source_labels_dir / label_name
            if label_src.exists():
                if not args.dry_run:
                    transfer_fn(str(label_src), str(dest_labels_dir / label_name))
                moved_labels += 1
            else:
                missing_labels.append(label_name)

    verb = "copied" if args.copy else "moved"
    verb_dry = f"would be {verb}" if args.dry_run else verb

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Images {verb_dry}    : {moved_images}")
    print(f"  Source folder       : {source_dir}")
    print(f"  Destination folder  : {dest_dir}")

    if source_labels_dir:
        print(f"  Labels {verb_dry}    : {moved_labels}")
        print(f"  Labels source       : {source_labels_dir}")
        print(f"  Labels destination  : {dest_labels_dir}")
        if missing_labels:
            print(f"  Labels not found    : {len(missing_labels)}")

    remaining = total_available - moved_images if not args.copy else total_available
    print(f"  Remaining in source : {remaining}")
    print("=" * 60)

    if missing_labels:
        print(f"\n Missing labels (showing first 10):")
        for n in missing_labels[:10]:
            print(f"   - {n}")
        if len(missing_labels) > 10:
            print(f"   ... and {len(missing_labels) - 10} more")

    print(f"\nDone")


if __name__ == "__main__":
    main()