import argparse
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def find_matching_image(stem: str, source_images: Path) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = source_images / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Copy images that match label filenames from a labels directory."
    )
    parser.add_argument(
        "--source-labels",
        type=str,
        required=True,
        help="Path to the folder containing filtered .txt label files.",
    )
    parser.add_argument(
        "--source-images",
        type=str,
        required=True,
        help="Path to the folder containing the original image files.",
    )
    parser.add_argument(
        "--dest-images",
        type=str,
        required=True,
        help="Path to the destination folder for matched images (will be created).",
    )
    args = parser.parse_args()

    label_dir = Path(args.source_labels)
    image_dir = Path(args.source_images)
    dest_dir = Path(args.dest_images)

    if not label_dir.exists():
        raise FileNotFoundError(f"Source labels folder not found: {label_dir}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Source images folder not found: {image_dir}")

    label_files = sorted(label_dir.glob("*.txt"))
    print(f"Label files found        : {len(label_files)}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    not_found = []

    for lbl in label_files:
        img_path = find_matching_image(lbl.stem, image_dir)
        if img_path is not None:
            shutil.copy2(img_path, dest_dir / img_path.name)
            copied += 1
        else:
            not_found.append(lbl.stem)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Labels scanned     : {len(label_files)}")
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