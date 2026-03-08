from pathlib import Path
import csv
import re
import argparse


def get_class_names(classes_path: Path) -> dict:
    with open(classes_path, "r") as f:
        return {i: line.strip() for i, line in enumerate(f.readlines()) if line.strip()}


def extract_bookyear(class_name: str) -> str:

    match = re.search(r'(\d{4})', class_name)
    if match:
        return match.group(1)
    return "unknown"


def extract_labels(labels_dir: Path, class_map: dict) -> list[dict]:
    rows = []
    label_files = sorted(labels_dir.glob("*.txt"))
    print(f"Labels dir  : {labels_dir}")
    print(f"Found       : {len(label_files)} label files")

    for lbl_path in label_files:
        stem = lbl_path.stem
        clean_stem = stem.split("__", 1)[1] if "__" in stem else stem

        with open(lbl_path, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    class_id = int(parts[0])
                    cx       = float(parts[1])
                    cy       = float(parts[2])
                    w        = float(parts[3])
                    h        = float(parts[4])
                except ValueError:
                    continue

                class_name = class_map.get(class_id, f"Unknown-{class_id}")
                bookyear   = extract_bookyear(class_name)

                rows.append({
                    "filename"      : f"{clean_stem}.png",
                    "class"         : class_name,
                    "bookyear"      : bookyear,
                    "coordinate_x"  : round(cx, 6),
                    "coordinate_y"  : round(cy, 6),
                    "width"         : round(w, 6),
                    "height"        : round(h, 6),
                })

    return rows


def main():
    parser = argparse.ArgumentParser(description="Extract label details to CSV")
    parser.add_argument("--labels", required=True,
                        help="Path to labels directory (e.g. results/dataset/labels/train)")
    parser.add_argument("--classes", required=True,
                        help="Path to classes.txt")
    parser.add_argument("--out", default="label_details.csv",
                        help="Output CSV path (default: label_details.csv)")
    args = parser.parse_args()

    class_map = get_class_names(Path(args.classes))
    print(f"Classes     : {len(class_map)} → {list(class_map.values())}")

    rows = extract_labels(Path(args.labels), class_map)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["filename", "class", "bookyear",
               "coordinate_x", "coordinate_y", "width", "height"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    from collections import Counter
    year_counts  = Counter(r["bookyear"] for r in rows)
    class_counts = Counter(r["class"] for r in rows)

    print(f"\nTotal annotations: {len(rows)}")
    print(f"Output CSV       : {out_path}")
    print(f"\nBy book year:")
    for year, count in sorted(year_counts.items()):
        print(f"  {year}: {count}")
    print(f"\nBy class:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count}")


if __name__ == "__main__":
    main()
