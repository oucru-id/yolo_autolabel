import argparse
import os
import tempfile
from pathlib import Path
import sys


def predict_to_labels(
    model_path: str,
    dataset_dir: Path,
    out_dir: Path,
    conf_threshold: float = 0.25,
    save_conf: bool = False,
    split: str = "val",
):
    """
    model_path     : path to the .pt model weights
    dataset_dir    : root of the dataset (should contain images/)
    out_dir        : directory where .txt label files will be written
    conf_threshold : minimum confidence to keep a detection
    save_conf      : if True, append the confidence score as a 6th column
    split          : sub-folder name under images/ (e.g. val, train, test)
    """
    from ultralytics import YOLO
    try:
        os.getcwd()
    except FileNotFoundError:
        os.chdir(tempfile.gettempdir())

    model = YOLO(model_path)
    class_names = model.names

    images_dir = dataset_dir / "images" / split
    if not images_dir.exists():
        images_dir = dataset_dir / "images"
    if not images_dir.exists():
        images_dir = dataset_dir

    print(f"Model           : {model_path}")
    print(f"Images dir      : {images_dir}")
    print(f"Output dir      : {out_dir}")
    print(f"Conf threshold  : {conf_threshold}")
    print(f"Save confidence : {save_conf}")
    print(f"Class names     : {class_names}")

    image_files = sorted([
        f for f in images_dir.glob("*.*")
        if f.suffix.lower() in ('.png', '.jpg', '.jpeg')
    ])
    if not image_files:
        print("No images found.")
        sys.exit(1)
    print(f"Images found    : {len(image_files)}\n")

    out_dir.mkdir(parents=True, exist_ok=True)

    total_images = len(image_files)
    total_detections = 0
    images_with_detections = 0

    for idx, img_path in enumerate(image_files, 1):
        results = model(str(img_path), conf=conf_threshold, verbose=False)
        result = results[0]
        h, w = result.orig_shape

        lines = []
        if len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                x_center = ((x1 + x2) / 2.0) / w
                y_center = ((y1 + y2) / 2.0) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                if save_conf:
                    lines.append(
                        f"{cls_id} {x_center:.6f} {y_center:.6f} "
                        f"{bw:.6f} {bh:.6f} {conf:.6f}"
                    )
                else:
                    lines.append(
                        f"{cls_id} {x_center:.6f} {y_center:.6f} "
                        f"{bw:.6f} {bh:.6f}"
                    )

        txt_path = out_dir / f"{img_path.stem}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")   

        n_det = len(lines)
        total_detections += n_det
        if n_det > 0:
            images_with_detections += 1

        if idx % 50 == 0 or idx == total_images:
            print(f"  [{idx:>5}/{total_images}] {img_path.name}  →  {n_det} detections")

    print("\n" + "═" * 55)
    print("  PREDICTION SUMMARY")
    print("═" * 55)
    print(f"  Total images processed       : {total_images}")
    print(f"  Images with detections       : {images_with_detections}")
    print(f"  Images without detections    : {total_images - images_with_detections}")
    print(f"  Total detections             : {total_detections}")
    print(f"  Avg detections / image       : {total_detections / total_images:.2f}")
    print(f"  Label files written to       : {out_dir}")
    print("═" * 55)
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO predictions and export as label .txt files."
    )
    parser.add_argument("--model", required=True,
                        help="Path to trained YOLO model (.pt)")
    parser.add_argument("--data", required=True,
                        help="Dataset dir (with images/ sub-folder)")
    parser.add_argument("--out", default="predicted_labels",
                        help="Output directory for predicted label .txt files")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--save-conf", action="store_true",
                        help="Append confidence score as 6th column in each line")
    parser.add_argument("--split", default="val",
                        help="Sub-folder under images/ to use (default: val)")
    args = parser.parse_args()

    model_path  = str(Path(args.model).resolve())
    dataset_dir = Path(args.data).resolve()
    out_dir     = Path(args.out).resolve()

    predict_to_labels(
        model_path     = model_path,
        dataset_dir    = dataset_dir,
        out_dir        = out_dir,
        conf_threshold = args.conf,
        save_conf      = args.save_conf,
        split          = args.split,
    )

if __name__ == "__main__":
    main()