import argparse
import collections
import csv
from pathlib import Path
import sys

def compute_iou_box(box1, box2) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter)


def load_labels_xyxy(txt_path: Path, w: int, h: int) -> list[dict]:
    gts = []
    if txt_path.exists():
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    c = int(parts[0])
                    x_c, y_c, bw, bh = map(float, parts[1:5])
                    x1 = int((x_c - bw / 2) * w)
                    y1 = int((y_c - bh / 2) * h)
                    x2 = int((x_c + bw / 2) * w)
                    y2 = int((y_c + bh / 2) * h)
                    gts.append({"class_id": c, "bbox": [x1, y1, x2, y2]})
    return gts

def evaluate_iou_full(model_path: str, dataset_dir: Path, iou_threshold: float = 0.5):

    from ultralytics import YOLO
    model = YOLO(model_path)
    class_names = model.names

    images_dir = dataset_dir / "images" / "val"
    labels_dir = dataset_dir / "labels" / "val"
    if not images_dir.exists():
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"
    if not images_dir.exists():
        images_dir = dataset_dir
        labels_dir = dataset_dir.parent / "labels" if dataset_dir.name == "images" else dataset_dir

    print(f"Images dir      : {images_dir}")
    print(f"Labels dir      : {labels_dir}")
    print(f"IoU threshold   : {iou_threshold}  (standard: 0.5)")

    image_files = sorted([
        f for f in images_dir.glob("*.*")
        if f.suffix.lower() in ('.png', '.jpg', '.jpeg')
    ])
    if not image_files:
        print("No images found.")
        sys.exit(1)
    print(f"Images found    : {len(image_files)}\n")

    records       = []   
    image_records = []   

    for img_path in image_files:
        results = model(str(img_path), verbose=False)
        result  = results[0]
        h, w    = result.orig_shape

        txt_path = labels_dir / f"{img_path.stem}.txt"
        gts      = load_labels_xyxy(txt_path, w, h)

        preds = []
        if len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                preds.append({
                    "class_id"  : cls_id,
                    "class"     : class_names.get(cls_id, str(cls_id)),
                    "confidence": conf,
                    "bbox"      : [x1, y1, x2, y2],
                    "iou"       : 0.0,
                    "matched"   : False,
                })

        if len(gts) == 0 and len(preds) == 0:
            records.append({
                "filename"   : img_path.name,
                "class"      : "N/A",
                "confidence" : 0.0,
                "iou"        : 0.0,
                "type"       : "True Negative",
                "iou_threshold": iou_threshold,
            })
            image_records.append({
                "filename": img_path.name,
                "n_gt"    : 0,
                "n_pred"  : 0,
                "tp": 0, "fp": 0, "fn": 0, "tn": 1,
            })
            continue

        all_matches = []
        for i, p in enumerate(preds):
            for j, gt in enumerate(gts):
                if p["class_id"] == gt["class_id"]:
                    iou = compute_iou_box(p["bbox"], gt["bbox"])
                    if iou >= iou_threshold:
                        all_matches.append((iou, i, j))

        all_matches.sort(key=lambda x: x[0], reverse=True)
        matched_preds = set()
        matched_gts   = set()

        for iou_val, i, j in all_matches:
            if i not in matched_preds and j not in matched_gts:
                preds[i]["iou"]     = iou_val
                preds[i]["matched"] = True
                matched_preds.add(i)
                matched_gts.add(j)

        img_tp = img_fp = img_fn = 0

        for p in preds:
            det_type = "True Positive" if p["matched"] else "False Positive"
            if p["matched"]:
                img_tp += 1
            else:
                img_fp += 1
            records.append({
                "filename"     : img_path.name,
                "class"        : p["class"],
                "confidence"   : round(p["confidence"], 4),
                "iou"          : round(p["iou"], 4),
                "type"         : det_type,
                "iou_threshold": iou_threshold,
            })

        for j, gt in enumerate(gts):
            if j not in matched_gts:
                cls_id = gt["class_id"]
                img_fn += 1
                records.append({
                    "filename"     : img_path.name,
                    "class"        : class_names.get(cls_id, str(cls_id)),
                    "confidence"   : 0.0,
                    "iou"          : 0.0,
                    "type"         : "False Negative",
                    "iou_threshold": iou_threshold,
                })

        image_records.append({
            "filename": img_path.name,
            "n_gt"    : len(gts),
            "n_pred"  : len(preds),
            "tp"      : img_tp,
            "fp"      : img_fp,
            "fn"      : img_fn,
            "tn"      : 0,
        })

    return records, image_records

def compute_metrics(records: list[dict], class_names: dict) -> dict:

    total_tp = sum(1 for r in records if r["type"] == "True Positive")
    total_fp = sum(1 for r in records if r["type"] == "False Positive")
    total_fn = sum(1 for r in records if r["type"] == "False Negative")
    total_tn = sum(1 for r in records if r["type"] == "True Negative")

    def safe_div(n, d):
        return n / d if d > 0 else 0.0

    overall_precision   = safe_div(total_tp, total_tp + total_fp)
    overall_recall      = safe_div(total_tp, total_tp + total_fn)
    overall_f1          = safe_div(2 * overall_precision * overall_recall,
                                   overall_precision + overall_recall)
    overall_specificity = safe_div(total_tn, total_tn + total_fp)

    class_counts = collections.defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for r in records:
        t = r["type"]
        c = r["class"]
        if t == "True Positive":
            class_counts[c]["tp"] += 1
        elif t == "False Positive":
            class_counts[c]["fp"] += 1
        elif t == "False Negative":
            class_counts[c]["fn"] += 1

    per_class = {}
    for cls_name, counts in sorted(class_counts.items()):
        tp  = counts["tp"]
        fp  = counts["fp"]
        fn  = counts["fn"]
        p   = safe_div(tp, tp + fp)
        r   = safe_div(tp, tp + fn)
        f1  = safe_div(2 * p * r, p + r)
        per_class[cls_name] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4),
            "recall"   : round(r, 4),
            "f1"       : round(f1, 4),
        }

    return {
        "overall": {
            "tp": total_tp, "fp": total_fp,
            "fn": total_fn, "tn": total_tn,
            "precision"  : round(overall_precision, 4),
            "recall"     : round(overall_recall, 4),
            "f1"         : round(overall_f1, 4),
            "specificity": round(overall_specificity, 4),
        },
        "per_class": per_class,
    }


def print_metrics(metrics: dict):
    ov = metrics["overall"]
    print("\n" + "═" * 65)
    print("  OVERALL METRICS")
    print("═" * 65)
    print(f"  {'Type':<20} {'Count':>8}")
    print(f"  {'-' * 30}")
    print(f"  {'True Positive  (TP)':<20} {ov['tp']:>8}")
    print(f"  {'False Positive (FP)':<20} {ov['fp']:>8}  ← predicted, no GT match")
    print(f"  {'False Negative (FN)':<20} {ov['fn']:>8}  ← GT exists, not predicted")
    print(f"  {'True Negative  (TN)':<20} {ov['tn']:>8}  ← empty image, no prediction")
    print(f"  {'-' * 30}")
    print(f"  {'Precision':<20} {ov['precision']:>8.4f}  TP / (TP+FP)")
    print(f"  {'Recall':<20} {ov['recall']:>8.4f}  TP / (TP+FN)")
    print(f"  {'F1 Score':<20} {ov['f1']:>8.4f}")
    print(f"  {'Specificity':<20} {ov['specificity']:>8.4f}  TN / (TN+FP)")
    print("═" * 65)

    print("\n  PER-CLASS METRICS")
    print(f"  {'Class':<35} {'TP':>5} {'FP':>5} {'FN':>5} {'P':>7} {'R':>7} {'F1':>7}")
    print(f"  {'-' * 72}")
    for cls, m in sorted(metrics["per_class"].items()):
        print(f"  {cls:<35} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5} "
              f"{m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f}")
    print("═" * 65 + "\n")

def plot_results(records: list[dict], metrics: dict, save_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Skipping plots.")
        return

    per_class = metrics["per_class"]
    classes   = sorted(per_class.keys())

    # ── Plot 1: IoU Distribution per Class (TP only) ─────────────
    class_ious = collections.defaultdict(list)
    for r in records:
        if r["type"] == "True Positive":
            class_ious[r["class"]].append(r["iou"])

    if class_ious:
        fig, ax = plt.subplots(figsize=(max(10, len(class_ious) * 1.5), 6),
                               facecolor="#0d1117")
        ax.set_facecolor("#161b22")
        data = [class_ious[c] for c in sorted(class_ious)]
        ax.boxplot(
            data, patch_artist=True,
            boxprops   =dict(facecolor="#58a6ff", color="white", alpha=0.7),
            capprops   =dict(color="white"),
            whiskerprops=dict(color="white"),
            flierprops =dict(markeredgecolor="white", markerfacecolor="#f97583", alpha=0.5),
            medianprops=dict(color="#56d364", linewidth=2),
        )
        ax.set_xticks(range(1, len(class_ious) + 1))
        ax.set_xticklabels(sorted(class_ious), rotation=45, ha="right", color="white")
        ax.set_title("IoU Distribution per Class (True Positives)", color="white",
                     fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("IoU", color="white")
        ax.set_ylim(0, 1.05)
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        plt.tight_layout()
        plt.savefig(str(save_dir / "iou_distribution_per_class.png"),
                    dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()

    # ── Plot 2: TP / FP / FN per Class ───────────────────────────
    if classes:
        tps = [per_class[c]["tp"] for c in classes]
        fps = [per_class[c]["fp"] for c in classes]
        fns = [per_class[c]["fn"] for c in classes]
        x   = np.arange(len(classes))
        w   = 0.25

        fig, ax = plt.subplots(figsize=(max(10, len(classes) * 1.5), 6),
                               facecolor="#0d1117")
        ax.set_facecolor("#161b22")
        ax.bar(x - w, tps, w, label="TP", color="#56d364", alpha=0.85)
        ax.bar(x,     fps, w, label="FP", color="#f97583", alpha=0.85)
        ax.bar(x + w, fns, w, label="FN", color="#d2a8ff", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha="right", color="white")
        ax.set_title("TP / FP / FN per Class", color="white",
                     fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("Count", color="white")
        ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        plt.tight_layout()
        plt.savefig(str(save_dir / "tp_fp_fn_per_class.png"),
                    dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()

    # ── Plot 3: Precision / Recall / F1 per Class ────────────────
    if classes:
        prec = [per_class[c]["precision"] for c in classes]
        rec  = [per_class[c]["recall"]    for c in classes]
        f1s  = [per_class[c]["f1"]        for c in classes]
        x    = np.arange(len(classes))
        w    = 0.25

        fig, ax = plt.subplots(figsize=(max(10, len(classes) * 1.5), 6),
                               facecolor="#0d1117")
        ax.set_facecolor("#161b22")
        ax.bar(x - w, prec, w, label="Precision", color="#58a6ff", alpha=0.85)
        ax.bar(x,     rec,  w, label="Recall",    color="#56d364", alpha=0.85)
        ax.bar(x + w, f1s,  w, label="F1",        color="#f0883e", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha="right", color="white")
        ax.set_ylim(0, 1.15)
        ax.set_title("Precision / Recall / F1 per Class", color="white",
                     fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("Score", color="white")
        ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_color("#30363d")

        for i, v in enumerate(f1s):
            ax.text(x[i] + w, v + 0.02, f"{v:.2f}", ha="center",
                    fontsize=8, color="white")
        plt.tight_layout()
        plt.savefig(str(save_dir / "precision_recall_f1_per_class.png"),
                    dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()

    # ── Plot 4: Overall confusion counts (TP/FP/FN/TN) ──────────
    ov     = metrics["overall"]
    labels = ["TP\n(correct)", "FP\n(false alarm)", "FN\n(missed)", "TN\n(correct empty)"]
    values = [ov["tp"], ov["fp"], ov["fn"], ov["tn"]]
    colors = ["#56d364", "#f97583", "#d2a8ff", "#58a6ff"]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                str(val), ha="center", fontsize=12, fontweight="bold", color="white")
    ax.set_title("Detection Type Counts (Overall)", color="white",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Count", color="white")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    plt.tight_layout()
    plt.savefig(str(save_dir / "detection_type_counts.png"),
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"  Plots saved to: {save_dir}")


def save_csvs(records, image_records, metrics, out_dir: Path):
    # 1. Per-detection CSV
    det_csv = out_dir / "iou_full_per_detection.csv"
    headers = ["filename", "class", "confidence", "iou", "type", "iou_threshold"]
    with open(det_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(records)
    print(f"  Per-detection CSV : {det_csv}")

    # 2. Per-image CSV
    img_csv = out_dir / "iou_full_per_image.csv"
    with open(img_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "n_gt", "n_pred",
                                                "tp", "fp", "fn", "tn"])
        writer.writeheader()
        writer.writerows(image_records)
    print(f"  Per-image CSV     : {img_csv}")

    # 3. Per-class metrics CSV
    cls_csv = out_dir / "iou_full_per_class_metrics.csv"
    with open(cls_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "tp", "fp", "fn",
                                                "precision", "recall", "f1"])
        writer.writeheader()
        for cls, m in sorted(metrics["per_class"].items()):
            writer.writerow({"class": cls, **m})
    print(f"  Per-class CSV     : {cls_csv}")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO model with full TP/FP/FN/TN metrics."
    )
    parser.add_argument("--model", required=True,
                        help="Path to trained YOLO model (.pt)")
    parser.add_argument("--data",  required=True,
                        help="Validation dataset dir (with images/ and labels/)")
    parser.add_argument("--out",   default=".",
                        help="Output directory for CSV and PNG results")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold to count a match as TP (default: 0.5)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nModel           : {args.model}")
    print(f"Dataset         : {args.data}")
    print(f"Output          : {out_dir}")
    print(f"IoU threshold   : {args.iou_threshold}\n")

    records, image_records = evaluate_iou_full(
        model_path    = args.model,
        dataset_dir   = Path(args.data),
        iou_threshold = args.iou_threshold,
    )

    from ultralytics import YOLO
    class_names = YOLO(args.model).names

    metrics = compute_metrics(records, class_names)
    print_metrics(metrics)

    save_csvs(records, image_records, metrics, out_dir)
    plot_results(records, metrics, out_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()