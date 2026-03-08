import argparse
import collections
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    BASE, TEST_IMAGES, TEST_RESULTS, CONF,
    get_best_model, ensure_drive_mounted,
)

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0: return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter)


def load_labels(txt_path, w, h):
    gts = []
    if txt_path.exists():
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    c = int(parts[0])
                    x_c, y_c, bw, bh = map(float, parts[1:5])
                    x1 = int((x_c - bw / 2) * w)
                    y1 = int((y_c - bh / 2) * h)
                    x2 = int((x_c + bw / 2) * w)
                    y2 = int((y_c + bh / 2) * h)
                    gts.append({'class_id': c, 'bbox': [x1, y1, x2, y2]})
    return gts


def compute_metrics(det_records, class_names):

    def safe_div(n, d):
        return n / d if d > 0 else 0.0

    total_tp = sum(1 for r in det_records if r["type"] == "True Positive")
    total_fp = sum(1 for r in det_records if r["type"] == "False Positive")
    total_fn = sum(1 for r in det_records if r["type"] == "False Negative")
    total_tn = sum(1 for r in det_records if r["type"] == "True Negative")

    p   = safe_div(total_tp, total_tp + total_fp)
    r   = safe_div(total_tp, total_tp + total_fn)
    f1  = safe_div(2 * p * r, p + r)
    spe = safe_div(total_tn, total_tn + total_fp)

    class_counts = collections.defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0,
                                                     "ious": [], "confs": []})
    for rec in det_records:
        t = rec["type"]; c = rec["class"]
        if t == "True Positive":
            class_counts[c]["tp"] += 1
            class_counts[c]["ious"].append(rec["iou"])
            class_counts[c]["confs"].append(rec["confidence"])
        elif t == "False Positive":
            class_counts[c]["fp"] += 1
            class_counts[c]["confs"].append(rec["confidence"])
        elif t == "False Negative":
            class_counts[c]["fn"] += 1

    per_class = {}
    for cls_name, counts in sorted(class_counts.items()):
        tp  = counts["tp"]; fp = counts["fp"]; fn = counts["fn"]
        cp  = safe_div(tp, tp + fp)
        cr  = safe_div(tp, tp + fn)
        cf1 = safe_div(2 * cp * cr, cp + cr)
        avg_iou  = np.mean(counts["ious"]) if counts["ious"] else 0.0
        avg_conf = np.mean(counts["confs"]) if counts["confs"] else 0.0
        per_class[cls_name] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(cp, 4),
            "recall"   : round(cr, 4),
            "f1"       : round(cf1, 4),
            "avg_iou"  : round(float(avg_iou), 4),
            "avg_conf" : round(float(avg_conf), 4),
        }

    return {
        "overall": {
            "tp": total_tp, "fp": total_fp, "fn": total_fn, "tn": total_tn,
            "precision"  : round(p, 4),
            "recall"     : round(r, 4),
            "f1"         : round(f1, 4),
            "specificity": round(spe, 4),
        },
        "per_class": per_class,
    }


def print_metrics(metrics):
    ov = metrics["overall"]
    print("\n" + "═" * 65)
    print("  OVERALL METRICS  [TEST SET]")
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

    print("\n  PER-CLASS METRICS  [TEST SET]")
    print(f"  {'Class':<35} {'TP':>5} {'FP':>5} {'FN':>5} {'P':>7} {'R':>7} {'F1':>7} {'AvgIoU':>8} {'AvgConf':>8}")
    print(f"  {'-' * 90}")
    for cls, m in sorted(metrics["per_class"].items()):
        print(f"  {cls:<35} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5} "
              f"{m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f} "
              f"{m['avg_iou']:>8.4f} {m['avg_conf']:>8.4f}")
    print("═" * 65 + "\n")

def save_csvs(det_records, image_records, metrics, out_dir):
    # Per-detection CSV
    det_csv = out_dir / "test_per_detection.csv"
    with open(det_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "class", "confidence",
                                                "iou", "type", "iou_threshold"])
        writer.writeheader(); writer.writerows(det_records)
    print(f"  Per-detection CSV : {det_csv}")

    # Per-image CSV
    img_csv = out_dir / "test_per_image.csv"
    with open(img_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "n_gt", "n_pred",
                                                "tp", "fp", "fn", "tn", "mean_iou"])
        writer.writeheader(); writer.writerows(image_records)
    print(f"  Per-image CSV     : {img_csv}")

    # Per-class CSV
    cls_csv = out_dir / "test_per_class_metrics.csv"
    with open(cls_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "tp", "fp", "fn",
                                                "precision", "recall", "f1",
                                                "avg_iou", "avg_conf"])
        writer.writeheader()
        for cls, m in sorted(metrics["per_class"].items()):
            writer.writerow({"class": cls, **m})
    print(f"  Per-class CSV     : {cls_csv}")

def main(conf=None, model_path=None, test_dir=None, iou_threshold=0.5):
    conf = conf or CONF

    ensure_drive_mounted()

    print("=" * 60)
    print("  YOLO MODEL TESTING")
    print("=" * 60)

    if test_dir:
        TEST_DIR = Path(test_dir)
    else:
        TEST_DIR = TEST_IMAGES

    if not TEST_DIR.exists():
        print(f"Test directory not found: {TEST_DIR}")
        print("Please check the path or provide one via --data")
        sys.exit(1)

    if (TEST_DIR / "images").is_dir():
        print(f"  Found 'images' subfolder in {TEST_DIR}, checking there instead.")
        TEST_DIR = TEST_DIR / "images"

    base_output_dir = TEST_RESULTS
    
    counter = 1
    OUTPUT_DIR = base_output_dir
    while OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        OUTPUT_DIR = base_output_dir.parent / f"{base_output_dir.name}{counter}"
        counter += 1

    (OUTPUT_DIR / "annotated").mkdir(parents=True, exist_ok=True)

    if model_path:
        MODEL_PATH = Path(model_path)
    else:
        MODEL_PATH = get_best_model()
    
    print(f"\n  Model: {MODEL_PATH}")
    print(f"  IoU Threshold: {iou_threshold}")

    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        print("Run training first or specify a valid --model path.")
        sys.exit(1)

    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))
    class_names = model.names
    num_classes = len(class_names)
    print(f"\n  Classes ({num_classes}): {list(class_names.values())}")

    image_files = sorted([
        f for f in TEST_DIR.iterdir()
        if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    ])
    
    if len(image_files) == 0:
        print(f"\n  Found 0 test images in {TEST_DIR}")
        print("  Skipping testing.")
        sys.exit(0)
        
    print(f"\n  Found {len(image_files)} test images")
    print("-" * 60)

    class_colors = [
        (46, 204, 113),   
        (52, 152, 219),   
        (241, 196, 15),   
        (155, 89, 182),  
        (230, 126, 34),  
        (26, 188, 156),   
        (192, 57, 43),   
    ]

    all_detections = []      
    per_image_results = []   
    det_records = []          
    image_records = []      

    for idx, img_path in enumerate(image_files):
        print(f"\n[{idx+1}/{len(image_files)}] {img_path.name}")

        results = model(str(img_path), conf=conf, verbose=False)
        result = results[0]

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        boxes = result.boxes
        num_det = len(boxes)

        txt_path = img_path.with_suffix('.txt')
        if not txt_path.exists() and img_path.parent.name == 'images':
            txt_path = img_path.parent.parent / 'labels' / f"{img_path.stem}.txt"
        gts = load_labels(txt_path, w, h)
        has_gt = txt_path.exists()

        preds = []
        if num_det > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                det_conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_name = class_names[cls_id]
                preds.append({
                    'class': cls_name,
                    'class_id': cls_id,
                    'confidence': det_conf,
                    'bbox': [x1, y1, x2, y2],
                    'iou': 0.0,
                    'matched': False,
                })

        if len(gts) == 0 and len(preds) == 0:
            det_records.append({
                "filename"     : img_path.name,
                "class"        : "N/A",
                "confidence"   : 0.0,
                "iou"          : 0.0,
                "type"         : "True Negative",
                "iou_threshold": iou_threshold,
            })
            image_records.append({
                "filename": img_path.name,
                "n_gt": 0, "n_pred": 0,
                "tp": 0, "fp": 0, "fn": 0, "tn": 1,
                "mean_iou": 1.0,
            })
            per_image_results.append({
                'filename': img_path.name,
                'num_detections': 0,
                'detections': [],
                'annotated_path': None,
                'has_gt': has_gt,
                'mean_iou': 1.0,
            })
            save_path = OUTPUT_DIR / "annotated" / f"result_{idx+1:02d}_{img_path.stem[:50]}.jpg"
            cv2.imwrite(str(save_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            per_image_results[-1]['annotated_path'] = save_path
            print(f"  No detections, no GT → True Negative")
            continue

        all_matches = []
        for i, p in enumerate(preds):
            for j, gt in enumerate(gts):
                if p['class_id'] == gt['class_id']:
                    iou_val = compute_iou(p['bbox'], gt['bbox'])
                    if iou_val >= iou_threshold:
                        all_matches.append((iou_val, i, j))

        all_matches.sort(key=lambda x: x[0], reverse=True)
        matched_preds = set()
        matched_gts = set()
        for iou_val, i, j in all_matches:
            if i not in matched_preds and j not in matched_gts:
                preds[i]['iou'] = iou_val
                preds[i]['matched'] = True
                matched_preds.add(i)
                matched_gts.add(j)

        img_tp = img_fp = img_fn = 0

        for p in preds:
            det_type = "True Positive" if p["matched"] else "False Positive"
            if p["matched"]:
                img_tp += 1
            else:
                img_fp += 1
            det_records.append({
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
                det_records.append({
                    "filename"     : img_path.name,
                    "class"        : class_names.get(cls_id, str(cls_id)),
                    "confidence"   : 0.0,
                    "iou"          : 0.0,
                    "type"         : "False Negative",
                    "iou_threshold": iou_threshold,
                })

        all_ious = [p['iou'] for p in preds]
        unmatched_gts_count = len(gts) - len(matched_gts)
        total_objects = len(preds) + unmatched_gts_count
        mean_iou = sum(all_ious) / total_objects if total_objects > 0 else 0.0

        image_records.append({
            "filename": img_path.name,
            "n_gt": len(gts), "n_pred": len(preds),
            "tp": img_tp, "fp": img_fp, "fn": img_fn, "tn": 0,
            "mean_iou": round(mean_iou, 4),
        })

        for d in preds:
            x1, y1, x2, y2 = d['bbox']
            cls_id = d['class_id']
            cls_name = d['class']
            det_conf = d['confidence']

            color = class_colors[cls_id % len(class_colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            iou_str = f" IoU:{d['iou']:.2f}" if d['iou'] > 0 else ""
            tag = "TP" if d['matched'] else "FP"
            label_text = f"{cls_name} {det_conf:.2f}{iou_str} [{tag}]"
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 5, y1), color, -1)
            cv2.putText(img, label_text, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        save_path = OUTPUT_DIR / "annotated" / f"result_{idx+1:02d}_{img_path.stem[:50]}.jpg"
        cv2.imwrite(str(save_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        pred_lbl_dir = OUTPUT_DIR / "predicted_labels"
        pred_lbl_dir.mkdir(parents=True, exist_ok=True)
        lbl_lines = []
        for d in preds:
            bx1, by1, bx2, by2 = d['bbox']
            x_center = ((bx1 + bx2) / 2.0) / w
            y_center = ((by1 + by2) / 2.0) / h
            bw_norm = (bx2 - bx1) / w
            bh_norm = (by2 - by1) / h
            lbl_lines.append(
                f"{d['class_id']} {x_center:.6f} {y_center:.6f} "
                f"{bw_norm:.6f} {bh_norm:.6f} {d['confidence']:.6f}"
            )
        lbl_path = pred_lbl_dir / f"{img_path.stem}.txt"
        with open(lbl_path, "w", encoding="utf-8") as lf:
            lf.write("\n".join(lbl_lines))
            if lbl_lines:
                lf.write("\n")

        all_detections.extend(preds)
        per_image_results.append({
            'filename': img_path.name,
            'num_detections': num_det,
            'detections': preds,
            'annotated_path': save_path,
            'has_gt': has_gt,
            'mean_iou': mean_iou,
        })

        print(f"  Detections: {num_det}  |  TP: {img_tp}  FP: {img_fp}  FN: {img_fn}")
        if has_gt:
            print(f"  Mean IoU  : {mean_iou:.3f}")
        for d in preds:
            tag = "TP" if d['matched'] else "FP"
            iou_str = f", IoU={d['iou']:.2f}" if d['iou'] > 0 else ""
            print(f"    - {d['class']}: {d['confidence']:.3f}{iou_str} [{tag}]")
        if img_fn > 0:
            print(f"    - {img_fn} ground truth object(s) missed [FN]")

    metrics = compute_metrics(det_records, class_names)
    print_metrics(metrics)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from collections import Counter, defaultdict

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    total_images = len(image_files)
    total_detections = len(all_detections)
    images_with_det = sum(1 for r in per_image_results if r['num_detections'] > 0)

    print(f"\n  Total Images    : {total_images}")
    print(f"  With Detections : {images_with_det}")
    print(f"  Without         : {total_images - images_with_det}")
    print(f"  Total Detections: {total_detections}")

    class_counts = Counter(d['class'] for d in all_detections)
    class_confidences = defaultdict(list)
    for d in all_detections:
        class_confidences[d['class']].append(d['confidence'])

    print(f"\n  Per-class breakdown:")
    for cls_name in sorted(class_counts.keys()):
        cnt = class_counts[cls_name]
        avg_c = np.mean(class_confidences[cls_name])
        print(f"    {cls_name}: {cnt} detections (avg conf: {avg_c:.3f})")

    plt.rcParams.update({
        'figure.facecolor': '#0d1117',
        'axes.facecolor': '#161b22',
        'axes.edgecolor': '#30363d',
        'axes.labelcolor': '#c9d1d9',
        'text.color': '#c9d1d9',
        'xtick.color': '#8b949e',
        'ytick.color': '#8b949e',
    })

    n_preview = min(6, len(per_image_results))
    n_top_charts = 5  
    n_cols = max(n_top_charts, n_preview)
    fig = plt.figure(figsize=(max(24, n_cols * 4.5), 14), facecolor='#0d1117')
    gs = fig.add_gridspec(3, n_cols, hspace=0.5, wspace=0.3)

    # --- Chart 1: Detections per Class ---
    ax1 = fig.add_subplot(gs[0, 0])
    if class_counts:
        classes_sorted = sorted(class_counts.keys())
        counts_sorted = [class_counts[c] for c in classes_sorted]
        bars = ax1.barh(classes_sorted, counts_sorted,
                        color=['#58a6ff', '#f97583', '#56d364', '#d2a8ff', '#f0883e', '#3fb950', '#da3633'],
                        height=0.6)
        for bar, cnt in zip(bars, counts_sorted):
            ax1.text(bar.get_width() + max(counts_sorted)*0.02, bar.get_y() + bar.get_height()/2,
                    str(cnt), va='center', fontsize=11, fontweight='bold', color='white')
    ax1.set_title('Detections per Class', fontsize=12, fontweight='bold', color='white')

    # --- Chart 2: Confidence Distribution ---
    ax2 = fig.add_subplot(gs[0, 1])
    if all_detections:
        confs = [d['confidence'] for d in all_detections]
        ax2.hist(confs, bins=20, range=(0, 1), color='#58a6ff', edgecolor='white', linewidth=0.5, alpha=0.8)
        ax2.axvline(x=np.mean(confs), color='#f97583', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(confs):.2f}')
        ax2.legend(fontsize=10)
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Frequency')
    ax2.set_title('Confidence Distribution', fontsize=12, fontweight='bold', color='white')

    # --- Chart 3: Per-class avg confidence ---
    ax3 = fig.add_subplot(gs[0, 2])
    if class_confidences:
        classes_sorted = sorted(class_confidences.keys())
        avg_confs = [np.mean(class_confidences[c]) for c in classes_sorted]
        min_confs = [np.min(class_confidences[c]) for c in classes_sorted]
        max_confs = [np.max(class_confidences[c]) for c in classes_sorted]

        x_pos = range(len(classes_sorted))
        ax3.bar(x_pos, avg_confs, color='#56d364', alpha=0.8, label='Avg')
        ax3.errorbar(x_pos, avg_confs,
                     yerr=[[a-mn for a, mn in zip(avg_confs, min_confs)],
                           [mx-a for a, mx in zip(avg_confs, max_confs)]],
                     fmt='none', capsize=5, color='white', alpha=0.5)
        ax3.set_xticks(list(x_pos))
        ax3.set_xticklabels(classes_sorted, rotation=45, ha='right', fontsize=9)
        ax3.set_ylim(0, 1.1)

        for i, v in enumerate(avg_confs):
            ax3.text(i, v + 0.03, f'{v:.2f}', ha='center', fontsize=10, fontweight='bold', color='white')

    ax3.set_title('Avg Confidence per Class', fontsize=12, fontweight='bold', color='white')

    # --- Chart 4: Per-class Mean IoU ---
    ax4 = fig.add_subplot(gs[0, 3])
    pc = metrics["per_class"]
    if pc:
        cls_sorted = sorted(pc.keys())
        avg_ious = [pc[c]["avg_iou"] for c in cls_sorted]
        x_pos = range(len(cls_sorted))
        bars4 = ax4.bar(x_pos, avg_ious, color='#d2a8ff', alpha=0.85)
        ax4.set_xticks(list(x_pos))
        ax4.set_xticklabels(cls_sorted, rotation=45, ha='right', fontsize=9)
        ax4.set_ylim(0, 1.1)
        for i, v in enumerate(avg_ious):
            ax4.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9, fontweight='bold', color='white')
    ax4.set_title('Avg IoU per Class (TP)', fontsize=12, fontweight='bold', color='white')

    # --- Chart 5: TP / FP / FN per Class ---
    ax5 = fig.add_subplot(gs[0, 4])
    if pc:
        cls_sorted = sorted(pc.keys())
        tps = [pc[c]["tp"] for c in cls_sorted]
        fps = [pc[c]["fp"] for c in cls_sorted]
        fns = [pc[c]["fn"] for c in cls_sorted]
        x = np.arange(len(cls_sorted)); bw = 0.25
        ax5.bar(x - bw, tps, bw, label="TP", color="#56d364", alpha=0.85)
        ax5.bar(x,      fps, bw, label="FP", color="#f97583", alpha=0.85)
        ax5.bar(x + bw, fns, bw, label="FN", color="#d2a8ff", alpha=0.85)
        ax5.set_xticks(x)
        ax5.set_xticklabels(cls_sorted, rotation=45, ha='right', fontsize=9)
        ax5.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor='white')
    ax5.set_title('TP / FP / FN per Class', fontsize=12, fontweight='bold', color='white')

    for i in range(n_preview):
        ax_img = fig.add_subplot(gs[1:, i])

        result_info = per_image_results[i]
        if result_info['annotated_path']:
            img_preview = cv2.imread(str(result_info['annotated_path']))
            if img_preview is not None:
                img_preview = cv2.cvtColor(img_preview, cv2.COLOR_BGR2RGB)
                ax_img.imshow(img_preview)
        ax_img.axis('off')

        short_name = result_info['filename'][:30] + ('...' if len(result_info['filename']) > 30 else '')
        det_labels = [f"{d['class']} ({d['iou']:.2f})" if d.get('iou') and d['iou'] > 0 else f"{d['class']}" for d in result_info['detections']]
        det_text = ', '.join(det_labels) if det_labels else 'No detections'
        iou_suffix = f"\nMean IoU: {result_info['mean_iou']:.2f}" if result_info.get('has_gt') and result_info.get('mean_iou') is not None else ""
        ax_img.set_title(f"{short_name}\n{det_text}{iou_suffix}", fontsize=10, color='#58a6ff', pad=8)

    ov = metrics["overall"]
    fig.suptitle(
        f"Test Results — {total_images} images, {total_detections} detections  |  "
        f"P: {ov['precision']:.3f}  R: {ov['recall']:.3f}  F1: {ov['f1']:.3f}",
        fontsize=16, fontweight='bold', color='white', y=0.99
    )

    dashboard_path = OUTPUT_DIR / "test_dashboard.png"
    fig.savefig(str(dashboard_path), dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

    print(f"\n  Dashboard saved: {dashboard_path}")
    
    valid_ious = [r['mean_iou'] for r in per_image_results if r.get('has_gt') and r['mean_iou'] is not None]
    if valid_ious:
        fig_iou = plt.figure(figsize=(8, 6), facecolor='#0d1117')
        ax_iou = fig_iou.add_subplot(111)
        ax_iou.set_facecolor('#161b22')
        ax_iou.hist(valid_ious, bins=min(10, max(3, len(valid_ious))), range=(0, 1), color='#d2a8ff', edgecolor='white', linewidth=0.5, alpha=0.8)
        avg_iou_all = sum(valid_ious) / len(valid_ious)
        ax_iou.axvline(x=avg_iou_all, color='#56d364', linestyle='--', linewidth=2,
                    label=f'Mean: {avg_iou_all:.2f}')
        ax_iou.legend(fontsize=10, facecolor='#161b22', edgecolor='#30363d', labelcolor='white')
        ax_iou.set_xlabel('Mean IoU per Image', color='#c9d1d9')
        ax_iou.set_ylabel('Frequency', color='#c9d1d9')
        ax_iou.tick_params(colors='#8b949e')
        for spine in ax_iou.spines.values():
            spine.set_color('#30363d')
            
        ax_iou.set_title('Mean IoU Distribution', fontsize=14, fontweight='bold', color='white')
        
        iou_chart_path = OUTPUT_DIR / "iou_distribution.png"
        fig_iou.savefig(str(iou_chart_path), dpi=150, bbox_inches='tight', facecolor=fig_iou.get_facecolor())
        plt.close(fig_iou)
        print(f"  IoU Distribution saved: {iou_chart_path}")
    else:
        iou_chart_path = None

    report_lines = ["YOLO Test Results Summary", "=" * 40]
    report_lines.append(f"Total Images: {total_images}")
    report_lines.append(f"With Detections: {images_with_det}")
    report_lines.append(f"Total Detections: {total_detections}")
    report_lines.append(f"IoU Threshold: {iou_threshold}")
    report_lines.append("")
    report_lines.append(f"Overall Precision: {ov['precision']:.4f}")
    report_lines.append(f"Overall Recall:    {ov['recall']:.4f}")
    report_lines.append(f"Overall F1:        {ov['f1']:.4f}")
    report_lines.append(f"TP: {ov['tp']}  FP: {ov['fp']}  FN: {ov['fn']}  TN: {ov['tn']}")
    
    valid_ious_rep = [r['mean_iou'] for r in per_image_results if r.get('has_gt') and r['mean_iou'] is not None]
    if valid_ious_rep:
        avg_iou_all = sum(valid_ious_rep) / len(valid_ious_rep)
        report_lines.append(f"Overall Mean IoU (across evaluated images): {avg_iou_all:.3f}")
        
    report_lines.append("")
    report_lines.append("Per-class metrics:")
    report_lines.append(f"{'Class':<35} {'TP':>5} {'FP':>5} {'FN':>5} {'P':>7} {'R':>7} {'F1':>7} {'AvgIoU':>8} {'AvgConf':>8}")
    report_lines.append("-" * 90)
    for cls, m in sorted(metrics["per_class"].items()):
        report_lines.append(
            f"{cls:<35} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5} "
            f"{m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f} "
            f"{m['avg_iou']:>8.4f} {m['avg_conf']:>8.4f}"
        )

    report_path = OUTPUT_DIR / "test_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n  Saving CSV files...")
    save_csvs(det_records, image_records, metrics, OUTPUT_DIR)

    print(f"\n  Output Files:")
    print(f"    Dashboard  : {dashboard_path}")
    if iou_chart_path:
        print(f"    IoU Chart  : {iou_chart_path}")
    print(f"    Annotated  : {OUTPUT_DIR / 'annotated'}/")
    print(f"    Pred Labels: {OUTPUT_DIR / 'predicted_labels'}/")
    print(f"    Report     : {report_path}")
    print(f"    CSV (det)  : {OUTPUT_DIR / 'test_per_detection.csv'}")
    print(f"    CSV (img)  : {OUTPUT_DIR / 'test_per_image.csv'}")
    print(f"    CSV (class): {OUTPUT_DIR / 'test_per_class_metrics.csv'}")
    print("\n" + "=" * 60)
    print("  TESTING SELESAI!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test YOLO Model")
    parser.add_argument("--conf", type=float, default=None,
                        help=f"Confidence threshold (default: {CONF})")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific path to a model file to test (e.g. results/models/custom.pt)")
    parser.add_argument("--data", type=str, default=None,
                        help="Specific path to a directory containing test images")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for TP matching (default: 0.5)")
    args = parser.parse_args()
    main(conf=args.conf, model_path=args.model, test_dir=args.data,
         iou_threshold=args.iou_threshold)