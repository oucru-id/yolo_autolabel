
import argparse
import collections
import csv
from pathlib import Path
import sys

def compute_iou_box(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0: return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter)

def load_labels_xyxy(txt_path, w, h):
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

def evaluate_iou(model_path: str, dataset_dir: Path):
    from ultralytics import YOLO
    model = YOLO(model_path)
    class_names = model.names
    
    # Try different combinations of image/label dirs based on whether
    # the user passed a `dataset` root or the direct `images` dir.
    images_dir = dataset_dir / "images" / "train"
    labels_dir = dataset_dir / "labels" / "train"
    
    if not images_dir.exists():
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"
        if not images_dir.exists():
            images_dir = dataset_dir
            labels_dir = dataset_dir.parent / "labels" if dataset_dir.name == "images" else dataset_dir
            
    print(f"Reading images from: {images_dir}")
    print(f"Reading labels from: {labels_dir}")
    
    image_files = sorted([f for f in images_dir.glob("*.*") if f.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    if not image_files:
        print("[ERROR] No images found!")
        sys.exit(1)
        
    print(f"Found {len(image_files)} images for evaluation.")
    
    iou_records = []
    
    for img_path in image_files:
        results = model(str(img_path), verbose=False)
        result = results[0]
        
        h, w = result.orig_shape
        boxes = result.boxes
        
        txt_path = labels_dir / f"{img_path.stem}.txt"
        gts = load_labels_xyxy(txt_path, w, h)
        
        preds = []
        if len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                preds.append({
                    'class_id': cls_id,
                    'class': class_names[cls_id],
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2],
                    'iou': 0.0,
                    'matched': False
                })
                
        matches = []
        for i, p in enumerate(preds):
            for j, gt in enumerate(gts):
                if p['class_id'] == gt['class_id']:
                    iou = compute_iou_box(p['bbox'], gt['bbox'])
                    if iou > 0:
                        matches.append((iou, i, j))
                        
        matches.sort(key=lambda x: x[0], reverse=True)
        matched_preds = set()
        matched_gts = set()
        
        for iou, i, j in matches:
            if i not in matched_preds and j not in matched_gts:
                preds[i]['iou'] = iou
                preds[i]['matched'] = True
                matched_preds.add(i)
                matched_gts.add(j)
                
        for p in preds:
            iou_records.append({
                'filename': img_path.name,
                'class': p['class'],
                'confidence': p['confidence'],
                'iou': p['iou'],
                'type': 'True Positive' if p['matched'] else 'False Positive'
            })
            
        for j, gt in enumerate(gts):
            if j not in matched_gts:
                cls_id = gt['class_id']
                iou_records.append({
                    'filename': img_path.name,
                    'class': class_names[cls_id] if cls_id in class_names else str(cls_id),
                    'confidence': 0.0,
                    'iou': 0.0,
                    'type': 'False Negative'
                })
                
    return iou_records

def plot_iou_distribution(records: list[dict], save_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed. Skipping IoU distribution plot.")
        return
        
    class_ious = collections.defaultdict(list)
    for r in records:
        if r['type'] == 'True Positive':
            class_ious[r['class']].append(r['iou'])
            
    if not class_ious:
        print("[WARN] No Top Positive matched records found to plot IoU.")
        return
        
    classes = sorted(class_ious.keys())
    data_to_plot = [class_ious[c] for c in classes]
    
    fig = plt.figure(figsize=(12, 6), facecolor="#0d1117")
    ax = plt.gca()
    ax.set_facecolor("#161b22")
    
    boxparts = ax.boxplot(data_to_plot, patch_artist=True,
                          boxprops=dict(facecolor="#58a6ff", color="white", alpha=0.7),
                          capprops=dict(color="white"),
                          whiskerprops=dict(color="white"),
                          flierprops=dict(markeredgecolor="white", markerfacecolor="#f97583", alpha=0.5),
                          medianprops=dict(color="#56d364", linewidth=2))
                          
    ax.set_xticks(range(1, len(classes) + 1))
    ax.set_xticklabels(classes, rotation=45, ha='right', color='white')
    ax.set_title('IoU Distribution Per Segment (True Positives)', color='white', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('IoU Value', color='white', fontsize=11)
    
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_color('#30363d')
        
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    save_path = save_dir / "iou_distribution_per_segment.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"IoU Distribution plot saved: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate standalone IoU distribution for an existing model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained YOLO model (.pt)")
    parser.add_argument("--data", type=str, required=True, help="Path to the validation dataset (should contain images/ and labels/)")
    parser.add_argument("--out", type=str, default=".", help="Output directory to save CSV and PNG results")
    args = parser.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Evaluating IoU for model: {args.model}")
    records = evaluate_iou(args.model, Path(args.data))
    
    if records:
        csv_path = out_dir / "iou_distribution_per_image.csv"
        headers = ["filename", "class", "confidence", "iou", "type"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for rec in records:
                writer.writerow(rec)
        print(f"\nSaved CSV to: {csv_path}")
        
        plot_iou_distribution(records, out_dir)

if __name__ == "__main__":
    main()
