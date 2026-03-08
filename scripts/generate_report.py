#!/usr/bin/env python3

import csv
import sys
from datetime import datetime
from pathlib import Path

# Add scripts dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BASE, VERIFICATION, RUNS_DIR, MODELS_DIR, OUTPUT_DIR

# Paths
REPORT_OUT = OUTPUT_DIR / "END_TO_END_REPORT.md"
VERIFY_CSV = VERIFICATION / "csv_summary" / "metrics_summary.csv"
FT_RESULTS_CSV = RUNS_DIR / "finetune" / "results.csv"
MODEL_PATH = MODELS_DIR / "best.pt"

def get_verification_summary():
    if not VERIFY_CSV.exists():
        return None
    
    stats = {
        "total": 0,
        "trusted": 0,
        "retrain": 0,
        "review": 0,
        "avg_f1": 0.0,
        "avg_iou": 0.0,
        "best_img": "",
        "worst_img": ""
    }
    
    rows = []
    with open(VERIFY_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Nama_Gambar") == "=== TOTAL ===":
                stats["avg_f1"] = float(row.get("F1_Score", 0))
                stats["avg_iou"] = float(row.get("Rata2_IoU", 0))
                continue
            
            rows.append(row)
            f1 = float(row.get("F1_Score", 0))
            if f1 >= 0.8:
                stats["trusted"] += 1
            elif f1 < 0.5:
                stats["retrain"] += 1
            else:
                stats["review"] += 1
                
    stats["total"] = len(rows)
    if rows:
        sorted_rows = sorted(rows, key=lambda x: float(x.get("F1_Score", 0)))
        stats["worst_img"] = f"{sorted_rows[0]['Nama_Gambar']} ({float(sorted_rows[0]['F1_Score'])*100:.1f}%)"
        stats["best_img"] = f"{sorted_rows[-1]['Nama_Gambar']} ({float(sorted_rows[-1]['F1_Score'])*100:.1f}%)"
        
    return stats

def get_finetune_summary():
    """Read finetune results.csv and return stats."""
    if not FT_RESULTS_CSV.exists():
        return None

    stats = {
        "epochs": 0,
        "final_map50": 0.0,
        "final_loss": 0.0
    }
    
    with open(FT_RESULTS_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
        
        if rows:
            stats["epochs"] = len(rows)
            def find_col(name):
                for i, h in enumerate(header):
                    if name in h.strip(): return i
                return -1
            
            idx_map = find_col("metrics/mAP50(B)")
            idx_loss = find_col("train/box_loss")
            
            last_row = rows[-1]
            if idx_map != -1:
                stats["final_map50"] = float(last_row[idx_map].strip())
            if idx_loss != -1:
                stats["final_loss"] = float(last_row[idx_loss].strip())
                
    return stats

def generate_report():
    print(f"Generating End-to-End Report to {REPORT_OUT}...")
    
    verify_stats = get_verification_summary()
    ft_stats = get_finetune_summary()
    
    lines = []
    lines.append(f"# YOLO Pipeline End-to-End Report")
    lines.append(f"**Generated:** {datetime.now().strftime('%d %B %Y, %H:%M')}\n")
    
    lines.append("## 1. Executive Summary")
    if verify_stats:
        pass_rate = (verify_stats['trusted'] / verify_stats['total']) * 100 if verify_stats['total'] > 0 else 0
        grade = "A" if pass_rate >= 90 else "B" if pass_rate >= 75 else "C"
        lines.append(f"- **Overall Grade**: {grade}")
        lines.append(f"- **Verification Pass Rate**: {pass_rate:.1f}% ({verify_stats['trusted']}/{verify_stats['total']} images)")
        lines.append(f"- **Action Required**: {'Fine-tuning needed' if verify_stats['retrain'] > 0 else 'Model is stable'}")
    else:
        lines.append("- **Status**: No verification data found. Please run `verify_segmentation.py` first.")
    lines.append("")
    
    lines.append("## 2. Verification Results")
    if verify_stats:
        lines.append(f"| Metric | Value |")
        lines.append(f"|---|---|")
        lines.append(f"| Total Images | {verify_stats['total']} |")
        lines.append(f"| Trusted (≥80%) | {verify_stats['trusted']} |")
        lines.append(f"| Review (50-80%) | {verify_stats['review']} |")
        lines.append(f"| Retrain (<50%) | {verify_stats['retrain']} |")
        lines.append(f"| Best Image | {verify_stats['best_img']} |")
        lines.append(f"| Worst Image | {verify_stats['worst_img']} |")
        lines.append("")
        lines.append("> Full metrics available in `verification/csv_summary/metrics_summary.csv`")
    else:
        lines.append("No verification data available.")
    lines.append("")
    
    lines.append("## 3. Fine-Tuning Status")
    if ft_stats:
        lines.append(f"- **Status**: Fine-tuning completed.")
        lines.append(f"- **Epochs Run**: {ft_stats['epochs']}")
        lines.append(f"- **Final mAP50**: {ft_stats['final_map50']:.4f}")
        lines.append(f"- **Final Loss**: {ft_stats['final_loss']:.4f}")
        lines.append(f"- **Logs**: Check `runs/finetune/` for detailed charts.")
    else:
        lines.append("- **Status**: No fine-tuning run detected (or `results.csv` missing).")
        lines.append("- If accuracy was low, consider running `python scripts/fine_tune.py`.")
    lines.append("")
    
    lines.append("## 4. Current Model")
    if MODEL_PATH.exists():
        mod_time = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        lines.append(f"- **Path**: `models/best.pt`")
        lines.append(f"- **Last Updated**: {mod_time}")
        lines.append(f"- **Size**: {size_mb:.2f} MB")
    else:
        lines.append("`models/best.pt` not found.")
    lines.append("")

    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    print(f"Report saved to: {REPORT_OUT}")

if __name__ == "__main__":
    generate_report()