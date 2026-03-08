import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODEL_NAME

try:
    from config import BASE, VERIFICATION, TRAIN_RUN
except ImportError:
    BASE = Path(__file__).resolve().parent.parent
    VERIFICATION = BASE / "results" / "verification"
    TRAIN_RUN = BASE / "results" / "runs" / "train"

def load_results(results_dir: Path) -> dict:
    csv_path = results_dir / "results.csv"
    if not csv_path.exists():
        print(f"results.csv not found at: {csv_path}")
        return {}

    data = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                clean_key = key.strip()
                if clean_key not in data:
                    data[clean_key] = []
                try:
                    data[clean_key].append(float(val.strip()))
                except (ValueError, AttributeError):
                    data[clean_key].append(0.0)

    if data:
        print(f"Loaded results.csv ({len(list(data.values())[0])} epochs)")
        print(f"   Columns: {list(data.keys())[:6]}...")

    iou_csv = results_dir / "iou_log.csv"
    if iou_csv.exists():
        iou_values = []
        with open(iou_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    iou_values.append(float(row["mean_iou"]))
                except (ValueError, KeyError):
                    iou_values.append(0.0)
        if iou_values:
            data["mean_iou"] = iou_values
            print(f"   Loaded iou_log.csv ({len(iou_values)} epochs)")

    return data


def find_key(data: dict, substring: str) -> str | None:
    for k in data.keys():
        if substring.lower() in k.lower():
            return k
    return None

def detect_plateau(values: list[float], window: int = 5, threshold: float = 0.005) -> dict:

    if len(values) < window + 2:
        return {
            "detected": False,
            "start_epoch": None,
            "duration": 0,
            "final_value": values[-1] if values else 0,
            "improvement_last_10": 0,
        }

    arr = np.array(values, dtype=float)
    n = len(arr)

    smoothed = np.convolve(arr, np.ones(window) / window, mode="valid")

    grad = np.gradient(smoothed)

    near_zero = np.abs(grad) < threshold
    plateau_start = None
    plateau_duration = 0
    max_plateau_start = None
    max_plateau_duration = 0

    for i in range(len(near_zero)):
        if near_zero[i]:
            if plateau_start is None:
                plateau_start = i + window
            plateau_duration += 1
        else:
            if plateau_duration > max_plateau_duration:
                max_plateau_start = plateau_start
                max_plateau_duration = plateau_duration
            plateau_start = None
            plateau_duration = 0

    if plateau_duration > max_plateau_duration:
        max_plateau_start = plateau_start
        max_plateau_duration = plateau_duration

    tail = min(10, n)
    improvement = arr[-1] - arr[-tail] if n >= tail else 0

    base_val = arr[-tail] if (n >= tail and arr[-tail] != 0) else 1
    rel_improvement = abs(improvement / base_val) * 100

    return {
        "detected": max_plateau_duration >= 5,
        "start_epoch": max_plateau_start,
        "duration": max_plateau_duration,
        "final_value": float(arr[-1]),
        "improvement_last_10": float(improvement),
        "rel_improvement_pct": float(rel_improvement),
    }

def early_stopping_analysis(data: dict) -> dict:

    map50_key = find_key(data, "mAP50(B)")
    if map50_key is None:
        map50_key = find_key(data, "map50")

    if map50_key is None or not data[map50_key]:
        return {"best_epoch": 0, "best_value": 0, "should_stop": False}

    values = data[map50_key]
    best_val = max(values)
    best_epoch = values.index(best_val) + 1
    total_epochs = len(values)

    patience = 10
    delta = 0.001

    epochs_since_best = total_epochs - best_epoch
    should_stop = epochs_since_best >= patience
    if total_epochs > patience:
        recent_max = max(values[-patience:])
        older_max = max(values[:-patience]) if len(values) > patience else 0
        meaningful_improvement = (recent_max - older_max) > delta
    else:
        meaningful_improvement = True

    return {
        "best_epoch": best_epoch,
        "best_value": float(best_val),
        "total_epochs": total_epochs,
        "epochs_since_best": epochs_since_best,
        "should_stop": should_stop,
        "meaningful_improvement": meaningful_improvement,
        "recommendation": (
            f"Best model found at epoch {best_epoch}. "
            + ("Training can be stopped early." if should_stop
               else "Training is still improving.")
        ),
    }

def diagnose_bottleneck(data: dict, num_train: int = 0, num_classes: int = 0) -> list[dict]:
    diagnoses = []

    def tail_avg(key, n=5):
        k = find_key(data, key)
        if k and data[k]:
            vals = data[k][-n:]
            return float(np.mean(vals))
        return None

    train_box_loss = tail_avg("train/box_loss")
    train_cls_loss = tail_avg("train/cls_loss")
    val_box_loss = tail_avg("val/box_loss")
    val_cls_loss = tail_avg("val/cls_loss")
    precision = tail_avg("precision")
    recall = tail_avg("recall")
    map50 = tail_avg("mAP50(B)")
    map5095 = tail_avg("mAP50-95(B)")

    if map50 is None:
        map50 = tail_avg("map50")
    if map5095 is None:
        map5095 = tail_avg("map50-95")

    if num_train > 0 and num_train < 100:
        severity = "HIGH" if num_train < 30 else "MEDIUM"
        diagnoses.append({
            "issue": "Small Dataset",
            "severity": severity,
            "detail": f"Only {num_train} training images. "
                      f"At least 100-300 images per class recommended for good results.",
            "suggestion": "Add more training images or apply data augmentation.",
        })

    if train_box_loss is not None and val_box_loss is not None:
        if val_box_loss > train_box_loss * 1.5:
            diagnoses.append({
                "issue": "Overfitting Detected",
                "severity": "HIGH",
                "detail": f"Val loss ({val_box_loss:.4f}) is much higher than "
                          f"train loss ({train_box_loss:.4f}).",
                "suggestion": "Try: data augmentation, dropout, or add more training data.",
            })

    if train_box_loss is not None and train_box_loss > 1.0:
        diagnoses.append({
            "issue": "Underfitting / Model Too Small",
            "severity": "MEDIUM",
            "detail": f"Train loss is still high ({train_box_loss:.4f}). "
                      f"The model may not have enough capacity.",
            "suggestion": f"Try a larger model ({MODEL_NAME.replace('n.', 's.')} or {MODEL_NAME.replace('n.', 'm.')}), "
                          "or train for more epochs.",
        })

    if recall is not None and recall < 0.5:
        diagnoses.append({
            "issue": "Low Recall",
            "severity": "HIGH",
            "detail": f"Recall is only {recall:.2f}. The model is missing many objects.",
            "suggestion": "Check label quality, add more training data, or lower the confidence threshold.",
        })

    if map50 is not None and map5095 is not None:
        if map50 > 0.5 and map5095 < map50 * 0.5:
            diagnoses.append({
                "issue": "Imprecise Bounding Boxes",
                "severity": "MEDIUM",
                "detail": f"mAP50={map50:.2f} is good, but mAP50-95={map5095:.2f} is low. "
                          f"Box localisation is not accurate enough.",
                "suggestion": "Review the quality of bounding box labels. "
                              "Ensure boxes tightly surround the target objects.",
            })

    if map50 is not None and map50 >= 0.8:
        diagnoses.append({
            "issue": "Good Performance!",
            "severity": "INFO",
            "detail": f"mAP50 = {map50:.2f}. The model is performing well enough for production.",
            "suggestion": "Continue testing on new data to validate further.",
        })

    if not diagnoses:
        diagnoses.append({
            "issue": "No Issues Detected",
            "severity": "INFO",
            "detail": "All metrics are within normal range.",
            "suggestion": "Continue testing on new unseen data.",
        })

    return diagnoses

def plot_training_analysis(data: dict, plateau_info: dict, es_info: dict,
                           diagnoses: list, save_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping visualization.")
        return

    fig = plt.figure(figsize=(20, 18), facecolor="#0d1117")
    fig.suptitle("YOLO Training Analysis Dashboard",
                 fontsize=20, fontweight="bold", color="white", y=0.98)

    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "text.color": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
    })

    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

    def style_ax(ax, title, xlabel="Epoch", ylabel=""):
        ax.set_title(title, fontsize=13, fontweight="bold", color="white", pad=10)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.15, color="#30363d")
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    # --- 1. Loss Curves ---
    ax1 = fig.add_subplot(gs[0, 0])
    for loss_name, color in [("box_loss", "#58a6ff"), ("cls_loss", "#f97583"), ("dfl_loss", "#56d364")]:
        for prefix in ["train", "val"]:
            key = find_key(data, f"{prefix}/{loss_name}")
            if key and data[key]:
                ls = "--" if prefix == "val" else "-"
                ax1.plot(data[key], label=f"{prefix}/{loss_name}", linestyle=ls, color=color,
                         alpha=0.7 if prefix == "val" else 1.0, linewidth=1.5)
    style_ax(ax1, "Training & Validation Loss", ylabel="Loss")
    ax1.legend(fontsize=7, loc="upper right")

    # --- 2. mAP Curves ---
    ax2 = fig.add_subplot(gs[0, 1])
    for metric, color in [("mAP50(B)", "#58a6ff"), ("mAP50-95(B)", "#f97583")]:
        key = find_key(data, metric)
        if key and data[key]:
            ax2.plot(data[key], label=metric, color=color, linewidth=2)
    if es_info.get("best_epoch"):
        ax2.axvline(x=es_info["best_epoch"] - 1, color="#56d364", linestyle="--",
                     alpha=0.7, label=f"Best: epoch {es_info['best_epoch']}")
    style_ax(ax2, "mAP Scores", ylabel="mAP")
    ax2.legend(fontsize=9)

    # --- 3. Precision & Recall ---
    ax3 = fig.add_subplot(gs[0, 2])
    for metric, color in [("precision", "#58a6ff"), ("recall", "#f97583")]:
        key = find_key(data, metric)
        if key and data[key]:
            ax3.plot(data[key], label=metric.capitalize(), color=color, linewidth=2)
    # F1 (computed)
    p_key = find_key(data, "precision")
    r_key = find_key(data, "recall")
    if p_key and r_key and data[p_key] and data[r_key]:
        f1_vals = []
        for p, r in zip(data[p_key], data[r_key]):
            f1_vals.append(2 * p * r / (p + r) if (p + r) > 0 else 0)
        ax3.plot(f1_vals, label="F1", color="#56d364", linewidth=2, linestyle="--")
    style_ax(ax3, "Precision, Recall & F1", ylabel="Score")
    ax3.legend(fontsize=9)

    # --- 4. Learning Rate ---
    ax4 = fig.add_subplot(gs[1, 0])
    lr_key = find_key(data, "lr/pg0")
    if lr_key and data[lr_key]:
        ax4.plot(data[lr_key], color="#d2a8ff", linewidth=1.5)
    style_ax(ax4, "Learning Rate Schedule", ylabel="LR")

    # --- 5. Plateau Analysis ---
    ax5 = fig.add_subplot(gs[1, 1])
    map50_key = find_key(data, "mAP50(B)")
    if map50_key is None:
        map50_key = find_key(data, "map50")
    if map50_key and data[map50_key]:
        vals = data[map50_key]
        ax5.plot(vals, color="#58a6ff", linewidth=2, label="mAP50")
        if plateau_info["detected"] and plateau_info["start_epoch"] is not None:
            s = plateau_info["start_epoch"]
            ax5.axvspan(s, s + plateau_info["duration"], alpha=0.15, color="red",
                         label=f"Plateau (epoch {s}+)")
        if es_info.get("best_epoch"):
            ax5.axvline(x=es_info["best_epoch"] - 1, color="#56d364",
                         linestyle="--", alpha=0.8,
                         label=f"Best = {es_info['best_value']:.4f}")
    style_ax(ax5, "Plateau Detection", ylabel="mAP50")
    ax5.legend(fontsize=9)

    # --- 6. Diagnosis Panel ---
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    text_lines = []
    for d in diagnoses:
        sev = d["severity"]
        marker = {"HIGH": "[!]", "MEDIUM": "[~]", "INFO": "[i]"}.get(sev, "[ ]")
        text_lines.append(f"{marker} [{sev}] {d['issue']}")
        text_lines.append(f"   {d['detail'][:80]}")
        text_lines.append(f"   → {d['suggestion'][:80]}")
        text_lines.append("")
    diag_text = "\n".join(text_lines) if text_lines else "No issues detected."
    ax6.text(0.05, 0.95, diag_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment="top", fontfamily="monospace", color="#c9d1d9",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#161b22", edgecolor="#30363d"))
    ax6.set_title("Diagnosis & Recommendations", fontsize=13,
                   fontweight="bold", color="white", pad=10)

    # --- 7. Mean IoU per Epoch ---
    ax_iou = fig.add_subplot(gs[2, 0])
    iou_key = find_key(data, "mean_iou")
    if iou_key and data[iou_key]:
        iou_vals = data[iou_key]
        epochs_x = list(range(1, len(iou_vals) + 1))
        ax_iou.plot(epochs_x, iou_vals, color="#3fb950", linewidth=2.5, marker="o",
                     markersize=4, zorder=3)
        ax_iou.fill_between(epochs_x, iou_vals, alpha=0.15, color="#3fb950", zorder=2)
        if len(iou_vals) > 0:
            best_iou = max(iou_vals)
            best_epoch = iou_vals.index(best_iou) + 1
            ax_iou.axhline(y=best_iou, color="#56d364", linestyle="--", alpha=0.5,
                            label=f"Best: {best_iou:.4f} (ep {best_epoch})")
            ax_iou.legend(fontsize=9)
    else:
        ax_iou.text(0.5, 0.5, "No IoU data available",
                     transform=ax_iou.transAxes, ha="center", va="center",
                     fontsize=11, color="#8b949e")
    style_ax(ax_iou, "Mean IoU per Epoch", ylabel="IoU")

    # --- 8-9. Summary metrics ---
    ax7 = fig.add_subplot(gs[3, :])
    ax7.axis("off")

    summary_items = []
    if es_info:
        summary_items.append(f"Best Epoch: {es_info.get('best_epoch', '?')} / {es_info.get('total_epochs', '?')}")
        summary_items.append(f"Best mAP50: {es_info.get('best_value', 0):.4f}")
    if plateau_info:
        summary_items.append(f"Plateau: {'Yes' if plateau_info['detected'] else 'No'}")
        summary_items.append(f"Last 10 Improvement: {plateau_info.get('improvement_last_10', 0):.4f}")
    iou_key = find_key(data, "mean_iou")
    if iou_key and data[iou_key]:
        final_iou = data[iou_key][-1]
        best_iou = max(data[iou_key])
        summary_items.append(f"Final IoU: {final_iou:.4f}")
        summary_items.append(f"Best IoU: {best_iou:.4f}")
    summary_text = "    |    ".join(summary_items) if summary_items else ""

    ax7.text(0.5, 0.5, summary_text, transform=ax7.transAxes, fontsize=12,
             ha="center", va="center", color="white",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#161b22", edgecolor="#58a6ff"))

    save_path = save_dir / "training_analysis.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Dashboard saved: {save_path}")

def print_report(data: dict, plateau_info: dict, es_info: dict, diagnoses: list):
    print("\n" + "=" * 50)
    print("TRAINING ANALYSIS REPORT")
    print("=" * 50)

    if es_info:
        print(f"\nEarly Stopping Analysis:")
        print(f"   Best Epoch  : {es_info.get('best_epoch', '?')}")
        print(f"   Best mAP50  : {es_info.get('best_value', 0):.4f}")
        print(f"   Recommendation: {es_info.get('recommendation', '')}")

    if plateau_info:
        print(f"\nPlateau Detection:")
        if plateau_info["detected"]:
            print(f"   Plateau detected starting at epoch {plateau_info['start_epoch']}")
            print(f"   Duration: {plateau_info['duration']} epochs")
        else:
            print(f"   No significant plateau detected")
        print(f"   Last 10 epoch improvement: {plateau_info.get('improvement_last_10', 0):.4f}")

    iou_key = find_key(data, "mean_iou") if data else None
    if iou_key and data.get(iou_key):
        iou_vals = data[iou_key]
        print(f"\nMean IoU:")
        print(f"   Final : {iou_vals[-1]:.4f}")
        print(f"   Best  : {max(iou_vals):.4f}")
        print(f"   Epochs: {len(iou_vals)}")

    if diagnoses:
        print(f"\nDiagnoses:")
        for d in diagnoses:
            print(f"   [{d['severity']}] {d['issue']}")
            print(f"     {d['detail']}")
            print(f"     → {d['suggestion']}")
    print()

def run_analysis(results_dir: str | Path, num_train: int = 0, num_classes: int = 0) -> dict:
    results_dir = Path(results_dir)

    data = load_results(results_dir)
    if not data:
        print("No data available to analyze.")
        return {}

    map50_key = find_key(data, "mAP50(B)")
    if map50_key is None:
        map50_key = find_key(data, "map50")
    map50_vals = data.get(map50_key, []) if map50_key else []
    plateau_info = detect_plateau(map50_vals)
    es_info = early_stopping_analysis(data)
    diagnoses = diagnose_bottleneck(data, num_train, num_classes)
    print_report(data, plateau_info, es_info, diagnoses)

    try:
        plot_training_analysis(data, plateau_info, es_info, diagnoses, results_dir)
    except Exception as e:
        print(f"Could not generate plot: {e}")

    return {
        "plateau": plateau_info,
        "early_stopping": es_info,
        "diagnoses": diagnoses,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze YOLO training results")
    parser.add_argument("--results_dir", type=str,
                        default=str(TRAIN_RUN),
                        help="Path to training results directory")
    parser.add_argument("--num_train", type=int, default=0,
                        help="Number of training images")
    parser.add_argument("--num_classes", type=int, default=6,
                        help="Number of classes")
    args = parser.parse_args()

    run_analysis(args.results_dir, args.num_train, args.num_classes)

def log_training_history(
    model_name: str,
    num_train: int,
    epochs_trained: int,
    metrics: dict,
    save_dir: Path | None = None,
):

    if save_dir is None:
        save_dir = VERIFICATION
    
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "training_history.csv"
    
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    map50    = metrics.get("metrics/mAP50(B)", 0)
    map5095  = metrics.get("metrics/mAP50-95(B)", 0)
    precision = metrics.get("metrics/precision(B)", 0)
    recall    = metrics.get("metrics/recall(B)", 0)
    mean_iou  = metrics.get("mean_iou", 0)
    
    if map50 == 0: map50 = metrics.get("map50", 0)
    if map5095 == 0: map5095 = metrics.get("map50-95", 0)
    if precision == 0: precision = metrics.get("precision", 0)
    if recall == 0: recall = metrics.get("recall", 0)

    headers = [
        "timestamp", "model_name", "num_train", "epochs",
        "map50", "map50_95", "precision", "recall", "mean_iou"
    ]
    
    file_exists = csv_path.exists()
    
    row = {
        "timestamp": now_str,
        "model_name": model_name,
        "num_train": num_train,
        "epochs": epochs_trained,
        "map50": f"{map50:.5f}",
        "map50_95": f"{map5095:.5f}",
        "precision": f"{precision:.5f}",
        "recall": f"{recall:.5f}",
        "mean_iou": f"{mean_iou:.5f}",
    }
    
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        
    print(f"History logged to: {csv_path}")


def plot_training_history(history_csv: Path | None = None, save_path: Path | None = None):

    if history_csv is None:
        history_csv = VERIFICATION / "training_history.csv"
    
    if not history_csv.exists():
        print(f"History file not found: {history_csv}")
        return

    if save_path is None:
        save_path = VERIFICATION / "history_chart.png"

    data = []
    with open(history_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                item = {
                    "num_train": int(row["num_train"]),
                    "map50_95": float(row["map50_95"]),
                    "map50": float(row["map50"]),
                    "model": row["model_name"]
                }
                data.append(item)
            except (ValueError, KeyError):
                continue
    
    if not data:
        print("No valid data in history CSV.")
        return

    data.sort(key=lambda x: x["num_train"])
    
    x = [d["num_train"] for d in data]
    y = [d["map50_95"] for d in data]
    y_pct = [val * 100 for val in y]
    
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not found.")
        return

    plt.style.use('default') 
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    ax.grid(which='major', color='#eeeeee', linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    
    ax.plot(x, y_pct, color='#2c82c9', linewidth=2.5, marker='o', markersize=6, zorder=3)
    
    ax.fill_between(x, y_pct, color='#ebf5fb', alpha=0.8, zorder=2)
    
    for i, _ in enumerate(x):
        val = y_pct[i]
        ax.annotate(f"{val:.1f}%", 
                    xy=(x[i], y_pct[i]), 
                    xytext=(0, 10), 
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='#2c82c9')

    ax.set_title("Incremental Improvement: mAP50-95 vs Training Size", 
                 fontsize=14, fontweight='bold', pad=20, color='#2c3e50')
    ax.set_xlabel("Total Training Images", fontsize=11, fontweight='bold', color='#34495e', labelpad=10)
    ax.set_ylabel("mAP50-95 (%)", fontsize=11, fontweight='bold', color='#34495e', labelpad=10)
    
    last_acc = y_pct[-1]
    last_n = x[-1]
    
    summary_text = f"Latest Accuracy: {last_acc:.1f}%\nTotal Data: {last_n} images"
    
    y_min, y_max = min(y_pct), max(y_pct)
    padding = (y_max - y_min) * 0.2 if (y_max - y_min) > 0 else 5
    ax.set_ylim(max(0, y_min - padding), min(100, y_max + padding * 1.5))
    
    if len(x) > 1:
        ax.set_xlim(min(x) * 0.9, max(x) * 1.1)
        
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"History chart saved: {save_path}")


def plot_dual_panel_history(history_csv: Path | None = None, save_path: Path | None = None):

    if history_csv is None:
        history_csv = VERIFICATION / "training_history.csv"

    if not history_csv.exists():
        print(f"History file not found: {history_csv}")
        return

    if save_path is None:
        save_path = VERIFICATION / "dual_panel_chart.png"

    data = []
    with open(history_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                item = {
                    "num_train": int(row["num_train"]),
                    "map50": float(row.get("map50", 0)),
                    "map50_95": float(row.get("map50_95", 0)),
                }
                data.append(item)
            except (ValueError, KeyError):
                continue

    if not data:
        print("No valid data in history CSV.")
        return

    data.sort(key=lambda x: x["num_train"])

    x = [d["num_train"] for d in data]
    y_map50 = [d["map50"] * 100 for d in data]
    y_map5095 = [d["map50_95"] * 100 for d in data]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not found.")
        return

    plt.style.use("default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor="white")

    fig.suptitle(
        fontsize=14, fontweight="bold", color="#2c3e50", y=1.02,
    )

    ax1.set_facecolor("#f8fbff")
    ax1.grid(which="major", color="#e0e0e0", linestyle="-", linewidth=0.8)
    ax1.set_axisbelow(True)

    ax1.plot(x, y_map50, color="#2c82c9", linewidth=2.5, marker="o", markersize=6, zorder=3)
    ax1.fill_between(x, y_map50, color="#d6eaf8", alpha=0.8, zorder=2)

    for i in range(len(x)):
        ax1.annotate(
            f"{y_map50[i]:.1f}%",
            xy=(x[i], y_map50[i]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=8, fontweight="bold", color="#2c82c9",
        )

    ax1.set_title("mAP50 (higher is better)", fontsize=12, fontweight="bold", color="#2c3e50", pad=12)
    ax1.set_xlabel("Training Data Size", fontsize=10, fontweight="bold", color="#34495e", labelpad=8)
    ax1.set_ylabel("mAP50 (%)", fontsize=10, fontweight="bold", color="#34495e", labelpad=8)

    if y_map50:
        y_min, y_max = min(y_map50), max(y_map50)
        pad = max((y_max - y_min) * 0.2, 3)
        ax1.set_ylim(max(0, y_min - pad), min(100, y_max + pad * 1.5))

    ax2.set_facecolor("#fff8f8")
    ax2.grid(which="major", color="#e0e0e0", linestyle="-", linewidth=0.8)
    ax2.set_axisbelow(True)

    ax2.plot(x, y_map5095, color="#c0392b", linewidth=2.5, marker="o", markersize=6, zorder=3)
    ax2.fill_between(x, y_map5095, color="#fadbd8", alpha=0.8, zorder=2)

    for i in range(len(x)):
        ax2.annotate(
            f"{y_map5095[i]:.1f}%",
            xy=(x[i], y_map5095[i]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=8, fontweight="bold", color="#c0392b",
        )

    ax2.set_title("mAP50-95 (higher is better)", fontsize=12, fontweight="bold", color="#2c3e50", pad=12)
    ax2.set_xlabel("Training Data Size", fontsize=10, fontweight="bold", color="#34495e", labelpad=8)
    ax2.set_ylabel("mAP50-95 (%)", fontsize=10, fontweight="bold", color="#34495e", labelpad=8)

    if y_map5095:
        y_min, y_max = min(y_map5095), max(y_map5095)
        pad = max((y_max - y_min) * 0.2, 3)
        ax2.set_ylim(max(0, y_min - pad), min(100, y_max + pad * 1.5))

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Dual-panel chart saved: {save_path}")