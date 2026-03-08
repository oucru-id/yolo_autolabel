"""
YOLO Hyperparameter Fine-Tuning Script
"""

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    BASE, DEVICE, WORKERS, MODELS_DIR, RUNS_DIR, DATASET,
    IMGSZ, OPTIMIZER, LRF, COS_LR, PATIENCE,
    get_best_model, ensure_drive_mounted, print_config,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


DEFAULT_TRIAL_EPOCHS = 10
DEFAULT_STEPS = 10
DEFAULT_FINAL_EPOCHS = 20
DATA_YAML = DATASET / "data.yaml"

PARAM_CONFIG = {
    "lr0": {
        "low": 0.0001,
        "high": 0.01,
        "default": 0.001,
        "log_scale": True,
        "description": "Initial learning rate",
    },
    "lrf": {
        "low": 0.001,
        "high": 0.1,
        "default": LRF,
        "log_scale": True,
        "description": "Final LR ratio",
    },
}

PARAM_ORDER = ["lr0", "lrf"]

def generate_values(low: float, high: float, steps: int,
                    log_scale: bool = True) -> list[float]:
    if log_scale:
        return list(np.logspace(np.log10(low), np.log10(high), steps))
    return list(np.linspace(low, high, steps))


def get_search_dir(runs_dir: Path) -> Path:
    base_name = "hyperparam_search"
    candidate = runs_dir / base_name
    if not candidate.exists():
        return candidate
    idx = 2
    while (runs_dir / f"{base_name}{idx}").exists():
        idx += 1
    return runs_dir / f"{base_name}{idx}"


def extract_metrics(results) -> dict:
    metrics = {
        "mAP50": 0.0,
        "mAP50-95": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }

    if hasattr(results, "results_dict") and results.results_dict:
        rd = results.results_dict
        for key, metric_name in [
            ("metrics/mAP50(B)", "mAP50"),
            ("metrics/mAP50-95(B)", "mAP50-95"),
            ("metrics/precision(B)", "precision"),
            ("metrics/recall(B)", "recall"),
        ]:
            if key in rd:
                metrics[metric_name] = float(rd[key])
    return metrics



def run_single_trial(
    model_path: Path,
    data_path: Path,
    trial_name: str,
    project_dir: Path,
    epochs: int,
    lr0: float,
    lrf: float,
    batch: int = -1,
    workers: int = WORKERS,
) -> dict:

    from ultralytics import YOLO

    print(f"\n  Trial: {trial_name}  |  lr0={lr0:.6f}  lrf={lrf:.6f}  epochs={epochs}")

    model = YOLO(str(model_path))

    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=IMGSZ,
        batch=batch,
        lr0=float(lr0),
        lrf=float(lrf),
        optimizer=OPTIMIZER,
        cos_lr=COS_LR,
        patience=min(5, epochs),
        device=DEVICE,
        workers=workers,
        project=str(project_dir),
        name=trial_name,
        resume=False,
        verbose=False,
    )

    metrics = extract_metrics(results)

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else project_dir / trial_name
    weights_dir = save_dir / "weights"
    if weights_dir.exists():
        shutil.rmtree(weights_dir)

    print(f"    mAP50={metrics['mAP50']:.4f}  mAP50-95={metrics['mAP50-95']:.4f}  "
          f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}")

    return metrics

def sweep_parameter(
    param_name: str,
    values: list[float],
    fixed_params: dict,
    model_path: Path,
    data_path: Path,
    project_dir: Path,
    trial_epochs: int,
    batch: int = -1,
    workers: int = WORKERS,
) -> list[dict]:

    results = []

    print(f"\n{'=' * 60}")
    print(f"  SWEEP: {param_name}  ({len(values)} values)")
    print(f"  Fixed: {fixed_params}")
    print(f"{'=' * 60}")

    for i, val in enumerate(values):
        trial_params = {**fixed_params, param_name: val}
        trial_name = f"run_{param_name}_{val:.6f}"

        metrics = run_single_trial(
            model_path=model_path,
            data_path=data_path,
            trial_name=trial_name,
            project_dir=project_dir,
            epochs=trial_epochs,
            lr0=trial_params.get("lr0", PARAM_CONFIG["lr0"]["default"]),
            lrf=trial_params.get("lrf", PARAM_CONFIG["lrf"]["default"]),
            batch=batch,
            workers=workers,
        )

        trial_result = {
            "round": param_name,
            "trial": i + 1,
            "param_name": param_name,
            "param_value": val,
            **trial_params,
            **metrics,
        }
        results.append(trial_result)

    results.sort(key=lambda r: r["mAP50"], reverse=True)

    best = results[0]
    print(f"\n  ★ Best {param_name} = {best['param_value']:.6f}  "
          f"(mAP50={best['mAP50']:.4f})")

    return results


def coordinate_descent(
    model_path: Path,
    data_path: Path,
    output_dir: Path,
    trial_epochs: int,
    steps: int,
    batch: int = -1,
    workers: int = WORKERS,
) -> tuple[dict, list[dict]]:

    best_params = {
        name: cfg["default"] for name, cfg in PARAM_CONFIG.items()
    }
    all_results = []

    print("\n" + "=" * 60)
    print("  HYPERPARAMETER SEARCH — Coordinate Descent")
    print("=" * 60)
    print(f"  Parameters : {', '.join(PARAM_ORDER)}")
    print(f"  Steps/param: {steps}")
    print(f"  Epochs/trial: {trial_epochs}")
    print(f"  Total trials: {len(PARAM_ORDER) * steps}")
    print(f"  Total epochs: {len(PARAM_ORDER) * steps * trial_epochs}")
    print(f"  Model      : {model_path}")
    print(f"  Data       : {data_path}")
    print(f"  Output     : {output_dir}")
    print("=" * 60)

    for param_name in PARAM_ORDER:
        cfg = PARAM_CONFIG[param_name]
        values = generate_values(cfg["low"], cfg["high"], steps, cfg["log_scale"])

        fixed = {k: v for k, v in best_params.items() if k != param_name}

        sweep_results = sweep_parameter(
            param_name=param_name,
            values=values,
            fixed_params=fixed,
            model_path=model_path,
            data_path=data_path,
            project_dir=output_dir,
            trial_epochs=trial_epochs,
            batch=batch,
            workers=workers,
        )

        all_results.extend(sweep_results)

        best_value = sweep_results[0]["param_value"]
        best_params[param_name] = best_value

    print(f"\n{'=' * 60}")
    print(f"  SEARCH COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Best parameters:")
    for k, v in best_params.items():
        print(f"    {k}: {v:.6f}")
    print(f"{'=' * 60}")

    return best_params, all_results



def generate_heatmap(param_name: str, results: list[dict],
                     output_dir: Path):
    if not HAS_MPL:
        return

    param_results = [r for r in results if r["param_name"] == param_name]
    if not param_results:
        return

    param_results.sort(key=lambda r: r["param_value"])

    values = [r["param_value"] for r in param_results]
    map50 = [r["mAP50"] for r in param_results]
    map50_95 = [r["mAP50-95"] for r in param_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

    desc = PARAM_CONFIG[param_name]["description"]
    fig.suptitle(f"Hyperparameter Sweep: {desc} ({param_name})",
                 fontsize=14, fontweight="bold", y=1.02)

    colors_50 = []
    best_idx_50 = int(np.argmax(map50))
    for i in range(len(values)):
        if i == best_idx_50:
            colors_50.append("#27ae60")
        else:
            colors_50.append("#3498db")

    labels = [f"{v:.5f}" for v in values]
    bars1 = ax1.bar(range(len(values)), map50, color=colors_50,
                    edgecolor="white", linewidth=0.5)
    ax1.set_xlabel(param_name, fontsize=11)
    ax1.set_ylabel("mAP50", fontsize=11)
    ax1.set_title("mAP50 vs " + param_name, fontsize=12, fontweight="bold")
    ax1.set_xticks(range(len(values)))
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    for i, v in enumerate(map50):
        ax1.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=7,
                 fontweight="bold" if i == best_idx_50 else "normal")

    colors_95 = []
    best_idx_95 = int(np.argmax(map50_95))
    for i in range(len(values)):
        if i == best_idx_95:
            colors_95.append("#e67e22")
        else:
            colors_95.append("#9b59b6")

    bars2 = ax2.bar(range(len(values)), map50_95, color=colors_95,
                    edgecolor="white", linewidth=0.5)
    ax2.set_xlabel(param_name, fontsize=11)
    ax2.set_ylabel("mAP50-95", fontsize=11)
    ax2.set_title("mAP50-95 vs " + param_name, fontsize=12, fontweight="bold")
    ax2.set_xticks(range(len(values)))
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    for i, v in enumerate(map50_95):
        ax2.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=7,
                 fontweight="bold" if i == best_idx_95 else "normal")

    plt.tight_layout()
    save_path = output_dir / f"heatmap_{param_name}.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Heatmap → {save_path}")


def generate_scatter_gradient(all_results: list[dict], output_dir: Path):

    if not HAS_MPL or not all_results:
        return

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")

    for param_name in PARAM_ORDER:
        param_results = [r for r in all_results if r["param_name"] == param_name]
        param_results.sort(key=lambda r: r["param_value"])

        if not param_results:
            continue

        values = [r["param_value"] for r in param_results]
        map50 = [r["mAP50"] for r in param_results]

        gradients = [0.0]
        for i in range(1, len(map50)):
            dv = values[i] - values[i - 1]
            dm = map50[i] - map50[i - 1]
            gradients.append(dm / dv if dv != 0 else 0)

        grad_arr = np.array(gradients)
        if grad_arr.max() != grad_arr.min():
            norm_grad = (grad_arr - grad_arr.min()) / (grad_arr.max() - grad_arr.min())
        else:
            norm_grad = np.zeros_like(grad_arr)

        sc = ax.scatter(
            values, map50,
            c=norm_grad, cmap="RdYlGn", s=100, edgecolors="black",
            linewidth=0.5, zorder=3, label=param_name,
        )
        ax.plot(values, map50, "--", alpha=0.4, linewidth=1)

        best_idx = int(np.argmax(map50))
        ax.annotate(
            f"{map50[best_idx]:.4f}",
            (values[best_idx], map50[best_idx]),
            textcoords="offset points", xytext=(0, 12),
            ha="center", fontsize=9, fontweight="bold",
            color="#27ae60",
        )

    ax.set_xlabel("Parameter Value", fontsize=11)
    ax.set_ylabel("mAP50", fontsize=11)
    ax.set_title("Improvement Gradient Across Trials",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("Gradient (improvement rate)", fontsize=10)

    plt.tight_layout()
    save_path = output_dir / "scatter_gradient.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Scatter gradient → {save_path}")


def save_trial_results(all_results: list[dict], output_dir: Path):
    csv_path = output_dir / "trial_results.csv"
    fieldnames = [
        "round", "trial", "param_name", "param_value",
        "lr0", "lrf",
        "mAP50", "mAP50-95", "precision", "recall",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            row = {}
            for k in fieldnames:
                v = r.get(k, "")
                if isinstance(v, float):
                    row[k] = f"{v:.6f}"
                else:
                    row[k] = v
            writer.writerow(row)
    print(f"  Trial results → {csv_path}")


def save_best_params(best_params: dict, all_results: list[dict],
                     output_dir: Path):
    json_path = output_dir / "best_params.json"

    best_trial = max(all_results, key=lambda r: r["mAP50"])

    data = {
        "best_params": {k: round(v, 6) for k, v in best_params.items()},
        "best_mAP50": round(best_trial["mAP50"], 4),
        "best_mAP50-95": round(best_trial["mAP50-95"], 4),
        "timestamp": datetime.now().isoformat(),
        "total_trials": len(all_results),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f" Best params → {json_path}")



def final_training(
    model_path: Path,
    data_path: Path,
    best_params: dict,
    output_dir: Path,
    final_epochs: int,
    batch: int = -1,
    workers: int = WORKERS,
) -> Path | None:

    from ultralytics import YOLO

    print(f"\n{'=' * 60}")
    print(f"  FINAL TRAINING with best parameters")
    print(f"{'=' * 60}")
    print(f"  lr0    : {best_params['lr0']:.6f}")
    print(f"  lrf    : {best_params['lrf']:.6f}")
    print(f"  epochs : {final_epochs}")
    print(f"{'=' * 60}")

    model = YOLO(str(model_path))

    results = model.train(
        data=str(data_path),
        epochs=final_epochs,
        imgsz=IMGSZ,
        batch=batch,
        lr0=float(best_params["lr0"]),
        lrf=float(best_params["lrf"]),
        optimizer=OPTIMIZER,
        cos_lr=COS_LR,
        patience=PATIENCE,
        device=DEVICE,
        workers=workers,
        project=str(output_dir),
        name="final_best",
        resume=False,
    )

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else output_dir / "final_best"
    ft_best = save_dir / "weights" / "best.pt"

    if ft_best.exists():
        dest_local = output_dir / "best.pt"
        shutil.copy2(str(ft_best), str(dest_local))

        dest_models = MODELS_DIR / "best.pt"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(ft_best), str(dest_models))

        for fname in ["results.csv", "results.png"]:
            src = save_dir / fname
            if src.exists():
                shutil.copy2(str(src), str(output_dir / f"final_{fname}"))
                print(f" Copied {fname} → {output_dir / f'final_{fname}'}")

        print(f"\n Final model → {dest_local}")
        print(f" Copied to   → {dest_models}")
        return dest_local

    print(f"  best.pt not found at {ft_best}")
    return None

def archive_previous_model(model_path: Path):
    if not model_path.exists():
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"{model_path.stem}_v{ts}{model_path.suffix}"
    archive_path = MODELS_DIR / archive_name
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(str(model_path), str(archive_path))
    print(f"  Previous model archived → {archive_path}")


def load_best_params(search_dir: Path) -> dict | None:
    json_path = search_dir / "best_params.json"
    if not json_path.exists():
        print(f"  best_params.json not found in {search_dir}")
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    params = {k: float(v) for k, v in data.get("best_params", {}).items()}
    print(f"  Loaded best params from {json_path}:")
    for k, v in params.items():
        print(f"    {k}: {v:.6f}")
    return params

def main(cli_args=None):
    parser = argparse.ArgumentParser(
        description="Hyperparameter fine-tuning via sequential coordinate descent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/fine_tune.py                                         # default search
  python scripts/fine_tune.py --trial-epochs 15                       # more epochs per trial
  python scripts/fine_tune.py --steps 5                               # fewer steps (faster)
  python scripts/fine_tune.py --final-epochs 30                       # longer final training
  python scripts/fine_tune.py --no-final                              # skip final training
  python scripts/fine_tune.py --resume-from results/runs/hyperparam_search  # skip search, run final only
        """,
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Path to base model (default: best.pt from training)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to data YAML (default: dataset/data.yaml)")
    parser.add_argument("--trial-epochs", type=int, default=DEFAULT_TRIAL_EPOCHS,
                        help=f"Epochs per trial (default: {DEFAULT_TRIAL_EPOCHS})")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"Number of values per parameter (default: {DEFAULT_STEPS})")
    parser.add_argument("--final-epochs", type=int, default=DEFAULT_FINAL_EPOCHS,
                        help=f"Epochs for final training (default: {DEFAULT_FINAL_EPOCHS})")
    parser.add_argument("--no-final", action="store_true",
                        help="Skip final training run (only search)")
    parser.add_argument("--no-archive", action="store_true",
                        help="Skip archiving previous model")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to existing hyperparam_search/ folder — skips search, runs final training only")
    parser.add_argument("--batch", type=int, default=-1,
                        help="Batch size (-1 = auto, default: -1)")
    parser.add_argument("--workers", type=int, default=WORKERS,
                        help=f"Dataloader workers (default: {WORKERS})")

    args = parser.parse_args(cli_args)

    ensure_drive_mounted()

    print("=" * 60)
    print("  YOLO HYPERPARAMETER FINE-TUNING")
    print("=" * 60)
    print_config()

    if args.model:
        model_path = Path(args.model)
    else:
        model_path = get_best_model()

    print(f"\n  Model: {model_path}")
    if not model_path.exists():
        print(f"  Model not found: {model_path}")
        print("     Run training first: python scripts/train.py")
        sys.exit(1)

    data_path = Path(args.data) if args.data else DATA_YAML
    if not data_path.exists():
        print(f"  Data YAML not found: {data_path}")
        print("     Run training first: python scripts/train.py")
        sys.exit(1)

    print(f"  Data : {data_path}")

    if args.resume_from:
        resume_dir = Path(args.resume_from)
        print(f"\n  RESUME MODE: skipping search, loading from {resume_dir}")
        best_params = load_best_params(resume_dir)
        if best_params is None:
            print("  Cannot resume — best_params.json not found. Run full search first.")
            sys.exit(1)
        output_dir = resume_dir
        if not args.no_final:
            final_training(
                model_path=model_path,
                data_path=data_path,
                best_params=best_params,
                output_dir=output_dir,
                final_epochs=args.final_epochs,
                batch=args.batch,
                workers=args.workers,
            )
        return

    if not args.no_archive:
        archive_previous_model(model_path)

    output_dir = get_search_dir(RUNS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output: {output_dir}")

    best_params, all_results = coordinate_descent(
        model_path=model_path,
        data_path=data_path,
        output_dir=output_dir,
        trial_epochs=args.trial_epochs,
        steps=args.steps,
        batch=args.batch,
        workers=args.workers,
    )

    save_trial_results(all_results, output_dir)
    save_best_params(best_params, all_results, output_dir)

    print(f"\n{'=' * 60}")
    print(f"  GENERATING VISUALIZATIONS")
    print(f"{'=' * 60}")

    for param_name in PARAM_ORDER:
        generate_heatmap(param_name, all_results, output_dir)
    generate_scatter_gradient(all_results, output_dir)

    if not args.no_final:
        final_model = final_training(
            model_path=model_path,
            data_path=data_path,
            best_params=best_params,
            output_dir=output_dir,
            final_epochs=args.final_epochs,
            batch=args.batch,
            workers=args.workers,
        )

    print(f"\n{'=' * 60}")
    print(f"  HYPERPARAMETER FINE-TUNING COMPLETE!")
    print(f"{'=' * 60}")
    print(f"  Best parameters:")
    for k, v in best_params.items():
        print(f"    {k}: {v:.6f}")
    print(f"  Results: {output_dir}")
    print(f"  CSV    : {output_dir / 'trial_results.csv'}")
    print(f"  Params : {output_dir / 'best_params.json'}")
    if not args.no_final:
        print(f"  Model  : {MODELS_DIR / 'best.pt'}")
    print(f"\n  Next steps:")
    print(f"    python scripts/test_model.py    (evaluate model)")
    print(f"    python scripts/predict.py       (run predictions)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()