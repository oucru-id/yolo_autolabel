"""
YOLO Hyperparameter Fine-Tuning Script
=======================================
Uses Ultralytics' built-in model.tune()
"""

import argparse
import json
import shutil
import sys
import yaml
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    BASE, DEVICE, WORKERS, MODELS_DIR, RUNS_DIR, DATASET,
    IMGSZ, OPTIMIZER, LR0, LRF, COS_LR, PATIENCE,
    get_best_model, ensure_drive_mounted, print_config,
)


DEFAULT_TUNE_EPOCHS = 10
DEFAULT_ITERATIONS = 10
DEFAULT_FINAL_EPOCHS = 25
DATA_YAML = DATASET / "data.yaml"


def get_search_dir(runs_dir: Path) -> Path:
    base_name = "hyperparam_search"
    candidate = runs_dir / base_name
    if not candidate.exists():
        return candidate
    idx = 2
    while (runs_dir / f"{base_name}{idx}").exists():
        idx += 1
    return runs_dir / f"{base_name}{idx}"


def run_tune(
    model_path: Path,
    data_path: Path,
    output_dir: Path,
    tune_epochs: int,
    iterations: int,
    batch: int = -1,
    workers: int = WORKERS,
) -> dict:
    from ultralytics import YOLO

    print(f"\n{'=' * 60}")
    print(f"  HYPERPARAMETER SEARCH — Ultralytics Tuner")
    print(f"{'=' * 60}")
    print(f"  Iterations  : {iterations}")
    print(f"  Epochs/trial: {tune_epochs}")
    print(f"  Total epochs: {iterations * tune_epochs}")
    print(f"  Model       : {model_path}")
    print(f"  Data        : {data_path}")
    print(f"  Output      : {output_dir}")
    print(f"  Batch       : {batch}")
    print(f"  Workers     : {workers}")
    print(f"{'=' * 60}")

    model = YOLO(str(model_path))

    model.tune(
        data=str(data_path),
        epochs=tune_epochs,
        iterations=iterations,
        optimizer=OPTIMIZER,
        plots=False,
        save=False,
        val=True,
        imgsz=IMGSZ,
        batch=batch,
        device=DEVICE,
        workers=workers,
        project=str(output_dir),
        name="tune",
    )

    best_yaml = output_dir / "tune" / "best_hyperparameters.yaml"
    if not best_yaml.exists():
        for candidate in [
            output_dir / "best_hyperparameters.yaml",
            Path("runs/detect/tune/best_hyperparameters.yaml"),
        ]:
            if candidate.exists():
                best_yaml = candidate
                break

    if not best_yaml.exists():
        print(f"  best_hyperparameters.yaml not found, using defaults")
        return {"lr0": LR0, "lrf": LRF}

    with open(best_yaml, "r", encoding="utf-8") as f:
        best_params = yaml.safe_load(f)

    dest = output_dir / "best_hyperparameters.yaml"
    if dest != best_yaml:
        shutil.copy2(str(best_yaml), str(dest))

    json_path = output_dir / "best_params.json"
    json_data = {
        "best_params": {k: round(float(v), 6) for k, v in best_params.items()},
        "timestamp": datetime.now().isoformat(),
        "iterations": iterations,
        "tune_epochs": tune_epochs,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  SEARCH COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Best hyperparameters:")
    for k, v in best_params.items():
        print(f"    {k}: {v}")
    print(f"  Saved to: {dest}")
    print(f"{'=' * 60}")

    return best_params


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
    from iou_callback import create_iou_callback

    lr0 = float(best_params.get("lr0", LR0))
    lrf = float(best_params.get("lrf", LRF))

    print(f"\n{'=' * 60}")
    print(f"  FINAL TRAINING with best parameters")
    print(f"{'=' * 60}")
    print(f"  lr0    : {lr0:.6f}")
    print(f"  lrf    : {lrf:.6f}")
    print(f"  epochs : {final_epochs}")
    print(f"  batch  : {batch}")
    print(f"  workers: {workers}")
    print(f"{'=' * 60}")

    model = YOLO(str(model_path))

    iou_cb = create_iou_callback()
    model.add_callback("on_fit_epoch_end", iou_cb.on_epoch_end)
    print("[Callback registered — Mean IoU will be logged each epoch")

    train_kwargs = dict(
        data=str(data_path),
        epochs=final_epochs,
        imgsz=IMGSZ,
        batch=batch,
        lr0=lr0,
        lrf=lrf,
        optimizer=OPTIMIZER,
        cos_lr=COS_LR,
        patience=PATIENCE,
        device=DEVICE,
        workers=workers,
        project=str(output_dir),
        name="final_best",
        resume=False,
    )

    extra_keys = ["momentum", "weight_decay", "warmup_epochs",
                  "warmup_momentum", "warmup_bias_lr", "box", "cls",
                  "dfl", "hsv_h", "hsv_s", "hsv_v", "degrees",
                  "translate", "scale", "shear", "perspective",
                  "flipud", "fliplr", "mosaic", "mixup", "copy_paste"]
    for key in extra_keys:
        if key in best_params:
            train_kwargs[key] = float(best_params[key])

    results = model.train(**train_kwargs)

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


def load_best_params(source: Path) -> dict | None:
    if source.is_dir():
        candidates = [
            source / "best_hyperparameters.yaml",
            source / "tune" / "best_hyperparameters.yaml",
            source / "best_params.json",
        ]
        for c in candidates:
            if c.exists():
                source = c
                break
        else:
            print(f"  No best params found in {source}")
            return None

    if source.suffix in (".yaml", ".yml"):
        with open(source, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
    elif source.suffix == ".json":
        with open(source, "r", encoding="utf-8") as f:
            data = json.load(f)
        params = data.get("best_params", data)
    else:
        print(f"  Unsupported file format: {source}")
        return None

    params = {k: float(v) for k, v in params.items()}
    print(f"  Loaded best params from {source}:")
    for k, v in params.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.6f}")
        else:
            print(f"    {k}: {v}")
    return params


def main(cli_args=None):
    parser = argparse.ArgumentParser(
        description="Hyperparameter fine-tuning using Ultralytics model.tune()",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/fine_tune.py                                           # default tune
  python scripts/fine_tune.py --iterations 30 --tune-epochs 15          # more trials/epochs
  python scripts/fine_tune.py --final-epochs 30                         # longer final training
  python scripts/fine_tune.py --no-final                                # tune only, skip final
  python scripts/fine_tune.py --resume-from results/runs/hyperparam_search  # skip tune, final only
        """,
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Path to base model (default: best.pt from training)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to data YAML (default: dataset/data.yaml)")
    parser.add_argument("--tune-epochs", type=int, default=DEFAULT_TUNE_EPOCHS,
                        help=f"Epochs per tuning trial (default: {DEFAULT_TUNE_EPOCHS})")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                        help=f"Number of tuning iterations (default: {DEFAULT_ITERATIONS})")
    parser.add_argument("--final-epochs", type=int, default=DEFAULT_FINAL_EPOCHS,
                        help=f"Epochs for final training (default: {DEFAULT_FINAL_EPOCHS})")
    parser.add_argument("--no-final", action="store_true",
                        help="Skip final training (only tune)")
    parser.add_argument("--no-archive", action="store_true",
                        help="Skip archiving previous model")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to best_hyperparameters.yaml or search dir — skips tune, runs final only")
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
        resume_source = Path(args.resume_from)
        print(f"\n  RESUME MODE: skipping tune, loading from {resume_source}")
        best_params = load_best_params(resume_source)
        if best_params is None:
            print("  Cannot resume — no best params found. Run full tune first.")
            sys.exit(1)
        output_dir = resume_source if resume_source.is_dir() else resume_source.parent
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

    best_params = run_tune(
        model_path=model_path,
        data_path=data_path,
        output_dir=output_dir,
        tune_epochs=args.tune_epochs,
        iterations=args.iterations,
        batch=args.batch,
        workers=args.workers,
    )

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
        if isinstance(v, float):
            print(f"    {k}: {v:.6f}")
        else:
            print(f"    {k}: {v}")
    print(f"  Results: {output_dir}")
    print(f"  Params : {output_dir / 'best_hyperparameters.yaml'}")
    if not args.no_final:
        print(f"  Model  : {MODELS_DIR / 'best.pt'}")
    print(f"\n  Next steps:")
    print(f"    python scripts/test_model.py    (evaluate model)")
    print(f"    python scripts/predict.py       (run predictions)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()