import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    BASE, DEVICE, WORKERS, IS_COLAB,
    RAW_IMAGES, EXPORT_LABELS, CLASSES_TXT,
    DATASET, TRAIN_IMG, VAL_IMG, TRAIN_LBL, VAL_LBL,
    RUNS_DIR, TRAIN_RUN,
    ensure_drive_mounted, print_config,
    MODEL_NAME, PRETRAINED_MODEL, EPOCHS, IMGSZ,
    OPTIMIZER, LR0, LRF, COS_LR, PATIENCE, RESUME,
)


FL_GAMMA_DEFAULT = 1.5
FL_ALPHA_DEFAULT = 0.25

def check_gpu():
    import torch
    print("=" * 50)
    print("1. CHECK GPU AVAILABILITY")
    print("=" * 50)
    print("CUDA (GPU) Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device Name:", torch.cuda.get_device_name(0))
    else:
        print("Device: CPU (training will be slower)")
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO model (with Focal Loss)")
    parser.add_argument("--model", type=str, default=str(PRETRAINED_MODEL),
                        help=f"Model name or path (default: {MODEL_NAME})")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Number of epochs (default: {EPOCHS})")
    parser.add_argument("--imgsz", type=int, default=IMGSZ,
                        help=f"Image size (default: {IMGSZ})")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit total number of images (absolute count)")
    parser.add_argument("--fraction", type=float, default=0.0,
                        help="Use a fraction of available data, e.g. 0.2 = 20%% (overrides --limit)")
    parser.add_argument("--batch", type=int, default=-1,
                        help="Batch size (-1 = auto, default: -1)")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Custom image folder (default: input_files/raw_images)")
    parser.add_argument("--split-dir", type=str, default=None,
                        help="Path to a pre-split dataset directory. Must contain "
                             "images/train, images/val, labels/train, labels/val. "
                             "When provided, the script skips the split step.")
    parser.add_argument("--fl-gamma", type=float, default=FL_GAMMA_DEFAULT,
                        help=f"Focal Loss gamma – controls how much easy examples "
                             f"are down-weighted (default: {FL_GAMMA_DEFAULT}). "
                             f"Set to 0.0 to disable Focal Loss (standard BCE).")
    parser.add_argument("--fl-alpha", type=float, default=FL_ALPHA_DEFAULT,
                        help=f"Focal Loss alpha – class-balance factor "
                             f"(default: {FL_ALPHA_DEFAULT}).")

    return parser.parse_args()


def reset_dataset():
    print("=" * 50)
    print("3. RESET DATASET FOLDER")
    print("=" * 50)

    if DATASET.exists():
        shutil.rmtree(DATASET)

    for p in [TRAIN_IMG, VAL_IMG, TRAIN_LBL, VAL_LBL]:
        p.mkdir(parents=True, exist_ok=True)

    print("Dataset folders created:", DATASET)
    print()


def use_presplit_dataset(split_dir):

    print("=" * 50)
    print("3-4. USING PRE-SPLIT DATASET (skip split)")
    print("=" * 50)

    src = Path(split_dir)
    required = [
        src / "images" / "train",
        src / "images" / "val",
        src / "labels" / "train",
        src / "labels" / "val",
    ]
    for p in required:
        if not p.exists():
            print(f"ERROR: Required directory not found: {p}")
            sys.exit(1)

    if DATASET.exists():
        shutil.rmtree(DATASET)
    for p in [TRAIN_IMG, VAL_IMG, TRAIN_LBL, VAL_LBL]:
        p.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        img_src = src / "images" / split
        lbl_src = src / "labels" / split
        img_dst = DATASET / "images" / split
        lbl_dst = DATASET / "labels" / split

        for f in img_src.iterdir():
            if f.is_file():
                shutil.copy(f, img_dst / f.name)
        for f in lbl_src.iterdir():
            if f.is_file():
                shutil.copy(f, lbl_dst / f.name)

    num_train = len(list(TRAIN_IMG.glob("*.*")))
    num_val = len(list(VAL_IMG.glob("*.*")))

    print(f"Source       : {src}")
    print(f"Train Images : {num_train}")
    print(f"Val Images   : {num_val}")
    print()

    return num_train


def prepare_dataset(limit=0, images_dir=None):

    print("=" * 50)
    print("4. PREPARE & SPLIT DATASET")
    print("=" * 50)

    src_images = Path(images_dir) if images_dir else RAW_IMAGES
    print(f"Images dir : {src_images}")
    labels = list(EXPORT_LABELS.glob("*.txt"))
    images = list(src_images.glob("*.*"))

    img_map = {p.stem: p for p in images}
    img_map_lower = {p.stem.lower(): p for p in images}

    pairs = []
    missing_images = []

    print(f"Processing {len(labels)} labels against {len(images)} images...")

    for lbl in labels:
        if lbl.stat().st_size == 0:
            continue

        stem = lbl.stem
        clean_stem = stem.split("__", 1)[1] if "__" in stem else stem

        found_img = img_map.get(clean_stem) or img_map_lower.get(clean_stem.lower())

        if found_img:
            pairs.append((found_img, lbl))
        else:
            missing_images.append(lbl.name)

    print("-" * 30)
    print(f"Total Labels: {len(labels)}")
    print(f"Pairs Found : {len(pairs)}")

    if limit > 0 and len(pairs) > limit:
        print("-" * 30)
        print(f"  LIMITING DATASET: {len(pairs)} -> {limit}")
        random.seed(42)
        random.shuffle(pairs)
        pairs = pairs[:limit]

    if missing_images:
        print(f"Missing Images: {len(missing_images)}")

    if not pairs:
        print("No pairs found. Please check your image filenames.")
        sys.exit(1)

    random.seed(42)
    random.shuffle(pairs)
    cut = int(0.8 * len(pairs))
    train_pairs = pairs[:cut]
    val_pairs = pairs[cut:]

    if not val_pairs and len(pairs) > 1:
        val_pairs = [train_pairs.pop()]

    def copy_pairs(pairs_list, split):
        for img, lbl in pairs_list:
            shutil.copy(img, DATASET / f"images/{split}/{img.name}")
            shutil.copy(lbl, DATASET / f"labels/{split}/{img.stem}.txt")

    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")

    num_train = len(list(TRAIN_IMG.glob("*.*")))
    num_val = len(list(VAL_IMG.glob("*.*")))

    print("-" * 30)
    print("Final Dataset Counts:")
    print("Train Images:", num_train)
    print("Val Images:  ", num_val)
    print()

    return num_train


def create_data_yaml():

    print("=" * 50)
    print("5. CREATE DATA.YAML")
    print("=" * 50)

    classes = [c.strip() for c in CLASSES_TXT.read_text().splitlines() if c.strip()]
    names_block = "\n".join([f"  {i}: {name}" for i, name in enumerate(classes)])
    dataset_path = str(DATASET).replace("\\", "/")

    data_yaml = f"""path: {dataset_path}
train: images/train
val: images/val
names:
{names_block}
"""

    (DATASET / "data.yaml").write_text(data_yaml)
    print("Saved:", DATASET / "data.yaml")
    print()

    return classes


def _make_focal_callback(fl_gamma, fl_alpha):

    from ultralytics.utils.loss import FocalLoss

    def _inject_focal_loss(trainer):
        if hasattr(trainer, "criterion") and hasattr(trainer.criterion, "bce"):
            trainer.criterion.bce = FocalLoss(gamma=fl_gamma, alpha=fl_alpha)
            print(f"Replaced BCE → FocalLoss(gamma={fl_gamma}, alpha={fl_alpha})")
        else:
            print("Could not locate trainer.criterion.bce — Focal Loss was NOT applied.")

    return _inject_focal_loss


def run_training(model_path, epochs, imgsz, batch=-1,
                 fl_gamma=FL_GAMMA_DEFAULT, fl_alpha=FL_ALPHA_DEFAULT):
    print("=" * 50)
    print("6. START TRAINING (FOCAL LOSS)")
    print("=" * 50)

    from ultralytics import YOLO
    from iou_callback import create_iou_callback

    print(f"Model    : {model_path}")
    print(f"FL gamma : {fl_gamma}")
    print(f"FL alpha : {fl_alpha}")
    if fl_gamma == 0.0:
        print("  → Focal Loss DISABLED (standard BCE loss)")
    else:
        print(f"  → Focal Loss ENABLED (gamma={fl_gamma}, alpha={fl_alpha})")

    model = YOLO(model_path)

    iou_cb = create_iou_callback()
    model.add_callback("on_fit_epoch_end", iou_cb.on_epoch_end)
    print("[IoU] Callback registered — Mean IoU will be logged each epoch")

    if fl_gamma > 0.0:
        focal_cb = _make_focal_callback(fl_gamma, fl_alpha)
        model.add_callback("on_pretrain_routine_end", focal_cb)

    train_kwargs = dict(
        data=str(DATASET / "data.yaml"),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch if batch != -1 else 16,
        device=DEVICE,
        workers=WORKERS,
        project=str(RUNS_DIR),
        name="train",

        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        cos_lr=COS_LR,
        patience=PATIENCE,
        resume=RESUME,
        augment=False,
        warmup_epochs=3,
        close_mosaic=15,
    )

    results = model.train(**train_kwargs)

    save_dir = Path(results.save_dir) if hasattr(results, 'save_dir') else TRAIN_RUN

    iou_history = iou_cb.get_iou_values()
    if iou_history:
        print()
        print(f"[IoU] Final Mean IoU : {iou_history[-1]:.4f}")
        print(f"[IoU] Best Mean IoU  : {max(iou_history):.4f}")
        print(f"[IoU] Log saved to   : {save_dir / 'iou_log.csv'}")

    print()
    print("=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)
    print(f"Results saved to: {save_dir}")
    print(f"Best model: {save_dir / 'weights' / 'best.pt'}")

    return results, save_dir, iou_cb


def post_training_analysis(classes, num_train, model_name, results, save_dir, iou_cb=None):
    print()
    print("=" * 50)
    print("7. POST-TRAINING ANALYSIS")
    print("=" * 50)

    from analyze import run_analysis

    run_analysis(
        results_dir=str(save_dir),
        num_classes=len(classes),
    )

    try:
        from analyze import log_training_history, load_results

        if hasattr(results, 'results_dict') and results.results_dict:
            metrics_to_log = results.results_dict
        else:
            d = load_results(save_dir)
            metrics_to_log = {}
            if d:
                for k, v in d.items():
                    if v:
                        metrics_to_log[k] = v[-1]

        if iou_cb is not None:
            final_iou = iou_cb.get_final_iou()
            if final_iou > 0:
                metrics_to_log["mean_iou"] = final_iou

        log_training_history(
            model_name=model_name,
            num_train=num_train,
            epochs_trained=len(classes),
            metrics=metrics_to_log,
        )
    except Exception as e:
        print(f"Warning: Failed to log training history: {e}")

    print()
    print("=" * 50)
    print("ALL DONE!")
    print("=" * 50)
    print(f"  Best model : {save_dir / 'weights' / 'best.pt'}")
    print(f"  Dashboard  : {save_dir / 'training_analysis.png'}")
    print(f"  IoU log    : {save_dir / 'iou_log.csv'}")
    print(f"  Open the PNG file above to view the full visualization.")


def main():
    check_gpu()

    args = parse_args()

    print("=" * 50)
    print("2. ENVIRONMENT, PATHS & ARGS")
    print("=" * 50)
    print(f"Model    : {args.model}")
    print(f"Epochs   : {args.epochs}")
    print(f"FL gamma : {args.fl_gamma}")
    print(f"FL alpha : {args.fl_alpha}")
    if args.fraction > 0.0:
        print(f"Subset: {args.fraction:.0%} of available data")
    elif args.limit > 0:
        print(f"Limit : {args.limit} images")
    else:
        print(f"Limit : No Limit (full dataset)")

    ensure_drive_mounted()
    print_config()
    print()

    if args.split_dir:
        num_train = use_presplit_dataset(args.split_dir)
    else:
        reset_dataset()

        effective_limit = args.limit
        if args.fraction > 0.0:
            total_pairs = len(list(EXPORT_LABELS.glob("*.txt")))
            effective_limit = max(1, int(total_pairs * args.fraction))
            print(f"Fraction : {args.fraction:.0%} of ~{total_pairs} -> {effective_limit} images")

        images_dir = args.images_dir if args.images_dir else None
        num_train = prepare_dataset(limit=effective_limit, images_dir=images_dir)

    classes = create_data_yaml()

    results, save_dir, iou_cb = run_training(
        args.model, args.epochs, args.imgsz, args.batch,
        fl_gamma=args.fl_gamma,
        fl_alpha=args.fl_alpha,
    )

    print("=" * 50)
    print("7. POST-TRAINING ANALYSIS")
    print("=" * 50)
    post_training_analysis(classes, num_train, args.model, results, save_dir, iou_cb=iou_cb)


if __name__ == "__main__":
    main()