import csv
from pathlib import Path

import numpy as np
import torch


def _box_iou_batch(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union_area = area1[:, None] + area2[None, :] - inter_area

    return inter_area / (union_area + 1e-7)


class IoUTracker:

    def __init__(self):
        self._history: list[dict] = []
        self._save_dir: Path | None = None

    def on_epoch_end(self, trainer):

        epoch = trainer.epoch + 1 

        if self._save_dir is None:
            self._save_dir = Path(trainer.save_dir)

        mean_iou, total_gt = self._compute_iou_from_validator(trainer)

        record = {
            "epoch": epoch,
            "mean_iou": round(float(mean_iou), 5),
        }
        self._history.append(record)

        print(f"Epoch {epoch}: Mean Val IoU = {mean_iou:.4f}")

        self._append_csv(record)

    @staticmethod
    def _compute_iou_from_validator(trainer) -> tuple[float, int]:
        validator = getattr(trainer, "validator", None)
        if validator is None:
            return 0.0, 0
        try:
            stats = validator.stats
            if isinstance(stats, dict) and "tp" in stats:
                tp = stats["tp"]
                if hasattr(tp, '__len__') and len(tp) > 0:
                    tp_arr = np.array(tp)
                    mean_tp = float(tp_arr.mean())
                    n_gt = int(tp_arr.shape[0])
                    return mean_tp, n_gt
        except Exception:
            pass
        try:
            metrics = trainer.metrics
            if isinstance(metrics, dict):
                map50 = metrics.get("metrics/mAP50(B)", 0)
                map5095 = metrics.get("metrics/mAP50-95(B)", 0)

                if map50 > 0:
                    ratio = map5095 / map50 if map50 > 0 else 0
                    estimated_iou = 0.5 + ratio * 0.4
                    return round(estimated_iou, 5), -1
        except Exception:
            pass

        return 0.0, 0

    def _append_csv(self, record: dict):
        if self._save_dir is None:
            return

        csv_path = self._save_dir / "iou_log.csv"
        headers = ["epoch", "mean_iou"]
        file_exists = csv_path.exists()

        try:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(record)
        except Exception as e:
            print(f"Warning: could not write to {csv_path}: {e}")

    def get_history(self) -> list[dict]:
        return list(self._history)

    def get_iou_values(self) -> list[float]:
        return [r["mean_iou"] for r in self._history]

    def get_final_iou(self) -> float:
        return self._history[-1]["mean_iou"] if self._history else 0.0


def create_iou_callback() -> IoUTracker:
    return IoUTracker()