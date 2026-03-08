import os
import sys
from pathlib import Path


def _is_colab() -> bool:
    try:
        import google.colab 
        return True
    except ImportError:
        return False


IS_COLAB = _is_colab()

if IS_COLAB:
    BASE = Path("/content/drive/MyDrive/Yolo_Autolabel")
    DEVICE = 0 
    WORKERS = 4
elif sys.platform == "darwin":
    BASE = Path(__file__).resolve().parent.parent
    WORKERS = 2

    try:
        import torch
        DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    except (ImportError, AttributeError):
        DEVICE = "cpu"
elif sys.platform == "linux":
    BASE = Path(__file__).resolve().parent.parent
    WORKERS = 2

    try:
        import torch
        DEVICE = 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        DEVICE = "cpu"
else:
    BASE = Path(__file__).resolve().parent.parent
    WORKERS = 0

    try:
        import torch
        DEVICE = 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        DEVICE = "cpu"


INPUT_DIR  = BASE / "input_files"
OUTPUT_DIR = BASE / "results"

MODEL_NAME       = os.environ.get("YOLO_MODEL", "yolo26n.pt")
TRAINING_MODELS  = INPUT_DIR / "training_models"
TRAINING_MODELS.mkdir(parents=True, exist_ok=True)
PRETRAINED_MODEL = TRAINING_MODELS / MODEL_NAME
if not PRETRAINED_MODEL.exists():
    PRETRAINED_MODEL = MODEL_NAME

import os
os.environ["YOLO_CONFIG_DIR"] = str(TRAINING_MODELS)
try:
    from ultralytics.utils import SETTINGS
    SETTINGS.update({'weights_dir': str(TRAINING_MODELS)})
except ImportError:
    pass

# Training
EPOCHS    = int(os.environ.get("YOLO_EPOCHS", "80"))
IMGSZ     = int(os.environ.get("YOLO_IMGSZ", "640"))
CONF      = float(os.environ.get("YOLO_CONF", "0.25"))
OPTIMIZER = os.environ.get("YOLO_OPTIMIZER", "AdamW")
LR0       = float(os.environ.get("YOLO_LR0", "0.008"))
LRF       = float(os.environ.get("YOLO_LRF", "0.01"))
COS_LR    = os.environ.get("YOLO_COS_LR", "True").lower() in ("true", "1", "yes")
PATIENCE  = int(os.environ.get("YOLO_PATIENCE", "25"))
RESUME    = os.environ.get("YOLO_RESUME", "False").lower() in ("true", "1", "yes")

# Fine-tune defaults
FT_EPOCHS = int(os.environ.get("YOLO_FT_EPOCHS", "20"))
FT_LR0    = float(os.environ.get("YOLO_FT_LR0", "0.001"))
FT_FREEZE = int(os.environ.get("YOLO_FT_FREEZE", "10"))

# Input paths
RAW_IMAGES     = INPUT_DIR / "raw_images"
EXPORT_DIR     = INPUT_DIR / "export"
EXPORT_LABELS  = EXPORT_DIR / "labels"
CLASSES_TXT    = EXPORT_DIR / "classes.txt"
INPUT_IMAGES   = INPUT_DIR / "input_images"
TEST_IMAGES    = INPUT_DIR / "test-dataset"

# Output paths
DATASET        = OUTPUT_DIR / "dataset"
TRAIN_IMG      = DATASET / "images" / "train"
VAL_IMG        = DATASET / "images" / "val"
TRAIN_LBL      = DATASET / "labels" / "train"
VAL_LBL        = DATASET / "labels" / "val"
DATA_YAML      = DATASET / "data.yaml"
MODELS_DIR     = OUTPUT_DIR / "models"
RUNS_DIR       = OUTPUT_DIR / "runs"

def get_latest_train_run(runs_dir: Path) -> Path:
    if not runs_dir.exists():
        return runs_dir / "train"
    
    train_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("train")]
    if not train_dirs:
        return runs_dir / "train"
    
    train_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return train_dirs[0]

TRAIN_RUN      = get_latest_train_run(RUNS_DIR)
BEST_MODEL     = TRAIN_RUN / "weights" / "best.pt"
BEST_MODEL_ALT = MODELS_DIR / "best.pt"
OUTPUT_RESULTS = OUTPUT_DIR / "output_results"
TEST_RESULTS   = OUTPUT_DIR / "test_results"
VERIFICATION   = OUTPUT_DIR / "verification"
PREANNOTATIONS_JSON = OUTPUT_DIR / "preannotations.json"
LS_URL   = os.environ.get("LS_URL", "")
TOKEN_FILE = INPUT_DIR / "token.txt"
LS_TOKEN = os.environ.get("LS_TOKEN", "")
if not LS_TOKEN and TOKEN_FILE.exists():
    LS_TOKEN = TOKEN_FILE.read_text("utf-8").strip()
if not LS_TOKEN:
    LS_TOKEN = ""

def get_best_model() -> Path:
    if BEST_MODEL.exists():
        return BEST_MODEL
    if BEST_MODEL_ALT.exists():
        return BEST_MODEL_ALT
    tm = TRAINING_MODELS / "best.pt"
    if tm.exists():
        return tm
    return BEST_MODEL


def ensure_drive_mounted():
    if IS_COLAB and not Path("/content/drive/MyDrive").exists():
        from google.colab import drive
        drive.mount("/content/drive")


def print_config():
    if IS_COLAB:
        env = "Google Colab"
    elif sys.platform == "darwin":
        env = "Local (macOS)"
    elif sys.platform == "linux":
        env = "Local (Linux/WSL)"
    else:
        env = "Local (Windows)"
    print("=" * 50)
    print(f"Environment  : {env}")
    print(f"BASE         : {BASE}")
    print(f"INPUT_DIR    : {INPUT_DIR}")
    print(f"OUTPUT_DIR   : {OUTPUT_DIR}")
    print(f"DEVICE       : {DEVICE}")
    print(f"WORKERS      : {WORKERS}")
    print(f"MODEL        : {MODEL_NAME}")
    print(f"EPOCHS       : {EPOCHS}  |  IMGSZ: {IMGSZ}  |  CONF: {CONF}")
    print(f"OPTIMIZER    : {OPTIMIZER}  |  LR0: {LR0}  |  LRF: {LRF}  |  COS_LR: {COS_LR}")
    print(f"PATIENCE     : {PATIENCE}  |  RESUME: {RESUME}")
    print(f"FT_EPOCHS    : {FT_EPOCHS}  |  FT_LR0: {FT_LR0}  |  FT_FREEZE: {FT_FREEZE}")
    print(f"RAW_IMAGES   : {RAW_IMAGES}  | exists: {RAW_IMAGES.exists()}")
    print(f"EXPORT_LABELS: {EXPORT_LABELS} | exists: {EXPORT_LABELS.exists()}")
    print(f"CLASSES_TXT  : {CLASSES_TXT} | exists: {CLASSES_TXT.exists()}")
    print(f"BEST_MODEL   : {get_best_model()} | exists: {get_best_model().exists()}")
    print(f"LS_URL       : {LS_URL}")
    print(f"LS_TOKEN     : {'*' * 8 + '...' + LS_TOKEN[-4:] if LS_TOKEN else '(not set)'}")
    print("=" * 50)