import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

IMAGE_DIR = BASE_DIR / "images"
CSV_PATH = BASE_DIR / "result.csv"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "outputs"

TASK_DEFINITIONS = [
    {
        "name": "tongue_color",
        "name_cn": "舌体颜色",
        "classes": ["淡红", "白", "红", "绛", "紫"],
        "num_classes": 5
    },
    {
        "name": "tongue_feature",
        "name_cn": "舌体特征",
        "classes": ["老", "嫩"],
        "num_classes": 2
    },
    {
        "name": "tongue_shape",
        "name_cn": "舌体形状",
        "classes": ["胖大", "瘦小", "正常"],
        "num_classes": 3
    },
    {
        "name": "teeth_marks",
        "name_cn": "齿痕",
        "classes": ["无齿痕", "有齿痕"],
        "num_classes": 2
    },
    {
        "name": "prickles",
        "name_cn": "点刺",
        "classes": ["无点刺", "有点刺"],
        "num_classes": 2
    },
    {
        "name": "cracks",
        "name_cn": "裂纹",
        "classes": ["无裂纹", "有裂纹"],
        "num_classes": 2
    },
    {
        "name": "coating_thickness",
        "name_cn": "舌苔特征",
        "classes": ["舌苔厚", "舌苔薄", "舌苔正常"],
        "num_classes": 3
    },
    {
        "name": "slippery_coating",
        "name_cn": "滑苔",
        "classes": ["无滑苔", "有滑苔"],
        "num_classes": 2
    },
    {
        "name": "rough_coating",
        "name_cn": "糙苔",
        "classes": ["无糙苔", "有糙苔"],
        "num_classes": 2
    },
    {
        "name": "peeled_coating",
        "name_cn": "剥落苔",
        "classes": ["无剥落苔", "有剥落苔"],
        "num_classes": 2
    },
    {
        "name": "curd_coating",
        "name_cn": "腐苔",
        "classes": ["无腐苔", "有腐苔"],
        "num_classes": 2
    },
    {
        "name": "greasy_coating",
        "name_cn": "腻苔",
        "classes": ["无腻苔", "有腻苔"],
        "num_classes": 2
    },
    {
        "name": "coating_color",
        "name_cn": "舌苔颜色",
        "classes": ["白", "黄", "灰", "黑"],
        "num_classes": 4
    },
    {
        "name": "red_tip",
        "name_cn": "舌尖红",
        "classes": ["无舌尖红", "有舌尖红"],
        "num_classes": 2
    },
    {
        "name": "tip_depression",
        "name_cn": "舌尖凹陷",
        "classes": ["无舌尖凹陷", "有舌尖凹陷"],
        "num_classes": 2
    }
]

NUM_TASKS = len(TASK_DEFINITIONS)
MISSING_LABEL = -1

TASK_NAMES = [task["name"] for task in TASK_DEFINITIONS]
TASK_NAMES_CN = [task["name_cn"] for task in TASK_DEFINITIONS]
NUM_CLASSES_PER_TASK = [task["num_classes"] for task in TASK_DEFINITIONS]

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
VAL_SPLIT = 0.2
SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

for d in [CHECKPOINT_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)
