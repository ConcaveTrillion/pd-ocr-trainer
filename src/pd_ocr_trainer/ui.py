"""NiceGUI training interface for OCR configuration and dataset management."""

import gc
import json
import os
import platform
import shutil
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

import torch
from doctr.datasets import VOCABS
from nicegui import ui

from pd_ocr_trainer.train_detect import detect_from_config
from pd_ocr_trainer.train_recog import train_from_config

# Get the project root (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent.parent
ML_TRAINING_DIR = Path(os.getenv("PD_OCR_TRAINER_ML_TRAINING_DIR", PROJECT_ROOT / "ml-training"))
ML_VALIDATION_DIR = Path(os.getenv("PD_OCR_TRAINER_ML_VALIDATION_DIR", PROJECT_ROOT / "ml-validation"))
APP_NAME = "pd-ocr-labeler"
MODEL_STORE_DIRNAME = "pd-ml-models"
MODEL_NAME_PREFIX = "pd"
BASE_OCR_PROFILE = "all"
LEGACY_BASE_OCR_PROFILE = "base-ocr"
DATASET_TASKS = ("detection", "recognition")


def _available_cpu_count() -> int:
    """Return CPU count visible to this process/container."""
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def _default_worker_count() -> int:
    """Choose a conservative default that scales with available CPUs."""
    cpu_count = _available_cpu_count()
    if cpu_count <= 2:
        return 1
    return min(8, max(2, cpu_count // 2))


def _release_cuda_memory() -> None:
    """Best-effort CUDA cleanup after training ends or fails."""
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()


def get_os_data_parent() -> Path:
    """Return OS-aware parent directory for application data roots."""
    system_name = platform.system()

    if system_name == "Linux":
        data_home = os.getenv("XDG_DATA_HOME")
        base_dir = Path(data_home).expanduser() if data_home else Path.home() / ".local" / "share"
    elif system_name == "Darwin":
        base_dir = Path.home() / "Library" / "Application Support"
    elif system_name == "Windows":
        appdata = os.getenv("APPDATA")
        base_dir = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    else:
        base_dir = Path.home() / ".local" / "share"

    return base_dir


APP_DATA_ROOT = Path(os.getenv("PD_OCR_TRAINER_APP_DATA_ROOT", get_os_data_parent() / APP_NAME))
SHARED_MODELS_DIR = Path(os.getenv("PD_OCR_TRAINER_SHARED_MODELS_DIR", get_os_data_parent() / MODEL_STORE_DIRNAME))
TRAINER_SETTINGS_PATH = APP_DATA_ROOT / "trainer_settings.json"

DEFAULT_VOCAB_LIBRARY = ["multilingual", "currency"]
DEFAULT_CUSTOM_CHARACTERS = "⸺¡¿—‘’“”′″"

DETECTION_ARCH_OPTIONS = [
    "db_resnet34",
    "db_resnet50",
    "db_mobilenet_v3_large",
    "linknet_resnet18",
    "linknet_resnet34",
    "linknet_resnet50",
    "fast_tiny",
    "fast_small",
    "fast_base",
]

RECOGNITION_ARCH_OPTIONS = [
    "crnn_vgg16_bn",
    "crnn_mobilenet_v3_small",
    "crnn_mobilenet_v3_large",
    "sar_resnet31",
    "master",
    "vitstr_small",
    "vitstr_base",
    "parseq",
    "viptr_tiny",
]

DETECTION_ARCH_HELP = {
    "db_resnet34": "DBNet with ResNet-34 backbone: balanced speed/accuracy with lower compute than ResNet-50.",
    "db_resnet50": "DBNet with ResNet-50 backbone: strong baseline for robust text detection quality.",
    "db_mobilenet_v3_large": "DBNet with MobileNetV3-Large: lightweight and faster, usually best on smaller GPUs.",
    "linknet_resnet18": "LinkNet with ResNet-18: efficient segmentation-style detector with lower parameter count.",
    "linknet_resnet34": "LinkNet with ResNet-34: moderate compute/accuracy option in the LinkNet family.",
    "linknet_resnet50": "LinkNet with ResNet-50: heavier LinkNet variant for stronger feature capacity.",
    "fast_tiny": "FAST detector with tiny TextNet backbone: fastest FAST variant, lowest compute.",
    "fast_small": "FAST detector with small TextNet backbone: speed/quality middle ground.",
    "fast_base": "FAST detector with base TextNet backbone: strongest FAST variant, heavier than tiny/small.",
}

RECOGNITION_ARCH_HELP = {
    "crnn_vgg16_bn": "CRNN with VGG16 backbone: classic strong baseline for text recognition.",
    "crnn_mobilenet_v3_small": "CRNN with MobileNetV3-Small: very fast and lightweight recognizer.",
    "crnn_mobilenet_v3_large": "CRNN with MobileNetV3-Large: still efficient with somewhat higher capacity.",
    "sar_resnet31": "SAR with ResNet-31 encoder: attention-based recognizer for irregular text.",
    "master": "MASTER recognizer: high-capacity transformer-style model, usually slower and heavier.",
    "vitstr_small": "ViTSTR-Small: transformer recognizer with moderate compute.",
    "vitstr_base": "ViTSTR-Base: larger ViTSTR variant with higher memory and compute needs.",
    "parseq": "PARSeq recognizer: permuted autoregressive sequence model with strong accuracy-speed tradeoff.",
    "viptr_tiny": "VIPTR-Tiny: compact modern recognizer focused on efficient inference.",
}

# Ensure directories exist
ML_TRAINING_DIR.mkdir(exist_ok=True)
ML_VALIDATION_DIR.mkdir(exist_ok=True)
(ML_TRAINING_DIR / BASE_OCR_PROFILE).mkdir(parents=True, exist_ok=True)
(ML_VALIDATION_DIR / BASE_OCR_PROFILE).mkdir(parents=True, exist_ok=True)
SHARED_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_profile_name(name: str) -> str:
    value = (name or "").strip().lower().replace(" ", "-").replace("_", "-")
    if value == LEGACY_BASE_OCR_PROFILE:
        return BASE_OCR_PROFILE
    return value or BASE_OCR_PROFILE


def _merge_profile_tree(src_root: Path, dest_root: Path) -> None:
    """Merge one profile tree into another and remove the source when done."""
    if not src_root.exists() or src_root == dest_root:
        return

    dest_root.mkdir(parents=True, exist_ok=True)

    for child in src_root.iterdir():
        target = dest_root / child.name
        if child.is_dir():
            if target.exists():
                _merge_profile_tree(child, target)
            else:
                shutil.move(str(child), str(target))
        else:
            if target.exists():
                target.unlink()
            shutil.move(str(child), str(target))

    shutil.rmtree(src_root, ignore_errors=True)


def _profile_model_root(profile: str) -> Path:
    return SHARED_MODELS_DIR / _normalize_profile_name(profile)


def _model_output_dir(profile: str, model_type: str) -> Path:
    return _profile_model_root(profile) / model_type


def _split_profile_root(split: str, profile: str = BASE_OCR_PROFILE) -> Path:
    split_map = {"train": ML_TRAINING_DIR, "val": ML_VALIDATION_DIR}
    root = split_map.get(split)
    if root is None:
        raise ValueError(f"Unknown split '{split}'")
    return root / _normalize_profile_name(profile)


def _migrate_legacy_dataset_layout() -> None:
    """Move legacy split/task datasets into the all-profile layout.

    Legacy layout:
      ml-training/detection, ml-training/recognition, ...

    New layout:
      ml-training/<profile>/detection, ml-training/<profile>/recognition, ...
    """
    base_profile = _normalize_profile_name(BASE_OCR_PROFILE)
    for split_root in (ML_TRAINING_DIR, ML_VALIDATION_DIR):
        has_legacy = any((split_root / task).exists() for task in DATASET_TASKS)
        if not has_legacy:
            continue

        profile_root = split_root / base_profile
        profile_root.mkdir(parents=True, exist_ok=True)

        for task in DATASET_TASKS:
            legacy_task_root = split_root / task
            if not legacy_task_root.exists():
                continue
            target_task_root = profile_root / task

            if not target_task_root.exists():
                shutil.move(str(legacy_task_root), str(target_task_root))
                continue

            target_images = target_task_root / "images"
            target_images.mkdir(parents=True, exist_ok=True)
            legacy_images = legacy_task_root / "images"
            if legacy_images.exists():
                for src_img in legacy_images.iterdir():
                    if src_img.is_file():
                        shutil.move(str(src_img), str(target_images / src_img.name))

            legacy_labels = ExportManager._load_json_map(legacy_task_root / "labels.json")
            target_labels_path = target_task_root / "labels.json"
            target_labels = ExportManager._load_json_map(target_labels_path)
            target_labels.update(legacy_labels)
            ExportManager._write_json_map(target_labels_path, target_labels)

            shutil.rmtree(legacy_task_root, ignore_errors=True)

    legacy_profile = LEGACY_BASE_OCR_PROFILE
    target_profile = _normalize_profile_name(BASE_OCR_PROFILE)
    if _normalize_profile_name(legacy_profile) == target_profile:
        for root in (ML_TRAINING_DIR, ML_VALIDATION_DIR, SHARED_MODELS_DIR):
            legacy_root = root / legacy_profile
            target_root = root / target_profile
            if legacy_root.exists() and legacy_root != target_root:
                _merge_profile_tree(legacy_root, target_root)


def _iter_export_profile_dirs(export_root: Path):
    """Yield export subdirectories that directly contain DocTR task folders."""
    if not export_root.exists():
        return

    for project_dir in sorted(export_root.iterdir()):
        if not project_dir.is_dir():
            continue
        for subdir in sorted(project_dir.rglob("*")):
            if not subdir.is_dir():
                continue
            if any((subdir / task / "labels.json").exists() for task in DATASET_TASKS):
                yield project_dir, subdir


def get_available_model_profiles() -> list[str]:
    """List trainable model profiles derived from export subfolders plus all."""
    profiles = {BASE_OCR_PROFILE}
    for split_root in (ML_TRAINING_DIR, ML_VALIDATION_DIR):
        if not split_root.exists():
            continue
        for profile_dir in split_root.iterdir():
            if not profile_dir.is_dir():
                continue
            if any((profile_dir / task).exists() for task in DATASET_TASKS):
                profiles.add(_normalize_profile_name(profile_dir.name))
    if SHARED_MODELS_DIR.exists():
        for profile_dir in SHARED_MODELS_DIR.iterdir():
            if profile_dir.is_dir():
                profiles.add(_normalize_profile_name(profile_dir.name))
    export_root = ExportManager.get_export_root()
    for _project_dir, subfolder in _iter_export_profile_dirs(export_root):
        profiles.add(_normalize_profile_name(subfolder.name))
    return sorted(profiles)


def _unique_chars_in_order(chars: str) -> str:
    """Keep first occurrence order while removing duplicate characters."""
    return "".join(dict.fromkeys(chars))


def build_custom_vocab_arg(vocab_names: list[str], custom_chars: str) -> str:
    """Build CUSTOM vocab argument from selected library vocab names and custom chars."""
    library_chars = "".join(VOCABS[name] for name in vocab_names if name in VOCABS)
    combined_chars = _unique_chars_in_order(library_chars + (custom_chars or ""))
    if not combined_chars:
        raise ValueError("Vocabulary cannot be empty. Select at least one library vocab or custom character.")
    return f"CUSTOM:{combined_chars}"


def _prefixed_model_name(model_type: str, base_name: str, profile: str = BASE_OCR_PROFILE) -> str:
    """Return a normalized model name with enforced prefix, profile, and type."""
    normalized = (base_name or "").strip().replace(" ", "-")
    normalized_profile = _normalize_profile_name(profile)
    if normalized.startswith(f"{MODEL_NAME_PREFIX}-"):
        normalized = normalized.removeprefix(f"{MODEL_NAME_PREFIX}-")
    if normalized.startswith(f"{normalized_profile}-"):
        normalized = normalized.removeprefix(f"{normalized_profile}-")
    if normalized.startswith(f"{model_type}-"):
        normalized = normalized.removeprefix(f"{model_type}-")
    if not normalized:
        normalized = "finetuned"
    return f"{MODEL_NAME_PREFIX}-{normalized_profile}-{model_type}-{normalized}"


def _default_model_timestamp() -> str:
    """Return yyyymmddhh24 timestamp for default model names."""
    return datetime.now().strftime("%Y%m%d%H")


def _project_from_stem(stem: str) -> str:
    """Strip trailing digit-only segments from an image stem to recover the project ID."""
    parts = stem.split("_")
    end = len(parts)
    while end > 1 and parts[end - 1].isdigit():
        end -= 1
    return "_".join(parts[:end])


def _detection_page_from_recognition_name(img_name: str) -> str:
    """Best-effort page filename from a recognition crop filename.

    Recognition crops are expected to be <page_stem>_<x1>_<x2>_<y1>_<y2>.<ext>.
    If the suffix shape does not match, return the original name.
    """
    path = Path(img_name)
    parts = path.stem.split("_")
    if len(parts) > 4 and all(p.isdigit() for p in parts[-4:]):
        return f"{'_'.join(parts[:-4])}{path.suffix}"
    return img_name


def _group_existing_by_project(split_root: Path) -> dict[str, list[str]]:
    """Return {project_id: [page_img_name, ...]} for an existing split.

    Prefer detection labels (one entry per page). If absent, derive page names from
    recognition crops so projects still appear in the on-disk UI.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    detection_labels = ExportManager._load_json_map(split_root / "detection" / "labels.json")
    if detection_labels:
        for img_name in detection_labels:
            groups[_project_from_stem(Path(img_name).stem)].append(img_name)
        return {k: sorted(v) for k, v in sorted(groups.items())}

    recognition_labels = ExportManager._load_json_map(split_root / "recognition" / "labels.json")
    if recognition_labels:
        by_project_pages: dict[str, set[str]] = defaultdict(set)
        for img_name in recognition_labels:
            page_name = _detection_page_from_recognition_name(img_name)
            by_project_pages[_project_from_stem(Path(page_name).stem)].add(page_name)
        return {k: sorted(v) for k, v in sorted(by_project_pages.items())}

    return {}


class ExportManager:
    """Manages pd-ocr-labeler DocTR export assignments for training."""

    def __init__(self) -> None:
        self.active_profile = _normalize_profile_name(BASE_OCR_PROFILE)
        self.assignments: dict[str, str | None] = {}
        self.page_assignments: dict[tuple[str, str], str] = {}
        self.changed_keys: set[str] = set()
        self.scan()

    def set_profile(self, profile: str) -> None:
        self.active_profile = _normalize_profile_name(profile)

    def split_root(self, split: str) -> Path:
        root = _split_profile_root(split, self.active_profile)
        root.mkdir(parents=True, exist_ok=True)
        return root

    @staticmethod
    def get_export_root() -> Path:
        """OS-aware path to the pd-ocr-labeler DocTR export root."""
        return APP_DATA_ROOT / "doctr-export"

    def scan(self) -> None:
        """Scan the export root and rebuild available exports, preserving existing assignments."""
        export_root = self.get_export_root()

        # Index existing images in both split directories for change detection.
        existing_image_names: set[str] = set()
        for split_root in (self.split_root("train"), self.split_root("val")):
            for task in DATASET_TASKS:
                images_dir = split_root / task / "images"
                if images_dir.exists():
                    for img in images_dir.iterdir():
                        existing_image_names.add(img.name)

        new_assignments: dict[str, str | None] = {}
        new_changed: set[str] = set()

        for _project_dir, subfolder in _iter_export_profile_dirs(export_root):
            if _normalize_profile_name(subfolder.name) != self.active_profile:
                continue
            key = subfolder.relative_to(export_root).as_posix()

            # If every source page/crop from this export is already present in a dataset split,
            # keep it out of the pending export list and represent it only via the on-disk view.
            export_pages = self.get_export_pages(key)
            if export_pages and all(page_name in existing_image_names for page_name in export_pages):
                continue

            # Preserve existing assignment.
            new_assignments[key] = self.assignments.get(key)
            # Mark as "changed" if any source image already exists in a split dir.
            for task in DATASET_TASKS:
                src_images = subfolder / task / "images"
                if src_images.exists() and any(img.name in existing_image_names for img in src_images.iterdir()):
                    new_changed.add(key)
                    break

        self.assignments = new_assignments
        self.page_assignments = {
            (key, page): split
            for (key, page), split in self.page_assignments.items()
            if key in self.assignments and page in set(self.get_export_pages(key)) and split in {"train", "val"}
        }
        self.changed_keys = new_changed

    def get_by_split(self) -> dict[str, dict[str, list[str]]]:
        """Return exports grouped by split then project: {split: {project: [keys]}}."""
        result: dict[str, dict[str, list[str]]] = {
            "unassigned": defaultdict(list),
            "train": defaultdict(list),
            "val": defaultdict(list),
        }
        for key, split in self.assignments.items():
            col = split if split in {"train", "val"} else "unassigned"
            project = key.split("/")[0]
            result[col][project].append(key)
        return {k: dict(v) for k, v in result.items()}

    def get_export_pages_by_split(self) -> dict[str, dict[str, dict[str, list[str]]]]:
        """Return pending export pages grouped as {split: {project: {key: [page_names]}}}."""
        result: dict[str, dict[str, dict[str, list[str]]]] = {
            "unassigned": defaultdict(lambda: defaultdict(list)),
            "train": defaultdict(lambda: defaultdict(list)),
            "val": defaultdict(lambda: defaultdict(list)),
        }

        for key in self.assignments:
            project = key.split("/")[0]
            pages = self.get_export_pages(key)
            split = self.assignments.get(key)
            if split in {"train", "val"}:
                result[split][project][key].extend(pages)
                continue

            for page_name in pages:
                page_split = self.page_assignments.get((key, page_name))
                col = page_split if page_split in {"train", "val"} else "unassigned"
                result[col][project][key].append(page_name)

        return {
            split: {
                project: {key: sorted(pages) for key, pages in sorted(keys.items())}
                for project, keys in sorted(projects.items())
            }
            for split, projects in result.items()
        }

    def assign(self, key: str, target: str | None) -> None:
        if key in self.assignments:
            self.assignments[key] = target if target in {"train", "val"} else None
            self.page_assignments = {(k, page): split for (k, page), split in self.page_assignments.items() if k != key}

    def assign_page(self, key: str, page_name: str, target: str | None) -> None:
        if key not in self.assignments:
            return

        pages = self.get_export_pages(key)
        full_split = self.assignments.get(key)
        if full_split in {"train", "val"}:
            for page in pages:
                self.page_assignments[(key, page)] = full_split
            self.assignments[key] = None

        if target in {"train", "val"}:
            self.page_assignments[(key, page_name)] = target
        else:
            self.page_assignments.pop((key, page_name), None)

    def assign_project(self, project_id: str, target: str | None) -> None:
        for key in self.assignments:
            if key.startswith(f"{project_id}/"):
                self.assignments[key] = target if target in {"train", "val"} else None
                self.page_assignments = {
                    (k, page): split for (k, page), split in self.page_assignments.items() if k != key
                }

    def assign_pages(self, pages: list[tuple[str, str]], target: str | None) -> None:
        for key, page_name in pages:
            self.assign_page(key, page_name, target)

    def clear_split(self, split: str) -> None:
        if split not in {"train", "val"}:
            return
        for key in list(self.assignments):
            if self.assignments.get(key) == split:
                self.assignments[key] = None
        self.page_assignments = {(k, page): s for (k, page), s in self.page_assignments.items() if s != split}

    def is_changed(self, key: str) -> bool:
        return key in self.changed_keys

    def export_path(self, key: str) -> Path:
        """Return the filesystem path for an export key."""
        parts = key.split("/", 1)
        return self.get_export_root() / parts[0] / parts[1]

    def get_export_pages(self, key: str) -> list[str]:
        """Return sorted page image names for an export key."""
        export_root = self.export_path(key)
        for task in ("detection", "recognition"):
            labels = self._load_json_map(export_root / task / "labels.json")
            if labels:
                return sorted(labels)
        return []

    @staticmethod
    def _load_json_map(path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            with open(path) as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _write_json_map(path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def get_existing_projects(split_root: Path) -> dict[str, int]:
        """Return {project_id: page_count} already present in a split directory."""
        return {k: len(v) for k, v in _group_existing_by_project(split_root).items()}

    @staticmethod
    def get_existing_pages(split_root: Path, project_id: str) -> list[str]:
        """Return sorted image names for project_id already present in split_root."""
        return _group_existing_by_project(split_root).get(project_id, [])

    def move_existing_project(self, project_id: str, from_split: str, to_split: str) -> int:
        """Physically move an on-disk project between ml-training and ml-validation."""
        src_root = self.split_root(from_split) if from_split in {"train", "val"} else None
        dest_root = self.split_root(to_split) if to_split in {"train", "val"} else None
        if src_root is None or dest_root is None or src_root == dest_root:
            return 0
        moved = 0
        for task in DATASET_TASKS:
            src_lp = src_root / task / "labels.json"
            if not src_lp.exists():
                continue
            try:
                with open(src_lp) as f:
                    src_labels = json.load(f)
            except Exception:
                continue
            to_move = {k: v for k, v in src_labels.items() if _project_from_stem(Path(k).stem) == project_id}
            if not to_move:
                continue
            dest_lp = dest_root / task / "labels.json"
            dest_images = dest_root / task / "images"
            dest_images.mkdir(parents=True, exist_ok=True)
            dest_labels: dict = {}
            if dest_lp.exists():
                try:
                    with open(dest_lp) as f:
                        dest_labels = json.load(f)
                except Exception:
                    pass
            src_images = src_root / task / "images"
            for img_name, meta in to_move.items():
                src_img = src_images / img_name
                if src_img.exists():
                    shutil.move(str(src_img), dest_images / img_name)
                del src_labels[img_name]
                dest_labels[img_name] = meta
                if task == "detection":
                    moved += 1
            with open(src_lp, "w") as f:
                json.dump(src_labels, f, indent=2)
            with open(dest_lp, "w") as f:
                json.dump(dest_labels, f, indent=2)
        return moved

    def move_existing_page(self, page_name: str, from_split: str, to_split: str) -> int:
        """Physically move one on-disk page (and its recognition crops) between splits."""
        src_root = self.split_root(from_split) if from_split in {"train", "val"} else None
        dest_root = self.split_root(to_split) if to_split in {"train", "val"} else None
        if src_root is None or dest_root is None or src_root == dest_root:
            return 0

        page_stem = Path(page_name).stem
        moved = 0
        for task in DATASET_TASKS:
            src_lp = src_root / task / "labels.json"
            if not src_lp.exists():
                continue
            try:
                with open(src_lp) as f:
                    src_labels = json.load(f)
            except Exception:
                continue

            if task == "detection":
                keys = [k for k in src_labels if Path(k).stem == page_stem]
            else:
                keys = [k for k in src_labels if Path(k).stem.startswith(f"{page_stem}_")]

            if not keys:
                continue

            dest_lp = dest_root / task / "labels.json"
            dest_images = dest_root / task / "images"
            dest_images.mkdir(parents=True, exist_ok=True)

            dest_labels: dict = {}
            if dest_lp.exists():
                try:
                    with open(dest_lp) as f:
                        dest_labels = json.load(f)
                except Exception:
                    pass

            src_images = src_root / task / "images"
            for key in keys:
                src_img = src_images / key
                if src_img.exists():
                    shutil.move(str(src_img), dest_images / key)
                dest_labels[key] = src_labels[key]
                del src_labels[key]

            if task == "detection":
                moved += len(keys)

            with open(src_lp, "w") as f:
                json.dump(src_labels, f, indent=2)
            with open(dest_lp, "w") as f:
                json.dump(dest_labels, f, indent=2)

        return moved

    def save_assignments(
        self,
        include_detection: bool = True,
        include_recognition: bool = True,
    ) -> dict[str, int]:
        """Merge assigned DocTR exports into ML_TRAINING_DIR / ML_VALIDATION_DIR."""
        full_copy = [(k, v) for k, v in self.assignments.items() if v in {"train", "val"}]
        page_copy = [
            ((k, page), split) for (k, page), split in self.page_assignments.items() if split in {"train", "val"}
        ]
        if not full_copy and not page_copy:
            return {"copied": 0}

        task_flags = {
            "detection": include_detection,
            "recognition": include_recognition,
        }
        plan: dict[tuple[str, str], set[str] | None] = {}
        for key, split in full_copy:
            plan[(key, split)] = None
        for (key, page_name), split in page_copy:
            bucket = plan.get((key, split))
            if bucket is None and (key, split) in plan:
                continue
            if bucket is None:
                bucket = set()
                plan[(key, split)] = bucket
            bucket.add(page_name)

        count = 0
        for (key, split), selected_pages in plan.items():
            src_root = self.export_path(key)
            dest_root = self.split_root("train" if split == "train" else "val")
            copied_any_task = False
            for task, include in task_flags.items():
                if not include:
                    continue
                src_labels_path = src_root / task / "labels.json"
                if not src_labels_path.exists():
                    continue
                src_images_dir = src_root / task / "images"
                dest_images_dir = dest_root / task / "images"
                dest_images_dir.mkdir(parents=True, exist_ok=True)

                src_labels = self._load_json_map(src_labels_path)
                if selected_pages is not None:
                    if task == "detection":
                        src_labels = {k: v for k, v in src_labels.items() if k in selected_pages}
                    else:
                        selected_stems = {Path(name).stem for name in selected_pages}
                        src_labels = {
                            k: v
                            for k, v in src_labels.items()
                            if any(Path(k).stem.startswith(f"{stem}_") for stem in selected_stems)
                        }

                if not src_labels:
                    continue

                dest_labels_path = dest_root / task / "labels.json"
                dest_labels = self._load_json_map(dest_labels_path)

                for img_name in src_labels:
                    src_img = src_images_dir / img_name
                    if src_img.exists():
                        shutil.copy2(src_img, dest_images_dir / img_name)

                dest_labels.update(src_labels)
                self._write_json_map(dest_labels_path, dest_labels)
                copied_any_task = True
                count += 1

            if copied_any_task:
                if selected_pages is None:
                    self.assignments[key] = None
                    self.page_assignments = {(k, page): s for (k, page), s in self.page_assignments.items() if k != key}
                else:
                    for page_name in selected_pages:
                        self.page_assignments.pop((key, page_name), None)

        self.scan()
        return {"copied": count}


class DetectionTrainingConfig:
    """Manages detection fine-tuning configuration."""

    def __init__(self):
        self.enabled = True
        self.arch = "db_resnet50"
        self.epochs = 100
        self.batch_size = 2
        self.workers = _default_worker_count()
        self.learning_rate = 0.002
        self.weight_decay = 0.0
        self.optimizer = "adam"
        self.scheduler = "poly"
        self.input_size = 1024
        self.rotation = False
        self.amp = False
        self.pretrained = True
        self.early_stop = False
        self.early_stop_epochs = 5
        self.early_stop_delta = 0.01
        self.model_name = _prefixed_model_name("detection", f"model-finetuned-{_default_model_timestamp()}")
        self.device = 0 if torch.cuda.is_available() else None


class RecognitionTrainingConfig:
    """Manages recognition fine-tuning configuration."""

    def __init__(self):
        self.enabled = True
        self.arch = "crnn_vgg16_bn"
        self.epochs = 100
        self.batch_size = 64
        self.learning_rate = 0.001
        self.weight_decay = 0.0
        self.optimizer = "adam"
        self.scheduler = "cosine"
        self.input_size = 32
        self.pretrained = True
        self.model_name = _prefixed_model_name("recognition", f"model-finetuned-{_default_model_timestamp()}")
        self.amp = False
        self.early_stop = False
        self.early_stop_epochs = 5
        self.early_stop_delta = 0.01
        default_vocab_library = [name for name in DEFAULT_VOCAB_LIBRARY if name in VOCABS]
        if not default_vocab_library:
            default_vocab_library = ["french"] if "french" in VOCABS else [next(iter(VOCABS))]
        self.vocab_library = default_vocab_library
        self.custom_characters = DEFAULT_CUSTOM_CHARACTERS
        self.vocab = build_custom_vocab_arg(self.vocab_library, self.custom_characters)
        self.workers = _default_worker_count()
        self.device = 0 if torch.cuda.is_available() else None


def _reset_model_names_to_defaults() -> None:
    """Recompute model names at load time; names are never persisted."""
    detection_config.model_name = _prefixed_model_name("detection", f"model-finetuned-{_default_model_timestamp()}")
    recognition_config.model_name = _prefixed_model_name("recognition", f"model-finetuned-{_default_model_timestamp()}")


def _detection_settings_payload() -> dict:
    return {
        "arch": detection_config.arch,
        "epochs": int(detection_config.epochs),
        "batch_size": int(detection_config.batch_size),
        "workers": int(detection_config.workers),
        "learning_rate": float(detection_config.learning_rate),
        "weight_decay": float(detection_config.weight_decay),
        "optimizer": detection_config.optimizer,
        "scheduler": detection_config.scheduler,
        "input_size": int(detection_config.input_size),
        "rotation": bool(detection_config.rotation),
        "amp": bool(detection_config.amp),
        "pretrained": bool(detection_config.pretrained),
        "early_stop": bool(detection_config.early_stop),
        "early_stop_epochs": int(detection_config.early_stop_epochs),
        "early_stop_delta": float(detection_config.early_stop_delta),
    }


def _recognition_settings_payload() -> dict:
    return {
        "arch": recognition_config.arch,
        "epochs": int(recognition_config.epochs),
        "batch_size": int(recognition_config.batch_size),
        "learning_rate": float(recognition_config.learning_rate),
        "weight_decay": float(recognition_config.weight_decay),
        "optimizer": recognition_config.optimizer,
        "scheduler": recognition_config.scheduler,
        "input_size": int(recognition_config.input_size),
        "workers": int(recognition_config.workers),
        "pretrained": bool(recognition_config.pretrained),
        "amp": bool(recognition_config.amp),
        "early_stop": bool(recognition_config.early_stop),
        "early_stop_epochs": int(recognition_config.early_stop_epochs),
        "early_stop_delta": float(recognition_config.early_stop_delta),
        "vocab_library": list(recognition_config.vocab_library),
        "custom_characters": recognition_config.custom_characters,
    }


def _save_trainer_settings() -> None:
    payload = {
        "version": 1,
        "detection": _detection_settings_payload(),
        "recognition": _recognition_settings_payload(),
    }
    TRAINER_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAINER_SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _apply_detection_settings(data: dict) -> None:
    detection_config.arch = str(data.get("arch", detection_config.arch))
    detection_config.epochs = int(data.get("epochs", detection_config.epochs))
    detection_config.batch_size = int(data.get("batch_size", detection_config.batch_size))
    detection_config.workers = int(data.get("workers", detection_config.workers))
    detection_config.learning_rate = float(data.get("learning_rate", detection_config.learning_rate))
    detection_config.weight_decay = float(data.get("weight_decay", detection_config.weight_decay))
    detection_config.optimizer = str(data.get("optimizer", detection_config.optimizer))
    detection_config.scheduler = str(data.get("scheduler", detection_config.scheduler))
    detection_config.input_size = int(data.get("input_size", detection_config.input_size))
    detection_config.rotation = bool(data.get("rotation", detection_config.rotation))
    detection_config.amp = bool(data.get("amp", detection_config.amp))
    detection_config.pretrained = bool(data.get("pretrained", detection_config.pretrained))
    detection_config.early_stop = bool(data.get("early_stop", detection_config.early_stop))
    detection_config.early_stop_epochs = int(data.get("early_stop_epochs", detection_config.early_stop_epochs))
    detection_config.early_stop_delta = float(data.get("early_stop_delta", detection_config.early_stop_delta))


def _apply_recognition_settings(data: dict) -> None:
    recognition_config.arch = str(data.get("arch", recognition_config.arch))
    recognition_config.epochs = int(data.get("epochs", recognition_config.epochs))
    recognition_config.batch_size = int(data.get("batch_size", recognition_config.batch_size))
    recognition_config.learning_rate = float(data.get("learning_rate", recognition_config.learning_rate))
    recognition_config.weight_decay = float(data.get("weight_decay", recognition_config.weight_decay))
    recognition_config.optimizer = str(data.get("optimizer", recognition_config.optimizer))
    recognition_config.scheduler = str(data.get("scheduler", recognition_config.scheduler))
    recognition_config.input_size = int(data.get("input_size", recognition_config.input_size))
    recognition_config.workers = int(data.get("workers", recognition_config.workers))
    recognition_config.pretrained = bool(data.get("pretrained", recognition_config.pretrained))
    recognition_config.amp = bool(data.get("amp", recognition_config.amp))
    recognition_config.early_stop = bool(data.get("early_stop", recognition_config.early_stop))
    recognition_config.early_stop_epochs = int(data.get("early_stop_epochs", recognition_config.early_stop_epochs))
    recognition_config.early_stop_delta = float(data.get("early_stop_delta", recognition_config.early_stop_delta))

    loaded_vocab_library = data.get("vocab_library")
    if isinstance(loaded_vocab_library, list):
        cleaned = [str(name) for name in loaded_vocab_library if isinstance(name, str) and name in VOCABS]
        if cleaned:
            recognition_config.vocab_library = cleaned

    loaded_custom_characters = data.get("custom_characters")
    if isinstance(loaded_custom_characters, str):
        recognition_config.custom_characters = loaded_custom_characters

    recognition_config.vocab = build_custom_vocab_arg(
        recognition_config.vocab_library,
        recognition_config.custom_characters,
    )


def _load_trainer_settings() -> None:
    if not TRAINER_SETTINGS_PATH.exists():
        _reset_model_names_to_defaults()
        return

    try:
        with open(TRAINER_SETTINGS_PATH, encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            _reset_model_names_to_defaults()
            return

        detection_data = payload.get("detection", {})
        recognition_data = payload.get("recognition", {})
        if isinstance(detection_data, dict):
            _apply_detection_settings(detection_data)
        if isinstance(recognition_data, dict):
            _apply_recognition_settings(recognition_data)
    except Exception:
        pass
    finally:
        _reset_model_names_to_defaults()

    # Global state
    _model_output_dir(BASE_OCR_PROFILE, "detection").mkdir(parents=True, exist_ok=True)
    _model_output_dir(BASE_OCR_PROFILE, "recognition").mkdir(parents=True, exist_ok=True)


_migrate_legacy_dataset_layout()
export_manager = ExportManager()
detection_config = DetectionTrainingConfig()
recognition_config = RecognitionTrainingConfig()
_load_trainer_settings()
training_thread: threading.Thread | None = None
training_cancelled = False


def create_ui():
    """Build the NiceGUI interface."""

    model_profile_state = {"value": BASE_OCR_PROFILE}
    export_manager.set_profile(model_profile_state["value"])

    with ui.header().classes("w-full bg-blue-500 text-white"):
        ui.label("OCR Training Suite").classes("text-2xl font-bold")

    with ui.column().classes("w-full"):
        output_labels: dict[str, object] = {}
        dataset_scope_label: dict[str, object] = {}

        def _safe_int(value, fallback: int) -> int:
            if value in (None, ""):
                return fallback
            try:
                return int(value)
            except (TypeError, ValueError):
                return fallback

        def _safe_float(value, fallback: float) -> float:
            if value in (None, ""):
                return fallback
            try:
                return float(value)
            except (TypeError, ValueError):
                return fallback

        # ==================== PROFILE SECTION ====================
        with ui.card().classes("w-full"):
            ui.label("🎯 Model Profile").classes("font-semibold")
            profile_options = get_available_model_profiles()
            if model_profile_state["value"] not in profile_options:
                profile_options = [model_profile_state["value"], *profile_options]

            def refresh_model_output_labels() -> None:
                profile = _normalize_profile_name(model_profile_state["value"])
                det_label = output_labels.get("detection")
                rec_label = output_labels.get("recognition")
                dataset_label = dataset_scope_label.get("value")
                if det_label is not None:
                    det_label.set_text(f"Detection output root: {_model_output_dir(profile, 'detection')}")
                if rec_label is not None:
                    rec_label.set_text(f"Recognition output root: {_model_output_dir(profile, 'recognition')}")
                if dataset_label is not None:
                    dataset_label.set_text(f"Showing exports and dataset files for profile '{profile}'.")
                export_manager.set_profile(profile)
                export_manager.scan()

            ui.select(
                label="Training profile",
                options=profile_options,
                value=model_profile_state["value"],
                on_change=lambda v: (
                    model_profile_state.__setitem__("value", _normalize_profile_name(str(v.value))),
                    refresh_model_output_labels(),
                    refresh_kanban(),
                ),
            ).classes("w-64")

            ui.label(
                "Select the profile first. Dataset management below shows only exports and datasets for that profile."
            ).classes("text-xs text-gray-500")

            ui.label(
                "Use all for all-word training. Create/select style-specific profiles (e.g. italics)"
                " to keep model artifacts separated."
            ).classes("text-xs text-gray-500")

            refresh_model_output_labels()

        # ==================== DATASET SECTION ====================
        with ui.card().classes("w-full"):
            ui.label("📂 Dataset Management").classes("text-lg font-bold")
            dataset_scope_label["value"] = ui.label("").classes("text-xs text-gray-500")
            ui.label(
                "Auto-populated from the pd-ocr-labeler DocTR export root. "
                "Yellow items are already present in the training/validation datasets. "
                "Drag project rows or individual subfolder chips between columns. "
                "For on-disk pages: click to select, Ctrl/Cmd-click to toggle, Shift-click for range."
            ).classes("text-xs text-gray-500 mb-2")

            copy_detection = {"value": True}
            copy_recognition = {"value": True}

            with ui.row().classes("items-center gap-6 mb-3"):
                ui.checkbox(
                    text="Include detection dataset",
                    value=True,
                    on_change=lambda v: copy_detection.__setitem__("value", bool(v.value)),
                )
                ui.checkbox(
                    text="Include recognition dataset",
                    value=True,
                    on_change=lambda v: copy_recognition.__setitem__("value", bool(v.value)),
                )

            kanban_status = ui.label("").classes("text-xs text-gray-600")
            dragging: dict = {"type": None, "key": None}
            col_containers: dict = {}
            selected_existing_pages: set[tuple[str, str]] = set()
            page_selection_anchor: dict[str, str] = {}

            def _select_existing_page(
                event,
                split_col: str,
                project_id: str,
                page_name: str,
                ordered_pages: list[str],
            ) -> None:
                args = event.args if hasattr(event, "args") and isinstance(event.args, dict) else {}
                shift_pressed = bool(args.get("shiftKey"))
                toggle_pressed = bool(args.get("ctrlKey") or args.get("metaKey"))
                key = (split_col, page_name)
                scope_key = f"{split_col}:{project_id}"

                if shift_pressed and page_selection_anchor.get(scope_key) in ordered_pages:
                    start = ordered_pages.index(page_selection_anchor[scope_key])
                    end = ordered_pages.index(page_name)
                    if start > end:
                        start, end = end, start
                    range_keys = {(split_col, p) for p in ordered_pages[start : end + 1]}
                    if toggle_pressed:
                        selected_existing_pages.update(range_keys)
                    else:
                        selected_existing_pages.difference_update(
                            {selected for selected in selected_existing_pages if selected[0] == split_col}
                        )
                        selected_existing_pages.update(range_keys)
                elif toggle_pressed:
                    if key in selected_existing_pages:
                        selected_existing_pages.remove(key)
                    else:
                        selected_existing_pages.add(key)
                else:
                    selected_existing_pages.clear()
                    selected_existing_pages.add(key)

                page_selection_anchor[scope_key] = page_name
                refresh_kanban()

            def handle_drop(target: str) -> None:
                dtype = dragging.get("type")
                key = dragging.get("key")
                if dtype == "export" and isinstance(key, str):
                    export_manager.assign(key, target)
                elif dtype == "export_page" and isinstance(key, tuple) and len(key) == 2:
                    export_key, page_name = key
                    if isinstance(export_key, str) and isinstance(page_name, str):
                        export_manager.assign_page(export_key, page_name, target)
                elif dtype == "project_pages" and isinstance(key, list):
                    page_tuples = [k for k in key if isinstance(k, tuple) and len(k) == 2]
                    normalized = [(k, p) for k, p in page_tuples if isinstance(k, str) and isinstance(p, str)]
                    if normalized:
                        export_manager.assign_pages(normalized, target)
                elif dtype == "project" and isinstance(key, str):
                    export_manager.assign_project(key, target)
                elif dtype == "existing" and isinstance(key, str):
                    from_split = dragging.get("from_split")
                    if isinstance(from_split, str) and from_split != target and target in ("train", "val"):
                        export_manager.move_existing_project(key, from_split, target)
                elif dtype == "existing_page" and isinstance(key, str):
                    from_split = dragging.get("from_split")
                    if isinstance(from_split, str) and from_split != target and target in ("train", "val"):
                        export_manager.move_existing_page(key, from_split, target)
                elif dtype == "existing_page" and isinstance(key, list):
                    from_split = dragging.get("from_split")
                    if isinstance(from_split, str) and from_split != target and target in ("train", "val"):
                        for page_name in key:
                            if isinstance(page_name, str):
                                export_manager.move_existing_page(page_name, from_split, target)
                        selected_existing_pages.difference_update(
                            {selected for selected in selected_existing_pages if selected[0] == from_split}
                        )
                dragging["type"] = None
                dragging["key"] = None
                dragging["from_split"] = None
                refresh_kanban()

            def render_column(col_id: str, container) -> None:
                container.clear()
                by_split_pages = export_manager.get_export_pages_by_split()
                pending_projects = dict(sorted(by_split_pages.get(col_id, {}).items()))

                # Existing data already in ml-training / ml-validation.
                existing_projects: dict[str, int] = {}
                if col_id == "train":
                    existing_projects = export_manager.get_existing_projects(export_manager.split_root("train"))
                elif col_id == "val":
                    existing_projects = export_manager.get_existing_projects(export_manager.split_root("val"))

                has_content = pending_projects or existing_projects
                with container:
                    if not has_content:
                        ui.label("(empty — drop items here)").classes("text-xs text-gray-400 italic p-2")
                        return

                    # --- Pending (from labeler export root) ---
                    for project_id, pages_by_key in pending_projects.items():
                        page_count = sum(len(pages) for pages in pages_by_key.values())
                        exp_label = f"{project_id}  ·  {page_count} pages  [export]"
                        with ui.expansion(exp_label).classes(
                            "w-full mb-1 bg-slate-50 border border-slate-200 rounded"
                        ) as exp_card:
                            exp_card.props("dense")
                            with ui.column().classes("w-full gap-1 pl-2"):
                                with ui.card().classes(
                                    "w-full mb-1 px-2 py-1 border border-dashed rounded shadow-none bg-slate-100 text-slate-700 cursor-grab"
                                ) as project_drag_row:
                                    project_drag_row.props("draggable=true")
                                    project_drag_row.on(
                                        "dragstart",
                                        lambda e, pages=pages_by_key: (
                                            dragging.__setitem__("type", "project_pages"),
                                            dragging.__setitem__(
                                                "key",
                                                [
                                                    (k, page_name)
                                                    for k, page_names in pages.items()
                                                    for page_name in page_names
                                                ],
                                            ),
                                        ),
                                    )
                                    ui.label("Drag to move all pages").classes("text-xs")
                                for key, pages in pages_by_key.items():
                                    changed = export_manager.is_changed(key)
                                    for page_name in pages:
                                        row_cls = "w-full mb-1 px-2 py-1 border rounded shadow-none cursor-grab " + (
                                            "bg-yellow-100 border-yellow-400 text-yellow-800"
                                            if changed
                                            else "bg-white border-slate-200 text-slate-600"
                                        )
                                        with ui.card().classes(row_cls) as export_row:
                                            export_row.props("draggable=true")
                                            export_row.on(
                                                "dragstart.stop",
                                                lambda e, k=key, p=page_name: (
                                                    dragging.__setitem__("type", "export_page"),
                                                    dragging.__setitem__("key", (k, p)),
                                                ),
                                            )
                                            ui.label(page_name).classes("text-xs font-mono")

                    # --- Already present in ml-training / ml-validation ---
                    if existing_projects:
                        if col_id in ("train", "val") and pending_projects:
                            ui.separator()
                        for project_id, page_count in existing_projects.items():
                            pages = export_manager.get_existing_pages(
                                export_manager.split_root("train" if col_id == "train" else "val"),
                                project_id,
                            )
                            exp_label = f"{project_id}  ·  {page_count} pages  [on disk]"
                            with ui.expansion(exp_label).classes(
                                "w-full mb-1 bg-slate-50 border border-slate-200 rounded"
                            ) as exp_card:
                                exp_card.props("dense")
                                with ui.column().classes("w-full gap-0 pl-2"):
                                    with ui.card().classes(
                                        "w-full mb-1 px-2 py-1 border border-dashed rounded shadow-none bg-slate-100 text-slate-700 cursor-grab"
                                    ) as project_drag_row:
                                        project_drag_row.props("draggable=true")
                                        project_drag_row.on(
                                            "dragstart",
                                            lambda e, p=project_id, s=col_id: (
                                                dragging.__setitem__("type", "existing"),
                                                dragging.__setitem__("key", p),
                                                dragging.__setitem__("from_split", s),
                                            ),
                                        )
                                        ui.label("Drag to move all pages").classes("text-xs")
                                    for img_name in pages:
                                        selected = (col_id, img_name) in selected_existing_pages
                                        selected_cls = (
                                            "bg-blue-100 border-blue-400 text-blue-800"
                                            if selected
                                            else "bg-white border-slate-200 text-slate-600"
                                        )
                                        with ui.card().classes(
                                            "w-full mb-1 px-2 py-1 border rounded shadow-none cursor-grab "
                                            + selected_cls
                                        ) as page_row:
                                            page_row.props("draggable=true")
                                            page_row.on(
                                                "click",
                                                lambda e, p=img_name, s=col_id, pr=project_id, ordered=pages: (
                                                    _select_existing_page(e, s, pr, p, ordered)
                                                ),
                                            )
                                            page_row.on(
                                                "dragstart.stop",
                                                lambda e, p=img_name, s=col_id: (
                                                    dragging.__setitem__("type", "existing_page"),
                                                    dragging.__setitem__(
                                                        "key",
                                                        sorted(
                                                            [
                                                                name
                                                                for split, name in selected_existing_pages
                                                                if split == s
                                                            ]
                                                        )
                                                        if (s, p) in selected_existing_pages
                                                        and any(split == s for split, _ in selected_existing_pages)
                                                        else p,
                                                    ),
                                                    dragging.__setitem__("from_split", s),
                                                ),
                                            )
                                            ui.label(img_name).classes("text-xs font-mono")

            def refresh_kanban() -> None:
                for col_id, container in col_containers.items():
                    render_column(col_id, container)

            COLUMN_DEFS = [
                ("unassigned", "📋 Unassigned", "border-gray-300"),
                ("train", "🔵 Training", "border-blue-400"),
                ("val", "🟢 Validation", "border-teal-400"),
            ]

            with ui.row().classes("w-full gap-4"):
                for col_id, col_title, col_border_cls in COLUMN_DEFS:
                    with ui.card().classes(f"flex-1 min-h-40 border-2 {col_border_cls}") as column_card:
                        with ui.row().classes("items-center justify-between w-full mb-1"):
                            ui.label(col_title).classes("font-semibold text-sm")
                            if col_id != "unassigned":

                                def _make_clear(t: str):
                                    def _clear():
                                        export_manager.clear_split(t)
                                        refresh_kanban()

                                    return _clear

                                ui.button("Clear", on_click=_make_clear(col_id)).props("size=xs color=negative flat")
                        column_card.on("dragover.prevent", lambda e: None)
                        column_card.on("drop", lambda e, t=col_id: handle_drop(t))
                        drop_area = ui.column().classes("w-full gap-1 min-h-16")
                        col_containers[col_id] = drop_area

            with ui.row().classes("items-center gap-4 mt-3"):
                ui.button(
                    "🔄 Refresh",
                    on_click=lambda: (
                        export_manager.scan(),
                        refresh_kanban(),
                        kanban_status.set_text("Refreshed exports from labeler."),
                    ),
                ).props("color=secondary")

                def save_assignments() -> None:
                    try:
                        result = export_manager.save_assignments(
                            include_detection=copy_detection["value"],
                            include_recognition=copy_recognition["value"],
                        )
                        count = result.get("copied", 0)
                        if count == 0:
                            ui.notify("No assigned exports to copy.", type="warning")
                            return
                        refresh_kanban()
                        kanban_status.set_text(
                            f"💾 Copied {count} export task(s) into '{export_manager.active_profile}' datasets."
                        )
                        ui.notify("Assignments saved.", type="positive")
                    except Exception as e:
                        ui.notify(str(e), type="negative")

                ui.button("💾 Copy to Datasets", on_click=save_assignments).props("color=primary")

            refresh_kanban()

        # ==================== TRAINING CONFIG SECTION ====================
        with ui.row().classes("w-full mb-2"):
            ui.button(
                "Start Full Training (Detection -> Recognition)",
                on_click=lambda: run_training("both"),
            ).props("color=primary")
            ui.button(
                "Save Settings",
                on_click=lambda: (_save_trainer_settings(), ui.notify("Settings saved.", type="positive")),
            ).props("color=secondary")

        with ui.row().classes("w-full gap-4 items-start"):
            with ui.card().classes("flex-1"):
                with ui.row().classes("w-full items-center justify-between"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("settings")
                        ui.label("Detection Configuration").classes("font-semibold")
                    ui.button(
                        "Start Detection Training",
                        on_click=lambda: run_training("detection"),
                    ).props("color=green")

                ui.label("Detection fine-tune settings").classes("font-semibold text-sm")
                output_labels["detection"] = ui.label("").classes("text-xs text-gray-600")
                refresh_model_output_labels()

                with ui.expansion("Configuration", icon="tune").classes("w-full"):
                    detection_arch_help_label = ui.label(
                        DETECTION_ARCH_HELP.get(detection_config.arch, "Select an architecture to see details.")
                    ).classes("text-xs text-gray-600")

                    with ui.row().classes("items-center gap-2"):
                        ui.label("Architecture").classes("text-sm")
                        with ui.icon("help_outline").classes("text-gray-500 cursor-help"):
                            ui.tooltip(
                                "Architecture notes update below when you change the selection. "
                                "Summaries are based on docTR docs and model zoo references."
                            )

                    def _on_detection_arch_change(event):
                        detection_config.arch = event.value
                        detection_arch_help_label.set_text(
                            DETECTION_ARCH_HELP.get(event.value, "No details available for this architecture.")
                        )

                    ui.select(
                        label="",
                        options=DETECTION_ARCH_OPTIONS,
                        value=detection_config.arch,
                        on_change=_on_detection_arch_change,
                    ).classes("w-full")

                    ui.number(
                        label="Epochs",
                        value=detection_config.epochs,
                        min=1,
                        max=300,
                        on_change=lambda v: setattr(
                            detection_config,
                            "epochs",
                            _safe_int(v.value, detection_config.epochs),
                        ),
                    ).classes("w-full")

                    ui.number(
                        label="Batch Size",
                        value=detection_config.batch_size,
                        min=1,
                        max=64,
                        on_change=lambda v: setattr(
                            detection_config,
                            "batch_size",
                            _safe_int(v.value, detection_config.batch_size),
                        ),
                    ).classes("w-full")

                    ui.number(
                        label="Workers",
                        value=detection_config.workers,
                        min=0,
                        max=16,
                        on_change=lambda v: setattr(
                            detection_config,
                            "workers",
                            _safe_int(v.value, detection_config.workers),
                        ),
                    ).classes("w-full")

                    ui.number(
                        label="Learning Rate",
                        value=detection_config.learning_rate,
                        min=0.00001,
                        max=0.1,
                        step=0.0001,
                        format="%.5f",
                        on_change=lambda v: setattr(
                            detection_config,
                            "learning_rate",
                            _safe_float(v.value, detection_config.learning_rate),
                        ),
                    ).classes("w-full")

                    ui.number(
                        label="Weight Decay",
                        value=detection_config.weight_decay,
                        min=0,
                        max=0.1,
                        step=0.001,
                        format="%.4f",
                        on_change=lambda v: setattr(
                            detection_config,
                            "weight_decay",
                            _safe_float(v.value, detection_config.weight_decay),
                        ),
                    ).classes("w-full")

                    ui.select(
                        label="Optimizer",
                        options=["adam", "adamw"],
                        value=detection_config.optimizer,
                        on_change=lambda v: setattr(detection_config, "optimizer", v.value),
                    ).classes("w-full")

                    ui.select(
                        label="Scheduler",
                        options=["cosine", "onecycle", "poly"],
                        value=detection_config.scheduler,
                        on_change=lambda v: setattr(detection_config, "scheduler", v.value),
                    ).classes("w-full")

                    ui.checkbox(
                        text="Mixed Precision (AMP)",
                        value=detection_config.amp,
                        on_change=lambda v: setattr(detection_config, "amp", v.value),
                    ).classes("w-full")

                    ui.checkbox(
                        text="Use Pretrained Weights",
                        value=detection_config.pretrained,
                        on_change=lambda v: setattr(detection_config, "pretrained", v.value),
                    ).classes("w-full")

                    ui.checkbox(
                        text="Early Stopping",
                        value=detection_config.early_stop,
                        on_change=lambda v: setattr(detection_config, "early_stop", v.value),
                    ).classes("w-full")

                    ui.number(
                        label="Early Stop Patience",
                        value=detection_config.early_stop_epochs,
                        min=1,
                        max=20,
                        on_change=lambda v: setattr(
                            detection_config,
                            "early_stop_epochs",
                            _safe_int(v.value, detection_config.early_stop_epochs),
                        ),
                    ).classes("w-full")

                    ui.number(
                        label="Early Stop Delta",
                        value=detection_config.early_stop_delta,
                        min=0,
                        max=1,
                        step=0.001,
                        format="%.3f",
                        on_change=lambda v: setattr(
                            detection_config,
                            "early_stop_delta",
                            _safe_float(v.value, detection_config.early_stop_delta),
                        ),
                    ).classes("w-full")

                    ui.separator().classes("my-2")
                    ui.label("Advanced").classes("text-sm font-medium")

                    ui.number(
                        label="Input Size",
                        value=detection_config.input_size,
                        min=256,
                        max=2048,
                        step=32,
                        on_change=lambda v: setattr(
                            detection_config,
                            "input_size",
                            _safe_int(v.value, detection_config.input_size),
                        ),
                    ).classes("w-full")

                    ui.checkbox(
                        text="Enable Rotation",
                        value=detection_config.rotation,
                        on_change=lambda v: setattr(detection_config, "rotation", v.value),
                    ).classes("w-full")

                    ui.input(
                        label="Model Name",
                        value=detection_config.model_name,
                        on_change=lambda v: setattr(detection_config, "model_name", v.value),
                    ).classes("w-full")

            with ui.card().classes("flex-1"):
                with ui.row().classes("w-full items-center justify-between"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("settings")
                        ui.label("Recognition Configuration").classes("font-semibold")
                    ui.button(
                        "Start Recognition Training",
                        on_click=lambda: run_training("recognition"),
                    ).props("color=green")

                ui.label("Recognition fine-tune settings").classes("font-semibold text-sm")
                output_labels["recognition"] = ui.label("").classes("text-xs text-gray-600")
                refresh_model_output_labels()

                with ui.expansion("Configuration", icon="tune").classes("w-full"):
                    recognition_arch_help_label = ui.label(
                        RECOGNITION_ARCH_HELP.get(recognition_config.arch, "Select an architecture to see details.")
                    ).classes("text-xs text-gray-600")

                    with ui.row().classes("items-center gap-2"):
                        ui.label("Architecture").classes("text-sm")
                        with ui.icon("help_outline").classes("text-gray-500 cursor-help"):
                            ui.tooltip(
                                "Architecture notes update below when you change the selection. "
                                "Summaries are based on docTR docs and model zoo references."
                            )

                    def _on_recognition_arch_change(event):
                        recognition_config.arch = event.value
                        recognition_arch_help_label.set_text(
                            RECOGNITION_ARCH_HELP.get(event.value, "No details available for this architecture.")
                        )

                    ui.select(
                        label="",
                        options=RECOGNITION_ARCH_OPTIONS,
                        value=recognition_config.arch,
                        on_change=_on_recognition_arch_change,
                    ).classes("w-full")

                    ui.number(
                        label="Epochs",
                        value=recognition_config.epochs,
                        min=1,
                        max=300,
                        on_change=lambda v: setattr(
                            recognition_config,
                            "epochs",
                            _safe_int(v.value, recognition_config.epochs),
                        ),
                    ).classes("w-full")

                    ui.number(
                        label="Batch Size",
                        value=recognition_config.batch_size,
                        min=1,
                        max=512,
                        on_change=lambda v: setattr(
                            recognition_config,
                            "batch_size",
                            _safe_int(v.value, recognition_config.batch_size),
                        ),
                    ).classes("w-full")

                    ui.number(
                        label="Learning Rate",
                        value=recognition_config.learning_rate,
                        min=0.00001,
                        max=0.1,
                        step=0.0001,
                        format="%.5f",
                        on_change=lambda v: setattr(
                            recognition_config,
                            "learning_rate",
                            _safe_float(v.value, recognition_config.learning_rate),
                        ),
                    ).classes("w-full")

                    ui.number(
                        label="Weight Decay",
                        value=recognition_config.weight_decay,
                        min=0,
                        max=0.1,
                        step=0.001,
                        format="%.4f",
                        on_change=lambda v: setattr(
                            recognition_config,
                            "weight_decay",
                            _safe_float(v.value, recognition_config.weight_decay),
                        ),
                    ).classes("w-full")

                    ui.select(
                        label="Optimizer",
                        options=["adam", "adamw"],
                        value=recognition_config.optimizer,
                        on_change=lambda v: setattr(recognition_config, "optimizer", v.value),
                    ).classes("w-full")

                    ui.select(
                        label="Scheduler",
                        options=["cosine", "onecycle", "poly"],
                        value=recognition_config.scheduler,
                        on_change=lambda v: setattr(recognition_config, "scheduler", v.value),
                    ).classes("w-full")

                    ui.number(
                        label="Input Height",
                        value=recognition_config.input_size,
                        min=16,
                        max=64,
                        step=4,
                        on_change=lambda v: setattr(
                            recognition_config,
                            "input_size",
                            _safe_int(v.value, recognition_config.input_size),
                        ),
                    ).classes("w-full")

                    ui.number(
                        label="Workers",
                        value=recognition_config.workers,
                        min=0,
                        max=16,
                        on_change=lambda v: setattr(
                            recognition_config,
                            "workers",
                            _safe_int(v.value, recognition_config.workers),
                        ),
                    ).classes("w-full")

                    ui.checkbox(
                        text="Use Pretrained Weights",
                        value=recognition_config.pretrained,
                        on_change=lambda v: setattr(recognition_config, "pretrained", v.value),
                    ).classes("w-full")

                    ui.checkbox(
                        text="Mixed Precision (AMP)",
                        value=recognition_config.amp,
                        on_change=lambda v: setattr(recognition_config, "amp", v.value),
                    ).classes("w-full")

                    ui.checkbox(
                        text="Early Stopping",
                        value=recognition_config.early_stop,
                        on_change=lambda v: setattr(recognition_config, "early_stop", v.value),
                    ).classes("w-full")

                    ui.number(
                        label="Early Stop Patience",
                        value=recognition_config.early_stop_epochs,
                        min=1,
                        max=20,
                        on_change=lambda v: setattr(
                            recognition_config,
                            "early_stop_epochs",
                            _safe_int(v.value, recognition_config.early_stop_epochs),
                        ),
                    ).classes("w-full")

                    ui.number(
                        label="Early Stop Delta",
                        value=recognition_config.early_stop_delta,
                        min=0,
                        max=1,
                        step=0.001,
                        format="%.3f",
                        on_change=lambda v: setattr(
                            recognition_config,
                            "early_stop_delta",
                            _safe_float(v.value, recognition_config.early_stop_delta),
                        ),
                    ).classes("w-full")

                    ui.input(
                        label="Model Name",
                        value=recognition_config.model_name,
                        on_change=lambda v: setattr(recognition_config, "model_name", v.value),
                    ).classes("w-full")

                    vocab_tags_row = ui.row().classes("w-full flex-wrap gap-2")
                    vocab_preview_label = ui.label("").classes("text-xs text-gray-600")

                    def refresh_vocab_ui():
                        recognition_config.vocab = build_custom_vocab_arg(
                            recognition_config.vocab_library,
                            recognition_config.custom_characters,
                        )
                        vocab_preview_label.set_text(
                            "Resolved as CUSTOM vocab. Note: custom vocab is deduplicated and sorted by the trainer."
                        )
                        vocab_tags_row.clear()
                        with vocab_tags_row:
                            for selected_vocab in recognition_config.vocab_library:
                                ui.badge(selected_vocab).props("outline")

                    ui.select(
                        label="Vocab Library (shown as tags)",
                        options=sorted(VOCABS.keys()),
                        value=recognition_config.vocab_library,
                        multiple=True,
                        on_change=lambda v: (
                            setattr(recognition_config, "vocab_library", list(v.value or [])),
                            refresh_vocab_ui(),
                        ),
                    ).props("use-chips").classes("w-full")

                    ui.input(
                        label="Custom Characters",
                        value=recognition_config.custom_characters,
                        on_change=lambda v: (
                            setattr(recognition_config, "custom_characters", v.value or ""),
                            refresh_vocab_ui(),
                        ),
                    ).classes("w-full")

                    ui.button(
                        "Reset to Default Vocab Preset",
                        on_click=lambda: (
                            setattr(
                                recognition_config,
                                "vocab_library",
                                [name for name in DEFAULT_VOCAB_LIBRARY if name in VOCABS],
                            ),
                            setattr(recognition_config, "custom_characters", DEFAULT_CUSTOM_CHARACTERS),
                            refresh_vocab_ui(),
                        ),
                    ).props("color=secondary")

                    def export_vocab_file() -> None:
                        # Refresh vocab to reflect any pending edits, then resolve to the
                        # exact character set the trainer would use and write it as a
                        # sidecar next to where the recognition .pt would land.
                        recognition_config.vocab = build_custom_vocab_arg(
                            recognition_config.vocab_library,
                            recognition_config.custom_characters,
                        )
                        try:
                            from .train_recog import resolve_vocab

                            resolved = resolve_vocab(recognition_config.vocab)
                        except Exception as exc:
                            ui.notify(f"Failed to resolve vocab: {exc}", type="negative")
                            return
                        profile = _normalize_profile_name(model_profile_state["value"])
                        out_dir = _model_output_dir(profile, "recognition")
                        out_dir.mkdir(parents=True, exist_ok=True)
                        vocab_path = out_dir / f"{recognition_config.model_name}.vocab"
                        try:
                            vocab_path.write_text(resolved, encoding="utf-8")
                        except Exception as exc:
                            ui.notify(f"Failed to write vocab file: {exc}", type="negative")
                            return
                        ui.notify(
                            f"Exported {len(resolved)} chars to {vocab_path}",
                            type="positive",
                        )

                    ui.button(
                        "Export Vocab File",
                        on_click=export_vocab_file,
                    ).props("color=primary")

                    refresh_vocab_ui()

    # ==================== TRAINING OUTPUT SECTION ====================
    with ui.card().classes("w-full"):
        ui.label("🧾 Training Output").classes("text-lg font-bold")

        status_label = ui.label("Ready").classes("text-sm text-gray-600")
        output_area = (
            ui.textarea(
                value="Training output will appear here...",
            )
            .props("readonly")
            .classes("w-full h-64")
        )
        ui_events: Queue = Queue()

        def queue_ui_event(event: str, **payload) -> None:
            ui_events.put((event, payload))

        def flush_ui_events() -> None:
            while True:
                try:
                    event, payload = ui_events.get_nowait()
                except Empty:
                    break

                if event == "set_status":
                    status_label.set_text(str(payload.get("text", "")))
                elif event == "append_output":
                    output_area.value += str(payload.get("text", ""))
                    output_area.update()

        # Process worker-thread UI events on the main UI thread.
        ui.timer(0.1, flush_ui_events)

        def run_training(mode: str):
            """Run detection, recognition, or both sequentially in a background thread."""
            global training_thread, training_cancelled

            if training_thread and training_thread.is_alive():
                ui.notify("Training is already running!", type="warning")
                return

            def label_count(split_root: Path, task: str) -> int:
                labels_path = split_root / task / "labels.json"
                if not labels_path.exists():
                    return 0
                try:
                    with open(labels_path) as f:
                        labels = json.load(f)
                    return len(labels) if isinstance(labels, dict) else 0
                except Exception:
                    return 0

            try:
                run_detection = mode in {"detection", "both"}
                run_recognition = mode in {"recognition", "both"}
                selected_profile = _normalize_profile_name(model_profile_state["value"])
                detection_out_dir = _model_output_dir(selected_profile, "detection")
                recognition_out_dir = _model_output_dir(selected_profile, "recognition")
                detection_out_dir.mkdir(parents=True, exist_ok=True)
                recognition_out_dir.mkdir(parents=True, exist_ok=True)

                # Auto-save any pending export assignments before training.
                pending = any(v in {"train", "val"} for v in export_manager.assignments.values())
                if pending:
                    save_result = export_manager.save_assignments(
                        include_detection=copy_detection["value"],
                        include_recognition=copy_recognition["value"],
                    )
                    refresh_kanban()
                    ui.notify(
                        f"Auto-copied {save_result.get('copied', 0)} export task(s) before training.",
                        type="positive",
                    )

                if run_detection:
                    detection_config.model_name = _prefixed_model_name(
                        "detection", detection_config.model_name, selected_profile
                    )
                    det_train_count = label_count(_split_profile_root("train", selected_profile), "detection")
                    det_val_count = label_count(_split_profile_root("val", selected_profile), "detection")
                    if det_train_count == 0 or det_val_count == 0:
                        ui.notify(
                            "Please assign and save pages to both train/val for detection before training.",
                            type="warning",
                        )
                        return

                if run_recognition:
                    recognition_config.model_name = _prefixed_model_name(
                        "recognition", recognition_config.model_name, selected_profile
                    )
                    rec_train_count = label_count(_split_profile_root("train", selected_profile), "recognition")
                    rec_val_count = label_count(_split_profile_root("val", selected_profile), "recognition")
                    if rec_train_count == 0 or rec_val_count == 0:
                        ui.notify(
                            "Please assign and save pages to both train/val for recognition before training.",
                            type="warning",
                        )
                        return
                    recognition_config.vocab = build_custom_vocab_arg(
                        recognition_config.vocab_library,
                        recognition_config.custom_characters,
                    )

                if run_detection and run_recognition:
                    status_label.set_text("⏳ Training starting (detection -> recognition)...")
                    output_area.value = "Starting training...\n\n[1/2] Detection\n"
                elif run_detection:
                    status_label.set_text("⏳ Training starting (detection)...")
                    output_area.value = "Starting training...\n\n[1/1] Detection\n"
                else:
                    status_label.set_text("⏳ Training starting (recognition)...")
                    output_area.value = "Starting training...\n\n[1/1] Recognition\n"
                output_area.update()

                training_cancelled = False

                def train_worker():
                    global training_thread, training_cancelled
                    try:
                        selected_train_root = _split_profile_root("train", selected_profile)
                        selected_val_root = _split_profile_root("val", selected_profile)

                        def ui_progress(payload: dict) -> None:
                            event = payload.get("event")
                            if event == "log":
                                message = str(payload.get("message", ""))
                                if message:
                                    queue_ui_event("append_output", text=f"{message}\n")
                                return

                            if event == "train_batch":
                                batch = payload.get("batch")
                                total = payload.get("total_batches")
                                loss = float(payload.get("loss", 0.0))
                                lr = float(payload.get("lr", 0.0))
                                queue_ui_event(
                                    "set_status",
                                    text=f"⏳ Training batch {batch}/{total} · loss={loss:.4f} · lr={lr:.6f}",
                                )
                                return

                            if event == "val_batch":
                                batch = payload.get("batch")
                                total = payload.get("total_batches")
                                loss = float(payload.get("loss", 0.0))
                                queue_ui_event(
                                    "set_status",
                                    text=f"⏳ Validation batch {batch}/{total} · loss={loss:.4f}",
                                )

                        if run_detection:
                            queue_ui_event("set_status", text="⏳ Running detection fine-tuning...")
                            detect_from_config(
                                train_path=selected_train_root / "detection",
                                val_path=selected_val_root / "detection",
                                arch=detection_config.arch,
                                epochs=detection_config.epochs,
                                batch_size=detection_config.batch_size,
                                lr=detection_config.learning_rate,
                                weight_decay=detection_config.weight_decay,
                                optimizer=detection_config.optimizer,
                                scheduler=detection_config.scheduler,
                                input_size=detection_config.input_size,
                                rotation=detection_config.rotation,
                                workers=detection_config.workers,
                                amp=detection_config.amp,
                                early_stop=detection_config.early_stop,
                                early_stop_epochs=detection_config.early_stop_epochs,
                                early_stop_delta=detection_config.early_stop_delta,
                                pretrained=detection_config.pretrained,
                                output_dir=str(detection_out_dir),
                                device=detection_config.device,
                                name=detection_config.model_name,
                                progress_hook=ui_progress,
                            )
                            queue_ui_event("append_output", text="✅ Detection fine-tuning completed.\n")

                        if run_recognition:
                            if run_detection:
                                queue_ui_event("append_output", text="\n[2/2] Recognition\n")
                            queue_ui_event("set_status", text="⏳ Running recognition fine-tuning...")
                            train_from_config(
                                train_path=selected_train_root / "recognition",
                                val_path=selected_val_root / "recognition",
                                arch=recognition_config.arch,
                                epochs=recognition_config.epochs,
                                batch_size=recognition_config.batch_size,
                                lr=recognition_config.learning_rate,
                                weight_decay=recognition_config.weight_decay,
                                optimizer=recognition_config.optimizer,
                                scheduler=recognition_config.scheduler,
                                input_size=recognition_config.input_size,
                                vocab=recognition_config.vocab,
                                workers=recognition_config.workers,
                                amp=recognition_config.amp,
                                early_stop=recognition_config.early_stop,
                                early_stop_epochs=recognition_config.early_stop_epochs,
                                early_stop_delta=recognition_config.early_stop_delta,
                                output_dir=str(recognition_out_dir),
                                device=recognition_config.device,
                                pretrained=recognition_config.pretrained,
                                name=recognition_config.model_name,
                                progress_hook=ui_progress,
                            )
                            queue_ui_event("append_output", text="✅ Recognition fine-tuning completed.\n")

                        if not training_cancelled:
                            queue_ui_event("set_status", text="✅ Training completed!")
                            queue_ui_event("append_output", text="\n✅ Training completed successfully!")
                        else:
                            queue_ui_event("set_status", text="⏹️ Training stopped by user")
                            queue_ui_event("append_output", text="\n⏹️ Training stopped by user.")

                    except Exception as e:
                        queue_ui_event("set_status", text=f"❌ Error: {e}")
                        queue_ui_event("append_output", text=f"\n\nError: {e}\n")
                    finally:
                        _release_cuda_memory()
                        training_thread = None

                training_thread = threading.Thread(target=train_worker, daemon=True)
                training_thread.start()

            except Exception as e:
                status_label.set_text(f"❌ Error preparing training: {e}")
                ui.notify(str(e), type="negative")

        def stop_training():
            """Signal training to stop."""
            global training_cancelled
            training_cancelled = True
            status_label.set_text("⏹️ Stopping training...")

        with ui.row():
            ui.button("⏹️ Stop Training", on_click=stop_training).props("color=red")
            ui.button(
                "Clear Output",
                on_click=lambda: (output_area.set_value(""), status_label.set_text("Ready")),
            ).props("color=blue")


def main():
    """Entry point for the training UI."""
    # CLI entrypoints can fail with NiceGUI auto-reload subprocess startup;
    # keep reload opt-in via env var for local debugging.
    reload_enabled = os.getenv("NICEGUI_RELOAD", "false").lower() in {"1", "true", "yes"}
    host = os.getenv("PD_OCR_TRAINER_HOST", "127.0.0.1")
    port = int(os.getenv("PD_OCR_TRAINER_PORT", "8000"))
    show_browser = os.getenv("PD_OCR_TRAINER_SHOW_BROWSER", "true").lower() in {"1", "true", "yes"}
    ui.run(create_ui, host=host, port=port, reload=reload_enabled, show=show_browser)


if __name__ in {"__main__", "__mp_main__"}:
    main()
