"""NiceGUI training interface for OCR configuration and dataset management."""

import json
import os
import shutil
import threading
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import cv2
import torch
from nicegui import ui

from train_pgdp_ocr.trainer import train_from_config

# Get the project root (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent.parent
MATCHED_OCR_DIR = PROJECT_ROOT / "matched-ocr"
ML_TRAINING_DIR = PROJECT_ROOT / "ml-training"
ML_VALIDATION_DIR = PROJECT_ROOT / "ml-validation"

# Ensure directories exist
ML_TRAINING_DIR.mkdir(exist_ok=True)
ML_VALIDATION_DIR.mkdir(exist_ok=True)


class DatasetManager:
    """Manages dataset loading and page assignments."""

    def __init__(self):
        self.loaded_files = {}  # {filename: json_data}
        self.assignments = {}  # {filename: {page_index: 'train'|'val'|None}}
        self.existing_assignments = {}  # {page_key: 'train'|'val'|None}
        self.existing_page_split = {}  # {page_key: 'train'|'val'}
        self.existing_page_keys = set()
        self.page_diff_flags = {}  # {(filename, page_index): bool}
        self.existing_detection_by_page = {}
        self.existing_recognition_by_page = defaultdict(list)
        self.total_pages = 0
        self.assigned_pages = 0
        self.refresh_existing_page_keys()

    @staticmethod
    def page_instance_key(filename: str, page_index: int) -> str:
        """Build a stable key for a matched-ocr page."""
        return f"{Path(filename).stem}_{page_index}"

    @staticmethod
    def page_key_from_image_stem(image_stem: str) -> str:
        """Extract the page key used by dataset images.

        Recognition crops use <page_key>_<x1>_<x2>_<y1>_<y2> naming,
        while detection images are stored directly as <page_key>.
        """
        parts = image_stem.split("_")
        if len(parts) >= 6 and all(part.isdigit() for part in parts[-4:]):
            return "_".join(parts[:-4])
        return image_stem

    def refresh_existing_page_keys(self):
        """Read current train/val datasets and index all existing page keys."""
        indexed_keys = set()
        detection_by_page = {}
        recognition_by_page = defaultdict(list)
        split_by_page = {}
        dataset_roots = [("train", ML_TRAINING_DIR), ("val", ML_VALIDATION_DIR)]
        tasks = ["detection", "recognition"]

        for split_name, dataset_root in dataset_roots:
            for task in tasks:
                images_dir = dataset_root / task / "images"
                if not images_dir.exists():
                    continue

                for image_path in images_dir.glob("*.*"):
                    page_key = self.page_key_from_image_stem(image_path.stem)
                    indexed_keys.add(page_key)
                    if page_key not in split_by_page:
                        split_by_page[page_key] = split_name

                labels_path = dataset_root / task / "labels.json"
                if not labels_path.exists():
                    continue

                try:
                    with open(labels_path) as f:
                        labels_data = json.load(f)
                except Exception:
                    continue

                if task == "detection":
                    for image_name, meta in labels_data.items():
                        page_key = Path(image_name).stem
                        indexed_keys.add(page_key)
                        if page_key not in split_by_page:
                            split_by_page[page_key] = split_name
                        if page_key not in detection_by_page:
                            detection_by_page[page_key] = meta
                else:
                    for crop_name, text in labels_data.items():
                        page_key = self.page_key_from_image_stem(Path(crop_name).stem)
                        indexed_keys.add(page_key)
                        if page_key not in split_by_page:
                            split_by_page[page_key] = split_name
                        recognition_by_page[page_key].append(str(text))

        self.existing_page_keys = indexed_keys
        self.existing_page_split = split_by_page
        self.existing_assignments = {page_key: split_by_page.get(page_key) for page_key in indexed_keys}
        self.existing_detection_by_page = detection_by_page
        self.existing_recognition_by_page = recognition_by_page

    def is_existing_page(self, filename: str, page_index: int) -> bool:
        """Check whether a matched-ocr page already exists in train/val datasets."""
        return self.page_instance_key(filename, page_index) in self.existing_page_keys

    @staticmethod
    def _hash_values(values: list[str]) -> str:
        return sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()

    def _extract_words(self, node: dict | list | None, words: list[dict]):
        """Recursively collect word nodes from a page tree."""
        if isinstance(node, list):
            for child in node:
                self._extract_words(child, words)
            return
        if not isinstance(node, dict):
            return

        if node.get("type") == "Word":
            words.append(node)
            return

        if "items" in node:
            self._extract_words(node.get("items"), words)

    @staticmethod
    def _bbox_to_pixels(word: dict, width: int, height: int) -> str | None:
        bbox = word.get("bounding_box", {})
        top_left = bbox.get("top_left") if isinstance(bbox, dict) else None
        bottom_right = bbox.get("bottom_right") if isinstance(bbox, dict) else None
        if not (isinstance(top_left, dict) and isinstance(bottom_right, dict)):
            return None

        try:
            x1 = int(round(float(top_left["x"]) * width))
            y1 = int(round(float(top_left["y"]) * height))
            x2 = int(round(float(bottom_right["x"]) * width))
            y2 = int(round(float(bottom_right["y"]) * height))
        except Exception:
            return None

        return f"{x1}_{x2}_{y1}_{y2}"

    def is_page_different(self, filename: str, page_index: int, page: dict) -> bool:
        """Return True when a matched page exists already but content appears different."""
        page_key = self.page_instance_key(filename, page_index)
        if page_key not in self.existing_page_keys:
            return False

        existing_detection = self.existing_detection_by_page.get(page_key, {})
        existing_texts = self.existing_recognition_by_page.get(page_key, [])

        width = page.get("width")
        height = page.get("height")
        words = []
        self._extract_words(page.get("items", []), words)

        matched_texts = []
        matched_boxes = []
        for word in words:
            token = word.get("ground_truth_text") or word.get("text") or ""
            token = str(token).strip()
            if token:
                matched_texts.append(token)
            if isinstance(width, int) and isinstance(height, int):
                box_key = self._bbox_to_pixels(word, width, height)
                if box_key:
                    matched_boxes.append(box_key)

        existing_dims = existing_detection.get("img_dimensions") if isinstance(existing_detection, dict) else None
        if (
            isinstance(existing_dims, list)
            and len(existing_dims) == 2
            and isinstance(width, int)
            and isinstance(height, int)
        ):
            if [width, height] != existing_dims:
                return True

        existing_polygons = existing_detection.get("polygons") if isinstance(existing_detection, dict) else None
        if isinstance(existing_polygons, list):
            existing_polygon_boxes = []
            for polygon in existing_polygons:
                if not isinstance(polygon, list) or not polygon:
                    continue
                try:
                    xs = [int(point[0]) for point in polygon if isinstance(point, list) and len(point) >= 2]
                    ys = [int(point[1]) for point in polygon if isinstance(point, list) and len(point) >= 2]
                except Exception:
                    continue
                if xs and ys:
                    existing_polygon_boxes.append(f"{min(xs)}_{max(xs)}_{min(ys)}_{max(ys)}")

            if matched_boxes and self._hash_values(matched_boxes) != self._hash_values(existing_polygon_boxes):
                return True

        if existing_texts:
            if self._hash_values(matched_texts) != self._hash_values(existing_texts):
                return True

        return False

    def load_json_file(self, filepath: Path) -> dict:
        """Load a JSON file from matched-ocr."""
        try:
            with open(filepath) as f:
                data = json.load(f)
            filename = filepath.name
            self.loaded_files[filename] = data

            # Count pages
            pages = data.get("pages", [])
            self.page_diff_flags = {k: v for k, v in self.page_diff_flags.items() if k[0] != filename}

            assignable_page_indices = self.get_assignable_page_indices(filename, pages)
            self.assignments[filename] = dict.fromkeys(assignable_page_indices)

            return data
        except Exception as e:
            raise ValueError(f"Failed to load {filepath.name}: {e}") from e

    def get_assignable_page_indices(self, filename: str, pages: list[dict]) -> list[int]:
        """Return all page indices; existing overlaps are still shown for reassignment."""
        assignable = []
        for idx, page in enumerate(pages):
            assignable.append(idx)
            if not self.is_existing_page(filename, idx):
                self.page_diff_flags[(filename, idx)] = False
                continue

            is_different = self.is_page_different(filename, idx, page)
            self.page_diff_flags[(filename, idx)] = is_different

        return assignable

    def load_all_matched_files(self):
        """Auto-load all matched-ocr JSON files so assignment can start without manual add."""
        if not MATCHED_OCR_DIR.exists():
            return

        for filepath in sorted(MATCHED_OCR_DIR.glob("*.json")):
            if filepath.name in self.loaded_files:
                continue
            self.load_json_file(filepath)

    def is_flagged_different(self, filename: str, page_index: int) -> bool:
        """Whether the page is an overlap that differs from existing datasets."""
        return self.page_diff_flags.get((filename, page_index), False)

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
    def _write_json_map(path: Path, data: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def project_from_page_key(page_key: str) -> str:
        parts = page_key.split("_")
        if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
            return "_".join(parts[:-2])
        if "_" in page_key:
            return page_key.rsplit("_", 1)[0]
        return page_key

    def _page_words_with_boxes(self, page: dict) -> list[dict]:
        words = []
        self._extract_words(page.get("items", []), words)
        width = page.get("width")
        height = page.get("height")
        if not isinstance(width, int) or not isinstance(height, int):
            return []

        out = []
        for word in words:
            text = str(word.get("ground_truth_text") or word.get("text") or "").strip()
            if not text:
                continue
            box = self._bbox_to_pixels(word, width, height)
            if not box:
                continue
            x1s, x2s, y1s, y2s = box.split("_")
            x1, x2, y1, y2 = int(x1s), int(x2s), int(y1s), int(y2s)
            if x2 <= x1 or y2 <= y1:
                continue
            out.append({"text": text, "box": [x1, x2, y1, y2]})
        return out

    def _source_image_path(self, filename: str, data: dict) -> Path:
        source_path = data.get("source_path")
        candidates = []
        if isinstance(source_path, str) and source_path:
            source_obj = Path(source_path)
            candidates.append(PROJECT_ROOT / source_obj)
            candidates.append(PROJECT_ROOT / source_obj.name)
            candidates.append(MATCHED_OCR_DIR / source_obj.name)
        candidates.append(MATCHED_OCR_DIR / f"{Path(filename).stem}.png")

        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Source image not found for {filename}")

    @staticmethod
    def _remove_page_prefixed_entries(data: dict, page_key: str):
        prefix = f"{page_key}_"
        to_delete = [key for key in data if key.startswith(prefix)]
        for key in to_delete:
            data.pop(key, None)

    @staticmethod
    def _remove_page_prefixed_images(images_dir: Path, page_key: str):
        if not images_dir.exists():
            return
        pattern = f"{page_key}_*.png"
        for image_path in images_dir.glob(pattern):
            image_path.unlink(missing_ok=True)

    def _existing_pages_for_split(self, split: str) -> dict:
        split_root = ML_TRAINING_DIR if split == "train" else ML_VALIDATION_DIR
        detection_labels = self._load_json_map(split_root / "detection" / "labels.json")
        recognition_labels = self._load_json_map(split_root / "recognition" / "labels.json")

        page_keys = set()
        for image_name in detection_labels:
            page_keys.add(Path(image_name).stem)
        for crop_name in recognition_labels:
            page_keys.add(self.page_key_from_image_stem(Path(crop_name).stem))

        projects = defaultdict(list)
        for page_key in sorted(page_keys):
            project = self.project_from_page_key(page_key)
            projects[project].append(f"{page_key} [existing]")

        mirrored = {project: list(labels) for project, labels in projects.items()}
        return {"detection": mirrored, "recognition": {k: list(v) for k, v in mirrored.items()}}

    @staticmethod
    def _merge_projects(primary: dict, secondary: dict) -> dict:
        merged = {}
        all_projects = set(primary.keys()) | set(secondary.keys())
        for project in sorted(all_projects):
            values = list(primary.get(project, [])) + list(secondary.get(project, []))
            merged[project] = sorted(set(values))
        return merged

    def get_combined_split_task_pages(self) -> dict:
        return self.get_split_task_pages()

    def save_assignments(self) -> dict:
        """Persist assignments into combined labels and prune saved pages from matched-ocr."""
        pending = []
        by_file = defaultdict(list)

        for filename, file_assignments in self.assignments.items():
            data = self.loaded_files.get(filename)
            if not data:
                continue
            pages = data.get("pages", [])
            for page_index, split in file_assignments.items():
                if split not in {"train", "val"}:
                    continue
                if page_index >= len(pages):
                    continue
                pending.append((filename, page_index, split))
                by_file[filename].append(page_index)

        split_data = {
            "train": {
                "root": ML_TRAINING_DIR,
                "detection_labels": self._load_json_map(ML_TRAINING_DIR / "detection" / "labels.json"),
                "recognition_labels": self._load_json_map(ML_TRAINING_DIR / "recognition" / "labels.json"),
            },
            "val": {
                "root": ML_VALIDATION_DIR,
                "detection_labels": self._load_json_map(ML_VALIDATION_DIR / "detection" / "labels.json"),
                "recognition_labels": self._load_json_map(ML_VALIDATION_DIR / "recognition" / "labels.json"),
            },
        }

        for split_cfg in split_data.values():
            (split_cfg["root"] / "detection" / "images").mkdir(parents=True, exist_ok=True)
            (split_cfg["root"] / "recognition" / "images").mkdir(parents=True, exist_ok=True)

        existing_pending = []
        for page_key, target_split in self.existing_assignments.items():
            if target_split not in {"train", "val", None}:
                continue
            source_split = self.existing_page_split.get(page_key)
            if target_split != source_split:
                existing_pending.append((page_key, source_split, target_split))

        if not pending and not existing_pending:
            return {"saved_pages": 0, "removed_files": 0}

        for page_key, source_split, target_split in existing_pending:
            det_name = f"{page_key}.png"
            source_cfg = split_data.get(source_split) if source_split in {"train", "val"} else None
            target_cfg = split_data.get(target_split) if target_split in {"train", "val"} else None

            moved_detection = None
            moved_recognition = {}

            if source_cfg:
                moved_detection = source_cfg["detection_labels"].pop(det_name, None)
                src_det_path = source_cfg["root"] / "detection" / "images" / det_name
                src_crop_dir = source_cfg["root"] / "recognition" / "images"

                for key in list(source_cfg["recognition_labels"].keys()):
                    if key.startswith(f"{page_key}_"):
                        moved_recognition[key] = source_cfg["recognition_labels"].pop(key)

                if target_cfg and src_det_path.exists():
                    shutil.copy2(src_det_path, target_cfg["root"] / "detection" / "images" / det_name)
                src_det_path.unlink(missing_ok=True)

                for crop_name in moved_recognition:
                    src_crop = src_crop_dir / crop_name
                    if target_cfg and src_crop.exists():
                        shutil.copy2(src_crop, target_cfg["root"] / "recognition" / "images" / crop_name)
                    src_crop.unlink(missing_ok=True)
            else:
                for cfg in split_data.values():
                    cfg["detection_labels"].pop(det_name, None)
                    self._remove_page_prefixed_entries(cfg["recognition_labels"], page_key)
                    (cfg["root"] / "detection" / "images" / det_name).unlink(missing_ok=True)
                    self._remove_page_prefixed_images(cfg["root"] / "recognition" / "images", page_key)

            if target_cfg:
                if moved_detection is not None:
                    target_cfg["detection_labels"][det_name] = moved_detection
                for crop_name, text in moved_recognition.items():
                    target_cfg["recognition_labels"][crop_name] = text

        for filename, page_index, split in pending:
            data = self.loaded_files[filename]
            page = data["pages"][page_index]
            page_key = self.page_instance_key(filename, page_index)
            source_image = self._source_image_path(filename, data)

            # Replace semantics by page key: remove stale entries from both train/val before adding.
            for existing_split_cfg in split_data.values():
                det_name = f"{page_key}.png"
                existing_split_cfg["detection_labels"].pop(det_name, None)
                self._remove_page_prefixed_entries(existing_split_cfg["recognition_labels"], page_key)

                det_existing_path = existing_split_cfg["root"] / "detection" / "images" / det_name
                det_existing_path.unlink(missing_ok=True)
                self._remove_page_prefixed_images(existing_split_cfg["root"] / "recognition" / "images", page_key)

            split_cfg = split_data[split]
            det_images_dir = split_cfg["root"] / "detection" / "images"
            rec_images_dir = split_cfg["root"] / "recognition" / "images"

            det_filename = f"{page_key}.png"
            det_target = det_images_dir / det_filename
            shutil.copy2(source_image, det_target)

            words = self._page_words_with_boxes(page)
            polygons = []
            src_img = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
            if src_img is None:
                raise ValueError(f"Failed to read source image: {source_image}")
            height, width = src_img.shape[:2]

            for word in words:
                x1, x2, y1, y2 = word["box"]
                x1 = max(0, min(x1, width - 1))
                x2 = max(1, min(x2, width))
                y1 = max(0, min(y1, height - 1))
                y2 = max(1, min(y2, height))
                if x2 <= x1 or y2 <= y1:
                    continue

                polygons.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

                crop_name = f"{page_key}_{x1}_{x2}_{y1}_{y2}.png"
                crop_target = rec_images_dir / crop_name
                crop_img = src_img[y1:y2, x1:x2]
                cv2.imwrite(str(crop_target), crop_img)
                split_cfg["recognition_labels"][crop_name] = word["text"]

            split_cfg["detection_labels"][det_filename] = {
                "img_dimensions": [page.get("width", 0), page.get("height", 0)],
                "img_hash": self._file_sha256(det_target),
                "polygons": polygons,
            }

        for split_cfg in split_data.values():
            self._write_json_map(split_cfg["root"] / "detection" / "labels.json", split_cfg["detection_labels"])
            self._write_json_map(split_cfg["root"] / "recognition" / "labels.json", split_cfg["recognition_labels"])

        removed_files = 0
        for filename, page_indices in by_file.items():
            filepath = MATCHED_OCR_DIR / filename
            data = self.loaded_files.get(filename)
            if not data:
                continue

            for page_index in sorted(page_indices, reverse=True):
                if page_index < len(data.get("pages", [])):
                    data["pages"].pop(page_index)

            for new_idx, page in enumerate(data.get("pages", [])):
                page["page_index"] = new_idx

            if data.get("pages"):
                with open(filepath, "w") as f:
                    json.dump(data, f, indent=2)
            else:
                filepath.unlink(missing_ok=True)
                source_png = MATCHED_OCR_DIR / f"{Path(filename).stem}.png"
                source_png.unlink(missing_ok=True)
                removed_files += 1

        self.loaded_files.clear()
        self.assignments.clear()
        self.page_diff_flags.clear()
        self.refresh_existing_page_keys()

        return {"saved_pages": len(pending) + len(existing_pending), "removed_files": removed_files}

    def get_assignable_page_count_for_file(self, filename: str) -> int:
        """Get assignable page count for a matched-ocr file without loading it in UI state."""
        filepath = MATCHED_OCR_DIR / filename
        if not filepath.exists():
            return 0

        try:
            with open(filepath) as f:
                data = json.load(f)
            pages = data.get("pages", [])
            return len(self.get_assignable_page_indices(filename, pages))
        except Exception:
            return 0

    def get_available_files(self) -> list[str]:
        """Get list of available JSON files in matched-ocr."""
        if not MATCHED_OCR_DIR.exists():
            return []

        visible_files = []
        for candidate in sorted(MATCHED_OCR_DIR.glob("*.json")):
            if self.get_assignable_page_count_for_file(candidate.name) > 0:
                visible_files.append(candidate.name)
        return visible_files

    def get_page_count(self, filename: str) -> int:
        """Get number of pages in a file."""
        if filename in self.assignments:
            return len(self.assignments[filename])
        return 0

    def assign_page(self, filename: str, page_index: int, target: str):
        """Assign a page to train, val, or None."""
        if filename in self.assignments:
            self.assignments[filename][page_index] = target if target in {"train", "val"} else None

    def update_stats(self):
        """Update total and assigned page counts."""
        matched_total = sum(len(file_assigns) for file_assigns in self.assignments.values())
        matched_assigned = sum(
            1
            for file_assigns in self.assignments.values()
            for assignment in file_assigns.values()
            if assignment is not None
        )
        existing_total = len(self.existing_assignments)
        existing_assigned = sum(1 for split in self.existing_assignments.values() if split in {"train", "val"})

        self.total_pages = matched_total + existing_total
        self.assigned_pages = matched_assigned + existing_assigned

    @staticmethod
    def project_from_filename(filename: str) -> str:
        """Extract a project key from a matched-ocr filename."""
        stem = Path(filename).stem
        if "_" not in stem:
            return stem
        return stem.rsplit("_", 1)[0]

    def get_split_task_pages(self) -> dict:
        """Group matched and existing pages by split and project, mirrored for both tasks."""
        split_projects = {
            "unassigned": defaultdict(list),
            "train": defaultdict(list),
            "val": defaultdict(list),
        }

        for filename, file_assignments in self.assignments.items():
            if filename not in self.loaded_files:
                continue

            project_key = self.project_from_filename(filename)
            for page_index, assignment in file_assignments.items():
                split_key = assignment if assignment in {"train", "val"} else "unassigned"
                diff_suffix = " [different]" if self.is_flagged_different(filename, page_index) else ""
                page_label = f"{Path(filename).stem} · page {page_index + 1}{diff_suffix}"
                split_projects[split_key][project_key].append(page_label)

        matched_page_keys = {
            self.page_instance_key(filename, page_index)
            for filename, file_assignments in self.assignments.items()
            for page_index in file_assignments
        }

        for page_key, assignment in self.existing_assignments.items():
            if page_key in matched_page_keys:
                continue
            split_key = assignment if assignment in {"train", "val"} else "unassigned"
            project_key = self.project_from_page_key(page_key)
            split_projects[split_key][project_key].append(f"{page_key} [existing]")

        split_views = {}
        for split_key, projects in split_projects.items():
            ordered_projects = {
                project: sorted(project_pages)
                for project, project_pages in sorted(projects.items(), key=lambda item: item[0])
            }
            split_views[split_key] = {
                "detection": {project: list(pages) for project, pages in ordered_projects.items()},
                "recognition": {project: list(pages) for project, pages in ordered_projects.items()},
            }

        return split_views

    def export_datasets(self) -> dict:
        """Export training and validation datasets."""
        train_pages = []
        val_pages = []

        for filename, assignments in self.assignments.items():
            if filename not in self.loaded_files:
                continue

            data = self.loaded_files[filename]
            pages = data.get("pages", [])

            for page_index, assignment in assignments.items():
                if page_index < len(pages):
                    page = pages[page_index]
                    page_with_meta = {
                        "source_file": filename,
                        "source_path": data.get("source_path"),
                        **page,
                    }

                    if assignment == "train":
                        train_pages.append(page_with_meta)
                    elif assignment == "val":
                        val_pages.append(page_with_meta)

        return {"train": train_pages, "val": val_pages}


class TrainingConfig:
    """Manages training configuration."""

    def __init__(self):
        # Detection
        self.det_arch = "db_resnet50"
        self.det_epochs = 10
        self.det_batch_size = 16
        self.det_learning_rate = 0.001
        self.det_weight_decay = 0.0
        self.det_optimizer = "adam"
        self.det_scheduler = "cosine"
        self.det_workers = 4
        self.det_amp = False
        self.det_early_stop = False
        self.det_early_stop_epochs = 5

        # Recognition
        self.arch = "crnn_vgg16_bn"
        self.epochs = 10
        self.batch_size = 64
        self.learning_rate = 0.001
        self.weight_decay = 0.0
        self.optimizer = "adam"
        self.scheduler = "cosine"
        self.input_size = 32
        self.amp = False
        self.early_stop = False
        self.early_stop_epochs = 5
        self.vocab = "french"
        self.workers = 4
        self.device = 0 if torch.cuda.is_available() else -1


# Global state
dataset_manager = DatasetManager()
training_config = TrainingConfig()
training_thread: threading.Thread | None = None
training_cancelled = False


def create_ui():
    """Build the NiceGUI interface."""

    with ui.header().classes("w-full bg-blue-500 text-white"):
        ui.label("OCR Training Suite").classes("text-2xl font-bold")

    with ui.row().classes("w-full"):
        # ==================== DATASET SECTION ====================
        with ui.card().classes("flex-1"):
            ui.label("📂 Dataset Management").classes("text-lg font-bold")

            # Available files browser
            with ui.card().classes("w-full"):
                ui.label("Available JSON Files (non-overlapping)").classes("font-semibold")

                async def load_file(filename: str):
                    try:
                        filepath = MATCHED_OCR_DIR / filename
                        dataset_manager.load_json_file(filepath)
                        page_count = dataset_manager.get_page_count(filename)
                        diff_count = sum(
                            1
                            for idx in dataset_manager.assignments.get(filename, {})
                            if dataset_manager.is_flagged_different(filename, idx)
                        )

                        if page_count == 0:
                            files_label.set_text(
                                f"↷ Skipped: {filename} (all pages already exist in ml-training/ml-validation)"
                            )
                        elif diff_count:
                            files_label.set_text(
                                f"✓ Loaded: {filename} ({page_count} assignable pages, {diff_count} different/newer)"
                            )
                        else:
                            files_label.set_text(f"✓ Loaded: {filename} ({page_count} assignable pages)")
                        refresh_page_grid()
                        update_stats()
                    except Exception as e:
                        files_label.set_text(f"✗ Error: {e}")

                files_label = ui.label("Select file to load...")
                file_buttons_container = ui.row()

                def refresh_available_file_buttons():
                    file_buttons_container.clear()
                    available_files = dataset_manager.get_available_files()
                    with file_buttons_container:
                        if not available_files:
                            ui.label("No assignable matched-ocr files").classes("text-xs text-gray-500")
                            return
                        for fname in available_files[:10]:  # Show first 10
                            ui.button(
                                fname[:20] + "..." if len(fname) > 20 else fname,
                                on_click=lambda f=fname: load_file(f),
                            ).props("size=sm").tooltip(fname)

            # Page assignment grid
            ui.separator().classes("mt-4")
            ui.label("Assign Pages to Sets").classes("font-semibold mt-2")

            page_container = ui.column().classes("w-full border-l-4 border-blue-300 pl-4")
            drag_state = {"unassigned": [], "train": [], "val": []}
            container_to_split = {}

            ui.separator().classes("mt-4")
            split_view_container = ui.column().classes("w-full mt-2")

            def render_split_view(split_title: str, projects_by_task: dict):
                with split_view_container:
                    with ui.expansion(split_title, value=False).classes("w-full"):
                        with ui.row().classes("w-full gap-4"):
                            for task_name in ("detection", "recognition"):
                                with ui.card().classes("flex-1"):
                                    task_projects = projects_by_task.get(task_name, {})
                                    ui.label(task_name).classes("font-medium text-sm")

                                    if not task_projects:
                                        ui.label("No pages").classes("text-xs text-gray-500")
                                        continue

                                    for project_key, page_labels in task_projects.items():
                                        with ui.expansion(f"{project_key} ({len(page_labels)} pages)").classes("w-full"):
                                            for page_label in page_labels:
                                                ui.label(page_label).classes("text-xs text-gray-600")

            def resolve_split_from_event_value(raw_value):
                if raw_value is None:
                    return None

                if isinstance(raw_value, dict):
                    for key in ("id", "value", "name"):
                        split_key = resolve_split_from_event_value(raw_value.get(key))
                        if split_key:
                            return split_key
                    return None

                as_str = str(raw_value).strip().lstrip("#")
                if as_str in {"unassigned", "train", "val"}:
                    return as_str

                return container_to_split.get(as_str)

            def apply_drag_state_to_assignments():
                for split_key, page_ids in drag_state.items():
                    target = split_key if split_key in {"train", "val"} else None
                    for page_id in page_ids:
                        filename, page_index = page_id.rsplit("::", 1)
                        dataset_manager.assign_page(filename, int(page_index), target)

            def on_sort_end(e, source_split: str):
                args = getattr(e, "args", {}) or {}
                old_index = getattr(e, "old_index", None)
                new_index = getattr(e, "new_index", None)

                if old_index is None:
                    old_index = args.get("oldIndex")
                if new_index is None:
                    new_index = args.get("newIndex")

                if old_index is None:
                    return

                source = resolve_split_from_event_value(args.get("from")) or source_split
                target = resolve_split_from_event_value(args.get("to")) or source

                if source not in drag_state or target not in drag_state:
                    return

                source_items = drag_state[source]
                if old_index < 0 or old_index >= len(source_items):
                    return

                moved_page = source_items.pop(old_index)

                if new_index is None:
                    new_index = len(drag_state[target])

                target_items = drag_state[target]
                insert_at = max(0, min(new_index, len(target_items)))
                target_items.insert(insert_at, moved_page)

                apply_drag_state_to_assignments()
                refresh_page_grid()
                update_stats()

            def refresh_page_grid():
                page_container.clear()
                split_view_container.clear()
                container_to_split.clear()
                drag_state["unassigned"] = []
                drag_state["train"] = []
                drag_state["val"] = []

                split_items = {"unassigned": [], "train": [], "val": []}

                for filename, assignments in dataset_manager.assignments.items():
                    if filename not in dataset_manager.loaded_files:
                        continue
                    if not assignments:
                        continue

                    for page_idx, current in sorted(assignments.items(), key=lambda item: item[0]):
                        split_key = current if current in {"train", "val"} else "unassigned"
                        is_different = dataset_manager.is_flagged_different(filename, page_idx)
                        label_suffix = " [different/newer]" if is_different else ""
                        page_label = f"{Path(filename).stem} · page {page_idx + 1}{label_suffix}"
                        page_id = f"{filename}::{page_idx}"

                        split_items[split_key].append(
                            {
                                "id": page_id,
                                "label": page_label,
                                "different": is_different,
                            }
                        )

                with page_container:
                    ui.label("Drag pages between Unassigned, Training, and Validation").classes(
                        "text-sm text-gray-600"
                    )

                    with ui.row().classes("w-full gap-4 items-start"):
                        for split_key, title in (
                            ("unassigned", "Unassigned"),
                            ("train", "Training"),
                            ("val", "Validation"),
                        ):
                            items = sorted(split_items[split_key], key=lambda item: item["label"])
                            drag_state[split_key] = [item["id"] for item in items]

                            with ui.card().classes("flex-1"):
                                ui.label(f"{title} ({len(items)})").classes("font-semibold text-sm")

                                drop_column = ui.column().classes("w-full min-h-40 gap-2 bg-gray-50 p-2 rounded")
                                container_to_split[str(drop_column.id)] = split_key

                                with drop_column:
                                    for item in items:
                                        with ui.card().classes("w-full p-2 cursor-grab active:cursor-grabbing"):
                                            ui.label(item["label"]).classes("text-xs")
                                            if item["different"]:
                                                ui.label("different/newer").classes("text-xs text-amber-700")

                                drop_column.make_sortable(
                                    group="page-assignment",
                                    on_end=lambda e, src=split_key: on_sort_end(e, src),
                                )

                split_task_pages = dataset_manager.get_combined_split_task_pages()
                render_split_view("ml-training", split_task_pages["train"])
                render_split_view("ml-validation", split_task_pages["val"])
                render_split_view("unassigned", split_task_pages["unassigned"])

            # Stats
            stats_label = ui.label("No data loaded").classes("text-sm text-gray-600 mt-4")

            def update_stats():
                dataset_manager.update_stats()
                stats_label.set_text(
                    f"📊 Total pages: {dataset_manager.total_pages} | Assigned: {dataset_manager.assigned_pages}"
                )

            def save_assignments():
                try:
                    result = dataset_manager.save_assignments()
                    saved_pages = result.get("saved_pages", 0)
                    removed_files = result.get("removed_files", 0)
                    if saved_pages == 0:
                        ui.notify("No assigned pages to save.", type="warning")
                        return

                    files_label.set_text(
                        f"💾 Saved {saved_pages} pages to combined labels; removed {removed_files} fully-consumed matched files"
                    )
                    refresh_available_file_buttons()
                    refresh_page_grid()
                    update_stats()
                    ui.notify("Assignments saved.", type="positive")
                except Exception as e:
                    ui.notify(str(e), type="negative")
                    files_label.set_text(f"✗ Save failed: {e}")

            with ui.row().classes("mt-3"):
                ui.button("💾 Save Assignments", on_click=save_assignments).props("color=primary")

            refresh_available_file_buttons()
            refresh_page_grid()
            update_stats()

        # ==================== TRAINING CONFIG SECTION ====================
        with ui.card().classes("flex-1"):
            ui.label("⚙️ Training Configuration").classes("text-lg font-bold")

            with ui.expansion("🔍 Detection", value=True).classes("w-full"):
                ui.select(
                    label="Architecture",
                    options=[
                        "db_resnet50",
                        "db_resnet34",
                        "db_mobilenet_v3_large",
                        "linknet_resnet18",
                        "linknet_resnet34",
                    ],
                    value=training_config.det_arch,
                    on_change=lambda v: setattr(training_config, "det_arch", v.value),
                ).classes("w-full")

                ui.number(
                    label="Epochs",
                    value=training_config.det_epochs,
                    min=1,
                    max=100,
                    on_change=lambda v: setattr(training_config, "det_epochs", int(v.value)),
                ).classes("w-full")

                ui.number(
                    label="Batch Size",
                    value=training_config.det_batch_size,
                    min=1,
                    max=512,
                    on_change=lambda v: setattr(training_config, "det_batch_size", int(v.value)),
                ).classes("w-full")

                ui.select(
                    label="Optimizer",
                    options=["adam", "adamw"],
                    value=training_config.det_optimizer,
                    on_change=lambda v: setattr(training_config, "det_optimizer", v.value),
                ).classes("w-full")

                ui.number(
                    label="Learning Rate",
                    value=training_config.det_learning_rate,
                    min=0.00001,
                    max=0.1,
                    step=0.0001,
                    format="%.5f",
                    on_change=lambda v: setattr(training_config, "det_learning_rate", v.value),
                ).classes("w-full")

                ui.number(
                    label="Weight Decay",
                    value=training_config.det_weight_decay,
                    min=0,
                    max=0.1,
                    step=0.001,
                    format="%.4f",
                    on_change=lambda v: setattr(training_config, "det_weight_decay", v.value),
                ).classes("w-full")

                ui.select(
                    label="Scheduler",
                    options=["cosine", "onecycle", "poly"],
                    value=training_config.det_scheduler,
                    on_change=lambda v: setattr(training_config, "det_scheduler", v.value),
                ).classes("w-full")

                ui.number(
                    label="Workers",
                    value=training_config.det_workers,
                    min=0,
                    max=16,
                    on_change=lambda v: setattr(training_config, "det_workers", int(v.value)),
                ).classes("w-full")

                ui.checkbox(
                    text="Mixed Precision (AMP)",
                    value=training_config.det_amp,
                    on_change=lambda v: setattr(training_config, "det_amp", v.value),
                ).classes("w-full")

                ui.checkbox(
                    text="Early Stopping",
                    value=training_config.det_early_stop,
                    on_change=lambda v: setattr(training_config, "det_early_stop", v.value),
                ).classes("w-full")

                ui.number(
                    label="Early Stop Patience",
                    value=training_config.det_early_stop_epochs,
                    min=1,
                    max=20,
                    on_change=lambda v: setattr(training_config, "det_early_stop_epochs", int(v.value)),
                ).classes("w-full")

            with ui.expansion("🔤 Recognition", value=True).classes("w-full"):
                ui.select(
                    label="Architecture",
                    options=[
                        "crnn_vgg16_bn",
                        "crnn_mobilenet_v3_small",
                        "crnn_mobilenet_v3_large",
                    ],
                    value=training_config.arch,
                    on_change=lambda v: setattr(training_config, "arch", v.value),
                ).classes("w-full")

                ui.number(
                    label="Epochs",
                    value=training_config.epochs,
                    min=1,
                    max=100,
                    on_change=lambda v: setattr(training_config, "epochs", int(v.value)),
                ).classes("w-full")

                ui.number(
                    label="Batch Size",
                    value=training_config.batch_size,
                    min=1,
                    max=512,
                    on_change=lambda v: setattr(training_config, "batch_size", int(v.value)),
                ).classes("w-full")

                ui.select(
                    label="Vocabulary",
                    options=["french", "english", "digits"],
                    value=training_config.vocab,
                    on_change=lambda v: setattr(training_config, "vocab", v.value),
                ).classes("w-full")

                ui.select(
                    label="Optimizer",
                    options=["adam", "adamw"],
                    value=training_config.optimizer,
                    on_change=lambda v: setattr(training_config, "optimizer", v.value),
                ).classes("w-full")

                ui.number(
                    label="Learning Rate",
                    value=training_config.learning_rate,
                    min=0.00001,
                    max=0.1,
                    step=0.0001,
                    format="%.5f",
                    on_change=lambda v: setattr(training_config, "learning_rate", v.value),
                ).classes("w-full")

                ui.number(
                    label="Weight Decay",
                    value=training_config.weight_decay,
                    min=0,
                    max=0.1,
                    step=0.001,
                    format="%.4f",
                    on_change=lambda v: setattr(training_config, "weight_decay", v.value),
                ).classes("w-full")

                ui.select(
                    label="Scheduler",
                    options=["cosine", "onecycle", "poly"],
                    value=training_config.scheduler,
                    on_change=lambda v: setattr(training_config, "scheduler", v.value),
                ).classes("w-full")

                ui.number(
                    label="Input Height",
                    value=training_config.input_size,
                    min=16,
                    max=64,
                    step=4,
                    on_change=lambda v: setattr(training_config, "input_size", int(v.value)),
                ).classes("w-full")

                ui.number(
                    label="Workers",
                    value=training_config.workers,
                    min=0,
                    max=16,
                    on_change=lambda v: setattr(training_config, "workers", int(v.value)),
                ).classes("w-full")

                ui.checkbox(
                    text="Mixed Precision (AMP)",
                    value=training_config.amp,
                    on_change=lambda v: setattr(training_config, "amp", v.value),
                ).classes("w-full")

                ui.checkbox(
                    text="Early Stopping",
                    value=training_config.early_stop,
                    on_change=lambda v: setattr(training_config, "early_stop", v.value),
                ).classes("w-full")

                ui.number(
                    label="Early Stop Patience",
                    value=training_config.early_stop_epochs,
                    min=1,
                    max=20,
                    on_change=lambda v: setattr(training_config, "early_stop_epochs", int(v.value)),
                ).classes("w-full")

    # ==================== TRAINING CONTROL SECTION ====================
    with ui.card().classes("w-full"):
        ui.label("🚀 Training Control").classes("text-lg font-bold")

        status_label = ui.label("Ready").classes("text-sm text-gray-600")
        output_area = (
            ui.textarea(
                value="Training output will appear here...",
            )
            .props("readonly")
            .classes("w-full h-64")
        )

        def run_training():
            """Run training in background thread."""
            global training_thread, training_cancelled

            if training_thread and training_thread.is_alive():
                ui.notify("Training is already running!", type="warning")
                return

            def recognition_label_count(split_root: Path) -> int:
                labels_path = split_root / "recognition" / "labels.json"
                if not labels_path.exists():
                    return 0
                try:
                    with open(labels_path) as f:
                        labels = json.load(f)
                    return len(labels) if isinstance(labels, dict) else 0
                except Exception:
                    return 0

            try:
                # Persist any pending selections before training.
                pending_assignments = any(
                    assignment in {"train", "val"}
                    for file_assignments in dataset_manager.assignments.values()
                    for assignment in file_assignments.values()
                )
                if pending_assignments:
                    save_result = dataset_manager.save_assignments()
                    refresh_available_file_buttons()
                    refresh_page_grid()
                    update_stats()
                    ui.notify(
                        f"Auto-saved {save_result.get('saved_pages', 0)} assigned pages before training.",
                        type="positive",
                    )

                train_count = recognition_label_count(ML_TRAINING_DIR)
                val_count = recognition_label_count(ML_VALIDATION_DIR)
                if train_count == 0 or val_count == 0:
                    ui.notify(
                        "Please assign and save pages to both training and validation sets!",
                        type="warning",
                    )
                    return

                status_label.set_text("⏳ Training starting...")
                output_area.value = "Starting training...\n"
                training_cancelled = False

                def train_worker():
                    global training_thread, training_cancelled
                    try:
                        # Train against the persisted combined recognition datasets.
                        train_from_config(
                            train_path=ML_TRAINING_DIR / "recognition",
                            val_path=ML_VALIDATION_DIR / "recognition",
                            arch=training_config.arch,
                            epochs=training_config.epochs,
                            batch_size=training_config.batch_size,
                            lr=training_config.learning_rate,
                            weight_decay=training_config.weight_decay,
                            optimizer=training_config.optimizer,
                            scheduler=training_config.scheduler,
                            input_size=training_config.input_size,
                            vocab=training_config.vocab,
                            workers=training_config.workers,
                            amp=training_config.amp,
                            early_stop=training_config.early_stop,
                            early_stop_epochs=training_config.early_stop_epochs,
                            early_stop_delta=0.01,
                            output_dir=str(PROJECT_ROOT),
                            device=training_config.device,
                        )

                        if not training_cancelled:
                            status_label.set_text("✅ Training completed!")
                            ui.notify("Training finished successfully!", type="positive")
                            output_area.value += "\n\n✅ Training completed successfully!"
                        else:
                            status_label.set_text("⏹️ Training stopped by user")

                    except Exception as e:
                        status_label.set_text(f"❌ Error: {e}")
                        output_area.value += f"\n\nError: {e}\n"
                        ui.notify(str(e), type="negative")

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
            ui.button("▶️ Start Training", on_click=run_training).props("color=green")
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
    ui.run(create_ui, host="127.0.0.1", port=8000, reload=reload_enabled)


if __name__ in {"__main__", "__mp_main__"}:
    main()
