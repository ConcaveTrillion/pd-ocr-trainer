"""NiceGUI training interface for OCR configuration and dataset management."""

import contextlib
import gc
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

from nicegui import ui
from pdomain_book_tools.ocr.doctr_support import (
    DEFAULT_VOCAB_EXTRA_CHARS as _DEFAULT_VOCAB_EXTRA_CHARS,
)
from pdomain_book_tools.ocr.doctr_support import (
    DEFAULT_VOCAB_LIBRARY as _DEFAULT_VOCAB_LIBRARY,
)

from .dataset_store import (
    APP_DATA_ROOT,
    BASE_OCR_PROFILE,
    ML_TRAINING_DIR,
    MODEL_NAME_PREFIX,
    TRAINER_SETTINGS_PATH,
    ExportManager,
    get_available_model_profiles,
    migrate_legacy_dataset_layout,
    model_output_dir,
    normalize_profile_name,
    split_profile_root,
)
from .dataset_ui import build_dataset_section

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler(stream=__import__("sys").stdout)
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)


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
    import torch

    if not torch.cuda.is_available():
        return
    with contextlib.suppress(Exception):
        torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()


_VOCABS_CACHE_PATH = APP_DATA_ROOT / "_vocabs_cache.json"
_vocabs_cache: "dict | None" = None


def _get_vocabs() -> dict:
    """Return the doctr VOCABS dict, using a JSON cache to avoid a slow import on every startup."""
    global _vocabs_cache
    if _vocabs_cache is not None:
        return _vocabs_cache
    if _VOCABS_CACHE_PATH.exists():
        try:
            with open(_VOCABS_CACHE_PATH, encoding="utf-8") as f:
                _vocabs_cache = json.load(f)
            return _vocabs_cache
        except Exception:
            pass
    # Cache miss — import doctr.datasets and write cache for next startup.
    from doctr.datasets import VOCABS as _DOCTR_VOCABS

    _vocabs_cache = dict(_DOCTR_VOCABS)
    try:
        _VOCABS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_VOCABS_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_vocabs_cache, f, ensure_ascii=False)
    except Exception:
        pass
    return _vocabs_cache


DEFAULT_VOCAB_LIBRARY = list(_DEFAULT_VOCAB_LIBRARY)
DEFAULT_CUSTOM_CHARACTERS = _DEFAULT_VOCAB_EXTRA_CHARS

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


def _unique_chars_in_order(chars: str) -> str:
    """Keep first occurrence order while removing duplicate characters."""
    return "".join(dict.fromkeys(chars))


def build_custom_vocab_arg(vocab_names: list[str], custom_chars: str) -> str:
    """Build CUSTOM vocab argument from selected library vocab names and custom chars."""
    library_chars = "".join(_get_vocabs()[name] for name in vocab_names if name in _get_vocabs())
    combined_chars = _unique_chars_in_order(library_chars + (custom_chars or ""))
    if not combined_chars:
        raise ValueError("Vocabulary cannot be empty. Select at least one library vocab or custom character.")
    return f"CUSTOM:{combined_chars}"


def suggest_batch_size(task: str) -> int:
    """Suggest a batch size based on available GPU VRAM.

    Uses simple per-task heuristics:
    - recognition: ~16 samples per GB (CRNN/ViT-style, 32px height crops)
    - detection: ~0.5 samples per GB (full-page inputs are much larger)

    Falls back to the task default when no CUDA GPU is available.
    """
    default = 64 if task == "recognition" else 2
    try:
        import torch

        if not torch.cuda.is_available():
            logger.info("[suggest_batch_size] No CUDA GPU available, using default=%d for task=%s", default, task)
            return default
        props = torch.cuda.get_device_properties(0)
        vram_bytes = props.total_memory
        vram_gb = vram_bytes / (1024**3)
        logger.info(
            "[suggest_batch_size] GPU: %s, VRAM: %.2f GB, task=%s",
            props.name,
            vram_gb,
            task,
        )
    except Exception:
        logger.exception("[suggest_batch_size] Failed to query GPU, using default=%d for task=%s", default, task)
        return default

    if task == "recognition":
        raw = int(vram_gb * 16)
        # Round down to nearest power of 2, clamped to [1, 512]
        size = max(1, min(512, 1 << (raw.bit_length() - 1)))
    else:
        raw = int(vram_gb * 0.5)
        size = max(1, min(64, 1 << max(0, raw.bit_length() - 1)))
    logger.info("[suggest_batch_size] task=%s vram_gb=%.2f raw=%d -> batch_size=%d", task, vram_gb, raw, size)
    return size


def _with_case_pair(ch: str) -> str:
    """Return ch plus its upper/lower counterpart when each is a single distinct character."""
    result = ch
    upper = ch.upper()
    if len(upper) == 1 and upper != ch:
        result += upper
    lower = ch.lower()
    if len(lower) == 1 and lower != ch:
        result += lower
    return result


def scan_training_set_for_missing_chars(profile: str, vocab_names: list[str], custom_chars: str) -> str:
    """Scan recognition labels.json for the given profile and return characters not in the current vocab."""
    import json

    current_vocab = build_custom_vocab_arg(vocab_names, custom_chars)
    vocab_chars = set(current_vocab.removeprefix("CUSTOM:"))

    labels_path = ML_TRAINING_DIR / profile / "recognition" / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"No training labels found at {labels_path}")

    with open(labels_path, encoding="utf-8") as f:
        labels: dict = json.load(f)

    missing: set[str] = set()
    for text in labels.values():
        for ch in text:
            if ch not in vocab_chars:
                missing.add(ch)

    # Expand each missing char to include its case pair
    expanded = "".join(_with_case_pair(ch) for ch in sorted(missing))
    # Deduplicate while preserving order, excluding chars already in vocab
    return _unique_chars_in_order("".join(ch for ch in expanded if ch not in vocab_chars))


def _prefixed_model_name(model_type: str, base_name: str, profile: str = BASE_OCR_PROFILE) -> str:
    """Return a normalized model name with enforced prefix, profile, and type."""
    normalized = (base_name or "").strip().replace(" ", "-")
    normalized_profile = normalize_profile_name(profile)
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
        self.device = None  # set lazily at training start via _detect_cuda_device()


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
        vocabs = _get_vocabs()
        default_vocab_library = [name for name in DEFAULT_VOCAB_LIBRARY if name in vocabs]
        if not default_vocab_library:
            default_vocab_library = ["french"] if "french" in vocabs else [next(iter(vocabs))]
        self.vocab_library = default_vocab_library
        self.custom_characters = DEFAULT_CUSTOM_CHARACTERS
        self.vocab = build_custom_vocab_arg(self.vocab_library, self.custom_characters)
        self.workers = _default_worker_count()
        self.device = None  # set lazily at training start via _detect_cuda_device()


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
        cleaned = [str(name) for name in loaded_vocab_library if isinstance(name, str) and name in _get_vocabs()]
        if cleaned:
            recognition_config.vocab_library = cleaned

    loaded_custom_characters = data.get("custom_characters")
    if isinstance(loaded_custom_characters, str):
        # Merge any new default chars that weren't in the saved settings
        merged = _unique_chars_in_order(loaded_custom_characters + DEFAULT_CUSTOM_CHARACTERS)
        recognition_config.custom_characters = merged

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
    model_output_dir(BASE_OCR_PROFILE, "detection").mkdir(parents=True, exist_ok=True)
    model_output_dir(BASE_OCR_PROFILE, "recognition").mkdir(parents=True, exist_ok=True)


migrate_legacy_dataset_layout()
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
        _kanban_refresh: list = [lambda: None]

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
                profile = normalize_profile_name(model_profile_state["value"])
                det_label = output_labels.get("detection")
                rec_label = output_labels.get("recognition")
                dataset_label = dataset_scope_label.get("value")
                if det_label is not None:
                    det_label.set_text(f"Detection output root: {model_output_dir(profile, 'detection')}")
                if rec_label is not None:
                    rec_label.set_text(f"Recognition output root: {model_output_dir(profile, 'recognition')}")
                if dataset_label is not None:
                    dataset_label.set_text(f"Showing exports and dataset files for profile '{profile}'.")
                export_manager.set_profile(profile)
                export_manager.scan()

            ui.select(
                label="Training profile",
                options=profile_options,
                value=model_profile_state["value"],
                on_change=lambda v: (
                    model_profile_state.__setitem__("value", normalize_profile_name(str(v.value))),
                    refresh_model_output_labels(),
                    _kanban_refresh[0](),
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
        refresh_kanban_fn, scope_label_widget = build_dataset_section(export_manager, model_profile_state)
        _kanban_refresh[0] = refresh_kanban_fn
        dataset_scope_label["value"] = scope_label_widget

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

                    with ui.row().classes("w-full items-center gap-2"):
                        det_batch_input = ui.number(
                            label="Batch Size",
                            value=detection_config.batch_size,
                            min=1,
                            max=64,
                            on_change=lambda v: setattr(
                                detection_config,
                                "batch_size",
                                _safe_int(v.value, detection_config.batch_size),
                            ),
                        ).classes("flex-1")

                        def _toggle_auto_det_batch(v) -> None:
                            if v.value:
                                suggested = suggest_batch_size("detection")
                                logger.debug("[Auto batch] Detection: suggested batch_size=%d", suggested)
                                detection_config.batch_size = suggested
                                det_batch_input.set_value(suggested)
                                det_batch_input.disable()
                            else:
                                det_batch_input.enable()

                        ui.checkbox("Auto", on_change=_toggle_auto_det_batch)

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

                    with ui.row().classes("w-full items-center gap-2"):
                        rec_batch_input = ui.number(
                            label="Batch Size",
                            value=recognition_config.batch_size,
                            min=1,
                            max=512,
                            on_change=lambda v: setattr(
                                recognition_config,
                                "batch_size",
                                _safe_int(v.value, recognition_config.batch_size),
                            ),
                        ).classes("flex-1")

                        def _toggle_auto_rec_batch(v) -> None:
                            if v.value:
                                suggested = suggest_batch_size("recognition")
                                logger.debug("[Auto batch] Recognition: suggested batch_size=%d", suggested)
                                recognition_config.batch_size = suggested
                                rec_batch_input.set_value(suggested)
                                rec_batch_input.disable()
                            else:
                                rec_batch_input.enable()

                        ui.checkbox("Auto", on_change=_toggle_auto_rec_batch)

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
                        options=sorted(_get_vocabs().keys()),
                        value=recognition_config.vocab_library,
                        multiple=True,
                        on_change=lambda v: (
                            setattr(recognition_config, "vocab_library", list(v.value or [])),
                            refresh_vocab_ui(),
                        ),
                    ).props("use-chips").classes("w-full")

                    custom_chars_input = ui.input(
                        label="Custom Characters",
                        value=recognition_config.custom_characters,
                        on_change=lambda v: (
                            setattr(recognition_config, "custom_characters", v.value or ""),
                            refresh_vocab_ui(),
                        ),
                    ).classes("w-full")

                    def scan_and_add_missing_chars() -> None:
                        profile = normalize_profile_name(model_profile_state["value"])
                        try:
                            missing = scan_training_set_for_missing_chars(
                                profile,
                                recognition_config.vocab_library,
                                recognition_config.custom_characters,
                            )
                        except FileNotFoundError as exc:
                            ui.notify(str(exc), type="warning")
                            return
                        except Exception as exc:
                            ui.notify(f"Scan failed: {exc}", type="negative")
                            return
                        if not missing:
                            ui.notify("No missing characters found in training set.", type="positive")
                            return
                        recognition_config.custom_characters = _unique_chars_in_order(
                            recognition_config.custom_characters + missing
                        )
                        custom_chars_input.set_value(recognition_config.custom_characters)
                        refresh_vocab_ui()
                        ui.notify(
                            f"Added {len(missing)} missing character(s) to custom vocab: {missing}",
                            type="positive",
                        )

                    ui.button(
                        "Scan Training Data for Missing Chars",
                        on_click=scan_and_add_missing_chars,
                    ).props("color=accent")

                    ui.button(
                        "Reset to Default Vocab Preset",
                        on_click=lambda: (
                            setattr(
                                recognition_config,
                                "vocab_library",
                                [name for name in DEFAULT_VOCAB_LIBRARY if name in _get_vocabs()],
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
                        profile = normalize_profile_name(model_profile_state["value"])
                        out_dir = model_output_dir(profile, "recognition")
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
                selected_profile = normalize_profile_name(model_profile_state["value"])
                detection_out_dir = model_output_dir(selected_profile, "detection")
                recognition_out_dir = model_output_dir(selected_profile, "recognition")
                detection_out_dir.mkdir(parents=True, exist_ok=True)
                recognition_out_dir.mkdir(parents=True, exist_ok=True)

                # Auto-save any pending export assignments before training.
                pending = any(v in {"train", "val"} for v in export_manager.assignments.values())
                if pending:
                    save_result = export_manager.save_assignments()
                    _kanban_refresh[0]()
                    ui.notify(
                        f"Auto-copied {save_result.get('copied', 0)} export task(s) before training.",
                        type="positive",
                    )

                if run_detection:
                    detection_config.model_name = _prefixed_model_name(
                        "detection", detection_config.model_name, selected_profile
                    )
                    det_train_count = label_count(split_profile_root("train", selected_profile), "detection")
                    det_val_count = label_count(split_profile_root("val", selected_profile), "detection")
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
                    rec_train_count = label_count(split_profile_root("train", selected_profile), "recognition")
                    rec_val_count = label_count(split_profile_root("val", selected_profile), "recognition")
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
                    # Heavy imports are deferred here so startup stays fast.
                    import torch

                    from pd_ocr_trainer.train_detect import detect_from_config
                    from pd_ocr_trainer.train_recog import train_from_config

                    cuda_device = 0 if torch.cuda.is_available() else None
                    try:
                        selected_train_root = split_profile_root("train", selected_profile)
                        selected_val_root = split_profile_root("val", selected_profile)

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
                                device=cuda_device,
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
                                device=cuda_device,
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
