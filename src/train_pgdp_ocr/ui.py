"""NiceGUI training interface for OCR configuration and dataset management."""

import json
import threading
from pathlib import Path

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
        self.total_pages = 0
        self.assigned_pages = 0

    def load_json_file(self, filepath: Path) -> dict:
        """Load a JSON file from matched-ocr."""
        try:
            with open(filepath) as f:
                data = json.load(f)
            filename = filepath.name
            self.loaded_files[filename] = data

            # Count pages
            pages = data.get("pages", [])
            if filename not in self.assignments:
                self.assignments[filename] = dict.fromkeys(range(len(pages)))

            return data
        except Exception as e:
            raise ValueError(f"Failed to load {filepath.name}: {e}") from e

    def get_available_files(self) -> list[str]:
        """Get list of available JSON files in matched-ocr."""
        if not MATCHED_OCR_DIR.exists():
            return []
        return [f.name for f in MATCHED_OCR_DIR.glob("*.json")]

    def get_page_count(self, filename: str) -> int:
        """Get number of pages in a file."""
        if filename in self.loaded_files:
            return len(self.loaded_files[filename].get("pages", []))
        return 0

    def assign_page(self, filename: str, page_index: int, target: str):
        """Assign a page to train, val, or None."""
        if filename in self.assignments:
            self.assignments[filename][page_index] = target if target != "none" else None

    def update_stats(self):
        """Update total and assigned page counts."""
        self.total_pages = sum(len(self.loaded_files.get(f, {}).get("pages", [])) for f in self.loaded_files)
        self.assigned_pages = sum(
            1 for file_assigns in self.assignments.values() if any(v is not None for v in file_assigns.values())
        )

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
                ui.label("Available JSON Files").classes("font-semibold")
                available_files = dataset_manager.get_available_files()

                async def load_file(filename: str):
                    try:
                        filepath = MATCHED_OCR_DIR / filename
                        dataset_manager.load_json_file(filepath)
                        page_count = dataset_manager.get_page_count(filename)
                        files_label.set_text(f"✓ Loaded: {filename} ({page_count} pages)")
                        refresh_page_grid()
                        update_stats()
                    except Exception as e:
                        files_label.set_text(f"✗ Error: {e}")

                files_label = ui.label("Select file to load...")
                with ui.row():
                    for fname in available_files[:10]:  # Show first 10
                        ui.button(
                            fname[:20] + "..." if len(fname) > 20 else fname,
                            on_click=lambda f=fname: load_file(f),
                        ).props("size=sm").tooltip(fname)

            # Page assignment grid
            ui.label("Assign Pages to Sets").classes("font-semibold mt-4")

            page_container = ui.column().classes("w-full border-l-4 border-blue-300 pl-4")

            async def refresh_page_grid():
                page_container.clear()

                for filename, assignments in dataset_manager.assignments.items():
                    if filename not in dataset_manager.loaded_files:
                        continue

                    with page_container:
                        ui.label(f"📄 {filename}").classes("font-semibold text-sm mt-2")

                        with ui.row().classes("flex-wrap gap-2"):
                            for page_idx in assignments:
                                current = assignments[page_idx]

                                async def update_assignment(f=filename, p=page_idx, v=None):
                                    dataset_manager.assign_page(f, p, v)
                                    await refresh_page_grid()
                                    update_stats()

                                ui.select(
                                    options=["none", "train", "val"],
                                    value=current or "none",
                                    on_change=lambda v, f=filename, p=page_idx: (
                                        dataset_manager.assign_page(f, p, v.value),
                                        update_stats(),
                                    ),
                                ).props("size=sm dense").classes("w-24")

            # Stats
            stats_label = ui.label("No data loaded").classes("text-sm text-gray-600 mt-4")

            def update_stats():
                dataset_manager.update_stats()
                stats_label.set_text(
                    f"📊 Total pages: {dataset_manager.total_pages} | Assigned: {dataset_manager.assigned_pages}"
                )

        # ==================== TRAINING CONFIG SECTION ====================
        with ui.card().classes("flex-1"):
            ui.label("⚙️ Training Configuration").classes("text-lg font-bold")

            with ui.tabs().classes("w-full"):
                with ui.tab_panel("Basic"):
                    ui.label("Model & Data").classes("font-semibold text-sm")

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
                        on_change=lambda v: setattr(training_config, "epochs", v.value),
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

                with ui.tab_panel("Optimizer"):
                    ui.label("Optimizer & Learning Rate").classes("font-semibold text-sm")

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

                with ui.tab_panel("Advanced"):
                    ui.label("Advanced Options").classes("font-semibold text-sm")

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
        output_area = ui.textarea(
            value="Training output will appear here...",
            readonly=True,
        ).classes("w-full h-64")

        def run_training():
            """Run training in background thread."""
            global training_thread, training_cancelled

            if training_thread and training_thread.is_alive():
                ui.notify("Training is already running!", type="warning")
                return

            # Export datasets
            try:
                datasets = dataset_manager.export_datasets()
                if not datasets["train"] or not datasets["val"]:
                    ui.notify(
                        "Please assign pages to both training and validation sets!",
                        type="warning",
                    )
                    return

                # Save dataset info
                train_info_file = ML_TRAINING_DIR / "dataset_info.json"
                val_info_file = ML_VALIDATION_DIR / "dataset_info.json"

                train_info_file.write_text(json.dumps({"pages": datasets["train"]}, indent=2))
                val_info_file.write_text(json.dumps({"pages": datasets["val"]}, indent=2))

                status_label.set_text("⏳ Training starting...")
                output_area.value = "Starting training...\n"
                training_cancelled = False

                def train_worker():
                    global training_thread, training_cancelled
                    try:
                        # Call training directly
                        train_from_config(
                            train_path=ML_TRAINING_DIR,
                            val_path=ML_VALIDATION_DIR,
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
    create_ui()
    ui.run(host="127.0.0.1", port=8000, reload=True)


if __name__ in {"__main__", "__mp_main__"}:
    main()
