from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@pytest.fixture(scope="session")
def browser_dataset_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("trainer-browser-data")

    train_root = root / "ml-training" / "all"
    val_root = root / "ml-validation" / "all"

    detection_labels = {
        "demo_project_001.png": {"boxes": []},
        "demo_project_002.png": {"boxes": []},
        "demo_project_003.png": {"boxes": []},
    }
    recognition_labels = {
        "demo_project_001_0_10_0_10.png": "one",
        "demo_project_002_0_10_0_10.png": "two",
        "demo_project_003_0_10_0_10.png": "three",
    }

    _write_json(train_root / "detection" / "labels.json", detection_labels)
    _write_json(train_root / "recognition" / "labels.json", recognition_labels)
    _write_json(val_root / "detection" / "labels.json", {})
    _write_json(val_root / "recognition" / "labels.json", {})

    export_root = root / "app-data" / "doctr-export" / "pending_project" / "all"
    export_detection_labels = {
        "pending_project_001.png": {"boxes": []},
        "pending_project_002.png": {"boxes": []},
        "pending_project_003.png": {"boxes": []},
    }
    export_recognition_labels = {
        "pending_project_001_0_10_0_10.png": "one",
        "pending_project_002_0_10_0_10.png": "two",
        "pending_project_003_0_10_0_10.png": "three",
    }
    _write_json(export_root / "detection" / "labels.json", export_detection_labels)
    _write_json(export_root / "recognition" / "labels.json", export_recognition_labels)

    (root / "app-data").mkdir(parents=True, exist_ok=True)
    (root / "shared-models").mkdir(parents=True, exist_ok=True)

    return root


@pytest.fixture(scope="session")
def browser_app_url(browser_dataset_root: Path) -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    env = {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}
    env["PD_OCR_TRAINER_ML_TRAINING_DIR"] = str(browser_dataset_root / "ml-training")
    env["PD_OCR_TRAINER_ML_VALIDATION_DIR"] = str(browser_dataset_root / "ml-validation")
    env["PD_OCR_TRAINER_APP_DATA_ROOT"] = str(browser_dataset_root / "app-data")
    env["PD_OCR_TRAINER_SHARED_MODELS_DIR"] = str(browser_dataset_root / "shared-models")
    env["PD_OCR_TRAINER_HOST"] = "127.0.0.1"
    env["PD_OCR_TRAINER_PORT"] = str(port)
    env["PD_OCR_TRAINER_SHOW_BROWSER"] = "false"

    process = subprocess.Popen(
        [sys.executable, "-m", "pd_ocr_trainer.ui"],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    url = f"http://127.0.0.1:{port}/"
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                raise RuntimeError(f"Trainer exited before ready. Output:\n{output}")
            try:
                with urlopen(url, timeout=1):
                    break
            except URLError:
                time.sleep(0.25)
        else:
            raise TimeoutError("Timed out waiting for trainer app startup")

        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


@pytest.fixture(scope="session")
def _browser_instance():
    from playwright.sync_api import sync_playwright

    playwright = sync_playwright().start()
    try:
        browser = playwright.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-software-rasterizer",
            ],
        )
        yield browser
        browser.close()
    finally:
        playwright.stop()


@pytest.fixture
def browser_page(_browser_instance):
    context = _browser_instance.new_context(reduced_motion="reduce")
    context.set_default_navigation_timeout(60_000)
    context.set_default_timeout(30_000)
    page = context.new_page()
    yield page
    context.close()
