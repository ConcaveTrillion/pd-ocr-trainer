from __future__ import annotations

import re

import pytest
from playwright.sync_api import expect


@pytest.mark.browser
def test_dragging_single_page_moves_only_that_page(browser_app_url: str, browser_page) -> None:
    page = browser_page
    page.goto(browser_app_url, wait_until="networkidle")

    expect(page.get_by_text("📂 Dataset Management", exact=True)).to_be_visible()

    train_col = page.get_by_text("🔵 Training", exact=True).locator("xpath=ancestor::*[contains(@class,'q-card')][1]")
    val_col = page.get_by_text("🟢 Validation", exact=True).locator("xpath=ancestor::*[contains(@class,'q-card')][1]")

    train_header = train_col.get_by_text(re.compile(r"demo_project\s+·\s+3\s+pages\s+\[on disk\]"))
    expect(train_header).to_be_visible()

    train_header.click()
    page_to_move = train_col.get_by_text("demo_project_001.png").first
    expect(page_to_move).to_be_visible()

    val_drop_area = val_col.locator(".min-h-16").first
    page_to_move.drag_to(val_drop_area)

    expect(train_col.get_by_text(re.compile(r"demo_project\s+·\s+2\s+pages\s+\[on disk\]"))).to_be_visible()
    expect(val_col.get_by_text(re.compile(r"demo_project\s+·\s+1\s+pages\s+\[on disk\]"))).to_be_visible()
