from __future__ import annotations

import re

import pytest
from playwright.sync_api import expect


@pytest.mark.browser
def test_dragging_one_unassigned_export_page_moves_only_that_page(browser_app_url: str, browser_page) -> None:
    page = browser_page
    page.goto(browser_app_url, wait_until="networkidle")

    expect(page.get_by_text("📂 Dataset Management", exact=True)).to_be_visible()

    unassigned_col = page.get_by_text("📋 Unassigned", exact=True).locator(
        "xpath=ancestor::*[contains(@class,'q-card')][1]"
    )
    train_col = page.get_by_text("🔵 Training", exact=True).locator("xpath=ancestor::*[contains(@class,'q-card')][1]")

    unassigned_header = unassigned_col.get_by_text(re.compile(r"pending_project\s+·\s+3\s+pages\s+\[export\]"))
    expect(unassigned_header).to_be_visible()

    unassigned_header.click()
    page_to_move = unassigned_col.get_by_text("pending_project_001.png").first
    expect(page_to_move).to_be_visible()

    train_drop_area = train_col.locator(".min-h-16").first
    page_to_move.drag_to(train_drop_area)

    expect(unassigned_col.get_by_text(re.compile(r"pending_project\s+·\s+2\s+pages\s+\[export\]"))).to_be_visible()
    expect(train_col.get_by_text(re.compile(r"pending_project\s+·\s+1\s+pages\s+\[export\]"))).to_be_visible()
