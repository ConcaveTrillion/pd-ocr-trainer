from __future__ import annotations

import re

import pytest
from playwright.sync_api import expect


@pytest.mark.browser
def test_export_project_handle_moves_only_pages_currently_in_column(browser_app_url: str, browser_page) -> None:
    page = browser_page
    page.goto(browser_app_url, wait_until="networkidle")

    expect(page.get_by_text("📂 Dataset Management", exact=True)).to_be_visible()

    unassigned_col = page.get_by_text("📋 Unassigned", exact=True).locator(
        "xpath=ancestor::*[contains(@class,'q-card')][1]"
    )
    train_col = page.get_by_text("🔵 Training", exact=True).locator("xpath=ancestor::*[contains(@class,'q-card')][1]")
    val_col = page.get_by_text("🟢 Validation", exact=True).locator("xpath=ancestor::*[contains(@class,'q-card')][1]")

    # Reset export assignments so this test is deterministic when run with others.
    train_col.get_by_role("button", name="Clear").click()
    val_col.get_by_role("button", name="Clear").click()

    # First split one page out from Unassigned -> Training.
    unassigned_header = unassigned_col.get_by_text(re.compile(r"pending_project\s+·\s+3\s+pages\s+\[export\]"))
    expect(unassigned_header).to_be_visible()
    unassigned_header.click()
    first_page = unassigned_col.get_by_text("pending_project_001.png").first
    first_page.drag_to(train_col)

    expect(unassigned_col.get_by_text(re.compile(r"pending_project\s+·\s+2\s+pages\s+\[export\]"))).to_be_visible()
    expect(train_col.get_by_text(re.compile(r"pending_project\s+·\s+1\s+pages\s+\[export\]"))).to_be_visible()

    # Now drag the Unassigned project handle: only those 2 pages should move to Validation.
    unassigned_col.get_by_text(re.compile(r"pending_project\s+·\s+2\s+pages\s+\[export\]")).click()
    handle = unassigned_col.get_by_text("Drag to move all pages", exact=True).first
    expect(handle).to_be_visible()
    handle.drag_to(val_col)

    expect(train_col.get_by_text(re.compile(r"pending_project\s+·\s+1\s+pages\s+\[export\]"))).to_be_visible()
    expect(val_col.get_by_text(re.compile(r"pending_project\s+·\s+2\s+pages\s+\[export\]"))).to_be_visible()
