"""Playwright end-to-end checks for AngularJS UI workflows."""

from __future__ import annotations

import os

import pytest

playwright = pytest.importorskip("playwright.sync_api")
sync_playwright = playwright.sync_playwright


@pytest.mark.e2e
def test_viewer_login_and_sidebar_navigation() -> None:
    base_url = os.getenv("E2E_BASE_URL", "http://127.0.0.1:8000")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(f"{base_url}/ui/index.html", wait_until="domcontentloaded")
        page.fill("#login-username", "viewer")
        page.fill("#login-password", "viewer123")
        page.click("#login-submit")

        page.wait_for_selector("text=Session", timeout=15000)
        page.click("a:has-text('NDT Explorer')")
        page.wait_for_selector("text=Available Samples", timeout=20000)
        page.click("a:has-text('Dashboard')")
        page.wait_for_selector("text=System Overview", timeout=15000)

        preprocessing_link = page.locator("a:has-text('Preprocessing Lab')")
        assert "disabled" in (preprocessing_link.get_attribute("class") or "")
        browser.close()


@pytest.mark.e2e
def test_admin_sees_server_analytics_panel() -> None:
    base_url = os.getenv("E2E_BASE_URL", "http://127.0.0.1:8000")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(f"{base_url}/ui/index.html", wait_until="domcontentloaded")
        page.fill("#login-username", "admin")
        page.fill("#login-password", "admin123")
        page.click("#login-submit")

        page.wait_for_selector("text=Session", timeout=15000)
        page.wait_for_selector("text=Server Error Analytics", timeout=20000)
        browser.close()
