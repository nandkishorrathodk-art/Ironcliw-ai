#!/usr/bin/env python3
"""
NotebookLM Upload Script

Uses Playwright browser automation to upload sources to NotebookLM.
You'll need to log in to Google once, then it automates the rest.

Usage:
    python upload_to_notebooklm.py
    python upload_to_notebooklm.py --source-dir ~/NotebookLM_Sources
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout


class NotebookLMUploader:
    """Automates uploading sources to NotebookLM via browser."""

    NOTEBOOKLM_URL = "https://notebooklm.google.com/"

    def __init__(self, source_dir: Path, headless: bool = False):
        self.source_dir = source_dir
        self.headless = headless
        self.browser = None
        self.context = None
        self.page = None
        # Persistent browser data to keep login
        self.user_data_dir = Path.home() / ".notebooklm_browser"

    async def setup(self):
        """Initialize browser with persistent context."""
        self.playwright = await async_playwright().start()

        # Use persistent context to save login
        self.context = await self.playwright.chromium.launch_persistent_context(
            user_data_dir=str(self.user_data_dir),
            headless=self.headless,
            slow_mo=100,  # Slow down for visibility
            viewport={"width": 1280, "height": 900},
        )
        self.page = self.context.pages[0] if self.context.pages else await self.context.new_page()

    async def cleanup(self):
        """Close browser."""
        if self.context:
            await self.context.close()
        if self.playwright:
            await self.playwright.stop()

    async def ensure_logged_in(self) -> bool:
        """Navigate to NotebookLM and ensure user is logged in."""
        print("Opening NotebookLM...")
        await self.page.goto(self.NOTEBOOKLM_URL, wait_until="networkidle")

        # Check if we're on login page or NotebookLM
        await asyncio.sleep(2)
        current_url = self.page.url

        if "accounts.google.com" in current_url:
            print("\n" + "=" * 60)
            print("GOOGLE LOGIN REQUIRED")
            print("=" * 60)
            print("Please log in to your Google account in the browser window.")
            print("This script will wait until you're logged in...")
            print("=" * 60 + "\n")

            # Wait for redirect back to NotebookLM (max 5 minutes)
            try:
                await self.page.wait_for_url(
                    "**/notebooklm.google.com/**",
                    timeout=300000
                )
                print("Login successful!")
                await asyncio.sleep(2)
            except PlaywrightTimeout:
                print("Login timeout - please try again")
                return False

        return True

    async def create_notebook(self, name: str) -> bool:
        """Create a new notebook with the given name."""
        print(f"\nCreating notebook: {name}")

        try:
            # Look for "New notebook" or "Create" button
            # NotebookLM UI varies, so try multiple selectors
            new_btn_selectors = [
                'button:has-text("New notebook")',
                'button:has-text("Create")',
                '[aria-label="Create new notebook"]',
                'button:has-text("New")',
                '.create-notebook-button',
            ]

            clicked = False
            for selector in new_btn_selectors:
                try:
                    btn = self.page.locator(selector).first
                    if await btn.is_visible(timeout=2000):
                        await btn.click()
                        clicked = True
                        break
                except:
                    continue

            if not clicked:
                # Try clicking the + icon or floating action button
                try:
                    fab = self.page.locator('[aria-label*="new"], [aria-label*="create"], .fab, .floating-button').first
                    await fab.click()
                    clicked = True
                except:
                    pass

            if not clicked:
                print(f"  Could not find 'New notebook' button")
                print(f"  Please create notebook '{name}' manually in the browser")
                input("  Press Enter when ready to continue...")
                return True

            await asyncio.sleep(2)

            # Try to set notebook name if there's an input
            try:
                name_input = self.page.locator('input[type="text"], [contenteditable="true"]').first
                if await name_input.is_visible(timeout=3000):
                    await name_input.fill(name)
                    await self.page.keyboard.press("Enter")
            except:
                pass

            await asyncio.sleep(2)
            print(f"  Notebook created (or ready for manual creation)")
            return True

        except Exception as e:
            print(f"  Error creating notebook: {e}")
            return False

    async def upload_file(self, file_path: Path) -> bool:
        """Upload a single file to the current notebook."""
        print(f"  Uploading: {file_path.name}")

        try:
            # Look for "Add source" or upload button
            add_source_selectors = [
                'button:has-text("Add source")',
                'button:has-text("Upload")',
                '[aria-label*="Add source"]',
                '[aria-label*="Upload"]',
                'button:has-text("Add")',
            ]

            clicked = False
            for selector in add_source_selectors:
                try:
                    btn = self.page.locator(selector).first
                    if await btn.is_visible(timeout=2000):
                        await btn.click()
                        clicked = True
                        break
                except:
                    continue

            if not clicked:
                print(f"    Could not find 'Add source' button")
                return False

            await asyncio.sleep(1)

            # Look for file upload option
            upload_selectors = [
                'button:has-text("Upload")',
                'button:has-text("File")',
                '[aria-label*="Upload file"]',
                'input[type="file"]',
            ]

            for selector in upload_selectors:
                try:
                    if selector == 'input[type="file"]':
                        # Direct file input
                        file_input = self.page.locator(selector).first
                        await file_input.set_input_files(str(file_path))
                        await asyncio.sleep(2)
                        return True
                    else:
                        btn = self.page.locator(selector).first
                        if await btn.is_visible(timeout=2000):
                            await btn.click()
                            await asyncio.sleep(1)

                            # Now look for file input
                            file_input = self.page.locator('input[type="file"]').first
                            await file_input.set_input_files(str(file_path))
                            await asyncio.sleep(2)
                            return True
                except:
                    continue

            print(f"    Could not upload file automatically")
            return False

        except Exception as e:
            print(f"    Error uploading: {e}")
            return False

    async def upload_notebook_sources(self, notebook_name: str, source_files: list) -> dict:
        """Upload all sources for a notebook."""
        results = {
            "notebook": notebook_name,
            "uploaded": [],
            "failed": [],
        }

        # Create notebook
        if not await self.create_notebook(notebook_name):
            results["failed"] = [str(f) for f in source_files]
            return results

        await asyncio.sleep(2)

        # Upload each file (limit to avoid overwhelming)
        max_files = 20  # NotebookLM works best with fewer focused sources
        files_to_upload = source_files[:max_files]

        if len(source_files) > max_files:
            print(f"  Note: Uploading {max_files} of {len(source_files)} files (NotebookLM works best with fewer sources)")

        for file_path in files_to_upload:
            success = await self.upload_file(file_path)
            if success:
                results["uploaded"].append(str(file_path))
            else:
                results["failed"].append(str(file_path))

            # Small delay between uploads
            await asyncio.sleep(1)

        return results

    async def run(self) -> dict:
        """Main upload process."""
        results = {
            "notebooks": [],
            "total_uploaded": 0,
            "total_failed": 0,
        }

        try:
            await self.setup()

            # Ensure logged in
            if not await self.ensure_logged_in():
                return results

            # Find notebook folders
            notebook_dirs = [d for d in self.source_dir.iterdir() if d.is_dir()]

            if not notebook_dirs:
                print(f"No notebook folders found in {self.source_dir}")
                return results

            print(f"\nFound {len(notebook_dirs)} notebooks to upload")

            for notebook_dir in notebook_dirs:
                notebook_name = notebook_dir.name.replace("_", " ")
                source_files = sorted(notebook_dir.glob("*.md"))

                if not source_files:
                    print(f"\nSkipping {notebook_name} - no files")
                    continue

                print(f"\n{'='*60}")
                print(f"NOTEBOOK: {notebook_name}")
                print(f"{'='*60}")
                print(f"Files: {len(source_files)}")

                notebook_results = await self.upload_notebook_sources(notebook_name, source_files)
                results["notebooks"].append(notebook_results)
                results["total_uploaded"] += len(notebook_results["uploaded"])
                results["total_failed"] += len(notebook_results["failed"])

                # Navigate back to main page for next notebook
                await self.page.goto(self.NOTEBOOKLM_URL, wait_until="networkidle")
                await asyncio.sleep(2)

        finally:
            # Don't close browser immediately - let user see results
            print("\n" + "=" * 60)
            print("UPLOAD COMPLETE")
            print("=" * 60)
            print(f"Uploaded: {results['total_uploaded']} files")
            print(f"Failed: {results['total_failed']} files")

            if results["total_failed"] > 0:
                print("\nSome files failed. You may need to upload them manually.")

            print("\nBrowser will stay open for 30 seconds so you can verify...")
            await asyncio.sleep(30)

            await self.cleanup()

        return results


async def main():
    parser = argparse.ArgumentParser(description="Upload sources to NotebookLM")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path.home() / "NotebookLM_Sources",
        help="Directory containing notebook source folders"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (not recommended - you need to log in)"
    )

    args = parser.parse_args()

    if not args.source_dir.exists():
        print(f"Source directory not found: {args.source_dir}")
        print("Run sync_to_notebooklm.py first to prepare sources")
        return 1

    uploader = NotebookLMUploader(
        source_dir=args.source_dir,
        headless=args.headless
    )

    results = await uploader.run()

    # Save results
    results_path = args.source_dir / "upload_results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to: {results_path}")

    return 0 if results["total_failed"] == 0 else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
