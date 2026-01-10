#!/usr/bin/env python3
"""
Scrape SWE-bench Pro dashboard to verify statistics.
"""

import json
import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright

def scrape_dashboard_stats(headless: bool = True):
    """
    Scrape basic statistics from the SWE-bench Pro dashboard.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()

        print("Loading dashboard...")
        page.goto('https://docent.transluce.org/dashboard/032fb63d-4992-4bfc-911d-3b7dafcb931f',
                  wait_until='networkidle', timeout=60000)
        page.wait_for_timeout(3000)

        # Get page content for inspection
        html = page.content()

        # Save HTML for manual inspection
        with open('swebench_pro_dashboard.html', 'w') as f:
            f.write(html)
        print("Saved dashboard HTML to swebench_pro_dashboard.html")

        # Try to extract statistics from the page
        stats = {}

        # Look for text patterns that might indicate counts
        text = page.inner_text('body')

        # Save the text content too
        with open('swebench_pro_dashboard.txt', 'w') as f:
            f.write(text)
        print("Saved dashboard text to swebench_pro_dashboard.txt")

        browser.close()

        return stats

def scrape_single_trajectory(trajectory_url: str, headless: bool = True):
    """
    Scrape a single trajectory page to understand the structure.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()

        print(f"Loading trajectory: {trajectory_url}")
        page.goto(trajectory_url, wait_until='networkidle', timeout=60000)
        page.wait_for_timeout(3000)

        # Get page content
        html = page.content()
        text = page.inner_text('body')

        # Save for inspection
        with open('swebench_pro_trajectory.html', 'w') as f:
            f.write(html)
        with open('swebench_pro_trajectory.txt', 'w') as f:
            f.write(text)

        print("Saved trajectory HTML to swebench_pro_trajectory.html")
        print("Saved trajectory text to swebench_pro_trajectory.txt")

        browser.close()

def main():
    parser = argparse.ArgumentParser(description="Scrape SWE-bench Pro dashboard")
    parser.add_argument("--no-headless", action="store_true",
                       help="Run browser in visible mode")
    parser.add_argument("--trajectory", type=str,
                       help="Scrape a specific trajectory URL instead of dashboard")
    args = parser.parse_args()

    if args.trajectory:
        scrape_single_trajectory(args.trajectory, headless=not args.no_headless)
    else:
        stats = scrape_dashboard_stats(headless=not args.no_headless)
        print("\nDashboard statistics:")
        print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
