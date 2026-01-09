#!/usr/bin/env python3
"""
Test what data we can access from a trajectory page.
"""

import json
from playwright.sync_api import sync_playwright
from pathlib import Path

def test_trajectory_access(agent_run_id: str = "eaa8e4b1-dda6-46ec-9787-ba7ccebfafa2"):
    """Test different ways to access trajectory data."""

    collection_id = "032fb63d-4992-4bfc-911d-3b7dafcb931f"
    url = f"https://docent.transluce.org/dashboard/{collection_id}/agent_run/{agent_run_id}"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        # Intercept network requests to see if we can capture API calls
        api_calls = []
        def handle_response(response):
            if 'api' in response.url or 'json' in response.headers.get('content-type', ''):
                api_calls.append({
                    'url': response.url,
                    'status': response.status,
                    'content-type': response.headers.get('content-type', '')
                })

        page.on('response', handle_response)

        print(f"Loading: {url}")
        page.goto(url, wait_until='networkidle', timeout=60000)
        page.wait_for_timeout(5000)

        print("\n=== API Calls Captured ===")
        for call in api_calls:
            print(f"  {call['url']}")
            print(f"    Status: {call['status']}, Type: {call['content-type']}")

        # Try to find trajectory data in page
        print("\n=== Searching for Trajectory Data ===")

        # Method 1: Check for embedded JSON
        scripts = page.locator('script').all()
        print(f"Found {len(scripts)} script tags")

        for i, script in enumerate(scripts[:5]):  # Check first 5
            content = script.inner_text()
            if 'trajectory' in content.lower() or 'agentrun' in content.lower() or len(content) > 1000:
                print(f"\nScript {i+1} (first 500 chars):")
                print(content[:500])

        # Method 2: Check window objects
        print("\n=== Checking Window Objects ===")
        window_data = page.evaluate("""
            () => {
                const keys = Object.keys(window).filter(k =>
                    k.includes('__') ||
                    k.includes('data') ||
                    k.includes('state') ||
                    k.includes('trajectory') ||
                    k.includes('agent')
                );
                return keys.slice(0, 20);
            }
        """)
        print(f"Interesting window keys: {window_data}")

        # Method 3: Check for Next.js data
        print("\n=== Checking Next.js Data ===")
        nextjs_data = page.evaluate("""
            () => {
                if (window.__NEXT_DATA__) {
                    return {
                        hasNextData: true,
                        keys: Object.keys(window.__NEXT_DATA__),
                        propsKeys: window.__NEXT_DATA__.props ? Object.keys(window.__NEXT_DATA__.props) : []
                    };
                }
                return { hasNextData: false };
            }
        """)
        print(json.dumps(nextjs_data, indent=2))

        if nextjs_data.get('hasNextData'):
            # Try to get the actual data
            full_data = page.evaluate("() => window.__NEXT_DATA__")
            output_file = Path("swebench_pro_nextjs_data.json")
            with open(output_file, 'w') as f:
                json.dump(full_data, f, indent=2)
            print(f"\nSaved Next.js data to: {output_file}")
            print(f"Size: {output_file.stat().st_size:,} bytes")

        # Keep browser open for manual inspection
        print("\n=== Browser will stay open for 30 seconds for manual inspection ===")
        print("Check the browser to see if there's a download button visible")
        page.wait_for_timeout(30000)

        browser.close()

        return api_calls

if __name__ == "__main__":
    api_calls = test_trajectory_access()
