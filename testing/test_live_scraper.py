import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

async def check_live_races():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # HKJC Race Card URL
        url = "https://bet.hkjc.com/en/racing/wp/2026-02-28/S1/1"
        print(f"Navigating to {url}")
        
        try:
            await page.goto(url, timeout=60000)
            # Wait for the table or a specific element to load
            await page.wait_for_load_state('networkidle')
            await asyncio.sleep(5) # Give it a few seconds to render JS
            
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            
            print("Page title:", await page.title())
            
            # Find the race table
            # The betting site uses different classes, let's just look for any table first
            tables = soup.find_all('table')
            if tables:
                print(f"Found {len(tables)} tables!")
                for i, table in enumerate(tables):
                    print(f"Table {i} classes: {table.get('class')}")
                    rows = table.find_all('tr')
                    if len(rows) > 1:
                        print(f"Table {i} first row: {[td.text.strip() for td in rows[1].find_all(['th', 'td'])]}")
            else:
                print("No table found.")
                
            with open("test_live.html", "w", encoding="utf-8") as f:
                f.write(content)
            print("Saved HTML to test_live.html")
                
        except Exception as e:
            print(f"Error: {e}")
            # Save HTML even on error if possible
            try:
                content = await page.content()
                with open("test_live_error.html", "w", encoding="utf-8") as f:
                    f.write(content)
            except:
                pass
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(check_live_races())
