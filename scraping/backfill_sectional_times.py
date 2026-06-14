import asyncio
import logging
import os
import csv
import random
from typing import Dict, Any
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from playwright_stealth import Stealth
from bs4 import BeautifulSoup
import re
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()

# --- Configuration & Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:your_password@127.0.0.1:5432/hkjc_db")

class SectionalTimeScraper:
    def __init__(self, proxy_url: str = None):
        self.proxy_url = proxy_url
        self.rate_limit_delay = 3.0
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0"
        ]

    async def fetch_sectional_times(self, page, date_str: str, race_no: int) -> Dict[str, Dict[str, float]]:
        """Fetches and parses sectional times for a given race."""
        # date_str is expected to be YYYY/MM/DD or YYYY-MM-DD
        date_parts = date_str.replace('-', '/').split('/')
        formatted_date = f"{date_parts[2]}/{date_parts[1]}/{date_parts[0]}"
        url = f"https://racing.hkjc.com/racing/information/English/Racing/DisplaySectionalTime.aspx?RaceDate={formatted_date}&RaceNo={race_no}"
        
        logger.info(f"Fetching sectional times from {url}")
        try:
            delay = self.rate_limit_delay + random.uniform(1.0, 3.0)
            await asyncio.sleep(delay)
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            
            try:
                await page.wait_for_selector('.table_bd.f_tac.race_table', timeout=10000)
            except PlaywrightTimeoutError:
                logger.info(f"No sectional times table found for {date_str} Race {race_no}")
                return {}
                
            html_content = await page.content()
            soup = BeautifulSoup(html_content, 'html.parser')
            table = soup.find('table', class_='table_bd f_tac race_table')
            if not table:
                return {}
                
            sectional_data = {}
            rows = table.find_all('tr')[3:] # Skip headers
            for row in rows:
                cols = [td.text.strip() for td in row.find_all('td')]
                if len(cols) < 10:
                    continue
                    
                horse_name_raw = cols[2]
                horse_code_match = re.search(r'\(([^)]+)\)', horse_name_raw)
                horse_code = horse_code_match.group(1) if horse_code_match else None
                
                if not horse_code:
                    continue
                    
                sec_times = {}
                for i in range(1, 7):
                    col_idx = 2 + i
                    td = row.find_all('td')[col_idx]
                    p_tags = td.find_all('p')
                    if p_tags:
                        try:
                            # The last <p> tag usually contains the sectional time
                            time_str = p_tags[-1].contents[0].strip()
                            sec_times[f'sec{i}_time'] = float(time_str)
                        except (ValueError, IndexError):
                            sec_times[f'sec{i}_time'] = None
                    else:
                        sec_times[f'sec{i}_time'] = None
                        
                sectional_data[horse_code] = sec_times
                
            return sectional_data
        except Exception as e:
            logger.error(f"Error fetching sectional times for {date_str} Race {race_no}: {e}")
            return {}

async def update_database_schema(engine):
    """Adds sectional time columns to the database if they don't exist."""
    try:
        async with engine.begin() as conn:
            # Check if columns exist
            res = await conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='race_results'"))
            columns = [r[0] for r in res.fetchall()]
            
            if 'sec1_time' not in columns:
                logger.info("Adding sectional time columns to database...")
                for i in range(1, 7):
                    await conn.execute(text(f"ALTER TABLE race_results ADD COLUMN sec{i}_time NUMERIC(6, 2)"))
                logger.info("Database schema updated.")
    except Exception as e:
        logger.warning(f"Could not update database schema (is the DB running?): {e}")

async def update_database_records(engine, date_str, race_no, horse_name, distance, sec_times):
    """Updates the database with sectional times for a specific horse in a race."""
    try:
        async with engine.begin() as conn:
            # Find race_id
            date_parts = date_str.replace('-', '/').split('/')
            formatted_date = f"{date_parts[0]}-{date_parts[1]}-{date_parts[2]}"
            
            query = text("""
                UPDATE race_results rr
                SET sec1_time = :sec1, sec2_time = :sec2, sec3_time = :sec3, 
                    sec4_time = :sec4, sec5_time = :sec5, sec6_time = :sec6
                FROM horses h, races r
                WHERE rr.horse_id = h.horse_id 
                  AND rr.race_id = r.race_id
                  AND r.race_date = :race_date
                  AND r.distance = :distance
                  AND h.name = :horse_name
            """)
            
            await conn.execute(query, {
                'sec1': sec_times.get('sec1_time'),
                'sec2': sec_times.get('sec2_time'),
                'sec3': sec_times.get('sec3_time'),
                'sec4': sec_times.get('sec4_time'),
                'sec5': sec_times.get('sec5_time'),
                'sec6': sec_times.get('sec6_time'),
                'race_date': formatted_date,
                'distance': int(distance),
                'horse_name': horse_name
            })
    except Exception as e:
        pass # Ignore DB errors during backfill to allow CSVs to update

async def main():
    engine = create_async_engine(DB_URL, echo=False)
    await update_database_schema(engine)
    
    scraper = SectionalTimeScraper(proxy_url="http://127.0.0.1:7890")
    csv_dir = "data/raw_csvs"
    
    if not os.path.exists(csv_dir):
        logger.error(f"Directory {csv_dir} does not exist.")
        return
        
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    csv_files.sort()
    
    async with async_playwright() as p:
        launch_args = {"headless": True}
        if scraper.proxy_url:
            launch_args["proxy"] = {"server": scraper.proxy_url}
            
        browser = await p.chromium.launch(**launch_args)
        user_agent = random.choice(scraper.user_agents)
        context = await browser.new_context(
            user_agent=user_agent,
            viewport={"width": random.randint(1366, 1920), "height": random.randint(768, 1080)}
        )
        page = await context.new_page()
        await Stealth().apply_stealth_async(page)
        
        try:
            for filename in csv_files:
                filepath = os.path.join(csv_dir, filename)
                date_str = filename.replace('.csv', '')
                
                # Read CSV
                with open(filepath, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    fieldnames = reader.fieldnames
                    
                if not rows:
                    continue
                    
                # Check if already backfilled
                if 'sec1_time' in fieldnames and any(row.get('sec1_time') for row in rows):
                    logger.info(f"Skipping {filename}, already backfilled.")
                    continue
                    
                logger.info(f"Backfilling {filename}...")
                
                # Group rows by race_number
                races = {}
                for row in rows:
                    race_no = int(row['race_number'])
                    if race_no not in races:
                        races[race_no] = []
                    races[race_no].append(row)
                    
                # Fetch sectional times for each race
                for race_no, race_rows in races.items():
                    sec_times = await scraper.fetch_sectional_times(page, date_str, race_no)
                    
                    for row in race_rows:
                        horse_code = row.get('horse_code')
                        horse_name = row.get('horse_name')
                        distance = row.get('distance')
                        if horse_code and horse_code in sec_times:
                            horse_sec_times = sec_times[horse_code]
                            row.update(horse_sec_times)
                            # Update database
                            await update_database_records(engine, date_str, race_no, horse_name, distance, horse_sec_times)
                        else:
                            empty_sec_times = {f'sec{i}_time': None for i in range(1, 7)}
                            row.update(empty_sec_times)
                            
                # Save updated CSV
                new_fieldnames = fieldnames.copy()
                for i in range(1, 7):
                    col = f'sec{i}_time'
                    if col not in new_fieldnames:
                        new_fieldnames.append(col)
                        
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=new_fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                    
                logger.info(f"Successfully backfilled and saved {filename}")
                
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(main())