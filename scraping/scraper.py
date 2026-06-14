import asyncio
import logging
import random
import os
import csv
from typing import List, Dict, Any
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from playwright_stealth import Stealth
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, Date, Numeric, ForeignKey, select, UniqueConstraint
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime, date, timedelta
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import re

load_dotenv()

# --- Configuration & Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

Base = declarative_base()
DB_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:your_password@127.0.0.1:5432/hkjc_db")

# --- SQLAlchemy Models ---
class Race(Base):
    __tablename__ = 'races'
    race_id = Column(Integer, primary_key=True, autoincrement=True)
    track = Column(String(50), nullable=False)
    race_date = Column(Date, nullable=False)
    race_class = Column(String(20))
    distance = Column(Integer, nullable=False)
    __table_args__ = (UniqueConstraint('track', 'race_date', 'distance', name='_track_date_dist_uc'),)

class Horse(Base):
    __tablename__ = 'horses'
    horse_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)
    origin = Column(String(50))
    age = Column(Integer)

class RaceResult(Base):
    __tablename__ = 'race_results'
    result_id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(Integer, ForeignKey('races.race_id', ondelete='CASCADE'))
    horse_id = Column(Integer, ForeignKey('horses.horse_id', ondelete='CASCADE'))
    finish_position = Column(String(10))
    finishing_time = Column(Numeric(6, 2))
    jockey = Column(String(100))
    trainer = Column(String(100))
    weight_carried = Column(Numeric(5, 2))
    barrier_draw = Column(Integer)
    win_odds = Column(Numeric(6, 2))
    sec1_time = Column(Numeric(6, 2))
    sec2_time = Column(Numeric(6, 2))
    sec3_time = Column(Numeric(6, 2))
    sec4_time = Column(Numeric(6, 2))
    sec5_time = Column(Numeric(6, 2))
    sec6_time = Column(Numeric(6, 2))
    __table_args__ = (UniqueConstraint('race_id', 'horse_id', name='_race_horse_uc'),)

# --- Database Manager ---
class DatabaseManager:
    def __init__(self, db_url: str):
        self.engine = create_async_engine(db_url, echo=False)
        self.async_session = sessionmaker(
            self.engine, expire_on_commit=False, class_=AsyncSession
        )

    async def init_db(self):
        """Creates tables if they don't exist."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            
            # Check if sectional time columns exist and add them if not
            try:
                from sqlalchemy import text
                res = await conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='race_results'"))
                columns = [r[0] for r in res.fetchall()]
                if 'sec1_time' not in columns:
                    logger.info("Adding sectional time columns to database...")
                    for i in range(1, 7):
                        await conn.execute(text(f"ALTER TABLE race_results ADD COLUMN sec{i}_time NUMERIC(6, 2)"))
            except Exception as e:
                logger.warning(f"Could not check/update database schema: {e}")
                
        logger.info("Database tables initialized.")

    async def save_scraped_data(self, race: Dict, horses: List[Dict], results: List[Dict]):
        """Handles the insertion of scraped data into the normalized tables."""
        async with self.async_session() as session:
            async with session.begin():
                # 1. Insert/Get Race
                race_date_obj = datetime.strptime(race['race_date'], '%Y/%m/%d').date()
                stmt = insert(Race).values(
                    track=race['track'],
                    race_date=race_date_obj,
                    race_class=race['race_class'],
                    distance=race['distance']
                ).on_conflict_do_nothing(
                    index_elements=['track', 'race_date', 'distance']
                ).returning(Race.race_id)
                
                result = await session.execute(stmt)
                race_id = result.scalar()
                
                if not race_id:
                    stmt = select(Race.race_id).where(
                        Race.track == race['track'],
                        Race.race_date == race_date_obj,
                        Race.distance == race['distance']
                    )
                    result = await session.execute(stmt)
                    race_id = result.scalar()
                
                # 2. Insert/Get Horses
                horse_ids = {}
                for horse in horses:
                    stmt = insert(Horse).values(
                        name=horse['name'],
                        origin=horse['origin'],
                        age=horse['age']
                    ).on_conflict_do_nothing(
                        index_elements=['name']
                    ).returning(Horse.horse_id)
                    
                    result = await session.execute(stmt)
                    horse_id = result.scalar()
                    
                    if not horse_id:
                        stmt = select(Horse.horse_id).where(Horse.name == horse['name'])
                        result = await session.execute(stmt)
                        horse_id = result.scalar()
                        
                    horse_ids[horse['name']] = horse_id
                
                # 3. Insert Race Results mapping the IDs
                for res in results:
                    stmt = insert(RaceResult).values(
                        race_id=race_id,
                        horse_id=horse_ids[res['horse_name']],
                        finish_position=res['finish_position'],
                        finishing_time=res['finishing_time'],
                        jockey=res['jockey'],
                        trainer=res['trainer'],
                        weight_carried=res['weight_carried'],
                        barrier_draw=res['barrier_draw'],
                        win_odds=res['win_odds'],
                        sec1_time=res.get('sec1_time'),
                        sec2_time=res.get('sec2_time'),
                        sec3_time=res.get('sec3_time'),
                        sec4_time=res.get('sec4_time'),
                        sec5_time=res.get('sec5_time'),
                        sec6_time=res.get('sec6_time')
                    ).on_conflict_do_nothing(
                        index_elements=['race_id', 'horse_id']
                    )
                    await session.execute(stmt)

# --- Scraper ---
class HKJCScraper:
    def __init__(self, base_url: str, proxy_url: str = None):
        self.base_url = base_url
        self.proxy_url = proxy_url
        self.rate_limit_delay = 3.0  # Base seconds to wait between requests
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0"
        ]

    async def scrape_date(self, date_str: str, max_retries: int = 3) -> List[Dict[str, Any]]:
        """Scrapes all race results for a given date with retries."""
        for attempt in range(max_retries):
            try:
                return await self._scrape_date_internal(date_str)
            except PlaywrightTimeoutError:
                logger.warning(f"Timeout error on attempt {attempt + 1}/{max_retries} for {date_str}. Retrying...")
                await asyncio.sleep(random.uniform(5, 10))
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}/{max_retries} for {date_str}: {str(e)}")
                await asyncio.sleep(random.uniform(5, 10))
        
        logger.error(f"Failed to scrape {date_str} after {max_retries} attempts.")
        return []

    async def fetch_sectional_times(self, page, date_str: str, race_no: int) -> Dict[str, Dict[str, float]]:
        """Fetches and parses sectional times for a given race."""
        date_parts = date_str.split('/')
        formatted_date = f"{date_parts[2]}/{date_parts[1]}/{date_parts[0]}"
        url = f"https://racing.hkjc.com/racing/information/English/Racing/DisplaySectionalTime.aspx?RaceDate={formatted_date}&RaceNo={race_no}"
        
        logger.info(f"Fetching sectional times from {url}")
        try:
            delay = self.rate_limit_delay + random.uniform(1.0, 3.0)
            await asyncio.sleep(delay)
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            # Wait for the table to load, but it might not exist if there are no sectional times
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

    async def _scrape_date_internal(self, date_str: str) -> List[Dict[str, Any]]:
        all_races_data = []
        async with async_playwright() as p:
            # Launch browser (headless=True for production)
            # Pass proxy settings to the browser instance
            launch_args = {"headless": True}
            if self.proxy_url:
                launch_args["proxy"] = {"server": self.proxy_url}
                
            browser = await p.chromium.launch(**launch_args)
            
            # Use a realistic user agent to avoid basic bot detection
            user_agent = random.choice(self.user_agents)
            context = await browser.new_context(
                user_agent=user_agent,
                viewport={"width": random.randint(1366, 1920), "height": random.randint(768, 1080)}
            )
            page = await context.new_page()
            await Stealth().apply_stealth_async(page)

            try:
                target_url = f"{self.base_url}?RaceDate={date_str}"
                logger.info(f"Navigating to {target_url}")
                
                # Wait until network is mostly idle to ensure JS has rendered the tables
                # Increased timeout for residential proxies
                await page.goto(target_url, wait_until="domcontentloaded", timeout=60000)
                
                # Check if there is actually race data on this page before waiting for the table
                # If the page says "No information", we can skip it immediately
                html_content = await page.content()
                if "No information" in html_content or "No race meeting" in html_content:
                    logger.info(f"No race data found for {date_str}... Skipping")
                    return []
                
                # Wait for the main results table to load
                try:
                    await page.wait_for_selector('.f_tac.table_bd', timeout=15000)
                except PlaywrightTimeoutError:
                    logger.info(f"No results table found for {date_str}... Skipping")
                    return []
                
                html_content = await page.content()
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Find all race links for the day
                race_links = []
                racecard_table = soup.find('table', class_='f_fs12 js_racecard')
                if racecard_table:
                    for link in racecard_table.find_all('a'):
                        href = link.get('href')
                        if href and 'RaceNo=' in href:
                            race_links.append(href)
                
                # Parse the first race (already loaded)
                logger.info(f"Parsing Race 1 for {date_str}")
                race_data = self.parse_race_html(html_content, date_str, 1)
                if race_data:
                    # Fetch sectional times for Race 1
                    sec_times = await self.fetch_sectional_times(page, date_str, 1)
                    for res in race_data['results']:
                        horse_code = res.get('horse_code')
                        if horse_code and horse_code in sec_times:
                            res.update(sec_times[horse_code])
                        else:
                            res.update({f'sec{i}_time': None for i in range(1, 7)})
                    all_races_data.append(race_data)
                
                # Navigate to and parse subsequent races
                for link in race_links:
                    try:
                        # Extract race number from the link (e.g., ?RaceDate=2020/04/01&RaceNo=2)
                        race_no_match = re.search(r'RaceNo=(\d+)', link)
                        race_no = int(race_no_match.group(1)) if race_no_match else 0
                        
                        full_url = f"https://racing.hkjc.com{link}"
                        logger.info(f"Navigating to {full_url}")
                        # Randomize rate limit delay to mimic human behavior
                        delay = self.rate_limit_delay + random.uniform(1.0, 4.0)
                        await asyncio.sleep(delay)
                        await page.goto(full_url, wait_until="domcontentloaded", timeout=60000)
                        await page.wait_for_selector('.f_tac.table_bd', timeout=30000)
                        
                        html_content = await page.content()
                        race_data = self.parse_race_html(html_content, date_str, race_no)
                        if race_data:
                            # Fetch sectional times for this race
                            sec_times = await self.fetch_sectional_times(page, date_str, race_no)
                            for res in race_data['results']:
                                horse_code = res.get('horse_code')
                                if horse_code and horse_code in sec_times:
                                    res.update(sec_times[horse_code])
                                else:
                                    res.update({f'sec{i}_time': None for i in range(1, 7)})
                            all_races_data.append(race_data)
                    except Exception as e:
                        logger.error(f"Error scraping race {link}: {str(e)}")
                
                logger.info(f"Successfully retrieved {len(all_races_data)} races for {date_str}")

            finally:
                await browser.close()
                
            # --- Rate Limiting ---
            # Crucial for HKJC to prevent IP bans
            delay = self.rate_limit_delay + random.uniform(2.0, 5.0)
            logger.info(f"Sleeping for {delay:.2f}s to respect rate limits...")
            await asyncio.sleep(delay)
            
            return all_races_data

    def parse_race_html(self, html_content: str, date_str: str, race_number: int) -> Dict[str, Any]:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Track
        meeting_info_tag = soup.find(string=lambda text: text and 'Race Meeting' in text)
        if not meeting_info_tag:
            return None
        meeting_info = meeting_info_tag.parent.text.strip()
        track = meeting_info.split('  ')[-1].strip()
        
        # Race Class and Distance
        race_tab = soup.find('div', class_='race_tab')
        if not race_tab:
            return None
        
        # Extract Course Type and Track Condition
        course_type = None
        track_condition = None
        try:
            # Look for the specific td elements containing Course and Going
            course_td = soup.find('td', string=re.compile(r'Course\s*:'))
            if course_td:
                course_type = course_td.find_next_sibling('td').text.strip()
                
            going_td = soup.find('td', string=re.compile(r'Going\s*:'))
            if going_td:
                track_condition = going_td.find_next_sibling('td').text.strip()
        except Exception as e:
            logger.warning(f"Could not parse course/condition for {date_str} Race {race_number}: {e}")

        race_info_str = race_tab.find('tbody').find_all('tr')[1].text.strip()
        race_class = race_info_str.split('-')[0].strip()
        distance_match = re.search(r'(\d+)M', race_info_str)
        distance = int(distance_match.group(1)) if distance_match else 0
        
        race_data = {
            "track": track,
            "race_date": date_str,
            "race_number": race_number,
            "race_class": race_class,
            "distance": distance,
            "course_type": course_type,
            "track_condition": track_condition
        }
        
        horses_data = []
        results_data = []
        
        table = soup.find('table', class_='f_tac table_bd draggable')
        if not table:
            return None
            
        for row in table.find('tbody').find_all('tr'):
            cols = [td.text.strip() for td in row.find_all('td')]
            if len(cols) < 12:
                continue
                
            finish_position = cols[0]
            
            # Extract Horse Number
            horse_number_str = cols[1]
            horse_number = int(horse_number_str) if horse_number_str.isdigit() else None
            
            # Extract Horse Name and Horse Code
            horse_name_raw = cols[2]
            horse_name = re.sub(r'\s*\([^)]*\)', '', horse_name_raw).strip()
            
            # The horse code is usually in parentheses, e.g., "LUCKY STAR (V123)"
            horse_code_match = re.search(r'\(([^)]+)\)', horse_name_raw)
            horse_code = horse_code_match.group(1) if horse_code_match else None
            
            jockey = cols[3]
            trainer = cols[4]
            
            weight_carried = cols[5]
            weight_carried = float(weight_carried) if weight_carried.replace('.', '', 1).isdigit() else None
            
            # Extract Horse Body Weight
            horse_weight_str = cols[6]
            horse_weight = float(horse_weight_str) if horse_weight_str.replace('.', '', 1).isdigit() else None
            
            barrier_draw = cols[7]
            barrier_draw = int(barrier_draw) if barrier_draw.isdigit() else None
            
            finish_time_str = cols[10]
            finishing_time = None
            if finish_time_str and finish_time_str != '-':
                parts = finish_time_str.split(':')
                if len(parts) == 2:
                    finishing_time = float(parts[0]) * 60 + float(parts[1])
                else:
                    try:
                        finishing_time = float(parts[0])
                    except ValueError:
                        pass
                        
            # Extract Running Position
            # The running position is usually in column 9, formatted like "7 3 1"
            running_position_str = cols[9]
            running_position = None
            if running_position_str:
                # Clean up the string to just be space-separated numbers
                running_position = ' '.join(running_position_str.split())
                
            win_odds = cols[11]
            win_odds = float(win_odds) if win_odds.replace('.', '', 1).isdigit() else None
            
            horses_data.append({
                "name": horse_name,
                "code": horse_code,
                "origin": None,
                "age": None
            })
            
            results_data.append({
                "horse_number": horse_number,
                "horse_name": horse_name,
                "horse_code": horse_code,
                "finish_position": finish_position,
                "finishing_time": finishing_time,
                "jockey": jockey,
                "trainer": trainer,
                "weight_carried": weight_carried,
                "horse_weight": horse_weight,
                "barrier_draw": barrier_draw,
                "running_position": running_position,
                "win_odds": win_odds
            })
            
        # Remove horse_number from results_data as it's not needed in the final output
        for result in results_data:
            result.pop('horse_number', None)
            
        return {"race": race_data, "horses": horses_data, "results": results_data}

# --- Orchestrator ---
def generate_race_dates(start_year=2020, end_year=2026):
    start_date = date(start_year, 1, 1)
    end_date = date.today() # Automatically use the current date
    
    dates = []
    current_date = start_date
    while current_date <= end_date:
        # HKJC races are typically on Wednesdays (2), Saturdays (5), and Sundays (6)
        # Skip the "Dead Zone" (mid-July through August)
        if current_date.month == 8 or (current_date.month == 7 and current_date.day > 15):
            current_date += timedelta(days=1)
            continue
            
        if current_date.weekday() in [2, 5, 6]:
            dates.append(current_date.strftime("%Y/%m/%d"))
        current_date += timedelta(days=1)
    return dates

async def main():
    # 1. Initialize DB
    db = DatabaseManager(DB_URL)
    await db.init_db() # Uncomment when DB is running
    
    # 2. Initialize Scraper
    # If you have a residential proxy, replace the proxy_url below.
    # Example: proxy_url="http://username:password@proxy.example.com:8080"
    # If you are using a local proxy like Clash/V2Ray, use "http://127.0.0.1:7890"
    scraper = HKJCScraper(
        base_url="https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx",
        proxy_url="http://127.0.0.1:7890" # Update or set to None if not using a proxy
    )
    
    # 3. Define dates to scrape (e.g., generating a list of historical Wednesdays/Weekends)
    all_dates = generate_race_dates(2020, 2026)
    
    # 4. Check DB for already scraped dates to avoid re-scraping
    async with db.async_session() as session:
        result = await session.execute(select(Race.race_date).distinct())
        scraped_dates = {row[0].strftime("%Y/%m/%d") for row in result.fetchall()}
    
    # Create a directory for CSV backups
    csv_dir = "data/raw_csvs"
    os.makedirs(csv_dir, exist_ok=True)
    
    # Check the last CSV file to determine resuming date
    last_csv_date = None
    all_csvs = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    
    if all_csvs:
        # Sort files to find the latest date
        # Filename format YYYY-MM-DD.csv sorts chronologically correctly
        all_csvs.sort()
        last_csv_filename = all_csvs[-1]
        try:
            last_csv_date_str = last_csv_filename.split('.')[0].replace('-', '/')
            last_csv_date = datetime.strptime(last_csv_date_str, "%Y/%m/%d").date()
            logger.info(f"Examples of CSVs found: {all_csvs[:3]} ... {all_csvs[-3:]}")
            logger.info(f"Last scraped date from CSV: {last_csv_date_str}")
        except ValueError:
            logger.warning(f"Could not parse date from filename: {last_csv_filename}")
    
    # Filter dates
    dates_to_scrape = []
    
    # Convert all_dates to date objects for comparison
    for d_str in all_dates:
        d_obj = datetime.strptime(d_str, "%Y/%m/%d").date()
        
        # Condition 1: Not in DB
        if d_str in scraped_dates:
            continue
            
        # Condition 2: If we have a last CSV date, only scrape dates AFTER that date
        if last_csv_date and d_obj <= last_csv_date:
            continue
            
        dates_to_scrape.append(d_str)

    logger.info(f"Found {len(dates_to_scrape)} new dates to scrape out of {len(all_dates)} total possible dates.")
    
    for date_str in dates_to_scrape:
        scraped_payloads = await scraper.scrape_date(date_str)
        
        if scraped_payloads:
            # Save to Database
            for payload in scraped_payloads:
                await db.save_scraped_data(**payload)
            logger.info(f"Data saved to DB for {date_str}")
            
            # Save to CSV Backup
            flattened_data = []
            for payload in scraped_payloads:
                race_info = payload['race']
                for result in payload['results']:
                    row = {**race_info, **result}
                    flattened_data.append(row)
            
            if flattened_data:
                csv_filename = os.path.join(csv_dir, f"{date_str.replace('/', '-')}.csv")
                fieldnames = list(flattened_data[0].keys())
                with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(flattened_data)
                logger.info(f"Data saved to CSV backup: {csv_filename}")

if __name__ == "__main__":
    asyncio.run(main())
