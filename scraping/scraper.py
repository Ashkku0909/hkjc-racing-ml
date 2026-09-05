import asyncio
import logging
import random
import os
import csv
import json
from typing import List, Dict, Any
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from playwright_stealth import Stealth
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, Date, Numeric, ForeignKey, select, UniqueConstraint, Text
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

# Modern HKJC race card data is served from this GraphQL endpoint
GRAPHQL_BASE_URL = "https://info.cld.hkjc.com/graphql/base/"

# Exact query copied from the HKJC race card SPA bundle (schema-valid).
# Runners expose currentRating / gearInfo / handicapWeight; horse codes appear
# embedded in horse.name_en (e.g. 'GOLDEN SIXTY (C834)') for local runners.
RACECARD_PROFILE_QUERY = """
query RaceCardProfile($date: String, $venueCode: String, $type: STStatType, $ids: [String!], $raceNumber: String, $meetingDate: String) {
  raceMeetingProfile(date: $date, venueCode: $venueCode) {
    totalNumberOfRace
    status
    pmPools {
      leg {
        races
      }
      status
      oddsType
    }
    races {
      id
      no
      status
      postTime
      raceName_en
      raceName_ch
      raceResults {
        status
      }
      countryCodeNm {
        code
        english
        chinese
      }
      distance
      raceCourse {
        code
        description_en
        description_ch
      }
      raceTrack {
        code
        description_en
        description_ch
      }
      raceType_en
      raceType_ch
      raceClass_en
      raceClass_ch
      country_en
      country_ch
      winningMargin {
        seqNo
        lbw
      }
      go_en
      go_ch
      remarks {
        name_en
        name_ch
        seqNo
      }
      runners {
        horse {
          name_en
          name_ch
          id
        }
        status
        color
        no
        handicapWeight
        jockey {
          code
          name_en
          name_ch
        }
        trainer {
          code
          name_en
          name_ch
        }
        id
        last6run
        internationalRating
        currentRating
        sire
        sexNm {
          chinese
          english
          code
        }
        age
        barrierDrawNumber
        gearInfo
        stat(type: $type) {
          statType
          numStarts
          numFirst
          numSecond
          numThird
        }
        damNm {
          code
          chinese
          english
        }
        sireOfDamNm {
          code
          chinese
          english
        }
        ownerNm {
          code
          chinese
          english
        }
        colorNm {
          code
          chinese
          english
        }
      }
    }
    date
    venueCode
  }

  simulcastHorse(ids: $ids, raceNumber: $raceNumber, meetingDate: $meetingDate, venCode: $venueCode) {
    id
    brandNumber
    earings
    performanceStats {
      type
      firstPlace
      secondPlace
      thirdPlace
      totalRun
      ssn
    }
    horseFormRecord {
      videoReplayLink {
        chinese
        code
        english
      }
    }
  }
}
"""

# --- SQLAlchemy Models ---
class Race(Base):
    __tablename__ = 'races'
    race_id = Column(Integer, primary_key=True, autoincrement=True)
    track = Column(String(50), nullable=False)
    race_date = Column(Date, nullable=False)
    race_class = Column(String(20))
    distance = Column(Integer, nullable=False)
    place_dividends = Column(Text)          # PLA pool dividend line(s)
    quinella_dividend = Column(Numeric(6, 2))  # QIN dividend
    qpl_dividends = Column(Text)            # QPL pair dividends
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
    horse_rating = Column(Integer)            # Official HKJC rating (None for debutants)
    rating_change = Column(Integer)           # Rating delta vs previous start
    gear = Column(String(50))                 # Equipment, e.g. 'B', 'TT', 'V', 'B/TT', '--'
    jockey_allowance = Column(Integer)        # Apprentice claim, e.g. -10, -5, -2, 0
    incident_report = Column(Text)            # Stewards' incident paragraph(s) for this horse
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
            
            # Check for missing columns (added in later versions) and add them if not
            try:
                from sqlalchemy import text
                res = await conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='race_results'"))
                columns = [r[0] for r in res.fetchall()]
                new_columns = {
                    **{f'sec{i}_time': 'NUMERIC(6, 2)' for i in range(1, 7)},
                    'horse_rating': 'INTEGER',
                    'rating_change': 'INTEGER',
                    'gear': 'VARCHAR(50)',
                    'jockey_allowance': 'INTEGER',
                    'incident_report': 'TEXT',
                }
                for col_name, col_type in new_columns.items():
                    if col_name not in columns:
                        logger.info(f"Adding missing column '{col_name}' to race_results table...")
                        await conn.execute(text(f"ALTER TABLE race_results ADD COLUMN {col_name} {col_type}"))
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
                    distance=race['distance'],
                    place_dividends=race.get('place_dividends'),
                    quinella_dividend=race.get('quinella_dividend'),
                    qpl_dividends=race.get('qpl_dividends')
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
                        sec6_time=res.get('sec6_time'),
                        horse_rating=res.get('horse_rating'),
                        rating_change=res.get('rating_change'),
                        gear=res.get('gear'),
                        jockey_allowance=res.get('jockey_allowance'),
                        incident_report=res.get('incident_report')
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

    async def intercept_hkjc_api(self, page, target_url: str, replay_vars: Dict[str, Any] = None, wait_seconds: int = 12) -> Dict[str, Any]:
        """Navigates to target_url while intercepting HKJC racing JSON API responses.

        The modern HKJC site loads race card / horse detail data through background
        GraphQL requests. If the SPA never fires the race card query (e.g. the race
        already ran), the query is actively replayed against the GraphQL endpoint
        using the exact query text from the SPA bundle.

        Returns {url: json_body} for every intercepted/replayed JSON payload.
        """
        captured: Dict[str, Any] = {}

        async def handle_response(response):
            try:
                url = response.url
                is_hkjc_data = ('hkjc.com' in url) and (
                    'graphql' in url or 'content-api' in url
                    or ('/racing/' in url and '.json' in url.lower())
                )
                if not is_hkjc_data:
                    return
                content_type = response.headers.get('content-type', '')
                if response.status == 200 and 'application/json' in content_type:
                    json_body = await response.json()
                    captured[url] = json_body
            except Exception:
                pass

        page.on('response', handle_response)
        try:
            await page.goto(target_url, wait_until='domcontentloaded', timeout=60000)
            # Poll while the SPA fires its GraphQL queries; stop early once race
            # card data (raceMeetingProfile) has been captured.
            for _ in range(max(1, wait_seconds // 3)):
                await page.wait_for_timeout(3000)
                try:
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                except Exception:
                    pass
                if any('raceMeetingProfile' in json.dumps(b.get('data', {}))
                       for b in captured.values() if isinstance(b, dict)):
                    break
        finally:
            page.remove_listener('response', handle_response)

        # Active replay fallback: query the GraphQL endpoint directly.
        # Only used when passive interception did not capture race card data.
        has_racecard = any('raceMeetingProfile' in json.dumps(b.get('data', {}))
                           for b in captured.values() if isinstance(b, dict))
        if replay_vars and not has_racecard:
            try:
                api = page.context.request
                replay_resp = await api.post(
                    GRAPHQL_BASE_URL,
                    headers={'content-type': 'application/json'},
                    data=json.dumps({'variables': replay_vars, 'query': RACECARD_PROFILE_QUERY}),
                )
                replay_body = await replay_resp.json()
                if isinstance(replay_body, dict) and 'data' in replay_body:
                    captured[f'{GRAPHQL_BASE_URL}<replayed:RaceCardProfile>'] = replay_body
            except Exception as e:
                logger.warning(f"GraphQL replay failed: {e}")

        return captured

    @staticmethod
    def _extract_horse_code(runner: Dict[str, Any]) -> str:
        """Best-effort extraction of the HKJC horse code (e.g. 'K152') from a runner dict."""
        horse = runner.get('horse') or {}
        for source in [horse.get('code'), horse.get('name_en'), horse.get('name_ch'), runner.get('id')]:
            if source:
                match = re.search(r'[A-Z]\d{3}', str(source).upper())
                if match:
                    return match.group(0)
        return None

    def parse_graphql_racecard(self, payloads: Dict[str, Any], target_race_no: int = None) -> Dict[str, Dict[str, Any]]:
        """Maps intercepted GraphQL payloads into {horse_code: {horse_rating, gear, jockey_allowance}}.

        Handles both local runners (with official HKJC horse codes) and simulcast
        runners (which lack HK codes and are skipped).
        """
        details: Dict[str, Dict[str, Any]] = {}
        for body in payloads.values():
            data = body.get('data') if isinstance(body, dict) else None
            if not isinstance(data, dict):
                continue
            for profile in data.get('raceMeetingProfile', []) or []:
                for race in profile.get('races', []) or []:
                    try:
                        race_no = int(race.get('no') or 0)
                    except (TypeError, ValueError):
                        race_no = 0
                    if target_race_no and race_no != target_race_no:
                        continue
                    for runner in race.get('runners', []) or []:
                        horse_code = self._extract_horse_code(runner)
                        if not horse_code:
                            continue

                        # Official rating (GraphQL zero-pads it, e.g. '060')
                        rating_raw = runner.get('currentRating')
                        horse_rating = None
                        if rating_raw not in (None, ''):
                            try:
                                horse_rating = int(str(rating_raw))
                            except ValueError:
                                horse_rating = None

                        gear = runner.get('gearInfo') or None

                        jockey = runner.get('jockey') or {}
                        jockey_allowance = 0
                        allowance_raw = jockey.get('allowance')
                        if allowance_raw not in (None, ''):
                            try:
                                jockey_allowance = int(allowance_raw)
                            except (TypeError, ValueError):
                                jockey_allowance = 0
                        else:
                            # Fallback: claims sometimes embedded in the jockey name, e.g. 'M F Poon (-5)'
                            name_match = re.search(r'\(([+-]?\d+)\)', str(jockey.get('name_en') or ''))
                            if name_match:
                                jockey_allowance = int(name_match.group(1))

                        details[horse_code] = {
                            'horse_rating': horse_rating,
                            'gear': gear,
                            'jockey_allowance': jockey_allowance,
                        }
        return details

    async def fetch_racecard_details(self, page, date_str: str, race_no: int, venue_code: str = 'ST') -> Dict[str, Dict[str, Any]]:
        """Fetches race card details (rating, gear, jockey allowance).

        Strategy:
          1. Intercept the modern site's GraphQL JSON (works for upcoming/current meetings).
          2. Fall back to classic race card pages parsed from HTML.
        """
        date_parts = date_str.split('/')
        formatted_date = f"{date_parts[2]}/{date_parts[1]}/{date_parts[0]}"
        iso_date = f"{date_parts[0]}-{date_parts[1]}-{date_parts[2]}"

        # --- 1. Modern approach: intercept GraphQL traffic on the new race card page ---
        modern_url = (f"https://racing.hkjc.com/en-us/local/information/racecard"
                      f"?racedate={iso_date}&Racecourse={venue_code}&RaceNo={race_no}")
        replay_vars = {
            'date': iso_date,
            'venueCode': venue_code,
            'type': 'LIEF_TIME',
            'ids': [],
            'raceNumber': str(race_no),
            'meetingDate': date_str.replace('/', ''),
            'venCode': venue_code,
        }
        try:
            delay = self.rate_limit_delay + random.uniform(1.0, 3.0)
            await asyncio.sleep(delay)
            logger.info(f"Intercepting race card API traffic from {modern_url}")
            payloads = await self.intercept_hkjc_api(page, modern_url, replay_vars=replay_vars)
            if payloads:
                details = self.parse_graphql_racecard(payloads, target_race_no=race_no)
                if details:
                    logger.info(f"Extracted {len(details)} horse(s) from intercepted GraphQL payloads")
                    return details
        except Exception as e:
            logger.error(f"GraphQL interception failed for {date_str} Race {race_no}: {e}")

        # --- 2. Fallback: classic race card pages (mostly decommissioned) ---
        urls = [
            f"https://racing.hkjc.com/racing/information/English/Racing/LocalRaceCard.aspx?RaceDate={formatted_date}&RaceNo={race_no}",
            f"https://racing.hkjc.com/racing/information/en-us/Content/racing.aspx?RaceDate={iso_date}&RaceNo={race_no}",
        ]
        for url in urls:
            try:
                delay = self.rate_limit_delay + random.uniform(1.0, 3.0)
                await asyncio.sleep(delay)
                logger.info(f"Fetching race card from {url}")
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                await page.wait_for_timeout(3000)  # allow JS rendering
                html_content = await page.content()
                details = self.parse_racecard_html(html_content)
                if details:
                    return details
            except Exception as e:
                logger.error(f"Error fetching race card from {url}: {e}")

        logger.info(f"No race card details found for {date_str} Race {race_no}")
        return {}

    def parse_racecard_html(self, html_content: str) -> Dict[str, Dict[str, Any]]:
        """Parses the race card HTML into {horse_code: {horse_rating, gear, jockey_allowance}}."""
        soup = BeautifulSoup(html_content, 'html.parser')
        details: Dict[str, Dict[str, Any]] = {}

        table = soup.find('table', class_='f_tac table_bd')
        if not table:
            return details

        # Locate columns by header text (robust against column order changes)
        header_cells = []
        header_row = None
        for row in table.find_all('tr'):
            cells = [td.text.strip().lower() for td in row.find_all(['th', 'td'])]
            if any('rating' in c or 'gear' in c or 'allow' in c or 'wt.' in c for c in cells):
                header_cells = cells
                header_row = row
                break

        col_map = {}
        for idx, txt in enumerate(header_cells):
            if txt == 'rating':
                col_map['rating'] = idx
            elif 'gear' in txt:
                col_map['gear'] = idx
            elif 'allow' in txt or 'claim' in txt:
                col_map['allowance'] = idx

        # Parse the class rating band from the race title, e.g. "CLASS 4 - 1200M - (60-40)"
        band_match = re.search(r'\((\d+)\s*-\s*(\d+)\)', soup.get_text(' ', strip=True))
        if band_match:
            details['__race__'] = {'class_max_rating': max(int(band_match.group(1)), int(band_match.group(2)))}

        def _get(row, idx):
            cols = row.find_all(['th', 'td'])
            return cols[idx].text.strip() if idx is not None and len(cols) > idx else ''

        rows = table.find_all('tr')
        if header_row is not None:
            rows = rows[rows.index(header_row) + 1:]
        for row in rows:
            cells = row.find_all(['th', 'td'])
            if not cells:
                continue
            horse_cell = cells[1].text.strip() if len(cells) > 1 else ''
            code_match = re.search(r'\(([A-Z]\d{3})\)', horse_cell)
            horse_code = code_match.group(1) if code_match else None
            if not horse_code:
                continue

            rating_raw = _get(row, col_map.get('rating'))
            gear_raw = _get(row, col_map.get('gear'))
            allowance_raw = _get(row, col_map.get('allowance'))

            # Allowance parsing depends on the column type:
            #  - combined "Wt. / Allowance" column -> only parenthesized values, e.g. "128 (-5)"
            #  - dedicated "Allowance" column -> the whole cell, e.g. "-5"
            jockey_allowance = 0
            allow_header = header_cells[col_map['allowance']] if 'allowance' in col_map else ''
            if allowance_raw:
                paren_match = re.search(r'\(([+-]?\d+)\)', allowance_raw)
                if paren_match:
                    jockey_allowance = int(paren_match.group(1))
                elif 'wt' not in allow_header:
                    bare_match = re.search(r'([+-]?\d+)', allowance_raw.strip())
                    if bare_match:
                        jockey_allowance = int(bare_match.group(1))
            if jockey_allowance == 0 and len(cells) > 2:
                # Last resort: apprentice claims shown as "( -5 )" in the jockey cell
                jockey_cell = cells[2].text.strip()
                jockey_cell_match = re.search(r'\(([+-]?\d+)\)', jockey_cell)
                if jockey_cell_match:
                    jockey_allowance = int(jockey_cell_match.group(1))

            details[horse_code] = {
                'horse_rating': int(float(rating_raw)) if rating_raw.replace('.', '', 1).isdigit() else None,
                'gear': gear_raw if gear_raw else None,
                'jockey_allowance': jockey_allowance,
            }

        # Fallback: classic 'gear' table (table.gear) lists horse -> equipment
        for gear_table in soup.find_all('table', class_='gear'):
            for g_row in gear_table.find_all('tr'):
                g_cells = [td.text.strip() for td in g_row.find_all(['th', 'td'])]
                if len(g_cells) >= 2 and 'gear' not in g_cells[0].lower():
                    name_cell = g_cells[0]
                    g_match = re.search(r'\(([A-Z]\d{3})\)', name_cell)
                    g_code = g_match.group(1) if g_match else None
                    if g_code and g_code not in details:
                        details[g_code] = {'horse_rating': None, 'gear': g_cells[1] or None, 'jockey_allowance': 0}
                    elif g_code:
                        details[g_code]['gear'] = g_cells[1] or None

        return details

    def parse_incident_report(self, html_content: str, horse_map: Dict[str, str]) -> Dict[str, str]:
        """Parses the Racing Incident Report section and assigns paragraphs to horse codes.

        The heading may appear both in the site navigation and in the race content;
        the container that mentions the most horse codes is selected.

        horse_map: {horse_code: horse_name} of runners in the race.
        Returns: {horse_code: incident_text}
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        code_re = re.compile(r'\(([A-Z]\d{3})\)')

        # Collect every container that carries an incident heading
        headings = soup.find_all(string=re.compile(r'Racing\s*Incident', re.I))
        containers = []
        for heading in headings:
            container = heading.find_parent(['div', 'section', 'table'])
            if container is None:
                continue
            raw_text = container.get_text('\n')
            n_codes = len(code_re.findall(raw_text))
            n_names = sum(1 for name in horse_map.values()
                          if name and re.search(rf'\b{re.escape(name)}\b', raw_text, re.I))
            containers.append((n_codes + n_names, container))

        if not containers:
            # Fallback: use any element whose id mentions incident
            incident_el = soup.find(id=re.compile(r'incident', re.I))
            if incident_el is not None:
                containers.append((1, incident_el))

        if not containers:
            return {}

        # Pick the container with the strongest horse association (skip the nav bar)
        _, container = max(containers, key=lambda item: item[0])
        section_text = container.get_text('\n')
        lines = [ln.strip() for ln in section_text.split('\n') if len(ln.strip()) > 20]

        incidents: Dict[str, str] = {}
        for line in lines:
            codes = code_re.findall(line)
            if not codes:
                # Match full horse names on word boundaries (longest names first)
                for code, name in sorted(horse_map.items(), key=lambda kv: -len(kv[1] or '')):
                    if name and re.search(rf'\b{re.escape(name)}\b', line, re.I):
                        codes.append(code)
            for code in set(codes):
                incidents[code] = (incidents.get(code, '') + ' ' + line).strip()
        return incidents

    def parse_dividends(self, html_content: str) -> Dict[str, Any]:
        """Parses the Dividends section (WIN/PLA/QIN/QPL/...) from a race results page.

        Handles BOTH layouts:
          - classic pages: dividend `<table>` elements
          - new-site pages: a div-based block whose text runs
            `Dividend | Pool | Winning Combination | Dividend (HK$) | WIN | 1 | 60.50 | ...`

        Returns {'place_dividends': str, 'quinella_dividend': float|None,
                 'qpl_dividends': str, 'win_dividend': float|None,
                 'all_dividends': list of (pool, combo, dividend)}.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        result = {'place_dividends': '', 'quinella_dividend': None, 'qpl_dividends': '',
                  'win_dividend': None, 'all_dividends': []}

        # --- Primary: token state machine over the rendered text ---
        # Works for both the classic table and the new-site table (`table_bd f_tac f_fs13 f_fl`)
        # whose rows read: Dividend | Pool | Winning Combination | Dividend (HK$) | WIN | 1 | 60.50 | ...
        text = soup.get_text(' ', strip=True)
        idx = text.lower().find('dividend')
        if idx >= 0:
            section = text[idx:]
            note_idx = section.lower().find('dividend note')
            if note_idx >= 0:
                section = section[:note_idx]

            tokens = section.split()
            # Normalize multi-word pool names: 'QUINELLA PLACE' -> QPL, 'FIRST 4' -> FIRST4
            normalized = []
            i = 0
            while i < len(tokens):
                t = tokens[i]
                if t.upper() == 'QUINELLA' and i + 1 < len(tokens) and tokens[i + 1].upper() == 'PLACE':
                    normalized.append('QPL')
                    i += 2
                    continue
                if t.upper() == 'FIRST' and i + 1 < len(tokens) and tokens[i + 1] == '4':
                    normalized.append('FIRST4')
                    i += 2
                    continue
                normalized.append(t)
                i += 1

            pool_counts = {'WIN': 1, 'PLACE': 3, 'QUINELLA': 1, 'QPL': 3,
                           'FORECAST': 1, 'TIERCE': 1, 'TRIO': 1, 'FIRST4': 1, 'QUARTET': 1}
            i = 0
            while i < len(normalized):
                t = normalized[i].upper()
                if t in pool_counts:
                    pool = t
                    need = pool_counts[t]
                    i += 1
                    for _ in range(need):
                        if i >= len(normalized):
                            break
                        combo = normalized[i]
                        if not re.search(r'\d', combo):
                            break
                        i += 1
                        if i >= len(normalized):
                            break
                        amount = normalized[i]
                        if not re.fullmatch(r'[\d,]+(?:\.\d+)?', amount):
                            break
                        i += 1
                        div = float(amount.replace(',', ''))
                        result['all_dividends'].append((pool, combo, div))
                        if pool == 'WIN':
                            result['win_dividend'] = div
                        elif pool == 'QUINELLA':
                            result['quinella_dividend'] = div
                else:
                    i += 1

        # --- Fallback: regex extraction from classic dividend tables ---
        if not result['all_dividends']:
            def table_text(t):
                return ' | '.join(' '.join(r.get_text(' ', strip=True).split()) for r in t.find_all('tr'))

            candidates = []
            for t in soup.find_all('table'):
                cls = ' '.join(t.get('class') or [])
                txt = table_text(t)
                low = (cls + ' ' + txt).lower()
                if 'dividend' in low or 'quinella' in low or ('pla' in txt and 'qin' in txt):
                    candidates.append(txt)
            if candidates:
                block = ' || '.join(candidates)
                for m in re.finditer(r'PLA[\s:]*([\d./\s|-]+?)(?:\|\||$)', block, re.I):
                    result['place_dividends'] = m.group(1).strip()[:300]
                    break
                for m in re.finditer(r'QIN[\s:]*([\d./\s|-]+?)(?:\|\||QPL)', block, re.I):
                    num = re.search(r'(\d+\.\d+)', m.group(1))
                    if num:
                        result['quinella_dividend'] = float(num.group(1))
                    break
                for m in re.finditer(r'QPL[\s:]*([\d./\s|-]+?)(?:\|\||$)', block, re.I):
                    result['qpl_dividends'] = m.group(1).strip()[:300]
                    break

        pla = [f"{c}:{d}" for p, c, d in result['all_dividends'] if p == 'PLACE']
        qpl = [f"{c}:{d}" for p, c, d in result['all_dividends'] if p == 'QPL']
        result['place_dividends'] = ', '.join(pla)
        result['qpl_dividends'] = ', '.join(qpl)
        return result

    async def _attach_race_extras(self, page, date_str: str, race_no: int, race_data: Dict, results_html: str = None) -> Dict:
        """Attaches sectional times, race card details (rating/gear/allowance) and the
        incident report text to every result record of a race."""
        incidents = {}
        if results_html:
            horse_map = {r.get('horse_code'): r.get('horse_name') for r in race_data['results']}
            incidents = self.parse_incident_report(results_html, horse_map)
            # Attach dividend data (PLA / QIN / QPL) to the race record
            race_data['race'].update(self.parse_dividends(results_html))

        sec_times = await self.fetch_sectional_times(page, date_str, race_no)

        # Venue code for the modern API (ST = Sha Tin, HV = Happy Valley)
        track = (race_data['race'].get('track') or '').lower()
        venue_code = 'HV' if 'happy valley' in track else 'ST'
        card_details = await self.fetch_racecard_details(page, date_str, race_no, venue_code=venue_code)

        # Use the race card's class band if the results page did not provide one
        if race_data['race'].get('class_max_rating') in (None, 0) and '__race__' in card_details:
            race_data['race']['class_max_rating'] = card_details['__race__'].get('class_max_rating')
        card_details.pop('__race__', None)

        for res in race_data['results']:
            horse_code = res.get('horse_code')
            if horse_code and horse_code in sec_times:
                res.update(sec_times[horse_code])
            else:
                res.update({f'sec{i}_time': None for i in range(1, 7)})

            if horse_code and horse_code in card_details:
                res.update(card_details[horse_code])
            else:
                res.update({'horse_rating': None, 'gear': None, 'jockey_allowance': 0})

            res['incident_report'] = incidents.get(horse_code) or res.get('incident_report', '')

        return race_data

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

                # Give the JS-rendered incident report section a moment to appear
                await page.wait_for_timeout(2500)
                
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
                    # Fetch sectional times, race card details and incident report for Race 1
                    race_data = await self._attach_race_extras(page, date_str, 1, race_data, results_html=html_content)
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
                        # Give the JS-rendered incident report section a moment to appear
                        await page.wait_for_timeout(2500)
                        
                        html_content = await page.content()
                        race_data = self.parse_race_html(html_content, date_str, race_no)
                        if race_data:
                            # Fetch sectional times, race card details and incident report for this race
                            race_data = await self._attach_race_extras(page, date_str, race_no, race_data, results_html=html_content)
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

        # Class rating band, e.g. "CLASS 4 - 1200M - (60-40)" -> max benchmark = 60
        class_max_rating = None
        band_match = re.search(r'\((\d+)\s*-\s*(\d+)\)', race_info_str)
        if band_match:
            class_max_rating = max(int(band_match.group(1)), int(band_match.group(2)))
        
        race_data = {
            "track": track,
            "race_date": date_str,
            "race_number": race_number,
            "race_class": race_class,
            "distance": distance,
            "course_type": course_type,
            "track_condition": track_condition,
            "class_max_rating": class_max_rating,
            "place_dividends": "",
            "quinella_dividend": None,
            "qpl_dividends": ""
        }
        
        horses_data = []
        results_data = []
        
        table = soup.find('table', class_='f_tac table_bd draggable')
        if not table:
            return None

        # Detect optional columns (Rating / Gear / Allowance) by header text
        extra_col_map = {}
        try:
            header_row = None
            for row in table.find_all('tr'):
                cells = [td.text.strip().lower() for td in row.find_all(['th', 'td'])]
                if any('rating' in c or 'gear' in c or 'allow' in c for c in cells):
                    header_row = row
                    for idx, txt in enumerate(cells):
                        if txt == 'rating':
                            extra_col_map['horse_rating'] = idx
                        elif 'gear' in txt:
                            extra_col_map['gear'] = idx
                        elif 'allow' in txt or 'claim' in txt:
                            extra_col_map['jockey_allowance'] = idx
                    break
        except Exception:
            pass

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

            # Apprentice claim allowance, e.g. jockey cell "M F Poon (-5)"
            jockey_allowance = 0
            if 'jockey_allowance' in extra_col_map and len(cols) > extra_col_map['jockey_allowance']:
                allow_raw = cols[extra_col_map['jockey_allowance']]
                allow_digits = re.search(r'([+-]?\d+)', allow_raw)
                jockey_allowance = int(allow_digits.group(1)) if allow_digits else 0
            else:
                allow_match = re.search(r'\(([+-]?\d+)\)', jockey)
                if allow_match:
                    jockey_allowance = int(allow_match.group(1))

            # Official rating (empty for debutants / international races)
            horse_rating = None
            if 'horse_rating' in extra_col_map and len(cols) > extra_col_map['horse_rating']:
                rating_raw = cols[extra_col_map['horse_rating']]
                horse_rating = int(float(rating_raw)) if rating_raw.replace('.', '', 1).isdigit() else None

            # Gear / equipment, e.g. "B", "TT", "V", "B/TT", "--"
            gear = None
            if 'gear' in extra_col_map and len(cols) > extra_col_map['gear']:
                gear_raw = cols[extra_col_map['gear']]
                gear = gear_raw if gear_raw else None
            
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
                "win_odds": win_odds,
                "horse_rating": horse_rating,
                "gear": gear,
                "jockey_allowance": jockey_allowance,
                "incident_report": ""  # filled by parse_incident_report in _attach_race_extras
            })
            
        # NOTE: horse_number is intentionally KEPT (used to map dividend combos
        # from horse numbers to horse codes for data/historical_dividends.csv).

        return {"race": race_data, "horses": horses_data, "results": results_data}

# --- Dividend store integration (real settled payouts) ---
DIVIDEND_STORE_CSV = "data/historical_dividends.csv"
DIVIDEND_STORE_FIELDS = ['race_id', 'race_date', 'venue', 'race_no', 'pool', 'combo',
                         'combo_codes', 'dividend']


def append_dividend_store(payload: Dict[str, Any], out_path: str = DIVIDEND_STORE_CSV) -> int:
    """Writes granular per-combination dividend rows for one race payload into
    the historical dividend store (resume/dedupe-safe). Returns rows written.

    Uses parse_dividends' 'all_dividends' list (already attached to the race
    dict by _attach_race_extras) plus the horse-number -> horse-code map.
    """
    race = payload.get('race') or {}
    all_divs = race.get('all_dividends') or []
    if not all_divs:
        return 0

    num2code = {str(r.get('horse_number')): r.get('horse_code')
                for r in payload.get('results', []) if r.get('horse_number') and r.get('horse_code')}
    track = str(race.get('track', '')).lower()
    venue = 'HV' if 'happy valley' in track else 'ST'
    date_str = str(race.get('race_date', '')).replace('/', '-')
    race_id = f"{date_str}_Race{race.get('race_number')}"

    existing = set()
    if os.path.exists(out_path):
        with open(out_path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row.get('race_id') == race_id:
                    existing.add((row.get('pool'), row.get('combo')))

    rows = []
    for pool, combo, dividend in all_divs:
        if (pool, combo) in existing:
            continue
        numbers = [p for p in re.split(r'[,\s]', combo) if p.isdigit()]
        codes = [num2code.get(n, n) for n in numbers]
        rows.append({
            'race_id': race_id,
            'race_date': date_str,
            'venue': venue,
            'race_no': race.get('race_number'),
            'pool': pool,
            'combo': combo,
            'combo_codes': '/'.join(codes),
            'dividend': dividend,
        })
    if not rows:
        return 0

    exists = os.path.exists(out_path)
    with open(out_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=DIVIDEND_STORE_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    logger.info("Appended %d dividend row(s) for %s to %s", len(rows), race_id, out_path)
    return len(rows)
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
    # 1. Initialize DB (optional: CSV + dividend store work without it)
    db = DatabaseManager(DB_URL)
    db_available = True
    try:
        await db.init_db()
    except Exception as e:
        logger.warning("PostgreSQL unavailable (%s) - continuing CSV-only.", e)
        db_available = False
    
    # 2. Initialize Scraper
    # Proxy is optional: set HKJC_PROXY=http://127.0.0.1:7890 to use a local proxy
    scraper = HKJCScraper(
        base_url="https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx",
        proxy_url=os.getenv("HKJC_PROXY") or None
    )
    
    # 3. Define dates to scrape (e.g., generating a list of historical Wednesdays/Weekends)
    all_dates = generate_race_dates(2020, 2026)
    
    # 4. Check DB for already scraped dates to avoid re-scraping (optional)
    scraped_dates = set()
    if db_available:
        try:
            async with db.async_session() as session:
                result = await session.execute(select(Race.race_date).distinct())
                scraped_dates = {row[0].strftime("%Y/%m/%d") for row in result.fetchall()}
        except Exception as e:
            logger.warning("Could not read scraped dates from DB (%s) - falling back to CSV resume.", e)
            db_available = False
    
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
            # Save to Database (optional)
            if db_available:
                for payload in scraped_payloads:
                    try:
                        await db.save_scraped_data(**payload)
                    except Exception as e:
                        logger.warning("DB save failed for %s (%s) - continuing CSV-only.", date_str, e)
                        db_available = False
                        break
                if db_available:
                    logger.info(f"Data saved to DB for {date_str}")
            # Keep the granular dividend store fresh (real settled payouts) — always
            for payload in scraped_payloads:
                append_dividend_store(payload)
            
            # Save to CSV Backup (scalar fields only)
            flattened_data = []
            for payload in scraped_payloads:
                race_info = payload['race']
                for result in payload['results']:
                    row = {**race_info, **result}
                    # exclude non-scalar race-level values (e.g. all_dividends list)
                    row = {k: v for k, v in row.items() if not isinstance(v, (list, dict))}
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
