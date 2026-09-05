import asyncio
import csv
import os
import re
import threading
import time
import json
import gzip
import urllib.request
from typing import Optional, Tuple
import pandas as pd
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from datetime import datetime
from cachetools import TTLCache
from async_lru import alru_cache

from scraping.scraper import GRAPHQL_BASE_URL, RACECARD_PROFILE_QUERY

# Race header metadata parsed from the WP page (used for the terminal countdown).
# key = (date_str, venue, race_num) -> {'title', 'post_hhmm', 'fetched_at'}
RACE_META: dict = {}

# Cache for formguide which doesn't change often
_formguide_cache = TTLCache(maxsize=100, ttl=3600)  # 1 hour cache

# =====================================================================
# In-memory live odds snapshot cache (smart money flow engine)
# =====================================================================
# Tracks odds snapshots per (race_date, venue, race_no) over time so the
# smart-money scorer can compare current odds against a baseline scraped
# >= BASELINE_LEAD_SECONDS before the jump.
_odds_snapshots = {}          # key -> [{'timestamp': float, 'df': DataFrame}]
_snapshot_lock = threading.Lock()
SNAPSHOT_TTL_SECONDS = 4 * 3600   # prune snapshots older than 4 hours
BASELINE_LEAD_SECONDS = 900       # baseline must be >= 15m old to be "mature"


def _snapshot_key(date_str, venue, race_num):
    return (str(date_str), str(venue), int(race_num))


def record_odds_snapshot(date_str, venue, race_num, df):
    """Stores an immutable copy of a polled odds frame with its wall-clock
    timestamp. Call this every time scrape_live_odds succeeds."""
    if df is None or len(df) == 0:
        return
    key = _snapshot_key(date_str, venue, race_num)
    now = time.time()
    snap = df.copy()
    with _snapshot_lock:
        snaps = _odds_snapshots.setdefault(key, [])
        snaps[:] = [s for s in snaps if now - s['timestamp'] < SNAPSHOT_TTL_SECONDS]
        snaps.append({'timestamp': now, 'df': snap})


def get_odds_baseline(date_str, venue, race_num, min_age_seconds=BASELINE_LEAD_SECONDS):
    """Returns (baseline_df, age_seconds, mature) for the race.

    The baseline is the OLDEST retained snapshot. `mature` is True only when
    that snapshot is at least min_age_seconds old (default: 15 minutes before
    jump). Returns (None, 0.0, False) when no snapshots exist yet.
    """
    key = _snapshot_key(date_str, venue, race_num)
    now = time.time()
    with _snapshot_lock:
        snaps = [s for s in _odds_snapshots.get(key, [])]
        snaps = [s for s in snaps if now - s['timestamp'] < SNAPSHOT_TTL_SECONDS]
        if not snaps:
            return None, 0.0, False
        oldest = min(snaps, key=lambda s: s['timestamp'])
        return oldest['df'].copy(), now - oldest['timestamp'], (now - oldest['timestamp']) >= min_age_seconds


# =====================================================================
# Odds snapshot PERSISTENCE (real-data store for the execution audit)
# Writes every poll to data/odds_snapshots/YYYYMMDD_VENUE.csv so the
# T-15m baseline / T-3m decision / T-0 close can be replayed later against
# the official final dividends (zero-lookahead dual-track backtest).
# =====================================================================
SNAPSHOT_DIR = os.path.join("data", "odds_snapshots")
SNAPSHOT_COLUMNS = ['timestamp', 'epoch', 'race_id', 'venue', 'horse_number',
                    'horse_name', 'win_odds', 'place_odds', 'time_to_post']
_snapshot_csv_lock = threading.Lock()


def _normalize_date(date_str) -> str:
    """Normalizes '2026-09-06' / '2026/09/06' / '20260906' to YYYY-MM-DD."""
    m = re.match(r'^(\d{4})[-/]?(\d{1,2})[-/]?(\d{1,2})$', str(date_str).strip())
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return str(date_str)


def _snapshot_path(date_str: str, venue: str) -> str:
    day = _normalize_date(date_str).replace('-', '')
    return os.path.join(SNAPSHOT_DIR, f"{day}_{str(venue).upper()}.csv")


def _append_snapshot_csv_rows(rows: list) -> None:
    """Synchronous per-day CSV append (called inside asyncio.to_thread)."""
    if not rows:
        return
    path = _snapshot_path(rows[0]['race_id'].split('_Race')[0], rows[0]['venue'])
    is_new = not os.path.exists(path)
    with _snapshot_csv_lock:
        with open(path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=SNAPSHOT_COLUMNS, extrasaction='ignore')
            if is_new:
                writer.writeheader()
            for r in rows:
                writer.writerow({c: r.get(c) for c in SNAPSHOT_COLUMNS})


async def persist_odds_snapshot(date_str: str, venue: str, race_num: int,
                                df: Optional[pd.DataFrame],
                                time_to_post: Optional[float] = None) -> None:
    """Non-blocking persistence of one poll.

    Updates the in-memory cache AND appends the poll to the per-meeting CSV
    (data/odds_snapshots/YYYYMMDD_VENUE.csv). time_to_post is seconds before
    post (positive); it labels the snapshot for the T-15m / T-3m / T-0 audit.
    """
    if df is None or len(df) == 0:
        return
    now_epoch = time.time()
    with _snapshot_lock:
        snaps = _odds_snapshots.setdefault(_snapshot_key(date_str, venue, race_num), [])
        snaps[:] = [s for s in snaps if now_epoch - s['timestamp'] < SNAPSHOT_TTL_SECONDS]
        snaps.append({'timestamp': now_epoch, 'df': df.copy()})

    race_id = f"{_normalize_date(date_str)}_Race{int(race_num)}"
    ts_iso = datetime.fromtimestamp(now_epoch).isoformat(timespec='seconds')
    rows = []
    for _, r in df.iterrows():
        rows.append({
            'timestamp': ts_iso,
            'epoch': round(now_epoch, 3),
            'race_id': race_id,
            'venue': str(venue).upper(),
            'horse_number': r.get('horse_number'),
            'horse_name': r.get('horse_name'),
            'win_odds': r.get('win_odds'),
            'place_odds': r.get('place_odds'),
            'time_to_post': round(float(time_to_post), 1) if time_to_post is not None else None,
        })
    try:
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        await asyncio.to_thread(_append_snapshot_csv_rows, rows)
    except Exception as e:
        print(f"Snapshot persistence failed ({race_id}): {e}")


def load_odds_snapshots(date_str: Optional[str] = None,
                        venue: Optional[str] = None) -> pd.DataFrame:
    """Loads persisted snapshot CSVs into one DataFrame (REAL data only).

    Optional filters: date_str (YYYY-MM-DD / YYYYMMDD) and venue ('ST'/'HV').
    Returns an empty typed frame when the store is empty.
    """
    if not os.path.isdir(SNAPSHOT_DIR):
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    frames = []
    want_day = _normalize_date(date_str).replace('-', '') if date_str else None
    for fn in sorted(os.listdir(SNAPSHOT_DIR)):
        if not fn.endswith('.csv'):
            continue
        stem = fn[:-4]
        parts = stem.split('_')
        if len(parts) < 2:
            continue
        fday, fvenue = parts[0], parts[1]
        if want_day is not None and fday != want_day:
            continue
        if venue is not None and fvenue.upper() != str(venue).upper():
            continue
        try:
            frames.append(pd.read_csv(os.path.join(SNAPSHOT_DIR, fn)))
        except Exception as e:
            print(f"Skipping snapshot file {fn}: {e}")
    if not frames:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
    return df


def get_labelled_snapshot(date_str: str, venue: str, race_num: int,
                          target_seconds: float,
                          window: Tuple[float, float] = (90.0, 330.0)) -> Optional[pd.DataFrame]:
    """Returns the persisted poll whose time_to_post is nearest `target_seconds`
    within `window` (e.g. the T-3m decision poll), as a horse-level frame.
    None when no matching labelled poll exists yet (real data only)."""
    snaps = load_odds_snapshots(date_str, venue)
    if len(snaps) == 0:
        return None
    race_id = f"{_normalize_date(date_str)}_Race{int(race_num)}"
    sub = snaps[(snaps['race_id'] == race_id) & snaps['time_to_post'].notna()].copy()
    if len(sub) == 0:
        return None
    t = pd.to_numeric(sub['time_to_post'], errors='coerce')
    in_win = t.between(window[0], window[1])
    if not in_win.any():
        return None
    best_idx = sub.index[in_win][(t[in_win] - target_seconds).abs().argmin()]
    epoch = sub.loc[best_idx, 'epoch']
    frame = snaps[(snaps['race_id'] == race_id) & (snaps['epoch'] == epoch)].copy()
    frame['win_odds'] = pd.to_numeric(frame['win_odds'], errors='coerce')
    frame['horse_number'] = pd.to_numeric(frame['horse_number'], errors='coerce')
    return frame.drop_duplicates(subset='horse_number')


def fetch_meeting_schedule(date_str, venue_code):
    """Synchronous GraphQL fetch of the meeting's race schedule.

    Uses the exact bundle-extracted RACECARD_PROFILE_QUERY (the server
    allow-lists operation text, so trimmed queries silently return nothing).
    Returns a list of dicts: {race_no, post_time (tz-aware datetime), status,
    race_name}, ordered by race number. Empty list when there is no meeting
    for the requested date/venue (e.g. HKJC off-season).
    """
    try:
        req = urllib.request.Request(
            GRAPHQL_BASE_URL,
            data=json.dumps({
                'query': RACECARD_PROFILE_QUERY,
                'variables': {'date': str(date_str), 'venueCode': str(venue_code)},
            }).encode(),
            headers={
                'content-type': 'application/json',
                'accept-encoding': 'gzip, deflate',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            })
        resp = urllib.request.urlopen(req, timeout=30)
        raw = resp.read()
        if resp.headers.get('Content-Encoding') == 'gzip':
            raw = gzip.decompress(raw)
        body = json.loads(raw.decode())
        profile = (body.get('data') or {}).get('raceMeetingProfile') or []
        if not profile:
            return []

        schedule = []
        for race in profile[0].get('races') or []:
            post_time_raw = race.get('postTime')
            if not post_time_raw:
                continue
            try:
                post_time = datetime.fromisoformat(str(post_time_raw))
            except (ValueError, TypeError):
                continue
            schedule.append({
                'race_no': int(race.get('no') or 0),
                'post_time': post_time,
                'status': race.get('status') or '',
                'race_name': race.get('raceName_en') or '',
            })
        schedule.sort(key=lambda r: r['race_no'])
        return schedule
    except Exception as e:
        print(f"Meeting schedule fetch failed for {date_str} {venue_code}: {e}")
        return []

class BrowserManager:
    """Manages a single Playwright browser instance to avoid spinning up new ones repeatedly.

    Includes recycling (every N live-odds polls or 2h) and a heartbeat so the
    Discord daemon watchdog can detect and respawn a frozen browser.
    """
    MAX_SCRAPES_BEFORE_RECYCLE = 3
    MAX_AGE_SECONDS = 2 * 3600

    def __init__(self):
        self.playwright = None
        self.browser = None
        self.opened_at: Optional[float] = None
        self.scrape_count = 0
        self._lock = threading.Lock()

    async def get_browser(self):
        if self.browser is None:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(headless=True)
            with self._lock:
                self.opened_at = time.time()
                self.scrape_count = 0
            print("Started reusable Playwright browser.")
        return self.browser

    async def close(self):
        with self._lock:
            self.opened_at = None
            self.scrape_count = 0
        if self.browser:
            try:
                await self.browser.close()
            except Exception as e:
                print(f"Browser close error: {e}")
            self.browser = None
        if self.playwright:
            try:
                await self.playwright.stop()
            except Exception as e:
                print(f"Playwright stop error: {e}")
            self.playwright = None
            print("Closed Playwright browser.")

    def age_seconds(self) -> float:
        with self._lock:
            return (time.time() - self.opened_at) if self.opened_at else 0.0

    def bump_scrape_count(self) -> None:
        with self._lock:
            self.scrape_count += 1

    def recycle_due(self) -> bool:
        with self._lock:
            return bool(self.browser is not None and (
                self.scrape_count >= self.MAX_SCRAPES_BEFORE_RECYCLE
                or (self.opened_at and time.time() - self.opened_at >= self.MAX_AGE_SECONDS)))

    async def recycle(self) -> None:
        """Tears down the current browser; the next get_browser() respawns fresh."""
        print("Recycling Playwright browser (memory hygiene).")
        await self.close()


# Global instance
browser_manager = BrowserManager()

# --- Live-daemon heartbeat / busy tracking (watchdog + safe recycle) ---
_scrape_active = 0
_first_start_since: Optional[float] = None
_busy_lock = threading.Lock()


def mark_scrape_started() -> None:
    global _scrape_active, _first_start_since
    with _busy_lock:
        if _scrape_active == 0:
            _first_start_since = time.time()
        _scrape_active += 1


def mark_scrape_finished() -> None:
    global _scrape_active, _first_start_since
    with _busy_lock:
        _scrape_active = max(0, _scrape_active - 1)
        if _scrape_active == 0:
            _first_start_since = None


def scrape_active() -> bool:
    with _busy_lock:
        return _scrape_active > 0


def scrape_stalled(seconds: float = 180.0) -> bool:
    """True when a scrape has been in flight (continuously) > `seconds`."""
    with _busy_lock:
        if _scrape_active == 0 or _first_start_since is None:
            return False
        return time.time() - _first_start_since > seconds

@alru_cache(maxsize=32, ttl=3600)
async def scrape_speedpro_formguide(race_num=1):
    """
    Scrapes the SpeedPRO form guide from the HKJC site for the given race number.
    Returns a dictionary mapping horse_name -> formguide_text.
    """
    url = f"https://racing.hkjc.com/en-us/local/info/speedpro/formguide?raceno={race_num}"
    
    browser = await browser_manager.get_browser()
    page = await browser.new_page()
        
    try:
        await page.goto(url, timeout=30000)
        await page.wait_for_selector('table.datatable', timeout=15000)
        await page.wait_for_timeout(1000)
        
        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')

        table = soup.find('table', {'class': 'datatable'})
        if not table:
            return {}
        
        formguide_data = {}
        current_horse = None
        current_horse_remarks = []
        
        for row in table.find_all('tr'):
            cells = [td.text.strip().replace('\xa0', ' ').replace('\n', ' ') for td in row.find_all(['th', 'td'])]
            if not cells: continue
            
            # Identify horse row
            if len(cells) == 8 and '(' in cells[1] and cells[0]:
                if current_horse:
                    formguide_data[current_horse] = " | ".join(current_horse_remarks[:3])
                    
                horse_str = cells[0]
                match = re.search(r'^\d+\s+(.*)', horse_str)
                if match:
                    current_horse = match.group(1).upper()
                else:
                    current_horse = horse_str.upper()
                
                current_horse = re.sub(r'\s*\([A-Z]+\)$', '', current_horse).strip()
                current_horse_remarks = []
            elif current_horse and len(cells) >= 10:
                comment_cell = cells[9]
                
                # Regex to find "Pace " and anything following it
                pace_match = re.search(r'(Pace\s+.*)', comment_cell)
                if pace_match:
                    comment = pace_match.group(1).strip()
                else:
                    comment = comment_cell
                        
                date_str = cells[0]
                pos_str = cells[7]
                
                current_horse_remarks.append(f"[{date_str} Pos:{pos_str}] {comment}")

        if current_horse:
            formguide_data[current_horse] = " | ".join(current_horse_remarks[:3])

        return formguide_data
        
    except Exception as e:
        print(f"Error scraping SpeedPRO Form Guide for race {race_num}: {e}")
        return {}
    finally:
        await page.close()


async def scrape_live_wpq(date_str, venue="S1", race_num=1):
    """
    Scrapes the WPQ page for top Quinella and Quinella Place combinations.
    """
    url = f"https://bet.hkjc.com/en/racing/wpq/{date_str}/{venue}/{race_num}"
    
    browser = await browser_manager.get_browser()
    page = await browser.new_page()
        
    try:
        await page.goto(url, timeout=30000)
        await page.wait_for_timeout(3000)
        
        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')
        
        qins = []
        qpls = []
        
        qin_table = soup.find('table', class_='qin-odds-table-QIN')
        if qin_table:
            for td in qin_table.find_all('td'):
                if 'id' in td.attrs and td['id'].startswith('qb_QIN_'):
                    parts = td['id'].split('_')
                    if len(parts) == 4:
                        odds = td.get_text(strip=True)
                        if odds and odds.replace('.', '', 1).isdigit():
                            qins.append(f"{parts[2]}-{parts[3]}: {float(odds)}")
                            
        qpl_table = soup.find('table', class_='qin-odds-table-QPL')
        if qpl_table:
            for td in qpl_table.find_all('td'):
                if 'id' in td.attrs and td['id'].startswith('qb_QPL_'):
                    parts = td['id'].split('_')
                    if len(parts) == 4:
                        odds = td.get_text(strip=True)
                        if odds and odds.replace('.', '', 1).isdigit():
                            qpls.append(f"{parts[2]}-{parts[3]}: {float(odds)}")
        
        top_qins = sorted(qins, key=lambda x: float(x.split(': ')[1]))[:10]
        top_qpls = sorted(qpls, key=lambda x: float(x.split(': ')[1]))[:10]
        
        result = ""
        if top_qins:
            result += f"Top 10 Lowest QIN (Quinella) Combinations: {', '.join(top_qins)}\n"
        if top_qpls:
            result += f"Top 10 Lowest QPL (Quinella Place) Combinations: {', '.join(top_qpls)}\n"
            
        return result.strip()
        
    except Exception as e:
        print(f"Error scraping WPQ for race {race_num}: {e}")
        return ""
    finally:
        await page.close()


@alru_cache(maxsize=32, ttl=3600)
async def scrape_draw_statistics(race_num=1):
    """
    Scrapes the historical draw statistics (Win%, Place%) for the given race number
    from the current HKJC upcoming race meeting.
    Returns a dictionary mapping draw_number -> {'win_pct': float, 'place_pct': float}.
    """
    url = "https://racing.hkjc.com/en-us/local/information/draw"
    
    browser = await browser_manager.get_browser()
    page = await browser.new_page()
        
    try:
        await page.goto(url, timeout=30000)
        # Give it a moment to render since the page often contains multiple tables
        await page.wait_for_timeout(3000)
        
        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')

        # Find the specific row for this race, e.g. <tr id="race1">
        tr = soup.find('tr', id=f"race{race_num}")
        if not tr:
            return {}

        table = tr.find_parent('table')
        if not table:
            return {}
        
        draw_stats = {}
        # The first two rows of this table are usually headers
        rows = table.find_all('tr')[2:]
        for row in rows:
            cols = [td.text.strip() for td in row.find_all('td')]
            # Ensure it's a valid data row (10+ columns) and Draw is a number
            if len(cols) >= 10 and cols[0].isdigit():
                draw = int(cols[0])
                win_pct_str = cols[6]
                place_pct_str = cols[8]
                
                win_pct = float(win_pct_str) if win_pct_str.replace('.', '', 1).isdigit() else 0.0
                place_pct = float(place_pct_str) if place_pct_str.replace('.', '', 1).isdigit() else 0.0
                
                draw_stats[draw] = {
                    'draw_win_pct': win_pct,
                    'draw_place_pct': place_pct
                }

        return draw_stats
        
    except Exception as e:
        print(f"Error scraping Draw Statistics for race {race_num}: {e}")
        return {}
    finally:
        await page.close()


@alru_cache(maxsize=32, ttl=3600)
async def scrape_speedpro(race_num=1):
    """
    Scrapes the SpeedPRO energy ratings from the HKJC site for the given race number.
    Returns a dictionary mapping horse_name -> speedpro_energy.
    """
    url = f"https://racing.hkjc.com/en-us/local/info/speedpro/speedguide?raceno={race_num}"
    
    browser = await browser_manager.get_browser()
    page = await browser.new_page()
        
    try:
        await page.goto(url, timeout=30000)
        # Give it time to load the dynamic content
        await page.wait_for_timeout(3000)
        
        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract base64 images (Pace speed / predicted position charts)
        images = []
        for img in soup.find_all('img'):
            src = img.get('src', '')
            if src.startswith('data:image'):
                images.append(src)

        # Find the energy table
        table = None
        for t in soup.find_all('table'):
            if 'Energy Required' in t.text or 'Energy' in t.text:
                table = t
                break
                
        if not table:
            return {}, images
            
        speedpro_data = {}
        for row in table.find_all('tr')[1:]:
            cells = [td.text.strip().replace('\xa0', ' ').replace('\n', ' ') for td in row.find_all(['th', 'td'])]
            if len(cells) > 12:
                horse_name_raw = cells[1]
                # Clean the name (remove (AUS) etc.)
                horse_name = re.sub(r'\s*\([A-Z]+\)$', '', horse_name_raw).strip()
                
                energy_str = cells[12]
                # Clean up the energy value
                if energy_str and energy_str.isdigit():
                    speedpro_data[horse_name] = int(energy_str)
                    
        return speedpro_data, images
        
    except Exception as e:
        print(f"Error scraping SpeedPRO for race {race_num}: {e}")
        return {}, []
    finally:
        await page.close()


async def scrape_live_odds(date_str, venue="S1", race_num=1, time_to_post: Optional[float] = None):
    """
    Scrapes live odds from the HKJC betting site.
    Example URL: https://bet.hkjc.com/en/racing/wp/2026-02-28/S1/1
    time_to_post: seconds before post (labels the persisted snapshot for the
    T-15m / T-3m / T-0 execution audit).
    """
    url = f"https://bet.hkjc.com/en/racing/wp/{date_str}/{venue}/{race_num}"
    print(f"Scraping live odds from: {url}")
    mark_scrape_started()

    browser = await browser_manager.get_browser()
    page = await browser.new_page()
        
    try:
        await page.goto(url, timeout=60000)
        # Wait for the odds table to load
        from playwright.async_api import TimeoutError
        try:
            await page.wait_for_selector('.rc-odds-table', timeout=15000)
        except TimeoutError:
            print(f"Timeout waiting for odds table at {url}. Odds might not be available yet.")
            return None
        await asyncio.sleep(2) # Give it a moment to populate odds
        
        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')
        
        # Verify the page is actually for the REQUESTED race: bet.hkjc.com serves
        # the default (Race 1) card for out-of-range race numbers (e.g. /ST/11
        # when the day only has 10 races), which previously poisoned card
        # discovery and duplicated Race 1 in the board.
        # Header shape: "Race 3\n06/09, SUN, 13:30, Group Three, ..."
        hm = re.search(r'Race\s*(\d+)\s*(\d{2}/\d{2}),\s*(\w{3}),\s*(\d{2}:\d{2})', soup.text)
        if hm and int(hm.group(1)) != int(race_num):
            print(f"WRONG RACE served for {url}: header says Race {hm.group(1)} -> ignore")
            return None
        if hm and int(hm.group(1)) == int(race_num):
            RACE_META[(date_str, venue, int(race_num))] = {
                'title': soup.text[hm.start():hm.start() + 180].strip(),
                'post_hhmm': hm.group(4),
                'fetched_at': time.time(),
            }
            
        # Check if it's a valid race page
        if "No race" in soup.text or "not available" in soup.text:
            return None
            
        # Find the main odds table
        table = soup.find('table', class_='rc-odds-table')
        if not table:
            return None
            
        horses = []
        rows = table.find_all('tr')
        
        # Skip header row
        for row in rows[1:]:
            cols = row.find_all(['th', 'td'])
            if len(cols) >= 9:
                try:
                    horse_num = cols[0].text.strip()
                    if not horse_num.isdigit():
                        continue
                    
                    # Sometimes there's an image/silk in col 1, name in col 2
                    horse_name_raw = cols[2].text.strip()
                    # Clean up name (remove country code like (AUS))
                    horse_name = re.sub(r'\s*\([A-Z]+\)$', '', horse_name_raw).strip()
                    
                    draw = cols[3].text.strip()
                    weight = cols[4].text.strip()
                    jockey = cols[5].text.strip()
                    trainer = cols[6].text.strip()
                    
                    win_odds_str = cols[7].text.strip()
                    place_odds_str = cols[8].text.strip()
                    
                    # Handle scratched horses or missing odds
                    # 999 usually means odds are not yet available
                    if win_odds_str == 'SCR':
                        continue
                        
                    win_odds = float(win_odds_str) if win_odds_str.replace('.','',1).isdigit() else None
                    place_odds = float(place_odds_str) if place_odds_str.replace('.','',1).isdigit() else None
                    
                    # If odds are 999, set them to None so we know they aren't real odds yet
                    if win_odds == 999.0: win_odds = None
                    if place_odds == 999.0: place_odds = None
                        
                    horses.append({
                        'horse_number': int(horse_num),
                        'horse_name': horse_name,
                        'barrier_draw': int(draw) if draw.isdigit() else None,
                        'weight_carried': float(weight) if weight.isdigit() else None,
                        'jockey': jockey,
                        'trainer': trainer,
                        'win_odds': win_odds,
                        'place_odds': place_odds
                    })
                except Exception as e:
                    print(f"Error parsing row: {e}")
                    continue
        
        if not horses:
            return None
            
        df = pd.DataFrame(horses)
        df['race_date'] = date_str
        df['race_number'] = race_num
        df['venue'] = venue

        # Add SpeedPRO data and Draw Statistics
        try:
            print("Fetching SpeedPRO ratings, Form Guide, WPQ, and Draw Stats...")
            speedpro_result, formguide_data, wpq_str, draw_stats = await asyncio.gather(
                scrape_speedpro(race_num),
                scrape_speedpro_formguide(race_num),
                scrape_live_wpq(date_str, venue, race_num),
                scrape_draw_statistics(race_num)
            )
            
            speedpro_data, speedpro_images = speedpro_result

            df['speedpro_energy'] = df['horse_name'].map(lambda name: speedpro_data.get(name.upper(), None))
            df['formguide_remarks'] = df['horse_name'].map(lambda name: formguide_data.get(name.upper(), ""))
            
            # Map draw stats to runners based on their barrier_draw
            df['draw_win_pct'] = df['barrier_draw'].map(lambda draw: draw_stats.get(draw, {}).get('draw_win_pct', None) if pd.notna(draw) else None)
            df['draw_place_pct'] = df['barrier_draw'].map(lambda draw: draw_stats.get(draw, {}).get('draw_place_pct', None) if pd.notna(draw) else None)

            df.attrs['wpq_str'] = wpq_str
            df.attrs['speedpro_images'] = speedpro_images
        except Exception as e:
            print(f"Skipping extra sources due to error: {e}")
            df['speedpro_energy'] = None
            df['formguide_remarks'] = ""
            df['draw_win_pct'] = None
            df['draw_place_pct'] = None
            df.attrs['wpq_str'] = ""
            df.attrs['speedpro_images'] = []

        # Persist this poll to memory + disk snapshot store (real-data audit trail)
        await persist_odds_snapshot(date_str, venue, race_num, df, time_to_post=time_to_post)
        return df

    except Exception as e:
        print(f"Error scraping live odds: {e}")
        return None
    finally:
        await page.close()
        mark_scrape_finished()
        # Memory hygiene: recycle ONLY when no other scrape is in flight
        try:
            browser_manager.bump_scrape_count()
            if browser_manager.recycle_due() and not scrape_active():
                await browser_manager.recycle()
        except Exception as e:
            print(f"Browser recycle error: {e}")

if __name__ == "__main__":
    async def test():
        df = await scrape_live_odds("2026-03-22", "S1", 1)
        if df is not None:
            print(df[['horse_name', 'barrier_draw', 'draw_win_pct', 'draw_place_pct']].head(10))
            print("Columns available:", df.columns.tolist())
        else:
            print("No valid race data found.")
        await browser_manager.close()
        
    asyncio.run(test())
