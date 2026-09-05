"""
HKJC Dividend Backfill (派彩回補)
================================
Scrapes SETTLED dividends (WIN / PLA / QIN / QPL / ...) and the horse-number ->
horse-code map for each race from the modern HKJC results pages, and appends
them to data/dividends.csv for settlement by modeling/exotics_pricing.py.

REAL DATA ONLY — every row is scraped from an actual settled race result.

Usage:
    python scraping/backfill_dividends.py            # last 10 race days in raw_csvs
    python scraping/backfill_dividends.py --days 30  # last 30 race days
"""

import argparse
import asyncio
import csv
import logging
import os
import random
import re
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from scraping.scraper import HKJCScraper

logger = logging.getLogger(__name__)

HISTORICAL_DIVIDENDS_CSV = "data/historical_dividends.csv"
DIVIDEND_FIELDS = ['race_id', 'race_date', 'venue', 'race_no', 'pool', 'combo',
                   'combo_codes', 'dividend']

RESULTS_URL = ("https://racing.hkjc.com/en-us/local/information/localresults"
               "?racedate={date}&Racecourse={venue}&RaceNo={no}")

# --- Anti-scraping & stability safeguards ---
MAX_RETRIES = 4
BACKOFF_SCHEDULE = [5, 15, 45, 90]     # seconds between attempts 1..4
RETRYABLE_STATUS = {429, 502, 503}     # HTTP statuses worth retrying
BETWEEN_RACE_JITTER = (1.0, 3.0)       # jitter between individual race requests
BETWEEN_DAY_JITTER = (3.0, 6.0)        # jitter between consecutive race days
SESSION_RECYCLE_MEETINGS = 50          # recycle browser session every N meetings
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]


def parse_runner_map(html_content: str) -> dict:
    """Maps horse NUMBER -> horse CODE from the results table (real page)."""
    soup = BeautifulSoup(html_content, 'html.parser')
    runner_map = {}
    table = soup.find('table', class_='f_tac table_bd')
    if table is None:
        for t in soup.find_all('table'):
            if len(t.find_all('tr')) > 3:
                table = t
                break
    if table is None:
        return runner_map
    for row in table.find_all('tr'):
        cells = [td.get_text(' ', strip=True) for td in row.find_all(['th', 'td'])]
        if len(cells) < 3:
            continue
        num = cells[1].strip()
        if not num.isdigit():
            continue
        code = None
        for c in cells[2:4]:
            m = re.search(r'\(([A-Z]\d{3})\)', c)
            if m:
                code = m.group(1)
                break
        if code:
            runner_map[num] = code
    return runner_map


async def scrape_one_race(page, scraper: HKJCScraper, date_str: str, venue: str, race_no: int):
    """Scrapes settled dividends + runner map for one race with status-aware retries.

    Retry policy (max 4 retries: 5s -> 15s -> 45s -> 90s, each + 1-3s jitter)
    applies to HTTP 429/502/503 and Playwright timeouts. 'No dividends found'
    is a valid end-of-day signal and is NOT retried.
    """
    url = RESULTS_URL.format(date=f"{date_str.replace('-', '/')}", venue=venue, no=race_no)
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = await page.goto(url, wait_until='domcontentloaded', timeout=60000)
            if response is not None and response.status in RETRYABLE_STATUS:
                raise RuntimeError(f"HTTP {response.status}")
            # The dividend block is JS-rendered; wait for its heading before parsing
            try:
                await page.wait_for_selector("text=Winning Combination", timeout=20000)
                await page.wait_for_timeout(1500)
            except Exception:
                await page.wait_for_timeout(9000)
            html = await page.content()

            divs = scraper.parse_dividends(html)
            all_divs = divs.get('all_dividends', [])
            if not all_divs:
                logger.info(f"no dividends found for {date_str} {venue} R{race_no}")
                return []
            runner_map = parse_runner_map(html)

            race_id = f"{date_str}_Race{race_no}"
            rows = []
            for pool, combo, dividend in all_divs:
                numbers = [p for p in re.split(r'[,\s]', combo) if p.isdigit()]
                codes = [runner_map.get(n, n) for n in numbers]
                rows.append({
                    'race_id': race_id,
                    'race_date': date_str,
                    'venue': venue,
                    'race_no': race_no,
                    'pool': pool,
                    'combo': combo,
                    'combo_codes': '/'.join(codes),
                    'dividend': dividend,
                })
            logger.info(f"{race_id}: {len(rows)} dividend rows (WIN={divs.get('win_dividend')} "
                        f"PLA={divs.get('place_dividends')} QIN={divs.get('quinella_dividend')})")
            return rows
        except Exception as e:
            last_error = e
            if attempt >= MAX_RETRIES:
                break
            wait = BACKOFF_SCHEDULE[attempt] + random.uniform(1.0, 3.0)
            logger.warning(f"attempt {attempt + 1}/{MAX_RETRIES + 1} failed for {date_str} R{race_no}: "
                           f"{e}; retrying in {wait:.1f}s")
            await asyncio.sleep(wait)
    logger.warning(f"giving up on {date_str} {venue} R{race_no}: {last_error}")
    return []


async def backfill_dates(dates_venues, limit_races: int = 12, out_path: str = HISTORICAL_DIVIDENDS_CSV):
    scraper = HKJCScraper("https://x")

    # --- Checkpoint load: existing race_ids are never re-requested ---
    existing = set()
    if os.path.exists(out_path):
        with open(out_path, newline='', encoding='utf-8') as f:
            existing = {row['race_id'] for row in csv.DictReader(f)}
    added_this_run = set()  # dedupe guard within this run
    logger.info("Resuming: %d race(s) already in %s", len(existing), out_path)

    total = 0
    total_days = len(dates_venues)
    start_time = time.time()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=random.choice(USER_AGENTS))
        page = await context.new_page()
        meetings_since_recycle = 0

        for idx, (date_str, venue) in enumerate(dates_venues, start=1):
            if idx > 1:
                await asyncio.sleep(random.uniform(*BETWEEN_DAY_JITTER))

            # Session recycling: new context + fresh user-agent every N meetings
            if meetings_since_recycle >= SESSION_RECYCLE_MEETINGS:
                await context.close()
                context = await browser.new_context(user_agent=random.choice(USER_AGENTS))
                page = await context.new_page()
                meetings_since_recycle = 0
                logger.info("Recycled browser session with a new user-agent")

            # --- Skip fully-covered meetings (contiguous race coverage) ---
            covered_count = sum(
                1 for n in range(1, limit_races + 1)
                if f"{date_str}_Race{n}" in existing or f"{date_str}_Race{n}" in added_this_run)
            if covered_count >= 9:
                elapsed = time.time() - start_time
                logger.info("[BACKFILL] [%d/%d Days] Date: %s (%s) | Skipped (already covered) | "
                            "Elapsed: %dm", idx, total_days, date_str, venue, int(elapsed // 60))
                meetings_since_recycle += 1
                continue

            day_rows = []
            races_done = 0
            for race_no in range(1, limit_races + 1):
                race_id = f"{date_str}_Race{race_no}"
                if race_id in existing or race_id in added_this_run:
                    continue
                await asyncio.sleep(random.uniform(*BETWEEN_RACE_JITTER))
                rows = await scrape_one_race(page, scraper, date_str, venue, race_no)
                if not rows:
                    if race_no >= 2 and not day_rows:
                        # no dividends for race 2+ with nothing collected -> no meeting
                        continue
                    if race_no >= 2 and day_rows:
                        break  # end of meeting
                    continue
                day_rows.extend(rows)
                added_this_run.update(r['race_id'] for r in rows)
                races_done += 1
                total += len(rows)
            meetings_since_recycle += 1

            # --- Atomic per-meeting flush (append + fsync) ---
            if day_rows:
                exists = os.path.exists(out_path)
                with open(out_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=DIVIDEND_FIELDS)
                    if not exists:
                        writer.writeheader()
                    writer.writerows(day_rows)
                    f.flush()
                    os.fsync(f.fileno())

            # --- CLI telemetry ---
            elapsed = time.time() - start_time
            avg_per_day = elapsed / max(idx, 1)
            est_remaining = avg_per_day * max(total_days - idx, 0)
            logger.info(
                "[BACKFILL] [%d/%d Days] Date: %s (%s) | Races: %d | Added: %d rows | "
                "Elapsed: %dm | Est. Remaining: %dm",
                idx, total_days, date_str, venue, races_done, len(day_rows),
                int(elapsed // 60), int(est_remaining // 60))

        await context.close()
        await browser.close()
    logger.info("Backfill complete: %d new dividend rows", total)


def venue_for_date(date_str: str) -> str:
    """Infers the venue code (ST / HV) from the raw scraped CSV for that day."""
    import pandas as pd
    path = os.path.join('data', 'raw_csvs', f"{date_str}.csv")
    if os.path.exists(path):
        try:
            track = str(pd.read_csv(path, usecols=['track'], nrows=1)['track'].iloc[0]).lower()
            if 'happy valley' in track:
                return 'HV'
        except Exception:
            pass
    return 'ST'


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=10, help='number of most recent race days to backfill')
    ap.add_argument('--venue', default=None, help='venue code (ST / HV); auto-detected from raw CSVs if omitted')
    ap.add_argument('--out', default=HISTORICAL_DIVIDENDS_CSV, help='output CSV path')
    ap.add_argument('--limit-races', type=int, default=12, help='max races attempted per meeting')
    args = ap.parse_args()

    # Take the most recent race days present in raw_csvs (REAL scraped data)
    raw_dir = 'data/raw_csvs'
    files = sorted(f for f in os.listdir(raw_dir) if f.endswith('.csv'))[-args.days:]
    dates = [f.replace('.csv', '') for f in files]
    logger.info(f"Backfilling dividends for {len(dates)} days: {dates[0]} .. {dates[-1]}")
    asyncio.run(backfill_dates([(d, args.venue or venue_for_date(d)) for d in dates],
                               limit_races=args.limit_races,
                               out_path=args.out))


if __name__ == "__main__":
    main()
