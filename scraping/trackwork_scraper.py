"""
HKJC Trackwork & Barrier Trial Scraper (晨操紀錄 / 試閘結果)
=========================================================
Scrapes a horse's trackwork profile and barrier trial history from HKJC
and persists them to CSVs consumed by data_pipeline/feature_engineering.py.

⚠️ EDUCATIONAL USE ONLY.
"""

import asyncio
import csv
import logging
import os
import re
from typing import Dict, List

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

TRACKWORK_URL = "https://racing.hkjc.com/racing/information/English/Horse/TrackworkResult.aspx?HorseNo={code}"
TRIALS_URL = "https://racing.hkjc.com/racing/information/English/Horse/HorseTrainingRecord.aspx?HorseNo={code}"

TRACKWORK_CSV = "data/trackwork.csv"
TRIALS_CSV = "data/trials.csv"

TRACKWORK_FIELDS = ['horse_code', 'activity_date', 'track_type', 'work_type',
                    'rider', 'time_800', 'time_400', 'gear']
TRIAL_FIELDS = ['horse_code', 'trial_date', 'finish_pos', 'field_size',
                'time', 'margin_behind_leader', 'trial_jockey']


# ---------------------------------------------------------------- parsers ---
def parse_trackwork_html(html_content: str, horse_code: str) -> List[Dict]:
    """Parses the HKJC TrackworkResult page into event records.

    Expected table rows look like:
      12 Aug 2026 | AWT | Fast Gallop | 24.6 22.4 (58.0) | J Moreira | B/TT
    Robust against column order via header detection; falls back to the
    classic fixed layout when no header is present.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    records: List[Dict] = []

    table = None
    for t in soup.find_all('table'):
        text = t.get_text(' ', strip=True)
        if any(k in text.lower() for k in ['trackwork', 'fast gallop', 'swimming', 'trotting']):
            table = t
            break
    if table is None:
        return records

    header = []
    header_row = None
    for row in table.find_all('tr'):
        cells = [td.get_text(strip=True).lower() for td in row.find_all(['th', 'td'])]
        if any('date' in c or 'work' in c or 'rider' in c or 'time' in c for c in cells):
            header = cells
            header_row = row
            break

    col = {}
    for idx, txt in enumerate(header):
        if 'date' in txt:
            col['activity_date'] = idx
        elif 'track' in txt or 'course' in txt or 'awt' in txt or 'turf' in txt:
            col.setdefault('track_type', idx)
        elif 'work' in txt or 'exercise' in txt or 'training' in txt:
            col['work_type'] = idx
        elif 'rider' in txt or 'jockey' in txt:
            col['rider'] = idx
        elif 'time' in txt or 'distance' in txt:
            col.setdefault('time_800', idx)
        elif 'gear' in txt:
            col['gear'] = idx

    rows = table.find_all('tr')
    if header_row is not None:
        rows = rows[rows.index(header_row) + 1:]

    for row in rows:
        cells = [td.get_text(' ', strip=True) for td in row.find_all(['th', 'td'])]
        if not cells or len(cells) < 3:
            continue
        # skip footer/header noise
        if not re.search(r'\d{1,2}\s+\w{3}\s+\d{4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}', cells[0] if cells else ''):
            continue

        def get(name):
            if name in col and len(cells) > col[name]:
                return cells[col[name]]
            return ''

        raw = get('activity_date')
        m = re.search(r'(\d{1,2})\s+(\w{3})\s+(\d{4})', raw)
        date_str = f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else raw

        time_raw = get('time_800')
        times = re.findall(r'\d+\.\d+', time_raw)
        time_800 = times[0] if times else None
        time_400 = times[1] if len(times) > 1 else None

        records.append({
            'horse_code': horse_code,
            'activity_date': date_str,
            'track_type': get('track_type'),
            'work_type': get('work_type'),
            'rider': get('rider'),
            'time_800': time_800,
            'time_400': time_400,
            'gear': get('gear'),
        })
    return records


def parse_trials_html(html_content: str, horse_code: str) -> List[Dict]:
    """Parses the HKJC barrier trial table into trial records."""
    soup = BeautifulSoup(html_content, 'html.parser')
    records: List[Dict] = []

    table = None
    for t in soup.find_all('table'):
        text = t.get_text(' ', strip=True)
        if 'trial' in text.lower() and 'barrier' in text.lower():
            table = t
            break
    if table is None:
        # second pass: any table whose rows contain finish-position patterns
        for t in soup.find_all('table'):
            rows = t.find_all('tr')
            if len(rows) >= 2 and any(re.search(r'\b\d{1,2}\b', r.get_text(strip=True)) for r in rows[:3]):
                table = t
                break
    if table is None:
        return records

    for row in table.find_all('tr'):
        cells = [td.get_text(' ', strip=True) for td in row.find_all(['th', 'td'])]
        if not cells:
            continue
        date_str = ''
        for c in cells:
            m = re.search(r'(\d{1,2})\s+(\w{3})\s+(\d{4})', c)
            if m:
                date_str = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
                break
        if not date_str:
            continue

        nums = [c for c in cells if re.fullmatch(r'\d{1,2}', c.strip())]
        finish_pos = int(nums[0]) if nums else None
        field_size = int(nums[1]) if len(nums) > 1 else None
        time_m = re.search(r'\d{1,2}:\d{2}\.\d{2}', ' '.join(cells))
        margin = None
        for c in cells:
            mm = re.search(r'(\d+(?:\.\d+)?)\s*[LNH]$', c.strip())
            if mm:
                margin = float(mm.group(1))
                break

        records.append({
            'horse_code': horse_code,
            'trial_date': date_str,
            'finish_pos': finish_pos,
            'field_size': field_size,
            'time': time_m.group(0) if time_m else None,
            'margin_behind_leader': margin,
            'trial_jockey': cells[-1] if len(cells) > 5 else '',
        })
    return records


# ---------------------------------------------------------------- scraping ---
async def scrape_horse_records(page, url: str, parser, horse_code: str) -> List[Dict]:
    try:
        await page.goto(url, wait_until='domcontentloaded', timeout=60000)
        await page.wait_for_timeout(2500)  # allow JS rendering
        html_content = await page.content()
        return parser(html_content, horse_code)
    except Exception as e:
        logger.warning(f"Failed to scrape {url}: {e}")
        return []


async def scrape_horse_trackwork(horse_code: str) -> List[Dict]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        records = await scrape_horse_records(
            page, TRACKWORK_URL.format(code=horse_code), parse_trackwork_html, horse_code)
        await browser.close()
        return records


async def scrape_horse_trials(horse_code: str) -> List[Dict]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        records = await scrape_horse_records(
            page, TRIALS_URL.format(code=horse_code), parse_trials_html, horse_code)
        await browser.close()
        return records


def append_records(records: List[Dict], path: str, fieldnames: List[str]):
    if not records:
        return
    exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    code = sys.argv[1] if len(sys.argv) > 1 else "C834"
    tw = asyncio.run(scrape_horse_trackwork(code))
    tr = asyncio.run(scrape_horse_trials(code))
    print(f"Trackwork records for {code}: {len(tw)}")
    for r in tw[:3]:
        print("  ", r)
    print(f"Trial records for {code}: {len(tr)}")
    for r in tr[:3]:
        print("  ", r)
