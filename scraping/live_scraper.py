import asyncio
import pandas as pd
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from datetime import datetime
import re
from cachetools import TTLCache
from async_lru import alru_cache

# Cache for formguide which doesn't change often
_formguide_cache = TTLCache(maxsize=100, ttl=3600)  # 1 hour cache

class BrowserManager:
    """Manages a single Playwright browser instance to avoid spinning up new ones repeatedly."""
    def __init__(self):
        self.playwright = None
        self.browser = None

    async def get_browser(self):
        if self.browser is None:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(headless=True)
            print("Started reusable Playwright browser.")
        return self.browser

    async def close(self):
        if self.browser:
            await self.browser.close()
            self.browser = None
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None
            print("Closed Playwright browser.")

# Global instance
browser_manager = BrowserManager()

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


async def scrape_live_odds(date_str, venue="S1", race_num=1):
    """
    Scrapes live odds from the HKJC betting site.
    Example URL: https://bet.hkjc.com/en/racing/wp/2026-02-28/S1/1
    """
    url = f"https://bet.hkjc.com/en/racing/wp/{date_str}/{venue}/{race_num}"
    print(f"Scraping live odds from: {url}")
    
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

        return df

    except Exception as e:
        print(f"Error scraping live odds: {e}")
        return None
    finally:
        await page.close()

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
