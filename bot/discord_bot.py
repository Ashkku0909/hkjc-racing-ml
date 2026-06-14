import sys
import os
import base64
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import discord
from discord.ext import commands
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from bot.llm_prompts import PROFESSIONAL_HANDICAPPER_PROMPT, THREE_MODE_ANALYSIS_PROMPT, MODE1_ONLY_PROMPT, MODE2_ONLY_PROMPT, MODE3_ONLY_PROMPT
import asyncio
from scraping.live_scraper import scrape_live_odds
from bot.analyzer_service import load_data, get_data, merge_live_odds_with_predictions, estimate_probabilities_from_history
from datetime import datetime

# Load environment variables
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini Client
# We need to configure the client to use the proxy if it's set
proxy_url = os.getenv("http_proxy") or os.getenv("HTTP_PROXY")

# Set the proxy environment variables for httpx (which google-genai uses under the hood)
if proxy_url:
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url

genai_client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize Discord Bot
intents = discord.Intents.default()
intents.message_content = True

# Use proxy if defined in environment
proxy_url = os.getenv("http_proxy") or os.getenv("HTTP_PROXY")
bot = commands.Bot(command_prefix="!", intents=intents, proxy=proxy_url)

bot.remove_command("help")

@bot.event
async def on_ready():
    load_data()
    print(f'Logged in as {bot.user.name}')
    print('Ready to analyze races! Use !analyze <query> in Discord.')
    print('⚠️ EDUCATIONAL USE ONLY — Not for actual gambling. See DISCLAIMER.md')

@bot.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Process commands first (like !analyze)
    await bot.process_commands(message)

    # Check if the bot is mentioned or if it's a DM
    is_mentioned = bot.user in message.mentions
    is_dm = isinstance(message.channel, discord.DMChannel)

    # If it's not a command, and the bot is mentioned or it's a DM
    if (is_mentioned or is_dm) and not message.content.startswith('!'):
        async with message.channel.typing():
            await handle_chat_or_vlm(message)

async def handle_chat_or_vlm(message):
    # Remove the bot mention from the text
    text_content = message.content.replace(f'<@{bot.user.id}>', '').strip()
    
    contents = []
    
    # Add text if present
    if text_content:
        contents.append(text_content)
    elif message.attachments:
        contents.append("Please analyze this image.")
        
    # Handle attachments (images for VLM)
    if message.attachments:
        for attachment in message.attachments:
            # Check if it's an image
            if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp', '.gif']):
                # Download the image bytes
                image_bytes = await attachment.read()
                
                # Add to contents for Gemini
                contents.append(
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=attachment.content_type
                    )
                )
            else:
                await message.channel.send(f"Skipping {attachment.filename}: Only images are supported for analysis.")

    if not contents:
        return

    # Construct a system prompt for general chat / VLM
    system_instruction = """
    You are an expert Professional Horse Racing Analyst and Master Handicapper.
    You are chatting with a user on Discord.
    If they provide an image (like a race card, form guide, or track map), analyze it in detail.
    If they ask general questions about horse racing, betting strategies, or specific horses, answer them professionally.
    Keep your answers concise and suitable for Discord (under 2000 characters if possible).
    """

    try:
        # Run the synchronous Gemini API call in a separate thread to avoid blocking the Discord event loop
        response = await asyncio.to_thread(
            genai_client.models.generate_content,
            model='gemini-2.5-flash',
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
            )
        )
        
        reply = response.text
        if len(reply) > 2000:
            chunks = [reply[i:i+1990] for i in range(0, len(reply), 1990)]
            for chunk in chunks:
                await message.channel.send(chunk)
        else:
            await message.channel.send(reply)
            
    except Exception as e:
        await message.channel.send(f"An error occurred while calling Gemini: {str(e)}")

@bot.command(name="live")
async def analyze_live_race(ctx, date_str: str = None, venue: str = "S1", race_num: int = 1, *, mode: str = "all"):
    """
    Scrapes live odds and analyzes the upcoming race.
    Usage: !live 2026-02-28 S1 1 [all|public|mode1|mode2|mode3]
    'all' = All 3 modes (Web + Model + Strategy)
    'public' or 'mode1' or 'mode 1' = Only Mode 1 (Web Scraping handicapping, ignores Model entirely)
    'mode2' or 'mode 2' = Only Mode 2 (Model Walk-Forward Logic)
    'mode3' or 'mode 3' = Only Mode 3 (Betting Execution Strategy)
    """
    if date_str is None:
        # Default to today
        date_str = datetime.now().strftime("%Y-%m-%d")
        
    await ctx.send(f"🏇 Scraping live data for {date_str} Venue {venue} Race {race_num}... Please wait.")
    
    try:
        # Scrape the live data
        live_df = await scrape_live_odds(date_str, venue, race_num)

        if live_df is None or len(live_df) == 0:
            await ctx.send(f"Could not find any live race data for {date_str} {venue} Race {race_num}. There might not be a race, or odds are not posted yet.")
            return

        # Back up attrs before any df merges
        backup_attrs = live_df.attrs.copy() if hasattr(live_df, 'attrs') else {}

        df = get_data()
        mode_str = mode.lower().replace(" ", "")
        
        # Try to merge with model predictions if available (skip if running purely in mode 1)
        if df is not None and mode_str not in ("public", "mode1"):
            # Create a race_id to match with predictions
            # Note: The format in all_predictions.csv is like "2020-01-08_Race7"
            race_id = f"{date_str}_Race{race_num}"
            
            # Filter predictions for this race
            pred_df = df[df['race_id'] == race_id]
            
            if len(pred_df) > 0:
                live_df = merge_live_odds_with_predictions(live_df, pred_df)
                await ctx.send("✅ Successfully merged live odds with model predictions!")
            else:
                await ctx.send("⚠️ Could not find exact race in predictions. Attempting to estimate using horses' most recent historical data...")
                live_df = estimate_probabilities_from_history(live_df, df)
                if 'true_prob' in live_df.columns and live_df['true_prob'].notna().any():
                    await ctx.send("✅ Successfully estimated probabilities using historical model data!")
                else:
                    await ctx.send("⚠️ Could not find any historical data for these horses. Proceeding with live data only.")
        
        # Format the data for Gemini
        data_str = live_df.to_string(index=False)
        
        # Add WPQ attributes if they exist
        wpq_info = backup_attrs.get('wpq_str', '')
        if wpq_info:
            data_str += f"\n\nAdditional Market Data:\n{wpq_info}"
        
        if mode_str in ["public", "mode1"]:
            selected_prompt = MODE1_ONLY_PROMPT
        elif mode_str == "mode2":
            selected_prompt = MODE2_ONLY_PROMPT
        elif mode_str == "mode3":
            selected_prompt = MODE3_ONLY_PROMPT
        else:
            selected_prompt = THREE_MODE_ANALYSIS_PROMPT
            
        prompt = f"""
{selected_prompt}

Here is the LIVE scraped data for an upcoming race ({date_str} Venue {venue} Race {race_num}):

{data_str}

Note: If 'win_odds' or 'place_odds' are 'None', it means the odds have not been posted yet by the HKJC.
If 'true_prob' and 'expected_value' are present, they are derived from our LightGBM model predictions.
"""

        contents = [prompt]
        
        # Add images if available
        speedpro_images = backup_attrs.get('speedpro_images', [])
        for img_src in speedpro_images:
            if img_src.startswith('data:image'):
                try:
                    mime_type = img_src.split(';')[0].split(':')[1]
                    base64_data = img_src.split(',')[1]
                    image_bytes = base64.b64decode(base64_data)
                    contents.append(
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type=mime_type
                        )
                    )
                except Exception as e:
                    print(f"Failed to parse base64 image: {e}")

        await ctx.send("🧠 Data scraped successfully! Analyzing with Gemini...")

        # Run the synchronous Gemini API call in a separate thread
        response = await asyncio.to_thread(
            genai_client.models.generate_content,
            model='gemini-2.5-flash',
            contents=contents,
        )

        reply = response.text
        if len(reply) > 2000:
            chunks = [reply[i:i+1990] for i in range(0, len(reply), 1990)]
            for chunk in chunks:
                await ctx.send(chunk)
        else:
            await ctx.send(reply)

    except Exception as e:
        await ctx.send(f"An error occurred: {str(e)}")

@bot.command(name="scan_overlays")
async def scan_overlays(ctx, date_str: str, venue: str = "ST", max_races: int = 11):
    """
    Scans all races on a given day to find and rank the best value bets (overlays).
    Usage: !scan_overlays 2026-03-01 ST
    """
    df = get_data()
    if df is None:
        await ctx.send("Data not loaded. Please run the model training script first.")
        return

    status_msg = await ctx.send(f"🔍 Scanning up to {max_races} races at {venue} on {date_str} for overlays. This might take a minute...\nProgress: 0/{max_races}")
    
    all_overlays = []
    
    # We will fetch sequentially to avoid overloading the site or playwright
    for race_num in range(1, max_races + 1):
        try:
            # Update status msg every few races to avoid API spam limits
            if race_num % 2 == 0 or race_num == max_races:
                await status_msg.edit(content=f"🔍 Scanning up to {max_races} races at {venue} on {date_str} for overlays. This might take a minute...\nProgress: {race_num}/{max_races}")
                     
            live_df = await scrape_live_odds(date_str, venue, race_num)
            if live_df is None or len(live_df) == 0:
                continue
                
            # Create a race_id to match with predictions
            race_id = f"{date_str}_Race{race_num}"
            pred_df = df[df['race_id'] == race_id]
            
            if len(pred_df) > 0:
                merged_df = merge_live_odds_with_predictions(live_df, pred_df)
                
                # Check if we have win_odds to calculate EV
                if 'expected_value' in merged_df.columns:
                    has_odds_mask = merged_df['win_odds'].notna() & merged_df['true_prob'].notna()

                    # Find positive EV (Overlay)
                    if 'prob_edge' in merged_df.columns:
                        overlays = merged_df[has_odds_mask & (merged_df['expected_value'] > 0) & (merged_df['prob_edge'] > 0.0)]
                    else:
                        overlays = merged_df[has_odds_mask & (merged_df['expected_value'] > 0)]

                    for _, row in overlays.iterrows():
                        all_overlays.append({
                            'Race': race_num,
                            'Horse': row['horse_name'],
                            'Odds': row['win_odds'],
                            'True Prob': f"{row['true_prob']*100:.1f}%",
                            'Expected_Win': row['true_prob'],
                            'EV': row['expected_value'],
                            'Edge': f"{row.get('prob_edge', 0.0)*100:.1f}%"
                        })
        except Exception as e:
            print(f"Error scanning race {race_num}: {e}")
            
    if not all_overlays:
        await ctx.send("No overlays found or odds are not available yet.")
        return
        
    await status_msg.edit(content=f"✅ Scan complete! Found {len(all_overlays)} value bets.")
        
    # Sort overlays by EV descending
    all_overlays = sorted(all_overlays, key=lambda x: x['EV'], reverse=True)
    
    # Format message
    msg = f"🏆 **Top Overlays Ranked for {date_str} {venue}** 🏆\n\n"
    
# Filter to only show the top 15 by Edge/EV for Discord limits.
    # ENFORCING OUR OPTIMIZED RULES: 3.0 <= Odds <= 8.0 AND EV > 0
    top_overlays = [
        x for x in all_overlays 
        if x['EV'] > 0 and 3.0 <= float(x['Odds']) <= 8.0 
    ][:15]

    if not top_overlays:
         msg += "No elite Tier 1 overlays found (Odds 3.0-8.0, EV > 0).\n"
    
    for i, overlay in enumerate(top_overlays):
        msg += f"**{i+1}. R{overlay['Race']} - {overlay['Horse']}**\n"
        msg += f"> 📊 Odds: {overlay['Odds']} | 🎯 True Prob: {overlay['True Prob']} | � Edge: {overlay['Edge']} | �💰 EV: **{overlay['EV']:.2f}**\n\n"
        
    if len(msg) > 2000:
        chunks = [msg[i:i+1990] for i in range(0, len(msg), 1990)]
        for chunk in chunks:
            await ctx.send(chunk)
    else:
        await ctx.send(msg)

@bot.command(name="analyze")
async def analyze_race(ctx, *, query: str = None):
    """
    Analyzes a race or horse based on the trained data and Gemini.
    Usage: !analyze race 2026-01-25_Race1
    Usage: !analyze horse BEAUTY GEMINI
    """
    if query is None:
        await ctx.send("Please provide a race ID or horse name to analyze. Example: `!analyze BEAUTY GEMINI` or `!analyze 2026-01-25_Race1`")
        return

    df = get_data()
    if df is None:
        await ctx.send("Data not loaded. Please run the model training script first.")
        return

    await ctx.send(f"🏇 Analyzing your request: '{query}'... Please wait.")

    query_upper = query.upper()
    filtered_data = None
    context_type = ""
    
    # 1. Try to find a specific race_id (e.g., 2026-01-25_Race1)
    if "RACE" in query_upper and "_" in query:
        words = query.split()
        for word in words:
            if "_" in word:
                race_id = word
                filtered_data = df[df['race_id'] == race_id]
                context_type = f"Race {race_id}"
                break
    
    # 2. Try to find a specific horse name
    if filtered_data is None or len(filtered_data) == 0:
        # Remove common words to isolate the horse name
        horse_name = query_upper.replace("HORSE", "").replace("ANALYZE", "").strip()
        
        # Exact match first
        filtered_data = df[df['horse_name'].str.upper() == horse_name]
        
        # Partial match if exact fails
        if len(filtered_data) == 0:
            filtered_data = df[df['horse_name'].str.upper().str.contains(horse_name, na=False)]
            
        if len(filtered_data) > 0:
            # Get the actual matched horse name
            matched_horse = filtered_data.iloc[0]['horse_name']
            context_type = f"Horse {matched_horse}"
            # Sort by date descending to get the most recent races
            filtered_data = filtered_data.sort_values(by='race_date', ascending=False).head(10)

    if filtered_data is None or len(filtered_data) == 0:
        await ctx.send(f"Could not find any data matching '{query}'. Try specifying a race_id (e.g., `2026-01-25_Race1`) or a horse name (e.g., `BEAUTY GEMINI`).")
        return

    # Format the data into a readable string for the LLM
    # Calculate Expected Value (EV) if not already present
    # Use .loc to avoid SettingWithCopyWarning
    if 'expected_value' not in filtered_data.columns:
        filtered_data = filtered_data.copy()
        filtered_data.loc[:, 'expected_value'] = (filtered_data['true_prob'] * filtered_data['win_odds']) - 1

    cols_to_include = [
        'race_id', 'race_date', 'horse_name', 'finishing_time', 'win_odds', 
        'true_prob', 'true_market_prob', 'prob_edge', 'implied_prob', 'expected_value', 'pred_score', 'distance', 'barrier_draw',
        'weight_carried', 'jockey_win_rate_50', 'trainer_win_rate_50'
    ]
    
    # Only include columns that actually exist in the dataframe
    actual_cols = [c for c in cols_to_include if c in filtered_data.columns]

    # Convert to string, rounding floats for readability
    # Limit the number of rows to prevent the prompt from getting too large
    # If it's a race, we want all horses (usually ~14). If it's a horse, we already limited to 10.
    max_rows = 20
    if len(filtered_data) > max_rows:
        filtered_data = filtered_data.head(max_rows)
        
    data_str = filtered_data[actual_cols].round(4).to_string(index=False)
    
    # Construct the prompt
    prompt = f"""
{PROFESSIONAL_HANDICAPPER_PROMPT}

Here is the quantitative data for {context_type} from our LightGBM model:

{data_str}

Definitions:
- 'true_prob': The model's calculated, frictionless probability of winning.
- 'implied_prob': The probability implied by the public 'win_odds', which includes the HKJC takeout (~1.17 overround).
- 'true_market_prob': The public probability perfectly normalized to sum to 1.0 (stripping the ~1.17 HKJC takeout vig).
- 'prob_edge': true_prob minus true_market_prob. A positive prob_edge (especially > 0.05) means our model severely out-handicapped the public.
- 'expected_value': The EV of a $1 bet, driven by the math.
- 'pred_score': The model's raw LightGBM ranking score (higher is better).
- If prob_edge > 0.05 AND win_odds are between 3.0 and 8.0, this is an elite Tier 1 Value Bet.

Based on this data, please answer the user's query: "{query}"

Task:
1. Identify the false favorites (Low EV, bad draw/map).
2. Identify the true value bets (High EV, good map/jockey).
3. Identify any tote anomalies (Win vs Place odds disparities, if applicable).
4. Output a final recommended betting strategy (e.g., Win Bet, Quinella, Trio) that maximizes our LightGBM edge while hedging against bad luck.
"""

    try:
        # Run the synchronous Gemini API call in a separate thread to avoid blocking the Discord event loop
        response = await asyncio.to_thread(
            genai_client.models.generate_content,
            model='gemini-2.5-flash',
            contents=prompt,
        )
        
        # Discord has a 2000 character limit per message
        reply = response.text
        if len(reply) > 2000:
            # Split into chunks
            chunks = [reply[i:i+1990] for i in range(0, len(reply), 1990)]
            for chunk in chunks:
                await ctx.send(chunk)
        else:
            await ctx.send(reply)
            
    except Exception as e:
        await ctx.send(f"An error occurred while calling Gemini: {str(e)}")

@bot.command(name="bot_help", aliases=["help", "commands"])
async def show_help(ctx):
    """Shows all available commands and how to use them."""
    help_text = """
🏇 **HKJC AI Analyst Bot - Available Commands** 🏇

**🔥 LIVE ANALYSIS (Mode 1, 2 & 3)**
`!live <YYYY-MM-DD> <Venue> <Race_Num> [Mode]`
*Scrapes current public odds, Form Guide, SpeedPRO, and merges with model predictions to find value.*
*Modes: 'all' (default, uses ML models) or 'public' (purely web scraped human analysis)*
*Example:* `!live 2026-03-01 ST 6 public`

**🔍 OVERLAY SCANNER**
`!scan_overlays <YYYY-MM-DD> <Venue> [Max_Races]`
*Scans all races at a meeting sequentially to flag the biggest value bets.*
*Example:* `!scan_overlays 2026-03-01 ST 11`

**📊 HISTORICAL ANALYSIS**
`!analyze race <Race_ID>`
*Analyzes a past race purely from model predictions.*
*Example:* `!analyze race 2026-01-25_Race1`

`!analyze horse <Horse_Name>`
*Analyzes a specific horse's past performances.*
*Example:* `!analyze horse BEAUTY GEMINI`

**⚙️ SYSTEM**
`!reload`
*Reloads the predictions data file.*

**💬 CHAT & VISION**
You can also tag the bot `@BotName` or send it a direct message with an image (e.g., of a track layout or odds screen) to chat with Gemini organically!
"""
    await ctx.send(help_text)

@bot.command(name="reload")
async def reload_data_command(ctx):
    """Reloads the predictions CSV file."""
    load_data()
    df = get_data()
    if df is not None:
        await ctx.send(f"Successfully reloaded data. {len(df)} rows available.")
    else:
        await ctx.send("Failed to load data. Check if data/all_predictions.csv exists.")

if __name__ == "__main__":
    print("Starting Discord Bot...")
    bot.run(DISCORD_TOKEN)
