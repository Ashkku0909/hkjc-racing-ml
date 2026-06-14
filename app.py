"""
HKJC Racing Analysis — Streamlit Web UI
========================================
Educational tool for studying horse racing data & ML predictions.
NOT for actual gambling. See DISCLAIMER.md
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

st.set_page_config(page_title="HKJC Racing Analysis", page_icon="🏇", layout="wide")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════
@st.cache_data
def load_predictions():
    path = "data/all_predictions.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['race_date'] = pd.to_datetime(df['race_date'], errors='coerce')
        return df
    return None

@st.cache_data
def load_features():
    path = "data/model_features.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def get_all_horses(_df):
    return sorted(_df['horse_name'].dropna().unique())

@st.cache_data
def get_all_race_ids(_df):
    return _df[['race_id','race_date','track','distance']].drop_duplicates('race_id').sort_values('race_date', ascending=False)

# ═══════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .disclaimer-banner {
        background: linear-gradient(135deg, #ff6b35 0%, #e63946 100%);
        color: white; padding: 14px 24px; border-radius: 10px;
        margin-bottom: 24px; font-weight: 600; text-align: center;
        box-shadow: 0 4px 15px rgba(230,57,70,0.25);
    }
    .disclaimer-banner a { color: white; text-decoration: underline; }

    .hero-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 32px; border-radius: 16px; margin-bottom: 24px;
        text-align: center; box-shadow: 0 8px 30px rgba(102,126,234,0.3);
    }
    .hero-card h1 { font-size: 2.2rem; margin: 0; }
    .hero-card p { opacity: 0.9; margin-top: 8px; font-size: 1.1rem; }

    .stat-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 24px; border-radius: 12px; text-align: center;
        border: 1px solid #dee2e6; transition: transform 0.15s;
    }
    .stat-box:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.08); }
    .stat-box .value { font-size: 2rem; font-weight: 700; color: #4361ee; }
    .stat-box .label { font-size: 0.85rem; color: #6c757d; margin-top: 4px; }

    .section-title {
        font-size: 1.3rem; font-weight: 700; color: #212529;
        padding-bottom: 8px; border-bottom: 3px solid #4361ee; display: inline-block;
        margin-bottom: 16px;
    }

    .card {
        background: white; border: 1px solid #e9ecef; border-radius: 12px;
        padding: 20px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    footer { visibility: hidden; }
    .stButton > button {
        background: linear-gradient(135deg, #4361ee, #3a0ca3);
        color: white; border: none; border-radius: 8px; padding: 8px 24px;
        font-weight: 600; transition: all 0.2s;
    }
    .stButton > button:hover { transform: scale(1.02); box-shadow: 0 4px 12px rgba(67,97,238,0.4); }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:16px 0;">
        <span style="font-size:3rem;">🏇</span>
        <h2 style="margin:4px 0;">HKJC Racing</h2>
        <p style="color:#6c757d;font-size:0.85rem;">ML Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio("", [
        "🏠 Home", "📊 Dashboard", "🔍 Race Lookup",
        "🐴 Horse Analysis", "🤖 AI Analysis", "📁 Data Overview"
    ], label_visibility="collapsed")

    st.divider()
    df = load_predictions()
    if df is not None:
        st.success(f"✅ {len(df):,} rows loaded")
        st.caption(f"📅 {df['race_id'].nunique():,} races · {df['horse_name'].nunique():,} horses")
    else:
        st.error("❌ No data")
        st.code("python main.py", language="bash")

    st.divider()
    st.caption("⚠️ Educational Use Only · [Disclaimer](DISCLAIMER.md)")

# ═══════════════════════════════════════════
# DISCLAIMER
# ═══════════════════════════════════════════
st.markdown("""
<div class="disclaimer-banner">
    ⚠️ EDUCATIONAL USE ONLY — Studying ML & statistics. NOT for gambling.
    <a href="DISCLAIMER.md">Full Disclaimer</a>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class="hero-card">
        <h1>🏇 HKJC Racing ML Platform</h1>
        <p>Explore Hong Kong horse racing data through machine learning & AI</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card">
            <h3>🔍 Race Lookup</h3>
            <p>Search any historical HKJC race. See full field data, model win probabilities, and visual charts.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
            <h3>🐴 Horse Deep-Dive</h3>
            <p>Analyze any horse's complete career — win/place rates, performance trends, jockey & trainer patterns.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="card">
            <h3>🤖 AI Analysis</h3>
            <p>Let Google Gemini break down race data with natural-language insights. Powered by your own API key.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    if df is not None:
        st.markdown('<p class="section-title">📊 Quick Stats</p>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="stat-box"><div class="value">{df["race_id"].nunique():,}</div><div class="label">Total Races</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-box"><div class="value">{df["horse_name"].nunique():,}</div><div class="label">Unique Horses</div></div>', unsafe_allow_html=True)
        with c3:
            tracks = df['track'].nunique()
            st.markdown(f'<div class="stat-box"><div class="value">{tracks}</div><div class="label">Race Tracks</div></div>', unsafe_allow_html=True)
        with c4:
            date_range = f"{df['race_date'].min().year}–{df['race_date'].max().year}"
            st.markdown(f'<div class="stat-box"><div class="value">{date_range}</div><div class="label">Years Covered</div></div>', unsafe_allow_html=True)

        st.divider()
        st.markdown('<p class="section-title">🚀 Quick Actions</p>', unsafe_allow_html=True)
        q1, q2 = st.columns(2)
        with q1:
            quick_search = st.text_input("Jump to a horse:", placeholder="Type horse name...", key="home_horse")
            if quick_search:
                st.switch_page("app.py")  # fallback
                st.session_state['horse_search'] = quick_search
        with q2:
            quick_race = st.text_input("Jump to a race:", placeholder="e.g. 2020-06-07_Race5", key="home_race")
            if quick_race:
                st.session_state['race_search'] = quick_race
    else:
        st.warning("No data loaded yet. Run the pipeline:")
        st.code("python main.py", language="bash")
        st.markdown("""
        Or use the one-click setup:
        - **Windows**: `setup.bat`
        - **Mac/Linux**: `./setup.sh`
        """)

    st.divider()
    st.markdown("""
    <div style="text-align:center; color:#6c757d; padding:20px;">
        <p>📖 <a href="README.md">Documentation</a> · 
        ⚠️ <a href="DISCLAIMER.md">Disclaimer</a> · 
        🤝 <a href="CONTRIBUTING.md">Contribute</a></p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.markdown('<p class="section-title">📊 Dashboard</p>', unsafe_allow_html=True)

    if df is None:
        st.warning("No data loaded. Run `python main.py` first.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="stat-box"><div class="value">{df["race_id"].nunique():,}</div><div class="label">Races</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-box"><div class="value">{df["horse_name"].nunique():,}</div><div class="label">Horses</div></div>', unsafe_allow_html=True)
    with c3:
        jockeys = df['jockey'].nunique() if 'jockey' in df.columns else '—'
        st.markdown(f'<div class="stat-box"><div class="value">{jockeys:,}</div><div class="label">Jockeys</div></div>', unsafe_allow_html=True)
    with c4:
        trainers = df['trainer'].nunique() if 'trainer' in df.columns else '—'
        st.markdown(f'<div class="stat-box"><div class="value">{trainers:,}</div><div class="label">Trainers</div></div>', unsafe_allow_html=True)

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📈 Trends", "🏆 Top Performers", "🔍 Search"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<p style="font-weight:600;">Races Per Month</p>', unsafe_allow_html=True)
            rpm = df.groupby(df['race_date'].dt.to_period('M'))['race_id'].nunique()
            rpm.index = rpm.index.astype(str)
            st.line_chart(rpm, height=300)
        with col_b:
            st.markdown('<p style="font-weight:600;">Track Distribution</p>', unsafe_allow_html=True)
            track_counts = df['track'].value_counts()
            st.bar_chart(track_counts, height=300)

    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<p style="font-weight:600;">Top Horses (Win Rate)</p>', unsafe_allow_html=True)
            if 'horse_win_rate_50' in df.columns:
                top = df.groupby('horse_name')['horse_win_rate_50'].mean().nlargest(10)
                st.bar_chart(top, height=300)
        with col_b:
            st.markdown('<p style="font-weight:600;">Top Jockeys (Win Rate)</p>', unsafe_allow_html=True)
            if 'jockey_win_rate_50' in df.columns:
                top_j = df.groupby('jockey')['jockey_win_rate_50'].mean().nlargest(10)
                st.bar_chart(top_j, height=300)

    with tab3:
        search = st.text_input("Search:", placeholder="Horse name or Race ID...", key="dash_search")
        if search:
            su = search.upper()
            if "_" in search:
                results = df[df['race_id'].str.upper() == su]
            else:
                results = df[df['horse_name'].str.upper().str.contains(su, na=False)]
            if len(results) > 0:
                st.success(f"{len(results)} results")
                st.dataframe(results[['race_date','horse_name','track','distance','jockey','trainer','win_odds','finish_position']].head(20),
                             use_container_width=True, hide_index=True)
            else:
                st.warning("No results")

# ═══════════════════════════════════════════════════════
# PAGE: RACE LOOKUP
# ═══════════════════════════════════════════════════════
elif page == "🔍 Race Lookup":
    st.markdown('<p class="section-title">🔍 Race Lookup</p>', unsafe_allow_html=True)
    if df is None: st.warning("No data."); st.stop()

    # Check for session state redirect from home
    default_race = st.session_state.pop('race_search', '')

    c1, c2 = st.columns([3, 1])
    with c1:
        race_search = st.text_input("Race ID:", placeholder="YYYY-MM-DD_RaceN", value=default_race)
    with c2:
        st.write(""); st.write("")
        clicked = st.button("🔍 Find Race", use_container_width=True)

    if race_search:
        race_df = df[df['race_id'].str.upper() == race_search.upper()]
        if len(race_df) > 0:
            info = race_df.iloc[0]
            st.markdown(f"""
            <div class="card">
                <h3>🏁 {info['track']} · {info['distance']}m · {info['race_date'].strftime('%B %d, %Y')}</h3>
                <p style="color:#6c757d;">Class: {info.get('race_class','—')} · Runners: {len(race_df)}</p>
            </div>
            """, unsafe_allow_html=True)

            show_cols = ['horse_name','barrier_draw','weight_carried','jockey','trainer','win_odds','finish_position','finishing_time']
            for extra in ['true_prob','pred_score','expected_value','prob_edge']:
                if extra in race_df.columns: show_cols.insert(-1, extra)
            available = [c for c in show_cols if c in race_df.columns]
            display = race_df[available].copy()
            if 'true_prob' in display.columns: display['true_prob'] = (display['true_prob']*100).round(1).astype(str)+'%'
            if 'expected_value' in display.columns: display['expected_value'] = display['expected_value'].round(3)
            if 'prob_edge' in display.columns: display['prob_edge'] = (display['prob_edge']*100).round(1).astype(str)+'%'
            if 'pred_score' in display.columns: display['pred_score'] = display['pred_score'].round(4)

            st.dataframe(display, use_container_width=True, hide_index=True)

            if 'true_prob' in race_df.columns:
                col_chart, col_info = st.columns([2, 1])
                with col_chart:
                    st.markdown('<p style="font-weight:600;">Model Win Probability (%)</p>', unsafe_allow_html=True)
                    chart = race_df[['horse_name','true_prob']].copy()
                    chart['true_prob'] = chart['true_prob']*100
                    st.bar_chart(chart.set_index('horse_name'), height=350)
                with col_info:
                    st.markdown('<p style="font-weight:600;">Race Insights</p>', unsafe_allow_html=True)
                    if 'true_prob' in race_df.columns:
                        top_horse = race_df.loc[race_df['true_prob'].idxmax()]
                        st.metric("Model's Top Pick", top_horse['horse_name'])
                        st.metric("Top Probability", f"{top_horse['true_prob']*100:.1f}%")
                    actual_winner = race_df[race_df['finish_position'].astype(str).str.match(r'^1(\s|$)')]
                    if len(actual_winner) > 0:
                        st.metric("Actual Winner", actual_winner.iloc[0]['horse_name'])
        else:
            st.warning(f"Race '{race_search}' not found. Format: YYYY-MM-DD_RaceN")

    st.divider()
    st.markdown('<p style="font-weight:600;">📋 Recent Races</p>', unsafe_allow_html=True)
    recent = get_all_race_ids(df).head(30)
    st.dataframe(recent, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════
# PAGE: HORSE ANALYSIS
# ═══════════════════════════════════════════════════════
elif page == "🐴 Horse Analysis":
    st.markdown('<p class="section-title">🐴 Horse Analysis</p>', unsafe_allow_html=True)
    if df is None: st.warning("No data."); st.stop()

    default_horse = st.session_state.pop('horse_search', '')
    horse_name = st.text_input("Horse Name:", placeholder="e.g. GOLDEN SIXTY", value=default_horse)

    if horse_name:
        hdf = df[df['horse_name'].str.upper().str.contains(horse_name.upper(), na=False)]
        if len(hdf) > 0:
            name = hdf.iloc[0]['horse_name']
            hdf = hdf.sort_values('race_date', ascending=False)

            # Hero card
            wins = hdf['finish_position'].astype(str).str.match(r'^1(\s|$)').sum()
            tries = len(hdf)
            places = (pd.to_numeric(hdf['finish_position'].astype(str).str.extract(r'^(\d+)')[0], errors='coerce').fillna(99) <= 3).sum()
            win_pct = (wins/tries*100) if tries else 0
            place_pct = (places/tries*100) if tries else 0

            st.markdown(f"""
            <div class="card" style="background:linear-gradient(135deg,#1a1a2e,#16213e);color:white;">
                <h2>🐴 {name}</h2>
                <div style="display:flex;gap:32px;margin-top:12px;flex-wrap:wrap;">
                    <div><span style="font-size:1.8rem;font-weight:700;">{tries}</span><br><small>Career Starts</small></div>
                    <div><span style="font-size:1.8rem;font-weight:700;">{wins}</span><br><small>Wins ({win_pct:.0f}%)</small></div>
                    <div><span style="font-size:1.8rem;font-weight:700;">{places}</span><br><small>Top 3 ({place_pct:.0f}%)</small></div>
                    <div><span style="font-size:1.8rem;font-weight:700;">{hdf['win_odds'].dropna().mean():.1f}</span><br><small>Avg Odds</small></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            tab1, tab2 = st.tabs(["📋 Race History", "📈 Charts"])
            with tab1:
                show = ['race_date','track','distance','jockey','trainer','weight_carried','barrier_draw','win_odds','finish_position']
                if 'true_prob' in hdf.columns: show.insert(-1,'true_prob')
                avail = [c for c in show if c in hdf.columns]
                st.dataframe(hdf[avail], use_container_width=True, hide_index=True)

            with tab2:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<p style="font-weight:600;">Finishing Position Trend</p>', unsafe_allow_html=True)
                    chart_df = hdf.copy()
                    chart_df['pos'] = pd.to_numeric(chart_df['finish_position'].astype(str).str.extract(r'^(\d+)')[0], errors='coerce')
                    chart_df = chart_df.sort_values('race_date')
                    st.line_chart(chart_df.set_index('race_date')['pos'], height=300)
                with c2:
                    st.markdown('<p style="font-weight:600;">Odds Distribution</p>', unsafe_allow_html=True)
                    odds = hdf['win_odds'].dropna()
                    if len(odds) > 0:
                        st.scatter_chart(pd.DataFrame({'race': range(len(odds)), 'odds': odds.values}).set_index('race'), height=300)
        else:
            st.warning(f"No horse matching '{horse_name}'")

    st.divider()
    st.markdown('<p style="font-weight:600;">🐎 Browse Horses</p>', unsafe_allow_html=True)
    all_h = get_all_horses(df)
    filt = st.text_input("Filter:", placeholder="Type to filter...", key="horse_filter")
    if filt:
        filtered = [h for h in all_h if filt.upper() in h.upper()]
        st.caption(f"{len(filtered)} of {len(all_h)} horses")
        st.dataframe(pd.DataFrame(filtered, columns=['Horse Name']), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════
# PAGE: AI ANALYSIS
# ═══════════════════════════════════════════════════════
elif page == "🤖 AI Analysis":
    st.markdown('<p class="section-title">🤖 AI-Powered Analysis</p>', unsafe_allow_html=True)
    st.info("Uses **Google Gemini** to analyze race data in plain English. Requires `GEMINI_API_KEY` in `.env`.")

    if df is None: st.warning("No data."); st.stop()

    from dotenv import load_dotenv; load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ GEMINI_API_KEY not found in `.env`")
        st.code("GEMINI_API_KEY=your_key_here", language="bash")
        st.stop()

    mode = st.selectbox("What would you like to analyze?", ["A specific race", "A specific horse"])
    st.caption("⚠️ Analysis is for educational/statistical study only — not betting advice.")

    if mode == "A specific race":
        rid = st.text_input("Race ID:", placeholder="2020-06-07_Race5", key="ai_race")
        if rid and st.button("🤖 Analyze", type="primary", key="ai_race_btn"):
            rdf = df[df['race_id'].str.upper() == rid.upper()]
            if len(rdf) > 0:
                with st.spinner("Gemini is analyzing..."):
                    try:
                        from google import genai
                        client = genai.Client(api_key=api_key)
                        resp = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=f"""Educational horse racing data analysis. No betting advice.

Race: {rdf.iloc[0]['track']}, {rdf.iloc[0]['distance']}m, {rdf.iloc[0]['race_date'].strftime('%Y-%m-%d')}

Data:
{rdf.to_string(index=False)}

Provide a clear, structured analysis: (1) Race overview, (2) Key statistical insights, (3) What data scientists can learn from this race. Keep it under 500 words."""
                        )
                        st.markdown(f'<div class="card">{resp.text}</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Gemini error: {e}")
            else: st.warning("Race not found")

    else:
        hname = st.text_input("Horse name:", placeholder="GOLDEN SIXTY", key="ai_horse")
        if hname and st.button("🤖 Analyze", type="primary", key="ai_horse_btn"):
            hdf = df[df['horse_name'].str.upper().str.contains(hname.upper(), na=False)]
            if len(hdf) > 0:
                with st.spinner("Gemini is analyzing..."):
                    try:
                        from google import genai
                        client = genai.Client(api_key=api_key)
                        resp = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=f"""Educational horse racing data analysis. No betting advice.

Horse: {hdf.iloc[0]['horse_name']}
Career starts: {len(hdf)}

Recent data:
{hdf.head(15).to_string(index=False)}

Provide a clear analysis: (1) Career summary, (2) Notable patterns (distance/track preferences), (3) Statistical takeaways for data science study. Under 500 words."""
                        )
                        st.markdown(f'<div class="card">{resp.text}</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Gemini error: {e}")
            else: st.warning("Horse not found")

# ═══════════════════════════════════════════════════════
# PAGE: DATA OVERVIEW
# ═══════════════════════════════════════════════════════
elif page == "📁 Data Overview":
    st.markdown('<p class="section-title">📁 Data Overview</p>', unsafe_allow_html=True)
    if df is None: st.warning("No data."); st.stop()

    tabs = st.tabs(["📋 Browse", "📊 Stats", "ℹ️ Columns", "🧬 Features"])

    with tabs[0]:
        st.dataframe(df.head(100), use_container_width=True)
        csv = df.head(500).to_csv(index=False)
        st.download_button("📥 Download Sample CSV", csv, "hkjc_sample.csv", "text/csv")

    with tabs[1]:
        st.dataframe(df.describe(), use_container_width=True)

    with tabs[2]:
        info = {
            "race_id":"Unique race ID (date + race number)",
            "race_date":"Date of race","track":"Sha Tin / Happy Valley",
            "distance":"Meters","race_class":"Class 1–5 / Group",
            "horse_name":"Horse name","jockey":"Jockey","trainer":"Trainer",
            "weight_carried":"Weight (lbs)","barrier_draw":"Gate number",
            "win_odds":"Tote win odds (decimal)","finish_position":"Finish pos",
            "finishing_time":"Seconds","true_prob":"Model win probability",
            "expected_value":"Theoretical EV","horse_win_rate_50":"Horse WR (50r)",
            "jockey_win_rate_50":"Jockey WR (50r)","trainer_win_rate_50":"Trainer WR (50r)",
            "speed_figure":"Normalized speed","days_since_last_race":"Rest days",
        }
        st.dataframe(pd.DataFrame(info.items(), columns=['Column','Description']), use_container_width=True, hide_index=True)

    with tabs[3]:
        feat_df = load_features()
        if feat_df is not None:
            st.metric("Feature Rows", f"{len(feat_df):,}")
            st.dataframe(feat_df.head(50), use_container_width=True)
        else:
            st.info("Run `python data_pipeline/feature_engineering.py`")


