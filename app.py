"""
app.py — The Interface
========================
Streamlit UI for the UAE AI Tourism Recommender.

Features:
    - Free-text semantic search (powered by ai_engine.py)
    - Interactive Folium HeatMap driven by AI relevance scores
    - Clickable CircleMarkers with location details
    - Gemini LLM-generated travel advice with waterfall fallback
    - Auto-detects teammate's CSV (uae_pois.csv) or falls back to placeholder

Run:
    streamlit run app.py
"""

import os
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from ai_engine import (
    UAETourismEngine,
    load_sentence_model,
    generate_ai_advice,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UAE Smart Tourism AI",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# CACHED ENGINE INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model (first run only)...")
def initialize_engine():
    """
    Load the Sentence Transformer model and initialize the engine ONCE.

    Streamlit reruns this entire script on every user interaction (click,
    toggle, etc). The @st.cache_resource decorator ensures the heavy
    model loading (~80 MB download) and embedding pre-computation only
    happen on the FIRST run. Every subsequent rerun reuses the cached
    engine object instantly.

    CSV Swap Logic:
        If uae_pois.csv exists in the project directory (delivered by
        Member A), the engine loads it automatically. Otherwise it falls
        back to the built-in 18-location placeholder dataset.
    """
    model = load_sentence_model()

    # --- Check for teammate's CSV ---
    csv_path = os.path.join(os.path.dirname(__file__), "uae_pois.csv")
    if os.path.exists(csv_path):
        df = UAETourismEngine.load_from_csv(csv_path)
        return UAETourismEngine(model, df=df)
    else:
        return UAETourismEngine(model, df=None)  # placeholder


engine = initialize_engine()


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE (persists across Streamlit reruns)
# ─────────────────────────────────────────────────────────────────────────────
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "search_query" not in st.session_state:
    st.session_state.search_query = ""
if "ai_advice" not in st.session_state:
    st.session_state.ai_advice = None
# Option 2: Cache AI responses to avoid repeat API calls for same query
if "advice_cache" not in st.session_state:
    st.session_state.advice_cache = {}  # {query_string: advice_text}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Build the Folium map
# ─────────────────────────────────────────────────────────────────────────────
def build_map(
    heat_data=None,
    results_df=None,
    all_locations=None,
    center=None,
    zoom=8,
):
    """
    Build a Folium map with optional HeatMap overlay and markers.

    Parameters
    ----------
    heat_data : list or None
        [[lat, lon, intensity], ...] for the HeatMap layer.
    results_df : pd.DataFrame or None
        Search results to render as CircleMarkers with popups.
    all_locations : pd.DataFrame or None
        Full dataset for the default view (small cyan dots).
    center : list or None
        [lat, lon] center point. Auto-calculated if None.
    zoom : int
        Initial zoom level.
    """
    # Auto-calculate center from data
    if center is None:
        source = results_df if results_df is not None else all_locations
        if source is not None and len(source) > 0:
            center = [source["Lat"].mean(), source["Lon"].mean()]
        else:
            center = [24.45, 54.65]  # UAE fallback

    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="CartoDB dark_matter",  # Dark theme — makes heatmap glow
    )

    # ── HeatMap Layer (search results) ──
    if heat_data:
        HeatMap(
            heat_data,
            radius=40,
            blur=30,
            max_zoom=13,
            min_opacity=0.3,
            gradient={
                0.2: "blue",
                0.4: "lime",
                0.6: "yellow",
                0.8: "orange",
                1.0: "red",
            },
        ).add_to(m)

    # ── CircleMarkers with popups (search results) ──
    if results_df is not None:
        for _, row in results_df.iterrows():
            relevance = row.get("Final_Relevance", 0)
            popup_html = (
                f"<b>{row['Name']}</b><br>"
                f"Type: {row['Type']}<br>"
                f"Relevance: {relevance:.2f}<br>"
                f"<small>{row['Description']}</small>"
            )
            folium.CircleMarker(
                location=[row["Lat"], row["Lon"]],
                radius=10 + (relevance * 10),  # Bigger = more relevant
                popup=folium.Popup(popup_html, max_width=250),
                color="white",
                weight=2,
                fill=True,
                fill_color="orange",
                fill_opacity=0.8,
            ).add_to(m)

    # ── Default dots (all locations, no search) ──
    if all_locations is not None and results_df is None:
        for _, row in all_locations.iterrows():
            folium.CircleMarker(
                location=[row["Lat"], row["Lon"]],
                radius=6,
                popup=folium.Popup(
                    f"<b>{row['Name']}</b><br>{row['Description']}",
                    max_width=200,
                ),
                color="cyan",
                weight=1,
                fill=True,
                fill_color="cyan",
                fill_opacity=0.5,
            ).add_to(m)

    return m


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — User Controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── API Key Resolution (3-tier priority) ──
    st.header("🔑 API Configuration")

    _auto_key = None
    _key_source = None

    # Priority 1: Streamlit Secrets (.streamlit/secrets.toml)
    try:
        _auto_key = st.secrets["GOOGLE_API_KEY"]
        _key_source = "Streamlit Secrets"
    except (FileNotFoundError, KeyError):
        pass

    # Priority 2: Environment Variable
    if not _auto_key:
        _auto_key = os.environ.get("GOOGLE_API_KEY")
        if _auto_key:
            _key_source = "Environment Variable"

    # Priority 3: Manual sidebar input (fallback)
    if _auto_key:
        st.success(f"API Key loaded from {_key_source}")
        api_key = _auto_key
    else:
        api_key = st.text_input(
            "Enter Gemini API Key:",
            type="password",
            help="Get a free key at https://aistudio.google.com/apikey",
        )
        if not api_key:
            st.warning("Paste your Gemini API Key to enable AI travel advice.")

    st.divider()

    # ── Preferences ──
    st.header("🔍 Your Preferences")
    st.caption("Describe what you want — the AI understands meaning, not just keywords.")

    user_query = st.text_area(
        "Describe your ideal UAE experience:",
        placeholder="e.g., quiet sunset near water, or fast cars and adrenaline...",
        height=100,
    )

    col1, col2 = st.columns(2)
    with col1:
        prefer_hidden = st.toggle("Hidden Gems", value=True, help="Boost lesser-known spots")
    with col2:
        avoid_heat = st.toggle("Stay Indoors", value=False, help="Only indoor locations")

    # Option 1: Make AI advice optional to save API quota
    enable_ai_advice = st.toggle(
        "Enable AI Advice",
        value=True,
        help="Disable to save API quota. Search still works without it."
    )

    search_clicked = st.button(
        "🚀 Find My Places",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    # Placeholder for AI advice (filled after search)
    advice_container = st.container()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA — Title
# ─────────────────────────────────────────────────────────────────────────────
st.title("🗺️ UAE AI Tourism Explorer")
st.caption(
    "Powered by **Vector Semantic Search** · all-MiniLM-L6-v2 · "
    "Cosine Similarity · Hybrid Anti-Bias Scoring"
)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOGIC — Search (stores results in session_state)
# ─────────────────────────────────────────────────────────────────────────────
if search_clicked and user_query.strip():
    # ── Run semantic search ──
    with st.spinner("Computing semantic vectors..."):
        results = engine.semantic_search(
            user_query=user_query.strip(),
            top_k=5,
            prefer_hidden_gems=prefer_hidden,
            avoid_heat=avoid_heat,
        )
    st.session_state.search_results = results
    st.session_state.search_query = user_query.strip()

    # ── Pre-fetch Gemini advice (with caching + toggle) ──
    query_key = user_query.strip().lower()  # Normalize for cache lookup

    if not enable_ai_advice:
        # Option 1: User disabled AI advice to save quota
        st.session_state.ai_advice = None
    elif query_key in st.session_state.advice_cache:
        # Option 2: Return cached response (no API call)
        st.session_state.ai_advice = st.session_state.advice_cache[query_key]
    elif api_key and len(results) > 0:
        # Fresh API call needed
        with st.spinner("Asking Gemini..."):
            advice = generate_ai_advice(results, user_query.strip(), api_key=api_key)
            st.session_state.ai_advice = advice
            # Cache the response for future identical queries
            st.session_state.advice_cache[query_key] = advice
    else:
        st.session_state.ai_advice = None

    # Force rerun to display fresh results
    st.rerun()

elif search_clicked:
    st.warning("✍️ Please describe what kind of experience you are looking for.")


# ─────────────────────────────────────────────────────────────────────────────
# RENDER — Display results from session_state (survives reruns)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.search_results is not None and len(st.session_state.search_results) > 0:
    results = st.session_state.search_results

    # Show current search query for clarity
    st.success(f"**Showing results for:** \"{st.session_state.search_query}\"")

    # ── Heatmap ──
    st.subheader("📍 Relevance Heat Map")
    st.caption(
        "Glow intensity = AI semantic relevance. "
        "Brighter areas match your query more closely."
    )

    heat_data = engine.get_map_data(results)
    m = build_map(heat_data=heat_data, results_df=results)
    st_folium(m, use_container_width=True, height=520)

    # ── Results Table ──
    st.subheader("🏆 Top Recommendations")
    display_df = results[
        ["Name", "Type", "Semantic_Score", "Final_Relevance", "Description"]
    ].copy()
    display_df.index = range(1, len(display_df) + 1)  # 1-based ranking
    display_df.index.name = "Rank"

    st.dataframe(
        display_df.style.format({
            "Semantic_Score": "{:.3f}",
            "Final_Relevance": "{:.3f}",
        }).background_gradient(
            subset=["Final_Relevance"],
            cmap="YlOrRd",
        ),
        use_container_width=True,
    )

    # ── Gemini AI Advice (in sidebar) ──
    with advice_container:
        st.subheader("🤖 AI Travel Advice")
        if st.session_state.ai_advice:
            st.info(st.session_state.ai_advice)
        elif not api_key:
            st.warning("Enter an API key in the sidebar to get AI travel advice.")

elif st.session_state.search_results is not None:
    st.warning(
        "No locations matched your filters. "
        "Try unchecking 'Stay Indoors' for more results."
    )

else:
    # ── Default state: show all locations ──
    st.info(
        "👈 Use the sidebar to describe your ideal experience, "
        "then click **Find My Places**."
    )
    all_locs = engine.get_all_locations()
    m = build_map(all_locations=all_locs)
    st_folium(m, use_container_width=True, height=520)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with Sentence Transformers · Scikit-Learn · Folium · "
    "Google Gemini · Streamlit"
)
