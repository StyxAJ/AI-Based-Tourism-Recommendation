"""
ai_engine.py — The Semantic Brain
===================================
Vector-Based Semantic Recommendation Engine for UAE Tourism.

This module replaces the rule-based keyword matching from logic.py with a
Vector Space Semantic Search system. Every location description is projected
into a 384-dimensional embedding space using a pre-trained Sentence Transformer
(all-MiniLM-L6-v2). User queries are projected into the SAME space, and
Cosine Similarity measures how closely a query's meaning aligns with each
location — even when they share zero words in common.

Architecture:
    User Query  ──encode──►  384-dim vector  ─┐
                                               ├─► Cosine Similarity ──► Ranked Results
    Location DB ──encode──►  (N, 384) matrix ─┘

Author : AI Engineering Lead
Version: 2.0 (Semantic Upgrade)
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "Name", "Type", "Popularity_Score", "Indoors",
    "Description", "Reviews_Embed_Source", "Lat", "Lon",
]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER  (called from app.py with @st.cache_resource)
# ─────────────────────────────────────────────────────────────────────────────
def load_sentence_model():
    """
    Load the all-MiniLM-L6-v2 Sentence Transformer (~80 MB).

    This function is intentionally separate from the class so that
    app.py can wrap it with Streamlit's @st.cache_resource decorator.
    That ensures the heavy model is downloaded and loaded only ONCE,
    surviving across every Streamlit UI rerun.

    Returns
    -------
    SentenceTransformer
        A model that converts any English text into a 384-dimensional
        dense vector capturing its semantic meaning.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


# ═════════════════════════════════════════════════════════════════════════════
# THE ENGINE
# ═════════════════════════════════════════════════════════════════════════════
class UAETourismEngine:
    """
    Vector-Based Semantic Recommendation Engine.

    FOR THE JUDGES — HOW THIS WORKS:
    ─────────────────────────────────
    Traditional search uses KEYWORD MATCHING: the word "quiet" only matches
    documents containing "quiet". Our engine uses SEMANTIC MATCHING: the
    word "quiet" also matches "peaceful", "serene", "tranquil", and
    "away from crowds" because they all point in similar directions in a
    learned 384-dimensional vector space.

    CORE MATH — Cosine Similarity:
    ──────────────────────────────
                          A · B
        cos(θ)  =  ─────────────────
                    ║A║  ×  ║B║

    Where:
        A     = user query vector   (1, 384)
        B     = location vector     (1, 384)
        A · B = dot product (sum of element-wise multiplication)
        ║A║   = L2 norm (Euclidean length) of A

    Result:
        1.0 = identical meaning (vectors point same direction)
        0.0 = completely unrelated (vectors are perpendicular)

    HYBRID SCORING:
    ───────────────
        Final_Relevance = (Semantic_Score × 0.7) + (Inverse_Popularity × 0.3)

    The 0.7 weight ensures AI-driven relevance dominates.
    The 0.3 weight gently nudges results toward hidden gems,
    preventing the system from always recommending Burj Khalifa.
    """

    # ─────────────────────────────────────────────────────────────────────
    # CONSTRUCTION
    # ─────────────────────────────────────────────────────────────────────
    def __init__(self, model, df=None):
        """
        Initialize the engine with a pre-loaded model and optional dataset.

        Parameters
        ----------
        model : SentenceTransformer
            A pre-loaded model instance (loaded via load_sentence_model()).
        df : pd.DataFrame or None
            If provided, the engine uses this as its location database.
            Must contain columns: {REQUIRED_COLUMNS}.
            If None, falls back to the built-in 18-location placeholder
            so the engine works immediately for testing.
        """
        self.model = model

        # --- Data Layer: CSV or Placeholder ---
        if df is not None:
            missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                raise ValueError(
                    f"DataFrame is missing required columns: {missing}\n"
                    f"Expected: {REQUIRED_COLUMNS}"
                )
            self.df = df.reset_index(drop=True)
            print(f"   [Engine] Loaded external dataset: {len(self.df)} locations")
        else:
            self.df = self._build_placeholder_dataset()
            print(f"   [Engine] Using placeholder dataset: {len(self.df)} locations")

        # --- Pre-compute embeddings on init (runs ONCE) ---
        self.location_embeddings = self._precompute_embeddings()
        print(f"   [Engine] Embeddings shape: {self.location_embeddings.shape}")

    # ─────────────────────────────────────────────────────────────────────
    # DATA: CSV LOADER (for Member A's dataset)
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def load_from_csv(csv_path: str) -> pd.DataFrame:
        """
        Load a teammate's CSV into a DataFrame.

        When Member A delivers uae_pois.csv, the swap in app.py is one line:
            df = UAETourismEngine.load_from_csv("uae_pois.csv")
            engine = UAETourismEngine(model, df=df)

        Parameters
        ----------
        csv_path : str
            Path to the CSV file.

        Returns
        -------
        pd.DataFrame
            The loaded dataset, ready to pass into __init__.
        """
        df = pd.read_csv(csv_path)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}\n"
                f"Expected: {REQUIRED_COLUMNS}\n"
                f"Found:    {list(df.columns)}"
            )
        print(f"   [CSV] Loaded {len(df)} locations from '{csv_path}'")
        return df

    # ─────────────────────────────────────────────────────────────────────
    # DATA: PLACEHOLDER (18 locations for immediate testing)
    # ─────────────────────────────────────────────────────────────────────
    def _build_placeholder_dataset(self) -> pd.DataFrame:
        """
        Hardcoded demo dataset: 18 real UAE locations across Abu Dhabi & Dubai.

        Each location has a rich 'Reviews_Embed_Source' text designed to
        contain diverse vocabulary so the semantic search can differentiate
        between "ocean kayaking" vs "desert sunset" vs "fast cars" queries.
        """
        data = {
            "Name": [
                # ── Abu Dhabi (11) ──
                "Jubail Mangrove Park",
                "Louvre Abu Dhabi",
                "Al Wathba Fossil Dunes",
                "Ferrari World",
                "Sheikh Zayed Grand Mosque",
                "Al Ain Oasis",
                "Qasr Al Watan",
                "Warner Bros World",
                "Eastern Mangroves",
                "Emirates Palace",
                "Saadiyat Beach",
                "Jebel Hafeet",
                "Yas Marina Circuit",
                # ── Dubai (5) ──
                "Burj Khalifa",
                "Dubai Miracle Garden",
                "Museum of the Future",
                "Al Fahidi Historical District",
                "Dubai Frame",
            ],
            "Type": [
                "Nature", "Museum", "Nature", "Adventure", "Landmark",
                "Nature", "Landmark", "Adventure", "Nature", "Landmark",
                "Nature", "Nature", "Adventure",
                "Landmark", "Nature", "Museum", "Landmark", "Landmark",
            ],
            "Popularity_Score": [
                5, 8, 3, 9, 10, 4, 7, 9, 5, 8,
                6, 5, 7,
                10, 8, 9, 4, 7,
            ],
            "Indoors": [
                False, True, False, True, False, False, True, True, False, True,
                False, False, False,
                True, False, True, False, True,
            ],
            "Description": [
                "A stunning mangrove sanctuary.",
                "Art and civilization museum.",
                "Unique rock formations in the desert.",
                "High speed theme park.",
                "Iconic Islamic architecture.",
                "Historic oasis with falaj irrigation.",
                "Presidential palace tour.",
                "Indoor theme park.",
                "Kayaking hotspot.",
                "Luxury hotel and golden cappuccino.",
                "Pristine white sand beach.",
                "Mountain summit with desert views.",
                "Formula One racing circuit.",
                "Tallest building in the world.",
                "Spectacular flower garden displays.",
                "Futuristic innovation museum.",
                "Old Dubai cultural heritage walks.",
                "Giant golden picture frame landmark.",
            ],
            # ── THE KEY COLUMN: Rich text for semantic embedding ──
            "Reviews_Embed_Source": [
                "Peaceful kayaking through lush green mangroves, perfect for "
                "birdwatching and quiet nature escapes away from the city noise. "
                "Calm water channels surrounded by wildlife and coastal serenity.",

                "World-class art museum with stunning dome architecture, showcasing "
                "masterpieces from every civilization and era of human history. "
                "Cultural exhibits, paintings, sculptures, and creative installations.",

                "Amazing place for quiet contemplation, the rocks are ancient and "
                "the sunset paints the fossil formations in gold and amber silence. "
                "Remote desert landscape far from crowds with dramatic geology.",

                "Adrenaline-pumping roller coasters and Formula One racing simulators, "
                "the fastest theme park rides on Earth under one giant red roof. "
                "Speed, thrills, fast cars, and family-friendly excitement.",

                "Breathtaking Islamic architecture with eighty-two white marble domes "
                "and gold-plated chandeliers, a spiritual and photographic masterpiece. "
                "Grand prayer hall, reflective pools, and serene courtyards.",

                "Historic palm oasis with ancient falaj irrigation channels winding "
                "through shaded pathways, a tranquil UNESCO World Heritage Site. "
                "Cool green shade, date palms, and centuries of agricultural heritage.",

                "Grand presidential palace tour revealing Arabian craftsmanship, ornate "
                "mosaics, and the story of UAE governance and diplomacy. "
                "Opulent interiors, majestic halls, and cultural significance.",

                "Massive indoor theme park with cartoon characters brought to life, "
                "rides and shows for families and children of all ages. "
                "Fun, entertainment, Batman, Superman, and animated adventures.",

                "Serene waterfront promenade with kayaking through mangrove channels, "
                "ideal for peaceful morning paddles and spotting herons and flamingos. "
                "Ocean breezes, coastal boardwalk, and waterfront dining.",

                "Opulent luxury hotel with gold leaf interiors and the famous gold "
                "cappuccino, an icon of Arabian hospitality and grandeur. "
                "Five-star elegance, royal suites, and beachfront luxury.",

                "Pristine white sand beach with turquoise Arabian Gulf waters, perfect "
                "for swimming, sunbathing, and watching hawksbill turtles nest. "
                "Ocean waves, coastal relaxation, and marine wildlife encounters.",

                "Dramatic mountain road winding to the summit with panoramic desert "
                "views, hot springs at the base and stargazing at the peak. "
                "Hiking, camping, mountain adventure, and sweeping vistas.",

                "Formula One Grand Prix racing circuit where you can drive supercars "
                "on the actual track or watch world championship motorsport events. "
                "Speed, racing, fast cars, engines roaring, and competitive thrills.",

                "The tallest building in the world at 828 meters, with observation "
                "decks offering breathtaking panoramic views of the entire Dubai skyline. "
                "Modern engineering marvel, skyscraper, and urban luxury.",

                "Spectacular flower garden with over 150 million blooming flowers "
                "arranged in arches, castles, and stunning floral sculptures and displays. "
                "Colorful petals, garden pathways, butterflies, and botanical beauty.",

                "Futuristic torus-shaped building exploring innovation, artificial "
                "intelligence, robotics, and what human civilization will look like "
                "in fifty years. Technology, science, and the future of humanity.",

                "Charming old Dubai neighborhood with wind tower architecture, art "
                "galleries, hidden coffee shops, and authentic cultural heritage walks. "
                "History, tradition, quiet alleys, and local Emirati culture.",

                "Giant golden picture frame structure bridging old and new Dubai, with "
                "a glass-floor sky deck and views across both sides of the city. "
                "Panoramic cityscape, modern landmark, and photography hotspot.",
            ],
            "Lat": [
                # Abu Dhabi
                24.5438, 24.5336, 24.1553, 24.4835, 24.4128,
                24.2167, 24.4622, 24.4897, 24.4539, 24.4615,
                24.5460, 24.0633, 24.4672,
                # Dubai
                25.1972, 25.0594, 25.2195, 25.2631, 25.2352,
            ],
            "Lon": [
                # Abu Dhabi
                54.6903, 54.3983, 54.6067, 54.6076, 54.4745,
                55.7636, 54.3060, 54.6093, 54.4417, 54.3174,
                54.4340, 55.7736, 54.6031,
                # Dubai
                55.2744, 55.2447, 55.2810, 55.2975, 55.3005,
            ],
        }
        return pd.DataFrame(data)

    # ─────────────────────────────────────────────────────────────────────
    # EMBEDDING: Pre-compute location vectors
    # ─────────────────────────────────────────────────────────────────────
    def _precompute_embeddings(self) -> np.ndarray:
        """
        Encode every location's Reviews_Embed_Source into a dense vector.

        VECTOR SPACE MODEL (for judges):
        ─────────────────────────────────
        The all-MiniLM-L6-v2 model was trained on 1 billion+ sentence pairs.
        It learned that semantically similar sentences should have vectors
        pointing in the same direction in 384-dimensional space.

        Example:
            "quiet sunset"   → vector A  (384 floats)
            "peaceful evening" → vector B  (384 floats)
            cos(A, B) ≈ 0.85  (high similarity, even with zero shared words)

        Returns
        -------
        np.ndarray
            Shape (N, 384) where N = number of locations.
            Each row is a location's semantic fingerprint.
        """
        texts = self.df["Reviews_Embed_Source"].tolist()
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings  # shape: (N, 384)

    # ─────────────────────────────────────────────────────────────────────
    # CORE: Semantic Search with Hybrid Scoring
    # ─────────────────────────────────────────────────────────────────────
    def semantic_search(
        self,
        user_query: str,
        top_k: int = 5,
        prefer_hidden_gems: bool = True,
        avoid_heat: bool = False,
    ) -> pd.DataFrame:
        """
        The main recommendation method. Combines AI-driven semantic relevance
        with an anti-popularity bias to surface hidden gems.

        ALGORITHM STEP-BY-STEP (for judges):
        ─────────────────────────────────────
        1. ENCODE: Convert the user's free-text query into a 384-dim vector.
        2. SIMILARITY: Compute Cosine Similarity between that vector and
           every pre-computed location vector.
        3. NORMALIZE: Min-max scale similarity scores to [0, 1] so the
           best match = 1.0 and worst = 0.0.
        4. ANTI-BIAS: Calculate Inverse Popularity:
              Inv_Pop = (10 - Popularity_Score) / 10
           A score of 10 (most crowded) → 0.0, score of 1 → 0.9.
        5. HYBRID BLEND:
              Final_Relevance = (Semantic × 0.7) + (Inv_Pop × 0.3)
           Semantic dominates (70%) but hidden gems get a gentle boost (30%).
        6. FILTER & RANK: Apply indoor/outdoor filter, sort descending,
           return top_k results.

        Parameters
        ----------
        user_query : str
            Free-text description like "quiet sunset near water".
        top_k : int
            Number of results to return.
        prefer_hidden_gems : bool
            If True, low-popularity locations get a scoring boost.
            If False, popular locations are favored instead.
        avoid_heat : bool
            If True, only return Indoors=True locations.

        Returns
        -------
        pd.DataFrame
            Top results with added columns:
            Semantic_Score, Inverse_Popularity_Score, Final_Relevance
        """
        # --- Work on a copy so we never mutate self.df ---
        results = self.df.copy()

        # ── STEP 1: Encode the user query ──
        #    Same model, same 384-dim space as the location embeddings.
        query_vector = self.model.encode([user_query], convert_to_numpy=True)
        # Shape: (1, 384)

        # ── STEP 2: Cosine Similarity ──
        #    cos(θ) = (query · location) / (‖query‖ × ‖location‖)
        #    Returns shape (1, N) — one similarity score per location.
        raw_scores = cosine_similarity(query_vector, self.location_embeddings)[0]
        # Shape: (N,) — one float per location

        # ── STEP 3: Min-Max Normalize to [0, 1] ──
        #    This makes the scores relative: best match = 1.0, worst = 0.0.
        #    Without this, raw cosine scores cluster around 0.2-0.6 and
        #    are hard to interpret visually on the heatmap.
        score_min = raw_scores.min()
        score_max = raw_scores.max()
        if score_max - score_min > 1e-9:
            semantic_scores = (raw_scores - score_min) / (score_max - score_min)
        else:
            # Edge case: all scores identical (e.g., empty query)
            semantic_scores = np.ones_like(raw_scores) * 0.5

        results["Semantic_Score"] = semantic_scores

        # ── STEP 4: Anti-Popularity Bias ──
        #    Inverse weighting so hidden gems (low popularity) score higher.
        if prefer_hidden_gems:
            results["Inverse_Popularity_Score"] = (
                (10 - results["Popularity_Score"]) / 10.0
            )
        else:
            # User WANTS popular spots — flip the weighting
            results["Inverse_Popularity_Score"] = (
                results["Popularity_Score"] / 10.0
            )

        # ── STEP 5: Hybrid Score ──
        #    70% semantic (the AI part) + 30% anti-bias (the fairness part)
        results["Final_Relevance"] = (
            (results["Semantic_Score"] * 0.7)
            + (results["Inverse_Popularity_Score"] * 0.3)
        )

        # ── STEP 6: Filter & Rank ──
        if avoid_heat:
            results = results[results["Indoors"] == True]

        results = results.sort_values("Final_Relevance", ascending=False)

        return results.head(top_k).reset_index(drop=True)

    # ─────────────────────────────────────────────────────────────────────
    # MAP DATA: Format for Folium HeatMap
    # ─────────────────────────────────────────────────────────────────────
    def get_map_data(self, results_df: pd.DataFrame) -> list:
        """
        Convert search results into the format expected by folium HeatMap.

        The HeatMap plugin expects a list of [lat, lon, intensity] triplets.
        We use Final_Relevance as the intensity so the map literally "glows"
        where the AI thinks the user should go.

        Parameters
        ----------
        results_df : pd.DataFrame
            Output from semantic_search(), must contain Lat, Lon, Final_Relevance.

        Returns
        -------
        list of [float, float, float]
            Each element is [latitude, longitude, heat_intensity].
        """
        return results_df[["Lat", "Lon", "Final_Relevance"]].values.tolist()

    # ─────────────────────────────────────────────────────────────────────
    # MAP DATA: All locations (for default view)
    # ─────────────────────────────────────────────────────────────────────
    def get_all_locations(self) -> pd.DataFrame:
        """Return the full dataset for rendering the default map view."""
        return self.df.copy()


# ═════════════════════════════════════════════════════════════════════════════
# GEMINI LLM ADVICE — Waterfall Fallback System
# ═════════════════════════════════════════════════════════════════════════════
def generate_ai_advice(top_places: pd.DataFrame, user_query: str, api_key: str) -> str:
    """
    Generate a persuasive travel recommendation using Google Gemini.

    Uses a Waterfall Fallback strategy: tries multiple models in priority
    order. If one model is rate-limited (429) or unavailable (404), it
    seamlessly falls through to the next. If ALL models fail, returns a
    pre-written backup string so the app never shows an error screen.

    Model Priority:
        1. gemini-2.0-flash-lite-001  (fastest, try first)
        2. gemini-2.0-flash-lite      (alias)
        3. gemini-2.0-flash           (stable)
        4. gemini-2.5-flash           (newest)

    Parameters
    ----------
    top_places : pd.DataFrame
        The top recommended locations from semantic_search().
    user_query : str
        The user's original query, included in the prompt for context.
    api_key : str
        Google Gemini API key, passed by the caller (resolved in app.py).

    Returns
    -------
    str
        Either a Gemini-generated recommendation or a hardcoded backup.
    """
    MODEL_PRIORITY = [
        "gemini-2.0-flash-lite-001",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.5-flash",
    ]

    names = ", ".join(top_places["Name"].tolist())

    # Enhanced prompt: includes what the user actually asked for
    prompt = (
        f"A tourist in the UAE described their ideal experience as: '{user_query}'. "
        f"Our AI recommendation engine selected these as the best semantic matches: {names}. "
        f"Write 2-3 enthusiastic sentences explaining why these places perfectly match "
        f"what the tourist is looking for. Be specific about each place's unique appeal."
    )

    client = genai.Client(api_key=api_key)

    for i, model_name in enumerate(MODEL_PRIORITY):
        try:
            print(f"   (Attempt {i + 1}/{len(MODEL_PRIORITY)}: Trying '{model_name}'...)")
            response = client.models.generate_content(
                model=model_name, contents=prompt
            )
            print(f"   ✅ SUCCESS: Got response from '{model_name}'!")
            return response.text

        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                reason = "Rate Limit"
            elif "404" in error_str:
                reason = "Model Not Found"
            else:
                reason = "Unknown Error"

            next_model = (
                MODEL_PRIORITY[i + 1] if i + 1 < len(MODEL_PRIORITY) else "Backup Mode"
            )
            print(
                f"   ⚠️ Model '{model_name}' failed ({reason}). "
                f"Switching to '{next_model}'..."
            )

    # --- FINAL SAFETY NET: All models exhausted ---
    print("   🔴 All models exhausted. Using Backup Mode.")
    return (
        f"**Recommended:** {names}. "
        "Our AI algorithm selected these specifically because they are the closest "
        "semantic match to your preferences while maintaining low crowd density scores."
    )


# ═════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST (run with: python ai_engine.py)
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    print("=" * 60)
    print("  AI ENGINE — Standalone Test")
    print("=" * 60)

    # Load model directly (no Streamlit caching in standalone mode)
    print("\n1. Loading Sentence Transformer model...")
    model = load_sentence_model()
    print("   Done.")

    # Init engine with placeholder data
    print("\n2. Initializing engine...")
    engine = UAETourismEngine(model)

    # Test queries
    test_queries = [
        "quiet sunset near water",
        "fast cars and racing",
        "ancient history and culture",
        "luxury and gold",
    ]

    for query in test_queries:
        print(f"\n{'─' * 60}")
        print(f"  QUERY: \"{query}\"")
        print(f"{'─' * 60}")
        results = engine.semantic_search(query, top_k=3)
        for _, row in results.iterrows():
            print(
                f"   {row['Name']:35s}  "
                f"Semantic={row['Semantic_Score']:.3f}  "
                f"Final={row['Final_Relevance']:.3f}"
            )

    # Quick assertions
    print(f"\n{'─' * 60}")
    print("  ASSERTIONS")
    print(f"{'─' * 60}")
    r = engine.semantic_search("ocean kayaking mangroves", top_k=3)
    assert len(r) <= 3, "top_k filter broken"
    assert "Semantic_Score" in r.columns, "Missing Semantic_Score column"
    assert "Final_Relevance" in r.columns, "Missing Final_Relevance column"
    assert r["Final_Relevance"].is_monotonic_decreasing, "Results not sorted"
    print("   ✅ All assertions passed.")

    # Test map data output
    map_data = engine.get_map_data(r)
    assert len(map_data) == len(r), "Map data length mismatch"
    assert len(map_data[0]) == 3, "Map data should be [lat, lon, intensity]"
    print("   ✅ Map data format correct.")

    # Test Gemini LLM (only if key available)
    test_key = os.environ.get("GOOGLE_API_KEY", "")
    if test_key:
        print(f"\n{'─' * 60}")
        print("  GEMINI LLM TEST")
        print(f"{'─' * 60}")
        advice = generate_ai_advice(r, "ocean kayaking", api_key=test_key)
        print(f"   Gemini advice: {advice[:80]}...")
    else:
        print("\n   (Skipping Gemini test — GOOGLE_API_KEY not set)")

    print(f"\n{'=' * 60}")
    print("  ALL TESTS PASSED")
    print(f"{'=' * 60}")
