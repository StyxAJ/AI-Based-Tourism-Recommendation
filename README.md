# UAE AI Tourism Recommender (Capstone 2026)

> A Vector-Based Semantic Search Engine that recommends UAE tourist destinations based on **meaning**, not keywords — with Probabilistic Heatmap Visualizations and LLM-powered travel advice.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.53-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8-F7931E?logo=scikit-learn&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-2.0%20Flash-4285F4?logo=google&logoColor=white)
![Folium](https://img.shields.io/badge/Folium-Maps-77B829?logo=leaflet&logoColor=white)
![Sentence Transformers](https://img.shields.io/badge/Sentence%20Transformers-MiniLM-FFD43B)

---

## The Problem

Traditional tourism platforms suffer from **popularity bias** — the same top-10 landmarks dominate every recommendation. Tourists miss lesser-known destinations that may better match their actual preferences. This creates **overtourism** at iconic sites while hidden gems remain empty.

## The Solution

This system uses **AI-driven semantic matching** instead of rule-based keyword filtering:

- A user typing *"quiet sunset near water"* will match locations described as *"peaceful kayaking through mangroves"* — even though those phrases share **zero words** in common.
- A **hybrid scoring formula** blends semantic relevance (70%) with inverse popularity (30%), gently nudging results toward hidden gems without sacrificing relevance.
- An interactive **Folium Heatmap** visually represents the AI's confidence — the map literally *glows* where the engine thinks you should go.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Semantic Search** | Free-text queries matched against location embeddings using Cosine Similarity (all-MiniLM-L6-v2) |
| **Anti-Bias Algorithm** | Inverse Popularity Scoring ensures lesser-known spots get a fair chance |
| **LLM Travel Advice** | Google Gemini generates personalized recommendations with a 4-model waterfall fallback |
| **Probabilistic Heatmap** | Folium map where glow intensity = AI semantic relevance scores |
| **CSV-Ready Architecture** | Auto-detects `uae_pois.csv` dataset or falls back to built-in 18-location placeholder |

---

## Architecture

```
User Query ──► Sentence Transformer ──► 384-dim vector ──┐
                                                          ├──► Cosine Similarity ──► Hybrid Score ──► Ranked Results
Location DB ──► Sentence Transformer ──► (N, 384) matrix ┘

                              ┌─────────────────────────────────────────┐
Hybrid Score Formula:         │                                         │
                              │  Final = (Semantic × 0.7)               │
                              │        + (Inverse_Popularity × 0.3)     │
                              │                                         │
                              └─────────────────────────────────────────┘

Cosine Similarity:   cos(θ) = (A · B) / (‖A‖ × ‖B‖)
```

- **Semantic_Score (0–1):** How closely the user's query matches a location's description in vector space
- **Inverse_Popularity (0–1):** `(10 - Popularity_Score) / 10` — hidden gems score higher
- **70/30 split:** AI relevance dominates, but crowds are penalized

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`, 384-dim) |
| Similarity | Scikit-learn `cosine_similarity` |
| LLM | Google Gemini (waterfall: flash-lite → flash → 2.5-flash) |
| UI | Streamlit |
| Maps | Folium + streamlit-folium (CartoDB dark_matter tiles) |
| Data | Pandas DataFrame / CSV |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/StyxAJ/AI-Based-Tourism-Recommendation.git
cd AI-Based-Tourism-Recommendation
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

- **Windows:** `venv\Scripts\activate`
- **macOS/Linux:** `source venv/bin/activate`

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will download the `all-MiniLM-L6-v2` model (~80 MB). This is cached automatically after the first download.

### 4. Set up your API key

```bash
cp .env.example .env
```

Edit `.env` and paste your [Google Gemini API key](https://aistudio.google.com/apikey):

```
GEMINI_API_KEY=your_actual_key_here
```

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Project Files

| File | Purpose |
|------|---------|
| `ai_engine.py` | Semantic recommendation engine — embeddings, cosine similarity, hybrid scoring, Gemini LLM waterfall |
| `app.py` | Streamlit UI — sidebar controls, Folium heatmap, results table, AI advice panel |
| `logic.py` | Original rule-based MVP (kept as backup / reference) |
| `requirements.txt` | Pinned Python dependencies |
| `.env.example` | Template for environment variables (API key) |
| `project_context.md` | Team architecture notes and constraints |

---

## Dataset

The system includes a **built-in placeholder dataset** of 18 UAE locations (Abu Dhabi + Dubai) for immediate testing.

When `uae_pois.csv` is placed in the project root, the engine **automatically detects and loads it** — no code changes needed.

**Required CSV columns:**

| Column | Type | Description |
|--------|------|-------------|
| `Name` | str | Location name |
| `Type` | str | Category (Nature, Museum, Adventure, Landmark) |
| `Popularity_Score` | int | 1–10 scale (10 = most crowded) |
| `Indoors` | bool | True if indoor location |
| `Description` | str | Short description |
| `Reviews_Embed_Source` | str | Rich text for semantic embedding |
| `Lat` | float | Latitude |
| `Lon` | float | Longitude |

---

## How It Works (For Judges)

1. **Vectorization:** Every location description and user query is encoded into a 384-dimensional vector using a pre-trained Sentence Transformer.
2. **Cosine Similarity:** We measure the angle between vectors — `cos(θ) = (A · B) / (‖A‖ × ‖B‖)`. A score of 1.0 means identical meaning.
3. **Hybrid Scoring:** Pure semantic score is blended 70/30 with inverse popularity to combat tourist-trap bias.
4. **Heatmap Visualization:** The Folium HeatMap intensity is driven directly by the AI relevance scores — the map IS the AI output.
5. **LLM Fallback:** The Gemini integration cascades across 4 models (flash-lite → flash → 2.5-flash) so the app never crashes, even on free-tier rate limits.

---

## Team

**Capstone Project** — 7th Semester, 2026
