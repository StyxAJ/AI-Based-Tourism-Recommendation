import os
import pandas as pd
from google import genai
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
API_KEY = os.environ.get("GOOGLE_API_KEY", "PASTE_YOUR_KEY_HERE")


def load_data():
    """
    Simulates the database from Member A.
    """
    data = {
        "Name": [
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
        ],
        "Type": [
            "Nature",
            "Museum",
            "Nature",
            "Adventure",
            "Landmark",
            "Nature",
            "Landmark",
            "Adventure",
            "Nature",
            "Landmark",
        ],
        "Popularity_Score": [5, 8, 3, 9, 10, 4, 7, 9, 5, 8],  # 10 = Crowded
        "Indoors": [False, True, False, True, False, False, True, True, False, True],
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
        ],
    }
    return pd.DataFrame(data)


def rank_places(user_preferences, df):
    """
    YOUR ALGORITHM: Inverse Popularity Weighting
    """
    ranked_df = df.copy()

    # 1. Filter by Indoors/Outdoors
    if user_preferences.get("avoid_heat"):
        ranked_df = ranked_df[ranked_df["Indoors"] == True]

    # 2. Anti-Bias Calculation
    if user_preferences.get("find_hidden_gems"):
        ranked_df["Discovery_Score"] = 10 - ranked_df["Popularity_Score"]
    else:
        ranked_df["Discovery_Score"] = ranked_df["Popularity_Score"]

    # 3. Interest Matching
    ranked_df["Interest_Match"] = ranked_df["Type"].apply(
        lambda x: 10 if x == user_preferences.get("interest") else 0
    )

    # 4. Final Score
    ranked_df["Final_Score"] = (ranked_df["Interest_Match"] * 0.6) + (
        ranked_df["Discovery_Score"] * 0.4
    )

    return ranked_df.sort_values(by="Final_Score", ascending=False).head(3)


def generate_ai_advice(top_places):
    """
    Waterfall Fallback System: tries multiple Gemini models in priority order.
    If all models fail (429 Rate Limit, 404 Not Found, etc.), returns a
    hardcoded backup recommendation so the app never crashes.
    """
    MODEL_PRIORITY = [
        "gemini-2.0-flash-lite-001",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.5-flash",
    ]

    names = ", ".join(top_places["Name"].tolist())
    prompt = (
        f"Write a one-sentence enthusiastic reason to visit these UAE spots: {names}. "
        "Focus on why they are great for avoiding crowds."
    )

    client = genai.Client(api_key=API_KEY)

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

            next_model = MODEL_PRIORITY[i + 1] if i + 1 < len(MODEL_PRIORITY) else "Backup Mode"
            print(f"   ⚠️ Model '{model_name}' failed ({reason}). Switching to '{next_model}'...")

    # --- FINAL SAFETY NET: All models exhausted ---
    print("   🔴 All models exhausted. Using Backup Mode.")
    return (
        f"**Recommended:** {names}. "
        "Our algorithm selected these specifically because they match your interests "
        "while maintaining a low crowd density score today."
    )


# --- TEST IT ---
if __name__ == "__main__":
    print("--- STARTING TOURISM ENGINE ---")
    user = {"interest": "Nature", "find_hidden_gems": True, "avoid_heat": False}

    df = load_data()
    results = rank_places(user, df)
    print(f"📍 Selected: {results['Name'].tolist()}")

    print("🤖 Asking Google Gemini (Model: gemini-2.0-flash)...")
    advice = generate_ai_advice(results)

    print(f"\n📝 FINAL OUTPUT:\n{advice}")
