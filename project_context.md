\# Project: AI Tourism Recommender (UAE)

\# Deadline: Feb 4th, 2026

\# Role: Capstone Project (MVP)



\## Tech Stack

\- Language: Python 3.10+

\- UI: Streamlit

\- AI: Google Gemini Pro API (`google-generativeai`)

\- Data: Pandas CSV (`uae\_pois.csv`)



\## Core Logic

1\. Load POIs from CSV.

2\. Filter by User Preferences (Indoors/Outdoors, Budget).

3\. "Anti-Bias" Algorithm: Rank = (Relevance \* 0.6) + ((10 - Popularity) \* 0.4).

4\. Send Top 3 to Gemini API to generate a "Why you should go" description.



\## Team Rules

\- No complex databases (SQL). Use CSV.

\- No training loops. Logic is rule-based.

\- Keep UI simple: One Sidebar, One Main Map, One Results List.

