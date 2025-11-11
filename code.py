# -*- coding: utf-8 -*-
"""
Vibe Matcher Prototype
OpenAI Embeddings (text-embedding-ada-002)
Compatible with VS Code + Python 3.11.6 virtual environment
"""

# =========================================================
# 1. IMPORTS & SETUP
# =========================================================
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# Load API key
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file!")

client = OpenAI(api_key=API_KEY)
print("✅ OpenAI client initialized successfully.")

# =========================================================
# 2. DATA PREPARATION
# =========================================================
products = [
    {"name": "Black Oversized Hoodie", "desc": "dark mysterious streetwear cozy cold weather", "vibes": ["streetwear", "dark", "cozy"]},
    {"name": "Pink Crop Top", "desc": "cute soft pastel girl summer vibes aesthetic", "vibes": ["girly", "summer", "aesthetic"]},
    {"name": "Denim Jacket", "desc": "casual classic cool urban wear for parties and outings", "vibes": ["urban", "casual", "party"]},
    {"name": "Beige Linen Shirt", "desc": "light minimal earthy calm peaceful natural look", "vibes": ["minimal", "earthy", "calm"]},
    {"name": "Athletic Set", "desc": "sporty energetic fitness wear for workout or running", "vibes": ["sporty", "active", "energetic"]},
    {"name": "Elegant Satin Gown", "desc": "formal luxury evening wear with graceful appearance", "vibes": ["formal", "elegant", "luxurious"]},
    {"name": "Boho Dress", "desc": "flowy colorful bohemian style for travel and festivals", "vibes": ["boho", "festival", "colorful"]},
    {"name": "Graphic Tee", "desc": "casual youthful street style with bold prints", "vibes": ["streetwear", "youthful", "bold"]},
    {"name": "White Blazer", "desc": "clean minimal chic modern office look outfit", "vibes": ["chic", "modern", "formal"]},
]

df = pd.DataFrame(products)
print("\n🛍️ Product catalog:")
print(df[["name", "vibes"]])

# =========================================================
# 3. GENERATE EMBEDDINGS
# =========================================================
EMBED_MODEL = "text-embedding-ada-002"

def get_embedding(text):
    """Generate OpenAI embeddings for a single text"""
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

# Generate embeddings for product descriptions
print("\n⚙️ Generating embeddings for products...")
embeddings = [get_embedding(desc) for desc in df["desc"]]
df["embedding"] = embeddings
print("✅ All embeddings generated!")

# =========================================================
# 4. COSINE SIMILARITY SEARCH
# =========================================================
def search_vibe_matches(query, top_k=3):
    """Find top-k product matches for a vibe query"""
    query_emb = get_embedding(query)
    product_embs = np.vstack(df["embedding"].values)
    sims = cosine_similarity([query_emb], product_embs)[0]

    df["score"] = sims
    top_matches = df.sort_values(by="score", ascending=False).head(top_k)

    fallback = None
    if top_matches.iloc[0]["score"] < 0.3:
        fallback = "No strong match found. Try rephrasing your vibe."

    return top_matches[["name", "desc", "vibes", "score"]], fallback

# =========================================================
# 5. TEST & EVALUATION
# =========================================================
queries = [
    "energetic urban chic",
    "dark mysterious street look",
    "boho travel outfit"
]

times, good_count = [], 0
for q in queries:
    start = time.time()
    results, fallback = search_vibe_matches(q)
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"\n🎯 Query: '{q}'  (Latency: {elapsed:.2f}s)")
    print(results)
    if fallback:
        print("⚠️", fallback)
    good_count += (results["score"] > 0.7).sum()

print(f"\n📊 Average latency: {np.mean(times):.2f}s | Good matches (>0.7): {good_count}")

# =========================================================
# 6. VISUALIZATION
# =========================================================
plt.bar(range(len(times)), times)
plt.xticks(range(len(times)), [f"Q{i+1}" for i in range(len(times))])
plt.xlabel("Query #")
plt.ylabel("Latency (s)")
plt.title("Vibe Matcher Query Latency")
plt.tight_layout()
plt.show()

# =========================================================
# 7. REFLECTION
# =========================================================
print("\n💡 Reflection:")
print("- Uses OpenAI embeddings for accurate semantic similarity.")
print("- Fallback message handles poor matches.")
print("- Could integrate Pinecone or FAISS for scalable search.")
print("- Future improvement: add UI (Gradio/Streamlit).")
print("- Extend dataset for richer vibe coverage.")
