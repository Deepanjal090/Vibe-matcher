# -*- coding: utf-8 -*-
"""
Vibe Matcher Prototype (Hybrid Version)
✅ Uses OpenAI embeddings if available and quota allows.
✅ Falls back to local FastEmbed (BAAI/bge-small-en-v1.5) if API unavailable.
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

# Load environment variables
load_dotenv()

# Try importing both OpenAI & FastEmbed
from openai import OpenAI
from fastembed import TextEmbedding

# =========================================================
# 2. CONFIGURATION
# =========================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_OPENAI = bool(OPENAI_API_KEY)

if USE_OPENAI:
    client = OpenAI(api_key=OPENAI_API_KEY)
    EMBED_MODEL = "text-embedding-ada-002"
    print("✅ OpenAI client detected — using cloud embeddings.")
else:
    print("ℹ️ No OpenAI key found — using local FastEmbed.")

# FastEmbed model for fallback
LOCAL_MODEL = "BAAI/bge-small-en-v1.5"
embedder = TextEmbedding(model_name=LOCAL_MODEL)

# =========================================================
# 3. DATA PREPARATION
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
# 4. EMBEDDING FUNCTIONS
# =========================================================
def get_openai_embedding(text):
    """Use OpenAI embeddings if available and quota allows."""
    try:
        response = client.embeddings.create(model=EMBED_MODEL, input=text)
        return np.array(response.data[0].embedding, dtype=np.float32)
    except Exception as e:
        print(f"⚠️ OpenAI embedding failed ({e}). Falling back to FastEmbed.")
        return get_fastembed_embedding(text)

def get_fastembed_embedding(text):
    """Use FastEmbed for free local embeddings."""
    return np.array(list(embedder.embed([text]))[0], dtype=np.float32)

def get_embedding(text):
    """Decide which embedding method to use dynamically."""
    if USE_OPENAI:
        return get_openai_embedding(text)
    else:
        return get_fastembed_embedding(text)

# =========================================================
# 5. GENERATE PRODUCT EMBEDDINGS
# =========================================================
print("\n⚙️ Generating embeddings for products...")
embeddings = [get_embedding(desc) for desc in df["desc"]]
df["embedding"] = embeddings
print("✅ All embeddings generated successfully!")

# =========================================================
# 6. COSINE SIMILARITY SEARCH
# =========================================================
def search_vibe_matches(query, top_k=3):
    """Find top-k matching products for a vibe query."""
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
# 7. TEST & EVALUATION
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
# 8. VISUALIZATION
# =========================================================
plt.bar(range(len(times)), times)
plt.xticks(range(len(times)), [f"Q{i+1}" for i in range(len(times))])
plt.xlabel("Query #")
plt.ylabel("Latency (s)")
plt.title("Vibe Matcher Query Latency (Hybrid Mode)")
plt.tight_layout()
plt.show()

# =========================================================
# 9. REFLECTION
# =========================================================
print("\n💡 Reflection:")
print("- Hybrid embedding pipeline: OpenAI (cloud) + FastEmbed (local fallback).")
print("- Robust against quota or connectivity errors.")
print("- Uses cosine similarity for vector search.")
print("- Could integrate FAISS or Pinecone for scalability.")
print("- Future: add Gradio/Streamlit UI for interactive vibe matching.")
