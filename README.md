# 🎧 Vibe Matcher – Fashion Recommendation Prototype

An AI-powered fashion recommendation prototype that matches outfits to user “vibes” using text embeddings and cosine similarity. Built with OpenAI and FastEmbed models, it suggests top-3 fashion items for any mood or style query.

---

## 🚀 Features

- 🧩 **Hybrid Embedding System:**  
  Uses OpenAI’s `text-embedding-ada-002` when available and automatically falls back to FastEmbed for free, local embeddings.

- 👗 **Fashion Dataset:**  
  Contains 9 mock fashion items with vibe tags such as *boho*, *minimal*, *streetwear*, and *formal*.

- 🔍 **Smart Vibe Matching:**  
  Embeds user vibe queries and product descriptions, then ranks top-3 results using cosine similarity.

- 📊 **Performance Evaluation:**  
  Measures average latency, counts high-similarity (“good”) matches, and visualizes query times with Matplotlib.

- 💡 **Offline Mode:**  
  Runs entirely locally when OpenAI API credits are unavailable.

---

## 🧠 How It Works

1. User enters a vibe query like _“energetic urban chic”_.  
2. The system generates vector embeddings for the query and product descriptions.  
3. Cosine similarity measures how closely each product matches the vibe.  
4. The top-3 matching outfits are displayed with similarity scores.  
5. Average response time and performance metrics are visualized.

---

## 🧩 Tech Stack

- **Language:** Python 3.11.6  
- **Libraries:** `openai`, `fastembed`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `python-dotenv`  
- **Environment:** Works in VS Code, Jupyter Notebook, and Google Colab  

---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/vibe-matcher.git
   cd vibe-matcher
