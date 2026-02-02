import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# ==============================================================
# 1️⃣ CONFIGURATION
# ==============================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # Small, fast, accurate model
KNOWLEDGE_DIR = "extracted_data"            # Folder containing extracted legal JSON files
VECTOR_DB_FILE = "legal_vector_db.json"     # Output vector database file


# ==============================================================
# 2️⃣ DATA LOADING
# ==============================================================

def load_legal_data() -> List[Dict]:
    """Load all extracted legal sections from JSON files inside the knowledge directory."""
    legal_texts = []

    if not os.path.exists(KNOWLEDGE_DIR):
        print(f"❌ FATAL: Knowledge directory '{KNOWLEDGE_DIR}' not found.")
        return []

    json_files = [f for f in os.listdir(KNOWLEDGE_DIR) if f.endswith(".json")]
    if not json_files:
        print("⚠️ No JSON files found in extracted_data/.")
        return []

    print(f"[INFO] Found {len(json_files)} files. Loading legal data...")
    for filename in json_files:
        filepath = os.path.join(KNOWLEDGE_DIR, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    legal_texts.extend(data)
        except Exception as e:
            print(f"⚠️ Skipped '{filename}' due to error: {e}")
            continue

    print(f"[INFO] Loaded {len(legal_texts)} total sections from all files.\n")
    return legal_texts


# ==============================================================
# 3️⃣ EMBEDDING CREATION
# ==============================================================

def create_embeddings():
    """Generate embeddings using a Sentence Transformer model and save them as JSON."""
    
    legal_data = load_legal_data()
    if not legal_data:
        print("❌ No legal data to embed. Exiting.")
        return

    # Initialize model
    print(f"[INFO] Loading model '{EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Fix random seed for consistency
    np.random.seed(42)

    # Extract valid texts
    texts = [item.get("text", "").strip() for item in legal_data if item.get("text")]
    print(f"[INFO] Generating embeddings for {len(texts)} legal sections...")

    # Create embeddings (handles batching internally)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32, convert_to_numpy=True)

    # Build vector DB
    db = []
    text_index = 0
    for section in tqdm(legal_data, desc="Building vector DB"):
        text = section.get("text", "").strip()
        if text:
            section_number = (
                section.get("section_number")
                or section.get("section")
                or f"N/A-{text_index}"
            )
            db.append({
                "section_number": section_number,
                "text": text,
                "embedding": embeddings[text_index].tolist(),
                "source_file": section.get("source", "unknown")
            })
            text_index += 1

    # Save vector database
    with open(VECTOR_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Successfully created {len(db)} embeddings.")
    print(f"💾 Saved vector database to: {VECTOR_DB_FILE}")


# ==============================================================
# 4️⃣ MAIN ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    if os.path.exists(VECTOR_DB_FILE):
        print(f"⚠️ Existing file '{VECTOR_DB_FILE}' found. Replacing it...\n")
        os.remove(VECTOR_DB_FILE)

    create_embeddings()
