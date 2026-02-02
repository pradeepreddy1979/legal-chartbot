import os
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch

# --- CONFIGURATION ---
INDEX_FILE_PATH = "data/indexed_legal_data.json"
EMBEDDINGS_FILE_PATH = "data/legal_text_embeddings.npy"
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2' # Excellent small, fast model for embeddings

def create_embeddings():
    """Loads indexed data, creates vector embeddings, and saves them."""
    
    if not os.path.exists(INDEX_FILE_PATH):
        print(f"FATAL ERROR: Index file not found at {INDEX_FILE_PATH}. Please run index_data.py.")
        return

    # Load the indexed data
    with open(INDEX_FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data:
        print("WARNING: Index data is empty. Nothing to embed.")
        return
        
    print(f"INFO: Loaded {len(data)} legal sections for vectorization.")

    # 1. Initialize tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"❌ ERROR: Could not load transformer model '{MODEL_NAME}'. Ensure you have an internet connection and the libraries installed.")
        print(f"Details: {e}")
        return

    # 2. Extract texts for embedding
    texts = [item['raw_text'] for item in data]
    
    # 3. Process in batches for efficiency (using a small batch size for demonstration)
    embeddings_list = []
    batch_size = 32 
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Creating Embeddings"):
        batch = texts[i:i + batch_size]
        
        # Tokenize and encode the batch
        encoded_input = tokenizer(
            batch, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        )

        # Generate embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
            # Use mean pooling to get sentence embeddings
            embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        
        embeddings_list.append(embeddings.cpu().numpy())

    # Concatenate all batch embeddings
    final_embeddings = np.concatenate(embeddings_list, axis=0)

    # 4. Save embeddings as a numpy file
    np.save(EMBEDDINGS_FILE_PATH, final_embeddings)
    print(f"\n✅ Successfully created and saved embeddings for {len(final_embeddings)} sections.")
    print(f"Embeddings saved to: {EMBEDDINGS_FILE_PATH}")
    
# Function to perform mean pooling (required for Sentence-Transformers models)
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


if __name__ == "__main__":
    create_embeddings()