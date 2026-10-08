import os
import time
import numpy as np
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(".env")
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

DATA_DIR = "data"
train_trans_path = os.path.join(DATA_DIR, "train_transcripts.csv")
test_trans_path = os.path.join(DATA_DIR, "test_transcripts.csv")

train_emb_path = os.path.join(DATA_DIR, "train_gemini_embeddings.npy")
test_emb_path = os.path.join(DATA_DIR, "test_gemini_embeddings.npy")

def get_embeddings_batched(texts, batch_size=80, model_name="models/gemini-embedding-001"):
    embeddings = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        clean_batch = [t.strip() if t.strip() else "[silent audio]" for t in batch]
        
        success = False
        for attempt in range(8):
            try:
                res = genai.embed_content(model=model_name, content=clean_batch)
                emb = res["embedding"]
                embeddings.extend(emb)
                print(f"Embedded [{i} to {min(i+batch_size, total)}] / {total} (Total so far: {len(embeddings)})")
                success = True
                break
            except Exception as e:
                err_str = str(e)
                wait_sec = 15.0 if "429" in err_str else 3.0
                print(f"Notice on batch {i}: {e}. Waiting {wait_sec}s before retry (attempt {attempt+1}/8)...")
                time.sleep(wait_sec)
        if not success:
            raise RuntimeError(f"Failed to embed batch starting at index {i}")
        time.sleep(2.0)
    return np.array(embeddings, dtype=np.float32)

def main():
    print("=== Extracting Gemini Dense Semantic Embeddings (3072-D) ===")
    train_df = pd.read_csv(train_trans_path)
    test_df = pd.read_csv(test_trans_path)
    
    train_texts = train_df["transcript"].fillna("").astype(str).tolist()
    test_texts = test_df["transcript"].fillna("").astype(str).tolist()
    
    if not os.path.exists(train_emb_path) or len(np.load(train_emb_path)) != len(train_texts):
        print(f"\nProcessing Train Transcripts ({len(train_texts)} items)...")
        train_embs = get_embeddings_batched(train_texts, batch_size=80)
        np.save(train_emb_path, train_embs)
        print(f"--> Saved {train_emb_path} shape: {train_embs.shape}")
    else:
        print(f"Loaded existing {train_emb_path}")
        train_embs = np.load(train_emb_path)
        print(f"Shape: {train_embs.shape}")
        
    if not os.path.exists(test_emb_path) or len(np.load(test_emb_path)) != len(test_texts):
        print(f"\nProcessing Test Transcripts ({len(test_texts)} items)...")
        test_embs = get_embeddings_batched(test_texts, batch_size=80)
        np.save(test_emb_path, test_embs)
        print(f"--> Saved {test_emb_path} shape: {test_embs.shape}")
    else:
        print(f"Loaded existing {test_emb_path}")
        test_embs = np.load(test_emb_path)
        print(f"Shape: {test_embs.shape}")

    print("\n[COMPLETE] Successfully extracted and saved all 3072-D dense embeddings!")

if __name__ == "__main__":
    main()
