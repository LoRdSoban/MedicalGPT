import os, json
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

def load_clinical_notes(base_path):
    data = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    note = json.load(f)
                    combined = "\n".join([note.get(f"input{i}", "") for i in range(1, 7)])
                    data.append({"path": os.path.join(root, file), "text": combined})
    return pd.DataFrame(data)

def embed_notes(df, model_name="all-MiniLM-L6-v2"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)
    df["embedding"] = df["text"].apply(lambda x: model.encode(x))
    return df

if __name__ == "__main__":
    df = load_clinical_notes("data/Finished")
    df = embed_notes(df)
    df.to_pickle("data/clinical_embeddings.pkl")
    print("✅ Embeddings saved to data/clinical_embeddings.pkl")
