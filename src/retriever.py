import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

class Retriever:
    def __init__(self, embedding_file="data/clinical_embeddings.pkl", model_name="all-MiniLM-L6-v2"):
        self.df = pd.read_pickle(embedding_file)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=device)
        self.index = self._build_index()

    def _build_index(self):
        dim = len(self.df["embedding"].iloc[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.stack(self.df["embedding"].values))
        return index

    def search(self, query, top_k=3):
        query_embedding = self.model.encode([query])
        D, I = self.index.search(query_embedding, top_k)
        return self.df.iloc[I[0]]

if __name__ == "__main__":
    r = Retriever()
    results = r.search("patient with chronic cough and low oxygen")
    print(results[["path", "text"]])
