from transformers import pipeline
import torch

class Generator:
    def __init__(self, model_name="tiiuae/falcon-rw-1b"):
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("text-generation", model=model_name, device=device, max_new_tokens=300)

    def generate(self, query, contexts):
        context_str = "\n".join(contexts)
        prompt = f"""You are a clinical assistant. Use the context below to answer the query:
--- CONTEXT ---
{context_str}
--- QUERY ---
{query}
--- ANSWER ---"""
        return self.pipe(prompt)[0]["generated_text"].split("--- ANSWER ---")[-1].strip()

if __name__ == "__main__":
    g = Generator()
    ctx = ["Patient has FEV1/FVC = 0.43, consistent with COPD", "Shortness of breath on exertion"]
    print(g.generate("Does the patient have COPD?", ctx))
