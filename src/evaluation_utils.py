from sklearn.metrics.pairwise import cosine_similarity
from evaluate import load
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Load Hugging Face evaluation metrics
rouge = load("rouge")
bleu = load("bleu")

# ✅ Retrieval Evaluation
def evaluate_retrieval(query_emb, doc_embs, relevant_idx):
    """
    Evaluate if the relevant document is in the top-3 retrieved results.
    Returns precision@3 (0 or 1).
    """
    sims = cosine_similarity([query_emb], doc_embs)[0]
    ranked = sims.argsort()[::-1]
    precision_at_k = int(relevant_idx in ranked[:3])
    return precision_at_k

# ✅ Generation Evaluation (ROUGE + BLEU + BLEU-1)
def evaluate_generation(pred, ref):
    """
    Evaluate text generation quality using ROUGE, BLEU, and BLEU-1.
    Returns dictionary of scores.
    """
    rouge_score = rouge.compute(predictions=[pred], references=[ref])
    bleu_score = bleu.compute(predictions=[pred], references=[[ref]])

    # BLEU-1: Unigram precision only
    pred_tokens = pred.split()
    ref_tokens = [ref.split()]
    bleu1 = sentence_bleu(
        ref_tokens,
        pred_tokens,
        weights=(1, 0, 0, 0),
        smoothing_function=SmoothingFunction().method1
    )

    return {
        "ROUGE": rouge_score,
        "BLEU": bleu_score,
        "BLEU-1": {"bleu1": bleu1}
    }

# ✅ Example test case
if __name__ == "__main__":
    print("\n=== 🔍 TESTING GENERATION EVALUATION ===")
    pred = "This patient likely has COPD due to chronic cough and low FEV1/FVC ratio."
    ref = "The findings are consistent with COPD based on pulmonary function tests and symptoms."

    scores = evaluate_generation(pred, ref)

    print("\n✅ Evaluation Results:")
    print("-" * 30)
    for metric, result in scores.items():
        print(f"{metric}:")
        for k, v in result.items():
            if isinstance(v, list):
                print(f"  {k}: {[round(val, 4) for val in v]}")
            else:
                print(f"  {k}: {v:.4f}")
    print("-" * 30)
