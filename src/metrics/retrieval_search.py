import numpy as np
from sentence_transformers import util
from . import sentence_model

def recall_at_k(relevant_docs: list, retrieved_docs: list, k: int) -> float:
    relevant_at_k = set(retrieved_docs[:k]).intersection(set(relevant_docs))
    return len(relevant_at_k) / len(relevant_docs) if relevant_docs else 0.0

def precision_at_k(relevant_docs: list, retrieved_docs: list, k: int) -> float:
    relevant_at_k = set(retrieved_docs[:k]).intersection(set(relevant_docs))
    return len(relevant_at_k) / k

def dcg(relevance_scores: list) -> int:
    return sum([
        (2**rel - 1) / np.log2(idx + 2)
        for idx, rel in enumerate(relevance_scores)
    ])

def ndcg_at_k(relevant_docs: list, retrieved_docs: list, k: int) -> float:
    rel_scores = [1 if doc in relevant_docs else 0 for doc in retrieved_docs[:k]]
    ideal_rel_scores = sorted(rel_scores, reverse=True)
    dcg_val = dcg(rel_scores)
    idcg_val = dcg(ideal_rel_scores)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0

def context_relevance(query: str, retrieved_contexts: list) -> list:
    query_emb = sentence_model.encode(query, convert_to_tensor=True)
    context_embs = sentence_model.encode(retrieved_contexts, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_emb, context_embs)
    return scores.squeeze().tolist()

