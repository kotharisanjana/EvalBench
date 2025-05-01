import numpy as np
from sentence_transformers import util
from evalbench_metrics.config import sentence_model
from utils import handle_output

@handle_output()
def recall_at_k(relevant_docs: list, retrieved_docs: list, k: int) -> float:
    if not relevant_docs or not retrieved_docs:
        return float('-inf')
    relevant_at_k = set(retrieved_docs[:k]).intersection(set(relevant_docs))
    return len(relevant_at_k) / len(relevant_docs) if relevant_docs else 0.0

@handle_output()
def precision_at_k(relevant_docs: list, retrieved_docs: list, k: int) -> float:
    if not relevant_docs or not retrieved_docs:
        return float('-inf')
    relevant_at_k = set(retrieved_docs[:k]).intersection(set(relevant_docs))
    return len(relevant_at_k) / k

@handle_output()
def dcg(relevance_scores: list) -> int:
    return sum([
        (2**rel - 1) / np.log2(idx + 2)
        for idx, rel in enumerate(relevance_scores)
    ])

@handle_output()
def ndcg_at_k(relevant_docs: list, retrieved_docs: list, k: int) -> float:
    if not relevant_docs or not retrieved_docs:
        return float('-inf')
    rel_scores = [1 if doc in relevant_docs else 0 for doc in retrieved_docs[:k]]
    ideal_rel_scores = sorted(rel_scores, reverse=True)
    dcg_val = dcg(rel_scores)
    idcg_val = dcg(ideal_rel_scores)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0

@handle_output()
def context_relevance_score(query: str, retrieved_contexts: list) -> list:
    if not query or not retrieved_contexts:
        return []
    query_emb = sentence_model.encode(query, convert_to_tensor=True)
    context_embs = sentence_model.encode(retrieved_contexts, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_emb, context_embs)
    return scores.squeeze().tolist()

@handle_output()
def evaluate_all(relevant_docs: list, retrieved_docs: list, k: int, query: str, retrieved_contexts: list) -> dict:
    scores = {}
    if relevant_docs and retrieved_docs and k:
        scores['recall@k'] = recall_at_k(relevant_docs, retrieved_docs, k, suppress_output=True)
        scores['precision@k'] = precision_at_k(relevant_docs, retrieved_docs, k, suppress_output=True)
        scores['ndcg@k'] = ndcg_at_k(relevant_docs, retrieved_docs, k, suppress_output=True)
    if query and retrieved_contexts:
        scores['context_relevance'] = context_relevance_score(query, retrieved_contexts, suppress_output=True)
    return scores