import numpy as np
from typing import List
from evalbench.utils.metrics_helper import handle_output, register_metric
import evalbench.error_handling.validation_helpers as validation

@register_metric(
    'recall_at_k',
    required_args=['relevant_docs', 'retrieved_docs', 'k'],
    arg_types=[List[List[str]], List[List[str]], int],
    module='retrieval'
)
@handle_output()
def recall_at_k(relevant_docs: List[List[str]], retrieved_docs: List[List[str]], k: int) -> List[float]:
    validation.validate_batch_inputs(('relevant_docs', relevant_docs), ('retrieved_docs', retrieved_docs))
    validation.validate_type_int_positive_integer(k, 'k')

    return [
        round(len(set(retrieved[:k]).intersection(set(relevant))) / len(relevant), 2) if len(relevant) > 0 else 0.0
        for retrieved, relevant in zip(retrieved_docs, relevant_docs)
    ]

@register_metric(
    'precision_at_k',
    required_args=['relevant_docs', 'retrieved_docs', 'k'],
    arg_types=[List[List[str]], List[List[str]], int],
    module='retrieval'
)
@handle_output()
def precision_at_k(relevant_docs: List[List[str]], retrieved_docs: List[List[str]], k: int) -> List[float]:
    validation.validate_batch_inputs(('relevant_docs', relevant_docs), ('retrieved_docs', retrieved_docs))
    validation.validate_type_int_positive_integer(k, 'k')

    return [
        round(len(set(retrieved[:k]).intersection(set(relevant))) / k, 2) if k > 0 else 0.0
        for retrieved, relevant in zip(retrieved_docs, relevant_docs)
    ]

def _dcg(relevance_scores: list) -> int:
    return sum([
        (2**rel - 1) / np.log2(idx + 2)
        for idx, rel in enumerate(relevance_scores)
    ])

@register_metric(
    'ndcg_at_k',
    required_args=['relevant_docs', 'retrieved_docs', 'k'],
    arg_types=[List[List[str]], List[List[str]], int],
    module='retrieval'
)
@handle_output()
def ndcg_at_k(relevant_docs: List[List[str]], retrieved_docs: List[List[str]], k: int) -> List[float]:
    validation.validate_batch_inputs(('relevant_docs', relevant_docs), ('retrieved_docs', retrieved_docs))
    validation.validate_type_int_positive_integer(k, 'k')

    results = []
    for rel_docs, ret_docs in zip(relevant_docs, retrieved_docs):
        rel_scores = [1 if doc in rel_docs else 0 for doc in ret_docs[:k]]
        ideal_rel_scores = sorted(rel_scores, reverse=True)
        dcg_val = _dcg(rel_scores)
        idcg_val = _dcg(ideal_rel_scores)
        ndcg = dcg_val / idcg_val if idcg_val > 0 else 0.0
        results.append(round(ndcg, 2))

    return results

@register_metric(
    'mrr',
    required_args=['retrieved_docs', 'relevant_docs', 'k'],
    arg_types=[List[List[str]], List[List[str]], int],
    module='retrieval'
)
@handle_output()
def mrr_score(relevant_docs: List[List[str]], retrieved_docs: List[List[str]], k: int) -> List[float]:
    validation.validate_batch_inputs(('relevant_docs', relevant_docs), ('retrieved_docs', retrieved_docs))
    validation.validate_type_int_positive_integer(k, 'k')

    results = []
    for rel, ret in zip(relevant_docs, retrieved_docs):
        rank = 0
        for idx, doc in enumerate(ret[:k]):
            if doc in rel:
                rank = 1.0 / (idx + 1)
                break
        results.append(round(rank, 2))

    return results
