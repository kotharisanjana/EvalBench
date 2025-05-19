import pytest
import numpy as np
from metrics.retrieval_search import (
    recall_at_k,
    precision_at_k,
    dcg,
    ndcg_at_k,
    context_relevance_score
)

@pytest.fixture
def test_data():
    return {
        'relevant_docs': ['doc1', 'doc2', 'doc3'],
        'retrieved_docs': ['doc2', 'doc3', 'doc4', 'doc5'],
        'k': 3
    }

def test_recall_at_k(test_data):
    recall = recall_at_k(test_data['relevant_docs'], test_data['retrieved_docs'], test_data['k'])
    assert isinstance(recall, float), f'Expected float, but got {type(recall)}'
    assert recall == 2.0 / 3, f'Expected recall to be 1.0, but got {recall}'

def test_precision_at_k(test_data):
    precision = precision_at_k(test_data['relevant_docs'], test_data['retrieved_docs'], test_data['k'])
    assert isinstance(precision, float), f'Expected float, but got {type(precision)}'
    assert precision == 2.0 / 3, f'Expected precision to be 1/3, but got {precision}'

def test_dcg():
    relevance_scores = [3, 2, 3, 0, 1, 2]
    dcg_val = dcg(relevance_scores)
    assert isinstance(dcg_val, float), f'Expected float, but got {type(dcg_val)}'
    assert dcg_val == (2 ** 3 - 1) / np.log2(2) + (2 ** 2 - 1) / np.log2(3) + (2 ** 3 - 1) / np.log2(4) + (
                2 ** 0 - 1) / np.log2(5) + (2 ** 1 - 1) / np.log2(6) + (2 ** 2 - 1) / np.log2(
        7), f'Expected dcg to be correctly calculated, but got {dcg_val}'

def test_ndcg_at_k(test_data):
    ndcg_val = ndcg_at_k(test_data['relevant_docs'], test_data['retrieved_docs'], test_data['k'])
    assert isinstance(ndcg_val, float), f'Expected float, but got {type(ndcg_val)}'
    assert 0.0 <= ndcg_val <= 1.0, f'Expected NDCG value between 0.0 and 1.0, but got {ndcg_val}'

def test_context_relevance():
    query = 'What is the capital of France?'
    retrieved_contexts = ['Paris is the capital of France.', 'Berlin is the capital of Germany.',
                          'London is the capital of the UK.']
    scores = context_relevance_score(query, retrieved_contexts)
    assert isinstance(scores, list), f'Expected list, but got {type(scores)}'
    assert len(scores) == len(
        retrieved_contexts), f'Expected number of scores to match number of contexts, but got {len(scores)}'
    assert all(isinstance(score, float) for score in scores), f'Expected all scores to be floats'
    assert max(scores) > 0.0, f'Expected relevance scores to be greater than 0, but got {scores}'