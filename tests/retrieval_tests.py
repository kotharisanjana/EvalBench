import pytest
import evalbench.metrics.predefined.retrieval as retrieval

@pytest.fixture
def test_data():
    return {
        'relevant_docs': [['doc1', 'doc4'], ['doc2', 'doc4', 'doc3'], ['doc1', 'doc3', 'doc4']],
        'retrieved_docs': [['doc2', 'doc5'], ['doc1', 'doc2'], ['doc1', 'doc3', 'doc4']],
        'k': 2
    }

def test_recall_at_k(test_data):
    score = retrieval.recall_at_k(test_data['relevant_docs'], test_data['retrieved_docs'], test_data['k'])
    expected = [0.0, 1/3, 2/3]
    assert all(abs(a - b) < 1e-5 for a, b in zip(score, expected)), \
        f'Expected {expected}, got {score}'

def test_precision_at_k(test_data):
    score = retrieval.precision_at_k(test_data['relevant_docs'], test_data['retrieved_docs'], test_data['k'])
    expected = [0.0, 1/2, 1.0]
    assert all(abs(a - b) < 1e-5 for a, b in zip(score, expected)), \
        f'Expected {expected}, got {score}'

def test_ndcg_at_k(test_data):
    score = retrieval.ndcg_at_k(test_data['relevant_docs'], test_data['retrieved_docs'], test_data['k'])
    for val in score:
        assert 0.0 <= val <= 1.0, \
            f'Expected between [0,1], but got {val}'

def test_mrr(test_data):
    score = retrieval.mrr_score(test_data['relevant_docs'], test_data['retrieved_docs'], test_data['k'])
    expected = [0.0, 0.5, 1.0]
    assert all(abs(a - b) < 1e-5 for a, b in zip(score, expected)), \
        f'Expected {expected}, got {score}'