import pytest
from error_handling.custom_error import MetricError
import metrics.generation as metric

@pytest.fixture
def reference_generated_pair():
    return 'The cat is on the mat', 'The cat sat on the mat'

def test_bleu_score(reference_generated_pair):
    ref, gen = reference_generated_pair
    score = bleu_score(ref, gen)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

def test_rouge_score(reference_generated_pair):
    ref, gen = reference_generated_pair
    scores = rouge_score(ref, gen)
    assert isinstance(scores, dict)
    assert all(0.0 <= val <= 1.0 for val in scores.values())

def test_meteor_score(reference_generated_pair):
    ref, gen = reference_generated_pair
    score = meteor_score(ref, gen)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

def test_semantic_similarity_score(reference_generated_pair):
    ref, gen = reference_generated_pair
    score = semantic_similarity_score(ref, gen)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

def test_bert_score(reference_generated_pair):
    ref, gen = reference_generated_pair
    scores = bert_score(ref, gen)
    for key in ['precision', 'recall', 'f1']:
        assert key in scores
        assert 0.0 <= scores[key] <= 1.0

def test_validations_missing_args():
    ref, gen = 'The cat is on the mat', None
    score = bleu_score(ref, gen)
    assert MetricError

def test_validations_type_mismatch():
    ref, gen = 'The cat is on the mat', 23
    score = bleu_score(ref, gen)
    assert MetricError

def test_validations_empty_input():
    ref, gen = '', None
    score = bleu_score(ref, gen)
    assert MetricError