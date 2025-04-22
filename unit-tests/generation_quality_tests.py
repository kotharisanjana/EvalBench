import pytest
from metrics.generation_quality import (
    bleu,
    rouge,
    meteor,
    semantic_similarity,
    bert_score_metric
)

@pytest.fixture
def reference_generated_pair():
    return 'The cat is on the mat', 'The cat sat on the mat'

def test_bleu(reference_generated_pair):
    ref, gen = reference_generated_pair
    score = bleu(ref, gen)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

def test_rouge(reference_generated_pair):
    ref, gen = reference_generated_pair
    scores = rouge(ref, gen)
    assert isinstance(scores, dict)
    assert all(0.0 <= val <= 1.0 for val in scores.values())

def test_meteor(reference_generated_pair):
    ref, gen = reference_generated_pair
    score = meteor(ref, gen)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

def test_semantic_similarity(reference_generated_pair):
    ref, gen = reference_generated_pair
    score = semantic_similarity(ref, gen)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

def test_bert_score_metric(reference_generated_pair):
    ref, gen = reference_generated_pair
    scores = bert_score_metric(ref, gen)
    for key in ['precision', 'recall', 'f1']:
        assert key in scores
        assert 0.0 <= scores[key] <= 1.0
