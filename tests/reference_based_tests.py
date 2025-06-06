import pytest
from evalbench import metrics as reference_based

@pytest.fixture
def test_data():
    return {
        'reference': [
            'The cat is on the mat',
            'A quick brown fox jumps over the lazy dog',
            'Machine learning is a subfield of artificial intelligence'
        ],
        'generated': [
            'The cat sat on the mat',
            'A fast brown fox leaped over the lazy dog',
            'Machine learning belongs to the field of AI'
        ]
    }

def test_bleu_score(test_data):
    score = reference_based.bleu_score(test_data['reference'], test_data['generated'])
    assert all(0.0 <= b <= 1.0 for b in score), \
        f'{score} has out of range values'

def test_rouge_score(test_data):
    score = reference_based.rouge_score(test_data['reference'], test_data['generated'])
    for score_dict in score:
        assert isinstance(score_dict, dict)
        for key, value in score_dict.items():
            assert 0.0 <= value <= 1.0, f'{key} score out of range: {value}'

def test_meteor_score(test_data):
    score = reference_based.meteor_score(test_data['reference'], test_data['generated'])
    assert all(0.0 <= b <= 1.0 for b in score), \
        f'{score} has out of range values'

def test_semantic_similarity_score(test_data):
    score = reference_based.semantic_similarity_score(test_data['reference'], test_data['generated'])
    assert all(0.0 <= b <= 1.0 for b in score), \
        f'{score} has out of range values'

def test_bert_score(test_data):
    score = reference_based.bert_score(test_data['reference'], test_data['generated'])
    for score_dict in score:
        for key in ['precision', 'recall', 'f1']:
            assert 0.0 <= score_dict[key] <= 1.0, f'{key} out of range: {score_dict[key]}'