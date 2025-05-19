from evalbench_metrics.conversational import (
    coherence_score,
    conciseness_score,
    helpfulness_score,
)

context = 'The Eiffel Tower is located in Paris.'
response_correct = 'The Eiffel Tower is a famous landmark in Paris.'

def test_coherence_score():
    score = coherence_score(context, response_correct)
    assert isinstance(score, int)
    assert 1 <= score <= 5

def test_conciseness_score():
    score = conciseness_score(response_correct)
    assert isinstance(score, float)
    assert 1 <= score <= 5

def test_helpfulness_score():
    score = helpfulness_score(context, response_correct)
    assert isinstance(score, float)
    assert 1 <= score <= 5