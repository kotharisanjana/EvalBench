from evalbench_metrics.conversational import (
    coherence_score,
    conciseness_score,
    helpfulness_score,
)

context = 'The Eiffel Tower is located in Paris.'
response_correct = 'The Eiffel Tower is a famous landmark in Paris.'
response_incorrect = 'The Eiffel Tower is in London.'

def test_coherence_score():
    score = coherence_score(context, response_correct)
    assert isinstance(score, float)
    assert 1 <= score <= 5

    score_incorrect = coherence_score(context, response_incorrect)
    assert isinstance(score_incorrect, float)
    assert 1 <= score_incorrect <= 5

def test_conciseness_score():
    score = conciseness_score(response_correct)
    assert isinstance(score, float)
    assert 1 <= score <= 5

    long_response = 'This is a long response that could have been more concise, but it provides detailed information about the Eiffel Tower in Paris and its history, location, and significance.'
    score_long = conciseness_score(long_response)
    assert isinstance(score_long, float)
    assert 1 <= score_long <= 5

def test_helpfulness_score():
    score = helpfulness_score(context, response_correct)
    assert isinstance(score, float)
    assert 1 <= score <= 5

    score_non_helpful = helpfulness_score(context, 'I don\'t know about the Eiffel Tower.')
    assert isinstance(score_non_helpful, float)
    assert 1 <= score_non_helpful <= 5