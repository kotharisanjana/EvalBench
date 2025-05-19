from error_handling.custom_error import MetricError
from evalbench_metrics.retrieval_generation import (
    faithfulness_score,
    hallucination_score,
    factuality_score,
    groundedness_score,
    answer_relevance_score
)

context = 'The Eiffel Tower is located in Paris and is one of the most iconic landmarks in the world.'
response_correct = 'The Eiffel Tower is located in Paris.'
response_wrong = 'The Eiffel Tower is located in New York.'
query = 'Where is the Eiffel Tower located?'

def test_faithfulness_score_positive():
    score = faithfulness_score(context, response_correct)
    assert isinstance(score, float)
    assert score > 0, f'Expected positive score, got {score}'

def test_faithfulness_score_negative():
    score = faithfulness_score(context, response_wrong)
    assert isinstance(score, float)
    assert score <= 0

def test_hallucination_score_positive():
    score = hallucination_score(context, response_correct)
    assert isinstance(score, float)
    assert score > 0

def test_hallucination_score_negative():
    score = hallucination_score(context, response_wrong)
    assert isinstance(score, float)
    assert score <= 0

def test_factuality_score_positive():
    score = factuality_score(response_correct)
    assert isinstance(score, float)
    assert score > 0

def test_factuality_negative():
    score = factuality_score(response_wrong)
    assert isinstance(score, float)
    assert score <= 0

def test_groundedness_score():
    score = groundedness_score(context, response_correct)
    assert score.strip().isdigit()
    assert 1 <= int(score.strip()) <= 5

def test_answer_relevance_score():
    score = answer_relevance_score(query, response_correct)
    assert isinstance(score, float)
    assert 0 < score <= 5
