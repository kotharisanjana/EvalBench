from evalbench_metrics.retrieval_generation import (
    faithfulness_score,
    hallucination_score,
    factuality_score,
    groundedness_score,
)

context = 'The Eiffel Tower is located in Paris and is one of the most iconic landmarks in the world.'
response_correct = 'The Eiffel Tower is located in Paris.'
response_wrong = 'The Eiffel Tower is located in New York.'

def test_faithfulness_score_valid():
    score = faithfulness_score(context, response_correct)
    assert isinstance(score, float), f'Expected float, got {type(score)}'
    assert score > 0, f'Expected positive score, got {score}'

def test_faithfulness_score_invalid():
    score = faithfulness_score(context, response_wrong)
    assert isinstance(score, float), f'Expected float, got {type(score)}'
    assert score > 0, f'Expected positive score, got {score}'

def test_hallucination_score_valid():
    score = hallucination_score(context, response_correct)
    assert isinstance(score, float), f'Expected float, got {type(score)}'
    assert score > 0, f'Expected positive score, got {score}'

def test_hallucination_score_invalid():
    score = hallucination_score(context, response_wrong)
    assert isinstance(score, float), f'Expected float, got {type(score)}'
    assert score > 0, f'Expected positive score, got {score}'

def test_factuality_score_factual_response():
    score = factuality_score(response_correct)
    assert isinstance(score, float), f'Expected float, got {type(score)}'
    assert score > 0, f'Expected positive score, got {score}'

def test_factuality_score_nonfactual_response():
    score = factuality_score(response_wrong)
    assert isinstance(score, float), f'Expected float, got {type(score)}'
    assert score > 0, f'Expected positive score, got {score}'

def test_g_eval_groundedness_score():
    score = groundedness_score(context, response_correct)
    assert score.strip().isdigit()
    assert 1 <= int(score.strip()) <= 5
