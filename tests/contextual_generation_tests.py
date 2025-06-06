import pytest
import evalbench.metrics.predefined.contextual_generation as contextual_generation

@pytest.fixture
def test_data():
    return {
        'context': [
            ['The Eiffel Tower is located in Paris, France.'],
            ['Water boils at 100 degrees Celsius.'],
            ['The capital of Germany is Berlin.']
        ],
        'generated': [
            'The Eiffel Tower one of the prime landmarks of France and is a very popular tourist site.',
            'Water freezes at 100 degrees Celsius.',
            'The world is battling hunger and poverty'
        ]
    }

def test_faithfulness_score(test_data):
    score = contextual_generation.faithfulness_score(test_data['context'], test_data['generated'])
    assert all(0.0 <= s <= 1.0 for s in score), \
            f'Expected score in [0,1], got {score}'

def test_hallucination_score(test_data):
    score = contextual_generation.hallucination_score(test_data['context'], test_data['generated'])
    assert all(0.0 <= s <= 1.0 for s in score), \
        f'Expected score in [0,1], got {score}'

def test_groundedness_score(test_data):
    score = contextual_generation.groundedness_score(test_data['context'], test_data['generated'])
    assert all(isinstance(s, str) for s in score), \
        f'Expected groundedness scores to be string in [1, 3], but got {score}'
