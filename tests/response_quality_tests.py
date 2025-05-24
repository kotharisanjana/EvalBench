import pytest
import metrics.predefined.response_quality as response_quality

@pytest.fixture
def test_data():
    return {
        'response': [
            "The Earth orbits the Sun once every 365.25 days.",
            "It takes 365 days for Earth to orbit, which we know because of the calendar that was made by humans over time.",
            "Bananas can be used to power spaceships through the moon tunnel.",
        ]
    }

def test_coherence_score(test_data):
    score = response_quality.coherence_score(test_data['response'])
    assert all(1 <= b <= 5 for b in score), \
        f'{score} has out of range values'

def test_conciseness_score(test_data):
    score = response_quality.conciseness_score(test_data['response'])
    assert all(1 <= b <= 5 for b in score), \
        f'{score} has out of range values'

def test_factuality_score(test_data):
    score = response_quality.factuality_score(test_data['response'])
    assert all(0.0 <= b <= 1.0 for b in score), \
        f'{score} has out of range values'