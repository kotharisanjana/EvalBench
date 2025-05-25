import pytest
import evalbench.metrics.predefined.response_alignment as response_alignment

@pytest.fixture
def test_data():
    return {
        'queries': [
            'What are the symptoms of heat stroke?',
            'How can I improve my public speaking skills?'
        ],
        'contexts': [
            ['Common symptoms include high body temperature and confusion.'],
            ['Just be yourself and it will work out on its own']
        ],
        'responses': [
            'The Eiffel Tower is in Paris.',
            'Practice regularly and join a speaking group like Toastmasters.'
        ]
    }
def test_answer_relevance(test_data):
    score = response_alignment.answer_relevance_score(test_data['queries'], test_data['responses'])
    assert all(isinstance(s, str) for s in score), \
        f'Expected relevance scores to be string in [1, 5], but got {score}'

def test_helpfulness_score(test_data):
    score = response_alignment.helpfulness_score(test_data['queries'], test_data['responses'])
    assert all(isinstance(s, str) for s in score), \
        f'Expected relevance scores to be string in [1, 5], but got {score}'