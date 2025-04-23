from evalbench_metrics.config import groq_client
from utils import handle_output

def evaluate_conversational_quality(context: str, response: str, metric_type: str) -> float:
    prompt = f'''
    You are a helpful evaluator. Given the following context and response, rate the response based on {metric_type} on a scale of 1 to 5.

    Context:
    \'\'\'{context}\'\'\'

    Response:
    \'\'\'{response}\'\'\'

    How {metric_type} is the response?

    Respond only with a number between 1 and 5. Do not include any explanation or text.
    '''

    completion = groq_client.chat.completions.create(
        model='llama3-8b-8192',
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.0,
    )

    score_str = completion.choices[0].message.content

    try:
        return float(score_str)
    except ValueError:
        print(f'Unexpected output from model: {score_str}')
        return -1.0

@handle_output()
def coherence_score(context: str, response: str) -> float:
    return evaluate_conversational_quality(context, response, 'coherence')

@handle_output()
def conciseness_score(response: str) -> float:
    return evaluate_conversational_quality('', response, 'conciseness')

@handle_output()
def helpfulness_score(context: str, response: str) -> float:
    return evaluate_conversational_quality(context, response, 'helpfulness')

@handle_output()
def evaluate_all(context: str, response: str) -> dict:
    scores = {}
    if context and response:
        scores['coherence_score'] = coherence_score(context, response, suppress_output=True)
        scores['conciseness_score'] = conciseness_score(response, suppress_output=True)
        scores['helpfulness_score'] = helpfulness_score(context, response, suppress_output=True)
    return scores