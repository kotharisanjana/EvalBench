from . import groq_client
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

def coherence_score(context: str, response: str) -> float:
    return evaluate_conversational_quality(context, response, 'coherence')

def conciseness_score(response: str) -> float:
    return evaluate_conversational_quality('', response, 'conciseness')

def helpfulness_score(context: str, response: str) -> float:
    return evaluate_conversational_quality(context, response, 'helpfulness')