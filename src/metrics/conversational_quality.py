import openai

def evaluate_conversational_quality(context: str, response: str, metric_type: str) -> float:
    prompt = f'''
    You are a helpful evaluator. Given the following context and response, rate the response based on {metric_type} on a scale of 1 to 5.

    Context:
    \'\'\'{context}\'\'\'

    Response:
    \'\'\'{response}\'\'\'

    How {metric_type} is the response? Give a score from 1 to 5.
    '''
    completion = openai.ChatCompletion.create(
        model='gpt-4',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return float(completion['choices'][0]['message']['content'].strip())

def coherence_score(context: str, response: str) -> float:
    return evaluate_conversational_quality(context, response, 'coherence')

def conciseness_score(response: str) -> float:
    return evaluate_conversational_quality('', response, 'conciseness')

def helpfulness_score(context: str, response: str) -> float:
    return evaluate_conversational_quality(context, response, 'helpfulness')