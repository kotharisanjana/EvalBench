from typing import List
from utils.helper import get_config, handle_output, register_metric
import error_handling.validation_helpers as validation

def evaluate(response: str, metric_type: str) -> float:
    cfg = get_config()

    prompt = f'''
    You are a helpful and fair evaluator. Your task is to assess the following response based on {metric_type} using a numeric rating between 1 (poor) and 5 (excellent). Respond with only the number.
    
    Instructions:
    - Use the full scale (1 to 5) when evaluating.
    - Do not include any explanation—just return a single number.
    - Assume you're evaluating as a human would: fair, consistent, and strict.
    
    Rate this:
    Metric: {metric_type}
    
    Response:
    \'\'\'{response}\'\'\'
    
    Rating:
    '''.strip()

    try:
        completion = cfg.groq_client.chat.completions.create(
            model='llama3-8b-8192',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.0,
        )
        score = completion.choices[0].message.content.strip()
        return float(score)
    except ValueError:
        return 0

@register_metric('coherence', required_args=['response'], module='response_quality')
@handle_output()
def coherence_score(response: List[str]) -> List[float]:
    validation.validate_type_list_non_empty(('response', response))

    return [
        evaluate(resp, 'coherence')
        for resp in response
    ]

@register_metric('conciseness', required_args=['response'], module='response_quality')
@handle_output()
def conciseness_score(response: List[str]) -> List[float]:
    validation.validate_type_list_non_empty(('response', response))

    return [
        evaluate(resp, 'conciseness')
        for resp in response
    ]

@register_metric('factuality', required_args=['response'], module='response_quality')
@handle_output()
def factuality_score(response: List[str]) -> List[float]:
    validation.validate_type_list_non_empty(('response', response))

    cfg = get_config()
    candidate_labels = ['factually correct', 'factually incorrect']
    results = []

    for resp in response:
        hypothesis = f"Is the following response factually correct. Response: ""{resp}"""
        result = cfg.fact_check_model(resp, candidate_labels, hypothesis=hypothesis)
        labels = result["labels"]
        scores = result["scores"]
        results.append(scores[labels.index("factually correct")])

    return results
