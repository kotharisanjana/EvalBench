from evalbench_metrics.config import groq_client
from utils.decorators import handle_output, register_metric
from error_handling.validation_helpers import(
    validate_type_string_non_empty,
    validate_num_args,
)

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

@register_metric('coherence', required_args=['context', 'generated'], category='conversational')
@handle_output()
def coherence_score(context: str, generated: str) -> float:
    validate_num_args((('context', context), ('generated', generated)), length=2)
    validate_type_string_non_empty(('context', context), ('generated', generated))
    return evaluate_conversational_quality(context, generated, 'coherence')

@register_metric('conciseness', required_args=['generated'], category='conversational')
@handle_output()
def conciseness_score(generated: str) -> float:
    validate_num_args(('generated', generated), length=1)
    validate_type_string_non_empty(('generated', generated))
    return evaluate_conversational_quality('', generated, 'conciseness')

@register_metric('helpfulness', required_args=['context', 'generated'], category='conversational')
@handle_output()
def helpfulness_score(context: str, generated: str) -> float:
    validate_num_args((('context', context), ('generated', generated)), length=2)
    validate_type_string_non_empty(('context', context), ('generated', generated))
    return evaluate_conversational_quality(context, generated, 'helpfulness')


