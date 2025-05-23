from typing import List
from utils.helper import get_config, handle_output, register_metric
import error_handling.validation_helpers as validation

cfg = get_config()

def evaluate_conversational_quality(context: str, response: str, metric_type: str) -> float:
    prompt = f'''
    You are a helpful and fair evaluator. Your task is to assess the following response based on **{metric_type}** using a numeric rating between **1 (poor)** and **5 (excellent)**. Respond with only the number.
    
    ### Instructions:
    - Use the full scale (1 to 5) when evaluating.
    - Do not include any explanation—just return a single number.
    - Assume you're evaluating as a human would: fair, consistent, and strict.
    
    ### Examples:
    
    #### Metric: Coherence
    Context: "I had a rough day at work today."
    Response: "Oh wow, the Eiffel Tower is in Paris."
    Rating: 1
    
    Context: "I had a rough day at work today."
    Response: "I'm sorry to hear that. Want to talk about it?"
    Rating: 5
    
    #### Metric: Conciseness
    Context: ""
    Response: "It is absolutely and unquestionably the case that yes, I do agree with that idea."
    Rating: 2
    
    Context: ""
    Response: "Yes, I agree."
    Rating: 5
    
    #### Metric: Helpfulness
    Context: "How can I improve my public speaking skills?"
    Response: "Maybe just try not to be nervous or something."
    Rating: 2
    
    Context: "How can I improve my public speaking skills?"
    Response: "Practice regularly, record yourself to evaluate progress, and consider joining a local speaking group like Toastmasters."
    Rating: 5
    
    ### Now rate this:
    
    Metric: {metric_type}
    Context:
    \"\"\"{context}\"\"\"
    
    Response:
    \"\"\"{response}\"\"\"
    
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
        return -1.0

@register_metric('coherence', required_args=['context', 'generated'], module='conversational')
@handle_output()
def coherence_score(context: List[List[str]], generated: List[str]) -> List[float]:
    """
    :param context: list of context strings
    :param generated: list of generated strings
    :return: list of coherence scores
    """
    validation.validate_batch_inputs(context, generated)
    return [
        evaluate_conversational_quality(' '.join(ctx), gen, 'coherence')
        for ctx, gen in zip(context, generated)
    ]

@register_metric('helpfulness', required_args=['context', 'generated'], module='conversational')
@handle_output()
def helpfulness_score(context: List[List[str]], generated: List[str]) -> List[float]:
    """
    :param context: list of context strings
    :param generated: list of generated strings
    :return: list of helpfulness scores
    """
    validation.validate_batch_inputs(context, generated)
    return [
        evaluate_conversational_quality(' '.join(ctx), gen, 'helpfulness')
        for ctx, gen in zip(context, generated)
    ]

# @register_metric('conciseness', required_args=['generated'], module='conversational')
# @handle_output()
# def conciseness_score(generated: str) -> float:
#     """
#     :param generated: list of generated strings
#     :return: list of conciseness scores
#     """
#     validate_num_args(('generated', generated), length=1)
#     validate_type_string_non_empty(('generated', generated))
#     return evaluate_conversational_quality('', generated, 'conciseness')
#
#
# @register_metric('factuality', required_args=['generated'], module='retrieval_generation')
# @handle_output()
# def factuality_score(generated: str) -> float:
#     validation.validate_batch_inputs(context, generated)
#
#     candidate_labels = ["factually correct", "factually incorrect"]
#     hypothesis = f"Is the following response factually correct. Response: '{generated}'"
#     result = cfg.fact_check_model(generated, candidate_labels, hypothesis=hypothesis)
#     score = result['scores'][1]
#     return score
